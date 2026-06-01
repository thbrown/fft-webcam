// The live processing pipeline. One WebGL2 context does all GPU work on a
// single offscreen N x N canvas: upload the webcam frame, pack to complex,
// forward FFT, then render each requested view (grayscale input, magnitude
// spectrum) to the offscreen canvas and blit it into a visible 2D canvas with
// drawImage. Doing every view from one context means one frame upload and one
// FFT per frame, and scales cleanly to the reconstruction view in later phases.

import {
  createGL,
  createProgram,
  createFullScreenTriangle,
  createComplexTarget,
  type RenderTarget,
} from "./glUtils.ts";
import {
  FULLSCREEN_VERT,
  DISPLAY_R_FRAG,
  MAGNITUDE_FRAG,
  MULTIPLY_FRAG,
  PACK_REAL_FRAG,
} from "./shaders.ts";
import { FFT2D } from "./FFT2D.ts";

export interface PipelineParams {
  /** Multiplier on log(1 + |F|) for spectrum brightness. */
  logGain: number;
  /** 0 = grayscale, 1 = viridis. */
  colormap: number;
  /** When true (and a mask is given), filter the spectrum before inverting. */
  filterEnabled: boolean;
  /** The filter mask source canvas, or null for no filtering. */
  mask: HTMLCanvasElement | null;
  /** Mask edit counter; the texture is re-uploaded only when this changes. */
  maskVersion: number;
}

export interface RenderTargets {
  grayscale?: CanvasRenderingContext2D | null;
  spectrum?: CanvasRenderingContext2D | null;
  reconstruction?: CanvasRenderingContext2D | null;
  /** Magnitude spectrum of the (real) reconstruction — the round-trip view. */
  roundtrip?: CanvasRenderingContext2D | null;
}

export class Pipeline {
  readonly size: number;
  private canvas: HTMLCanvasElement;
  private gl: WebGL2RenderingContext;
  private fft: FFT2D;
  private vao: WebGLVertexArrayObject;
  private videoTex: WebGLTexture;

  private displayProgram: WebGLProgram;
  private magProgram: WebGLProgram;
  private multiplyProgram: WebGLProgram;
  private packRealProgram: WebGLProgram;
  private uDisplayTex: WebGLUniformLocation | null;
  private uFreq: WebGLUniformLocation | null;
  private uLogGain: WebGLUniformLocation | null;
  private uColormap: WebGLUniformLocation | null;
  private uMulFreq: WebGLUniformLocation | null;
  private uMulMask: WebGLUniformLocation | null;
  private uPackRealSrc: WebGLUniformLocation | null;

  private maskTex: WebGLTexture;
  private filtered: RenderTarget;
  private realPacked: RenderTarget;
  private lastMaskVersion = -1;

  constructor(size: number) {
    this.size = size;
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    this.canvas = canvas;
    const gl = createGL(canvas);
    this.gl = gl;

    this.fft = new FFT2D(gl, size);
    this.vao = createFullScreenTriangle(gl);

    this.videoTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.videoTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    this.displayProgram = createProgram(gl, FULLSCREEN_VERT, DISPLAY_R_FRAG);
    this.magProgram = createProgram(gl, FULLSCREEN_VERT, MAGNITUDE_FRAG);
    this.multiplyProgram = createProgram(gl, FULLSCREEN_VERT, MULTIPLY_FRAG);
    this.packRealProgram = createProgram(gl, FULLSCREEN_VERT, PACK_REAL_FRAG);
    this.uDisplayTex = gl.getUniformLocation(this.displayProgram, "u_tex");
    this.uFreq = gl.getUniformLocation(this.magProgram, "u_freq");
    this.uLogGain = gl.getUniformLocation(this.magProgram, "u_logGain");
    this.uColormap = gl.getUniformLocation(this.magProgram, "u_colormap");
    this.uMulFreq = gl.getUniformLocation(this.multiplyProgram, "u_freq");
    this.uMulMask = gl.getUniformLocation(this.multiplyProgram, "u_mask");
    this.uPackRealSrc = gl.getUniformLocation(this.packRealProgram, "u_src");

    this.maskTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.maskTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    this.filtered = createComplexTarget(gl, size, size);
    this.realPacked = createComplexTarget(gl, size, size);
  }

  render(
    video: HTMLVideoElement,
    targets: RenderTargets,
    params: PipelineParams,
  ): void {
    const gl = this.gl;
    if (video.readyState < video.HAVE_CURRENT_DATA) return;

    // 1. Upload the current frame.
    gl.bindTexture(gl.TEXTURE_2D, this.videoTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);

    // 2. Pack to complex (center-square crop + luma) and forward FFT.
    this.fft.pack(this.videoTex, video.videoWidth, video.videoHeight, true);
    const freq = this.fft.forward();

    // 3. Present each requested view from the single offscreen canvas. The
    // spectrum must be drawn before the inverse runs, because the inverse
    // ping-pongs through (and overwrites) the frequency buffer.
    if (targets.grayscale) {
      this.drawReal(this.fft.inputTexture);
      this.present(targets.grayscale);
    }
    if (targets.spectrum) {
      this.drawMagnitude(freq.texture, params);
      this.present(targets.spectrum);
    }
    if (targets.reconstruction || targets.roundtrip) {
      // Optionally filter the spectrum, then invert. With an all-pass mask (or
      // filtering disabled) this is the plain round-trip reproducing the input.
      let spatialInput = freq.texture;
      if (params.filterEnabled && params.mask) {
        if (params.maskVersion !== this.lastMaskVersion) {
          this.uploadMask(params.mask);
          this.lastMaskVersion = params.maskVersion;
        }
        this.multiply(freq.texture);
        spatialInput = this.filtered.texture;
      }
      const spatial = this.fft.inverse(spatialInput);

      if (targets.reconstruction) {
        this.drawReal(spatial.texture);
        this.present(targets.reconstruction);
      }
      if (targets.roundtrip) {
        // Re-FFT only the real part (the displayed image). Pack first, before
        // forwardFrom overwrites the ping-pong buffers `spatial` lives in.
        this.packReal(spatial.texture);
        const rt = this.fft.forwardFrom(this.realPacked.texture);
        this.drawMagnitude(rt.texture, params);
        this.present(targets.roundtrip);
      }
    }
  }

  /** Renders src.r into the realPacked complex buffer as (r, 0). */
  private packReal(srcTex: WebGLTexture): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.realPacked.framebuffer);
    gl.viewport(0, 0, this.size, this.size);
    gl.useProgram(this.packRealProgram);
    gl.bindVertexArray(this.vao);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, srcTex);
    gl.uniform1i(this.uPackRealSrc, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  /** Blits the offscreen N×N canvas into a (possibly different-sized) 2D view. */
  private present(ctx: CanvasRenderingContext2D): void {
    ctx.drawImage(this.canvas, 0, 0, ctx.canvas.width, ctx.canvas.height);
  }

  private uploadMask(maskCanvas: HTMLCanvasElement): void {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.maskTex);
    // Flip Y on upload so the shader's fftshift also undoes drawImage's flip.
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, maskCanvas);
    // Reset so the next frame's (top-down) video upload isn't flipped.
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
  }

  private multiply(freqTex: WebGLTexture): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.filtered.framebuffer);
    gl.viewport(0, 0, this.size, this.size);
    gl.useProgram(this.multiplyProgram);
    gl.bindVertexArray(this.vao);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, freqTex);
    gl.uniform1i(this.uMulFreq, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.maskTex);
    gl.uniform1i(this.uMulMask, 1);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.activeTexture(gl.TEXTURE0);
  }

  private drawReal(tex: WebGLTexture): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.size, this.size);
    gl.useProgram(this.displayProgram);
    gl.bindVertexArray(this.vao); // fft passes leave their own VAO bound
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.uniform1i(this.uDisplayTex, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  private drawMagnitude(freqTex: WebGLTexture, params: PipelineParams): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.size, this.size);
    gl.useProgram(this.magProgram);
    gl.bindVertexArray(this.vao);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, freqTex);
    gl.uniform1i(this.uFreq, 0);
    gl.uniform1f(this.uLogGain, params.logGain);
    gl.uniform1i(this.uColormap, params.colormap);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  dispose(): void {
    const gl = this.gl;
    this.fft.dispose();
    gl.deleteTexture(this.videoTex);
    gl.deleteTexture(this.maskTex);
    for (const t of [this.filtered, this.realPacked]) {
      gl.deleteFramebuffer(t.framebuffer);
      gl.deleteTexture(t.texture);
    }
    gl.deleteProgram(this.displayProgram);
    gl.deleteProgram(this.magProgram);
    gl.deleteProgram(this.multiplyProgram);
    gl.deleteProgram(this.packRealProgram);
    gl.deleteVertexArray(this.vao);
    gl.getExtension("WEBGL_lose_context")?.loseContext();
  }
}
