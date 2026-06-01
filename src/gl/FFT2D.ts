// Separable 2D FFT on the GPU via a radix-2 Stockham auto-sort transform.
//
// The transform is run as log2(N) butterfly passes along rows, then log2(N)
// along columns, ping-ponging between two RGBA32F complex textures. Forward is
// unnormalized; inverse folds the 1/N^2 scale into its final pass.

import {
  createProgram,
  createFullScreenTriangle,
  createComplexTarget,
  type RenderTarget,
} from "./glUtils.ts";
import { FULLSCREEN_VERT, PACK_FRAG, FFT_FRAG } from "./shaders.ts";

interface FFTUniforms {
  src: WebGLUniformLocation | null;
  subtransformSize: WebGLUniformLocation | null;
  normalization: WebGLUniformLocation | null;
  horizontal: WebGLUniformLocation | null;
  forward: WebGLUniformLocation | null;
  resolution: WebGLUniformLocation | null;
}

interface PackUniforms {
  src: WebGLUniformLocation | null;
  flipY: WebGLUniformLocation | null;
  srcSize: WebGLUniformLocation | null;
}

function isPowerOfTwo(n: number): boolean {
  return n > 0 && (n & (n - 1)) === 0;
}

export class FFT2D {
  readonly size: number;
  private gl: WebGL2RenderingContext;
  private vao: WebGLVertexArrayObject;

  private fftProgram: WebGLProgram;
  private packProgram: WebGLProgram;
  private fftU: FFTUniforms;
  private packU: PackUniforms;

  /** Input for both packing and direct complex uploads. */
  private input: RenderTarget;
  /** Ping-pong scratch buffers. */
  private bufA: RenderTarget;
  private bufB: RenderTarget;

  private readonly iterations: number;

  constructor(gl: WebGL2RenderingContext, size: number) {
    if (!isPowerOfTwo(size)) {
      throw new Error(`FFT size must be a power of two, got ${size}.`);
    }
    this.gl = gl;
    this.size = size;
    this.iterations = Math.log2(size);

    this.vao = createFullScreenTriangle(gl);
    this.fftProgram = createProgram(gl, FULLSCREEN_VERT, FFT_FRAG);
    this.packProgram = createProgram(gl, FULLSCREEN_VERT, PACK_FRAG);

    this.fftU = {
      src: gl.getUniformLocation(this.fftProgram, "u_src"),
      subtransformSize: gl.getUniformLocation(this.fftProgram, "u_subtransformSize"),
      normalization: gl.getUniformLocation(this.fftProgram, "u_normalization"),
      horizontal: gl.getUniformLocation(this.fftProgram, "u_horizontal"),
      forward: gl.getUniformLocation(this.fftProgram, "u_forward"),
      resolution: gl.getUniformLocation(this.fftProgram, "u_resolution"),
    };
    this.packU = {
      src: gl.getUniformLocation(this.packProgram, "u_src"),
      flipY: gl.getUniformLocation(this.packProgram, "u_flipY"),
      srcSize: gl.getUniformLocation(this.packProgram, "u_srcSize"),
    };

    this.input = createComplexTarget(gl, size, size);
    this.bufA = createComplexTarget(gl, size, size);
    this.bufB = createComplexTarget(gl, size, size);
  }

  /**
   * Renders a source texture (e.g. a webcam frame) into the internal complex
   * input buffer as luma in the real channel. Call before `forward()`.
   */
  pack(srcTex: WebGLTexture, srcWidth: number, srcHeight: number, flipY: boolean): void {
    const gl = this.gl;
    gl.useProgram(this.packProgram);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.input.framebuffer);
    gl.viewport(0, 0, this.size, this.size);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, srcTex);
    gl.uniform1i(this.packU.src, 0);
    gl.uniform1i(this.packU.flipY, flipY ? 1 : 0);
    gl.uniform2f(this.packU.srcSize, srcWidth, srcHeight);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  /** The packed complex input texture (real = luma). Valid after `pack()`. */
  get inputTexture(): WebGLTexture {
    return this.input.texture;
  }

  /** Forward 2D FFT of the current input buffer. Returns the frequency target. */
  forward(): RenderTarget {
    return this.run(this.input.texture, true);
  }

  /**
   * Forward 2D FFT of an externally supplied complex texture. Pass a texture
   * that is not one of this engine's internal ping-pong buffers (the transform
   * overwrites those), or whose contents are no longer needed afterwards.
   */
  forwardFrom(complexTex: WebGLTexture): RenderTarget {
    return this.run(complexTex, true);
  }

  /** Inverse 2D FFT of `freqTex`. Returns the spatial-domain target. */
  inverse(freqTex: WebGLTexture): RenderTarget {
    return this.run(freqTex, false);
  }

  private run(inputTex: WebGLTexture, forward: boolean): RenderTarget {
    const gl = this.gl;
    const N = this.size;
    const totalPasses = this.iterations * 2;

    gl.useProgram(this.fftProgram);
    gl.bindVertexArray(this.vao);
    gl.uniform2f(this.fftU.resolution, N, N);
    gl.uniform1i(this.fftU.forward, forward ? 1 : 0);

    // Start writing into whichever buffer is not the input, so the first pass
    // never reads and writes the same texture.
    let ping = inputTex === this.bufA.texture ? this.bufB : this.bufA;
    let pong = ping === this.bufA ? this.bufB : this.bufA;

    let src = inputTex;
    let pass = 0;
    const doAxis = (horizontal: boolean) => {
      for (let i = 0; i < this.iterations; i++) {
        const subtransformSize = 1 << (i + 1);
        // Apply the full 1/N^2 inverse normalization on the very last pass.
        const norm = !forward && pass === totalPasses - 1 ? 1 / (N * N) : 1;
        this.fftPass(src, ping, horizontal, subtransformSize, norm);
        src = ping.texture;
        const next = pong;
        pong = ping;
        ping = next;
        pass++;
      }
    };

    doAxis(true); // rows
    doAxis(false); // columns

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return src === this.bufA.texture ? this.bufA : this.bufB;
  }

  private fftPass(
    srcTex: WebGLTexture,
    dst: RenderTarget,
    horizontal: boolean,
    subtransformSize: number,
    normalization: number,
  ): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, dst.framebuffer);
    gl.viewport(0, 0, this.size, this.size);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, srcTex);
    gl.uniform1i(this.fftU.src, 0);
    gl.uniform1f(this.fftU.subtransformSize, subtransformSize);
    gl.uniform1f(this.fftU.normalization, normalization);
    gl.uniform1i(this.fftU.horizontal, horizontal ? 1 : 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  // --- Validation / interop helpers -------------------------------------

  /** Uploads interleaved [re, im, re, im, ...] (length N*N*2) into the input. */
  uploadComplex(data: Float32Array): void {
    const gl = this.gl;
    const N = this.size;
    if (data.length !== N * N * 2) {
      throw new Error(`Expected ${N * N * 2} values, got ${data.length}.`);
    }
    const rgba = new Float32Array(N * N * 4);
    for (let i = 0; i < N * N; i++) {
      rgba[i * 4] = data[i * 2];
      rgba[i * 4 + 1] = data[i * 2 + 1];
    }
    gl.bindTexture(gl.TEXTURE_2D, this.input.texture);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, N, N, gl.RGBA, gl.FLOAT, rgba);
  }

  /** Reads back a target as interleaved [re, im, re, im, ...] (length N*N*2). */
  readComplex(target: RenderTarget): Float32Array {
    const gl = this.gl;
    const N = this.size;
    const rgba = new Float32Array(N * N * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, target.framebuffer);
    gl.readPixels(0, 0, N, N, gl.RGBA, gl.FLOAT, rgba);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    const out = new Float32Array(N * N * 2);
    for (let i = 0; i < N * N; i++) {
      out[i * 2] = rgba[i * 4];
      out[i * 2 + 1] = rgba[i * 4 + 1];
    }
    return out;
  }

  dispose(): void {
    const gl = this.gl;
    gl.deleteProgram(this.fftProgram);
    gl.deleteProgram(this.packProgram);
    gl.deleteVertexArray(this.vao);
    for (const t of [this.input, this.bufA, this.bufB]) {
      gl.deleteTexture(t.texture);
      gl.deleteFramebuffer(t.framebuffer);
    }
  }
}
