// Authoritative filter mask, held on an offscreen N x N 2D canvas.
//
// The mask is stored in the ALPHA channel (RGB kept white): alpha 1 = pass,
// alpha 0 = block. This makes both consumers simple:
//   - GPU (Phase 6): upload the canvas and multiply the spectrum by texel.a.
//   - Overlay: fill red, then `destination-out` the mask to erase red wherever
//     the mask passes, leaving red only over blocked frequencies.
//
// The mask lives in the same fftshifted (DC-centered) space the spectrum is
// displayed in, so the center is low frequency — intuitive to paint. Phase 6
// unshifts when applying it to the corner-DC frequency data.

export type BrushMode = "block" | "pass";

export class FilterMask {
  readonly size: number;
  readonly canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  /** Bumped on every edit so consumers can skip re-uploading an unchanged mask. */
  private _version = 0;

  constructor(size: number) {
    this.size = size;
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d", { willReadFrequently: true })!;
    this.allPass();
  }

  get version(): number {
    return this._version;
  }

  private get center(): number {
    return this.size / 2;
  }

  /** Opaque white everywhere → multiplier 1 → no filtering. */
  allPass(): void {
    const ctx = this.ctx;
    ctx.globalCompositeOperation = "source-over";
    ctx.clearRect(0, 0, this.size, this.size);
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, this.size, this.size);
    this._version++;
  }

  /** Transparent everywhere → multiplier 0 → blocks all frequencies. */
  blockAll(): void {
    this.ctx.globalCompositeOperation = "source-over";
    this.ctx.clearRect(0, 0, this.size, this.size);
    this._version++;
  }

  private dab(x: number, y: number, radius: number, mode: BrushMode): void {
    const ctx = this.ctx;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    if (mode === "pass") {
      ctx.globalCompositeOperation = "source-over";
      ctx.fillStyle = "#fff";
    } else {
      // Erase alpha to 0 (block); fill color is irrelevant for destination-out.
      ctx.globalCompositeOperation = "destination-out";
      ctx.fillStyle = "#000";
    }
    ctx.fill();
    ctx.globalCompositeOperation = "source-over";
  }

  /** Single brush dab. */
  point(x: number, y: number, radius: number, mode: BrushMode): void {
    this.dab(x, y, radius, mode);
    this._version++;
  }

  /** Brush stroke as overlapping dabs between two points (smooth line). */
  stroke(
    x0: number,
    y0: number,
    x1: number,
    y1: number,
    radius: number,
    mode: BrushMode,
  ): void {
    const dist = Math.hypot(x1 - x0, y1 - y0);
    const steps = Math.max(1, Math.ceil(dist / (radius * 0.4)));
    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      this.dab(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, radius, mode);
    }
    this._version++;
  }

  /** Pass a centered disk of the given radius; block everything else. */
  lowPass(radius: number): void {
    this.blockAll();
    this.dab(this.center, this.center, radius, "pass");
  }

  /** Block a centered disk of the given radius; pass everything else. */
  highPass(radius: number): void {
    this.allPass();
    this.dab(this.center, this.center, radius, "block");
  }

  /** Pass an annulus between `inner` and `outer`; block elsewhere. */
  bandPass(inner: number, outer: number): void {
    this.blockAll();
    this.dab(this.center, this.center, outer, "pass");
    this.dab(this.center, this.center, inner, "block");
  }

  /** Swap pass/block everywhere. */
  invert(): void {
    const img = this.ctx.getImageData(0, 0, this.size, this.size);
    const d = img.data;
    for (let i = 0; i < d.length; i += 4) {
      d[i] = 255; // keep RGB white so later composites stay consistent
      d[i + 1] = 255;
      d[i + 2] = 255;
      d[i + 3] = 255 - d[i + 3]; // invert the alpha multiplier
    }
    this.ctx.putImageData(img, 0, 0);
    this._version++;
  }

  /** Draws a translucent overlay (red over blocked frequencies) into `ctx`. */
  renderOverlay(ctx: CanvasRenderingContext2D): void {
    const N = this.size;
    ctx.save();
    ctx.globalCompositeOperation = "source-over";
    ctx.clearRect(0, 0, N, N);
    ctx.fillStyle = "rgba(255, 45, 45, 0.45)";
    ctx.fillRect(0, 0, N, N);
    ctx.globalCompositeOperation = "destination-out";
    ctx.drawImage(this.canvas, 0, 0);
    ctx.restore();
  }
}
