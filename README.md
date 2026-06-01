# FFT Webcam

A browser app that runs a **2D FFT on live webcam video on the GPU**, shows the
frequency spectrum, lets you **paint a filter mask** onto the spectrum, and
reconstructs the filtered video in real time via an inverse FFT.

- **Pipeline:** all on the GPU (WebGL2). The only per-frame CPU↔GPU traffic is
  uploading the webcam frame as a texture.
- **Processing:** grayscale / luma (one FFT).
- **FFT:** hand-rolled radix-2 **Stockham** FFT in fragment shaders, ping-ponging
  between `RG32F` (real, imag) float textures. Separable 2D = rows then columns.
- **Stack:** React + Vite + TypeScript.

## Requirements

- WebGL2 with `EXT_color_buffer_float` (float render targets). The app
  feature-detects and shows a banner if unsupported.
- A webcam, served from a secure context (`localhost` counts).

## Develop

```bash
npm install
npm run dev
```

Open the printed URL and grant webcam access.

## Per-frame pipeline

1. Webcam frame → texture (downscaled/cropped to N×N, N a power of two)
2. Grayscale + pack into complex texture (R=luma, G=0)
3. Forward 2D FFT → frequency texture `F`
4. Magnitude view: `log(1+|F|)` + fftshift + colormap → **Spectrum canvas**
5. Multiply `F` by the drawn **mask** → `F'`
6. Inverse 2D FFT of `F'`, scale by 1/N², real part → **Reconstruction canvas**
7. Re-pack that real reconstruction (imag = 0) and forward FFT it again →
   **Round-trip spectrum** (shows how keeping only the real part reshapes the
   spectrum; with no filter it matches the original spectrum)

## Build phases

- [x] **0** Scaffold (Vite + React + TS), WebGL2 / float-texture feature check
- [x] **1** Webcam live, downscaled to N×N grayscale
- [x] **2** Forward/inverse 2D FFT engine (Stockham); validated via impulse / round-trip
- [x] **3** Live spectrum visualization (log magnitude + fftshift + viridis/grayscale)
- [x] **4** Inverse FFT + reconstruction pane; live forward→inverse round-trip
- [x] **5** Mask drawing overlay (block/pass brush + low/high/band-pass, invert, clear)
- [x] **6** Mask multiply → inverse FFT; painted filter reshapes live video
- [x] **7** Controls (FFT size 256/512/1024, log gain, colormap, brush, FPS) + mask-upload dirty flag

## Layout

```
src/
  gl/
    glUtils.ts            context, feature detection, programs, RGBA32F targets
    shaders.ts            GLSL sources (string literals)
    FFT2D.ts              Stockham forward/inverse 2D FFT engine
    Pipeline.ts           live loop: upload → pack → FFT → present views
  filters/FilterMask.ts   offscreen mask canvas (alpha = pass/block) + presets
  hooks/useWebcam.ts
  App.tsx
scripts/
  fft-ref-check.mjs       CPU mirror of the FFT vs naive DFT (node, dev-only)
```

## Verifying the FFT

`node scripts/fft-ref-check.mjs` runs a CPU port of the exact butterfly + pass
schedule against a naive DFT (float64, errors ~1e-13) — impulse → flat spectrum
and `inverse(forward(x)) ≈ x`. A fast regression check when touching the FFT.
