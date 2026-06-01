// Throwaway CPU mirror of the GLSL Stockham butterfly + FFT2D pass schedule,
// checked against a naive DFT. Validates the algorithm and pass sequencing
// independent of WebGL. Run: node scripts/fft-ref-check.mjs
//
// Complex arrays are interleaved Float64 [re, im, re, im, ...], length N*N*2.

const PI = Math.PI;

// One butterfly stage along one axis, mirroring FFT_FRAG exactly.
function fftPass(src, N, subtransformSize, horizontal, forward, normalization) {
  const out = new Float64Array(N * N * 2);
  const half = subtransformSize * 0.5;
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const index = horizontal ? x : y;
      const evenIndex =
        Math.floor(index / subtransformSize) * half + (index % half);

      let ex, ey, ox, oy;
      if (horizontal) {
        ex = evenIndex; ey = y;
        ox = evenIndex + N / 2; oy = y;
      } else {
        ex = x; ey = evenIndex;
        ox = x; oy = evenIndex + N / 2;
      }
      const eRe = src[(ey * N + ex) * 2];
      const eIm = src[(ey * N + ex) * 2 + 1];
      const oRe = src[(oy * N + ox) * 2];
      const oIm = src[(oy * N + ox) * 2 + 1];

      const arg = (forward ? -1 : 1) * 2 * PI * (index / subtransformSize);
      const tr = Math.cos(arg);
      const ti = Math.sin(arg);
      // twiddle * odd
      const pr = tr * oRe - ti * oIm;
      const pi = tr * oIm + ti * oRe;

      const o = (y * N + x) * 2;
      out[o] = (eRe + pr) * normalization;
      out[o + 1] = (eIm + pi) * normalization;
    }
  }
  return out;
}

function run(input, N, forward) {
  const iterations = Math.log2(N);
  const totalPasses = iterations * 2;
  let data = input;
  let pass = 0;
  const axis = (horizontal) => {
    for (let i = 0; i < iterations; i++) {
      const subtransformSize = 1 << (i + 1);
      const norm =
        !forward && pass === totalPasses - 1 ? 1 / (N * N) : 1;
      data = fftPass(data, N, subtransformSize, horizontal, forward, norm);
      pass++;
    }
  };
  axis(true);
  axis(false);
  return data;
}

// Naive 2D DFT reference (unnormalized forward).
function naiveDFT2D(input, N) {
  const out = new Float64Array(N * N * 2);
  for (let ky = 0; ky < N; ky++) {
    for (let kx = 0; kx < N; kx++) {
      let re = 0, im = 0;
      for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
          const ang = -2 * PI * ((kx * x) / N + (ky * y) / N);
          const ir = input[(y * N + x) * 2];
          const ii = input[(y * N + x) * 2 + 1];
          re += ir * Math.cos(ang) - ii * Math.sin(ang);
          im += ir * Math.sin(ang) + ii * Math.cos(ang);
        }
      }
      out[(ky * N + kx) * 2] = re;
      out[(ky * N + kx) * 2 + 1] = im;
    }
  }
  return out;
}

function maxAbsDiff(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

let failures = 0;
function check(name, ok, detail) {
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}  ${detail}`);
  if (!ok) failures++;
}

// 1. Forward FFT vs naive DFT on a random image (small N).
{
  const N = 8;
  const x = new Float64Array(N * N * 2);
  let s = 42;
  const rnd = () => ((s = (s * 1103515245 + 12345) >>> 0) / 0xffffffff);
  for (let i = 0; i < N * N; i++) x[i * 2] = rnd();
  const fast = run(x, N, true);
  const ref = naiveDFT2D(x, N);
  const err = maxAbsDiff(fast, ref);
  check(`Forward FFT == naive DFT (N=${N})`, err < 1e-9, `maxErr=${err.toExponential(2)}`);
}

// 2. Impulse at origin -> flat spectrum of ones.
for (const N of [8, 256]) {
  const x = new Float64Array(N * N * 2);
  x[0] = 1;
  const f = run(x, N, true);
  let err = 0;
  for (let i = 0; i < N * N; i++)
    err = Math.max(err, Math.abs(f[i * 2] - 1), Math.abs(f[i * 2 + 1]));
  check(`Impulse -> flat ones (N=${N})`, err < 1e-9, `maxErr=${err.toExponential(2)}`);
}

// 3. Round-trip inverse(forward(x)) == x.
for (const N of [64, 512]) {
  const x = new Float64Array(N * N * 2);
  let s = 1337;
  const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 0xffffffff);
  for (let i = 0; i < N * N; i++) x[i * 2] = rnd();
  const back = run(run(x, N, true), N, false);
  const err = maxAbsDiff(back, x);
  check(`Round-trip (N=${N})`, err < 1e-9, `maxErr=${err.toExponential(2)}`);
}

console.log(failures === 0 ? "\nALL PASSED" : `\n${failures} FAILED`);
process.exit(failures === 0 ? 0 : 1);
