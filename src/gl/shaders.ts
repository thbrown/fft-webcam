// GLSL sources kept as string literals so no extra Vite loader is needed.
// All shaders target WebGL2 / GLSL ES 3.00.

/** Full-screen triangle. Passes a 0..1 UV to the fragment stage. */
export const FULLSCREEN_VERT = /* glsl */ `#version 300 es
layout(location = 0) in vec2 a_pos;
out vec2 v_uv;
void main() {
  v_uv = a_pos * 0.5 + 0.5;
  gl_Position = vec4(a_pos, 0.0, 1.0);
}
`;

/**
 * Displays the real channel of a complex texture as grayscale. Used to preview
 * exactly what feeds the FFT (the packed, center-cropped luma).
 */
export const DISPLAY_R_FRAG = /* glsl */ `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_tex;
out vec4 fragColor;

void main() {
  float r = texture(u_tex, v_uv).r;
  fragColor = vec4(vec3(r), 1.0);
}
`;

/**
 * Re-packs the real channel of a complex texture as a fresh complex value
 * (real = src.r, imag = 0). Used to forward-FFT the *displayed* reconstruction
 * (which keeps only the real part), so the round-trip spectrum reflects the
 * real image you see rather than trivially returning the filtered spectrum.
 */
export const PACK_REAL_FRAG = /* glsl */ `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_src;
out vec4 fragColor;

void main() {
  float r = texture(u_src, v_uv).r;
  fragColor = vec4(r, 0.0, 0.0, 1.0);
}
`;

/**
 * Multiplies a complex frequency texture by the (real) filter mask, writing the
 * filtered spectrum. The mask alpha is the attenuation (1 = pass, 0 = block).
 *
 * The mask is authored in fftshifted (DC-centered) display space, so we sample
 * it at the same `fract(uv + 0.5)` offset the magnitude view uses. The mask
 * texture is uploaded with UNPACK_FLIP_Y_WEBGL = true so this single shift also
 * undoes the vertical flip drawImage introduces when presenting the spectrum.
 */
export const MULTIPLY_FRAG = /* glsl */ `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_freq;
uniform sampler2D u_mask;
out vec4 fragColor;

void main() {
  vec2 c = texture(u_freq, v_uv).rg;
  float m = texture(u_mask, fract(v_uv + 0.5)).a;
  fragColor = vec4(c * m, 0.0, 1.0);
}
`;

/**
 * Renders the magnitude spectrum of a complex frequency texture.
 *
 *  - fftshift: the FFT stores DC at texel (0,0); we sample with a half-texture
 *    offset so DC lands in the center of the view.
 *  - log scale: spectra span many orders of magnitude, so we show
 *    log(1 + |F|) scaled by u_logGain (and clamped).
 *  - colormap: 0 = grayscale, 1 = viridis.
 */
export const MAGNITUDE_FRAG = /* glsl */ `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_freq;
uniform float u_logGain;
uniform int u_colormap;
out vec4 fragColor;

// Polynomial viridis approximation (Matt Zucker / Inigo Quilez).
vec3 viridis(float t) {
  const vec3 c0 = vec3(0.2777273272234177, 0.005407344544966578, 0.3340998053353061);
  const vec3 c1 = vec3(0.1050930431085774, 1.404613529898575, 1.384590162594685);
  const vec3 c2 = vec3(-0.3308618287255563, 0.214847559468213, 0.09509516302823659);
  const vec3 c3 = vec3(-4.634230498983486, -5.799100973351585, -19.33244095627987);
  const vec3 c4 = vec3(6.228269936347081, 14.17993336680509, 56.69055260068105);
  const vec3 c5 = vec3(4.776384997670288, -13.74514537774601, -65.35303263337234);
  const vec3 c6 = vec3(-5.435455855934631, 4.645852612178535, 26.3124352495832);
  return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

void main() {
  // fftshift: move DC (texel 0,0) to the center of the display.
  vec2 uv = fract(v_uv + 0.5);
  vec2 c = texture(u_freq, uv).rg;
  float mag = length(c);
  float t = clamp(log(1.0 + mag) * u_logGain, 0.0, 1.0);
  vec3 color = u_colormap == 1 ? viridis(t) : vec3(t);
  fragColor = vec4(color, 1.0);
}
`;

/**
 * Packs a source image into a complex texture for the FFT: R = luma (real
 * part), G = 0 (imaginary part). Performs a center-square crop so a non-square
 * webcam frame maps cleanly onto the N x N transform, plus optional Y flip.
 *
 * u_srcSize is the source texture's pixel dimensions, used for the crop aspect.
 */
export const PACK_FRAG = /* glsl */ `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_src;
uniform bool u_flipY;
uniform vec2 u_srcSize;
out vec4 fragColor;

void main() {
  vec2 uv = v_uv;
  // Center-square crop: shrink the longer axis around 0.5.
  float aspect = u_srcSize.x / u_srcSize.y;
  if (aspect > 1.0) {
    uv.x = 0.5 + (uv.x - 0.5) / aspect;
  } else {
    uv.y = 0.5 + (uv.y - 0.5) * aspect;
  }
  if (u_flipY) uv.y = 1.0 - uv.y;

  vec3 rgb = texture(u_src, uv).rgb;
  float luma = dot(rgb, vec3(0.2126, 0.7152, 0.0722));
  fragColor = vec4(luma, 0.0, 0.0, 1.0); // (real, imag, -, -)
}
`;

/**
 * One radix-2 Stockham auto-sort FFT butterfly stage along one axis.
 *
 * Run log2(N) times per axis, ping-ponging between two complex textures, with
 * u_subtransformSize doubling each pass (2, 4, ..., N). Stockham needs no
 * separate bit-reversal pass: each output texel gathers its even/odd inputs and
 * applies its own twiddle. Two outputs i and i+N/2 share the same even/odd; for
 * the upper one the twiddle argument is shifted by PI, which negates it -- so
 * the single expression `even + twiddle*odd` yields both butterfly branches.
 *
 * Forward is unnormalized; the inverse applies its 1/N^2 scale via
 * u_normalization on the final pass (linear, so the placement only affects
 * numerical conditioning).
 */
export const FFT_FRAG = /* glsl */ `#version 300 es
precision highp float;
precision highp sampler2D;
in vec2 v_uv;
uniform sampler2D u_src;
uniform float u_subtransformSize;
uniform float u_normalization;
uniform bool u_horizontal;
uniform bool u_forward;
uniform vec2 u_resolution;
out vec4 fragColor;

const float PI = 3.141592653589793;

vec2 cmul(vec2 a, vec2 b) {
  return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

void main() {
  float index = (u_horizontal ? gl_FragCoord.x : gl_FragCoord.y) - 0.5;
  float halfSize = u_subtransformSize * 0.5;
  float evenIndex =
    floor(index / u_subtransformSize) * halfSize + mod(index, halfSize);

  vec2 evenPos, oddPos;
  if (u_horizontal) {
    float y = gl_FragCoord.y;
    evenPos = vec2((evenIndex + 0.5) / u_resolution.x, y / u_resolution.y);
    oddPos = vec2((evenIndex + u_resolution.x * 0.5 + 0.5) / u_resolution.x,
                  y / u_resolution.y);
  } else {
    float x = gl_FragCoord.x;
    evenPos = vec2(x / u_resolution.x, (evenIndex + 0.5) / u_resolution.y);
    oddPos = vec2(x / u_resolution.x,
                  (evenIndex + u_resolution.y * 0.5 + 0.5) / u_resolution.y);
  }

  vec2 even = texture(u_src, evenPos).rg;
  vec2 odd = texture(u_src, oddPos).rg;

  float twiddleArg =
    (u_forward ? -1.0 : 1.0) * 2.0 * PI * (index / u_subtransformSize);
  vec2 twiddle = vec2(cos(twiddleArg), sin(twiddleArg));

  vec2 result = (even + cmul(twiddle, odd)) * u_normalization;
  fragColor = vec4(result, 0.0, 1.0);
}
`;
