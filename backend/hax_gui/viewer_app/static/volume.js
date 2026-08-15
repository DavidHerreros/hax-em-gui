// WebGL2 3D-texture volume ray-marcher (MIP + emission/absorption with a simple
// transfer function). Full-screen quad + ray-box intersection in the model-space cube
// the grids occupy, which the backend picks to coincide exactly with the marching-cubes
// surface so the map does not shift when you switch render modes; orbit camera shared
// in style with gl.js. Hand-written, no dependencies.
//
// Composites up to MAXV volumes in ONE march: each slot carries its own model-space
// box, contour threshold and colour, and every step samples all of them, so several
// maps can be compared in volume mode the way they can in surface mode. A single map
// keeps the exact rendering this file has always produced (grayscale MIP, tinted glow).
// Slot 0 additionally carries the morph partner (uVolB + uMix): a morph is a linear
// blend of two densities, and blending on the GPU rather than uploading a texture per
// frame is what makes playback run at screen rate.
import { projection, lookAt, multiply, invert, mulVec4, normalize } from "./mat4.js";
import { createOrbit } from "./orbit.js";

const MAXV = 4;

// Iso mode's lights, copied from gl.js so a shaded isosurface and a marching-cubes
// surface of the same map are lit identically — the point of the mode is that you can
// switch to it without the map appearing to change material. These are ChimeraX's
// Material defaults and its camera-space light directions (the direction each light
// *shines*, hence the minus signs in the shader), and the three presets are ChimeraX's
// simple / soft / full. `move_lights_with_camera` is why the directions are camera-space
// constants that drawScene carries back into model space every frame.
const KEY_DIR = normalize([0.577, -0.577, -0.577]);
const FILL_DIR = normalize([-0.2, -0.2, -0.959]);
const LIGHT_MODES = { standard: 0, smooth: 1, raytraced: 2 };

const VERT = `#version 300 es
precision highp float;
layout(location=0) in vec2 aPos;
out vec2 vUv;
void main() { vUv = aPos * 0.5 + 0.5; gl_Position = vec4(aPos, 0.0, 1.0); }`;

const FRAG = `#version 300 es
precision highp float;
precision highp sampler3D;
in vec2 vUv;
out vec4 outColor;
// GLSL ES 3.0 forbids dynamically indexed sampler arrays, hence four named samplers
// and an if-chain; the per-slot scalars ride in ordinary arrays.
uniform sampler3D uVol0;
uniform sampler3D uVol1;
uniform sampler3D uVol2;
uniform sampler3D uVol3;
uniform sampler3D uVolB;    // morph partner of slot 0; only sampled while mixing
uniform float uMix;
uniform int uCount;
uniform mat4 uInvVP;
uniform vec3 uEye;
uniform vec3 uBg;
uniform vec3 uBoxLo;        // union of the slot boxes: the segment the ray marches
uniform vec3 uBoxHi;
uniform vec3 uBoxLoA[${MAXV}];   // per-slot model-space boxes (maps can differ in size)
uniform vec3 uBoxHiA[${MAXV}];
uniform float uThrA[${MAXV}];    // per-slot 0..1 normalized contour
uniform vec3 uColA[${MAXV}];
uniform float uGamma;
uniform float uBright;
uniform int uMode;          // 0 = MIP, 1 = emission/absorption, 2 = shaded isosurface
uniform int uSteps;

// Iso mode. uVoxA is one texel of each slot's grid in NORMALIZED texture units (1/nx …),
// which is the step the central-difference gradient is taken over — a fixed model-space
// epsilon would straddle several voxels on a coarse grid and less than one on a fine one.
// The light directions arrive already rotated into model space (see drawScene): the march
// happens in model space, and the lights are fixed to the camera as in gl.js, so the
// rotation has to be undone once per frame rather than per fragment.
uniform vec3 uVoxA[${MAXV}];
uniform int uLightMode;     // 0 = standard, 1 = smooth, 2 = ray traced
uniform vec3 uKeyDir;       // model space, direction the key light shines
uniform vec3 uFillDir;

// Clip slab, the same model-space plane pair the surface renderer cuts with: keep the
// samples whose signed distance along the unit normal uClipN lands inside uClipRange,
// everything else contributes nothing. That is how ChimeraX's region clipping behaves
// for image renderings, generalised off the coordinate axes. Because the planes live in
// model space rather than in each slot's texture space, a multi-map scene now cuts every
// map on one plane instead of at the same fraction of each map's own box — which is the
// point, since the surfaces have always cut that way.
uniform int uClipOn;
uniform vec3 uClipN;
uniform vec2 uClipRange;

vec3 toWorld(vec3 ndc) { vec4 p = uInvVP * vec4(ndc, 1.0); return p.xyz / p.w; }

float sampleVol(int i, vec3 p) {
  if (i == 0) { float a = texture(uVol0, p).r; return uMix > 0.0 ? mix(a, texture(uVolB, p).r, uMix) : a; }
  if (i == 1) return texture(uVol1, p).r;
  if (i == 2) return texture(uVol2, p).r;
  return texture(uVol3, p).r;
}

// Density of slot i at a model-space point: zero outside its own box (the sampler's
// edge clamping would otherwise smear boundary values outward when boxes differ) and
// zero inside the clipped-away region.
float density(int i, vec3 mp) {
  vec3 p = (mp - uBoxLoA[i]) / (uBoxHiA[i] - uBoxLoA[i]);
  if (any(lessThan(p, vec3(0.0))) || any(greaterThan(p, vec3(1.0)))) return 0.0;
  if (uClipOn == 1) {
    float d = dot(mp, uClipN);
    if (d < uClipRange.x || d > uClipRange.y) return 0.0;
  }
  return sampleVol(i, p);
}

// Diverging (signed) grids put ZERO at 0.5 — outside the box or the clip slab the
// honest value is "no difference", i.e. 0.5, not 0 (0 would read as maximal loss).
uniform int uDiverging;
float densityN(vec3 mp) {
  vec3 p = (mp - uBoxLoA[0]) / (uBoxHiA[0] - uBoxLoA[0]);
  if (any(lessThan(p, vec3(0.0))) || any(greaterThan(p, vec3(1.0)))) return 0.5;
  if (uClipOn == 1) {
    float d = dot(mp, uClipN);
    if (d < uClipRange.x || d > uClipRange.y) return 0.5;
  }
  return sampleVol(0, p);
}

// ---- iso mode helpers -----------------------------------------------------

// Outward unit normal of slot i's isosurface at a model-space point: the central-difference
// density gradient, negated because density falls off *outwards*. Each offset is one texel
// of that slot's grid converted to model units, so the chain rule that carries a
// texture-space gradient into model space — divide each component by the box extent — is
// already paid by measuring the difference over that model-space distance. Anisotropic
// boxes and non-cubic grids therefore come out right, which a gradient taken in normalized
// texture space and used as-is would not. Sampling goes through density(), so slot 0's
// normals follow the morph blend and the clip slab cuts normals as well as geometry.
vec3 isoNormal(int i, vec3 mp) {
  vec3 h = max(uVoxA[i] * (uBoxHiA[i] - uBoxLoA[i]), vec3(1e-6));
  float dx = density(i, mp + vec3(h.x, 0.0, 0.0)) - density(i, mp - vec3(h.x, 0.0, 0.0));
  float dy = density(i, mp + vec3(0.0, h.y, 0.0)) - density(i, mp - vec3(0.0, h.y, 0.0));
  float dz = density(i, mp + vec3(0.0, 0.0, h.z)) - density(i, mp - vec3(0.0, 0.0, h.z));
  vec3 g = vec3(dx, dy, dz) / (2.0 * h);
  float len = length(g);
  return len < 1e-8 ? vec3(0.0) : -g / len;    // zero-length: the caller falls back
}

// Is any slot's interior here? Shadow and occlusion rays only need "something opaque is
// in the way", not which map it belongs to.
bool isoOccupied(vec3 mp) {
  for (int i = 0; i < ${MAXV}; i++) {
    if (i >= uCount) break;
    // The floor keeps a contour dragged to zero from declaring empty space solid — outside
    // the box and outside the clip slab density() returns exactly 0.
    if (density(i, mp) >= max(uThrA[i], 1e-4)) return true;
  }
  return false;
}

// Where a ray leaves the union box — the length a secondary ray can usefully march.
float boxFar(vec3 o, vec3 d) {
  vec3 inv = 1.0 / d;
  vec3 t0 = (uBoxLo - o) * inv, t1 = (uBoxHi - o) * inv;
  vec3 tmax = max(t0, t1);
  return min(min(tmax.x, tmax.y), tmax.z);
}

// Key-light shadow for the ray-traced preset, standing in for gl.js's 2048² shadow map:
// march towards the light and kill the key term on the first occupied sample. Twice the
// primary step, because a shadow's edge tolerates an error the silhouette does not, and
// the ray is started off the surface or step one would shadow the point against itself.
float isoShadow(vec3 p, vec3 n, float dt) {
  vec3 L = -uKeyDir;                    // uKeyDir is where the light shines, not where it is
  vec3 o = p + n * dt * 1.5;
  float tf = boxFar(o, L);
  float st = dt * 2.0;
  for (int s = 1; s < 256; s++) {
    float t = st * float(s);
    if (t > tf) break;
    if (isoOccupied(o + L * t)) return 0.0;
  }
  return 1.0;
}

// Cheap local ambient occlusion for the two presets whose look depends on it: gl.js gets
// this from 64 real shadow maps, which is out of reach inside a per-fragment ray march, so
// sample a narrow cone around the normal at two radii and darken the ambient term by the
// fraction that lands inside matter. It reproduces creases and pockets, not a whole domain
// shadowing its neighbour — deliberately subtle rather than wrong at a large scale.
float isoAO(vec3 p, vec3 n, float dt) {
  vec3 up = abs(n.z) < 0.9 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
  vec3 tx = normalize(cross(up, n)), ty = cross(n, tx);
  float occ = 0.0;
  for (int k = 0; k < 6; k++) {
    float a = 6.2831853 * float(k) / 6.0;
    vec3 d = normalize(n + 0.9 * (cos(a) * tx + sin(a) * ty));
    if (isoOccupied(p + d * (3.0 * dt))) occ += 1.0;
    if (isoOccupied(p + d * (7.0 * dt))) occ += 1.0;
  }
  return occ / 12.0;
}

// ChimeraX's lighting equation (gl.js's fragment shader), with the preset intensities and
// the Material reflectivities inlined: an isosurface drawn here has to match the same map
// drawn as a mesh, or switching modes would look like a change of material. rd is the
// unit eye-to-point direction, gl.js's v1 — every vector here is model space, and the
// model transform is rigid, so the dot products are the camera-space ones.
vec3 isoShade(int i, vec3 p, vec3 n, vec3 rd, float dt) {
  float lkey  = uLightMode == 1 ? 0.0 : (uLightMode == 2 ? 0.7 : 1.0);
  float lfill = uLightMode == 1 ? 0.0 : (uLightMode == 2 ? 0.3 : 0.5);
  float lamb  = uLightMode == 1 ? 1.5 : (uLightMode == 2 ? 0.8 : 0.4);
  float kf = max(-dot(n, uKeyDir), 0.0);
  float ff = max(-dot(n, uFillDir), 0.0);
  vec3 R = normalize(reflect(uKeyDir, n));
  float sf = pow(max(-dot(R, rd), 0.0), 30.0);      // MATERIAL.exponent
  float shadow = uLightMode == 2 ? isoShadow(p, n, dt) : 1.0;
  float amb = uLightMode == 0 ? 1.0 : 1.0 - 0.75 * isoAO(p, n, dt);
  vec3 base = uColA[i];
  vec3 Iamb = base * (0.8 * lamb) * amb;                                  // MATERIAL.ambient
  vec3 Idiff = base * (0.8 * lkey * kf * shadow + 0.8 * lfill * ff);      // MATERIAL.diffuse
  vec3 Ispec = vec3(0.3 * lkey * sf * shadow);                            // MATERIAL.specular
  return Iamb + Idiff + Ispec;
}

void main() {
  vec3 ro = uEye;
  vec3 rd = normalize(toWorld(vec3(vUv * 2.0 - 1.0, 1.0)) - ro);
  vec3 inv = 1.0 / rd;
  vec3 t0 = (uBoxLo - ro) * inv, t1 = (uBoxHi - ro) * inv;
  vec3 tmin = min(t0, t1), tmax = max(t0, t1);
  float tn = max(max(tmin.x, tmin.y), max(tmin.z, 0.0));
  float tf = min(min(tmax.x, tmax.y), tmax.z);
  if (tf <= tn) { outColor = vec4(uBg, 1.0); return; }

  int steps = uSteps;
  float dt = (tf - tn) / float(steps);
  float t = tn;

  // Signed difference rendering: two projections in one march — how far the diff
  // reaches above zero (blue, gained) and below it (red, lost). The contour
  // threshold does not apply (a signed field has no single inside), but gamma and
  // brightness do. Same hues as the painted surface and the diverging slices.
  if (uDiverging == 1) {
    const vec3 DBLUE = vec3(0.25, 0.44, 0.83);
    const vec3 DRED = vec3(0.83, 0.25, 0.25);
    if (uMode == 0) {                        // two-sided MIP
      float mpos = 0.0, mneg = 0.0;
      for (int s = 0; s < 512; s++) {
        if (s >= steps) break;
        float d = densityN(ro + rd * t) * 2.0 - 1.0;
        mpos = max(mpos, d); mneg = max(mneg, -d);
        t += dt;
      }
      float vp = clamp(pow(clamp(mpos, 0.0, 1.0), uGamma) * uBright, 0.0, 1.0);
      float vn = clamp(pow(clamp(mneg, 0.0, 1.0), uGamma) * uBright, 0.0, 1.0);
      vec3 srgb = (1.0 - vp * DBLUE) * (1.0 - vn * DRED);   // screen-blend the two sides
      outColor = vec4(uBg * (1.0 - vp) * (1.0 - vn) + (1.0 - srgb), 1.0);
    } else {                                 // two-sided emission
      vec3 acc = vec3(0.0); float alpha = 0.0;
      for (int s = 0; s < 512; s++) {
        if (s >= steps) break;
        float d = densityN(ro + rd * t) * 2.0 - 1.0;
        float dp = pow(clamp(d, 0.0, 1.0), uGamma) * uBright;
        float dn = pow(clamp(-d, 0.0, 1.0), uGamma) * uBright;
        float ap = clamp(dp * 0.12, 0.0, 1.0) * (1.0 - alpha);
        acc += ap * DBLUE; alpha += ap;
        float an = clamp(dn * 0.12, 0.0, 1.0) * (1.0 - alpha);
        acc += an * DRED; alpha += an;
        if (alpha >= 0.98) break;
        t += dt;
      }
      outColor = vec4(acc + uBg * (1.0 - alpha), 1.0);
    }
    return;
  }

  if (uMode == 0) {                          // per-map maximum-intensity projection
    float mv[${MAXV}];
    for (int i = 0; i < ${MAXV}; i++) mv[i] = 0.0;
    for (int s = 0; s < 512; s++) {
      if (s >= steps) break;
      vec3 mp = ro + rd * t;
      for (int i = 0; i < ${MAXV}; i++) {
        if (i >= uCount) break;
        float v = clamp((density(i, mp) - uThrA[i]) / max(1.0 - uThrA[i], 1e-3), 0.0, 1.0);
        mv[i] = max(mv[i], v);
      }
      t += dt;
    }
    if (uCount == 1) {                       // the classic single-map grayscale MIP
      float v = pow(mv[0], uGamma) * uBright;
      outColor = vec4(mix(uBg, vec3(1.0), clamp(v, 0.0, 1.0)), 1.0);
      return;
    }
    // Several maps: tint each projection with its map's colour and screen-blend them,
    // so overlaps brighten towards white instead of one map erasing the other.
    vec3 srgb = vec3(1.0);
    float cover = 1.0;
    for (int i = 0; i < ${MAXV}; i++) {
      if (i >= uCount) break;
      float v = clamp(pow(mv[i], uGamma) * uBright, 0.0, 1.0);
      srgb *= (1.0 - v * uColA[i]);
      cover *= (1.0 - v);
    }
    outColor = vec4(uBg * cover + (1.0 - srgb), 1.0);
  } else if (uMode == 2) {                   // shaded isosurface
    // March until some slot's density crosses its contour from below, then bisect and
    // shade that point opaquely. Stopping at the first hit is what makes iso cheaper than
    // MIP or emission, both of which must always run the whole segment.
    float prev[${MAXV}];
    for (int i = 0; i < ${MAXV}; i++) prev[i] = 0.0;
    for (int i = 0; i < ${MAXV}; i++) {
      if (i >= uCount) break;
      prev[i] = density(i, ro + rd * tn);
    }
    float tp = tn;
    int hitSlot = -1;
    float tHit = 0.0;
    for (int s = 1; s < 512; s++) {
      if (s > steps) break;
      float tc = tn + dt * float(s);
      vec3 mp = ro + rd * tc;
      float best = 1e30;
      int bi = -1;
      for (int i = 0; i < ${MAXV}; i++) {
        if (i >= uCount) break;
        float v = density(i, mp);
        float thr = max(uThrA[i], 1e-4);   // a zero contour has no crossing at all: see isoOccupied
        if (prev[i] < thr && v >= thr) {
          // One step spans several voxels at the default step count, and shading its far
          // end would terrace the surface into visible shells — bisect the bracket first.
          float a = tp, b = tc;
          for (int k = 0; k < 5; k++) {
            float m = 0.5 * (a + b);
            if (density(i, ro + rd * m) < thr) a = m; else b = m;
          }
          if (b < best) { best = b; bi = i; }   // two maps can cross in one step: nearest wins
        }
        prev[i] = v;
      }
      if (bi >= 0) { hitSlot = bi; tHit = best; break; }
      tp = tc;
    }
    if (hitSlot < 0) { outColor = vec4(uBg, 1.0); return; }
    vec3 hp = ro + rd * tHit;
    vec3 n = isoNormal(hitSlot, hp);
    // A cut face across a locally flat region has no gradient worth normalising; face the
    // viewer instead of shading from noise.
    if (dot(n, n) < 0.5) n = -rd;
    if (dot(n, rd) > 0.0) n = -n;              // two-sided, as gl.js is via gl_FrontFacing
    outColor = vec4(isoShade(hitSlot, hp, n, rd, dt), 1.0);
  } else {                                   // front-to-back emission/absorption
    vec3 acc = vec3(0.0); float alpha = 0.0;
    for (int s = 0; s < 512; s++) {
      if (s >= steps) break;
      vec3 mp = ro + rd * t;
      for (int i = 0; i < ${MAXV}; i++) {
        if (i >= uCount) break;
        float d = pow(clamp((density(i, mp) - uThrA[i]) / max(1.0 - uThrA[i], 1e-3), 0.0, 1.0), uGamma) * uBright;
        float a = clamp(d * 0.12, 0.0, 1.0) * (1.0 - alpha);
        acc += a * uColA[i]; alpha += a;
      }
      if (alpha >= 0.98) break;
      t += dt;
    }
    outColor = vec4(acc + uBg * (1.0 - alpha), 1.0);
  }
}`;

export function createVolume(canvas) {
  const gl = canvas.getContext("webgl2", { antialias: false, depth: false, preserveDrawingBuffer: true, premultipliedAlpha: false });
  if (!gl) throw new Error("WebGL2 is not available in this browser.");

  const prog = link(gl, VERT, FRAG);
  const u = (n) => gl.getUniformLocation(prog, n);
  const uInvVP = u("uInvVP"), uEye = u("uEye"), uBg = u("uBg"),
    uBoxLo = u("uBoxLo"), uBoxHi = u("uBoxHi"),
    uBoxLoA = u("uBoxLoA"), uBoxHiA = u("uBoxHiA"), uThrA = u("uThrA"), uColA = u("uColA"),
    uGamma = u("uGamma"), uBright = u("uBright"), uMode = u("uMode"), uSteps = u("uSteps"),
    uVolN = [u("uVol0"), u("uVol1"), u("uVol2"), u("uVol3")],
    uVolB = u("uVolB"), uMix = u("uMix"), uCount = u("uCount"),
    uClipOn = u("uClipOn"), uClipN = u("uClipN"), uClipRange = u("uClipRange"), uDiverging = u("uDiverging"),
    uVoxA = u("uVoxA"), uLightMode = u("uLightMode"), uKeyDirL = u("uKeyDir"), uFillDirL = u("uFillDir");

  // full-screen quad
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const quad = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quad);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);

  function makeVolTex() {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_3D, t);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
    // A 1-voxel placeholder makes the texture complete from birth: unused slots are
    // still bound (sampling an incomplete unit is undefined behaviour on some drivers).
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.R8, 1, 1, 1, 0, gl.RED, gl.UNSIGNED_BYTE, new Uint8Array(1));
    return t;
  }
  const slotTex = Array.from({ length: MAXV }, makeVolTex);
  const texB = makeVolTex();     // the morph's far end; only sampled while mixing

  let bg = [0.016, 0.024, 0.06];
  let gamma = 1.4, bright = 1.2, mode = 0, steps = 160;
  let hasTex = false;
  let count = 1;
  // Per-slot state, flat-packed on upload. Slot 0's defaults are the values the
  // single-map viewer has always used.
  const slotBoxLo = Array.from({ length: MAXV }, () => [-1, -1, -1]);
  const slotBoxHi = Array.from({ length: MAXV }, () => [1, 1, 1]);
  const slotThr = new Float32Array(MAXV).fill(0.45);
  const slotCol = Array.from({ length: MAXV }, () => [0.784, 0.784, 0.784]);
  // Grid dimensions per slot, kept only so iso mode can take its gradient over one texel
  // of the grid it is actually shading. The placeholder texture is 1³ until a real upload.
  const slotDim = Array.from({ length: MAXV }, () => [64, 64, 64]);
  let lightMode = 0;             // iso lighting preset; the other modes ignore it
  // Model-space clip slab, shared with the surface renderer: keep clipRange[0] <=
  // dot(p, clipN) <= clipRange[1]. A null normal is no clipping at all.
  let clipN = null, clipRange = [0, 0];
  let grid = null;   // { nx, ny, nz } of slot 0; null until one is, which is what the camera tests
  let gridB = null;  // the morph's far end, same shape; null when not morphing
  let mix = 0;       // blend weight in [0,1]; 0 = slot 0 alone
  let diverging = false;   // slot 0 holds a signed difference grid (zero at u8 midpoint)

  // The model-space box everything on screen occupies — the union over the active
  // slots. It bounds the ray march, the camera's pivot probe and the framing rule.
  function unionBox() {
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < count; i++) {
      for (let a = 0; a < 3; a++) {
        lo[a] = Math.min(lo[a], slotBoxLo[i][a]);
        hi[a] = Math.max(hi[a], slotBoxHi[i][a]);
      }
    }
    return { lo, hi };
  }

  // Where the sight line through the centre of the window meets the maps, as ChimeraX
  // reports it for a map drawn as an image: the *midpoint* of the segment the line spends
  // inside the data box — Volume.first_intercept's `image_shown` branch returns
  // 0.5*(xyz_in + xyz_out), not the first voxel above the contour. A ray-march has no
  // surface to hit, so ChimeraX does not pretend it has one, and the camera turns this into
  // its centre of rotation when you have zoomed in past the box.
  function centreDepth() {
    if (!grid) return null;
    const { lo, hi } = unionBox();
    const { origin, dir } = orbit.axisRay();   // model space, where the volumes fill the union box
    let tn = 0, tf = Infinity;                 // slab test against that box
    for (let a = 0; a < 3; a++) {
      if (Math.abs(dir[a]) < 1e-6) { if (origin[a] < lo[a] || origin[a] > hi[a]) return null; continue; }
      let t0 = (lo[a] - origin[a]) / dir[a], t1 = (hi[a] - origin[a]) / dir[a];
      if (t0 > t1) { const s = t0; t0 = t1; t1 = s; }
      tn = Math.max(tn, t0); tf = Math.min(tf, t1);
    }
    if (tf <= tn) return null;                 // the box is behind us, or we missed it
    return 0.5 * (tn + tf);
  }

  // The model-space box the grids occupy — ChimeraX's bounds for an image-rendered map,
  // which are the data box (the image drawing is the box; there is no contour to bound).
  function displayedBounds() {
    return grid ? unionBox() : null;
  }

  // 1.41 = the height this view has always shown (old camera: 3.4·tan(22.5°)), so the map
  // keeps its size and its registration with the surface view across a render-mode switch.
  const orbit = createOrbit(canvas, () => render(), null,
    { fitRadius: 1.41, probeDepth: centreDepth, modelBounds: displayedBounds });

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr)), h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  }
  const flat3 = (arr) => new Float32Array(arr.flat());
  function drawScene(alpha = 1) {
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(bg[0], bg[1], bg[2], alpha);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (!hasTex) return;
    const aspect = canvas.width / Math.max(1, canvas.height);
    const eye = orbit.eye, M = orbit.model;
    // march in MODEL space: transform = proj·view·M, so invert(mvp) maps clip→model
    // and the eye in model space is M⁻¹·eye_world (M = translate(pan)·R; camera is fixed on +Z).
    const view = lookAt(eye, [0, 0, 0], [0, 1, 0]);
    const mvp = multiply(multiply(projection(aspect, 0.01, 100), view), M);
    const ivp = invert(mvp); if (!ivp) return;
    const Minv = invert(M); if (!Minv) return;
    const em = mulVec4(Minv, [eye[0], eye[1], eye[2], 1]);
    // The lights are fixed to the camera (gl.js does the same, following ChimeraX's
    // move_lights_with_camera), but the march runs in model space, so each camera-space
    // direction is carried back through the rotation of view·M — rigid, so the inverse
    // rotation is the transpose, i.e. multiply by the 3x3 from the left. This is what keeps
    // the lit side facing the viewer's upper left however far the model is orbited.
    const mv = multiply(view, M);
    const toModelDir = (d) => [
      mv[0] * d[0] + mv[1] * d[1] + mv[2] * d[2],
      mv[4] * d[0] + mv[5] * d[1] + mv[6] * d[2],
      mv[8] * d[0] + mv[9] * d[1] + mv[10] * d[2],
    ];
    const ub = unionBox();
    gl.useProgram(prog);
    gl.uniformMatrix4fv(uInvVP, false, ivp);
    gl.uniform3fv(uEye, [em[0], em[1], em[2]]);
    gl.uniform3fv(uBg, bg);
    gl.uniform3fv(uBoxLo, ub.lo); gl.uniform3fv(uBoxHi, ub.hi);
    gl.uniform3fv(uBoxLoA, flat3(slotBoxLo)); gl.uniform3fv(uBoxHiA, flat3(slotBoxHi));
    gl.uniform1fv(uThrA, slotThr); gl.uniform3fv(uColA, flat3(slotCol));
    gl.uniform1f(uGamma, gamma); gl.uniform1f(uBright, bright);
    gl.uniform1i(uMode, mode); gl.uniform1i(uSteps, steps);
    gl.uniform3fv(uVoxA, flat3(slotDim.map((d) => [1 / Math.max(1, d[0]), 1 / Math.max(1, d[1]), 1 / Math.max(1, d[2])])));
    gl.uniform1i(uLightMode, lightMode);
    gl.uniform3fv(uKeyDirL, toModelDir(KEY_DIR)); gl.uniform3fv(uFillDirL, toModelDir(FILL_DIR));
    gl.uniform1i(uCount, count);
    gl.uniform1i(uDiverging, diverging ? 1 : 0);
    gl.uniform1f(uMix, gridB ? mix : 0);
    gl.uniform1i(uClipOn, clipN ? 1 : 0);
    if (clipN) { gl.uniform3fv(uClipN, clipN); gl.uniform2fv(uClipRange, clipRange); }
    // Every sampler stays bound (placeholder textures make the unused ones complete).
    for (let i = 0; i < MAXV; i++) {
      gl.activeTexture(gl.TEXTURE0 + i); gl.bindTexture(gl.TEXTURE_3D, slotTex[i]); gl.uniform1i(uVolN[i], i);
    }
    gl.activeTexture(gl.TEXTURE0 + MAXV); gl.bindTexture(gl.TEXTURE_3D, gridB ? texB : slotTex[0]); gl.uniform1i(uVolB, MAXV);
    gl.bindVertexArray(vao); gl.drawArrays(gl.TRIANGLES, 0, 3); gl.bindVertexArray(null);
  }
  function render() { resize(); drawScene(1); }

  function upload(target, nx, ny, nz, u8) {
    gl.bindTexture(gl.TEXTURE_3D, target);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.R8, nx, ny, nz, 0, gl.RED, gl.UNSIGNED_BYTE, u8);
  }

  function setVolume(nx, ny, nz, u8, lo, hi) {
    upload(slotTex[0], nx, ny, nz, u8);
    grid = { nx, ny, nz };
    slotDim[0] = [nx, ny, nz];
    count = 1;
    if (lo && hi) { slotBoxLo[0] = lo.slice(); slotBoxHi[0] = hi.slice(); }   // the cube the surface mesh uses, so the two modes coincide
    hasTex = true; orbit.invalidatePivot(); render();           // a new map is a new box to pivot on
  }

  // Several maps in one scene: each slot is {nx, ny, nz, data, lo, hi, color, threshold}.
  function setSlots(slots) {
    const n = Math.min(slots.length, MAXV);
    if (!n) return;
    for (let i = 0; i < n; i++) {
      const s = slots[i];
      upload(slotTex[i], s.nx, s.ny, s.nz, s.data);
      slotDim[i] = [s.nx, s.ny, s.nz];
      if (s.lo && s.hi) { slotBoxLo[i] = s.lo.slice(); slotBoxHi[i] = s.hi.slice(); }
      if (s.color) slotCol[i] = s.color.slice();
      if (Number.isFinite(s.threshold)) slotThr[i] = Math.max(0, Math.min(1, s.threshold));
    }
    grid = { nx: slots[0].nx, ny: slots[0].ny, nz: slots[0].nz };
    count = n;
    hasTex = true; orbit.invalidatePivot(); render();
  }

  window.addEventListener("resize", render);

  return {
    setVolume, setSlots, render,
    // How many maps a composited scene can hold; the controller reports truncation.
    maxSlots: MAXV,
    // The far end of a morph. Pass null to stop morphing and go back to one volume. The
    // camera is untouched either way: the two ends share a grid, so they share the box the
    // pivot is derived from.
    setVolumeB(nx, ny, nz, u8) {
      if (!u8) { gridB = null; mix = 0; render(); return; }
      upload(texB, nx, ny, nz, u8);
      gridB = { nx, ny, nz };
      render();
    },
    // Blend weight, 0 = the near map, 1 = the far one. Called every animation frame, so
    // it must stay cheap — which it is, being a few uniform writes and a draw.
    setMix(t) { mix = Math.max(0, Math.min(1, t)); render(); },
    setMode(m) { mode = m === "emission" ? 1 : m === "iso" ? 2 : 0; render(); },
    // Iso lighting preset, the same three names gl.js takes, so one control can drive
    // both renderers and the two views stay lit alike. Ignored by MIP and glow.
    setLighting(name) { lightMode = LIGHT_MODES[name] !== undefined ? LIGHT_MODES[name] : 0; render(); },
    // Signed-difference rendering of slot 0 (red = lost, blue = gained). The caller
    // is responsible for having uploaded a symmetric-range grid (see /api/paint/grid).
    setDiverging(on) { diverging = !!on; render(); },
    // No pivot invalidation: an image-rendered map pivots on its box, and the contour does
    // not move the box. Same in ChimeraX, where changing the level of a map shown as an
    // image leaves its bounds — and so its centre of rotation — where they were.
    setThreshold(v) { slotThr[0] = Math.max(0, Math.min(1, v)); render(); },
    setSlotThreshold(i, v) { if (i >= 0 && i < MAXV) { slotThr[i] = Math.max(0, Math.min(1, v)); render(); } },
    setGamma(g) { gamma = Math.max(0.2, g); render(); },
    // Clip slab: keep lo <= dot(p, n) <= hi, with `n` a unit normal in the model-space
    // [-1,1] cube — the same call the surface renderer takes, so one cut serves both
    // render modes. Pass a null normal to clear.
    setClip(n, lo, hi) { clipN = n ? [n[0], n[1], n[2]] : null; clipRange = [lo, hi]; render(); },
    setBrightness(b) { bright = Math.max(0.1, b); render(); },
    setColor(rgb) { slotCol[0] = rgb.slice(); render(); },
    setSlotColor(i, rgb) { if (i >= 0 && i < MAXV) { slotCol[i] = rgb.slice(); render(); } },
    setBackground(rgb) { bg = rgb; render(); },
    resetView() { orbit.reset(); },
    getCamera() { return orbit.getState(); },   // handed to the other renderer on a render-mode switch
    setCamera(cam) { orbit.setState(cam); },
    captureStream(fps) { return canvas.captureStream(fps); },
    async snapshot({ scale = 2, transparent = false } = {}) {
      const ow = canvas.width, oh = canvas.height;
      canvas.width = Math.floor(canvas.clientWidth * scale); canvas.height = Math.floor(canvas.clientHeight * scale);
      drawScene(transparent ? 0 : 1);
      const blob = await new Promise((res) => canvas.toBlob(res, "image/png"));
      canvas.width = ow; canvas.height = oh; render();
      return blob;
    },
  };
}

function link(gl, vsrc, fsrc) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vsrc));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fsrc));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error("shader link failed: " + gl.getProgramInfoLog(p));
  return p;
}
function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error("shader compile: " + gl.getShaderInfoLog(s));
  return s;
}
