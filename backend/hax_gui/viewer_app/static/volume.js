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

// gl.js's "multishadow" ambient light, cut down to a size a per-fragment ray march can
// afford: gl.js renders 64 depth maps from uniformly distributed directions and sums their
// visibility with a cosine weight; here the same sum is evaluated with occlusion rays from
// 16 directions. Sixteen is enough because the weight is cosine and the term is ambient —
// what matters is the *broad* gradient (deep in a pocket sees little of the sphere, an
// exposed face sees half of it), which is exactly what a local AO cone could not produce.
// The directions are a Fibonacci spiral over the whole sphere, generated once and baked
// into the shader source, and they live in MODEL space like gl.js's atlas, so the shading
// stays welded to the map while you orbit instead of swimming with the camera.
const MS_DIRS = (() => {
  const n = 16, ga = Math.PI * (3 - Math.sqrt(5)), out = [];
  for (let i = 0; i < n; i++) {
    const z = 1 - (2 * i + 1) / n, r = Math.sqrt(Math.max(0, 1 - z * z)), a = ga * i;
    out.push([r * Math.cos(a), r * Math.sin(a), z]);
  }
  return out;
})();
const MS_COUNT = MS_DIRS.length;
const msGLSL = `const vec3 MSDIR[${MS_COUNT}] = vec3[${MS_COUNT}](`
  + MS_DIRS.map((d) => `vec3(${d.map((v) => v.toFixed(6)).join(", ")})`).join(", ") + ");";

// Depth cue, mirroring gl.js (which mirrors ChimeraX): fractions of the near-to-far range
// derived from the scene's bounding sphere, and the padding/clamping that range is built
// with. Values copied from gl.js — the two views fade at the same distances or the same map
// looks nearer in one of them.
const DEPTH_CUE_START = 0.5, DEPTH_CUE_END = 1.0;
const NEAR_FAR_PAD = 0.01, MIN_NEAR_FRACTION = 0.001;

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
uniform float uMaxLodA[${MAXV}];   // last mip level of each slot's grid: floor(log2(max dim))
uniform int uLightMode;     // 0 = standard, 1 = smooth, 2 = ray traced
uniform vec3 uKeyDir;       // model space, direction the key light shines
uniform vec3 uFillDir;
// Depth cue. uDepthRow is the third row of view*M, which is all the march needs to turn a
// model-space hit into the camera-space axial distance the fade is a function of; the range
// is the pair of distances the fade runs between (see depthCueRange in the JS below).
uniform vec4 uDepthRow;
uniform vec2 uDepthRange;

${msGLSL}

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

// The volumes carry a mipmap chain now (iso's cone tracing walks it), which means the
// MIN filter is LINEAR_MIPMAP_LINEAR and a bare texture() would pick a level from screen-space
// derivatives — a ray march has no meaningful ones, and at silhouettes they explode, so every
// ordinary sample is pinned to level 0 with textureLod. Only the cone tracer chooses a level,
// and it computes it from the cone's radius rather than from derivatives.
float sampleVol(int i, vec3 p) {
  if (i == 0) { float a = textureLod(uVol0, p, 0.0).r; return uMix > 0.0 ? mix(a, textureLod(uVolB, p, 0.0).r, uMix) : a; }
  if (i == 1) return textureLod(uVol1, p, 0.0).r;
  if (i == 2) return textureLod(uVol2, p, 0.0).r;
  return textureLod(uVol3, p, 0.0).r;
}

// Same slot chain at an explicit mip level. Slot 0 keeps blending with the morph partner:
// a morphing map has to occlude and shadow as the blend it is showing, not as its near end.
float sampleVolLod(int i, vec3 p, float lod) {
  if (i == 0) { float a = textureLod(uVol0, p, lod).r; return uMix > 0.0 ? mix(a, textureLod(uVolB, p, lod).r, uMix) : a; }
  if (i == 1) return textureLod(uVol1, p, lod).r;
  if (i == 2) return textureLod(uVol2, p, lod).r;
  return textureLod(uVol3, p, lod).r;
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

// density() at an explicit mip level — the box test and the clip slab are unchanged, so a
// cone sample outside the map or inside the cut contributes nothing, exactly as a primary
// sample would.
float densityLod(int i, vec3 mp, float lod) {
  vec3 p = (mp - uBoxLoA[i]) / (uBoxHiA[i] - uBoxLoA[i]);
  if (any(lessThan(p, vec3(0.0))) || any(greaterThan(p, vec3(1.0)))) return 0.0;
  if (uClipOn == 1) {
    float d = dot(mp, uClipN);
    if (d < uClipRange.x || d > uClipRange.y) return 0.0;
  }
  return sampleVolLod(i, p, lod);
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
//
// The stencil is TWO texels wide, not one. The grid is uint8, so a one-texel difference is
// dominated by quantisation noise — it renders as a crumpled foil, where the mesh view gets
// smooth normals for free (averaged over the triangles at each vertex, then Taubin-smoothed).
// Two texels let the linear sampler low-pass the noise at a cost in fine detail that is far
// smaller than the noise it removes.
vec3 isoNormal(int i, vec3 mp) {
  vec3 h = max(2.0 * uVoxA[i] * (uBoxHiA[i] - uBoxLoA[i]), vec3(1e-6));
  float dx = density(i, mp + vec3(h.x, 0.0, 0.0)) - density(i, mp - vec3(h.x, 0.0, 0.0));
  float dy = density(i, mp + vec3(0.0, h.y, 0.0)) - density(i, mp - vec3(0.0, h.y, 0.0));
  float dz = density(i, mp + vec3(0.0, 0.0, h.z)) - density(i, mp - vec3(0.0, 0.0, h.z));
  vec3 g = vec3(dx, dy, dz) / (2.0 * h);
  float len = length(g);
  return len < 1e-8 ? vec3(0.0) : -g / len;    // zero-length: the caller falls back
}

// Where a ray leaves the union box — the length a secondary ray can usefully march.
float boxFar(vec3 o, vec3 d) {
  vec3 inv = 1.0 / d;
  vec3 t0 = (uBoxLo - o) * inv, t1 = (uBoxHi - o) * inv;
  vec3 tmax = max(t0, t1);
  return min(min(tmax.x, tmax.y), tmax.z);
}

// How opaque density has to be counted as: a voxel a little over the contour is solid, and a
// mip level that averaged it with its empty neighbours must still read as solid or a thin
// wall stops casting shadow. THE calibration knob for the whole shaded look — higher makes
// occlusion crisper and darker, lower makes it softer and brighter.
const float OCC_GAIN = 8.0;

// Visibility along a CONE, the estimator that replaced binary occlusion rays.
//
// Binary rays were the wrong tool and it showed: a ray steps over thin features, a grazing
// ray skims above a rough surface without ever registering, and each one answers only 0 or 1,
// so the ambient term came back at ~1 almost everywhere and the smooth preset rendered as
// blown-out white. gl.js's ambient is 64 PCF-filtered shadow maps, i.e. PARTIAL visibility
// including grazing self-occlusion, and that partiality is the entire look. A cone widens
// with distance and reads the map at the mip level matching its radius, so one sample covers
// everything inside it: an occluder that a ray would thread past still dims it.
//
// Samples march geometrically (each covers the gap the last one left), and the accumulated
// visibility multiplies, so the far end of the cone matters less than the near end — as it
// should, being both further away and coarser.
float coneVis(vec3 o, vec3 L, float vox, float tanHalf, float growth, int nsamp) {
  float voxEdge = max(vox / 1.7320508, 1e-6);   // voxel diagonal -> voxel edge
  float tf = boxFar(o, L);
  float vis = 1.0;
  float t = vox * 1.5;
  for (int k = 0; k < 12; k++) {
    if (k >= nsamp) break;
    if (t > tf) break;
    vec3 sp = o + L * t;
    float lod = log2(max(tanHalf * t / voxEdge, 1.0));
    float occ = 0.0;
    // Every active map occludes: in a composited scene one map casts ambient shadow on
    // another exactly as two halves of one map do.
    for (int j = 0; j < ${MAXV}; j++) {
      if (j >= uCount) break;
      float thrJ = max(uThrA[j], 1e-4);
      float v = densityLod(j, sp, clamp(lod, 0.0, uMaxLodA[j]));
      occ += clamp((v - thrJ) * OCC_GAIN / max(1.0 - thrJ, 1e-3), 0.0, 1.0);
    }
    vis *= 1.0 - clamp(occ, 0.0, 1.0);
    if (vis < 0.03) break;
    t *= growth;
  }
  return vis;
}

// Key-light shadow for the ray-traced preset: one narrow cone towards the light, standing in
// for gl.js's 2048² PCF shadow map. Narrow because a key shadow has a defined edge — but a
// cone, not a ray, so that edge is soft the way the filtered map's is, and the term is
// continuous in [0,1] rather than a hard 0/1 that would crawl as the camera moves.
//
// Every length here is in VOXELS of the map, never in units of the primary step dt: dt is
// (tf - tn) / steps and so changes with the view direction, and a visibility test scaled by
// it flips answers from frame to frame as you orbit — which is exactly what made the surface
// flash while rotating. The primary march stays on dt (it has to span whatever segment this
// pixel's ray covers); everything downstream of the hit is view-independent.
float isoShadow(vec3 p, vec3 n, float vox) {
  vec3 o = p + n * vox;                 // off the surface, or the cone shadows it on itself
  return coneVis(o, -uKeyDir, vox, 0.07, 1.45, 12);
}

// The ambient term, as gl.js computes it: NOT local ambient occlusion, but the
// cosine-weighted fraction of the whole sphere of directions this point can see out of —
//
//     mshadow = sum_i max(-dot(N, dir_i), 0) * vis_i / (0.25 * N)
//
// with dir_i the direction light i shines and vis_i its visibility. The 0.25 is the mean of
// max(cos, 0) over a uniformly sampled sphere, so a point that can see the whole sky comes
// out at exactly 1 and leaves the ambient intensity untouched. This is what draws the broad
// dark-in-the-pockets gradient the mesh view has: an occluder a whole domain away still
// blocks a direction, which no small-radius sampling could notice.
//
// Wide cones, because 16 of them have to cover the sphere: tan(half-angle) 0.577 is 30°,
// and ten geometric samples reach ~176 voxels — about 70% of a 256 box, so the far side of
// a molecule finally darkens the near side the way gl.js's whole-scene shadow maps do.
// (Seven samples reached only ~36 voxels, which is why the ambient term stayed too bright:
// nothing beyond the local neighbourhood could occlude anything.)
// Half the directions face away from the surface and are skipped before their cone is
// traced, so the real cost is ~8 cones of <=7 samples at hit pixels only.
float isoMultishadow(vec3 p, vec3 n, float vox) {
  vec3 o = p + n * vox;
  float sh = 0.0, wsum = 0.0;
  for (int i = 0; i < ${MS_COUNT}; i++) {
    float w = max(-dot(n, MSDIR[i]), 0.0);
    if (w <= 0.0) continue;
    sh += w * coneVis(o, -MSDIR[i], vox, 0.577, 1.7, 10);
    wsum += w;
  }
  // gl.js divides by the constant 0.25*N — the expectation of wsum — which its 64 directions
  // approach closely enough. At 16 directions that constant leaves a ±7% ripple in the
  // unoccluded ambient that reads as faint mottling on smooth faces, so normalise by the
  // actual weight sum instead: the same quantity both estimate, without the sample noise.
  return sh / max(wsum, 1e-4);
}

// ChimeraX's lighting equation (gl.js's fragment shader), with the preset intensities and
// the Material reflectivities inlined: an isosurface drawn here has to match the same map
// drawn as a mesh, or switching modes would look like a change of material. rd is the
// unit eye-to-point direction, gl.js's v1 — every vector here is model space, and the
// model transform is rigid, so the dot products are the camera-space ones.
vec3 isoShade(int i, vec3 p, vec3 n, vec3 rd, float vox) {
  float lkey  = uLightMode == 1 ? 0.0 : (uLightMode == 2 ? 0.7 : 1.0);
  float lfill = uLightMode == 1 ? 0.0 : (uLightMode == 2 ? 0.3 : 0.5);
  float lamb  = uLightMode == 1 ? 1.5 : (uLightMode == 2 ? 0.8 : 0.4);
  float kf = max(-dot(n, uKeyDir), 0.0);
  float ff = max(-dot(n, uFillDir), 0.0);
  vec3 R = normalize(reflect(uKeyDir, n));
  float sf = pow(max(-dot(R, rd), 0.0), 30.0);      // MATERIAL.exponent
  float shadow = uLightMode == 2 ? isoShadow(p, n, vox) : 1.0;
  float amb = uLightMode == 0 ? 1.0 : isoMultishadow(p, n, vox);
  vec3 base = uColA[i];
  vec3 Iamb = base * (0.8 * lamb) * amb;                                  // MATERIAL.ambient
  vec3 Idiff = base * (0.8 * lkey * kf * shadow + 0.8 * lfill * ff);      // MATERIAL.diffuse
  vec3 Ispec = vec3(0.3 * lkey * sf * shadow);                            // MATERIAL.specular
  // Not tone-mapped and not clamped, exactly as in gl.js: the soft preset's ambient is 1.5,
  // so an unoccluded face lands above 1 and clips against the framebuffer. That clipping is
  // what the preset looks like in ChimeraX.
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
    // A ray that enters the box already inside a map never crosses anything — prev starts
    // above the contour — and used to fall through to the background, which is why
    // silhouettes broke into holes where the surface meets the box face and why moving the
    // camera close punched the map open. Entering inside IS a hit, on the entry face.
    for (int i = 0; i < ${MAXV}; i++) {
      if (i >= uCount) break;
      if (prev[i] >= max(uThrA[i], 1e-4) && hitSlot < 0) { hitSlot = i; tHit = tn; }
    }
    for (int s = 1; s < 512; s++) {
      if (hitSlot >= 0) break;
      if (s > steps) break;
      float tc = tn + dt * float(s);
      vec3 mp = ro + rd * tc;
      float best = 1e30;
      int bi = -1;
      for (int i = 0; i < ${MAXV}; i++) {
        if (i >= uCount) break;
        float v = density(i, mp);
        float thr = max(uThrA[i], 1e-4);   // a contour dragged to zero has no crossing at all
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
    // The voxel diagonal of the map that was hit, in model units: the one length scale the
    // shading helpers are allowed to use (see isoShadow).
    vec3 vv = uVoxA[hitSlot] * (uBoxHiA[hitSlot] - uBoxLoA[hitSlot]);
    float vox = max(length(vv), 1e-6);
    // A cut face across a locally flat region has no gradient worth normalising; face the
    // viewer instead of shading from noise.
    if (dot(n, n) < 0.5) n = -rd;
    if (dot(n, rd) > 0.0) n = -n;              // two-sided, as gl.js is via gl_FrontFacing
    vec3 Ifrag = isoShade(hitSlot, hp, n, rd, vox);
    // Depth cue, applied for every preset because gl.js applies it for every preset: the
    // far side of the map fades into the background, and without it the surface reads flat.
    // -(row2 . hp + row2.w) is the camera-space axial distance of the hit (the camera looks
    // down -z), which is the quantity gl.js fades on.
    float dcue = -(dot(uDepthRow.xyz, hp) + uDepthRow.w);
    float dim = (uDepthRange.y - dcue) / max(uDepthRange.y - uDepthRange.x, 1e-6);
    outColor = vec4(mix(uBg, Ifrag, clamp(dim, 0.0, 1.0)), 1.0);
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

// Separable 5-tap Gaussian (sigma ~0.8 voxel) over a uint8 grid, clamped at the edges.
//
// For iso mode only, and deliberately AFTER quantisation. The grids arrive as uint8, so a
// smoothly varying density becomes a staircase of 1/255 plateaus: an isosurface cut through
// a plateau terraces, and the central-difference normal inside one flips sign on rounding
// noise — the faceted look with scattered dark patches. Averaging here manufactures the
// intermediate values 8 bits cannot hold, which is precisely what smoothing on the server
// before quantisation could not do. The mesh view gets the same effect for free: it is
// contoured from float32 and then Taubin-smoothed.
//
// Flat typed-array passes with an explicit 5-term inner sum: three passes over 256^3 is
// ~50M taps, tens of milliseconds, and it runs once per grid (callers tag the result).
// The taps of a sigma ~0.8 voxel Gaussian, normalised: the textbook numbers below sum to
// 1.0207, and an un-normalised kernel applied on three axes would brighten the whole map by
// 6% — visible as the contour appearing to move when iso mode smooths and the other styles
// do not.
const G5 = (() => {
  const w = [0.06136, 0.24477, 0.40842, 0.24477, 0.06136];
  const t = w.reduce((a, b) => a + b, 0);
  return w.map((v) => v / t);
})();

export function smoothGrid(data, nx, ny, nz) {
  const n = nx * ny * nz, plane = nx * ny;
  let src = new Float32Array(n);
  for (let i = 0; i < n; i++) src[i] = data[i];
  let dst = new Float32Array(n);
  const cl = (v, hi) => (v < 0 ? 0 : v > hi ? hi : v);

  // x: the taps are one element apart, so this axis is walked scalar. The two-wide border is
  // split out of the loop to keep the clamp out of the interior's inner sum.
  for (let base = 0; base < n; base += nx) {
    for (let i = 0; i < nx; i++) {
      if (i >= 2 && i < nx - 2) {
        const j = base + i;
        dst[j] = G5[0] * src[j - 2] + G5[1] * src[j - 1] + G5[2] * src[j] + G5[3] * src[j + 1] + G5[4] * src[j + 2];
      } else {
        dst[base + i] = G5[0] * src[base + cl(i - 2, nx - 1)] + G5[1] * src[base + cl(i - 1, nx - 1)]
          + G5[2] * src[base + i] + G5[3] * src[base + cl(i + 1, nx - 1)] + G5[4] * src[base + cl(i + 2, nx - 1)];
      }
    }
  }
  let t = src; src = dst; dst = t;

  // y and z: the taps are a row (or a plane) apart, so combine whole contiguous x-rows —
  // five streaming reads per output row instead of five strided reads per element. At 256³
  // that is the difference between a cache miss per tap and none.
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      const out = z * plane + y * nx;
      const r0 = z * plane + cl(y - 2, ny - 1) * nx, r1 = z * plane + cl(y - 1, ny - 1) * nx,
        r2 = z * plane + y * nx, r3 = z * plane + cl(y + 1, ny - 1) * nx, r4 = z * plane + cl(y + 2, ny - 1) * nx;
      for (let x = 0; x < nx; x++) {
        dst[out + x] = G5[0] * src[r0 + x] + G5[1] * src[r1 + x] + G5[2] * src[r2 + x]
          + G5[3] * src[r3 + x] + G5[4] * src[r4 + x];
      }
    }
  }
  t = src; src = dst; dst = t;

  for (let z = 0; z < nz; z++) {
    const p0 = cl(z - 2, nz - 1) * plane, p1 = cl(z - 1, nz - 1) * plane, p2 = z * plane,
      p3 = cl(z + 1, nz - 1) * plane, p4 = cl(z + 2, nz - 1) * plane;
    for (let i = 0; i < plane; i++) {
      dst[p2 + i] = G5[0] * src[p0 + i] + G5[1] * src[p1 + i] + G5[2] * src[p2 + i]
        + G5[3] * src[p3 + i] + G5[4] * src[p4 + i];
    }
  }
  src = dst;

  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const v = Math.round(src[i]);
    out[i] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
  return out;
}

export function createVolume(canvas) {
  const gl = canvas.getContext("webgl2", { antialias: false, depth: false, preserveDrawingBuffer: true, premultipliedAlpha: false });
  if (!gl) throw new Error("WebGL2 is not available in this browser.");

  // Can this context hold the grids as R16F?
  //
  // Iso shades the grid itself, so 8-bit quantisation is visible as terracing and as
  // gradient noise, and the client-side blur that hides it also eats real features. A
  // float16 grid has neither problem. Two things have to hold: half-float textures must be
  // filterable (core in WebGL2) and R16F must be colour-renderable, which is what
  // generateMipmap needs and what the extension below provides — iso's cone tracing reads
  // the mip chain, so a texture with no chain is not usable for it. The extension being
  // advertised is not quite proof, so the capability is probed for real: build a 1-voxel
  // R16F texture, mipmap it, and believe the result only if the driver raised nothing.
  // Callers read this back as `hasF16` and fetch 8-bit grids when it is false.
  const hasF16 = (() => {
    if (!gl.getExtension("EXT_color_buffer_float") && !gl.getExtension("EXT_color_buffer_half_float")) return false;
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_3D, t);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    while (gl.getError() !== gl.NO_ERROR) { /* start from a clean error state */ }
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.R16F, 1, 1, 1, 0, gl.RED, gl.HALF_FLOAT, new Uint16Array(1));
    gl.generateMipmap(gl.TEXTURE_3D);
    const ok = gl.getError() === gl.NO_ERROR;
    gl.bindTexture(gl.TEXTURE_3D, null); gl.deleteTexture(t);
    return ok;
  })();

  const prog = link(gl, VERT, FRAG);
  const u = (n) => gl.getUniformLocation(prog, n);
  const uInvVP = u("uInvVP"), uEye = u("uEye"), uBg = u("uBg"),
    uBoxLo = u("uBoxLo"), uBoxHi = u("uBoxHi"),
    uBoxLoA = u("uBoxLoA"), uBoxHiA = u("uBoxHiA"), uThrA = u("uThrA"), uColA = u("uColA"),
    uGamma = u("uGamma"), uBright = u("uBright"), uMode = u("uMode"), uSteps = u("uSteps"),
    uVolN = [u("uVol0"), u("uVol1"), u("uVol2"), u("uVol3")],
    uVolB = u("uVolB"), uMix = u("uMix"), uCount = u("uCount"),
    uClipOn = u("uClipOn"), uClipN = u("uClipN"), uClipRange = u("uClipRange"), uDiverging = u("uDiverging"),
    uVoxA = u("uVoxA"), uLightMode = u("uLightMode"), uKeyDirL = u("uKeyDir"), uFillDirL = u("uFillDir"),
    uDepthRow = u("uDepthRow"), uDepthRange = u("uDepthRange"), uMaxLodA = u("uMaxLodA");

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
    // Mipmapped minification: iso's cone tracing reads the map at the level matching each
    // cone's radius. Every other sample path pins level 0 with textureLod (see sampleVol).
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
    // A 1-voxel placeholder makes the texture complete from birth: unused slots are
    // still bound (sampling an incomplete unit is undefined behaviour on some drivers).
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.R8, 1, 1, 1, 0, gl.RED, gl.UNSIGNED_BYTE, new Uint8Array(1));
    gl.generateMipmap(gl.TEXTURE_3D);   // ...and mipmap-complete, now that the MIN filter wants levels
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

  // ChimeraX View.near_far_distances, transcribed from gl.js's depthCueRange: the scene's
  // bounding sphere projected onto the view axis, padded, then clamped so the near plane
  // stays in front of the eye — and the cue fades between fixed fractions of that. Deriving
  // it from the model rather than from the GL frustum (fixed here at 0.01..100) is what
  // makes the fade track the map as you zoom, and is what keeps this view's fade identical
  // to the mesh view's. The bounding sphere of a box is the sphere through its corners.
  function depthCueRange(mv, ub) {
    const c = [0, 0, 0], sz = [0, 0, 0];
    for (let a = 0; a < 3; a++) { c[a] = (ub.lo[a] + ub.hi[a]) / 2; sz[a] = ub.hi[a] - ub.lo[a]; }
    const radius = 0.5 * Math.hypot(sz[0], sz[1], sz[2]);
    const d = -(mv[2] * c[0] + mv[6] * c[1] + mv[10] * c[2] + mv[14]);   // camera looks down -z
    const r = (1 + NEAR_FAR_PAD) * radius;
    let near = d - r, far = d + r;
    const nearMin = far > near ? MIN_NEAR_FRACTION * (far - near) : 1;
    near = Math.max(near, nearMin);
    if (far <= near) far = 2 * near;
    return [near + (far - near) * DEPTH_CUE_START, near + (far - near) * DEPTH_CUE_END];
  }
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
    gl.uniform1fv(uMaxLodA, new Float32Array(slotDim.map((d) => Math.floor(Math.log2(Math.max(2, d[0], d[1], d[2]))))));
    gl.uniform1i(uLightMode, lightMode);
    gl.uniform3fv(uKeyDirL, toModelDir(KEY_DIR)); gl.uniform3fv(uFillDirL, toModelDir(FILL_DIR));
    // Third row of view·M (column-major: elements 2, 6, 10, 14) — the shader dots it with a
    // model-space hit to get the camera-space depth the cue fades on.
    gl.uniform4fv(uDepthRow, [mv[2], mv[6], mv[10], mv[14]]);
    gl.uniform2fv(uDepthRange, depthCueRange(mv, ub));
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

  // Every upload path — setVolume, setSlots and the morph's setVolumeB — goes through here,
  // so every texture on screen carries a full chain. Building it costs one pass over a third
  // more texels than the upload itself, once per map.
  //
  // The element type picks the storage: a Uint16Array is a float16 grid (parseGrid hands the
  // raw half bits straight through, which is what HALF_FLOAT wants), anything else is the
  // uint8 grid every other mode uses. Both normalise to 0..1 on sampling, so nothing
  // downstream — thresholds, gamma, the shader — can tell which one it got.
  function upload(target, nx, ny, nz, data) {
    const half = data instanceof Uint16Array;
    gl.bindTexture(gl.TEXTURE_3D, target);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    if (half) gl.texImage3D(gl.TEXTURE_3D, 0, gl.R16F, nx, ny, nz, 0, gl.RED, gl.HALF_FLOAT, data);
    else gl.texImage3D(gl.TEXTURE_3D, 0, gl.R8, nx, ny, nz, 0, gl.RED, gl.UNSIGNED_BYTE, data);
    gl.generateMipmap(gl.TEXTURE_3D);
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
    // Whether this context can hold float16 grids (see the probe above). The controllers ask
    // before they choose what to fetch: iso takes a native-resolution 16-bit grid where this
    // is true, and the 8-bit + client-side-blur path where it is not.
    hasF16,
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
