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
import { projection, lookAt, multiply, invert, mulVec4 } from "./mat4.js";
import { createOrbit } from "./orbit.js";

const MAXV = 4;

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
uniform int uMode;          // 0 = MIP, 1 = emission/absorption
uniform int uSteps;

// Clip slab in each slot's texture space [0,1]³ — voxels outside contribute nothing,
// which is how ChimeraX's per-axis region clipping behaves for image renderings.
uniform vec3 uClipLo;
uniform vec3 uClipHi;

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
  if (any(lessThan(p, uClipLo)) || any(greaterThan(p, uClipHi))) return 0.0;
  return sampleVol(i, p);
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
    uClipLo = u("uClipLo"), uClipHi = u("uClipHi");

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
  let clipLo = [0, 0, 0], clipHi = [1, 1, 1];    // texture-space clip slab; full box = no clipping
  let grid = null;   // { nx, ny, nz } of slot 0; null until one is, which is what the camera tests
  let gridB = null;  // the morph's far end, same shape; null when not morphing
  let mix = 0;       // blend weight in [0,1]; 0 = slot 0 alone

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
    const mvp = multiply(multiply(projection(aspect, 0.01, 100), lookAt(eye, [0, 0, 0], [0, 1, 0])), M);
    const ivp = invert(mvp); if (!ivp) return;
    const Minv = invert(M); if (!Minv) return;
    const em = mulVec4(Minv, [eye[0], eye[1], eye[2], 1]);
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
    gl.uniform1i(uCount, count);
    gl.uniform1f(uMix, gridB ? mix : 0);
    gl.uniform3fv(uClipLo, clipLo); gl.uniform3fv(uClipHi, clipHi);
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
    setMode(m) { mode = m === "emission" ? 1 : 0; render(); },
    // No pivot invalidation: an image-rendered map pivots on its box, and the contour does
    // not move the box. Same in ChimeraX, where changing the level of a map shown as an
    // image leaves its bounds — and so its centre of rotation — where they were.
    setThreshold(v) { slotThr[0] = Math.max(0, Math.min(1, v)); render(); },
    setSlotThreshold(i, v) { if (i >= 0 && i < MAXV) { slotThr[i] = Math.max(0, Math.min(1, v)); render(); } },
    setGamma(g) { gamma = Math.max(0.2, g); render(); },
    // Per-axis clip slab, texture space [0,1] in (x, y, z) order. Pass nulls to clear.
    setClip(lo, hi) { clipLo = lo || [0, 0, 0]; clipHi = hi || [1, 1, 1]; render(); },
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
