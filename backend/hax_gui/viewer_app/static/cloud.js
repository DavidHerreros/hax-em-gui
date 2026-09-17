// Minimal WebGL2 3D point-cloud renderer for the latent landscape (3 PCs).
// Orbit camera (shared style with gl.js), per-point color, selection/target
// rings, CPU-side picking, and high-DPI / transparent snapshots. No deps.
import { projection, lookAt, multiply, mulVec4, invert } from "./mat4.js";
import { createOrbit } from "./orbit.js";

const VERT = `#version 300 es
precision highp float;
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aColor;
layout(location=2) in float aFlag;   // 0 normal, 1 current, 2 path, 3 selection, 4 hover
uniform mat4 uProj;
uniform mat4 uView;
uniform float uSize;
out vec3 vColor;
out float vFlag;
void main() {
  vColor = aColor; vFlag = aFlag;
  gl_Position = uProj * uView * vec4(aPos, 1.0);
  gl_PointSize = uSize * (aFlag > 0.5 ? 2.1 : 1.0);
}`;

const FRAG = `#version 300 es
precision highp float;
in vec3 vColor;
in float vFlag;
uniform float uAlpha;                          // base point opacity (rings stay opaque)
out vec4 outColor;
void main() {
  vec2 d = gl_PointCoord - vec2(0.5);
  float r = length(d);
  if (r > 0.5) discard;                       // round dots
  if (vFlag > 0.5 && r > 0.34 && r < 0.5) {   // highlight ring (always visible)
    vec3 ring = vFlag > 3.5 ? vec3(0.886,0.910,0.941)   // hover (light slate, as in 2D)
              : vFlag > 2.5 ? vec3(0.95,0.96,0.98)      // selection (white)
              : vFlag > 1.5 ? vec3(0.957,0.447,0.714)   // path (pink)
                            : vec3(0.176,0.831,0.749);  // current (teal)
    outColor = vec4(ring, 1.0); return;
  }
  if (uAlpha < 0.01) discard;                  // points hidden
  float shade = 1.0 - r * 0.5;                 // subtle sphere shading
  outColor = vec4(vColor * shade, uAlpha);
}`;

// Density splats. One program covers every style except the ray-march; `uMode` picks the
// transfer function and the compositing the caller has already set up.
//
//   0 splats  a gaussian kernel, world-sized and depth-sorted, composited over
//   1 shells  nested quantile bands: each level encloses a stated share of the particles
//   2 energy  -ln(p), so the scale is dG in kT and six decades of occupancy stop clipping
//   3 points  one hue, opacity by occupancy — the density as a substrate, not the subject
//
// All of them draw back-to-front with ordinary over-blending. Additive blending is gone:
// summing a colormap over ~40 overlapping sprites made hue a function of ray depth and
// clipped dense cores to white, which is what this pass exists to avoid.
const VERT_SPLAT = `#version 300 es
precision highp float;
layout(location=0) in vec3 aPos;
layout(location=1) in float aDens;    // local density 0..1
layout(location=2) in float aQuant;   // share of particles in the densest set containing this point
layout(location=3) in vec3 aNorm;     // -grad(rho), for the headlight
uniform mat4 uProj;
uniform mat4 uView;
uniform float uPixPerUnit;   // world radius -> device px at unit depth
uniform float uRadius;       // splat radius, in normalized model units
uniform float uRadGain;      // share of the radius that scales with density
uniform vec3 uEyeModel;      // camera position in model space
out float vDens; out float vQuant; out float vZ; out float vShade;
void main() {
  vDens = aDens; vQuant = aQuant;
  vec3 ld = normalize(uEyeModel - aPos);
  vShade = dot(aNorm, aNorm) < 1e-8 ? 1.0 : 0.42 + 0.58 * max(dot(normalize(aNorm), ld), 0.0);
  vec4 vp = uView * vec4(aPos, 1.0);
  float z = max(0.05, -vp.z);
  vZ = z;
  gl_Position = uProj * vp;
  gl_PointSize = clamp(2.0 * uRadius * (1.0 - uRadGain + uRadGain * aDens) * uPixPerUnit / z, 1.0, 260.0);
}`;

const FRAG_SPLAT = `#version 300 es
precision highp float;
in float vDens; in float vQuant; in float vZ; in float vShade;
uniform int uMode;
uniform float uOpacity, uGamma;
uniform vec3 uRamp[6];     // density colormap stops (low->high)
uniform int uRampN;        // number of active stops
uniform vec3 uShellC[6];
uniform float uShellA[6];
uniform float uShellQ[6];  // ascending quantiles; the last is >1 (the tail)
uniform int uShellN;
uniform float uEps;        // density floor for the log transfer
uniform float uShadeAmt;
uniform float uFog;
uniform vec2 uZRange;
uniform vec3 uBg;
out vec4 outColor;
vec3 ramp(float t) {
  t = clamp(t, 0.0, 1.0);
  float f = t * float(uRampN - 1);
  int i = int(floor(f));
  i = min(i, uRampN - 2);
  return mix(uRamp[i], uRamp[i + 1], f - float(i));
}
void main() {
  vec2 d = gl_PointCoord - vec2(0.5);
  float a;
  if (uMode == 0) { a = exp(-dot(d, d) * 9.0); if (a < 0.01) discard; }
  else { float r = length(d) * 2.0; if (r > 1.0) discard; a = smoothstep(1.0, 0.72, r); }

  vec3 col; float alpha;
  if (uMode == 0) {
    col = ramp(vDens);
    alpha = a * uOpacity * pow(vDens, uGamma);
  } else if (uMode == 1) {
    int b = uShellN - 1;
    for (int i = 0; i < 6; i++) {
      if (i >= uShellN) break;
      if (vQuant <= uShellQ[i]) { b = i; break; }
    }
    col = uShellC[b];
    alpha = a * uShellA[b] * uOpacity * 1.9;
  } else if (uMode == 2) {
    float t = clamp((log(max(vDens, uEps)) - log(uEps)) / (-log(uEps)), 0.0, 1.0);
    col = ramp(t);
    alpha = a * uOpacity * (0.18 + 0.82 * pow(t, uGamma));
  } else {
    float t = pow(vDens, 0.7);
    col = mix(vec3(0.29, 0.34, 0.42), vec3(0.95, 0.97, 1.0), t);
    alpha = a * uOpacity * (0.10 + 0.90 * t);
  }

  if (uShadeAmt > 0.0) col *= mix(1.0, vShade, uShadeAmt);
  if (uFog > 0.0) {
    float f = clamp((vZ - uZRange.x) / max(0.001, uZRange.y - uZRange.x), 0.0, 1.0);
    col = mix(col, uBg, f * uFog * 0.8);
    alpha *= mix(1.0, 0.55, f * uFog);
  }
  outColor = vec4(col, clamp(alpha, 0.0, 1.0));
}`;

// Ray-marched density volume. The KDE grid the Slices view already builds is uploaded as
// an R8 3D texture and integrated front-to-back with emission + absorption, so the
// landscape occludes itself the way the decoded map does in the pane next to it.
const VERT_VOL = `#version 300 es
precision highp float;
out vec2 vNdc;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  vNdc = p * 2.0 - 1.0;
  gl_Position = vec4(vNdc, 0.0, 1.0);
}`;

// `precision highp sampler3D` is not optional: ES 3.00 gives sampler types no default
// precision in the fragment stage, and the shader fails to compile without it.
const FRAG_VOL = `#version 300 es
precision highp float;
precision highp sampler3D;
in vec2 vNdc;
uniform mat4 uInvVP;
uniform vec3 uEye, uLo, uHi, uDim, uBg;
uniform sampler3D uVol;
uniform float uOpacity, uGamma, uShadeAmt, uFog;
uniform vec3 uRamp[6];
uniform int uRampN;
out vec4 outColor;
vec3 ramp(float t) {
  t = clamp(t, 0.0, 1.0);
  float f = t * float(uRampN - 1);
  int i = int(floor(f));
  i = min(i, uRampN - 2);
  return mix(uRamp[i], uRamp[i + 1], f - float(i));
}
void main() {
  vec4 pn = uInvVP * vec4(vNdc, -1.0, 1.0);
  vec4 pf = uInvVP * vec4(vNdc,  1.0, 1.0);
  vec3 ro = pn.xyz / pn.w;
  vec3 rd = normalize(pf.xyz / pf.w - ro);

  vec3 inv = 1.0 / rd;
  vec3 t0 = (uLo - ro) * inv, t1 = (uHi - ro) * inv;
  vec3 lo3 = min(t0, t1), hi3 = max(t0, t1);
  float tmin = max(max(lo3.x, lo3.y), max(lo3.z, 0.0));
  float tmax = min(min(hi3.x, hi3.y), hi3.z);
  if (tmax <= tmin) { outColor = vec4(0.0); return; }

  const int STEPS = 128;
  float dt = (tmax - tmin) / float(STEPS);
  vec3 span = uHi - uLo;
  vec3 e = 1.0 / uDim;
  vec4 acc = vec4(0.0);
  float far = length(uEye - (uLo + uHi) * 0.5) + length(span) * 0.5;
  float near = max(0.05, far - length(span));

  for (int i = 0; i < STEPS; i++) {
    if (acc.a > 0.985) break;
    vec3 p = ro + rd * (tmin + (float(i) + 0.5) * dt);
    float s = texture(uVol, (p - uLo) / span).r;
    if (s > 0.004) {
      float v = pow(s, uGamma);
      vec3 c = ramp(v);
      if (uShadeAmt > 0.0) {
        vec3 tc = (p - uLo) / span;
        vec3 g = vec3(
          texture(uVol, tc + vec3(e.x, 0.0, 0.0)).r - texture(uVol, tc - vec3(e.x, 0.0, 0.0)).r,
          texture(uVol, tc + vec3(0.0, e.y, 0.0)).r - texture(uVol, tc - vec3(0.0, e.y, 0.0)).r,
          texture(uVol, tc + vec3(0.0, 0.0, e.z)).r - texture(uVol, tc - vec3(0.0, 0.0, e.z)).r);
        if (dot(g, g) > 1e-10) {
          vec3 nrm = normalize(-g), ld = normalize(uEye - p);
          float lam = 0.34 + 0.66 * max(dot(nrm, ld), 0.0);
          float rim = pow(1.0 - max(dot(nrm, ld), 0.0), 3.0) * 0.25;
          c = c * mix(1.0, lam, uShadeAmt) + rim * uShadeAmt * vec3(0.55, 0.62, 0.78);
        }
      }
      if (uFog > 0.0) {
        float f = clamp((length(p - uEye) - near) / max(0.001, far - near), 0.0, 1.0);
        c = mix(c, uBg, f * uFog * 0.6);
      }
      float al = 1.0 - exp(-v * uOpacity * 46.0 * dt);
      acc.rgb += (1.0 - acc.a) * al * c;
      acc.a += (1.0 - acc.a) * al;
    }
  }
  outColor = acc;
}`;

// Morph-path polyline (a line strip through the path waypoints, drawn on top).
const VERT_LINE = `#version 300 es
precision highp float;
layout(location=0) in vec3 aPos;
uniform mat4 uProj; uniform mat4 uView;
void main() { gl_Position = uProj * uView * vec4(aPos, 1.0); }`;
const FRAG_LINE = `#version 300 es
precision highp float;
uniform vec3 uColor;
out vec4 outColor;
void main() { outColor = vec4(uColor, 1.0); }`;

// ID pass: the same point sprites, but each fragment writes its point's *index* instead
// of a colour. Reading one pixel back then turns picking into an O(1) lookup. The CPU
// projection scans this replaces were O(N) and ran on every hover frame and every
// mouse-down — at a million points that is what made the 3D view crawl.
//
// `precision highp int` is not optional: an ES 3.00 fragment shader defaults to mediump
// int, which is only guaranteed 16 bits, and would silently mangle any index past 32767.
const VERT_PICK = `#version 300 es
precision highp float;
precision highp int;
layout(location=0) in vec3 aPos;
uniform mat4 uProj; uniform mat4 uView; uniform float uSize;
flat out uint vId;
void main() {
  vId = uint(gl_VertexID) + 1u;              // 0 is reserved for "background"
  gl_Position = uProj * uView * vec4(aPos, 1.0);
  gl_PointSize = uSize;
}`;

const FRAG_PICK = `#version 300 es
precision highp float;
precision highp int;
flat in uint vId;
out vec4 outColor;
void main() {
  if (length(gl_PointCoord - vec2(0.5)) > 0.5) discard;   // same round footprint as the drawn dot
  outColor = vec4(float(vId & 0xffu), float((vId >> 8) & 0xffu), float((vId >> 16) & 0xffu), 255.0) / 255.0;
}`;

// Editable control-point handles: white discs with a dark rim, drawn as point sprites.
const VERT_HANDLE = `#version 300 es
precision highp float;
layout(location=0) in vec3 aPos;
uniform mat4 uProj; uniform mat4 uView; uniform float uSize;
void main() { gl_Position = uProj * uView * vec4(aPos, 1.0); gl_PointSize = uSize; }`;
const FRAG_HANDLE = `#version 300 es
precision highp float;
out vec4 outColor;
void main() {
  float r = length(gl_PointCoord - vec2(0.5));
  if (r > 0.5) discard;
  outColor = r > 0.36 ? vec4(0.06, 0.09, 0.16, 1.0) : vec4(1.0, 1.0, 1.0, 0.95);
}`;

export function createCloud(canvas, onSelect) {
  const gl = canvas.getContext("webgl2", { antialias: true, depth: true, preserveDrawingBuffer: true, premultipliedAlpha: false });
  if (!gl) throw new Error("WebGL2 is not available in this browser.");
  let pickFilter = null;   // when set, selection picks (not camera/path picks) only accept ids where pickFilter(i) is true

  const prog = link(gl, VERT, FRAG);
  const u = (n) => gl.getUniformLocation(prog, n);
  const uProj = u("uProj"), uView = u("uView"), uSize = u("uSize"), uAlpha = u("uAlpha");

  const splatProg = link(gl, VERT_SPLAT, FRAG_SPLAT);
  const su = (n) => gl.getUniformLocation(splatProg, n);
  const sProj = su("uProj"), sView = su("uView"), sRamp = su("uRamp"), sRampN = su("uRampN");
  let rampFlat = new Float32Array([0.27, 0.0, 0.33, 0.23, 0.32, 0.55, 0.13, 0.57, 0.55, 0.37, 0.79, 0.38, 0.99, 0.91, 0.15, 0, 0, 0]);   // viridis (normalized)
  let rampN = 5;

  // The ray-marcher is the only program that needs a 3D sampler. If a driver refuses it we
  // lose one style, not the landscape — so it links on its own and may come back null.
  let volProg = null, vu = () => null;
  try { volProg = link(gl, VERT_VOL, FRAG_VOL); vu = (n) => gl.getUniformLocation(volProg, n); }
  catch (err) { console.warn("landscape: volumetric density unavailable —", err.message); }

  const vao = gl.createVertexArray();
  const posBuf = gl.createBuffer(), colBuf = gl.createBuffer(), flagBuf = gl.createBuffer(),
        densBuf = gl.createBuffer(), quantBuf = gl.createBuffer(), normBuf = gl.createBuffer(),
        orderBuf = gl.createBuffer();
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, colBuf); gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, flagBuf); gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 1, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);

  const splatVao = gl.createVertexArray();   // aPos (shared) + aDens + aQuant + aNorm
  gl.bindVertexArray(splatVao);
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, densBuf); gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 1, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, quantBuf); gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 1, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, normBuf); gl.enableVertexAttribArray(3); gl.vertexAttribPointer(3, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, orderBuf);
  gl.bindVertexArray(null);

  const emptyVao = gl.createVertexArray();   // the ray-march's full-screen triangle needs no attributes
  const volTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_3D, volTex);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
  gl.bindTexture(gl.TEXTURE_3D, null);

  const lineProg = link(gl, VERT_LINE, FRAG_LINE);
  const lProj = gl.getUniformLocation(lineProg, "uProj"), lView = gl.getUniformLocation(lineProg, "uView"), lColor = gl.getUniformLocation(lineProg, "uColor");
  const lineBuf = gl.createBuffer(), lineVao = gl.createVertexArray();
  gl.bindVertexArray(lineVao);
  gl.bindBuffer(gl.ARRAY_BUFFER, lineBuf); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);
  let lineCount = 0, lineColor = [0.957, 0.447, 0.714], lineData = new Float32Array(0);

  const handleProg = link(gl, VERT_HANDLE, FRAG_HANDLE);
  const hProj = gl.getUniformLocation(handleProg, "uProj"), hView = gl.getUniformLocation(handleProg, "uView"), hSize = gl.getUniformLocation(handleProg, "uSize");
  const handleBuf = gl.createBuffer(), handleVao = gl.createVertexArray();
  gl.bindVertexArray(handleVao);
  gl.bindBuffer(gl.ARRAY_BUFFER, handleBuf); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);
  let handleData = new Float32Array(0), handleCount = 0;
  gl.enable(gl.DEPTH_TEST);

  let bg = [0.016, 0.024, 0.06];
  let norm = null;        // Float32Array(N*3) normalized to [-1,1] cube
  let nCenter = [0, 0, 0], nScale = 1;   // data->norm: (p - nCenter) * nScale
  let freeCurves = [];    // [{ flat:Float32Array(norm), n, color:[r,g,b] }] free-spline overlays
  let count = 0;
  let colorData = null;   // persisted per-point RGB so points are never left black
  let showDens = false, hasDens = false, densIntensity = 0.22, pointAlpha = 1.0;
  // ── density render style ──────────────────────────────────────────────────
  // MODE_OF maps the public style name onto the shader's uMode. "volume" has no splat
  // mode of its own: it ray-marches, and falls back to sorted splats where the 3D-texture
  // program failed to link.
  const MODE_OF = { splats: 0, shells: 1, energy: 2, points: 3, volume: 0 };
  let densStyle = "splats", densGamma = 1.4, densKernel = 0.12, densShade = 1, densFog = 1;
  let hasQuant = false, hasNorm = false, hasVol = false;
  let volDim = [1, 1, 1], volLo = [0, 0, 0], volHi = [0, 0, 0];
  let shellQ = new Float32Array([0.25, 0.5, 0.75, 0.9, 1.01, 1.01]);
  let shellC = new Float32Array(18), shellA = new Float32Array(6), shellN = 5;
  // Back-to-front order for the splats. A counting sort into depth buckets is O(N) and
  // allocation-free, which matters because it re-runs on every camera change.
  const SORT_BUCKETS = 2048;
  let order = null, depths = null, bucketOf = null, sortCounts = new Int32Array(SORT_BUCKETS);
  let sortDirty = true, lastZRange = null;
  const invalidateSort = () => { sortDirty = true; };
  let selected = -1, hover = -1, pathArr = [], selArr = [];
  // Mid-export the canvas is deliberately off its on-screen size. gl_PointSize is in
  // device px, so the dots have to grow with it or a 4x PNG comes out full of specks.
  let snapping = false;
  const ratio = () => (snapping
    ? canvas.width / Math.max(1, canvas.clientWidth)
    : Math.min(window.devicePixelRatio || 1, 2));
  let pathSet = new Set(), selSet = new Set();   // membership tests for the per-point flag ladder

  // ── id buffer ─────────────────────────────────────────────────────────────
  // Off-screen render of the point indices, re-made only when the camera or the point set
  // moves under it (a hover, which changes neither, reuses it). Everything that needs to
  // know "which point is at this pixel" reads it back instead of projecting all N points.
  const pickProg = link(gl, VERT_PICK, FRAG_PICK);
  const pProj = gl.getUniformLocation(pickProg, "uProj");
  const pView = gl.getUniformLocation(pickProg, "uView");
  const pSize = gl.getUniformLocation(pickProg, "uSize");
  let pickFbo = null, pickTex = null, pickRbo = null, pickW = 0, pickH = 0, pickDirty = true;
  const invalidatePick = () => { pickDirty = true; };

  function ensurePickTargets() {
    const w = canvas.width, h = canvas.height;
    if (pickFbo && pickW === w && pickH === h) return;
    if (pickTex) gl.deleteTexture(pickTex);
    if (pickRbo) gl.deleteRenderbuffer(pickRbo);
    if (pickFbo) gl.deleteFramebuffer(pickFbo);
    pickW = w; pickH = h;
    pickTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, pickTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    pickRbo = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, pickRbo);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, w, h);
    pickFbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, pickFbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, pickTex, 0);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, pickRbo);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    pickDirty = true;
  }

  function ensurePickBuffer() {
    if (!count || !norm) return false;
    ensurePickTargets();
    if (!pickDirty) return true;
    const { proj, view } = matrices();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    gl.bindFramebuffer(gl.FRAMEBUFFER, pickFbo);
    gl.viewport(0, 0, pickW, pickH);
    gl.clearColor(0, 0, 0, 0);                       // id 0 = nothing here
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.disable(gl.BLEND); gl.enable(gl.DEPTH_TEST);
    gl.useProgram(pickProg);
    gl.uniformMatrix4fv(pProj, false, proj);
    gl.uniformMatrix4fv(pView, false, view);
    gl.uniform1f(pSize, 6.0 * dpr);                  // the sprite size the points are drawn at
    gl.bindVertexArray(vao);                         // aPos is location 0 here too
    gl.drawArrays(gl.POINTS, 0, count);
    gl.bindVertexArray(null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    pickDirty = false;
    return true;
  }

  // Nearest drawn point to (px, py), both in device px with a GL bottom-up y, within
  // `tolPx`. Depth-tested, so an occluded point can no longer win over the one in front
  // of it — which the old screen-space scan happily let happen.
  function pickAt(px, py, tolPx, useFilter) {
    if (!ensurePickBuffer()) return -1;
    const r = Math.max(1, Math.round(tolPx));
    const x0 = Math.max(0, Math.round(px) - r), x1 = Math.min(pickW - 1, Math.round(px) + r);
    const y0 = Math.max(0, Math.round(py) - r), y1 = Math.min(pickH - 1, Math.round(py) + r);
    const w = x1 - x0 + 1, h = y1 - y0 + 1;
    if (w <= 0 || h <= 0) return -1;
    const buf = new Uint8Array(w * h * 4);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, pickFbo);
    gl.readPixels(x0, y0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, buf);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
    let best = -1, bestD = Infinity;
    for (let j = 0; j < h; j++) {
      for (let i = 0; i < w; i++) {
        const o = (j * w + i) * 4;
        const id = buf[o] | (buf[o + 1] << 8) | (buf[o + 2] << 16);
        if (!id) continue;
        if (useFilter && pickFilter && !pickFilter(id - 1)) continue;   // out of the pick scope
        const dx = (x0 + i) - px, dy = (y0 + j) - py;
        const dd = dx * dx + dy * dy;
        if (dd < bestD) { bestD = dd; best = id - 1; }
      }
    }
    return best;
  }
  const glY = (cssY) => canvas.height - 1 - cssY * Math.min(window.devicePixelRatio || 1, 2);

  // Distance from the eye to the nearest point under the centre of the window, which the
  // camera turns into a ChimeraX-style centre of rotation: zoom into a lobe of the
  // landscape and rotation happens about that lobe rather than the cloud's centre.
  // Null when the centre is empty, which leaves the camera on the pivot it already had.
  const COFR_TOL_PX = 24;   // a point cloud is sparse, so accept points a little off-centre
  function centreDepth() {
    if (!norm || !count) return null;
    const r = Math.min(window.devicePixelRatio || 1, 2);
    const i = pickAt(canvas.width / 2, canvas.height / 2, COFR_TOL_PX * r);
    if (i < 0) return null;
    const { view } = matrices();   // view = lookAt · model, so this lands straight in view space
    const v = mulVec4(view, [norm[i * 3], norm[i * 3 + 1], norm[i * 3 + 2], 1]);
    const t = -v[2];               // view-space z is negative in front of the eye
    return t > 0 ? t : null;
  }

  // invalidatePick on every camera change: the id buffer is rendered from this camera, so
  // it goes stale the moment the camera moves — but *not* when only the hover changes.
  const orbit = createOrbit(canvas, () => { invalidatePick(); invalidateSort(); render(); }, (e) => { const i = pick(e.clientX, e.clientY); onSelect(i, { shift: e.shiftKey, cx: e.clientX, cy: e.clientY }); },
    // normalizePoints() scales the largest extent to 1.8; fitRadius 1.41 is the height the
    // old 45°-vertical camera showed at its default distance, so the cloud keeps its framing.
    { probeDepth: centreDepth, modelWidth: 1.8, fitRadius: 1.41 });
  function matrices() {
    const aspect = canvas.width / Math.max(1, canvas.height);
    const proj = projection(aspect, 0.01, 100);
    // camera fixed on +Z; the model transform (trackball rotation + pan) is baked into view.
    const view = multiply(lookAt(orbit.eye, [0, 0, 0], [0, 1, 0]), orbit.model);
    return { proj, view };
  }
  function resize() {
    if (snapping) return;   // the export canvas is sized deliberately; leave it alone
    const dpr = ratio();
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr)), h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  }
  function drawScene(alpha = 1) {
    const { proj, view } = matrices();
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(bg[0], bg[1], bg[2], alpha);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (!count) return;
    const dpr = ratio();
    // density overlay, behind the points and never writing depth
    if (showDens && hasDens) drawDensity(proj, view);
    gl.useProgram(prog);
    gl.uniformMatrix4fv(uProj, false, proj);
    gl.uniformMatrix4fv(uView, false, view);
    gl.uniform1f(uSize, 6.0 * dpr);
    gl.uniform1f(uAlpha, pointAlpha);
    gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.bindVertexArray(vao);
    gl.drawArrays(gl.POINTS, 0, count);
    gl.bindVertexArray(null);
    gl.disable(gl.BLEND);
    // morph-path polyline + free-spline overlays, drawn on top (depth test off)
    if (lineCount >= 2 || freeCurves.length) {
      gl.useProgram(lineProg);
      gl.uniformMatrix4fv(lProj, false, proj);
      gl.uniformMatrix4fv(lView, false, view);
      gl.disable(gl.DEPTH_TEST);
      gl.bindVertexArray(lineVao);
      if (lineCount >= 2) {
        gl.uniform3fv(lColor, lineColor);
        gl.bindBuffer(gl.ARRAY_BUFFER, lineBuf); gl.bufferData(gl.ARRAY_BUFFER, lineData, gl.DYNAMIC_DRAW);
        gl.drawArrays(gl.LINE_STRIP, 0, lineCount);
      }
      for (const fc of freeCurves) {
        if (fc.n < 2) continue;
        gl.uniform3fv(lColor, fc.color);
        gl.bindBuffer(gl.ARRAY_BUFFER, lineBuf); gl.bufferData(gl.ARRAY_BUFFER, fc.flat, gl.DYNAMIC_DRAW);
        gl.drawArrays(gl.LINE_STRIP, 0, fc.n);
      }
      gl.bindVertexArray(null);
      gl.enable(gl.DEPTH_TEST);
    }
    // editable control-point handles (point sprites), drawn on top of everything
    if (handleCount) {
      gl.useProgram(handleProg);
      gl.uniformMatrix4fv(hProj, false, proj);
      gl.uniformMatrix4fv(hView, false, view);
      gl.uniform1f(hSize, 11.0 * dpr);
      gl.disable(gl.DEPTH_TEST);
      gl.bindVertexArray(handleVao); gl.drawArrays(gl.POINTS, 0, handleCount); gl.bindVertexArray(null);
      gl.enable(gl.DEPTH_TEST);
    }
  }
  // Setting up a density style touches ~10 uploads, and every setter redraws. Batch them
  // so a style change is one frame rather than ten — at a million points that is the
  // difference between instant and a visible hitch.
  let batching = 0;
  function render() {
    if (batching > 0) return;
    resize(); drawScene(1); if (onRenderCb) onRenderCb();
  }
  let onRenderCb = null;

  // Order the splats far-to-near for this camera. Counting sort over SORT_BUCKETS depth
  // bins: one pass to bin, one prefix sum, one pass to place — no comparisons, no garbage.
  function sortSplats(view) {
    if (!order || order.length !== count) {
      order = new Uint32Array(count); depths = new Float32Array(count); bucketOf = new Int32Array(count);
    }
    const m2 = view[2], m6 = view[6], m10 = view[10], m14 = view[14];
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < count; i++) {
      const z = -(m2 * norm[i * 3] + m6 * norm[i * 3 + 1] + m10 * norm[i * 3 + 2] + m14);
      depths[i] = z; if (z < lo) lo = z; if (z > hi) hi = z;
    }
    const span = (hi - lo) || 1;
    sortCounts.fill(0);
    for (let i = 0; i < count; i++) {
      // reversed, so the farthest points land in bucket 0 and are drawn first
      let k = (SORT_BUCKETS - 1) - (((depths[i] - lo) / span) * (SORT_BUCKETS - 1) | 0);
      if (k < 0) k = 0; else if (k >= SORT_BUCKETS) k = SORT_BUCKETS - 1;
      bucketOf[i] = k; sortCounts[k]++;
    }
    let acc = 0;
    for (let k = 0; k < SORT_BUCKETS; k++) { const c = sortCounts[k]; sortCounts[k] = acc; acc += c; }
    for (let i = 0; i < count; i++) order[sortCounts[bucketOf[i]]++] = i;
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, orderBuf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, order, gl.DYNAMIC_DRAW);
    sortDirty = false;
    lastZRange = [lo, hi];
    return lastZRange;
  }

  function drawDensity(proj, view) {
    const useVolume = densStyle === "volume" && volProg && hasVol;
    let mode = MODE_OF[densStyle] ?? 0;
    if (mode === 1 && !hasQuant) mode = 0;   // shells without quantiles would be one flat band
    // eye in model space: the model transform is baked into `view`, and aPos is not
    const iv = invert(view);
    const eye = iv ? [iv[12], iv[13], iv[14]] : [0, 0, 3];

    if (useVolume) {
      const mvp = multiply(proj, view), inv = invert(mvp);
      if (inv) {
        gl.useProgram(volProg);
        gl.uniformMatrix4fv(vu("uInvVP"), false, inv);
        gl.uniform3fv(vu("uEye"), eye);
        gl.uniform3fv(vu("uLo"), volLo);
        gl.uniform3fv(vu("uHi"), volHi);
        gl.uniform3f(vu("uDim"), volDim[0], volDim[1], volDim[2]);
        gl.uniform3fv(vu("uBg"), bg);
        gl.uniform1f(vu("uOpacity"), densIntensity);
        gl.uniform1f(vu("uGamma"), densGamma);
        gl.uniform1f(vu("uShadeAmt"), densShade ? 0.85 : 0);
        gl.uniform1f(vu("uFog"), densFog ? 1 : 0);
        gl.uniform3fv(vu("uRamp"), rampFlat);
        gl.uniform1i(vu("uRampN"), rampN);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_3D, volTex);
        gl.uniform1i(vu("uVol"), 0);
        gl.depthMask(false); gl.disable(gl.DEPTH_TEST);
        gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);   // already premultiplied
        gl.bindVertexArray(emptyVao); gl.drawArrays(gl.TRIANGLES, 0, 3); gl.bindVertexArray(null);
        gl.disable(gl.BLEND); gl.enable(gl.DEPTH_TEST); gl.depthMask(true);
      }
      return;
    }

    if (sortDirty || !order || order.length !== count) sortSplats(view);
    // Reuse the sort's own near/far so the depth cue does not jump on frames that skip it.
    let zr = lastZRange;
    if (!zr) {
      const dist = Math.hypot(orbit.eye[0], orbit.eye[1], orbit.eye[2]);
      zr = [Math.max(0.05, dist - 1.1), dist + 1.1];
    }

    gl.useProgram(splatProg);
    gl.uniformMatrix4fv(sProj, false, proj);
    gl.uniformMatrix4fv(sView, false, view);
    gl.uniform3fv(su("uEyeModel"), eye);
    gl.uniform3fv(sRamp, rampFlat);
    gl.uniform1i(sRampN, rampN);
    gl.uniform1i(su("uMode"), mode);
    gl.uniform1f(su("uOpacity"), densIntensity);
    gl.uniform1f(su("uGamma"), densGamma);
    gl.uniform1f(su("uEps"), 0.004);
    gl.uniform1f(su("uShadeAmt"), densShade && hasNorm ? 0.85 : 0);
    gl.uniform1f(su("uFog"), densFog ? 1 : 0);
    gl.uniform2f(su("uZRange"), zr[0], zr[1]);
    gl.uniform3fv(su("uBg"), bg);
    gl.uniform3fv(su("uShellC"), shellC);
    gl.uniform1fv(su("uShellA"), shellA);
    gl.uniform1fv(su("uShellQ"), shellQ);
    gl.uniform1i(su("uShellN"), shellN);
    // Kernel width in model units, not in pixels: the picture of the density then stops
    // depending on how far the camera happens to be zoomed in. proj[0] carries the
    // horizontal 1/tan(fov/2), so this is device px per model unit at unit depth.
    gl.uniform1f(su("uPixPerUnit"), proj[0] * canvas.width * 0.5);
    gl.uniform1f(su("uRadius"), mode === 0 ? densKernel : mode === 3 ? 0.010 : 0.016);
    gl.uniform1f(su("uRadGain"), mode === 0 ? 0.55 : mode === 3 ? 0.6 : 0.35);

    gl.depthMask(false); gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.bindVertexArray(splatVao);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, orderBuf);
    gl.drawElements(gl.POINTS, count, gl.UNSIGNED_INT, 0);
    gl.bindVertexArray(null);
    gl.disable(gl.BLEND); gl.enable(gl.DEPTH_TEST); gl.depthMask(true);
  }

  function normalizePoints(pts) {
    const n = pts.length;
    const mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
    for (const p of pts) for (let d = 0; d < 3; d++) { const v = p[d] || 0; if (v < mn[d]) mn[d] = v; if (v > mx[d]) mx[d] = v; }
    let ext = 0; for (let d = 0; d < 3; d++) ext = Math.max(ext, mx[d] - mn[d]);
    if (!(ext > 0)) ext = 1;
    nScale = 1.8 / ext;
    for (let d = 0; d < 3; d++) nCenter[d] = (mn[d] + mx[d]) / 2;
    const out = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) for (let d = 0; d < 3; d++) out[i * 3 + d] = ((pts[i][d] || 0) - nCenter[d]) * nScale;
    return out;
  }
  const dataToNorm = (p) => [((p[0] || 0) - nCenter[0]) * nScale, ((p[1] || 0) - nCenter[1]) * nScale, ((p[2] || 0) - nCenter[2]) * nScale];
  const normToData = (q) => [q[0] / nScale + nCenter[0], q[1] / nScale + nCenter[1], q[2] / nScale + nCenter[2]];

  // Written weakest-first, so a point that is several things at once shows the one that
  // matters most: hover < selection < path < current.
  let flags = null;   // CPU mirror of the flag attribute, so single points can be patched
  function uploadFlags() {
    if (!flags || flags.length !== count) flags = new Float32Array(count);
    flags.fill(0);
    if (hover >= 0 && hover < count) flags[hover] = 4;                    // hover = light
    for (const si of selArr) if (si >= 0 && si < count) flags[si] = 3;    // selection = white
    for (const pi of pathArr) if (pi >= 0 && pi < count) flags[pi] = 2;   // path = pink
    if (selected >= 0 && selected < count) flags[selected] = 1;           // current = teal (wins)
    gl.bindBuffer(gl.ARRAY_BUFFER, flagBuf); gl.bufferData(gl.ARRAY_BUFFER, flags, gl.DYNAMIC_DRAW);
  }
  // The same ladder, for one point — the value it *should* have right now.
  function flagValue(i) {
    if (i === selected) return 1;
    if (pathSet.has(i)) return 2;
    if (selSet.has(i)) return 3;
    if (i === hover) return 4;
    return 0;
  }
  // A hover move touches exactly two points, so patch those two floats in place. The full
  // rebuild above reallocates and re-uploads the whole attribute — 4 MB at a million
  // points, which it used to do on every mouse move.
  function patchFlag(i) {
    if (i < 0 || i >= count || !flags) return;
    const v = flagValue(i);
    if (flags[i] === v) return;
    flags[i] = v;
    gl.bindBuffer(gl.ARRAY_BUFFER, flagBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, i * 4, flags, i, 1);
  }

  function pick(cx, cy) {
    if (!norm || !count) return -1;
    const rect = canvas.getBoundingClientRect();
    const r = Math.min(window.devicePixelRatio || 1, 2);
    return pickAt((cx - rect.left) * r, glY(cy - rect.top), 16 * r, true);   // selection pick: honor the pick filter
  }

  // Held so destroy() can undo them. A listener on `window` keeps this whole closure
  // alive for the life of the page — the WebGL2 context (preserveDrawingBuffer, so a
  // full-size back buffer), every GPU buffer, and the CPU copies of positions, colours
  // and density fields. One per viewer iframe is fine; a host that builds and discards
  // clouds (the GUI's metrics panels) leaked a context per panel until the tab died.
  let dead = false;
  const onResize = () => { if (!dead) render(); };
  window.addEventListener("resize", onResize);
  // Same reason as in plot.js: a canvas built while its tab is hidden has no size, so
  // whatever we drew went nowhere. Redraw when the box we actually occupy changes.
  const ro = window.ResizeObserver ? new ResizeObserver(() => { if (!dead) render(); }) : null;
  if (ro) ro.observe(canvas);

  // Hover, to match the 2D scatter. pick() is O(count), so coalesce to one test per
  // animation frame rather than one per mousemove event; and stay out of the way while
  // the camera is being dragged (any button down).
  let hoverPending = 0, hx = 0, hy = 0;
  function setHover(i) {
    if (i === hover) return;
    const prev = hover;
    hover = i;
    patchFlag(prev); patchFlag(i);   // two floats, not a 4 MB re-upload
    render();
  }
  canvas.addEventListener("mousemove", (e) => {
    if (e.buttons) { setHover(-1); return; }   // orbiting/panning, not pointing at anything
    hx = e.clientX; hy = e.clientY;
    if (hoverPending) return;
    hoverPending = requestAnimationFrame(() => { hoverPending = 0; setHover(pick(hx, hy)); });
  });
  canvas.addEventListener("mouseleave", () => setHover(-1));

  return {
    // Release what outlives the canvas: the global listener, the observer, pending frames,
    // and the GL context itself (browsers cap live contexts; waiting for GC is not a plan).
    destroy() {
      if (dead) return;
      dead = true;
      window.removeEventListener("resize", onResize);
      if (ro) ro.disconnect();
      if (hoverPending) { cancelAnimationFrame(hoverPending); hoverPending = 0; }
      norm = null; colorData = null; count = 0;
      const lose = gl.getExtension("WEBGL_lose_context");
      if (lose) lose.loseContext();
    },
    setPoints(pts) {
      norm = normalizePoints(pts); count = pts.length;
      hover = -1;                // the old hover index means nothing in a new cloud
      orbit.invalidatePivot();   // a new cloud is a new front surface
      invalidatePick();          // ...and new ids in the pick buffer
      invalidateSort();          // ...and a stale back-to-front order
      gl.bindBuffer(gl.ARRAY_BUFFER, posBuf); gl.bufferData(gl.ARRAY_BUFFER, norm, gl.STATIC_DRAW);
      // (re)seed colors so a fresh point set is never drawn black before applyColor()
      if (!colorData || colorData.length !== count * 3) { colorData = new Float32Array(count * 3); for (let i = 0; i < count * 3; i++) colorData[i] = [0.58, 0.64, 0.74][i % 3]; }
      gl.bindBuffer(gl.ARRAY_BUFFER, colBuf); gl.bufferData(gl.ARRAY_BUFFER, colorData, gl.STATIC_DRAW);
      uploadFlags(); render();
    },
    setColors(colors) { colorData = colors; gl.bindBuffer(gl.ARRAY_BUFFER, colBuf); gl.bufferData(gl.ARRAY_BUFFER, colors, gl.STATIC_DRAW); render(); },
    setPickFilter(fn) { pickFilter = typeof fn === "function" ? fn : null; },   // restrict selection picks to ids where fn(i) is true

    setDensityField(values) { hasDens = !!(values && values.length === count); gl.bindBuffer(gl.ARRAY_BUFFER, densBuf); gl.bufferData(gl.ARRAY_BUFFER, values || new Float32Array(count), gl.STATIC_DRAW); render(); },
    // Per-point share of the particles enclosed by the densest region containing it — what
    // makes a shell a statement about the data rather than about the colormap.
    setDensityQuantiles(values) { hasQuant = !!(values && values.length === count); gl.bindBuffer(gl.ARRAY_BUFFER, quantBuf); gl.bufferData(gl.ARRAY_BUFFER, values || new Float32Array(count), gl.STATIC_DRAW); render(); },
    setDensityNormals(values) { hasNorm = !!(values && values.length === count * 3); gl.bindBuffer(gl.ARRAY_BUFFER, normBuf); gl.bufferData(gl.ARRAY_BUFFER, values || new Float32Array(count * 3), gl.STATIC_DRAW); render(); },
    // { nx, ny, nz, data:Uint8Array (x fastest), lo, hi } in *data* coords — the object
    // landscapeDensityGrid() already returns for the Slices view.
    setDensityVolume(g) {
      hasVol = !!(g && g.data && g.data.length === g.nx * g.ny * g.nz);
      if (!hasVol || !volProg) { render(); return; }
      // The estimator caps its grid at WebGL2's guaranteed 256, but check the real limit
      // rather than trust it — an over-size upload is an INVALID_VALUE and a blank pane.
      const lim = gl.getParameter(gl.MAX_3D_TEXTURE_SIZE);
      if (Math.max(g.nx, g.ny, g.nz) > lim) {
        console.warn(`landscape: density grid ${g.nx}x${g.ny}x${g.nz} exceeds MAX_3D_TEXTURE_SIZE ${lim}`);
        hasVol = false; render(); return;
      }
      volDim = [g.nx, g.ny, g.nz];
      volLo = dataToNorm(g.lo || [0, 0, 0]);
      volHi = dataToNorm(g.hi || [1, 1, 1]);
      gl.bindTexture(gl.TEXTURE_3D, volTex);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage3D(gl.TEXTURE_3D, 0, gl.R8, g.nx, g.ny, g.nz, 0, gl.RED, gl.UNSIGNED_BYTE, g.data);
      gl.bindTexture(gl.TEXTURE_3D, null);
      render();
    },
    // "splats" | "shells" | "energy" | "volume" | "points"
    setDensityStyle(style) { densStyle = MODE_OF[style] === undefined ? "splats" : style; invalidateSort(); render(); },
    setDensityGamma(v) { densGamma = Math.max(0.1, v); render(); },
    setDensityKernel(v) { densKernel = Math.max(0.01, v); render(); },
    setDensityShade(on) { densShade = on ? 1 : 0; render(); },
    setDensityFog(on) { densFog = on ? 1 : 0; render(); },
    // levels: ascending percentages, e.g. [25, 50, 75, 90]; colors: [[r,g,b], ...] 0-255
    setDensityLevels(levels, colors, alphas) {
      const n = Math.min(6, levels.length + 1);
      shellN = n;
      for (let i = 0; i < 6; i++) {
        const q = i < levels.length ? levels[i] / 100 : 1.01;
        shellQ[i] = Math.min(1.01, q);
        const c = colors[Math.min(i, colors.length - 1)] || [255, 255, 255];
        shellC[i * 3] = c[0] / 255; shellC[i * 3 + 1] = c[1] / 255; shellC[i * 3 + 2] = c[2] / 255;
        shellA[i] = alphas[Math.min(i, alphas.length - 1)] ?? 0.1;
      }
      render();
    },
    hasVolumeSupport() { return !!volProg; },
    // batch(() => { ...many setters... }) redraws once, at the end
    batch(fn) { batching++; try { fn(); } finally { batching--; } render(); },
    showDensity(on) { showDens = on; render(); },
    setDensityIntensity(v) { densIntensity = Math.max(0, v); render(); },
    setColormap(stops) {
      if (!stops || stops.length < 2) return;
      const n = Math.min(6, stops.length);
      const flat = new Float32Array(18);
      for (let i = 0; i < n; i++) { flat[i * 3] = stops[i][0] / 255; flat[i * 3 + 1] = stops[i][1] / 255; flat[i * 3 + 2] = stops[i][2] / 255; }
      rampFlat = flat; rampN = n; if (showDens) render();
    },
    setPointAlpha(a) { pointAlpha = Math.max(0, Math.min(1, a)); render(); },
    select(i) { selected = i; uploadFlags(); render(); },
    setPath(arr) { pathArr = arr || []; pathSet = new Set(pathArr); uploadFlags(); render(); },
    setPathOrder(arr, colorRgb) {
      const ids = (arr || []).filter((i) => i >= 0 && i < count);
      const buf = new Float32Array(ids.length * 3);
      for (let k = 0; k < ids.length; k++) { buf[k * 3] = norm[ids[k] * 3]; buf[k * 3 + 1] = norm[ids[k] * 3 + 1]; buf[k * 3 + 2] = norm[ids[k] * 3 + 2]; }
      lineData = buf; lineCount = ids.length; if (colorRgb) lineColor = colorRgb;
      render();
    },
    // Free-spline overlays: list of { pts:[[x,y,z]...] data coords, color, handles? }.
    setFreeCurves(list) {
      const norms = (arr) => { const f = new Float32Array(arr.length * 3); for (let k = 0; k < arr.length; k++) { const q = dataToNorm(arr[k]); f[k * 3] = q[0]; f[k * 3 + 1] = q[1]; f[k * 3 + 2] = q[2]; } return f; };
      freeCurves = (list || []).map((c) => {
        const pts = c.pts || [];
        return { flat: norms(pts), n: pts.length, color: typeof c.color === "string" ? hexToRgb01(c.color) : (c.color || [0.957, 0.447, 0.714]) };
      });
      const handlePts = (list || []).flatMap((c) => c.handles || []);
      handleData = norms(handlePts); handleCount = handlePts.length;
      gl.bindBuffer(gl.ARRAY_BUFFER, handleBuf); gl.bufferData(gl.ARRAY_BUFFER, handleData, gl.DYNAMIC_DRAW);
      render();
    },
    // Project data-coord points to CSS screen px (null where behind the camera).
    projectData(pts) {
      const out = new Array(pts.length).fill(null);
      if (!norm) return out;
      const { proj, view } = matrices(), mvp = multiply(proj, view);
      const W = canvas.clientWidth, H = canvas.clientHeight;
      for (let i = 0; i < pts.length; i++) {
        const np = dataToNorm(pts[i]); const c = mulVec4(mvp, [np[0], np[1], np[2], 1]);
        if (c[3] <= 0) continue;
        out[i] = [(c[0] / c[3] * 0.5 + 0.5) * W, (1 - (c[1] / c[3] * 0.5 + 0.5)) * H];
      }
      return out;
    },
    // Unproject a CSS screen px to data coords. The screen tells us x and y; the depth has
    // to come from the cloud, and it comes from the point under the cursor — an id-buffer
    // read, where this used to scan all N points on every mouse move while drawing a curve.
    //
    // Over empty space the id buffer has nothing to offer, so fall back to the plane
    // through the cloud's centre. The old scan returned the globally nearest point's
    // depth, however far away it was; a stable centre plane is both cheaper and a steadier
    // surface to draw on.
    unproject(sx, sy) {
      if (!norm || !count) return null;
      const { proj, view } = matrices(), mvp = multiply(proj, view);
      const W = canvas.clientWidth, H = canvas.clientHeight;
      const r = Math.min(window.devicePixelRatio || 1, 2);
      const i = pickAt(sx * r, glY(sy), 96 * r);
      const at = i >= 0 ? [norm[i * 3], norm[i * 3 + 1], norm[i * 3 + 2], 1] : [0, 0, 0, 1];
      const c = mulVec4(mvp, at);
      if (c[3] <= 0) return null;
      const ndcZ = c[2] / c[3];
      const im = invert(mvp); if (!im) return null;
      const w = mulVec4(im, [sx / W * 2 - 1, (1 - sy / H) * 2 - 1, ndcZ, 1]);
      return normToData([w[0] / w[3], w[1] / w[3], w[2] / w[3]]);
    },
    setSelection(arr) { selArr = arr || []; selSet = new Set(selArr); uploadFlags(); render(); },
    // CSS-space screen coords of every point (null where behind the camera)
    projectedPoints() {
      const out = new Array(count).fill(null);
      if (!norm) return out;
      const { proj, view } = matrices();
      const mvp = multiply(proj, view);
      const W = canvas.clientWidth, H = canvas.clientHeight;
      for (let i = 0; i < count; i++) {
        const c = mulVec4(mvp, [norm[i * 3], norm[i * 3 + 1], norm[i * 3 + 2], 1]);
        if (c[3] <= 0) continue;
        out[i] = [(c[0] / c[3] * 0.5 + 0.5) * W, (1 - (c[1] / c[3] * 0.5 + 0.5)) * H];
      }
      return out;
    },
    setBackground(rgb) { bg = rgb; render(); },
    onRender(cb) { onRenderCb = cb; },   // called after every render (overlay re-syncs to the camera)
    orbitRotate(dx, dy) { orbit.drag(dx, dy); },   // rotate the camera from an external drag (overlay edit mode)
    orbitPan(dx, dy) { orbit.pan(dx, dy); },       // pan the camera from an external drag (overlay / middle-mouse)
    orbitZoom(e) { orbit.wheel(e); },              // wheel-zoom forwarded from the overlay
    resetView() { orbit.reset(); },
    render,
    get selected() { return selected; },
    async snapshot({ scale = 2, transparent = false } = {}) {
      const ow = canvas.width, oh = canvas.height;
      snapping = true;
      canvas.width = Math.max(1, Math.floor(canvas.clientWidth * scale));
      canvas.height = Math.max(1, Math.floor(canvas.clientHeight * scale));
      drawScene(transparent ? 0 : 1);
      const blob = await new Promise((res) => canvas.toBlob(res, "image/png"));
      snapping = false;
      canvas.width = ow; canvas.height = oh; render();
      return blob;
    },
  };
}

function hexToRgb01(h) {
  h = String(h).replace("#", "");
  if (h.length === 3) h = h.split("").map((x) => x + x).join("");
  const n = parseInt(h, 16) || 0;
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
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
