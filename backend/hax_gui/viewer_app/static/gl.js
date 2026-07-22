// WebGL2 isosurface renderer with screen-space ambient occlusion (SSAO).
//
// Pipeline (when EXT_color_buffer_float is available):
//   1. geometry prepass  -> offscreen MRT: view-space position + normal (RGBA16F)
//   2. SSAO pass          -> hemisphere-kernel occlusion into an AO texture
//   3. blur pass          -> 4x4 box blur to denoise the AO
//   4. beauty pass        -> re-render the mesh to the default (MSAA) framebuffer
//                            with Blinn-Phong key+fill+headlight + specular, the
//                            blurred AO multiplied in, and depth cueing.
// Re-rendering the geometry for the beauty pass keeps the default framebuffer's
// multisampling, so silhouettes stay smooth (deferred shading would alias them).
// Falls back to a single forward pass (derivative AO) when float targets are
// unsupported. Hand-written, no dependencies.
import { projection, lookAt } from "./mat4.js";
import { createOrbit } from "./orbit.js";

const KSIZE = 24;   // SSAO hemisphere samples

// Shared mesh vertex shader: view-space position + normal to the fragment stage.
const VERT = `#version 300 es
precision highp float;
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
uniform mat4 uProj;
uniform mat4 uView;
uniform mat4 uModel;
out vec3 vNormal;
out vec3 vViewPos;
void main() {
  vec4 vp = uView * uModel * vec4(aPos, 1.0);
  vViewPos = vp.xyz;
  vNormal = mat3(uView * uModel) * aNormal;
  gl_Position = uProj * vp;
}`;

// Geometry prepass: write view position (w=1 marks geometry) + view normal.
const FRAG_GEOM = `#version 300 es
precision highp float;
in vec3 vNormal;
in vec3 vViewPos;
layout(location=0) out vec4 outPos;
layout(location=1) out vec4 outNrm;
void main() {
  vec3 n = normalize(vNormal);
  if (!gl_FrontFacing) n = -n;
  outPos = vec4(vViewPos, 1.0);
  outNrm = vec4(n, 0.0);
}`;

// Fullscreen-triangle vertex shader (no buffers; derived from gl_VertexID).
const VERT_FS = `#version 300 es
precision highp float;
out vec2 vUv;
void main() {
  vec2 p = vec2(float((gl_VertexID & 1) << 2) - 1.0, float((gl_VertexID & 2) << 1) - 1.0);
  vUv = p * 0.5 + 0.5;
  gl_Position = vec4(p, 0.0, 1.0);
}`;

// SSAO: for each fragment, sample a normal-oriented hemisphere of points around
// the view-space position, project them to screen, and count how many are hidden
// behind nearer geometry. Range check fades occlusion when the occluder is far.
const FRAG_SSAO = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uPos;
uniform sampler2D uNrm;
uniform sampler2D uNoise;
uniform mat4 uProj;
uniform vec2 uNoiseScale;
uniform vec3 uKernel[${KSIZE}];
uniform float uRadius;
uniform float uBias;
out vec4 outColor;
void main() {
  vec4 pw = texture(uPos, vUv);
  if (pw.w < 0.5) { outColor = vec4(1.0); return; }   // background: unoccluded
  vec3 p = pw.xyz;
  vec3 n = normalize(texture(uNrm, vUv).xyz);
  vec3 rnd = texture(uNoise, vUv * uNoiseScale).xyz * 2.0 - 1.0;
  vec3 t = normalize(rnd - n * dot(rnd, n));
  vec3 b = cross(n, t);
  mat3 tbn = mat3(t, b, n);
  float occ = 0.0;
  for (int i = 0; i < ${KSIZE}; i++) {
    vec3 sp = p + tbn * uKernel[i] * uRadius;          // view-space sample point
    vec4 off = uProj * vec4(sp, 1.0);
    off.xyz /= off.w;
    off.xyz = off.xyz * 0.5 + 0.5;                     // -> [0,1] screen
    if (off.x < 0.0 || off.x > 1.0 || off.y < 0.0 || off.y > 1.0) continue;
    vec4 sd = texture(uPos, off.xy);
    if (sd.w < 0.5) continue;
    float sampleZ = sd.z;                              // stored geometry depth (view z, -forward)
    float rangeCheck = smoothstep(0.0, 1.0, uRadius / max(abs(p.z - sampleZ), 1e-4));
    occ += (sampleZ >= sp.z + uBias ? 1.0 : 0.0) * rangeCheck;
  }
  outColor = vec4(vec3(1.0 - occ / float(${KSIZE})), 1.0);
}`;

// Separable-ish 4x4 box blur over the noisy AO (matches the 4x4 noise tile).
const FRAG_BLUR = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uAO;
uniform vec2 uTexel;
out vec4 outColor;
void main() {
  float sum = 0.0;
  for (int x = -2; x < 2; x++)
    for (int y = -2; y < 2; y++)
      sum += texture(uAO, vUv + vec2(float(x), float(y)) * uTexel).r;
  outColor = vec4(vec3(sum / 16.0), 1.0);
}`;

// Beauty/lighting pass. uUseSSAO=1 samples the blurred AO texture by screen
// position; uUseSSAO=0 falls back to a cheap derivative AO (no float targets).
const FRAG_BEAUTY = `#version 300 es
precision highp float;
in vec3 vNormal;
in vec3 vViewPos;
uniform vec3 uColor;
uniform float uAmbient;   // base fill
uniform float uFill;      // upper-left key weight
uniform float uSpec;      // specular strength
uniform float uShine;     // specular exponent
uniform float uAO;        // derivative-AO strength (fallback)
uniform float uSSAO;      // SSAO strength (0..1)
uniform float uDepthCue;  // distance darkening
uniform float uCamDist;   // camera distance (depth-cue range)
uniform sampler2D uAOTex; // blurred SSAO
uniform vec2 uResolution;
uniform int uUseSSAO;
out vec4 outColor;
const vec3 KEY = vec3(-0.355, 0.450, 0.819);   // upper-left key (normalized below)
void main() {
  vec3 n = normalize(vNormal);
  if (!gl_FrontFacing) n = -n;
  vec3 v = normalize(-vViewPos);
  vec3 k = normalize(KEY);
  // bright, even base: half-Lambert camera headlight (never below 0.5).
  float nl = max(dot(n, v), 0.0);
  float wrap = nl * 0.5 + 0.5;
  float kd = max(dot(n, k), 0.0) * uFill;     // upper-left key adds gentle form shading
  float diff = uAmbient + (1.0 - uAmbient) * min(wrap + kd, 1.35);
  vec3 hh = normalize(k + v);
  float spec = pow(max(dot(n, hh), 0.0), uShine) * uSpec;
  float ao;
  if (uUseSSAO == 1) {
    float a = texture(uAOTex, gl_FragCoord.xy / uResolution).r;
    ao = mix(1.0, a, uSSAO);
  } else {
    // fallback: screen-space normal variation (no float render targets).
    ao = 1.0 - clamp(length(fwidth(n)) * 1.4 * uAO, 0.0, 0.16);
  }
  float far = (-vViewPos.z) - (uCamDist - 1.2);
  float dc = 1.0 - clamp(far * 0.26 * uDepthCue, 0.0, 0.32);
  vec3 col = uColor * diff * ao * dc + vec3(spec);
  outColor = vec4(col, 1.0);
}`;

// `radius` is the SSAO sampling scale (view-space units; the map is ~unit-sized).
// A tight radius darkens only deep contacts (crisp, for standard); a broad radius
// gives the soft, voluminous recess-shadowing ChimeraX shows in smooth mode.
const LIGHTING = {
  standard:  { ambient: 0.46, fill: 0.50, spec: 0.18, shine: 24.0, ao: 0.35, ssao: 0.55, radius: 0.12, depthCue: 0.14 },
  smooth:    { ambient: 0.47, fill: 0.30, spec: 0.08, shine: 16.0, ao: 0.15, ssao: 0.95, radius: 0.32, depthCue: 0.18 },
  raytraced: { ambient: 0.40, fill: 0.40, spec: 0.42, shine: 50.0, ao: 0.60, ssao: 0.80, radius: 0.18, depthCue: 0.22 },
};

const SSAO_RADIUS = 0.12, SSAO_BIAS = 0.018;   // defaults / fallback

export function createRenderer(canvas) {
  // preserveDrawingBuffer lets us read the canvas back for snapshots/recording.
  const gl = canvas.getContext("webgl2", { antialias: true, depth: true, preserveDrawingBuffer: true, premultipliedAlpha: false });
  if (!gl) throw new Error("WebGL2 is not available in this browser.");

  // SSAO needs to render into float color targets.
  let ssaoOK = !!gl.getExtension("EXT_color_buffer_float");

  const beauty = makeProg(gl, VERT, FRAG_BEAUTY);
  const geomP = ssaoOK ? makeProg(gl, VERT, FRAG_GEOM) : null;
  const ssaoP = ssaoOK ? makeProg(gl, VERT_FS, FRAG_SSAO) : null;
  const blurP = ssaoOK ? makeProg(gl, VERT_FS, FRAG_BLUR) : null;

  // Mesh geometry, one entry per model in the scene. A lone map is just the case where
  // there is one of them, so the single-map path is not a separate code path.
  //   { vao, posBuf, normBuf, idxBuf, count, color|null, visible, bounds }
  // Insertion order is draw order; `color: null` means "use the viewer's global colour",
  // which is what a single unnamed map has always used.
  const models = new Map();

  function makeModel() {
    const m = { vao: gl.createVertexArray(), posBuf: gl.createBuffer(), normBuf: gl.createBuffer(),
                idxBuf: gl.createBuffer(), count: 0, color: null, visible: true, bounds: null };
    gl.bindVertexArray(m.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, m.posBuf); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, m.normBuf); gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, m.idxBuf);
    gl.bindVertexArray(null);
    return m;
  }
  const drawable = () => [...models.values()].filter((m) => m.visible && m.count);
  // Whether *anything* is on screen. SSAO and the snapshot path both key off this.
  const anyGeometry = () => drawable().length > 0;
  const quadVao = gl.createVertexArray();   // empty VAO for fullscreen passes
  gl.enable(gl.DEPTH_TEST);

  // 1x1 white texture (bound to the AO sampler on the forward-fallback path).
  const whiteTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, whiteTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([255, 255, 255, 255]));
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

  // SSAO hemisphere kernel + tiling rotation-noise texture.
  let kernel = null, noiseTex = null;
  if (ssaoOK) {
    kernel = makeKernel(KSIZE);
    noiseTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, noiseTex);
    const noise = new Uint8Array(16 * 4);
    for (let i = 0; i < 16; i++) {
      noise[i * 4] = (Math.random() * 256) | 0;
      noise[i * 4 + 1] = (Math.random() * 256) | 0;
      noise[i * 4 + 2] = 128;   // z=0 after *2-1: rotation stays in the tangent plane
      noise[i * 4 + 3] = 255;
    }
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 4, 4, 0, gl.RGBA, gl.UNSIGNED_BYTE, noise);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
  }

  // offscreen targets (allocated lazily, resized to match the canvas)
  let aw = 0, ah = 0;
  let posTex = null, nrmTex = null, depthRbo = null, gFbo = null;
  let aoTex = null, aoFbo = null, blurTex = null, blurFbo = null;

  let color = [0.863, 0.863, 0.863];   // ChimeraX-style near-white neutral gray
  let bg = [0.016, 0.024, 0.06];
  let light = { ...LIGHTING.standard };
  // Distance from the eye to the front-most surface at the centre of the window, which
  // is what the camera turns into a ChimeraX-style centre of rotation. The SSAO
  // geometry prepass already parks view-space positions in `posTex` (w=1 marks
  // geometry) for exactly the camera we are looking at, so this costs a small readback
  // rather than another pass. Null on the forward fallback (no G-buffer) or over
  // background, which leaves the camera on the pivot it already had.
  function centreDepth() {
    const W = canvas.width, H = canvas.height, N = 5;   // a few px wide, so a 1px gap doesn't read as "empty"
    if (!ssaoOK || !anyGeometry() || !gFbo || W < N || H < N) return null;
    const x = (W >> 1) - (N >> 1), y = (H >> 1) - (N >> 1);
    let px;
    try {
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, gFbo);
      gl.readBuffer(gl.COLOR_ATTACHMENT0);
      // An RGBA16F attachment is only readable through the pair the driver advertises.
      if (gl.getParameter(gl.IMPLEMENTATION_COLOR_READ_FORMAT) !== gl.RGBA) return null;
      const type = gl.getParameter(gl.IMPLEMENTATION_COLOR_READ_TYPE);
      if (type === gl.FLOAT) {
        px = new Float32Array(4 * N * N);
        gl.readPixels(x, y, N, N, gl.RGBA, gl.FLOAT, px);
      } else if (type === gl.HALF_FLOAT) {
        const half = new Uint16Array(4 * N * N);
        gl.readPixels(x, y, N, N, gl.RGBA, gl.HALF_FLOAT, half);
        px = Float32Array.from(half, fromHalf);
      } else return null;
    } catch {
      return null;
    } finally {
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
    }
    let best = null;
    for (let i = 0; i < N * N; i++) {
      if (px[i * 4 + 3] < 0.5) continue;   // pos.w = 0 => background
      const t = -px[i * 4 + 2];            // view-space z is negative in front of the eye
      if (t > 0 && (best === null || t < best)) best = t;
    }
    return best;
  }

  // fitRadius 1.33 keeps the map the on-screen size it has always been (the old
  // 45°-vertical camera at dist 3.2 showed 3.2·tan(22.5°) ≈ 1.33 world units of height);
  // the camera now reaches that framing from further back, through ChimeraX's field.
  const orbit = createOrbit(canvas, () => render(), null,
    { fitRadius: 1.33, probeDepth: centreDepth, modelBounds: sceneBounds });

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr)), h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  }

  function mkTex(w, h, internal, format, type, filter) {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, type, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }
  function fboWith(attachments, depth) {
    const f = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, f);
    attachments.forEach((tex, i) => gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, tex, 0));
    if (depth) gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depth);
    return f;
  }

  function ensureTargets() {
    if (!ssaoOK) return;
    const W = canvas.width, H = canvas.height;
    if (W === aw && H === ah && gFbo) return;
    aw = W; ah = H;
    [posTex, nrmTex, aoTex, blurTex].forEach((t) => t && gl.deleteTexture(t));
    [gFbo, aoFbo, blurFbo].forEach((f) => f && gl.deleteFramebuffer(f));
    if (depthRbo) gl.deleteRenderbuffer(depthRbo);

    posTex = mkTex(W, H, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.NEAREST);
    nrmTex = mkTex(W, H, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.NEAREST);
    depthRbo = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, depthRbo);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, W, H);
    gFbo = fboWith([posTex, nrmTex], depthRbo);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) { ssaoOK = false; gl.bindFramebuffer(gl.FRAMEBUFFER, null); return; }

    aoTex = mkTex(W, H, gl.R8, gl.RED, gl.UNSIGNED_BYTE, gl.LINEAR);
    aoFbo = fboWith([aoTex], null);
    blurTex = mkTex(W, H, gl.R8, gl.RED, gl.UNSIGNED_BYTE, gl.LINEAR);
    blurFbo = fboWith([blurTex], null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  function camera() {
    const aspect = canvas.width / Math.max(1, canvas.height);
    const proj = projection(aspect, 0.01, 100);
    const view = lookAt(orbit.eye, [0, 0, 0], [0, 1, 0]);   // camera fixed; model carries the trackball rotation + pan
    return { proj, view, model: orbit.model };
  }
  // Draw every visible model. `colorLoc` is the beauty shader's uColor — passed so the
  // colour is set per model rather than once per pass; the geometry prepass has no
  // colour and passes null.
  function drawModels(colorLoc) {
    for (const m of drawable()) {
      if (colorLoc) gl.uniform3fv(colorLoc, m.color || color);
      gl.bindVertexArray(m.vao);
      gl.drawElements(gl.TRIANGLES, m.count, gl.UNSIGNED_INT, 0);
    }
    gl.bindVertexArray(null);
  }
  function drawQuad() { gl.bindVertexArray(quadVao); gl.drawArrays(gl.TRIANGLES, 0, 3); gl.bindVertexArray(null); }

  function setMatrices(p) {
    const { proj, view, model } = camera();
    gl.uniformMatrix4fv(p.u("uProj"), false, proj);
    gl.uniformMatrix4fv(p.u("uView"), false, view);
    gl.uniformMatrix4fv(p.u("uModel"), false, model);
  }
  function setLighting(useSSAO) {
    gl.uniform1f(beauty.u("uAmbient"), light.ambient);
    gl.uniform1f(beauty.u("uFill"), light.fill);
    gl.uniform1f(beauty.u("uSpec"), light.spec);
    gl.uniform1f(beauty.u("uShine"), light.shine);
    gl.uniform1f(beauty.u("uAO"), light.ao);
    gl.uniform1f(beauty.u("uSSAO"), light.ssao);
    gl.uniform1f(beauty.u("uDepthCue"), light.depthCue);
    gl.uniform1f(beauty.u("uCamDist"), orbit.dist);
    gl.uniform2f(beauty.u("uResolution"), canvas.width, canvas.height);
    gl.uniform1i(beauty.u("uUseSSAO"), useSSAO ? 1 : 0);
    gl.uniform1i(beauty.u("uAOTex"), 3);
  }

  function drawSSAO(alpha) {
    const W = canvas.width, H = canvas.height;
    // 1. geometry prepass -> position + normal G-buffer
    gl.bindFramebuffer(gl.FRAMEBUFFER, gFbo);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    gl.viewport(0, 0, W, H);
    gl.enable(gl.DEPTH_TEST);
    gl.clearColor(0, 0, 0, 0);   // pos.w=0 => background
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.useProgram(geomP.prog);
    setMatrices(geomP);
    drawModels(null);

    // 2. SSAO
    gl.bindFramebuffer(gl.FRAMEBUFFER, aoFbo);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
    gl.viewport(0, 0, W, H);
    gl.disable(gl.DEPTH_TEST);
    gl.useProgram(ssaoP.prog);
    bindTex(0, posTex); bindTex(1, nrmTex); bindTex(2, noiseTex);
    gl.uniform1i(ssaoP.u("uPos"), 0); gl.uniform1i(ssaoP.u("uNrm"), 1); gl.uniform1i(ssaoP.u("uNoise"), 2);
    gl.uniformMatrix4fv(ssaoP.u("uProj"), false, camera().proj);
    gl.uniform3fv(ssaoP.u("uKernel"), kernel);
    gl.uniform1f(ssaoP.u("uRadius"), light.radius || SSAO_RADIUS);
    gl.uniform1f(ssaoP.u("uBias"), SSAO_BIAS);
    gl.uniform2f(ssaoP.u("uNoiseScale"), W / 4, H / 4);
    drawQuad();

    // 3. blur
    gl.bindFramebuffer(gl.FRAMEBUFFER, blurFbo);
    gl.viewport(0, 0, W, H);
    gl.useProgram(blurP.prog);
    bindTex(0, aoTex);
    gl.uniform1i(blurP.u("uAO"), 0);
    gl.uniform2f(blurP.u("uTexel"), 1 / W, 1 / H);
    drawQuad();

    // 4. beauty pass into the default (MSAA) framebuffer
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, W, H);
    gl.enable(gl.DEPTH_TEST);
    gl.clearColor(bg[0], bg[1], bg[2], alpha);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.useProgram(beauty.prog);
    setMatrices(beauty);
    setLighting(true);
    bindTex(3, blurTex);
    drawModels(beauty.u("uColor"));
  }

  function drawForward(alpha) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.enable(gl.DEPTH_TEST);
    gl.clearColor(bg[0], bg[1], bg[2], alpha);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (!anyGeometry()) return;
    gl.useProgram(beauty.prog);
    setMatrices(beauty);
    setLighting(false);
    bindTex(3, whiteTex);
    drawModels(beauty.u("uColor"));
  }

  function bindTex(unit, tex) { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, tex); }

  function paint(alpha) {
    if (ssaoOK && anyGeometry()) { ensureTargets(); }
    if (ssaoOK && anyGeometry()) drawSSAO(alpha);
    else drawForward(alpha);
  }

  function render() { resize(); paint(1); }

  // Axis-aligned box enclosing the vertices, for the camera's centre-of-rotation rule.
  // A single pass over the positions, once per contour — the surface is what the box is
  // meant to describe, and it is the only thing here that knows where the surface is.
  function aabb(positions) {
    if (!positions.length) return null;
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < positions.length; i += 3) {
      for (let a = 0; a < 3; a++) {
        const v = positions[i + a];
        if (v < lo[a]) lo[a] = v;
        if (v > hi[a]) hi[a] = v;
      }
    }
    return isFinite(lo[0]) ? { lo, hi } : null;
  }

  // Union of what is on screen — the camera frames and pivots on the whole scene, not
  // on whichever model happens to be first.
  function sceneBounds() {
    let lo = null, hi = null;
    for (const m of drawable()) {
      if (!m.bounds) continue;
      if (!lo) { lo = m.bounds.lo.slice(); hi = m.bounds.hi.slice(); continue; }
      for (let a = 0; a < 3; a++) {
        lo[a] = Math.min(lo[a], m.bounds.lo[a]);
        hi[a] = Math.max(hi[a], m.bounds.hi[a]);
      }
    }
    return lo ? { lo, hi } : null;
  }

  function setModelMesh(id, positions, normals, indices) {
    let m = models.get(id);
    if (!m) models.set(id, (m = makeModel()));
    gl.bindBuffer(gl.ARRAY_BUFFER, m.posBuf); gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, m.normBuf); gl.bufferData(gl.ARRAY_BUFFER, normals, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, m.idxBuf); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
    m.count = indices.length;
    m.bounds = indices.length ? aabb(positions) : null;
    orbit.invalidatePivot();   // a new contour is a new front surface
    render();
  }

  function dropModel(id) {
    const m = models.get(id);
    if (!m) return;
    gl.deleteVertexArray(m.vao);
    gl.deleteBuffer(m.posBuf); gl.deleteBuffer(m.normBuf); gl.deleteBuffer(m.idxBuf);
    models.delete(id);
    orbit.invalidatePivot();
    render();
  }

  window.addEventListener("resize", render);

  return {
    render,
    // Single-map callers (the landscape viewer) address the one implicit model.
    setMesh: (positions, normals, indices) => setModelMesh(0, positions, normals, indices),
    setModelMesh, dropModel,
    setModelColor(id, rgb) { const m = models.get(id); if (m) { m.color = rgb; render(); } },
    setModelVisible(id, on) {
      const m = models.get(id);
      if (!m || m.visible === !!on) return;
      m.visible = !!on;
      orbit.invalidatePivot();   // hiding the front map exposes a different front surface
      render();
    },
    clear() { for (const id of [...models.keys()]) dropModel(id); render(); },
    setColor(rgb) { color = rgb; render(); },
    setBackground(rgb) { bg = rgb; render(); },
    setLighting(mode) { light = { ...(LIGHTING[mode] || LIGHTING.standard) }; render(); },
    resetView() { orbit.reset(); },
    getCamera() { return orbit.getState(); },   // handed to the other renderer on a render-mode switch
    setCamera(cam) { orbit.setState(cam); },
    captureStream(fps) { return canvas.captureStream(fps); },
    async snapshot({ scale = 2, transparent = false } = {}) {
      const ow = canvas.width, oh = canvas.height;
      canvas.width = Math.floor(canvas.clientWidth * scale);
      canvas.height = Math.floor(canvas.clientHeight * scale);
      if (ssaoOK && anyGeometry()) ensureTargets();
      paint(transparent ? 0 : 1);
      const blob = await new Promise((res) => canvas.toBlob(res, "image/png"));
      canvas.width = ow; canvas.height = oh; render();
      return blob;
    },
  };
}

// IEEE half -> float. readPixels only hands back the type the driver advertises, and
// on some drivers that is HALF_FLOAT rather than FLOAT.
function fromHalf(h) {
  const sign = h & 0x8000 ? -1 : 1, exp = (h >> 10) & 0x1f, frac = h & 0x3ff;
  if (exp === 0) return sign * 2 ** -14 * (frac / 1024);
  if (exp === 31) return frac ? NaN : sign * Infinity;
  return sign * 2 ** (exp - 15) * (1 + frac / 1024);
}

// Hemisphere kernel: points in the +z half, packed toward the origin so nearby
// occluders dominate (the usual accelerating distribution).
function makeKernel(n) {
  const k = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    let x = Math.random() * 2 - 1, y = Math.random() * 2 - 1, z = Math.random();
    const len = Math.hypot(x, y, z) || 1;
    let s = i / n; s = 0.1 + 0.9 * s * s;
    k[i * 3] = (x / len) * s; k[i * 3 + 1] = (y / len) * s; k[i * 3 + 2] = (z / len) * s;
  }
  return k;
}

function makeProg(gl, vs, fs) {
  const prog = link(gl, vs, fs);
  const cache = {};
  return { prog, u(n) { if (!(n in cache)) cache[n] = gl.getUniformLocation(prog, n); return cache[n]; } };
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
