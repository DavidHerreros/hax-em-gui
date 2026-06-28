// Minimal WebGL2 3D point-cloud renderer for the latent landscape (3 PCs).
// Orbit camera (shared style with gl.js), per-point color, selection/target
// rings, CPU-side picking, and high-DPI / transparent snapshots. No deps.
import { perspective, lookAt, multiply, mulVec4, invert } from "./mat4.js";
import { createOrbit } from "./orbit.js";

const VERT = `#version 300 es
precision highp float;
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aColor;
layout(location=2) in float aFlag;   // 0 normal, 1 current, 2 path, 3 selection
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
    vec3 ring = vFlag > 2.5 ? vec3(0.95,0.96,0.98)      // selection (white)
              : vFlag > 1.5 ? vec3(0.957,0.447,0.714)   // path (pink)
                            : vec3(0.176,0.831,0.749);  // current (teal)
    outColor = vec4(ring, 1.0); return;
  }
  if (uAlpha < 0.01) discard;                  // points hidden
  float shade = 1.0 - r * 0.5;                 // subtle sphere shading
  outColor = vec4(vColor * shade, uAlpha);
}`;

// Additive "density glow" splats — large soft sprites on a heat ramp, summed with
// additive blending so dense regions brighten (the 3D analogue of the 2D heatmap).
const VERT_SPLAT = `#version 300 es
precision highp float;
layout(location=0) in vec3 aPos;
layout(location=1) in float aDens;   // local density 0..1
uniform mat4 uProj;
uniform mat4 uView;
uniform float uSize;
out float vDens;
void main() { vDens = aDens; gl_Position = uProj * uView * vec4(aPos, 1.0); gl_PointSize = uSize * (0.45 + aDens); }`;

const FRAG_SPLAT = `#version 300 es
precision highp float;
in float vDens;
uniform float uIntensity;
uniform vec3 uRamp[6];     // density colormap stops (low->high)
uniform int uRampN;        // number of active stops
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
  float a = exp(-dot(d, d) * 9.0);          // gaussian falloff (tighter)
  if (a < 0.01) discard;
  outColor = vec4(ramp(vDens), a * (0.03 + 0.32 * vDens) * uIntensity);
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

  const prog = link(gl, VERT, FRAG);
  const u = (n) => gl.getUniformLocation(prog, n);
  const uProj = u("uProj"), uView = u("uView"), uSize = u("uSize"), uAlpha = u("uAlpha");

  const splatProg = link(gl, VERT_SPLAT, FRAG_SPLAT);
  const su = (n) => gl.getUniformLocation(splatProg, n);
  const sProj = su("uProj"), sView = su("uView"), sSize = su("uSize"), sIntensity = su("uIntensity");
  const sRamp = su("uRamp"), sRampN = su("uRampN");
  let rampFlat = new Float32Array([0.27, 0.0, 0.33, 0.23, 0.32, 0.55, 0.13, 0.57, 0.55, 0.37, 0.79, 0.38, 0.99, 0.91, 0.15, 0, 0, 0]);   // viridis (normalized)
  let rampN = 5;

  const vao = gl.createVertexArray();
  const posBuf = gl.createBuffer(), colBuf = gl.createBuffer(), flagBuf = gl.createBuffer(), densBuf = gl.createBuffer();
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, colBuf); gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, flagBuf); gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 1, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);

  const splatVao = gl.createVertexArray();   // aPos (shared) + aDens
  gl.bindVertexArray(splatVao);
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, densBuf); gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 1, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);

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
  let selected = -1, pathArr = [], selArr = [];

  const orbit = createOrbit(canvas, () => render(), (e) => { const i = pick(e.clientX, e.clientY); if (i >= 0) onSelect(i, { shift: e.shiftKey }); });
  function matrices() {
    const aspect = canvas.width / Math.max(1, canvas.height);
    const proj = perspective((45 * Math.PI) / 180, aspect, 0.01, 100);
    // camera fixed on +Z; the model is rotated by the trackball matrix (baked into view).
    const view = multiply(lookAt(orbit.eye, [0, 0, 0], [0, 1, 0]), orbit.rotation);
    return { proj, view };
  }
  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr)), h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  }
  function drawScene(alpha = 1) {
    const { proj, view } = matrices();
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(bg[0], bg[1], bg[2], alpha);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (!count) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    // density glow overlay (additive, behind the points, no depth writes)
    if (showDens && hasDens) {
      gl.useProgram(splatProg);
      gl.uniformMatrix4fv(sProj, false, proj);
      gl.uniformMatrix4fv(sView, false, view);
      gl.uniform1f(sSize, 34.0 * dpr);
      gl.uniform1f(sIntensity, densIntensity);
      gl.uniform3fv(sRamp, rampFlat);
      gl.uniform1i(sRampN, rampN);
      gl.depthMask(false); gl.disable(gl.DEPTH_TEST);
      gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
      gl.bindVertexArray(splatVao); gl.drawArrays(gl.POINTS, 0, count); gl.bindVertexArray(null);
      gl.disable(gl.BLEND); gl.enable(gl.DEPTH_TEST); gl.depthMask(true);
    }
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
  function render() { resize(); drawScene(1); if (onRenderCb) onRenderCb(); }
  let onRenderCb = null;

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

  function uploadFlags() {
    const f = new Float32Array(count);
    for (const si of selArr) if (si >= 0 && si < count) f[si] = 3;    // selection = white
    for (const pi of pathArr) if (pi >= 0 && pi < count) f[pi] = 2;   // path = pink
    if (selected >= 0 && selected < count) f[selected] = 1;           // current = teal (wins)
    gl.bindBuffer(gl.ARRAY_BUFFER, flagBuf); gl.bufferData(gl.ARRAY_BUFFER, f, gl.DYNAMIC_DRAW);
  }

  function pick(cx, cy) {
    if (!norm) return -1;
    const rect = canvas.getBoundingClientRect();
    const r = Math.min(window.devicePixelRatio || 1, 2);
    const px = (cx - rect.left) * r, py = (cy - rect.top) * r;
    const { proj, view } = matrices();
    const mvp = multiply(proj, view);
    let best = -1, bestD = Infinity;
    for (let i = 0; i < count; i++) {
      const c = mulVec4(mvp, [norm[i * 3], norm[i * 3 + 1], norm[i * 3 + 2], 1]);
      if (c[3] <= 0) continue;
      const sx = (c[0] / c[3] * 0.5 + 0.5) * canvas.width;
      const sy = (1 - (c[1] / c[3] * 0.5 + 0.5)) * canvas.height;
      const dd = (sx - px) ** 2 + (sy - py) ** 2;
      if (dd < bestD) { bestD = dd; best = i; }
    }
    const tol = (16 * r) ** 2;   // ignore clicks far from any point
    return bestD <= tol ? best : -1;
  }

  window.addEventListener("resize", render);

  return {
    setPoints(pts) {
      norm = normalizePoints(pts); count = pts.length;
      gl.bindBuffer(gl.ARRAY_BUFFER, posBuf); gl.bufferData(gl.ARRAY_BUFFER, norm, gl.STATIC_DRAW);
      // (re)seed colors so a fresh point set is never drawn black before applyColor()
      if (!colorData || colorData.length !== count * 3) { colorData = new Float32Array(count * 3); for (let i = 0; i < count * 3; i++) colorData[i] = [0.58, 0.64, 0.74][i % 3]; }
      gl.bindBuffer(gl.ARRAY_BUFFER, colBuf); gl.bufferData(gl.ARRAY_BUFFER, colorData, gl.STATIC_DRAW);
      uploadFlags(); render();
    },
    setColors(colors) { colorData = colors; gl.bindBuffer(gl.ARRAY_BUFFER, colBuf); gl.bufferData(gl.ARRAY_BUFFER, colors, gl.STATIC_DRAW); render(); },
    setDensityField(values) { hasDens = !!(values && values.length === count); gl.bindBuffer(gl.ARRAY_BUFFER, densBuf); gl.bufferData(gl.ARRAY_BUFFER, values || new Float32Array(count), gl.STATIC_DRAW); render(); },
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
    setPath(arr) { pathArr = arr || []; uploadFlags(); render(); },
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
    // Unproject a CSS screen px to data coords, taking depth from the nearest state.
    unproject(sx, sy) {
      if (!norm || !count) return null;
      const { proj, view } = matrices(), mvp = multiply(proj, view);
      const W = canvas.clientWidth, H = canvas.clientHeight;
      let best = -1, bestD = Infinity, ndcZ = 0;
      for (let i = 0; i < count; i++) {
        const c = mulVec4(mvp, [norm[i * 3], norm[i * 3 + 1], norm[i * 3 + 2], 1]);
        if (c[3] <= 0) continue;
        const px = (c[0] / c[3] * 0.5 + 0.5) * W, py = (1 - (c[1] / c[3] * 0.5 + 0.5)) * H;
        const dd = (px - sx) ** 2 + (py - sy) ** 2;
        if (dd < bestD) { bestD = dd; best = i; ndcZ = c[2] / c[3]; }
      }
      if (best < 0) return null;
      const im = invert(mvp); if (!im) return null;
      const w = mulVec4(im, [sx / W * 2 - 1, (1 - sy / H) * 2 - 1, ndcZ, 1]);
      return normToData([w[0] / w[3], w[1] / w[3], w[2] / w[3]]);
    },
    setSelection(arr) { selArr = arr || []; uploadFlags(); render(); },
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
    orbitZoom(deltaY) { orbit.zoom(deltaY); },     // wheel-zoom forwarded from the overlay
    resetView() { orbit.reset(); },
    render,
    get selected() { return selected; },
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
