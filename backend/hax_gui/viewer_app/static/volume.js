// WebGL2 3D-texture volume ray-marcher (MIP + emission/absorption with a simple
// transfer function). Full-screen quad + ray-box intersection in the model-space cube
// the grid occupies (uBoxLo..uBoxHi), which the backend picks to coincide exactly with
// the marching-cubes surface so the map does not shift when you switch render modes;
// orbit camera shared in style with gl.js. Hand-written, no dependencies.
import { projection, lookAt, multiply, invert, mulVec4 } from "./mat4.js";
import { createOrbit } from "./orbit.js";

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
uniform sampler3D uVol;
// Second volume + blend weight: a morph is a linear blend of two densities, and doing
// it here rather than by uploading a new texture per frame is what makes it run at
// screen rate instead of at the speed of the network. uMix 0 = uVol alone.
uniform sampler3D uVolB;
uniform float uMix;
uniform mat4 uInvVP;
uniform vec3 uEye;
uniform vec3 uColor;
uniform vec3 uBg;
uniform vec3 uBoxLo;        // model-space cube the grid occupies (matches the surface)
uniform vec3 uBoxHi;
uniform float uThreshold;   // 0..1 normalized contour
uniform float uGamma;
uniform float uBright;
uniform int uMode;          // 0 = MIP, 1 = emission/absorption
uniform int uSteps;

vec3 toWorld(vec3 ndc) { vec4 p = uInvVP * vec4(ndc, 1.0); return p.xyz / p.w; }

// The density at a texture coordinate, blended if a morph is running. Both volumes
// share a grid (the backend refuses to morph maps that do not), so one coordinate
// samples both.
float density(vec3 p) {
  float a = texture(uVol, p).r;
  return uMix > 0.0 ? mix(a, texture(uVolB, p).r, uMix) : a;
}

void main() {
  vec3 ro = uEye;
  vec3 rd = normalize(toWorld(vec3(vUv * 2.0 - 1.0, 1.0)) - ro);
  vec3 inv = 1.0 / rd;
  vec3 boxInv = 1.0 / (uBoxHi - uBoxLo);     // model -> [0,1]; box is in model/numpy-axis order
  vec3 t0 = (uBoxLo - ro) * inv, t1 = (uBoxHi - ro) * inv;
  vec3 tmin = min(t0, t1), tmax = max(t0, t1);
  float tn = max(max(tmin.x, tmin.y), max(tmin.z, 0.0));
  float tf = min(min(tmax.x, tmax.y), tmax.z);
  if (tf <= tn) { outColor = vec4(uBg, 1.0); return; }

  int steps = uSteps;
  float dt = (tf - tn) / float(steps);
  float t = tn;
  float denom = max(1.0 - uThreshold, 1e-3);

  if (uMode == 0) {                          // maximum-intensity projection
    float m = 0.0;
    for (int i = 0; i < 512; i++) {
      if (i >= steps) break;
      float s = density(((ro + rd * t - uBoxLo) * boxInv).zyx);
      m = max(m, s); t += dt;
    }
    float v = pow(clamp((m - uThreshold) / denom, 0.0, 1.0), uGamma) * uBright;
    outColor = vec4(mix(uBg, vec3(1.0), clamp(v, 0.0, 1.0)), 1.0);
  } else {                                   // front-to-back emission/absorption
    vec3 acc = vec3(0.0); float alpha = 0.0;
    for (int i = 0; i < 512; i++) {
      if (i >= steps) break;
      float s = density(((ro + rd * t - uBoxLo) * boxInv).zyx);
      float d = pow(clamp((s - uThreshold) / denom, 0.0, 1.0), uGamma) * uBright;
      float a = clamp(d * 0.12, 0.0, 1.0) * (1.0 - alpha);
      acc += a * uColor; alpha += a;
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
  const uInvVP = u("uInvVP"), uEye = u("uEye"), uColor = u("uColor"), uBg = u("uBg"),
    uBoxLo = u("uBoxLo"), uBoxHi = u("uBoxHi"),
    uThreshold = u("uThreshold"), uGamma = u("uGamma"), uBright = u("uBright"), uMode = u("uMode"), uSteps = u("uSteps"),
    uVol = u("uVol"), uVolB = u("uVolB"), uMix = u("uMix");

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
    return t;
  }
  const tex = makeVolTex();      // the map, or the "from" end of a morph
  const texB = makeVolTex();     // the "to" end; only sampled while mixing

  let color = [0.784, 0.784, 0.784], bg = [0.016, 0.024, 0.06];
  let threshold = 0.45, gamma = 1.4, bright = 1.2, mode = 0, steps = 160;
  let hasTex = false;
  let boxLo = [-1, -1, -1], boxHi = [1, 1, 1];   // model-space cube the grid fills; set per volume so it coincides with the surface mesh
  let grid = null;   // { nx, ny, nz, data } kept on the CPU so the ray-march can be repeated here
  let gridB = null;  // the morph's far end, same shape; null when not morphing
  let mix = 0;       // blend weight in [0,1]; 0 = `grid` alone

  // Reads the density the *shader* is drawing, so the centre of rotation and the
  // displayed bounds follow a morph instead of describing whichever map happens to be
  // at the near end. Returns a plain indexed read when nothing is blending, so the
  // common case pays nothing for the morph case.
  function sampler() {
    if (!gridB || mix <= 0) { const d = grid.data; return (i) => d[i]; }
    const a = grid.data, b = gridB.data, w = mix;
    return (i) => a[i] + (b[i] - a[i]) * w;
  }

  // Distance from the eye to the first voxel at or above the display threshold along
  // the view axis — the surface you see at the centre of the window. There is no
  // geometry to read back here, so re-walk the ray the shader walks, on the coarse
  // grid it is drawing. The camera turns this into a ChimeraX-style centre of rotation.
  function centreDepth() {
    if (!grid) return null;
    const { origin, dir } = orbit.axisRay();   // model space, where the volume fills [boxLo, boxHi]
    let tn = 0, tf = Infinity;                 // slab test against that box
    for (let a = 0; a < 3; a++) {
      if (Math.abs(dir[a]) < 1e-6) { if (origin[a] < boxLo[a] || origin[a] > boxHi[a]) return null; continue; }
      let t0 = (boxLo[a] - origin[a]) / dir[a], t1 = (boxHi[a] - origin[a]) / dir[a];
      if (t0 > t1) { const s = t0; t0 = t1; t1 = s; }
      tn = Math.max(tn, t0); tf = Math.min(tf, t1);
    }
    if (tf <= tn) return null;                 // the box is behind us, or we missed it

    const { nx, ny, nz } = grid;              // data is C-order (numpy axis0=nz slowest, axis2=nx fastest)
    const sample = sampler();
    const cut = threshold * 255;               // the shader compares the sample in [0,1] against uThreshold
    const steps3 = 2 * Math.max(nx, ny, nz);   // Nyquist-ish along the ray
    const texel = (v, lo, hi, n) => Math.min(n - 1, Math.max(0, Math.round(((v - lo) / (hi - lo)) * n - 0.5)));
    for (let i = 0; i <= steps3; i++) {
      const t = tn + ((tf - tn) * i) / steps3;
      // box/model axes are in numpy-axis order: model X -> axis0 (nz), Y -> axis1 (ny), Z -> axis2 (nx).
      const i0 = texel(origin[0] + dir[0] * t, boxLo[0], boxHi[0], nz);
      const i1 = texel(origin[1] + dir[1] * t, boxLo[1], boxHi[1], ny);
      const i2 = texel(origin[2] + dir[2] * t, boxLo[2], boxHi[2], nx);
      if (sample((i0 * ny + i1) * nx + i2) >= cut) return t;
    }
    return null;                               // the ray passed through empty space
  }

  // Model-space box enclosing the voxels at or above the display threshold — what the
  // ray-march actually paints, as opposed to the grid box, most of which is empty
  // padding. The camera uses it to decide whether the map is fully framed. One pass over
  // a ~96³ grid, recomputed only when the volume or the threshold changes.
  let boundsCache = null;
  function displayedBounds() {
    if (!grid) return null;
    if (boundsCache) return boundsCache;
    const { nx, ny, nz } = grid;
    const sample = sampler();
    const cut = threshold * 255;
    let a0 = nz, b0 = -1, a1 = ny, b1 = -1, a2 = nx, b2 = -1;
    for (let i0 = 0; i0 < nz; i0++) {
      for (let i1 = 0; i1 < ny; i1++) {
        const row = (i0 * ny + i1) * nx;
        for (let i2 = 0; i2 < nx; i2++) {
          if (sample(row + i2) < cut) continue;
          if (i0 < a0) a0 = i0; if (i0 > b0) b0 = i0;
          if (i1 < a1) a1 = i1; if (i1 > b1) b1 = i1;
          if (i2 < a2) a2 = i2; if (i2 > b2) b2 = i2;
        }
      }
    }
    if (b0 < 0) return null;   // nothing is above the threshold
    // Voxel index -> model coordinate, taking the outer face of the extreme voxels so
    // the box encloses them rather than passing through their centres. Model axes are in
    // numpy-axis order (X -> axis0 = nz, Y -> axis1 = ny, Z -> axis2 = nx), as in centreDepth.
    const at = (a, idx, n) => boxLo[a] + (idx / n) * (boxHi[a] - boxLo[a]);
    boundsCache = {
      lo: [at(0, a0, nz), at(1, a1, ny), at(2, a2, nx)],
      hi: [at(0, b0 + 1, nz), at(1, b1 + 1, ny), at(2, b2 + 1, nx)],
    };
    return boundsCache;
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
    gl.useProgram(prog);
    gl.uniformMatrix4fv(uInvVP, false, ivp);
    gl.uniform3fv(uEye, [em[0], em[1], em[2]]);
    gl.uniform3fv(uColor, color); gl.uniform3fv(uBg, bg);
    gl.uniform3fv(uBoxLo, boxLo); gl.uniform3fv(uBoxHi, boxHi);
    gl.uniform1f(uThreshold, threshold); gl.uniform1f(uGamma, gamma); gl.uniform1f(uBright, bright);
    gl.uniform1i(uMode, mode); gl.uniform1i(uSteps, steps);
    gl.uniform1f(uMix, gridB ? mix : 0);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_3D, tex); gl.uniform1i(uVol, 0);
    // Always bound, even when not morphing: sampling an incomplete texture unit is
    // undefined behaviour on some drivers even where the branch never reads it.
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_3D, gridB ? texB : tex); gl.uniform1i(uVolB, 1);
    gl.bindVertexArray(vao); gl.drawArrays(gl.TRIANGLES, 0, 3); gl.bindVertexArray(null);
  }
  function render() { resize(); drawScene(1); }

  function upload(target, nx, ny, nz, u8) {
    gl.bindTexture(gl.TEXTURE_3D, target);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.R8, nx, ny, nz, 0, gl.RED, gl.UNSIGNED_BYTE, u8);
  }

  function setVolume(nx, ny, nz, u8, lo, hi) {
    upload(tex, nx, ny, nz, u8);
    grid = { nx, ny, nz, data: u8 };
    if (lo && hi) { boxLo = lo.slice(); boxHi = hi.slice(); }   // the cube the surface mesh uses, so the two modes coincide
    hasTex = true; boundsCache = null; orbit.invalidatePivot(); render();
  }

  window.addEventListener("resize", render);

  return {
    setVolume, render,
    // The far end of a morph. Pass null to stop morphing and go back to one volume.
    setVolumeB(nx, ny, nz, u8) {
      if (!u8) { gridB = null; mix = 0; boundsCache = null; orbit.invalidatePivot(); render(); return; }
      upload(texB, nx, ny, nz, u8);
      gridB = { nx, ny, nz, data: u8 };
      boundsCache = null; orbit.invalidatePivot(); render();
    },
    // Blend weight, 0 = the near map, 1 = the far one. Called every animation frame, so
    // it must stay cheap: no pivot invalidation and no bounds rescan here — those are
    // gesture-time concerns, and re-deriving them 60 times a second would cost more
    // than the render.
    setMix(t) { mix = Math.max(0, Math.min(1, t)); render(); },
    setMode(m) { mode = m === "emission" ? 1 : 0; render(); },
    // A new contour is a new front surface — and a different set of voxels on screen.
    setThreshold(v) { threshold = Math.max(0, Math.min(1, v)); boundsCache = null; orbit.invalidatePivot(); render(); },
    setGamma(g) { gamma = Math.max(0.2, g); render(); },
    setBrightness(b) { bright = Math.max(0.1, b); render(); },
    setColor(rgb) { color = rgb; render(); },
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
