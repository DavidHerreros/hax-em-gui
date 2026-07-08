// WebGL2 3D-texture volume ray-marcher (MIP + emission/absorption with a simple
// transfer function). Full-screen quad + ray-box intersection in a [-1,1] cube;
// orbit camera shared in style with gl.js. Hand-written, no dependencies.
import { perspective, lookAt, multiply, invert, mulVec4 } from "./mat4.js";
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
uniform mat4 uInvVP;
uniform vec3 uEye;
uniform vec3 uColor;
uniform vec3 uBg;
uniform float uThreshold;   // 0..1 normalized contour
uniform float uGamma;
uniform float uBright;
uniform int uMode;          // 0 = MIP, 1 = emission/absorption
uniform int uSteps;

vec3 toWorld(vec3 ndc) { vec4 p = uInvVP * vec4(ndc, 1.0); return p.xyz / p.w; }

void main() {
  vec3 ro = uEye;
  vec3 rd = normalize(toWorld(vec3(vUv * 2.0 - 1.0, 1.0)) - ro);
  vec3 inv = 1.0 / rd;
  vec3 t0 = (vec3(-1.0) - ro) * inv, t1 = (vec3(1.0) - ro) * inv;
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
      float s = texture(uVol, (ro + rd * t) * 0.5 + 0.5).r;
      m = max(m, s); t += dt;
    }
    float v = pow(clamp((m - uThreshold) / denom, 0.0, 1.0), uGamma) * uBright;
    outColor = vec4(mix(uBg, vec3(1.0), clamp(v, 0.0, 1.0)), 1.0);
  } else {                                   // front-to-back emission/absorption
    vec3 acc = vec3(0.0); float alpha = 0.0;
    for (int i = 0; i < 512; i++) {
      if (i >= steps) break;
      float s = texture(uVol, (ro + rd * t) * 0.5 + 0.5).r;
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
    uThreshold = u("uThreshold"), uGamma = u("uGamma"), uBright = u("uBright"), uMode = u("uMode"), uSteps = u("uSteps"), uVol = u("uVol");

  // full-screen quad
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const quad = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quad);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);

  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_3D, tex);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);

  let color = [0.784, 0.784, 0.784], bg = [0.016, 0.024, 0.06];
  let threshold = 0.45, gamma = 1.4, bright = 1.2, mode = 0, steps = 160;
  let hasTex = false;
  const orbit = createOrbit(canvas, () => render(), null, { dist: 3.4 });

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
    const mvp = multiply(multiply(perspective((45 * Math.PI) / 180, aspect, 0.01, 100), lookAt(eye, [0, 0, 0], [0, 1, 0])), M);
    const ivp = invert(mvp); if (!ivp) return;
    const Minv = invert(M); if (!Minv) return;
    const em = mulVec4(Minv, [eye[0], eye[1], eye[2], 1]);
    gl.useProgram(prog);
    gl.uniformMatrix4fv(uInvVP, false, ivp);
    gl.uniform3fv(uEye, [em[0], em[1], em[2]]);
    gl.uniform3fv(uColor, color); gl.uniform3fv(uBg, bg);
    gl.uniform1f(uThreshold, threshold); gl.uniform1f(uGamma, gamma); gl.uniform1f(uBright, bright);
    gl.uniform1i(uMode, mode); gl.uniform1i(uSteps, steps);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_3D, tex); gl.uniform1i(uVol, 0);
    gl.bindVertexArray(vao); gl.drawArrays(gl.TRIANGLES, 0, 3); gl.bindVertexArray(null);
  }
  function render() { resize(); drawScene(1); }

  function setVolume(nx, ny, nz, u8) {
    gl.bindTexture(gl.TEXTURE_3D, tex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.R8, nx, ny, nz, 0, gl.RED, gl.UNSIGNED_BYTE, u8);
    hasTex = true; render();
  }

  window.addEventListener("resize", render);

  return {
    setVolume, render,
    setMode(m) { mode = m === "emission" ? 1 : 0; render(); },
    setThreshold(v) { threshold = Math.max(0, Math.min(1, v)); render(); },
    setGamma(g) { gamma = Math.max(0.2, g); render(); },
    setBrightness(b) { bright = Math.max(0.1, b); render(); },
    setColor(rgb) { color = rgb; render(); },
    setBackground(rgb) { bg = rgb; render(); },
    resetView() { orbit.reset(); },
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
