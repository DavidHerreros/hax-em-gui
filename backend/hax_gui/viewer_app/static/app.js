// Landscape explorer + volume viewer orchestration. Relative URLs (works behind
// the /viewer/<id>/ proxy). Zero dependencies.
import { createRenderer } from "./gl.js";
import { createVolume } from "./volume.js";
import { createScatter, clusterColor } from "./scatter.js";
import { createCloud } from "./cloud.js";
import { createHistogram } from "./histogram.js";

const $ = (id) => document.getElementById(id);
const status = (m, k = "") => { const e = $("status"); e.textContent = m; e.className = "status " + k; };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

let renderer, scatter, cloud = null, hist;
let L = null, umapData = null, points3 = [];
const clusterCache = {};
const state = { embedding: "pca", dims: 2, x: 0, y: 1, z: 2, colorBy: "none", k: 5, densBandwidth: 6, densIntensity: 0.22, current: -1, path: [], selection: [], sigma: 2.5, refMass: -1, render: "surface", vGamma: 1.4, vBright: 1.2, timeline: [], frame: 0, fps: 25, pointAlpha: 1, pathExpanded: false, smooth: 6 };
const frameDelay = () => 1000 / Math.max(2, Math.min(30, state.fps));
let decodeToken = 0;

// morph/series playback (a timeline of [a,b,t] frames; play/pause + scrubber)
let playing = false, recording = false;

// volume ray-marcher (created lazily on first switch to a Volume render mode)
let volRenderer = null, curVolMeta = null;

// lasso + layers
let lassoArmed = false, lassoing = false, lassoPts = [], lctx = null, lassoMode = "replace";
const layers = [];   // { name, color(hex), indices[], visible, kind? }
let layerSeq = 0;
// KMeans centroids: virtual indices >= L.n, with PCA coords + a per-centroid color
let indexColor = {};     // index -> hex (decoded maps of these indices are auto-colored)
let centroidPCA = {};    // index -> PCA vector (so centroids order along an axis for series)
let exportAvailable = false;

const is3d = () => state.dims === 3;
const activeLand = () => (is3d() && cloud ? cloud : scatter);

// Theme: the viewer runs in an iframe; the parent GUI passes its light/dark theme
// via ?theme= on load and a postMessage on toggle. We mirror it onto <html> (CSS
// vars do the chrome) and swap the 3D/volume canvas clear color to match.
const THEME_BG = { dark: "#040610", light: "#e7ecf3" };
function applyTheme(theme) {
  theme = theme === "light" ? "light" : "dark";
  document.documentElement.dataset.theme = theme;
  const bg = THEME_BG[theme];
  $("bg").value = bg;
  const c = hexToRgb(bg);
  renderer.setBackground(c);
  if (volRenderer) volRenderer.setBackground(c);
  if (cloud) cloud.setBackground(c);
}

async function boot() {
  try { renderer = createRenderer($("gl")); }
  catch (err) { status("WebGL2 unavailable: " + err.message, "err"); return; }
  scatter = createScatter($("scatter"), onSelect);
  hist = createHistogram($("hist"), onLevel);
  applyTheme(new URLSearchParams(location.search).get("theme"));
  window.addEventListener("message", (e) => { if (e.data && e.data.type === "hax-theme") applyTheme(e.data.theme); });

  try { L = await (await fetch("./api/landscape?k=5")).json(); }
  catch (err) { status("Could not load landscape: " + err, "err"); return; }
  clusterCache[5] = L.cluster.labels;
  $("title").textContent = L.title || "Conformational landscape";
  $("meta").textContent = `${L.n} states · latent dim ${L.dims}`;
  populateAxes();
  wireControls();
  setupLasso();
  setupResizers();
  fetchExportInfo();
  renderLayers();
  updatePoints();
  applyColor();
  status("Click a point to decode · shift-click to build a morph path · ▱ Lasso to select.");
  if (L.n > 0) { scatter.select(0); onSelect(0, {}); }
}

// Populate the X/Y/Z selectors for the active embedding (PCA / UMAP / latent),
// clamping the current picks to the number of axes that embedding exposes.
function populateAxes() {
  const n = axisCount();
  const opts = [];
  for (let i = 0; i < n; i++) opts.push(`<option value="${i}">${axisLabel(i)}</option>`);
  $("xaxis").innerHTML = opts.join(""); $("yaxis").innerHTML = opts.join(""); $("zaxis").innerHTML = opts.join("");
  if (state.x >= n) state.x = 0;
  if (state.y >= n) state.y = Math.min(1, n - 1);
  if (state.z >= n) state.z = Math.min(2, n - 1);
  $("xaxis").value = String(state.x); $("yaxis").value = String(state.y); $("zaxis").value = String(state.z);
}

function wireControls() {
  $("dims").addEventListener("change", (e) => {
    state.dims = +e.target.value; const is3 = is3d();
    $("scatter").style.display = is3 ? "none" : ""; $("scatter3d").style.display = is3 ? "" : "none";
    $("zwrap").style.display = is3 ? "" : "none";
    if (is3 && !cloud) { try { cloud = createCloud($("scatter3d"), onSelect); cloud.setBackground(hexToRgb($("bg").value)); } catch (err) { status("WebGL2 unavailable for 3D: " + err.message, "err"); state.dims = 2; $("dims").value = "2"; $("scatter").style.display = ""; $("scatter3d").style.display = "none"; return; } }
    refreshAxisEnabled();
    updatePoints(); applyColor(); syncSelection();   // re-apply color so 2D↔3D keeps density/cluster/layers
  });
  $("zaxis").addEventListener("change", (e) => { state.z = +e.target.value; updatePoints(); });
  $("embed").addEventListener("change", async (e) => {
    state.embedding = e.target.value;
    if (state.embedding === "umap" && !umapData) { status("Computing UMAP…", "busy"); try { umapData = (await (await fetch("./api/embedding?kind=umap")).json()).points; } catch (err) { status("UMAP failed: " + err, "err"); $("embed").value = "pca"; state.embedding = "pca"; return; } status(""); }
    populateAxes();        // each embedding has its own axis set (PCs / UMAP dims / latent dims)
    refreshAxisEnabled();
    updatePoints();
  });
  $("xaxis").addEventListener("change", (e) => { state.x = +e.target.value; updatePoints(); });
  $("yaxis").addEventListener("change", (e) => { state.y = +e.target.value; updatePoints(); });
  $("colorby").addEventListener("change", (e) => { state.colorBy = e.target.value; toggleColorWraps(); applyColor(); });
  $("lassoBtn").addEventListener("click", () => setLassoArmed(!lassoArmed));
  $("layersBtn").addEventListener("click", () => {
    const collapsed = $("layersRail").classList.toggle("collapsed");
    $("layersBtn").classList.toggle("active", !collapsed);
  });
  $("addLayer").addEventListener("click", addLayerFromSelection);
  $("clusterBtn").addEventListener("click", computeClusters);
  $("saveAll").addEventListener("click", saveAllLayers);
  $("layersBtn").classList.add("active");   // rail visible by default
  $("k").addEventListener("input", (e) => { state.k = +e.target.value; $("kval").textContent = state.k; if (state.colorBy === "cluster") applyColor(); });
  $("ptsalpha").addEventListener("input", (e) => { state.pointAlpha = +e.target.value; scatter.setPointAlpha(state.pointAlpha); if (cloud) cloud.setPointAlpha(state.pointAlpha); });
  $("pathToggle").addEventListener("click", () => { state.pathExpanded = !state.pathExpanded; updatePathLabel(); });
  $("denssigma").addEventListener("input", (e) => {
    state.densBandwidth = +e.target.value;
    if (state.colorBy !== "density") return;
    if (is3d() && cloud) applyColor(); else scatter.setDensityBandwidth(state.densBandwidth);
  });
  $("densint").addEventListener("input", (e) => { state.densIntensity = +e.target.value; if (cloud) cloud.setDensityIntensity(state.densIntensity); });
  $("light").addEventListener("change", (e) => renderer.setLighting(e.target.value));
  $("smooth").addEventListener("input", (e) => { state.smooth = +e.target.value; if (state.timeline.length && state.frame > 0) liveFrame(state.frame); else liveDecode(); });
  $("color").addEventListener("input", (e) => { const c = hexToRgb(e.target.value); renderer.setColor(c); if (volRenderer) volRenderer.setColor(c); });
  $("bg").addEventListener("input", (e) => { const c = hexToRgb(e.target.value); renderer.setBackground(c); if (volRenderer) volRenderer.setBackground(c); });
  $("reset").addEventListener("click", () => { renderer.resetView(); if (volRenderer) volRenderer.resetView(); });
  $("mrc").addEventListener("click", saveMrc);
  $("saveMap").addEventListener("click", () => savePng(state.render === "surface" ? renderer : volRenderer, "map"));
  $("saveLand").addEventListener("click", () => savePng(activeLand(), "landscape"));
  $("play").addEventListener("click", togglePlay);
  $("record").addEventListener("click", recordSeries);
  $("stop").addEventListener("click", stopPlayback);
  $("scrub").addEventListener("input", (e) => { setPlaying(false); if (!state.timeline.length) buildTimeline(); liveFrame(+e.target.value); });
  $("frames").addEventListener("change", () => { buildTimeline(); });
  $("speed").addEventListener("input", (e) => { state.fps = +e.target.value; });
  $("clearpath").addEventListener("click", clearPath);
  $("rendermode").addEventListener("change", (e) => setRenderMode(e.target.value));
  $("vgamma").addEventListener("input", (e) => { state.vGamma = +e.target.value; if (volRenderer) volRenderer.setGamma(state.vGamma); });
  $("vbright").addEventListener("input", (e) => { state.vBright = +e.target.value; if (volRenderer) volRenderer.setBrightness(state.vBright); });
}

function clearPath() {
  if (playing) setPlaying(false);
  state.path = []; state.frame = 0; syncSelection(); updatePathLabel();
}
function trajectory() {
  // [current, ...waypoints] with consecutive duplicates collapsed
  const raw = [state.current, ...state.path].filter((v) => v >= 0);
  return raw.filter((v, i) => i === 0 || v !== raw[i - 1]);
}
function updatePathLabel() {
  const traj = trajectory();
  const long = traj.length > 6;
  $("pathToggle").style.display = long ? "" : "none";
  $("pathToggle").textContent = state.pathExpanded ? "▾" : "▸";
  if (!traj.length) $("morphto").textContent = "—";
  else if (!long || state.pathExpanded) $("morphto").textContent = traj.map((i) => `#${i}`).join(" → ");
  else $("morphto").textContent = `${traj.length} states · #${traj[0]} → … → #${traj[traj.length - 1]}`;
  buildTimeline();        // cheap: just (re)computes the [a,b,t] frame list
  updateMorphButtons();
}
function updateMorphButtons() {
  const ready = state.timeline.length > 0;
  $("play").disabled = !ready; $("record").disabled = !ready;
  $("scrub").disabled = !ready;
}

const pcLabel = (i) => `PC${i + 1}${L.pca_var[i] != null ? ` (${(L.pca_var[i] * 100).toFixed(1)}%)` : ""}`;

// Each embedding exposes its own coordinate matrix + axis set, so X/Y/Z can be
// chosen the same way for PCA, UMAP and the raw latent space.
function activeCoords() {
  if (state.embedding === "umap") return umapData;
  if (state.embedding === "latent") return L.latents;
  return L.pca;
}
function axisCount() {
  if (state.embedding === "umap") return umapData ? umapData[0].length : 3;
  if (state.embedding === "latent") return L.dims;
  return L.pca_k;
}
function axisLabel(i) {
  if (state.embedding === "umap") return `UMAP${i + 1}`;
  if (state.embedding === "latent") return `Z${i + 1}`;
  return pcLabel(i);
}

// All embeddings now have selectable axes (PCA/UMAP/latent); keep them enabled.
function refreshAxisEnabled() {
  $("xaxis").disabled = false; $("yaxis").disabled = false; $("zaxis").disabled = false;
}

function updatePoints() {
  const c = activeCoords();
  if (!c) return;
  if (is3d() && cloud) {
    points3 = c.map((r) => [r[state.x], r[state.y], r[state.z] ?? 0]);
    $("axisinfo").textContent = `${axisLabel(state.x)} · ${axisLabel(state.y)} · ${axisLabel(state.z)}`;
    cloud.setPoints(points3);
    applyColor();
    return;
  }
  const pts = c.map((r) => [r[state.x], r[state.y]]);
  $("axisinfo").textContent = `${axisLabel(state.x)} · ${axisLabel(state.y)}`;
  scatter.setPoints(pts);
}

async function clusterLabels() {
  let labels = clusterCache[state.k];
  if (!labels) { try { labels = clusterCache[state.k] = (await (await fetch(`./api/cluster?k=${state.k}`)).json()).labels; } catch { labels = null; } }
  return labels;
}

function toggleColorWraps() {
  $("kwrap").style.display = state.colorBy === "cluster" ? "" : "none";
  $("denswrap").style.display = state.colorBy === "density" ? "" : "none";
}

async function applyColor() {
  const labels = state.colorBy === "cluster" ? await clusterLabels() : null;
  if (is3d() && cloud) {
    const dens = state.colorBy === "density";
    if (dens) { cloud.setDensityField(density3(points3, 0.04 + state.densBandwidth * 0.02)); cloud.setDensityIntensity(state.densIntensity); }
    cloud.showDensity(dens);
    cloud.setColors(computeColors(labels));   // neutral-gray points under the density glow
    return;
  }
  if (state.colorBy === "layers") { scatter.setDensity(false); scatter.setLabels(null); scatter.setPointColors(layerColorsCSS()); return; }
  scatter.setPointColors(null);
  if (state.colorBy === "density") { scatter.setLabels(null); scatter.setDensityBandwidth(state.densBandwidth); scatter.setDensity(true); return; }
  scatter.setDensity(false);
  scatter.setLabels(state.colorBy === "cluster" ? labels : null);
}

// Per-point RGB for the 3D cloud (cluster palette / local-density ramp / gray).
function computeColors(labels) {
  const n = L.n, out = new Float32Array(n * 3);
  if (state.colorBy === "cluster" && labels) {
    for (let i = 0; i < n; i++) { const c = hexToRgb(clusterColor(labels[i])); out[i * 3] = c[0]; out[i * 3 + 1] = c[1]; out[i * 3 + 2] = c[2]; }
  } else if (state.colorBy === "density") {
    // points stay neutral gray; the additive glow overlay shows the density (cf. 2D)
    for (let i = 0; i < n; i++) { out[i * 3] = 0.5; out[i * 3 + 1] = 0.55; out[i * 3 + 2] = 0.62; }
  } else if (state.colorBy === "layers") {
    for (let i = 0; i < n; i++) { out[i * 3] = 0.39; out[i * 3 + 1] = 0.45; out[i * 3 + 2] = 0.55; }
    for (const ly of layers) { if (!ly.visible) continue; const c = hexToRgb(ly.color); for (const i of ly.indices) { out[i * 3] = c[0]; out[i * 3 + 1] = c[1]; out[i * 3 + 2] = c[2]; } }
  } else {
    for (let i = 0; i < n; i++) { out[i * 3] = 0.58; out[i * 3 + 1] = 0.64; out[i * 3 + 2] = 0.74; }
  }
  return out;
}

// CSS colors for the 2D scatter in "layers" mode (dim gray + visible layer colors).
function layerColorsCSS() {
  const out = new Array(L.n).fill("rgba(100,116,139,0.4)");
  for (const ly of layers) { if (!ly.visible) continue; for (const i of ly.indices) out[i] = ly.color; }
  return out;
}

// O(N^2) neighbour count within `frac` of the bbox diagonal -> [0,1].
function density3(pts, frac = 0.12) {
  const n = pts.length, d = new Float32Array(n);
  const mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
  for (const p of pts) for (let k = 0; k < 3; k++) { const v = p[k] || 0; if (v < mn[k]) mn[k] = v; if (v > mx[k]) mx[k] = v; }
  let diag = 0; for (let k = 0; k < 3; k++) diag += (mx[k] - mn[k]) ** 2; diag = Math.sqrt(diag) || 1;
  const r2 = (diag * frac) ** 2;
  let max = 0;
  for (let i = 0; i < n; i++) {
    let c = 0;
    for (let j = 0; j < n; j++) { let dd = 0; for (let k = 0; k < 3; k++) { const a = (pts[i][k] || 0) - (pts[j][k] || 0); dd += a * a; } if (dd <= r2) c++; }
    d[i] = c; if (c > max) max = c;
  }
  if (max > 0) for (let i = 0; i < n; i++) d[i] /= max;
  return d;
}


// Keep both landscape views highlighting the same current state + path + selection.
function syncSelection() {
  scatter.select(state.current); scatter.setPath(state.path); scatter.setSelection(state.selection);
  if (cloud) { cloud.select(state.current); cloud.setPath(state.path); cloud.setSelection(state.selection); }
}

async function onSelect(i, opts) {
  if (playing) setPlaying(false);              // editing the path stops playback
  if (opts && opts.shift) {
    // append a waypoint to the morph path (skip if it repeats the last one)
    const last = state.path.length ? state.path[state.path.length - 1] : state.current;
    if (i !== last) state.path.push(i);
    syncSelection(); updatePathLabel();
    return;
  }
  state.current = i; syncSelection(); updatePathLabel(); $("sel").textContent = `state #${i}`;
  try {
    const h = await (await fetch(`./api/histogram?index=${i}`)).json();
    hist.setData(h, h.mean + state.sigma * h.std);
  } catch { /* keep last histogram */ }
  decode();
}

function onLevel(level) {
  state.sigma = hist.sigma;
  if (state.render !== "surface" && volRenderer && curVolMeta) { volRenderer.setThreshold(normThreshold()); return; }
  liveDecode();
}

// Coalescing single-flight runner: fires immediately, and if more updates arrive
// while a request is in flight, re-runs once with the latest values when it returns.
// Gives continuous (server-bounded) updates without piling up requests.
function makeLive(fn) {
  let busy = false, pending = false, args = [];
  async function run() { busy = true; do { pending = false; try { await fn(...args); } catch { /* ignore */ } } while (pending); busy = false; }
  return (...a) => { args = a; if (busy) pending = true; else run(); };
}
const liveDecode = makeLive(() => decode());
const liveFrame = makeLive((i) => renderFrame(i));

async function decode() {
  if (state.current < 0) return;
  const token = ++decodeToken;
  status("Decoding map…", "busy");
  try {
    const res = await fetch(`./api/decode?index=${state.current}&sigma=${state.sigma.toFixed(3)}&smooth=${state.smooth}`);
    if (!res.ok) throw new Error(await res.text());
    const m = parseMesh(await res.arrayBuffer());
    if (token !== decodeToken) return;
    state.refMass = m.mass;
    renderer.setMesh(m.positions, m.normals, m.indices);
    renderer.setColor(colorForIndex(state.current));   // centroids carry their cluster color
    if (state.render !== "surface") { loadVolume(state.current); return; }
    if (!m.indices.length) { renderer.clear(); status("No surface at this contour — lower the threshold.", "warn"); return; }
    status(`${(m.indices.length / 3).toLocaleString()} triangles · σ ${state.sigma.toFixed(1)} · level ${m.level.toFixed(3)}`);
  } catch (err) { if (token === decodeToken) status("Decode failed: " + err.message, "err"); }
}

// ---- volume ray-marching mode ----
function setRenderMode(mode) {
  state.render = mode; $("rendermode").value = mode;
  const surf = mode === "surface";
  $("gl").style.display = surf ? "" : "none";
  $("glvol").style.display = surf ? "none" : "";
  $("lightlbl").style.display = surf ? "" : "none";
  $("smoothlbl").style.display = surf ? "" : "none";
  $("volctrls").style.display = surf ? "none" : "";
  if (!surf && !volRenderer) {
    try { volRenderer = createVolume($("glvol")); volRenderer.setColor(hexToRgb($("color").value)); volRenderer.setBackground(hexToRgb($("bg").value)); volRenderer.setGamma(state.vGamma); volRenderer.setBrightness(state.vBright); }
    catch (err) { status("Volume rendering unavailable: " + err.message, "err"); setRenderMode("surface"); return; }
  }
  if (volRenderer) volRenderer.setMode(mode === "emission" ? "emission" : "mip");
  if (state.timeline.length && state.frame > 0) renderFrame(state.frame);
  else if (state.current >= 0) decode();
}

function normThreshold() {
  if (!curVolMeta) return 0.45;
  const span = curVolMeta.vmax - curVolMeta.vmin || 1;
  return (hist.level - curVolMeta.vmin) / span;
}

async function loadVolume(index) {
  if (!volRenderer || index < 0) return;
  const token = ++decodeToken;
  status("Loading volume…", "busy");
  try {
    const res = await fetch(`./api/grid?index=${index}&size=96`);
    if (!res.ok) throw new Error(await res.text());
    const g = parseGrid(await res.arrayBuffer());
    if (token !== decodeToken) return;
    curVolMeta = g;
    volRenderer.setVolume(g.nx, g.ny, g.nz, g.data);
    volRenderer.setThreshold(normThreshold());
    volRenderer.setColor(colorForIndex(index));
    status(`Volume ${g.nx}×${g.ny}×${g.nz} · ${state.render === "emission" ? "glow" : "MIP"}`);
  } catch (err) { if (token === decodeToken) status("Volume load failed: " + err.message, "err"); }
}

function parseGrid(buf) {
  const dv = new DataView(buf);
  const nx = dv.getInt32(0, true), ny = dv.getInt32(4, true), nz = dv.getInt32(8, true);
  const vmin = dv.getFloat32(12, true), vmax = dv.getFloat32(16, true), mean = dv.getFloat32(20, true), std = dv.getFloat32(24, true);
  const data = new Uint8Array(buf, 28, nx * ny * nz);
  return { nx, ny, nz, vmin, vmax, mean, std, data };
}

// ---- morph / volume series as a scrubbable timeline ----
// Build the ordered list of [a,b,t] frames for the current path (forward then back,
// so playback loops cleanly and the scrubber covers a full there-and-back). Cheap —
// no network — so it's safe to call on every path edit.
function buildTimeline() {
  const path = trajectory();
  if (path.length < 2) { state.timeline = []; state.frame = 0; $("scrub").max = "0"; $("scrub").value = "0"; $("framelbl").textContent = "—"; return false; }
  const frames = Math.max(2, Math.min(120, +$("frames").value || 36));
  const steps = [];
  for (let s = 0; s < path.length - 1; s++) { const a = path[s], b = path[s + 1]; for (let i = 0; i < frames; i++) steps.push([a, b, i / (frames - 1)]); }
  state.timeline = steps.concat(steps.slice(0, -1).reverse());
  state.frame = Math.min(state.frame, state.timeline.length - 1);
  $("scrub").max = String(state.timeline.length - 1);
  $("scrub").value = String(state.frame);
  $("framelbl").textContent = `${state.frame + 1}/${state.timeline.length}`;
  return true;
}

// Render one timeline frame in the current render mode (mass-constant for surfaces).
async function renderFrame(i) {
  if (!state.timeline.length) return;
  i = Math.max(0, Math.min(state.timeline.length - 1, i | 0));
  state.frame = i; $("scrub").value = String(i); $("framelbl").textContent = `${i + 1}/${state.timeline.length}`;
  $("stop").disabled = !(playing || i > 0);
  const [a, b, t] = state.timeline[i];
  const mass = state.refMass >= 0 ? state.refMass : null;
  const col = lerpColor(colorForIndex(a), colorForIndex(b), t);   // smooth cluster-color transition
  const tok = ++decodeToken;
  try {
    if (state.render === "surface") {
      const q = `index=${a}&index2=${b}&t=${t.toFixed(4)}&smooth=${state.smooth}` + (mass != null ? `&mass=${mass}` : `&sigma=${state.sigma.toFixed(3)}`);
      const m = parseMesh(await (await fetch(`./api/decode?${q}`)).arrayBuffer());
      if (tok !== decodeToken) return;
      if (m.indices.length) renderer.setMesh(m.positions, m.normals, m.indices);
      renderer.setColor(col);
    } else if (volRenderer) {
      const g = parseGrid(await (await fetch(`./api/grid?index=${a}&index2=${b}&t=${t.toFixed(4)}&size=96`)).arrayBuffer());
      if (tok !== decodeToken) return;
      curVolMeta = g; volRenderer.setVolume(g.nx, g.ny, g.nz, g.data); volRenderer.setThreshold(normThreshold());
      volRenderer.setColor(col);
    }
  } catch (err) { if (tok === decodeToken) status("Frame decode failed: " + err.message, "err"); }
}

function setPlaying(on) {
  playing = on;
  $("play").textContent = on ? "⏸ Pause" : "▶ Play";
  $("play").classList.toggle("active", on);
  $("stop").disabled = !on && state.frame === 0;
}

async function togglePlay() {
  if (playing) { setPlaying(false); return; }
  if (!state.timeline.length && !buildTimeline()) { status("Build a path first (shift-click states).", "warn"); return; }
  setPlaying(true);
  while (playing) {
    let next = state.frame + 1;
    if (next >= state.timeline.length) next = 0;     // loop
    await renderFrame(next);
    if (!playing) break;
    status(`Playing… frame ${state.frame + 1}/${state.timeline.length}`, "busy");
    await sleep(frameDelay());
  }
  if (!recording) status(`Paused at frame ${state.frame + 1}/${state.timeline.length}.`);
}

function stopPlayback() {
  setPlaying(false); recording = false;
  if (state.timeline.length) renderFrame(0);
  status("Stopped.");
}

async function recordSeries() {
  if (!state.timeline.length && !buildTimeline()) { status("Build a path first (shift-click states).", "warn"); return; }
  const active = state.render === "surface" ? renderer : volRenderer;
  if (!active || !active.captureStream) { status("Recording unavailable for this view.", "err"); return; }
  let rec = null; const chunks = [];
  try {
    const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp9") ? "video/webm;codecs=vp9" : "video/webm";
    rec = new MediaRecorder(active.captureStream(30), { mimeType: mime });
    rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
    rec.start();
  } catch (err) { status("Recording unavailable: " + err.message, "err"); return; }
  recording = true; setPlaying(true);
  await renderFrame(0);
  for (let i = 1; i < state.timeline.length && playing; i++) { await renderFrame(i); status(`Recording… ${i + 1}/${state.timeline.length}`, "busy"); await sleep(frameDelay()); }
  setPlaying(false); recording = false;
  rec.stop(); rec.onstop = () => downloadBlob(new Blob(chunks, { type: "video/webm" }), "morph.webm");
  status("Recording saved (morph.webm).");
}

// ---- resizable panels (drag the divider between panes / the layers-rail edge) ----
function setupResizers() {
  const split = $("paneDivider").parentElement;
  dragHandle($("paneDivider"), (e) => {
    const r = split.getBoundingClientRect();
    const w = Math.max(200, Math.min(r.width - 220, e.clientX - r.left));
    $("landPane").style.flex = `0 0 ${w}px`;
  });
  dragHandle($("railResize"), (e) => {
    const rail = $("layersRail");
    const w = Math.max(120, Math.min(440, e.clientX - rail.getBoundingClientRect().left));
    rail.style.width = `${w}px`;
  });
}
function dragHandle(el, onMove) {
  let active = false;
  el.addEventListener("pointerdown", (e) => { active = true; el.classList.add("dragging"); el.setPointerCapture(e.pointerId); e.preventDefault(); });
  el.addEventListener("pointermove", (e) => { if (!active) return; onMove(e); window.dispatchEvent(new Event("resize")); });
  el.addEventListener("pointerup", (e) => { active = false; el.classList.remove("dragging"); try { el.releasePointerCapture(e.pointerId); } catch {} window.dispatchEvent(new Event("resize")); });
}

// ---- lasso selection (app-owned overlay; works for the active 2D or 3D view) ----
function sizeLasso() {
  const lc = $("lassoCanvas"); const dpr = Math.min(window.devicePixelRatio || 1, 2);
  lc.width = Math.max(1, lc.clientWidth * dpr); lc.height = Math.max(1, lc.clientHeight * dpr);
  if (lctx) lctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
function setupLasso() {
  const lc = $("lassoCanvas"); lctx = lc.getContext("2d"); sizeLasso();
  window.addEventListener("resize", sizeLasso);
  lc.addEventListener("pointerdown", (e) => {
    if (!lassoArmed) return;
    lassoing = true;
    lassoMode = e.shiftKey ? "add" : (e.altKey || e.ctrlKey || e.metaKey) ? "remove" : "replace";
    lassoPts = [[e.offsetX, e.offsetY]]; lc.setPointerCapture(e.pointerId);
  });
  lc.addEventListener("pointermove", (e) => { if (!lassoing) return; lassoPts.push([e.offsetX, e.offsetY]); drawLasso(); });
  lc.addEventListener("pointerup", (e) => { if (!lassoing) return; lassoing = false; try { lc.releasePointerCapture(e.pointerId); } catch {} finishLasso(); });
}
const LASSO_RGB = { replace: "45,212,191", add: "74,222,128", remove: "248,113,113" };
function setLassoArmed(on) {
  lassoArmed = on;
  $("lassoCanvas").classList.toggle("armed", on);
  $("lassoBtn").classList.toggle("active", on);
  if (on) { sizeLasso(); status("Lasso: drag = replace · Shift = add · Alt/Ctrl = remove."); }
}
function drawLasso() {
  const lc = $("lassoCanvas"); lctx.clearRect(0, 0, lc.clientWidth, lc.clientHeight);
  if (lassoPts.length < 2) return;
  const rgb = LASSO_RGB[lassoMode] || LASSO_RGB.replace;
  lctx.beginPath(); lctx.moveTo(lassoPts[0][0], lassoPts[0][1]);
  for (let i = 1; i < lassoPts.length; i++) lctx.lineTo(lassoPts[i][0], lassoPts[i][1]);
  lctx.closePath();
  lctx.fillStyle = `rgba(${rgb},0.10)`; lctx.fill();
  lctx.strokeStyle = `rgba(${rgb},0.9)`; lctx.lineWidth = 1.5; lctx.setLineDash([5, 4]); lctx.stroke(); lctx.setLineDash([]);
}
function finishLasso() {
  const poly = lassoPts; lassoPts = [];
  const lc = $("lassoCanvas"); lctx.clearRect(0, 0, lc.clientWidth, lc.clientHeight);
  if (poly.length < 3) return;
  const proj = activeLand().projectedPoints();
  const enclosed = [];
  for (let i = 0; i < proj.length; i++) if (proj[i] && pointInPoly(proj[i], poly)) enclosed.push(i);
  let next;
  if (lassoMode === "add") { const s = new Set(state.selection); enclosed.forEach((i) => s.add(i)); next = [...s].sort((a, b) => a - b); }
  else if (lassoMode === "remove") { const s = new Set(state.selection); enclosed.forEach((i) => s.delete(i)); next = [...s]; }
  else next = enclosed;
  setSelectionState(next);
  const verb = lassoMode === "add" ? "added" : lassoMode === "remove" ? "removed" : "selected";
  status(`${enclosed.length} ${verb} · ${next.length} in selection — “+ selection” to make a layer.`, next.length ? "" : "warn");
}
function pointInPoly(pt, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i][0], yi = poly[i][1], xj = poly[j][0], yj = poly[j][1];
    if (((yi > pt[1]) !== (yj > pt[1])) && (pt[0] < ((xj - xi) * (pt[1] - yi)) / (yj - yi || 1e-9) + xi)) inside = !inside;
  }
  return inside;
}
function setSelectionState(idx) {
  state.selection = idx;
  scatter.setSelection(idx); if (cloud) cloud.setSelection(idx);
  $("addLayer").disabled = idx.length === 0;
}

// ---- layers (named index sets; can drive a volume series) ----
function addLayerFromSelection() {
  if (!state.selection.length) { status("Lasso-select points first.", "warn"); return; }
  layerSeq += 1;
  layers.push({ name: `Layer ${layers.length + 1}`, color: clusterColor(layerSeq - 1), indices: [...state.selection], visible: true });
  setSelectionState([]);                       // consume the selection
  $("colorby").value = "layers"; state.colorBy = "layers"; toggleColorWraps();
  renderLayers(); applyColor();
  status(`Created “${layers[layers.length - 1].name}” (${layers[layers.length - 1].indices.length} states).`);
}
function iconBtn(txt, title, fn) {
  const b = document.createElement("button"); b.className = "iconbtn"; b.textContent = txt; b.title = title;
  b.addEventListener("click", fn); return b;
}
function renderLayers() {
  const ul = $("layerList"); ul.innerHTML = "";
  $("noLayers").style.display = layers.length ? "none" : "";
  layers.forEach((ly) => {
    const li = document.createElement("li"); li.className = "layer-item" + (ly.visible ? "" : " hidden");
    const eye = iconBtn(ly.visible ? "◉" : "○", "Show / hide", () => { ly.visible = !ly.visible; renderLayers(); if (state.colorBy === "layers") applyColor(); });
    const sw = document.createElement("input"); sw.type = "color"; sw.className = "swatch"; sw.value = ly.color; sw.title = "Layer color";
    sw.addEventListener("input", (e) => { ly.color = e.target.value; if (state.colorBy === "layers") applyColor(); });
    const name = document.createElement("span"); name.className = "lname"; name.textContent = ly.name; name.title = "Click to rename";
    name.addEventListener("click", () => { const v = prompt("Layer name", ly.name); if (v) { ly.name = v.trim() || ly.name; renderLayers(); } });
    const cnt = document.createElement("span"); cnt.className = "lcount"; cnt.textContent = `${ly.indices.length}`;
    const series = iconBtn("▶", "Morph through this layer as a volume series", () => playLayerSeries(ly));
    const del = iconBtn("✕", "Delete layer", () => { layers.splice(layers.indexOf(ly), 1); renderLayers(); if (state.colorBy === "layers") applyColor(); });
    li.append(eye, sw, name, cnt, series);
    if (exportAvailable) li.append(iconBtn("⤓", "Save layer (metadata + average volume) to the project", () => saveLayer(ly)));
    li.append(del); ul.appendChild(li);
  });
}
const coordOf = (i) => (i >= L.n ? (centroidPCA[i] ? centroidPCA[i][state.x] : 0) : (activeCoords() || L.pca)[i][state.x]);
const orderAlongAxis = (idx) => [...idx].sort((a, b) => coordOf(a) - coordOf(b));

// ---- KMeans clustering -> one layer per cluster + a decodable centroid layer ----
async function computeClusters() {
  const k = Math.max(2, Math.min(20, +$("clusterK").value || 5));
  status(`Computing KMeans (k=${k})…`, "busy");
  let d;
  try { d = await (await fetch(`./api/kmeans?k=${k}`)).json(); }
  catch (err) { status("Clustering failed: " + err, "err"); return; }
  // drop previous auto-generated cluster/centroid layers (keep manual ones)
  for (let i = layers.length - 1; i >= 0; i--) if (layers[i].kind) layers.splice(i, 1);
  indexColor = {}; centroidPCA = {};
  const groups = Array.from({ length: d.k }, () => []);
  d.labels.forEach((lab, i) => groups[lab] && groups[lab].push(i));
  groups.forEach((idxs, j) => layers.push({ name: `Cluster ${j + 1}`, color: clusterColor(j), indices: idxs, visible: true, kind: "cluster" }));
  const cents = [];
  for (let j = 0; j < d.k; j++) { const idx = d.base + j; cents.push(idx); indexColor[idx] = clusterColor(j); centroidPCA[idx] = d.centroid_pca[j]; }
  layers.push({ name: "Centroids", color: "#ffffff", indices: cents, visible: true, kind: "centroid" });
  $("colorby").value = "layers"; state.colorBy = "layers"; toggleColorWraps();
  renderLayers(); applyColor();
  status(`KMeans k=${d.k}: ${d.k} cluster layers + a centroid layer (▶ to morph between clusters).`);
}

// Surface/volume color for a decoded state: cluster color for centroids, else the picker.
function colorForIndex(i) { return indexColor[i] ? hexToRgb(indexColor[i]) : hexToRgb($("color").value); }
function lerpColor(a, b, t) { return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]; }

// ---- export layers to disk (subset metadata + average-latent volume, per project) ----
async function fetchExportInfo() {
  try { const d = await (await fetch("./api/export_info")).json(); exportAvailable = !!d.available; }
  catch { exportAvailable = false; }
  $("saveRow").style.display = exportAvailable ? "" : "none";
}
async function saveLayer(ly) {
  if (!exportAvailable) { status("Export needs a project — open the viewer from a job.", "warn"); return; }
  const fmt = $("saveFmt").value;
  status(`Saving “${ly.name}”…`, "busy");
  try {
    const r = await fetch("./api/save_layer", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ name: ly.name, indices: ly.indices, format: fmt }) });
    if (!r.ok) throw new Error(await r.text());
    const d = await r.json();
    status(`Saved “${ly.name}”: ${d.files.join(", ")} (${d.n_particles} particles) → ${d.folder}`);
  } catch (err) { status("Save failed: " + err.message, "err"); }
}
async function saveAllLayers() {
  if (!layers.length) { status("No layers to save.", "warn"); return; }
  for (const ly of layers) await saveLayer(ly);
  status(`Saved ${layers.length} layer(s) to the project.`);
}

async function playLayerSeries(ly) {
  const ordered = orderAlongAxis(ly.indices);
  if (ordered.length < 2) { status("A series needs ≥2 states.", "warn"); return; }
  if (playing) setPlaying(false);
  state.current = ordered[0]; state.path = ordered.slice(1); state.frame = 0;
  syncSelection(); updatePathLabel(); $("sel").textContent = `state #${ordered[0]}`;
  try { const h = await (await fetch(`./api/histogram?index=${ordered[0]}`)).json(); hist.setData(h, h.mean + state.sigma * h.std); } catch { /* keep */ }
  await decode();                              // sets refMass for the constant-mass series
  togglePlay();
}

// ---- exports ----
async function saveMrc() {
  if (state.current < 0) return;
  try { const blob = await (await fetch(`./api/mrc?index=${state.current}`)).blob(); downloadBlob(blob, `state_${state.current}.mrc`); }
  catch (err) { status("MRC export failed: " + err.message, "err"); }
}
async function savePng(src, name) {
  const scale = +$("dpi").value || 2;
  const transparent = $("transparent").checked;
  const blob = await src.snapshot({ scale, transparent });
  if (blob) downloadBlob(blob, `${name}.png`);
}
function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a"); a.href = url; a.download = filename; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

function parseMesh(buf) {
  const dv = new DataView(buf);
  const nverts = dv.getInt32(0, true), nfaces = dv.getInt32(4, true);
  const level = dv.getFloat32(8, true), mass = dv.getFloat32(12, true);
  let off = 20;
  const positions = new Float32Array(buf, off, nverts * 3); off += nverts * 12;
  const normals = new Float32Array(buf, off, nverts * 3); off += nverts * 12;
  const indices = new Uint32Array(buf, off, nfaces * 3);
  return { positions, normals, indices, level, mass };
}
function hexToRgb(hex) { const n = parseInt(hex.slice(1), 16); return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255]; }

boot();
