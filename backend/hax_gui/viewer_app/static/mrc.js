// Standalone map (.mrc) viewer controller. Reuses the same renderers as the
// conformational-landscape app — the marching-cubes surface (gl.js), the volume
// ray-marcher (volume.js) and the contour histogram (histogram.js) — plus an
// orthogonal-slice view (slices.js). No latents, no network, no landscape: it
// just inspects one volume the backend loaded from disk.
import { createRenderer } from "./gl.js";
import { createVolume } from "./volume.js";
import { createHistogram } from "./histogram.js";
import { createSlices } from "./slices.js";

const $ = (id) => document.getElementById(id);
const THEME_BG = { dark: "#040610", light: "#e7ecf3" };

const renderer = createRenderer($("gl"));       // surface mesh
const volRenderer = createVolume($("glvol"));   // MIP / glow ray-march
const slices = createSlices($("slices"));       // orthogonal cuts
const hist = createHistogram($("hist"), onLevel);

const state = { render: "surface", sigma: 2.5, smooth: 6 };
let info = null;        // /api/info
let gridVol = null;     // parsed /api/grid for the ray-marcher (coarse)
let gridSlice = null;   // parsed /api/grid for slices (finer)

function status(msg, cls = "") {
  const el = $("status"); el.textContent = msg; el.title = msg;
  el.className = "status" + (cls ? " " + cls : "");
}

// ---- theme (mirrors app.js: ?theme= on load + postMessage on toggle) ----
function applyTheme(theme) {
  theme = theme === "light" ? "light" : "dark";
  document.documentElement.dataset.theme = theme;
  const bg = THEME_BG[theme];
  $("bg").value = bg;
  const rgb = hexToRgb(bg);
  renderer.setBackground(rgb); volRenderer.setBackground(rgb); slices.setBackground(bg);
}
applyTheme(new URLSearchParams(location.search).get("theme"));
window.addEventListener("message", (e) => { if (e.data && e.data.type === "hax-theme") applyTheme(e.data.theme); });

// ---- data loaders --------------------------------------------------------
async function loadInfo() {
  info = await (await fetch("./api/info")).json();
  const [nz, ny, nx] = info.shape;
  $("title").textContent = info.title || "Map";
  $("meta").textContent = `${nx}×${ny}×${nz}`;
  $("sel").textContent = info.filename ? info.filename.split("/").pop() : "";
  if (info.filename) document.title = "Hax — " + info.filename.split("/").pop();
}

async function loadHistogram() {
  const h = await (await fetch("./api/histogram")).json();
  state.sigma = 2.5;
  hist.setData(h, h.mean + state.sigma * h.std);
}

function parseMesh(buf) {
  const dv = new DataView(buf);
  const nverts = dv.getInt32(0, true), nfaces = dv.getInt32(4, true);
  let off = 20;
  const positions = new Float32Array(buf, off, nverts * 3); off += nverts * 12;
  const normals = new Float32Array(buf, off, nverts * 3); off += nverts * 12;
  const indices = new Uint32Array(buf, off, nfaces * 3);
  return { positions, normals, indices };
}

function parseGrid(buf) {
  const dv = new DataView(buf);
  const nx = dv.getInt32(0, true), ny = dv.getInt32(4, true), nz = dv.getInt32(8, true);
  const vmin = dv.getFloat32(12, true), vmax = dv.getFloat32(16, true), mean = dv.getFloat32(20, true), std = dv.getFloat32(24, true);
  const boxLo = [dv.getFloat32(28, true), dv.getFloat32(32, true), dv.getFloat32(36, true)];
  const boxHi = [dv.getFloat32(40, true), dv.getFloat32(44, true), dv.getFloat32(48, true)];
  const data = new Uint8Array(buf, 52, nx * ny * nz);
  return { nx, ny, nz, vmin, vmax, mean, std, boxLo, boxHi, data };
}

async function fetchGrid(size) {
  const buf = await (await fetch(`./api/grid?size=${size}`)).arrayBuffer();
  return parseGrid(buf);
}

// Run `fn` at most once at a time, re-running it after it settles if it was asked
// again meanwhile. Gives continuous (server-bounded) updates without piling up
// requests — same helper the landscape app uses.
function makeLive(fn) {
  let busy = false, pending = false, args = [];
  async function run() { busy = true; do { pending = false; try { await fn(...args); } catch { /* fn reports */ } } while (pending); busy = false; }
  return (...a) => { args = a; if (busy) pending = true; else run(); };
}

// One contour costs a full marching-cubes pass over the whole volume plus a multi-MB mesh
// download, and the histogram fires this on every pointermove. Left unthrottled a single
// drag queues one such request per pixel crossed; each reads the *current* level, so
// coalescing converges on the level you released at.
//
// Even coalesced, a full pass is ~750 ms and ~21 MB at a 320³ box — steady, but not live.
// So while the pointer is down we ask for a draft: a strided grid, no smoothing (~50 ms,
// ~3.8 MB), and take the full mesh once on release. The server scales the draft by its
// stride, so it sits in the same [-1,1] cube and refines in place rather than jumping.
const DRAFT_GRID = 160;

async function surfaceOnce(draft) {
  status("Contouring…", "busy");
  try {
    const q = `sigma=${state.sigma.toFixed(3)}`
            + (draft ? `&smooth=0&size=${DRAFT_GRID}` : `&smooth=${state.smooth}`);
    const buf = await (await fetch(`./api/decode?${q}`)).arrayBuffer();
    if (state.render !== "surface") return;   // mode switched while we were waiting
    const m = parseMesh(buf);
    renderer.setMesh(m.positions, m.normals, m.indices);
    if (!m.indices.length) { status("Empty surface at this contour", "warn"); return; }
    status(draft ? `Surface · σ ${state.sigma.toFixed(2)} · draft`
                 : `Surface · σ ${state.sigma.toFixed(2)}`);
  } catch (err) { status("Contour failed: " + err.message, "err"); }
}
const loadSurface = makeLive(surfaceOnce);

async function loadVolume() {
  status("Loading volume…", "busy");
  try {
    if (!gridVol) gridVol = await fetchGrid(96);
    volRenderer.setVolume(gridVol.nx, gridVol.ny, gridVol.nz, gridVol.data, gridVol.boxLo, gridVol.boxHi);
    volRenderer.setThreshold(normThreshold());
    status(`Volume ${gridVol.nx}×${gridVol.ny}×${gridVol.nz} · ${state.render === "emission" ? "glow" : "MIP"}`);
  } catch (err) { status("Volume load failed: " + err.message, "err"); }
}

async function loadSlices() {
  status("Loading slices…", "busy");
  try {
    if (!gridSlice) gridSlice = await fetchGrid(192);
    slices.setVolume(gridSlice);
    status(`Slices ${gridSlice.nx}×${gridSlice.ny}×${gridSlice.nz}`);
  } catch (err) { status("Slice load failed: " + err.message, "err"); }
}

// contour level (absolute) -> ray-marcher threshold in [0,1] over the grid range
function normThreshold() {
  if (!gridVol) return 0.45;
  const lvl = hist.level;
  return (lvl - gridVol.vmin) / ((gridVol.vmax - gridVol.vmin) || 1);
}

function onLevel(level, dragging) {
  state.sigma = info ? (level - info.mean) / (info.std || 1) : hist.sigma;
  if (state.render === "surface") loadSurface(!!dragging);
  else if (state.render === "mip" || state.render === "emission") volRenderer.setThreshold(normThreshold());
}

// ---- render-mode switching ----------------------------------------------
function show(el, on) { el.style.display = on ? "" : "none"; }

// Surface and volume are drawn on different canvases, each holding its own camera, so a
// mode switch would otherwise drop you at that renderer's default view. Carry the camera
// across instead. `camera` outlives the cameraless slice view; "Reset view" drops it so a
// reset is not undone by the next switch.
const camera3d = (mode) => (mode === "surface" ? renderer : mode === "mip" || mode === "emission" ? volRenderer : null);
let camera = null;

function setMode(mode) {
  const from = camera3d(state.render);
  if (from) camera = from.getCamera();
  state.render = mode;
  const isSurface = mode === "surface";
  const isVolume = mode === "mip" || mode === "emission";
  const isSlices = mode === "slices";

  show($("gl"), isSurface);
  show($("glvol"), isVolume);
  $("slices").style.display = isSlices ? "grid" : "none";

  const to = camera3d(mode);
  if (to && to !== from && camera) to.setCamera(camera);

  show($("lightlbl"), isSurface);
  show($("smoothlbl"), isSurface);
  show($("volctrls"), isVolume);
  show($("slicectrls"), isSlices);
  show($("colorlbl"), !isSlices);
  show($("histStrip"), !isSlices);   // contour is meaningless for raw slices

  if (isSurface) { renderer.render(); loadSurface(); }
  else if (isVolume) { volRenderer.setMode(mode === "emission" ? "emission" : "mip"); loadVolume(); }
  else { loadSlices(); }
}

// ---- controls ------------------------------------------------------------
$("rendermode").addEventListener("change", (e) => setMode(e.target.value));
$("light").addEventListener("change", (e) => renderer.setLighting(e.target.value));
$("smooth").addEventListener("input", (e) => { state.smooth = +e.target.value; if (state.render === "surface") loadSurface(); });
$("vgamma").addEventListener("input", (e) => volRenderer.setGamma(+e.target.value));
$("vbright").addEventListener("input", (e) => volRenderer.setBrightness(+e.target.value));
$("slicemap").addEventListener("change", (e) => slices.setColormap(e.target.value));
$("slicelo").addEventListener("input", () => slices.setContrast(+$("slicelo").value, +$("slicehi").value));
$("slicehi").addEventListener("input", () => slices.setContrast(+$("slicelo").value, +$("slicehi").value));
$("slicecross").addEventListener("change", (e) => slices.setCrosshair(e.target.checked));
$("sliceReset").addEventListener("click", () => slices.resetZoom());
$("color").addEventListener("input", (e) => { const rgb = hexToRgb(e.target.value); renderer.setColor(rgb); volRenderer.setColor(rgb); });
$("bg").addEventListener("input", (e) => { const rgb = hexToRgb(e.target.value); renderer.setBackground(rgb); volRenderer.setBackground(rgb); slices.setBackground(e.target.value); });
$("reset").addEventListener("click", () => { camera = null; renderer.resetView(); volRenderer.resetView(); slices.resetZoom(); });
$("mrc").addEventListener("click", () => { window.location.href = "./api/mrc"; });
$("saveMap").addEventListener("click", saveMap);

async function saveMap() {
  const scale = +$("dpi").value, transparent = $("transparent").checked;
  const r = state.render === "surface" ? renderer
    : state.render === "slices" ? slices : volRenderer;
  const blob = await r.snapshot({ scale, transparent });
  if (!blob) return;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = (info && info.filename ? info.filename.split("/").pop().replace(/\.[^.]+$/, "") : "map") + `_${state.render}.png`;
  a.click();
  URL.revokeObjectURL(a.href);
}

function hexToRgb(hex) { const n = parseInt(hex.slice(1), 16); return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255]; }

// ---- boot ----------------------------------------------------------------
(async function boot() {
  try {
    await loadInfo();
    await loadHistogram();
    setMode("surface");
  } catch (err) {
    status("Failed to load map: " + err.message, "err");
  }
})();
