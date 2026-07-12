// Landscape explorer + volume viewer orchestration. Relative URLs (works behind
// the /viewer/<id>/ proxy). Zero dependencies.
import { createRenderer } from "./gl.js";
import { createVolume } from "./volume.js";
import { createScatter, clusterColor } from "./scatter.js";
import { createCloud } from "./cloud.js";
import { createHistogram } from "./histogram.js";

// Density colormaps (RGB 0–255 control stops, low→high). Default inferno, to
// match the desktop annotate_space landscape (napari "inferno" over a dark
// canvas). Max 6 stops — the 3D glow shader's ramp array is sized 6.
const COLORMAPS = {
  viridis: [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]],
  magma: [[0, 0, 4], [81, 18, 124], [183, 55, 121], [252, 137, 97], [252, 253, 191]],
  plasma: [[13, 8, 135], [126, 3, 168], [204, 71, 120], [248, 149, 64], [240, 249, 33]],
  inferno: [[0, 0, 4], [50, 10, 94], [122, 28, 109], [188, 55, 84], [245, 135, 30], [252, 255, 164]],
  turbo: [[48, 18, 59], [33, 144, 141], [124, 211, 67], [253, 190, 61], [165, 14, 1]],
  cividis: [[0, 32, 76], [44, 70, 107], [124, 123, 120], [187, 175, 113], [255, 234, 70]],
  gray: [[20, 22, 30], [110, 114, 124], [180, 184, 192], [235, 237, 242]],
  teal: [[15, 23, 42], [30, 64, 120], [13, 148, 136], [250, 204, 21]],
};

const $ = (id) => document.getElementById(id);
const status = (m, k = "") => { const e = $("status"); e.textContent = m; e.className = "status " + k; };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

let renderer, scatter, cloud = null, hist;
let L = null, umapData = null, points3 = [];
const clusterCache = {};
const state = { embedding: "pca", dims: 2, x: 0, y: 1, z: 2, colorBy: "none", k: 5, densBandwidth: 6, densIntensity: 0.3, densMap: "inferno", current: -1, path: [], selection: [], sigma: 2.5, refMass: -1, render: "surface", vGamma: 1.4, vBright: 1.2, timeline: [], frame: 0, fps: 60, pointAlpha: 1, pathExpanded: false, smooth: 6 };
// `state.fps` is the *motion* rate: timeline frames advanced per second. We render
// at most DISPLAY_FPS_CAP frames/sec; once the motion rate exceeds that we keep the
// render rate at the cap and advance several frames per tick instead. That lets a
// long (many-frame) morph traverse fast — the protein appears to move more per
// second — without an absurd refresh rate. Below the cap, step == 1 (every frame
// shown: maximally smooth, slower traversal).
const DISPLAY_FPS_CAP = 30;
function playCadence() {
  const fps = Math.max(2, state.fps || 2);
  const displayFps = Math.min(DISPLAY_FPS_CAP, fps);
  return { delay: 1000 / displayFps, step: Math.max(1, Math.round(fps / displayFps)) };
}
let decodeToken = 0;

// morph/series playback (a timeline of [a,b,t] frames; play/pause + scrubber)
let playing = false, recording = false;
// When a *drawn curve* drives the morph, its snap to the nearest states is an
// internal implementation detail — we keep those "closest" picks out of the
// scene (only the smooth curve overlay shows). Display-only: state.path still
// holds the snapped indices that the decode walks.
let hidePathPicks = false;

// volume ray-marcher (created lazily on first switch to a Volume render mode)
let volRenderer = null, curVolMeta = null;

// lasso + layers
let lassoArmed = false, lassoing = false, lassoPts = [], lctx = null, lassoMode = "replace";
// freehand path tool (shares the lasso overlay): drag to trace a path across the
// points, or grab an existing waypoint to drag it onto another state. Works on the
// active 2D or 3D landscape via projectedPoints().
let pathDrawArmed = false, drawing = false, drawPts = [], draggingWp = -1, draggingCtrl = -1;
let dragProj = null;   // screen projection of every state, held for the duration of a waypoint drag
const DRAG_R = 12;   // px: grab radius for a waypoint / control handle
// A free-spline path layer keeps editable control points in data coords (not snapped
// to states); it is only snapped to the nearest states silently at morph time.
let editCurve = null;   // the curve layer currently being edited (control-point handles shown)
// orbit-while-editing (3D): an empty-space drag rotates the camera instead of editing
let orbiting = false, orbMoved = false, orbX = 0, orbY = 0, orbDownX = 0, orbDownY = 0;
const layers = [];   // { name, color(hex), indices[], visible, kind? }  (kind 'path' = ordered morph path)
let layerSeq = 0;
let editingPathLayer = null;   // the path layer currently being edited (shift-click add/remove)
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
// The density landscape always renders on a near-black backdrop (like the
// desktop app's napari canvas) so the inferno ramp reads correctly — even when
// the surrounding GUI is in light mode. Other color modes follow the theme.
const DENSITY_BG = "#050409";
function applyTheme(theme) {
  theme = theme === "light" ? "light" : "dark";
  document.documentElement.dataset.theme = theme;
  const bg = THEME_BG[theme];
  $("bg").value = bg;
  const c = hexToRgb(bg);
  renderer.setBackground(c);
  if (volRenderer) volRenderer.setBackground(c);
  applyLandscapeBg();   // keeps the dark density backdrop across theme toggles
  // Selection rings / path use a theme-aware contrast color computed at draw time,
  // so the landscape must repaint when the theme toggles.
  if (scatter) scatter.draw();
  if (cloud) cloud.render();
}

// The latent landscape (2D scatter + 3D cloud) goes dark in density mode, else
// follows the page theme. The molecular map view (renderer/volRenderer) is
// independent and always follows the theme.
function applyLandscapeBg() {
  const dens = state.colorBy === "density";
  if (scatter) scatter.setBackground(dens ? DENSITY_BG : null);
  if (cloud) cloud.setBackground(hexToRgb(dens ? DENSITY_BG : $("bg").value));
}

async function boot() {
  try { renderer = createRenderer($("gl")); }
  catch (err) { status("WebGL2 unavailable: " + err.message, "err"); return; }
  scatter = createScatter($("scatter"), onSelect);
  hist = createHistogram($("hist"), onLevel);
  applyColormap();
  applyTheme(new URLSearchParams(location.search).get("theme"));
  window.addEventListener("message", (e) => { if (e.data && e.data.type === "hax-theme") applyTheme(e.data.theme); });

  try {
    L = await (await fetch("./api/landscape")).json();   // metadata only; arrays follow as binary
    L.pca = await fetchMatrix("./api/pca.bin");
    L.latents = null;   // fetched on demand — only the "latent" embedding view needs them
    clusterCache[5] = await fetchLabels("./api/cluster.bin?k=5");
  } catch (err) { status("Could not load landscape: " + err, "err"); return; }
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
    if (is3 && !cloud) { try { cloud = createCloud($("scatter3d"), onSelect); cloud.setBackground(hexToRgb($("bg").value)); cloud.setColormap(COLORMAPS[state.densMap] || COLORMAPS.viridis); } catch (err) { status("WebGL2 unavailable for 3D: " + err.message, "err"); state.dims = 2; $("dims").value = "2"; $("scatter").style.display = ""; $("scatter3d").style.display = "none"; return; } }
    refreshAxisEnabled();
    updatePoints(); applyColor(); syncSelection();   // re-apply color so 2D↔3D keeps density/cluster/layers
  });
  $("zaxis").addEventListener("change", (e) => { state.z = +e.target.value; updatePoints(); });
  $("embed").addEventListener("change", async (e) => {
    state.embedding = e.target.value;
    if (state.embedding === "umap" && !umapData) { status("Computing UMAP…", "busy"); try { umapData = await fetchMatrix("./api/umap.bin"); } catch (err) { status("UMAP failed: " + err, "err"); $("embed").value = "pca"; state.embedding = "pca"; return; } status(""); }
    // The raw latent coords are big and only this view uses them, so they are not part of
    // the initial load — fetch them the first time someone actually asks for them.
    if (state.embedding === "latent" && !L.latents) { status("Loading latent coordinates…", "busy"); try { L.latents = await fetchMatrix("./api/latents.bin"); } catch (err) { status("Could not load latents: " + err, "err"); $("embed").value = "pca"; state.embedding = "pca"; return; } status(""); }
    populateAxes();        // each embedding has its own axis set (PCs / UMAP dims / latent dims)
    refreshAxisEnabled();
    updatePoints();
  });
  $("xaxis").addEventListener("change", (e) => { state.x = +e.target.value; updatePoints(); });
  $("yaxis").addEventListener("change", (e) => { state.y = +e.target.value; updatePoints(); });
  $("colorby").addEventListener("change", (e) => { state.colorBy = e.target.value; toggleColorWraps(); applyColor(); });
  $("lassoBtn").addEventListener("click", () => setLassoArmed(!lassoArmed));
  $("drawBtn").addEventListener("click", () => setPathDrawArmed(!pathDrawArmed));
  $("layersBtn").addEventListener("click", () => {
    const collapsed = $("layersRail").classList.toggle("collapsed");
    $("layersBtn").classList.toggle("active", !collapsed);
  });
  $("addLayer").addEventListener("click", addLayerFromSelection);
  $("addPath").addEventListener("click", makePathLayer);
  $("clusterBtn").addEventListener("click", () => computeClusters());
  $("clusterSeedBtn").addEventListener("click", clusterFromPicks);
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
  $("densmap").addEventListener("change", (e) => { state.densMap = e.target.value; applyColormap(); if (state.colorBy === "density") applyColor(); });
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
  $("speed").addEventListener("input", (e) => { state.fps = +e.target.value; if (playing) backgroundFill(); });
  $("clearpath").addEventListener("click", clearPath);
  $("rendermode").addEventListener("change", (e) => setRenderMode(e.target.value));
  $("vgamma").addEventListener("input", (e) => { state.vGamma = +e.target.value; if (volRenderer) volRenderer.setGamma(state.vGamma); });
  $("vbright").addEventListener("input", (e) => { state.vBright = +e.target.value; if (volRenderer) volRenderer.setBrightness(state.vBright); });
}

// Save the current morph path ([current, ...path]) as an editable path layer.
function makePathLayer() {
  const traj = [state.current, ...state.path].filter((v) => v >= 0);
  if (traj.length < 2) { status("Build a morph path first (shift-click states), then “+ path”.", "warn"); return; }
  layerSeq += 1;
  const ly = { name: `Path ${layers.length + 1}`, color: clusterColor(layerSeq - 1), indices: [...traj], visible: true, kind: "path" };
  layers.push(ly);
  renderLayers();
  status(`Saved “${ly.name}” (${traj.length} waypoints). Press ✎ on it to edit, ▶ to morph along it.`);
}
// Load a path layer into the live morph path for editing (toggle off to finish).
// Arms the freehand draw tool so the path can be retraced / its waypoints dragged.
function editPathLayer(ly) {
  if (editingPathLayer === ly) { editingPathLayer = null; setPathDrawArmed(false); renderLayers(); syncSelection(); status(`Finished editing “${ly.name}”.`); return; }
  editCurve = null; editingPathLayer = ly;
  const idx = ly.indices.filter((v) => v >= 0);
  state.path = idx.slice(1);
  renderLayers();
  setPathDrawArmed(true);
  status(`Editing “${ly.name}”: draw to retrace · drag a waypoint to move it · shift-click a state to add/remove.`, "busy");
  onSelect(idx[0]);   // make the first waypoint current + decode it
}

function clearPath() {
  if (editingPathLayer) { editingPathLayer = null; setPathDrawArmed(false); }
  if (playing) setPlaying(false);
  state.path = []; state.frame = 0; syncSelection(); updatePathLabel();
}

// The live morph trajectory as an ordered index list ([current, ...waypoints]).
const currentTrajectory = () => [state.current, ...state.path].filter((v) => v >= 0);

// Commit a new ordered trajectory to the live path (and the edited layer, if any).
// decodeCurrent re-decodes the head state when it changed (e.g. moving waypoint 0).
function applyTrajectory(traj, decodeCurrent) {
  traj = traj.filter((v) => v >= 0).filter((v, i, a) => i === 0 || v !== a[i - 1]);
  const prevCur = state.current;
  state.path = traj.slice(1);
  if (decodeCurrent && traj.length && traj[0] !== prevCur) {
    onSelect(traj[0], {});   // sets current, syncs the edited layer + decodes
    return;
  }
  state.current = traj.length ? traj[0] : -1;
  if (editingPathLayer) { editingPathLayer.indices = [...traj]; renderLayers(); if (state.colorBy === "layers") applyColor(); }
  syncSelection(); updatePathLabel();
  if (state.current >= 0) $("sel").textContent = `state #${state.current}`;
}

// ───────── free-spline paths (drawn curves, snapped to states only at morph time) ─────────
// A curve layer: { kind:"curve", ctrl:[[x,y(,z)]...] data coords, embed, axes, dims, color, name, visible }
const embedCoords = (embed) => (embed === "umap" ? umapData : embed === "latent" ? L.latents : L.pca);
const curveDims = () => (is3d() ? 3 : 2);
const viewAxes = () => (is3d() ? [state.x, state.y, state.z] : [state.x, state.y]);
const axesOf = (vec) => viewAxes().map((a) => vec[a]);   // full embedding vector -> current display axes
// A curve lives in its embedding; control points carry full embedding vectors, so it
// renders in any axis/2D/3D view of that embedding (we just read the displayed axes).
function curveMatchesView(ly) { return ly.embed === state.embedding; }
// ── nearest state, indexed ───────────────────────────────────────────────────
// Both curve tools need "which state lies closest to this point in embedding space", and
// both used to answer it by scanning every row: screenToVec on every mousemove while a
// handle is being dragged, curveToIndices ~550 times the moment a curve is played. At a
// million states those scans are what froze the tab.
//
// A uniform grid over the first few dimensions turns each query into a local search, and
// it stays EXACT rather than approximate: distance measured in a subset of the dimensions
// is a lower bound on the full distance, so once we hold a candidate closer than the
// nearest unscanned ring could possibly be, nothing further out can beat it.
//
// Convention: `dims` lists which coordinate indices to compare, and a query vector `q` is
// parallel to `dims` (q[j] pairs with dims[j]). That suits both callers — one searches on
// the two or three displayed axes, the other on every dimension.
const nnCache = new WeakMap();

function nnIndex(coords, dims) {
  let per = nnCache.get(coords);
  if (!per) { per = new Map(); nnCache.set(coords, per); }
  const key = dims.join(",");
  const hit = per.get(key);
  if (hit) return hit;

  const n = coords.length;
  const m = dims.length;
  // Flatten the compared columns into one typed array. The rows are arrays-of-arrays, and
  // chasing those pointers dominates any distance loop over them; this alone makes even a
  // straight scan several times faster, and it is what the fallback below relies on.
  const flat = new Float32Array(n * m);
  for (let i = 0; i < n; i++) {
    const row = coords[i];
    for (let j = 0; j < m; j++) flat[i * m + j] = row[dims[j]] || 0;
  }
  const gd = dims.slice(0, 3);            // grid on at most three axes
  const k = gd.length;
  const mn = new Array(k).fill(Infinity), mx = new Array(k).fill(-Infinity);
  for (let i = 0; i < n; i++) {
    for (let g = 0; g < k; g++) { const v = flat[i * m + g]; if (v < mn[g]) mn[g] = v; if (v > mx[g]) mx[g] = v; }
  }
  const G = Math.max(1, Math.min(128, Math.round(Math.cbrt(Math.max(1, n / 4)))));   // ~4 points per cell
  const span = Math.max(...mn.map((_, g) => (mx[g] - mn[g]) || 1));
  const h = Math.max(1e-9, span / G);
  const dim = new Array(k);
  for (let g = 0; g < k; g++) dim[g] = Math.max(1, Math.min(G, Math.ceil(((mx[g] - mn[g]) || 1) / h)));
  const strides = new Array(k);
  for (let g = k - 1, s = 1; g >= 0; g--) { strides[g] = s; s *= dim[g]; }
  const cells = dim.reduce((a, b) => a * b, 1);

  const cellOf = new Int32Array(n), start = new Int32Array(cells + 1);
  const cellIx = (g, v) => Math.min(dim[g] - 1, Math.max(0, ((v - mn[g]) / h) | 0));
  for (let i = 0; i < n; i++) {
    let c = 0;
    for (let g = 0; g < k; g++) c += cellIx(g, flat[i * m + g]) * strides[g];
    cellOf[i] = c; start[c + 1]++;
  }
  for (let c = 0; c < cells; c++) start[c + 1] += start[c];
  const items = new Int32Array(n), cur = start.slice(0, cells);
  for (let i = 0; i < n; i++) items[cur[cellOf[i]]++] = i;

  const maxR = Math.max(...dim);
  // The ring search is only worth doing when the grid spans *every* compared dimension.
  // Then the projected distance IS the full distance, the bound is tight, and a query is
  // a local lookup. With more dimensions than the grid covers (curveToIndices searches all
  // 8–10 at once) the bound is weak, the rings keep growing, and the search ends up
  // costing more than the scan it was meant to replace — measurably so. In that case skip
  // the grid and scan the flat array, which is itself several times faster than the
  // arrays-of-arrays scan this all replaced.
  const gridSpansAll = k === m;
  const qv = new Float32Array(m);   // scratch query vector, reused across queries

  const idx = {
    // `hint` is an index believed to be near the answer. It only seeds the distance bound,
    // never the result, so it cannot change what is returned — but a good hint lets the
    // ring search stop almost at once. curveToIndices walks a spline, so each sample's
    // answer is an excellent hint for the next.
    query(q, hint) {
      if (!n) return -1;
      for (let j = 0; j < m; j++) qv[j] = q[j] || 0;   // typed, so the scan below stays monomorphic
      let best = -1, bestD = Infinity;
      const consider = (i) => {
        let dd = 0;
        const o = i * m;
        for (let j = 0; j < m; j++) { const e = flat[o + j] - qv[j]; dd += e * e; }
        if (dd < bestD) { bestD = dd; best = i; }
      };
      if (hint >= 0 && hint < n) consider(hint);

      if (gridSpansAll) {
        const qc = new Array(k);
        for (let g = 0; g < k; g++) qc[g] = cellIx(g, qv[g]);
        const scan = (c) => { for (let t = start[c]; t < start[c + 1]; t++) consider(items[t]); };
        for (let r = 0; r <= maxR; r++) {
          ringCells(qc, r, dim, k, strides, scan);
          // every unscanned cell is at least r cells away on some axis, so nothing out
          // there can be nearer than r*h — once we beat that, we are done
          if (best >= 0 && (r * h) ** 2 >= bestD) return best;
        }
        return best;
      }
      // Flat scan, hoisted out of `consider` so the whole thing is one tight typed loop.
      for (let i = 0, o = 0; i < n; i++, o += m) {
        let dd = 0;
        for (let j = 0; j < m; j++) { const e = flat[o + j] - qv[j]; dd += e * e; }
        if (dd < bestD) { bestD = dd; best = i; }
      }
      return best;
    },
  };
  per.set(key, idx);
  return idx;
}

// Visit every cell whose Chebyshev distance from `qc` is exactly r (clipped to the grid).
function ringCells(qc, r, dim, k, strides, visit) {
  const at = new Array(k);
  const rec = (g) => {
    if (g === k) {
      let onRing = r === 0;
      for (let t = 0; t < k && !onRing; t++) if (Math.abs(at[t] - qc[t]) === r) onRing = true;
      if (!onRing) return;
      let c = 0;
      for (let t = 0; t < k; t++) c += at[t] * strides[t];
      visit(c);
      return;
    }
    const lo = Math.max(0, qc[g] - r), hi = Math.min(dim[g] - 1, qc[g] + r);
    for (let v = lo; v <= hi; v++) { at[g] = v; rec(g + 1); }
  };
  rec(0);
}

// Map a screen point to a full embedding vector: the displayed axes are free (where
// you drew), the remaining axes are borrowed from the nearest state (depth/structure).
function screenToVec(sx, sy) {
  const disp = activeLand().unproject(sx, sy);
  const coords = embedCoords(state.embedding);
  if (!disp || !coords || !coords.length) return null;
  const ax = viewAxes();
  const best = nnIndex(coords, ax).query(disp);   // `disp` is parallel to `ax`
  const vec = best >= 0 ? coords[best].slice() : new Array((coords[0] || disp).length).fill(0);
  ax.forEach((a, d) => { vec[a] = disp[d]; });
  return vec;
}
// Catmull-Rom spline sampled into a dense polyline (data coords); linear for 2 points.
function splineSamples(ctrl, perSeg = 16) {
  const n = ctrl.length, dim = (ctrl[0] || []).length || 2;
  if (n < 2) return ctrl.map((p) => p.slice());
  if (n === 2) { const out = []; for (let t = 0; t <= perSeg; t++) out.push(lerpVec(ctrl[0], ctrl[1], t / perSeg, dim)); return out; }
  const out = [], at = (i) => ctrl[Math.max(0, Math.min(n - 1, i))];
  for (let i = 0; i < n - 1; i++) {
    const p0 = at(i - 1), p1 = at(i), p2 = at(i + 1), p3 = at(i + 2);
    for (let s = 0; s < perSeg; s++) {
      const t = s / perSeg, t2 = t * t, t3 = t2 * t, v = new Array(dim);
      for (let d = 0; d < dim; d++)
        v[d] = 0.5 * ((2 * p1[d]) + (-p0[d] + p2[d]) * t + (2 * p0[d] - 5 * p1[d] + 4 * p2[d] - p3[d]) * t2 + (-p0[d] + 3 * p1[d] - 3 * p2[d] + p3[d]) * t3);
      out.push(v);
    }
  }
  out.push(at(n - 1).slice());
  return out;
}
const lerpVec = (a, b, t, dim) => { const v = new Array(dim); for (let d = 0; d < dim; d++) v[d] = a[d] + (b[d] - a[d]) * t; return v; };

// Show every visible curve layer that matches the current view on the active landscape.
function refreshFreeCurves() {
  if (!scatter) return;
  const land = activeLand();
  if (!land.setFreeCurves) return;
  const list = layers.filter((ly) => ly.kind === "curve" && ly.visible && curveMatchesView(ly))
    // sample full vectors, show displayed axes; the edited curve also carries its
    // control-point handles so the renderer draws them in the same pass (never drift).
    .map((ly) => ({ pts: splineSamples(ly.ctrl, 16).map(axesOf), color: ly.color, handles: ly === editCurve ? ly.ctrl.map(axesOf) : null }));
  land.setFreeCurves(list);
}
// Resample a freehand stroke into a handful of free control points (full embedding vectors).
function strokeToCtrl(strokePts) {
  const pts = resampleStroke(strokePts, 55);          // ~one handle every 55 px
  const ctrl = pts.map(([sx, sy]) => screenToVec(sx, sy)).filter(Boolean);
  return ctrl.length > 24 ? ctrl.filter((_, i) => i % Math.ceil(ctrl.length / 24) === 0) : ctrl;
}
// Snap a curve to an ordered, deduped state-index sequence (silent, at morph time).
//
// Answered by the server, which holds the very same coordinates and can put a k-d tree
// over them: the whole curve comes back in tens of milliseconds. Doing it here means
// finding the nearest of a million states for each of ~550 spline samples — billions of
// distance evaluations on the UI thread, which froze the tab for seconds on a click of ▶.
// A k-d tree in JS would fix the complexity but not the fact that the work sits on the
// one thread that has to stay responsive.
//
// The local scan is kept as a fallback (below): it is exact, just slow, so a viewer whose
// backend is unreachable degrades in speed rather than in behaviour.
async function curveToIndices(ly) {
  const coords = embedCoords(ly.embed);
  if (!coords || !coords.length) return [];
  const dense = splineSamples(ly.ctrl, 24);
  if (!dense.length) return [];
  try {
    const res = await fetch("./api/snap", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      // Array.from: a spline sample can be a Float32Array row view, and JSON.stringify
      // turns one of those into {"0":…,"1":…} rather than a list.
      body: JSON.stringify({ embed: ly.embed, points: dense.map((p) => Array.from(p)) }),
    });
    if (res.ok) return (await res.json()).indices || [];
  } catch { /* fall through to the local scan */ }
  return curveToIndicesLocal(coords, dense);
}

function curveToIndicesLocal(coords, dense) {
  const dim = coords[0].length, seq = [];
  const dims = Array.from({ length: dim }, (_, d) => d);
  const nn = nnIndex(coords, dims);
  let hint = -1;
  for (const p of dense) {
    const best = nn.query(p, hint);   // the previous sample's state is next door on the spline
    hint = best;
    if (best >= 0 && (!seq.length || seq[seq.length - 1] !== best)) seq.push(best);
  }
  return seq;
}
// Editing-hint text — in 3D an empty-space drag orbits, so retrace moves to Shift-drag.
const curveEditHint = () => is3d()
  ? "drag a handle to move · alt-click to delete · drag empty space to rotate · tap to insert · Shift-drag to retrace"
  : "drag a handle to move · alt-click to delete · tap the line to insert · draw to retrace";
function startCurveFromStroke(strokePts) {
  const ctrl = strokeToCtrl(strokePts);
  if (ctrl.length < 2) { status("Draw a longer line across the landscape.", "warn"); return; }
  layerSeq += 1;
  const ly = { name: `Curve ${layers.length + 1}`, color: clusterColor(layerSeq - 1), visible: true, kind: "curve", ctrl, embed: state.embedding, axes: viewAxes(), dims: curveDims() };
  layers.push(ly); editCurve = ly;
  renderLayers(); refreshFreeCurves();
  status(`Drew “${ly.name}” (${ctrl.length} points). Drag handles to edit · tap the line to add a point · ▶ to morph.`);
}
// Load a curve layer for editing (switches the view to the one it was drawn in).
function editCurveLayer(ly) {
  if (editCurve === ly) { editCurve = null; setPathDrawArmed(false); renderLayers(); status(`Finished editing “${ly.name}”.`); return; }
  editingPathLayer = null; editCurve = ly;
  setViewTo(ly.embed, ly.axes, ly.dims);
  if (!pathDrawArmed) setPathDrawArmed(true);
  renderLayers(); refreshFreeCurves();
  status(`Editing “${ly.name}”: ${curveEditHint()}.`, "busy");
}
// Switch the landscape view (embedding / axes / dims) by driving the existing controls.
function setViewTo(embed, axes, dims) {
  if (state.embedding !== embed) { $("embed").value = embed; $("embed").dispatchEvent(new Event("change")); }
  if (state.dims !== dims) { $("dims").value = String(dims); $("dims").dispatchEvent(new Event("change")); }
  if (state.x !== axes[0]) { $("xaxis").value = String(axes[0]); $("xaxis").dispatchEvent(new Event("change")); }
  if (state.y !== axes[1]) { $("yaxis").value = String(axes[1]); $("yaxis").dispatchEvent(new Event("change")); }
  if (dims === 3 && axes[2] != null && state.z !== axes[2]) { $("zaxis").value = String(axes[2]); $("zaxis").dispatchEvent(new Event("change")); }
}
// Insert a control point (full embedding vector) into the curve's nearest segment.
function insertCtrlPointVec(vec) {
  const ctrl = editCurve.ctrl;
  if (ctrl.length < 2) { ctrl.push(vec); return; }
  const disp = ctrl.map(axesOf), p = axesOf(vec);
  let bestSeg = 0, bestD = Infinity;
  for (let i = 0; i < ctrl.length - 1; i++) { const d = segDist(p, disp[i], disp[i + 1]); if (d < bestD) { bestD = d; bestSeg = i; } }
  ctrl.splice(bestSeg + 1, 0, vec);
}
function segDist(p, a, b) {
  const dim = a.length; let len2 = 0, t = 0;
  for (let d = 0; d < dim; d++) len2 += (b[d] - a[d]) ** 2;
  if (len2 > 0) { for (let d = 0; d < dim; d++) t += (p[d] - a[d]) * (b[d] - a[d]); t = Math.max(0, Math.min(1, t / len2)); }
  let dd = 0; for (let d = 0; d < dim; d++) dd += (a[d] + (b[d] - a[d]) * t - p[d]) ** 2;
  return dd;
}
const curveHandleAt = (sx, sy) => editCurve ? hitWaypointPts(sx, sy, activeLand().projectData(editCurve.ctrl.map(axesOf))) : -1;
function hitWaypointPts(sx, sy, proj) {
  let best = -1, bestD = DRAG_R * DRAG_R;
  for (let k = 0; k < proj.length; k++) { const p = proj[k]; if (!p) continue; const dd = (p[0] - sx) ** 2 + (p[1] - sy) ** 2; if (dd < bestD) { bestD = dd; best = k; } }
  return best;
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
    refreshFreeCurves();   // curve + its handles are drawn by the renderer (always aligned)
    return;
  }
  const pts = c.map((r) => [r[state.x], r[state.y]]);
  $("axisinfo").textContent = `${axisLabel(state.x)} · ${axisLabel(state.y)}`;
  scatter.setPoints(pts);
  refreshFreeCurves();
}

// ── binary transfers ─────────────────────────────────────────────────────────
// The landscape arrays arrive as raw buffers (see the server's _f32_matrix). At a million
// states the JSON they replace was a few hundred MB of text that the server spent minutes
// building — with the GIL held, so it stopped answering anything else — and that the
// browser then had to parse. There is nothing to parse here.
//
// Rows are *views* into the one buffer, not copies: `coords[i][d]`, `.length` and
// `.slice()` all behave as before, but a row costs a small view object rather than a
// fresh array of boxed doubles.
async function fetchMatrix(url) {
  const buf = await (await fetch(url)).arrayBuffer();
  const [rows, cols] = new Int32Array(buf, 0, 2);
  const data = new Float32Array(buf, 8, rows * cols);
  const out = new Array(rows);
  for (let i = 0; i < rows; i++) out[i] = data.subarray(i * cols, (i + 1) * cols);
  return out;
}
async function fetchLabels(url) {
  const buf = await (await fetch(url)).arrayBuffer();
  const n = new Int32Array(buf, 0, 1)[0];
  return new Int32Array(buf, 4, n);
}

async function clusterLabels() {
  let labels = clusterCache[state.k];
  if (!labels) { try { labels = clusterCache[state.k] = await fetchLabels(`./api/cluster.bin?k=${state.k}`); } catch { labels = null; } }
  return labels;
}

function toggleColorWraps() {
  $("kwrap").style.display = state.colorBy === "cluster" ? "" : "none";
  $("denswrap").style.display = state.colorBy === "density" ? "" : "none";
}

async function applyColor() {
  const labels = state.colorBy === "cluster" ? await clusterLabels() : null;
  applyLandscapeBg();   // dark backdrop in density mode, else the page theme
  if (is3d() && cloud) {
    const dens = state.colorBy === "density";
    if (dens) {
      const dv = density3(points3, 0.04 + state.densBandwidth * 0.02);
      cloud.setDensityField(dv); cloud.setDensityIntensity(state.densIntensity);
      cloud.setColors(densityRampColors(dv));   // inferno-tinted points, like the desktop app
    } else {
      cloud.setColors(computeColors(labels));
    }
    cloud.showDensity(dens);
    return;
  }
  if (state.colorBy === "layers") { scatter.setDensity(false); scatter.setLabels(null); scatter.setPointColors(layerColorsCSS()); return; }
  scatter.setPointColors(null);
  if (state.colorBy === "density") { scatter.setLabels(null); scatter.setDensityBandwidth(state.densBandwidth); scatter.setDensity(true); return; }
  scatter.setDensity(false);
  scatter.setLabels(state.colorBy === "cluster" ? labels : null);
}

// Per-point RGB (0..1) from local density through the active colormap, lifted off
// the dark low end so even sparse points stay visible (the 2D scatter does the
// equivalent internally). Matches the colored spheres of the desktop landscape.
function densityRampColors(dv) {
  const stops = COLORMAPS[state.densMap] || COLORMAPS.inferno;
  const out = new Float32Array(dv.length * 3);
  for (let i = 0; i < dv.length; i++) {
    const rgb = rampRGB(stops, 0.32 + 0.68 * dv[i]);
    out[i * 3] = rgb[0] / 255; out[i * 3 + 1] = rgb[1] / 255; out[i * 3 + 2] = rgb[2] / 255;
  }
  return out;
}
function rampRGB(stops, t) {
  t = Math.max(0, Math.min(1, t));
  const f = t * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(f)), u = f - i;
  const c = (a, b) => Math.round(a + (b - a) * u);
  return [c(stops[i][0], stops[i + 1][0]), c(stops[i][1], stops[i + 1][1]), c(stops[i][2], stops[i + 1][2])];
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
    for (const ly of layers) { if (!ly.visible || !ly.indices) continue; const c = hexToRgb(ly.color); for (const i of ly.indices) { out[i * 3] = c[0]; out[i * 3 + 1] = c[1]; out[i * 3 + 2] = c[2]; } }
  } else {
    for (let i = 0; i < n; i++) { out[i * 3] = 0.58; out[i * 3 + 1] = 0.64; out[i * 3 + 2] = 0.74; }
  }
  return out;
}

// CSS colors for the 2D scatter in "layers" mode (dim gray + visible layer colors).
function layerColorsCSS() {
  const out = new Array(L.n).fill("rgba(100,116,139,0.4)");
  for (const ly of layers) { if (!ly.visible || !ly.indices) continue; for (const i of ly.indices) out[i] = ly.color; }
  return out;
}

// Local density in [0,1], within `frac` of the bbox diagonal.
//
// Binned: drop the points into a uniform grid, blur it, then read each point back out
// of its cell. O(N + G) rather than the O(N²) all-pairs neighbour count this replaces
// — that scan was ~1e12 distance tests at a million points and simply hung the tab.
//
// The result is a kernel-density estimate instead of an exact neighbour count. That is
// no loss: the value only ever tints points and drives the glow overlay, and it is
// exactly how the 2D view has always computed its density, so the two now agree.
const DENS_GRID = 64;   // cells along the longest axis -> <=64³ = 262k cells, ~1 MB

function density3(pts, frac = 0.12) {
  const n = pts.length, d = new Float32Array(n);
  if (!n) return d;

  const mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
  for (const p of pts) for (let k = 0; k < 3; k++) { const v = p[k] || 0; if (v < mn[k]) mn[k] = v; if (v > mx[k]) mx[k] = v; }
  const ext = [0, 1, 2].map((k) => (mx[k] - mn[k]) || 1);
  const diag = Math.hypot(ext[0], ext[1], ext[2]) || 1;

  // Cubic cells, so a radius in world units is the same number of cells on every axis.
  const h = Math.max(ext[0], ext[1], ext[2]) / DENS_GRID;
  const dim = ext.map((e) => Math.max(1, Math.min(DENS_GRID, Math.ceil(e / h))));
  const [nx, ny, nz] = dim;
  const grid = new Float32Array(nx * ny * nz);

  // bin (O(N)), remembering each point's cell so the read-back is a lookup, not a search
  const cellOf = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const p = pts[i];
    const ix = Math.min(nx - 1, Math.max(0, ((p[0] || 0) - mn[0]) / h | 0));
    const iy = Math.min(ny - 1, Math.max(0, ((p[1] || 0) - mn[1]) / h | 0));
    const iz = Math.min(nz - 1, Math.max(0, ((p[2] || 0) - mn[2]) / h | 0));
    const c = (ix * ny + iy) * nz + iz;
    cellOf[i] = c;
    grid[c] += 1;
  }

  // Three separable box passes ≈ a gaussian of the requested radius. Each pass is a
  // running prefix sum, so the cost does not grow with the radius.
  const rad = Math.max(1, Math.round((frac * diag) / h / 2));
  let src = grid, dst = new Float32Array(grid.length);
  for (let pass = 0; pass < 3; pass++) {
    for (let axis = 0; axis < 3; axis++) {
      boxBlurAxis(src, dst, dim, axis, rad);
      const t = src; src = dst; dst = t;   // ping-pong
    }
  }

  let max = 0;
  for (let i = 0; i < n; i++) { const v = src[cellOf[i]]; d[i] = v; if (v > max) max = v; }
  if (max > 0) for (let i = 0; i < n; i++) d[i] /= max;
  return d;
}

// One separable box-blur pass along `axis` of a 3D grid, as a sliding window sum, so a
// cell costs O(1) however wide the kernel is. Edges clamp (the border cell repeats).
function boxBlurAxis(src, dst, dim, axis, rad) {
  const strides = [dim[1] * dim[2], dim[2], 1];   // grid index = (ix*ny + iy)*nz + iz
  const step = strides[axis], len = dim[axis];
  const [oa, ob] = [0, 1, 2].filter((k) => k !== axis);
  const cl = (i) => (i < 0 ? 0 : i >= len ? len - 1 : i);
  const win = 2 * rad + 1;
  for (let a = 0; a < dim[oa]; a++) {
    for (let b = 0; b < dim[ob]; b++) {
      const base = a * strides[oa] + b * strides[ob];
      let sum = 0;
      for (let k = -rad; k <= rad; k++) sum += src[base + cl(k) * step];
      for (let i = 0; i < len; i++) {
        dst[base + i * step] = sum / win;
        sum += src[base + cl(i + rad + 1) * step] - src[base + cl(i - rad) * step];
      }
    }
  }
}


// Keep both landscape views highlighting the same current state + path + selection.
// Push the selected density colormap to the 2D heatmap and the 3D glow shader.
function applyColormap() {
  const stops = COLORMAPS[state.densMap] || COLORMAPS.viridis;
  scatter.setColormap(stops);
  if (cloud) cloud.setColormap(stops);
}

function syncSelection() {
  // hidePathPicks: a curve-driven morph snaps to states only internally, so don't
  // mark those picks (dots / rings / connecting polyline) — the drawn curve
  // overlay still conveys the path. state.path is untouched, so decoding is fine.
  const pathPts = hidePathPicks ? [] : state.path;
  const traj = hidePathPicks ? [] : [state.current, ...state.path].filter((v) => v >= 0);
  const col = editingPathLayer ? editingPathLayer.color : "#f472b6";
  scatter.select(state.current); scatter.setPath(pathPts); scatter.setSelection(state.selection);
  scatter.setPathOrder(traj, col);
  if (cloud) { cloud.select(state.current); cloud.setPath(pathPts); cloud.setSelection(state.selection); cloud.setPathOrder(traj, hexToRgb(col)); }
}

async function onSelect(i, opts) {
  if (playing) setPlaying(false);              // editing the path stops playback
  hidePathPicks = false;                       // manual interaction -> show path picks again
  if (opts && opts.shift) {
    // shift-click toggles a waypoint: remove it if already on the path, else append
    const at = state.path.indexOf(i);
    if (at >= 0) state.path.splice(at, 1);
    else if (i !== state.current) {
      const last = state.path.length ? state.path[state.path.length - 1] : state.current;
      if (i !== last) state.path.push(i);
    }
    if (editingPathLayer) {                        // keep the edited path layer in sync
      editingPathLayer.indices = [state.current, ...state.path].filter((v) => v >= 0);
      renderLayers(); if (state.colorBy === "layers") applyColor();
    }
    syncSelection(); updatePathLabel();
    return;
  }
  state.current = i;
  if (editingPathLayer) { editingPathLayer.indices = [state.current, ...state.path].filter((v) => v >= 0); renderLayers(); if (state.colorBy === "layers") applyColor(); }
  syncSelection(); updatePathLabel(); $("sel").textContent = `state #${i}`;
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
//
// `onSupersede` fires when a newer call arrives while one is still running, which is the
// cue to abandon the running one. Without it a drag of the threshold slider would let
// every superseded decode finish and ship its mesh — tens of MB each at a 320³ box — only
// for the result to be thrown away on arrival.
function makeLive(fn, onSupersede) {
  let busy = false, pending = false, args = [];
  async function run() { busy = true; do { pending = false; try { await fn(...args); } catch { /* ignore */ } } while (pending); busy = false; }
  return (...a) => { args = a; if (busy) { pending = true; if (onSupersede) onSupersede(); } else run(); };
}
let decodeAbort = null;
const liveDecode = makeLive(() => decode(), () => decodeAbort && decodeAbort.abort());
const liveFrame = makeLive((i) => renderFrame(i));

async function decode() {
  if (state.current < 0) return;
  const token = ++decodeToken;
  const ac = new AbortController();
  decodeAbort = ac;
  status("Decoding map…", "busy");
  try {
    const res = await fetch(`./api/decode?index=${state.current}&sigma=${state.sigma.toFixed(3)}&smooth=${state.smooth}`,
                            { signal: ac.signal });
    if (!res.ok) throw new Error(await res.text());
    const m = parseMesh(await res.arrayBuffer());
    if (token !== decodeToken) return;
    state.refMass = m.mass;
    renderer.setMesh(m.positions, m.normals, m.indices);
    renderer.setColor(colorForIndex(state.current));   // centroids carry their cluster color
    if (state.render !== "surface") { loadVolume(state.current); return; }
    if (!m.indices.length) { renderer.clear(); status("No surface at this contour — lower the threshold.", "warn"); return; }
    status(`${(m.indices.length / 3).toLocaleString()} triangles · σ ${state.sigma.toFixed(1)} · level ${m.level.toFixed(3)}`);
  } catch (err) {
    if (err && err.name === "AbortError") return;   // superseded by a newer request, not a failure
    if (token === decodeToken) status("Decode failed: " + err.message, "err");
  } finally {
    if (decodeAbort === ac) decodeAbort = null;
  }
}

// ---- volume ray-marching mode ----
function setRenderMode(mode) {
  if (mode !== state.render) clearFrameCache();   // surface<->volume payloads differ
  // The surface and the ray-marcher live on different canvases, each with its own
  // camera. Hand the outgoing one over so the map does not jump back to a default view.
  const from = state.render === "surface" ? renderer : volRenderer;
  const camera = from ? from.getCamera() : null;
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
  const to = surf ? renderer : volRenderer;
  if (to && to !== from && camera) to.setCamera(camera);
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
    volRenderer.setVolume(g.nx, g.ny, g.nz, g.data, g.boxLo, g.boxHi);
    volRenderer.setThreshold(normThreshold());
    volRenderer.setColor(colorForIndex(index));
    status(`Volume ${g.nx}×${g.ny}×${g.nz} · ${state.render === "emission" ? "glow" : "MIP"}`);
  } catch (err) { if (token === decodeToken) status("Volume load failed: " + err.message, "err"); }
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

// ---- morph / volume series as a scrubbable timeline ----
// Build the ordered list of [a,b,t] frames for the current path (forward then back,
// so playback loops cleanly and the scrubber covers a full there-and-back). Cheap —
// no network — so it's safe to call on every path edit.
function buildTimeline() {
  clearFrameCache();   // the frame list changed -> any cached frames are stale
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

// ── decoded-frame cache + background fill ──────────────────────────────────
// Each morph frame needs a server-side decode (JAX + marching cubes / grid),
// which used to run synchronously inside the playback loop — so the speed
// slider (which only sets the inter-frame sleep) couldn't outrun it. We now
// decode off the critical path: a memory-budgeted LRU holds parsed frames keyed
// by their decode query, and a background filler decodes the whole timeline (in
// playhead order) as fast as the server allows, using the idle time during the
// fps sleeps. Once the timeline fits the budget it stays fully cached, so every
// loop after the first is pure cache hits and plays at the requested fps with
// no mid-loop stalls. The budget is what bounds memory — sized by bytes (not a
// frame count) because a 96³ grid and a marching-cubes mesh differ ~10×.
const CACHE_BUDGET = 768 * 1024 * 1024;   // ~768 MB of decoded frames (tune freely)
const frameCache = new Map();      // key -> { surface, payload, bytes }
const frameInflight = new Map();   // key -> Promise (single-flight per key)
const frameCacheKeys = [];         // insertion order, for LRU eviction
let cacheBytes = 0;
let fillToken = 0;                  // bumped to cancel an in-flight background fill

function clearFrameCache() {
  fillToken++;                     // cancel any running fill (its frames are stale)
  frameCache.clear(); frameInflight.clear(); frameCacheKeys.length = 0; cacheBytes = 0;
}

// The decode request (query + cache key) for timeline frame i, or null.
function frameRequest(i) {
  if (!state.timeline.length) return null;
  i = Math.max(0, Math.min(state.timeline.length - 1, i | 0));
  const [a, b, t] = state.timeline[i];
  const mass = state.refMass >= 0 ? state.refMass : null;
  if (state.render === "surface") {
    const q = `index=${a}&index2=${b}&t=${t.toFixed(4)}&smooth=${state.smooth}` + (mass != null ? `&mass=${mass}` : `&sigma=${state.sigma.toFixed(3)}`);
    return { surface: true, url: `./api/decode?${q}`, key: `s|${q}` };
  }
  const q = `index=${a}&index2=${b}&t=${t.toFixed(4)}&size=96`;
  return { surface: false, url: `./api/grid?${q}`, key: `v|${q}` };
}

// Decode (or reuse) frame i -> { surface, payload }. Caches the parsed result
// and coalesces concurrent requests for the same key, so the prefetcher and the
// render loop never fetch the same frame twice.
function decodeFrame(i) {
  const req = frameRequest(i);
  if (!req) return Promise.resolve(null);
  const hit = frameCache.get(req.key);
  if (hit) return Promise.resolve(hit);
  const flight = frameInflight.get(req.key);
  if (flight) return flight;
  const p = (async () => {
    const buf = await (await fetch(req.url)).arrayBuffer();
    const payload = req.surface ? parseMesh(buf) : parseGrid(buf);
    const bytes = req.surface
      ? payload.positions.byteLength + payload.normals.byteLength + payload.indices.byteLength
      : payload.data.byteLength;
    const entry = { surface: req.surface, payload, bytes };
    frameCache.set(req.key, entry); frameCacheKeys.push(req.key); cacheBytes += bytes;
    // evict oldest until under budget (never the entry we just added: it's at the tail)
    while (cacheBytes > CACHE_BUDGET && frameCacheKeys.length > 1) {
      const old = frameCacheKeys.shift();
      const e = frameCache.get(old);
      if (e) { cacheBytes -= e.bytes; frameCache.delete(old); }
    }
    return entry;
  })();
  p.finally(() => { if (frameInflight.get(req.key) === p) frameInflight.delete(req.key); });
  frameInflight.set(req.key, p);
  return p;
}

// Decode the whole timeline ahead of the playhead, as fast as the server allows
// (one outstanding request; the fps sleeps leave plenty of idle time). Stops if
// the cache fills the budget — past that point the look-ahead window is all we
// can keep, so further filling would just thrash. Re-launched on each play.
async function backgroundFill() {
  const my = ++fillToken;
  const n = state.timeline.length;
  const { step } = playCadence();
  const start = state.frame;                            // cover the frames the playhead will show
  const count = Math.ceil(n / step) + 1;               // one stride past the wrap, in playhead order
  for (let d = 0; d < count; d++) {
    if (my !== fillToken || !playing) return;          // superseded / stopped / timeline rebuilt
    if (cacheBytes > CACHE_BUDGET * 0.92) return;       // budget reached -> stop pre-filling
    await decodeFrame((start + d * step) % n).catch(() => {});
  }
}

// Render one timeline frame in the current render mode (mass-constant for surfaces).
async function renderFrame(i) {
  if (!state.timeline.length) return;
  i = Math.max(0, Math.min(state.timeline.length - 1, i | 0));
  state.frame = i; $("scrub").value = String(i); $("framelbl").textContent = `${i + 1}/${state.timeline.length}`;
  $("stop").disabled = !(playing || i > 0);
  const [a, b, t] = state.timeline[i];
  const col = lerpColor(colorForIndex(a), colorForIndex(b), t);   // smooth cluster-color transition
  const tok = ++decodeToken;
  try {
    const entry = await decodeFrame(i);
    if (tok !== decodeToken || !entry) return;   // a newer frame superseded this one
    if (entry.surface) {
      const m = entry.payload;
      if (m.indices.length) renderer.setMesh(m.positions, m.normals, m.indices);
      renderer.setColor(col);
    } else if (volRenderer) {
      const g = entry.payload;
      curVolMeta = g; volRenderer.setVolume(g.nx, g.ny, g.nz, g.data, g.boxLo, g.boxHi); volRenderer.setThreshold(normThreshold());
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
  backgroundFill();                            // race ahead of the playhead, filling the cache
  while (playing) {
    const { delay, step } = playCadence();
    let next = state.frame + step;
    if (next >= state.timeline.length) next = 0;     // loop (wrap to start)
    const t0 = performance.now();
    await renderFrame(next);
    if (!playing) break;
    status(`Playing… frame ${state.frame + 1}/${state.timeline.length}`, "busy");
    // hold the display rate without double-counting the decode we just paid for
    await sleep(Math.max(0, delay - (performance.now() - t0)));
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
  backgroundFill();
  await renderFrame(0);
  const { delay: recDelay, step: recStep } = playCadence();   // record what playback shows
  for (let i = recStep; i < state.timeline.length && playing; i += recStep) { await renderFrame(i); status(`Recording… ${i + 1}/${state.timeline.length}`, "busy"); await sleep(recDelay); }
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
  // Middle-mouse drag pans the active landscape even while a tool is armed (the
  // overlay otherwise swallows drags for lasso / draw). Capture phase + stop-
  // propagation so it beats the tool handlers below; routes to the 2D or 3D view.
  let overlayPanning = false, panLX = 0, panLY = 0;
  lc.addEventListener("pointerdown", (e) => {
    if (e.button !== 1) return;
    e.preventDefault(); e.stopImmediatePropagation();
    overlayPanning = true; panLX = e.clientX; panLY = e.clientY; lc.setPointerCapture(e.pointerId);
  }, true);
  lc.addEventListener("pointermove", (e) => {
    if (!overlayPanning) return;
    e.stopImmediatePropagation();
    const dx = e.clientX - panLX, dy = e.clientY - panLY; panLX = e.clientX; panLY = e.clientY;
    if (is3d() && cloud) cloud.orbitPan(dx, dy); else scatter.panBy(dx, dy);
  }, true);
  const endOverlayPan = (e) => {
    if (!overlayPanning) return;
    e.stopImmediatePropagation(); overlayPanning = false;
    try { lc.releasePointerCapture(e.pointerId); } catch {}
  };
  lc.addEventListener("pointerup", endOverlayPan, true);
  lc.addEventListener("auxclick", (e) => { if (e.button === 1) e.preventDefault(); });
  // The overlay only receives wheel events while armed (else pointer-events:none lets
  // them fall through to the landscape canvas, which zooms itself). Forward to the
  // active landscape so the wheel zooms even mid-lasso / mid-edit.
  lc.addEventListener("wheel", (e) => {
    e.preventDefault();
    if (is3d() && cloud) cloud.orbitZoom(e.deltaY);
    else scatter.zoomAt(e.offsetX, e.offsetY, Math.exp(-e.deltaY * 0.0015));
  }, { passive: false });
  lc.addEventListener("pointerdown", (e) => {
    if (!lassoArmed) return;
    lassoing = true;
    lassoMode = e.shiftKey ? "add" : (e.altKey || e.ctrlKey || e.metaKey) ? "remove" : "replace";
    lassoPts = [[e.offsetX, e.offsetY]]; lc.setPointerCapture(e.pointerId);
  });
  lc.addEventListener("pointermove", (e) => { if (!lassoing) return; lassoPts.push([e.offsetX, e.offsetY]); drawLasso(); });
  lc.addEventListener("pointerup", (e) => { if (!lassoing) return; lassoing = false; try { lc.releasePointerCapture(e.pointerId); } catch {} finishLasso(); });

  // Draw tool (same overlay). Dispatches on mode: editing a free curve, editing an
  // index path's waypoints, or a fresh draw (stroke -> new free curve, tap -> select).
  lc.addEventListener("pointerdown", (e) => {
    if (!pathDrawArmed) return;
    lc.setPointerCapture(e.pointerId);
    if (editCurve) {
      const h = curveHandleAt(e.offsetX, e.offsetY);
      if (h >= 0) {
        if (e.altKey || e.ctrlKey || e.metaKey) {        // alt-click a handle = delete it (keep >= 2)
          if (editCurve.ctrl.length > 2) { editCurve.ctrl.splice(h, 1); commitCurve(); status(`Removed a control point (${editCurve.ctrl.length} left).`); }
          return;
        }
        draggingCtrl = h; return;   // renderer already shows the handles
      }
      if (is3d() && cloud && !e.shiftKey) {              // 3D empty space: drag = orbit, tap = insert
        orbiting = true; orbMoved = false; orbX = orbDownX = e.clientX; orbY = orbDownY = e.clientY; return;
      }
      drawing = true; drawPts = [[e.offsetX, e.offsetY]]; return;   // 2D/shift: tap = insert · drag = retrace
    }
    if (editingPathLayer) {                                // index-path waypoint editing
      const proj = activeLand().projectedPoints();
      const wp = hitWaypoint(e.offsetX, e.offsetY, currentTrajectory(), proj);
      // Hold the projection for the whole drag: dragging a waypoint cannot move the
      // camera, so it never changes — and recomputing it per mousemove re-projects and
      // re-allocates every state (a million [x,y] arrays, which froze the drag outright).
      if (wp >= 0) { draggingWp = wp; dragProj = proj; drawDragPreview(currentTrajectory(), wp, [e.offsetX, e.offsetY], proj); return; }
    }
    drawing = true; drawPts = [[e.offsetX, e.offsetY]];     // stroke or tap (resolved on pointerup)
  });
  lc.addEventListener("pointermove", (e) => {
    if (orbiting) {                                       // rotate the 3D camera; curve + handles follow live
      if (Math.abs(e.clientX - orbDownX) + Math.abs(e.clientY - orbDownY) > 4) orbMoved = true;
      if (orbMoved && cloud) cloud.orbitRotate(e.clientX - orbX, e.clientY - orbY);
      orbX = e.clientX; orbY = e.clientY; return;
    }
    if (draggingCtrl >= 0) { const v = screenToVec(e.offsetX, e.offsetY); if (v) { editCurve.ctrl[draggingCtrl] = v; refreshFreeCurves(); } return; }
    if (draggingWp >= 0) { drawDragPreview(currentTrajectory(), draggingWp, [e.offsetX, e.offsetY], dragProj || activeLand().projectedPoints()); return; }
    if (drawing) { drawPts.push([e.offsetX, e.offsetY]); drawStroke(); }
  });
  lc.addEventListener("pointerup", (e) => {
    if (draggingCtrl < 0 && draggingWp < 0 && !drawing && !orbiting) return;
    try { lc.releasePointerCapture(e.pointerId); } catch {}
    if (orbiting) {                                        // a drag rotated; a tap inserts a control point
      orbiting = false;
      if (!orbMoved) { const v = screenToVec(e.offsetX, e.offsetY); if (v) { insertCtrlPointVec(v); commitCurve(); status(`Added a control point (${editCurve.ctrl.length}).`); } }
      return;
    }
    if (draggingCtrl >= 0) {                                // finish moving a control point
      const v = screenToVec(e.offsetX, e.offsetY), pos = draggingCtrl; draggingCtrl = -1;
      if (v) editCurve.ctrl[pos] = v;
      commitCurve(); status(`Moved control point ${pos + 1}.`); return;
    }
    lctx.clearRect(0, 0, lc.clientWidth, lc.clientHeight);
    if (draggingWp >= 0) {                                  // drop the dragged waypoint on the nearest state
      const proj = dragProj || activeLand().projectedPoints();
      const i = nearestPointTo(e.offsetX, e.offsetY, proj), pos = draggingWp; draggingWp = -1; dragProj = null;
      if (i >= 0) { const traj = currentTrajectory(); traj[pos] = i; applyTrajectory(traj, pos === 0); status(`Moved waypoint ${pos + 1} → state #${i}.`); }
      return;
    }
    drawing = false;
    const pts = drawPts; drawPts = []; const tap = strokeSpan(pts) <= 8;
    if (editCurve) {                                        // tap = insert a point · stroke = retrace
      if (tap) { const v = screenToVec(pts[0][0], pts[0][1]); if (v) { insertCtrlPointVec(v); commitCurve(); status(`Added a control point (${editCurve.ctrl.length}).`); } }
      else { const ctrl = strokeToCtrl(pts); if (ctrl.length >= 2) { editCurve.ctrl = ctrl; commitCurve(); status(`Retraced “${editCurve.name}” (${ctrl.length} points).`); } }
      return;
    }
    if (editingPathLayer) {                                 // index-path: tap = add/remove, stroke = retrace
      const proj = activeLand().projectedPoints();
      if (tap) { const i = nearestPointTo(pts[0][0], pts[0][1], proj); if (i >= 0) onSelect(i, { shift: e.shiftKey }); return; }
      const traj = strokeToPath(pts, proj);
      if (traj.length >= 1) { applyTrajectory(traj, true); status(`Retraced into “${editingPathLayer.name}” (${traj.length} states).`); }
      return;
    }
    // fresh: a tap selects a state, a real stroke draws a new free curve
    if (tap) { const proj = activeLand().projectedPoints(); const i = nearestPointTo(pts[0][0], pts[0][1], proj); if (i >= 0) onSelect(i, { shift: e.shiftKey }); return; }
    startCurveFromStroke(pts);
  });
}
function commitCurve() { renderLayers(); refreshFreeCurves(); }
const pathHexColor = () => (editCurve ? editCurve.color : editingPathLayer ? editingPathLayer.color : "#f472b6");
function nearestPointTo(sx, sy, proj) {
  let best = -1, bestD = Infinity;
  for (let i = 0; i < proj.length; i++) { const p = proj[i]; if (!p) continue; const dd = (p[0] - sx) ** 2 + (p[1] - sy) ** 2; if (dd < bestD) { bestD = dd; best = i; } }
  return best;
}
function hitWaypoint(sx, sy, traj, proj) {
  let best = -1, bestD = DRAG_R * DRAG_R;
  for (let k = 0; k < traj.length; k++) { const p = proj[traj[k]]; if (!p) continue; const dd = (p[0] - sx) ** 2 + (p[1] - sy) ** 2; if (dd < bestD) { bestD = dd; best = k; } }
  return best;
}
function strokeSpan(pts) {
  if (!pts || !pts.length) return 0;
  let a = Infinity, b = Infinity, c = -Infinity, d = -Infinity;
  for (const [x, y] of pts) { a = Math.min(a, x); b = Math.min(b, y); c = Math.max(c, x); d = Math.max(d, y); }
  return Math.hypot(c - a, d - b);
}
// Resample a freehand stroke at a fixed arc-length step so dense/jittery sampling
// doesn't bias the snap, then map each sample to its nearest state (deduped).
function resampleStroke(pts, step) {
  if (pts.length < 2) return pts.slice();
  const out = [pts[0]]; let [px, py] = pts[0];
  for (let i = 1; i < pts.length; i++) {
    let [x1, y1] = pts[i], dx = x1 - px, dy = y1 - py, d = Math.hypot(dx, dy);
    while (d >= step) { const t = step / d; px += dx * t; py += dy * t; out.push([px, py]); dx = x1 - px; dy = y1 - py; d = Math.hypot(dx, dy); }
  }
  out.push(pts[pts.length - 1]);
  return out;
}
function strokeToPath(pts, proj) {
  proj = proj || activeLand().projectedPoints();
  const seq = [];
  for (const [sx, sy] of resampleStroke(pts, 10)) {
    const i = nearestPointTo(sx, sy, proj);
    if (i >= 0 && (!seq.length || seq[seq.length - 1] !== i)) seq.push(i);
  }
  return seq;
}
function drawStroke() {
  const lc = $("lassoCanvas"); lctx.clearRect(0, 0, lc.clientWidth, lc.clientHeight);
  if (drawPts.length < 2) return;
  lctx.lineJoin = "round"; lctx.lineCap = "round";
  lctx.beginPath(); lctx.moveTo(drawPts[0][0], drawPts[0][1]);
  for (let i = 1; i < drawPts.length; i++) lctx.lineTo(drawPts[i][0], drawPts[i][1]);
  lctx.strokeStyle = "rgba(15,23,42,0.5)"; lctx.lineWidth = 5; lctx.setLineDash([]); lctx.stroke();
  lctx.strokeStyle = pathHexColor(); lctx.lineWidth = 2.5; lctx.stroke();
}
// Preview the path while a waypoint is being dragged (that node follows the cursor).
function drawDragPreview(traj, pos, cursor, proj) {
  const lc = $("lassoCanvas"); lctx.clearRect(0, 0, lc.clientWidth, lc.clientHeight);
  const pts = traj.map((idx, k) => (k === pos ? cursor : proj[idx] || null));
  const col = pathHexColor();
  lctx.lineJoin = "round"; lctx.lineCap = "round";
  lctx.beginPath(); let started = false;
  for (const p of pts) { if (!p) { started = false; continue; } if (!started) { lctx.moveTo(p[0], p[1]); started = true; } else lctx.lineTo(p[0], p[1]); }
  lctx.strokeStyle = "rgba(15,23,42,0.5)"; lctx.lineWidth = 5; lctx.setLineDash([]); lctx.stroke();
  lctx.strokeStyle = col; lctx.lineWidth = 2.5; lctx.stroke();
  for (let k = 0; k < pts.length; k++) {
    const p = pts[k]; if (!p) continue;
    lctx.beginPath(); lctx.arc(p[0], p[1], k === pos ? 7 : 4, 0, Math.PI * 2);
    lctx.fillStyle = k === pos ? col : "rgba(255,255,255,0.92)";
    lctx.strokeStyle = "rgba(15,23,42,0.65)"; lctx.lineWidth = 1.5; lctx.fill(); lctx.stroke();
  }
}
const LASSO_RGB = { replace: "45,212,191", add: "74,222,128", remove: "248,113,113" };
const syncOverlayArmed = () => $("lassoCanvas").classList.toggle("armed", lassoArmed || pathDrawArmed);
function setLassoArmed(on) {
  lassoArmed = on;
  if (on && pathDrawArmed) setPathDrawArmed(false);   // tools are mutually exclusive
  syncOverlayArmed();
  $("lassoBtn").classList.toggle("active", on);
  if (on) { sizeLasso(); status("Lasso: drag = replace · Shift = add · Alt/Ctrl = remove."); }
}
function setPathDrawArmed(on) {
  pathDrawArmed = on;
  if (on && lassoArmed) setLassoArmed(false);
  if (!on) {
    drawing = false; draggingWp = -1; draggingCtrl = -1; drawPts = []; orbiting = false; dragProj = null;
    const finished = editCurve || editingPathLayer;
    editCurve = null;                                  // curve layers stay visible via refreshFreeCurves
    if (lctx) lctx.clearRect(0, 0, $("lassoCanvas").clientWidth, $("lassoCanvas").clientHeight);
    if (finished) renderLayers();
  }
  syncOverlayArmed();
  $("drawBtn").classList.toggle("active", on);
  if (on) { sizeLasso(); status(editCurve ? `Editing a curve: ${curveEditHint()}.` : editingPathLayer ? "Editing path: draw to retrace · drag a waypoint · shift-tap a state to add/remove." : "Draw a smooth curve across the landscape · it morphs by snapping to states only at play time."); }
}
function drawLasso() {
  const lc = $("lassoCanvas"); lctx.clearRect(0, 0, lc.clientWidth, lc.clientHeight);
  if (lassoPts.length < 2) return;
  const rgb = LASSO_RGB[lassoMode] || LASSO_RGB.replace;
  lctx.beginPath(); lctx.moveTo(lassoPts[0][0], lassoPts[0][1]);
  for (let i = 1; i < lassoPts.length; i++) lctx.lineTo(lassoPts[i][0], lassoPts[i][1]);
  lctx.closePath();
  lctx.fillStyle = `rgba(${rgb},0.16)`; lctx.fill();
  // dark contrast halo first, so the loop reads on light backgrounds too...
  lctx.lineJoin = "round";
  lctx.strokeStyle = "rgba(15,23,42,0.55)"; lctx.lineWidth = 4; lctx.setLineDash([]); lctx.stroke();
  // ...then the bright dashed colored line on top
  lctx.strokeStyle = `rgba(${rgb},1)`; lctx.lineWidth = 2; lctx.setLineDash([6, 4]); lctx.stroke(); lctx.setLineDash([]);
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
    const isCurve = ly.kind === "curve";
    const editing = editingPathLayer === ly || editCurve === ly;
    const li = document.createElement("li"); li.className = "layer-item" + (ly.visible ? "" : " hidden") + (editing ? " editing" : "") + (ly.kind === "path" || isCurve ? " path" : "");
    const eye = iconBtn(ly.visible ? "◉" : "○", "Show / hide", () => { ly.visible = !ly.visible; renderLayers(); if (state.colorBy === "layers") applyColor(); });
    const sw = document.createElement("input"); sw.type = "color"; sw.className = "swatch"; sw.value = ly.color; sw.title = "Layer color";
    sw.addEventListener("input", (e) => { ly.color = e.target.value; if (state.colorBy === "layers") applyColor(); if (isCurve) refreshFreeCurves(); if (editingPathLayer === ly) syncSelection(); });
    const name = document.createElement("span"); name.className = "lname"; name.textContent = (ly.kind === "path" ? "↳ " : isCurve ? "∿ " : "") + ly.name; name.title = "Click to rename";
    name.addEventListener("click", () => { const v = prompt("Layer name", ly.name); if (v) { ly.name = v.trim() || ly.name; renderLayers(); } });
    const cnt = document.createElement("span"); cnt.className = "lcount"; cnt.textContent = `${isCurve ? ly.ctrl.length : ly.indices.length}`;
    li.append(eye, sw, name, cnt);
    if (ly.kind === "path") li.append(iconBtn(editing ? "✓" : "✎", editing ? "Finish editing this path" : "Edit waypoints (shift-click to add/remove)", () => editPathLayer(ly)));
    if (isCurve) li.append(iconBtn(editing ? "✓" : "✎", editing ? "Finish editing this curve" : "Edit the curve (drag handles · alt-click delete · tap to insert)", () => editCurveLayer(ly)));
    li.append(iconBtn("▶", isCurve ? "Morph along this curve (snaps to nearest states)" : ly.kind === "path" ? "Morph along this path" : "Morph through this layer as a volume series", () => playLayerSeries(ly)));
    if (exportAvailable && !isCurve) li.append(iconBtn("⤓", "Save layer (metadata + average volume) to the project", () => saveLayer(ly)));
    li.append(iconBtn("✕", "Delete layer", () => removeLayer(ly)));
    ul.appendChild(li);
  });
  refreshFreeCurves();
}

// Delete a layer. Removing a cluster layer also drops its matching point from the
// Centroids layer (and prunes the centroid's color/coords); if that empties the
// Centroids layer, it's removed too.
function removeLayer(ly) {
  if (editingPathLayer === ly) { editingPathLayer = null; setPathDrawArmed(false); }
  if (editCurve === ly) { editCurve = null; setPathDrawArmed(false); }
  if (ly.kind === "cluster" && ly.centroid != null) {
    const cen = layers.find((x) => x.kind === "centroid");
    if (cen) {
      const at = cen.indices.indexOf(ly.centroid);
      if (at >= 0) cen.indices.splice(at, 1);
      delete indexColor[ly.centroid]; delete centroidPCA[ly.centroid];
      if (!cen.indices.length) layers.splice(layers.indexOf(cen), 1);
    }
  }
  layers.splice(layers.indexOf(ly), 1);
  renderLayers(); refreshFreeCurves(); if (state.colorBy === "layers") applyColor();
}

const coordOf = (i) => (i >= L.n ? (centroidPCA[i] ? centroidPCA[i][state.x] : 0) : (activeCoords() || L.pca)[i][state.x]);
const orderAlongAxis = (idx) => [...idx].sort((a, b) => coordOf(a) - coordOf(b));

// ---- KMeans clustering -> one layer per cluster + a decodable centroid layer ----
// Use your shift-clicked states (the morph picks) as the initial cluster centers.
function clusterFromPicks() {
  const seeds = [...new Set([state.current, ...state.path].filter((v) => v >= 0 && v < L.n))];
  if (seeds.length < 2) { status("Shift-click at least two states first, then “Seed from picks”.", "warn"); return; }
  computeClusters(seeds);
}
async function computeClusters(seeds) {
  const seeded = Array.isArray(seeds) && seeds.length >= 2;
  const k = Math.max(2, Math.min(20, +$("clusterK").value || 5));
  const space = $("clusterSpace").value === "network" ? "network" : "view";
  const q = (seeded ? `seeds=${seeds.join(",")}` : `k=${k}`) + `&space=${space}`;
  const url = `./api/kmeans?${q}`;
  const spaceLbl = space === "network" ? "full network space" : "viewed space";
  const label = (seeded ? `seeded KMeans (${seeds.length} picks)` : `KMeans (k=${k})`) + ` on ${spaceLbl}`;
  status(`Computing ${label}…`, "busy");
  let d, kLabels;
  // Metadata and the per-state labels come over separately — the labels as binary. The
  // server memoises the clustering, so the second request is a lookup, not a second run.
  try {
    d = await (await fetch(url)).json();
    kLabels = await fetchLabels(`./api/kmeans.bin?${q}`);
  } catch (err) { status("Clustering failed: " + err, "err"); return; }
  if (seeded) $("clusterK").value = d.k;   // reflect the seed-derived cluster count
  // drop previous auto-generated cluster/centroid layers (keep manual ones)
  for (let i = layers.length - 1; i >= 0; i--) if (layers[i].kind === "cluster" || layers[i].kind === "centroid") layers.splice(i, 1);
  indexColor = {}; centroidPCA = {};
  const groups = Array.from({ length: d.k }, () => []);
  for (let i = 0; i < kLabels.length; i++) { const g = groups[kLabels[i]]; if (g) g.push(i); }
  // Tag each cluster with its centroid's virtual index so deleting the cluster
  // can drop the matching point from the Centroids layer.
  groups.forEach((idxs, j) => layers.push({ name: `Cluster ${j + 1}`, color: clusterColor(j), indices: idxs, visible: true, kind: "cluster", centroid: d.base + j }));
  const cents = [];
  for (let j = 0; j < d.k; j++) { const idx = d.base + j; cents.push(idx); indexColor[idx] = clusterColor(j); centroidPCA[idx] = d.centroid_pca[j]; }
  layers.push({ name: "Centroids", color: "#ffffff", indices: cents, visible: true, kind: "centroid" });
  $("colorby").value = "layers"; state.colorBy = "layers"; toggleColorWraps();
  renderLayers(); applyColor();
  status(`${seeded ? "Seeded KMeans" : "KMeans"} k=${d.k} on ${spaceLbl}: ${d.k} cluster layers + a centroid layer (▶ to morph between clusters).`);
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
  const saveable = layers.filter((ly) => ly.kind !== "curve");
  if (!saveable.length) { status("No layers to save.", "warn"); return; }
  for (const ly of saveable) await saveLayer(ly);
  status(`Saved ${saveable.length} layer(s) to the project.`);
}

async function playLayerSeries(ly) {
  // curve = snap the spline to states (silently); path = stored waypoint order; else order along the axis
  if (ly.kind === "curve") status("Snapping the curve to states…", "busy");
  const ordered = ly.kind === "curve" ? await curveToIndices(ly)
    : ly.kind === "path" ? ly.indices.filter((v) => v >= 0)
      : orderAlongAxis(ly.indices);
  if (ordered.length < 2) { status(ly.kind === "curve" ? "Draw the curve across at least two states." : "A series needs ≥2 states.", "warn"); return; }
  if (playing) setPlaying(false);
  hidePathPicks = ly.kind === "curve";   // curve snap stays internal-only; other series show their picks
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
