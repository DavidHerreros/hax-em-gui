// Landscape explorer + volume viewer orchestration. Relative URLs (works behind
// the /viewer/<id>/ proxy). Zero dependencies.
import { createRenderer } from "./gl.js";
import { createVolume, smoothGrid } from "./volume.js";
import { createScatter, clusterColor } from "./scatter.js";
import { createCloud } from "./cloud.js";
import { createHistogram } from "./histogram.js";
import { createSlices } from "./slices.js";
import { FMT, parseGrid, parseMesh, unwrap } from "./wire.js";
import { makeLive, startCanvasRecording } from "./viewerbase.js";

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
const state = { embedding: "pca", dims: 2, x: 0, y: 1, z: 2, colorBy: "none", k: 5, densBandwidth: 6, densIntensity: 0.3, densMap: "inferno",
  densStyle: "shells", densGamma: 1.4, densLevels: [25, 50, 75, 90], densShade: true, densFog: true, densCbar: true, current: -1, path: [], selection: [], sigma: 2.5, refMass: -1, render: "surface", vGamma: 1.4, vBright: 1.2, timeline: [], frame: 0, fps: 60, pointAlpha: 1, pathExpanded: false, smooth: 6, pickRestrict: false,
  // outlier filter: same defaults as `filter_latents` (--thr 1.0, --n_neighbours 10)
  zThr: 1, zK: 10, zMode: "dim" };
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
// orthogonal-slice viewer (created lazily on first switch to the Slices render mode)
let slices = null;
// landscape orthogonal-slice viewer (left pane; created lazily) + which view the
// landscape pane shows: the scatter/cloud points, or the density slices.
let landSlices = null, landView = "points";
// 2D context of the overlay canvas that draws KMeans centroid markers (virtual points
// that live off the base cloud), kept in sync with the active projection via onRender.
let centroidCtx = null;

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
// KMeans centroids: virtual indices >= L.n, with per-embedding coords + a per-centroid color
let indexColor = {};     // index -> hex (decoded maps of these indices are auto-colored)
// embed -> { index -> coord vector in THAT embedding }. Centroids used to be stored as a
// single PCA vector and drawn through whatever frame was active, so under UMAP (or raw
// latent) the markers were plotted in PCA units on a UMAP axis and sat well off their
// clusters. Each embedding now keeps its own coordinates, fetched on demand.
let centroidVecs = {};
let clusterQuery = null;   // the ?k=… / ?seeds=… of the current clustering, for those fetches
const centroidCoords = () => centroidVecs[state.embedding] || {};
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
  if (slices) slices.setBackground(bg);
  if (landSlices) landSlices.setBackground(bg);
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
  scatter.onRender(syncCentroidOverlay);   // keep centroid markers on the 2D projection
  scatter.setPickFilter(pickAllowed);      // pick-restrict blocks direct clicks too
  hist = createHistogram($("hist"), onLevel);
  applyColormap();
  applyTheme(new URLSearchParams(location.search).get("theme"));
  window.addEventListener("message", (e) => { if (e.data && e.data.type === "hax-theme") applyTheme(e.data.theme); });

  try {
    L = await (await fetch("./api/landscape")).json();   // metadata only; arrays follow as binary
    L.pca = await fetchMatrix("./api/pca.bin");
    L.rows = await fetchLabels("./api/rows.bin");   // state -> original particle, for filtering
    L.latents = null;   // fetched on demand — only the "latent" embedding view needs them
    clusterCache[5] = await fetchLabels("./api/cluster.bin?k=5");
  } catch (err) { status("Could not load landscape: " + err, "err"); return; }
  $("title").textContent = L.title || "Conformational landscape";
  updateMetaLabel();
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
    if (is3 && !cloud) { try { cloud = createCloud($("scatter3d"), onSelect); cloud.setBackground(hexToRgb($("bg").value)); cloud.setColormap(COLORMAPS[state.densMap] || COLORMAPS.viridis); cloud.onRender(syncCentroidOverlay); cloud.setPickFilter(pickAllowed); } catch (err) { status("WebGL2 unavailable for 3D: " + err.message, "err"); state.dims = 2; $("dims").value = "2"; $("scatter").style.display = ""; $("scatter3d").style.display = "none"; return; } }
    if (landView === "slices") setLandView("points");   // slices are the flat XY/XZ/YZ cuts; leaving them while toggling dims is confusing
    refreshAxisEnabled();
    updatePoints(); applyColor(); syncSelection();   // re-apply color so 2D↔3D keeps density/cluster/layers
    syncDensityPanel();   // the style list is a 3D control; say so when the pane is flat
  });
  $("zaxis").addEventListener("change", (e) => { state.z = +e.target.value; updatePoints(); if (landView === "slices") loadLandSlices(); });
  $("embed").addEventListener("change", async (e) => {
    state.embedding = e.target.value;
    if (state.embedding === "umap" && !umapData) { status("Computing UMAP…", "busy"); try { umapData = await fetchMatrix("./api/umap.bin"); } catch (err) { status("UMAP failed: " + err, "err"); $("embed").value = "pca"; state.embedding = "pca"; return; } status(""); }
    // The raw latent coords are big and only this view uses them, so they are not part of
    // the initial load — fetch them the first time someone actually asks for them.
    if (state.embedding === "latent" && !L.latents) { status("Loading latent coordinates…", "busy"); try { L.latents = await fetchMatrix("./api/latents.bin"); } catch (err) { status("Could not load latents: " + err, "err"); $("embed").value = "pca"; state.embedding = "pca"; return; } status(""); }
    await ensureCentroidCoords();   // centroids live in the new embedding's coordinates too
    populateAxes();        // each embedding has its own axis set (PCs / UMAP dims / latent dims)
    refreshAxisEnabled();
    updatePoints();
    if (landView === "slices") loadLandSlices();
  });
  $("xaxis").addEventListener("change", (e) => { state.x = +e.target.value; updatePoints(); if (landView === "slices") loadLandSlices(); });
  $("yaxis").addEventListener("change", (e) => { state.y = +e.target.value; updatePoints(); if (landView === "slices") loadLandSlices(); });
  $("colorby").addEventListener("change", (e) => { state.colorBy = e.target.value; toggleColorWraps(); applyColor(); });
  $("lassoBtn").addEventListener("click", () => setLassoArmed(!lassoArmed));
  $("landSlicesBtn").addEventListener("click", () => setLandView(landView === "slices" ? "points" : "slices"));
  $("drawBtn").addEventListener("click", () => setPathDrawArmed(!pathDrawArmed));
  $("layersBtn").addEventListener("click", () => {
    const collapsed = $("layersRail").classList.toggle("collapsed");
    $("layersBtn").classList.toggle("active", !collapsed);
  });
  $("addLayer").addEventListener("click", addLayerFromSelection);
  $("addPath").addEventListener("click", makePathLayer);
  // ---- layer visibility: show/hide all + bulk on the checked subset ----
  $("toggleAllLayers").addEventListener("click", toggleAllLayers);
  $("selAllLayers").addEventListener("change", (e) => { layers.forEach((l) => (l.selected = e.target.checked)); renderLayers(); });
  $("showSelLayers").addEventListener("click", () => setCheckedVisible(true));
  $("hideSelLayers").addEventListener("click", () => setCheckedVisible(false));
  $("pickRestrict").addEventListener("change", (e) => {
    state.pickRestrict = e.target.checked;
    renderLayers();
    status(state.pickRestrict ? pickScopeMsg() : "Picking any point again.");
  });
  $("clusterBtn").addEventListener("click", () => computeClusters());
  $("clusterSeedBtn").addEventListener("click", clusterFromPicks);
  $("saveAll").addEventListener("click", saveAllLayers);
  $("layersBtn").classList.add("active");   // rail visible by default
  $("k").addEventListener("input", (e) => { state.k = +e.target.value; $("kval").textContent = state.k; if (state.colorBy === "cluster") applyColor(); });
  $("ptsalpha").addEventListener("input", (e) => { state.pointAlpha = +e.target.value; scatter.setPointAlpha(state.pointAlpha); if (cloud) cloud.setPointAlpha(state.pointAlpha); });
  $("pathToggle").addEventListener("click", () => { state.pathExpanded = !state.pathExpanded; updatePathLabel(); });
  let bandwidthPending = 0;
  $("denssigma").addEventListener("input", (e) => {
    state.densBandwidth = +e.target.value;
    $("denssigmaOut").textContent = (0.04 + state.densBandwidth * 0.02).toFixed(2);
    if (bandwidthPending) return;
    bandwidthPending = requestAnimationFrame(() => {
      bandwidthPending = 0;
      if (landView === "slices") loadLandSlices();   // the slice field uses the same bandwidth
      if (state.colorBy !== "density") return;
      if (is3d() && cloud) applyColor(); else scatter.setDensityBandwidth(state.densBandwidth);
    });
  });
  $("densint").addEventListener("input", (e) => {
    state.densIntensity = +e.target.value;
    $("densintOut").textContent = state.densIntensity.toFixed(2);
    if (cloud) cloud.setDensityIntensity(state.densIntensity);
  });
  $("densgamma").addEventListener("input", (e) => {
    state.densGamma = +e.target.value;
    $("densgammaOut").textContent = state.densGamma.toFixed(1);
    if (cloud) cloud.setDensityGamma(state.densGamma);
  });
  $("densmap").addEventListener("change", (e) => { state.densMap = e.target.value; applyColormap(); if (state.colorBy === "density") applyColor(); if (landSlices) landSlices.setColormap(sliceColormap()); drawDensityBar(); });
  $("denslevels").addEventListener("change", (e) => {
    state.densLevels = e.target.value.split(",").map(Number);
    if (state.colorBy === "density") applyColor();
  });
  $("densshade").addEventListener("change", (e) => { state.densShade = e.target.checked; if (cloud) cloud.setDensityShade(state.densShade); });
  $("densfog").addEventListener("change", (e) => { state.densFog = e.target.checked; if (cloud) cloud.setDensityFog(state.densFog); });
  $("denscbar").addEventListener("change", (e) => { state.densCbar = e.target.checked; drawDensityBar(); });
  setupDensityPanel();
  setupOutlierPanel();
  $("light").addEventListener("change", (e) => renderer.setLighting(e.target.value));
  // Smoothing is baked into the cached meshes, so changing it invalidates every frame. That
  // flush also stops the prefetcher (clearFrameCache bumps fillToken), so restart it — and
  // while playing let the play loop redraw, rather than forcing a full-resolution mesh of
  // the current frame into the middle of the animation.
  $("smooth").addEventListener("input", (e) => {
    state.smooth = +e.target.value;
    clearFrameCache();
    scheduleWarm();          // re-decode the morph at the new smoothing, in the background
    if (playing) { prefetch(true); return; }
    if (state.timeline.length && state.frame > 0) liveFrame(state.frame, false);
    else liveDecode();
  });
  $("color").addEventListener("input", (e) => { const c = hexToRgb(e.target.value); renderer.setColor(c); if (volRenderer) volRenderer.setColor(c); });
  $("bg").addEventListener("input", (e) => { const c = hexToRgb(e.target.value); renderer.setBackground(c); if (volRenderer) volRenderer.setBackground(c); if (slices) slices.setBackground(e.target.value); if (landSlices) landSlices.setBackground(e.target.value); });
  $("reset").addEventListener("click", () => { renderer.resetView(); if (volRenderer) volRenderer.resetView(); if (slices) slices.resetZoom(); });
  $("mrc").addEventListener("click", saveMrc);
  $("saveMap").addEventListener("click", () => savePng(state.render === "surface" ? renderer : state.render === "slices" ? slices : volRenderer, "map"));
  $("saveLand").addEventListener("click", () => savePng(landView === "slices" ? landSlices : activeLand(), "landscape"));
  $("play").addEventListener("click", togglePlay);
  $("record").addEventListener("click", recordSeries);
  $("stop").addEventListener("click", stopPlayback);
  // Scrubbing is a drag: draft while it moves, full resolution once it settles.
  $("scrub").addEventListener("input", async (e) => {
    const v = +e.target.value;
    setPlaying(false);
    if (!state.timeline.length) await buildTimeline();   // async now: it prices the hops
    liveFrame(v, true);
  });
  $("scrub").addEventListener("change", (e) => { if (!playing) liveFrame(+e.target.value, false); });
  $("frames").addEventListener("change", () => { buildTimeline(); });
  $("speed").addEventListener("input", (e) => { state.fps = +e.target.value; if (playing) prefetch(true); });
  $("clearpath").addEventListener("click", clearPath);
  $("rendermode").addEventListener("change", (e) => setRenderMode(e.target.value));
  $("vgamma").addEventListener("input", (e) => { state.vGamma = +e.target.value; if (volRenderer) volRenderer.setGamma(state.vGamma); });
  $("vbright").addEventListener("input", (e) => { state.vBright = +e.target.value; if (volRenderer) volRenderer.setBrightness(state.vBright); });
  // ---- orthogonal slices (decoded map, right pane) ----
  $("slicemap").addEventListener("change", (e) => { if (slices) slices.setColormap(e.target.value); });
  $("slicelo").addEventListener("input", () => { if (slices) slices.setContrast(+$("slicelo").value, +$("slicehi").value); });
  $("slicehi").addEventListener("input", () => { if (slices) slices.setContrast(+$("slicelo").value, +$("slicehi").value); });
  $("slicecross").addEventListener("change", (e) => { if (slices) slices.setCrosshair(e.target.checked); });
  $("sliceReset").addEventListener("click", () => { if (slices) slices.resetZoom(); });
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
  buildTimeline();        // async (it prices the hops in latent space); nothing here awaits it
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
  const buf = unwrap(await (await fetch(url)).arrayBuffer(), FMT.f32matrix);
  const [rows, cols] = new Int32Array(buf, 0, 2);
  const data = new Float32Array(buf, 8, rows * cols);
  const out = new Array(rows);
  for (let i = 0; i < rows; i++) out[i] = data.subarray(i * cols, (i + 1) * cols);
  return out;
}
async function fetchLabels(url) {
  const buf = unwrap(await (await fetch(url)).arrayBuffer(), FMT.i32vector);
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
  const dens = state.colorBy === "density";
  $("denswrap").style.display = dens ? "" : "none";
  $("densBtn").classList.toggle("active", dens);
  if (!dens) openDensityPanel(false);
  syncDensityPanel();
}

async function applyColor() {
  const labels = state.colorBy === "cluster" ? await clusterLabels() : null;
  applyLandscapeBg();   // dark backdrop in density mode, else the page theme
  // While the filter panel is open the marking of what would be cut takes precedence over
  // the colour mode — you are choosing a threshold, and that is the only thing you need to
  // see. Closing the panel puts the colour mode back.
  if (zPreviewing()) {
    if (is3d() && cloud) { cloud.showDensity(false); cloud.setColors(zPreviewColors()); }
    else { scatter.setDensity(false); scatter.setLabels(null); scatter.setPointColors(zColorsCSS()); }
    return;
  }
  if (is3d() && cloud) {
    const dens = state.colorBy === "density";
    if (dens) {
      cloud.batch(() => {
        const d = landscapeDensity();
        if (d) {
          cloud.setDensityField(d.values);
          cloud.setDensityQuantiles(d.quantiles);
          cloud.setDensityNormals(d.normals);
          if (state.densStyle === "volume") cloud.setDensityVolume(d.grid);
          cloud.setColors(densityPointColors(d));
        }
        pushDensityParams();
      });
    } else {
      cloud.setColors(computeColors(labels));
    }
    cloud.showDensity(dens);
    drawDensityBar();
    return;
  }
  if (state.colorBy === "layers") { scatter.setDensity(false); scatter.setLabels(null); scatter.setPointColors(layerColorsCSS()); return; }
  scatter.setPointColors(null);
  if (state.colorBy === "density") { scatter.setLabels(null); scatter.setDensityBandwidth(state.densBandwidth); scatter.setDensity(true); return; }
  scatter.setDensity(false);
  scatter.setLabels(state.colorBy === "cluster" ? labels : null);
}

// Per-point RGB (0..1) for the 3D cloud in density mode. The points are drawn over the
// density render, so they take the same transfer function it does — otherwise the two
// disagree about what a colour means and the picture reads as two overlaid pictures.
const DENS_EPS = 0.004;                       // density floor of the log transfer
const DENS_KT = -Math.log(DENS_EPS);          // ...so the free-energy scale runs 0..DENS_KT
function densityPointColors(d) {
  const dv = d.values, qv = d.quantiles, n = dv.length;
  const out = new Float32Array(n * 3);
  const put = (i, rgb) => { out[i * 3] = rgb[0] / 255; out[i * 3 + 1] = rgb[1] / 255; out[i * 3 + 2] = rgb[2] / 255; };
  if (state.densStyle === "shells") {
    const bands = shellBands(state.densLevels);
    for (let i = 0; i < n; i++) {
      let b = bands.length - 1;
      for (let k = 0; k < bands.length; k++) if (qv[i] <= bands[k].q) { b = k; break; }
      put(i, bands[b].color);
    }
    return out;
  }
  const stops = COLORMAPS[state.densMap] || COLORMAPS.inferno;
  if (state.densStyle === "energy") {
    for (let i = 0; i < n; i++) {
      const t = (Math.log(Math.max(dv[i], DENS_EPS)) + DENS_KT) / DENS_KT;
      put(i, rampRGB(stops, 0.15 + 0.85 * t));
    }
    return out;
  }
  if (state.densStyle === "points" || state.densStyle === "volume") {
    for (let i = 0; i < n; i++) {
      const t = Math.pow(dv[i], 0.7);
      put(i, [74 + 168 * t, 87 + 160 * t, 107 + 148 * t]);
    }
    return out;
  }
  // splats: the ramp tint the viewer has always used, lifted off the dark end
  for (let i = 0; i < n; i++) put(i, rampRGB(stops, 0.32 + 0.68 * dv[i]));
  return out;
}

// Shell bands. The ramp runs light/warm at the core (read first) to dark/cool in the
// tail, and alpha falls geometrically, so nesting reads as nesting rather than as hue.
const SHELL_RAMP = [[252, 232, 180], [243, 168, 66], [221, 102, 58], [181, 58, 86], [104, 43, 104], [46, 56, 80]];
const SHELL_ALPHA = [0.95, 0.62, 0.38, 0.22, 0.13, 0.07];
function shellBands(levels) {
  const n = Math.min(6, levels.length + 1);
  return levels.slice(0, 5).concat([101]).map((pct, i) => ({
    q: Math.min(1.01, pct / 100),
    color: SHELL_RAMP[Math.round(i * (SHELL_RAMP.length - 1) / Math.max(1, n - 1))],
    alpha: SHELL_ALPHA[Math.min(i, SHELL_ALPHA.length - 1)],
    label: pct > 100 ? "tail" : pct + "%",
  }));
}

// ── the density panel ─────────────────────────────────────────────────────
// The landscape toolbar already carries eleven controls; six render styles plus their
// parameters will not fit in it. So they live behind one button, in a panel that shows
// only the rows the selected style actually reads — four to six at a time, never eleven.
const DENSITY_STYLES = [
  { id: "shells", name: "Population shells", sub: "nested quantile bands", uses: ["levels", "shade"],
    unit: "% of particles enclosed", note: "Each shell encloses the stated share of the particles." },
  { id: "volume", name: "Volumetric", name2: "volume", sub: "ray-marched KDE grid", uses: ["gamma", "cmap", "shade"],
    unit: "ρ / ρmax", note: "Marches the same grid the Slices view bins. Real occlusion." },
  { id: "energy", name: "Free energy", sub: "−ln p, in units of kT", uses: ["gamma", "cmap", "shade"],
    unit: "ΔG (kT), 0 = most populated", note: "The log compresses six decades of occupancy without clipping." },
  { id: "splats", name: "Sorted splats", sub: "density, composited", uses: ["gamma", "cmap", "shade"],
    unit: "ρ / ρmax", note: "The old kernel, depth-sorted and over-blended instead of summed." },
  { id: "points", name: "Graded points", sub: "one hue, no glow", uses: ["shade"],
    unit: "ρ / ρmax", note: "Density as a substrate, so the path and selection stay loudest." },
];
const densStyleDef = (id) => DENSITY_STYLES.find((d) => d.id === id) || DENSITY_STYLES[0];

function setupDensityPanel() {
  const pop = $("densPop"), btn = $("densBtn"), list = $("densStyles");
  DENSITY_STYLES.forEach((d) => {
    const b = document.createElement("button");
    b.type = "button"; b.className = "dens-style"; b.dataset.style = d.id;
    b.innerHTML = `<span class="rd">○</span><span>${d.name}<small>${d.sub}</small></span>`;
    b.addEventListener("click", () => setDensityStyle(d.id));
    list.appendChild(b);
  });
  btn.addEventListener("click", (e) => { e.stopPropagation(); openDensityPanel(pop.dataset.open !== "true"); });
  pop.addEventListener("click", (e) => e.stopPropagation());
  document.addEventListener("click", () => openDensityPanel(false));
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") openDensityPanel(false); });
  $("densReset").addEventListener("click", resetDensityPanel);
  $("denssigma").value = state.densBandwidth;
  $("densint").value = state.densIntensity;
  $("densgamma").value = state.densGamma;
  $("denssigmaOut").textContent = (0.04 + state.densBandwidth * 0.02).toFixed(2);
  $("densintOut").textContent = state.densIntensity.toFixed(2);
  $("densgammaOut").textContent = state.densGamma.toFixed(1);
  syncDensityPanel();
}

function openDensityPanel(on) {
  const pop = $("densPop");
  if (!pop) return;
  pop.dataset.open = String(!!on);
  $("densBtn").setAttribute("aria-expanded", String(!!on));
  if (!on) return;
  // Bound it against where the button actually sits, so the panel always ends above the
  // bottom of the window — the pane can be far shorter than the panel wants to be.
  const r = $("densBtn").getBoundingClientRect();
  pop.style.maxHeight = Math.max(200, window.innerHeight - r.bottom - 28) + "px";
}

function setDensityStyle(id) {
  state.densStyle = id;
  syncDensityPanel();
  if (state.colorBy === "density") applyColor(); else drawDensityBar();
}

// The panel's whole job: show the rows this style reads, and nothing else.
function syncDensityPanel() {
  const d = densStyleDef(state.densStyle);
  const volumeOk = !cloud || cloud.hasVolumeSupport();
  $("densBtnStyle").textContent = (d.name2 || d.name.split(" ")[0]).toLowerCase();
  $("densHint").textContent = d.unit;
  for (const b of $("densStyles").children) {
    const on = b.dataset.style === state.densStyle;
    b.setAttribute("aria-pressed", String(on));
    b.querySelector(".rd").textContent = on ? "◉" : "○";
    if (b.dataset.style === "volume") {
      b.disabled = !volumeOk;
      b.title = volumeOk ? "" : "This browser refused the 3D-texture shader.";
    }
  }
  for (const el of $("densPop").querySelectorAll("[data-when]")) {
    const k = el.dataset.when;
    el.classList.toggle("dens-hidden", !(k === "all" || d.uses.includes(k)));
  }
  // 2D keeps its own heatmap; the styles below are 3D renders of a 3D field.
  const flat = !is3d();
  $("densStyles").style.opacity = flat ? "0.45" : "";
  $("densNote").textContent = flat
    ? "The 2D view draws its own heatmap — these styles apply to the 3D landscape."
    : d.note;
  drawDensityBar();
}

function resetDensityPanel() {
  state.densBandwidth = 6; state.densIntensity = 0.3; state.densGamma = 1.4;
  state.densLevels = [25, 50, 75, 90]; state.densMap = "inferno";
  state.densShade = true; state.densFog = true; state.densCbar = true;
  $("denssigma").value = 6; $("densint").value = 0.3; $("densgamma").value = 1.4;
  $("denslevels").value = "25,50,75,90"; $("densmap").value = "inferno";
  $("densshade").checked = true; $("densfog").checked = true; $("denscbar").checked = true;
  $("denssigmaOut").textContent = (0.04 + 6 * 0.02).toFixed(2);
  $("densintOut").textContent = "0.30";
  $("densgammaOut").textContent = "1.4";
  applyColormap();
  if (landSlices) landSlices.setColormap(sliceColormap());
  if (state.colorBy === "density") applyColor(); else drawDensityBar();
}

// ── the outlier filter ────────────────────────────────────────────────────
// `filter_latents` as a control instead of a job, in two stages that cost very different
// things:
//
//   preview  the backend computes the neighbour z-scores once per k (/api/zscore.bin);
//            the threshold is then `z < thr` over a typed array we already hold, so it
//            can be dragged. Nothing is removed — the states that *would* go are marked.
//   apply    the states are dropped and the PCA, UMAP and clustering are refitted on what
//            is left. This is the part that is worth doing: outliers stretch the principal
//            axes toward themselves and sit in the k-NN graph UMAP is built from, so a
//            landscape drawn with them in it is distorted for every state that stays.
//
// Applying renumbers the states, but only inside this viewer — the backend keeps the map
// from each state to its original particle, so decoded maps and exported metadata still
// address the right rows. That map is exactly what `filtered_latents.npy` throws away.
let zHist = null;
let zData = null;      // { k, v: Float32Array, max, ms } — the fetched z-scores, for one k
let zInflight = null;  // { k, p } — the request in flight, so concurrent callers share it

// Unlike the other binary fetches this one is over a route that can genuinely fail (the
// neighbour search is the only computation in this backend that can run out of memory),
// so check the status rather than parsing a JSON error body as float32.
async function fetchVector(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error((await r.text()) || r.statusText);
  const buf = unwrap(await r.arrayBuffer(), FMT.f32vector);
  const n = new Int32Array(buf, 0, 1)[0];
  return new Float32Array(buf, 4, n);
}

// The z-scores for the current k, fetched once. Returns null if the search failed.
// Both the colour path and the k slider ask for this, and a slider release can arrive
// while the first request is still out, so the in-flight promise is shared rather than
// re-issued — the backend would otherwise repeat a search that costs real seconds.
function ensureZScores() {
  if (zData && zData.k === state.zK) return Promise.resolve(zData);
  if (zInflight && zInflight.k === state.zK) return zInflight.p;
  const k = state.zK;
  status(`Searching neighbours (k = ${k})…`, "busy");
  const t0 = performance.now();
  const p = fetchVector(`./api/zscore.bin?k=${k}`).then((v) => {
    let max = 0;
    for (let i = 0; i < v.length; i++) if (v[i] > max) max = v[i];
    // A late response for a k nobody is asking for any more must not overwrite the
    // current one, but it is still worth keeping if nothing else has landed.
    if (state.zK !== k && zData) return zData;
    zData = { k, v, max, ms: performance.now() - t0 };
    // The threshold range has to cover the tail we actually got, and that tail is
    // heavy — a fixed 0..4 would squeeze the whole distribution into a corner.
    $("zthr").max = Math.max(1, max).toFixed(2);
    if (state.zThr > max) { state.zThr = max; $("zthr").value = max; }
    zHist.setData(zHistogram(v, max), state.zThr);
    status(`Neighbour table for k = ${k} in ${(zData.ms / 1000).toFixed(1)} s — the threshold is free from here.`);
    return zData;
  }).catch((err) => {
    status("Neighbour search failed: " + err, "err");
    return null;
  }).finally(() => { if (zInflight && zInflight.k === k) zInflight = null; });
  zInflight = { k, p };
  return p;
}

// Bin the z-scores for the contour widget's data shape. Its own draw() takes the log,
// which is what makes a distribution with a long thin tail readable at all.
const Z_BINS = 72;
function zHistogram(v, max) {
  const counts = new Float32Array(Z_BINS);
  const hi = max || 1;
  for (let i = 0; i < v.length; i++) counts[Math.min(Z_BINS - 1, (v[i] / hi * Z_BINS) | 0)]++;
  return { counts, edges: null, min: 0, max: hi, mean: 0, std: 1 };
}

// CSS colours for the 2D scatter: kept points keep the scatter's own neutral (the filter
// marks what it removes, it does not restyle the rest); rejected ones are either pushed
// back or called out. Two modes, not three — "hidden" is only expressible in 2D (the 3D
// cloud's colours carry no alpha), and a filter that means different things in the two
// views is worse than one that means less.
//
// The array is allocated once and rewritten in place. This runs on every frame of a
// threshold drag, and a fresh million-entry array per frame is the kind of garbage that
// turns a free operation into a stuttering one.
const Z_KEEP_CSS = "rgba(148,163,184,0.6)";
const Z_CUT_CSS = { dim: "rgba(100,116,139,0.14)", flag: "rgba(248,113,113,0.9)" };
let zColorBuf = null;
function zColorsCSS() {
  if (!zColorBuf || zColorBuf.length !== L.n) zColorBuf = new Array(L.n);
  const cut = Z_CUT_CSS[state.zMode] || Z_CUT_CSS.dim;
  const v = zData ? zData.v : null, thr = state.zThr;
  for (let i = 0; i < L.n; i++) zColorBuf[i] = (!v || v[i] < thr) ? Z_KEEP_CSS : cut;
  return zColorBuf;
}

// The same preview for the 3D cloud. Its colours carry no alpha, so a dimmed point is a
// dark grey rather than a faded one.
function zPreviewColors() {
  const out = new Float32Array(L.n * 3);
  const cut = state.zMode === "flag" ? [0.97, 0.44, 0.44] : [0.19, 0.22, 0.28];
  const v = zData ? zData.v : null, thr = state.zThr;
  for (let i = 0; i < L.n; i++) {
    const keep = !v || v[i] < thr;
    out[i * 3] = keep ? 0.58 : cut[0];
    out[i * 3 + 1] = keep ? 0.64 : cut[1];
    out[i * 3 + 2] = keep ? 0.74 : cut[2];
  }
  return out;
}

function setupOutlierPanel() {
  zHist = createHistogram($("zhist"), (level, dragging) => {
    state.zThr = level;
    $("zthr").value = level;
    applyZThreshold(dragging);
  }, { keep: "below" });

  const pop = $("outPop");
  $("outBtn").addEventListener("click", (e) => { e.stopPropagation(); openOutlierPanel(pop.dataset.open !== "true"); });
  pop.addEventListener("click", (e) => e.stopPropagation());
  document.addEventListener("click", () => openOutlierPanel(false));
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") openOutlierPanel(false); });

  $("zthr").addEventListener("input", (e) => {
    state.zThr = +e.target.value;
    zHist.setLevel(state.zThr);       // the histogram line is the same threshold
    applyZThreshold(true);
  });
  $("zthr").addEventListener("change", () => status(zSummary()));
  // k is the one control that costs a round trip, so it commits on `change` (release)
  // rather than on every intermediate value the slider passes through on the way.
  $("zk").addEventListener("input", (e) => { state.zK = +e.target.value; $("zkOut").textContent = state.zK; });
  $("zk").addEventListener("change", async () => {
    if (zData && zData.k === state.zK) return;
    if (await ensureZScores()) { applyColor(); syncOutlierPanel(); }
  });
  $("zModes").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-zmode]");
    if (!b) return;
    state.zMode = b.dataset.zmode;
    for (const el of $("zModes").children) el.setAttribute("aria-pressed", String(el === b));
    applyColor();
  });
  $("zApply").addEventListener("click", () => commitFilter(state.zThr));
  $("zReset").addEventListener("click", () => commitFilter(null));
}

// Opening the panel turns the preview on, and closing it turns the preview off — the
// marking is a thing you do *while* choosing a threshold, so it has no reason to outlive
// the panel, and it composes with whatever the `color` select is set to instead of
// competing with it for the same control.
async function openOutlierPanel(on) {
  const pop = $("outPop");
  if (!pop) return;
  const was = pop.dataset.open === "true";
  pop.dataset.open = String(!!on);
  $("outBtn").setAttribute("aria-expanded", String(!!on));
  $("outBtn").classList.toggle("active", !!on);
  if (!on) {
    if (was) applyColor();          // drop the preview colours
    return;
  }
  const r = $("outBtn").getBoundingClientRect();
  pop.style.maxHeight = Math.max(200, window.innerHeight - r.bottom - 28) + "px";
  // A canvas inside a display:none panel has no size, so anything drawn into it while the
  // panel was shut went nowhere. The line above has already forced layout, so by here the
  // canvas has its real width and a redraw lands.
  if (zHist) zHist.setLevel(state.zThr);
  syncOutlierPanel();
  if (!was && await ensureZScores()) { syncOutlierPanel(); applyColor(); }
}

const zPreviewing = () => $("outPop").dataset.open === "true" && !!zData;

// Recolour on a new threshold. The panel's numbers update on every event (they are a
// single pass over the z-scores), but the repaint is coalesced to one per frame — a
// pointer can emit several moves between two frames, and painting each of them would be
// work nobody ever sees. The callback reads `state.zThr` when it fires, so the frame that
// does land is always the current threshold, never a stale one.
let zPending = 0;
function applyZThreshold(dragging) {
  syncOutlierPanel();
  if (!dragging) status(zSummary());
  if (zPending) return;
  zPending = requestAnimationFrame(() => { zPending = 0; applyColor(); });
}

function zSummary() {
  if (!zData) return "";
  const kept = zCount(), cut = L.n - kept;
  return `z < ${state.zThr.toFixed(2)} keeps ${kept} of ${L.n} states (${((cut / L.n) * 100).toFixed(1)}% cut).`;
}

function zCount() {
  if (!zData) return L ? L.n : 0;
  let kept = 0;
  for (let i = 0; i < zData.v.length; i++) if (zData.v[i] < state.zThr) kept++;
  return kept;
}

function syncOutlierPanel() {
  $("zthrOut").textContent = state.zThr.toFixed(2);
  $("zkOut").textContent = state.zK;
  if (!zData) { $("zNote").textContent = ""; $("zTiming").textContent = ""; return; }
  const kept = zCount(), cut = L.n - kept;
  $("zKept").textContent = kept.toLocaleString();
  $("zCut").textContent = `${cut.toLocaleString()} cut`;
  $("zBarKeep").style.width = `${(kept / L.n) * 100}%`;
  $("zBarCut").style.width = `${(cut / L.n) * 100}%`;
  $("zApply").disabled = filterBusy || cut === 0;
  $("zNote").textContent = cut === 0
    ? "Nothing is cut at this threshold — drag the line left."
    : `Applying drops ${((cut / L.n) * 100).toFixed(1)}% of the set and recomputes the PCA, `
      + `UMAP and clustering on the rest. Same cut as `
      + `filter_latents --thr ${state.zThr.toFixed(2)} --n_neighbours ${zData.k}.`;
  $("zTiming").textContent = `k-NN ${(zData.ms / 1000).toFixed(1)}s · once per k`;
}

// Drop the outliers for real: the backend refits the embeddings on what is left and
// renumbers the states, then everything the browser holds about the old numbering is
// rebuilt. `thr` of null restores the full set.
let filterBusy = false;
async function commitFilter(thr) {
  if (filterBusy) return;
  if (thr !== null && !zData) { status("No z-scores yet.", "warn"); return; }
  filterBusy = true;
  $("zApply").disabled = $("zReset").disabled = true;
  const oldRows = L.rows;
  status(thr === null ? "Restoring every state and refitting…" : "Dropping outliers and refitting the landscape…", "busy");
  try {
    const r = await fetch("./api/filter", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ thr, k: state.zK }),
    });
    if (!r.ok) throw new Error(((await r.json()).detail) || r.statusText);
    const d = await r.json();
    await reloadLandscape(oldRows);
    status(d.filtered
      ? `Refitted on ${d.n} of ${d.n_all} states — PCA, UMAP and clustering recomputed without the ${d.n_all - d.n} outliers.`
      : `Restored all ${d.n} states and refitted on the full set.`);
  } catch (err) {
    status("Filter failed: " + err.message, "err");
  } finally {
    filterBusy = false;
    $("zApply").disabled = $("zReset").disabled = false;
    syncOutlierPanel();
  }
}

// Rebuild every cached view of the landscape after the state set changed. Anything keyed
// on a state index is either remapped through the original-particle ids (which survive the
// renumbering) or dropped, and anything derived from the old embedding is refetched.
async function reloadLandscape(oldRows) {
  // A morph in flight walks state indices that are about to mean something else.
  setPlaying(false);
  state.frame = 0;
  const meta = await (await fetch("./api/landscape")).json();
  Object.assign(L, meta);
  L.pca = await fetchMatrix("./api/pca.bin");
  L.rows = await fetchLabels("./api/rows.bin");
  L.latents = null;                 // refetched on demand, like the initial load
  umapData = null;                  // a new UMAP graph; the old embedding is meaningless
  for (const k of Object.keys(clusterCache)) delete clusterCache[k];
  clusterCache[5] = await fetchLabels("./api/cluster.bin?k=5");
  zData = null; zInflight = null;   // z-scores are re-derived from the states that survived
  invalidateDensity();
  clearFrameCache();                // cached volumes were keyed on the old numbering
  centroidVecs = {}; indexColor = {}; clusterQuery = null;

  // old state index -> new one, via the original particle each one is. Layers and the
  // morph path are real work; losing them to a renumbering would make filtering something
  // you only dare do first, rather than whenever the picture warrants it.
  const pos = new Int32Array(L.n_all).fill(-1);
  for (let j = 0; j < L.rows.length; j++) pos[L.rows[j]] = j;
  const remap = (i) => (i >= 0 && i < oldRows.length ? pos[oldRows[i]] : -1);

  // Curve layers store control points in the *old* PCA coordinates. The basis just
  // changed, so those points no longer mean anything — they are dropped rather than
  // silently redrawn somewhere they were never put. Centroids are regenerated by
  // clustering, so they go too.
  const dropped = layers.filter((ly) => ly.kind === "curve" || ly.kind === "centroid").length;
  for (let i = layers.length - 1; i >= 0; i--) {
    const ly = layers[i];
    if (ly.kind === "curve" || ly.kind === "centroid") { layers.splice(i, 1); continue; }
    ly.indices = ly.indices.map(remap).filter((v) => v >= 0);
    if (!ly.indices.length) layers.splice(i, 1);
  }
  editingPathLayer = null; editCurve = null;
  state.path = state.path.map(remap).filter((v) => v >= 0);
  setSelectionState(state.selection.map(remap).filter((v) => v >= 0));
  state.current = remap(state.current);

  populateAxes();
  refreshAxisEnabled();
  renderLayers();
  updatePoints();
  await applyColor();
  syncSelection();
  updatePathLabel();
  // Rebuilt, not just emptied: buildTimeline is what owns the scrub range and the frame
  // counter, so clearing `state.timeline` behind its back left the morph bar advertising
  // a length the timeline no longer had. It also re-prices the hops in the new space.
  await buildTimeline();
  updateMetaLabel();
  if (landView === "slices") loadLandSlices();
  if (state.current >= 0) onSelect(state.current, {});
  else if (L.n > 0) { scatter.select(0); onSelect(0, {}); }
  if (dropped) status(`${dropped} curve/centroid layer(s) dropped — they were drawn in the old projection.`, "warn");
}

// "4,000 states · latent dim 10", or what is left of them once a filter is on.
function updateMetaLabel() {
  const base = L.filtered ? `${L.n} of ${L.n_all} states` : `${L.n} states`;
  $("meta").textContent = `${base} · latent dim ${L.dims}`;
  $("outBtnState").textContent = L.filtered ? ` · ${L.n_all - L.n} cut` : "";
  $("outBtn").classList.toggle("filtered", !!L.filtered);
  $("zReset").style.display = L.filtered ? "" : "none";
}

// ── colour bar ────────────────────────────────────────────────────────────
// Drawn onto the landscape rather than into the chrome: it costs no toolbar space, and
// it lands in Save PNG, so an exported landscape is a figure with a scale on it.
const densityBarVisible = () =>
  state.colorBy === "density" && is3d() && state.densCbar && landView !== "slices";

let densBarSig = "";
function drawDensityBar() {
  const cv = $("densBar");
  if (!cv) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = cv.clientWidth, h = cv.clientHeight;
  const on = densityBarVisible();
  const sig = [on, w, h, dpr, state.densStyle, state.densMap, state.densLevels.join(",")].join("|");
  if (sig === densBarSig) return;      // it does not move with the camera, so do not repaint per frame
  densBarSig = sig;
  const dw = Math.max(1, Math.floor(w * dpr)), dh = Math.max(1, Math.floor(h * dpr));
  if (cv.width !== dw || cv.height !== dh) { cv.width = dw; cv.height = dh; }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (on) paintDensityBar(ctx, w, h, 1);
}

const BAR_MONO = 'ui-monospace, "SF Mono", Menlo, Consolas, monospace';
function paintDensityBar(ctx, w, h, s) {
  const d = densStyleDef(state.densStyle);
  const x = 12 * s, yBottom = h - 12 * s;
  const plate = (px, py, pw, ph) => {
    ctx.fillStyle = "rgba(6,9,15,0.55)"; ctx.strokeStyle = "rgba(255,255,255,0.10)"; ctx.lineWidth = s;
    const r = 4 * s;
    ctx.beginPath();
    ctx.moveTo(px + r, py);
    ctx.arcTo(px + pw, py, px + pw, py + ph, r);
    ctx.arcTo(px + pw, py + ph, px, py + ph, r);
    ctx.arcTo(px, py + ph, px, py, r);
    ctx.arcTo(px, py, px + pw, py, r);
    ctx.closePath(); ctx.fill(); ctx.stroke();
  };
  ctx.save();
  ctx.textBaseline = "alphabetic"; ctx.textAlign = "left";

  if (state.densStyle === "shells") {
    const bands = shellBands(state.densLevels);
    const rowH = 13 * s, pad = 7 * s;
    const boxH = bands.length * rowH + pad * 2 + 12 * s, boxW = 124 * s;
    plate(x, yBottom - boxH, boxW, boxH);
    ctx.font = `500 ${9 * s}px ${BAR_MONO}`;
    ctx.fillStyle = "rgba(226,232,240,0.72)";
    ctx.fillText(d.unit.toUpperCase(), x + pad, yBottom - boxH + pad + 8 * s);
    bands.forEach((b, i) => {
      const y = yBottom - boxH + pad + 14 * s + i * rowH;
      ctx.fillStyle = `rgb(${b.color.join(",")})`;
      ctx.fillRect(x + pad, y, 10 * s, 10 * s);
      ctx.strokeStyle = "rgba(255,255,255,0.28)"; ctx.lineWidth = s;
      ctx.strokeRect(x + pad + s / 2, y + s / 2, 10 * s - s, 10 * s - s);
      ctx.fillStyle = "rgba(226,232,240,0.92)";
      ctx.font = `500 ${10 * s}px ${BAR_MONO}`;
      ctx.fillText(b.label, x + pad + 16 * s, y + 9 * s);
    });
    ctx.restore();
    return;
  }

  const barW = 128 * s, barH = 8 * s, pad = 8 * s;
  const boxW = barW + pad * 2, boxH = barH + pad * 2 + 24 * s;
  plate(x, yBottom - boxH, boxW, boxH);
  const stops = state.densStyle === "points"
    ? [[74, 87, 107], [242, 247, 255]]
    : (COLORMAPS[state.densMap] || COLORMAPS.inferno);
  const bx = x + pad, by = yBottom - boxH + pad + 12 * s;
  ctx.font = `500 ${9 * s}px ${BAR_MONO}`;
  ctx.fillStyle = "rgba(226,232,240,0.72)";
  ctx.fillText(d.unit.toUpperCase(), bx, by - 4 * s);
  const step = Math.max(1, s);
  for (let i = 0; i < barW; i += step) {
    const rgb = rampRGB(stops, i / (barW - step));
    ctx.fillStyle = `rgb(${rgb.join(",")})`;
    ctx.fillRect(bx + i, by, step + 0.5, barH);
  }
  ctx.strokeStyle = "rgba(255,255,255,0.3)"; ctx.lineWidth = s;
  ctx.strokeRect(bx + s / 2, by + s / 2, barW - s, barH - s);
  const ticks = state.densStyle === "energy"
    ? [DENS_KT.toFixed(1), (DENS_KT / 2).toFixed(1), "0"]
    : ["0", "0.5", "1"];
  ctx.font = `500 ${9.5 * s}px ${BAR_MONO}`;
  ctx.fillStyle = "rgba(226,232,240,0.92)";
  ticks.forEach((t, i) => {
    const tx = bx + (barW - step) * (i / (ticks.length - 1));
    ctx.textAlign = i === 0 ? "left" : i === ticks.length - 1 ? "right" : "center";
    ctx.fillText(t, tx, by + barH + 11 * s);
  });
  ctx.restore();
}

// Everything the renderer needs that is not per-point data.
function pushDensityParams() {
  if (!cloud) return;
  const style = state.densStyle === "volume" && !cloud.hasVolumeSupport() ? "splats" : state.densStyle;
  cloud.setDensityStyle(style);
  cloud.setDensityIntensity(state.densIntensity);
  cloud.setDensityGamma(state.densGamma);
  cloud.setDensityKernel(0.055 + state.densBandwidth * 0.009);
  cloud.setDensityShade(state.densShade);
  cloud.setDensityFog(state.densFog);
  const bands = shellBands(state.densLevels);
  cloud.setDensityLevels(state.densLevels, bands.map((b) => b.color), bands.map((b) => b.alpha));
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

// Draw the visible Centroids layer's points as ringed markers. Centroids are virtual
// states (index >= L.n) that live off the base cloud, so they are never part of the
// scatter/cloud point set — we project their stored coords onto the active view and
// stamp them on a dedicated overlay, re-synced after every render (onRender) so the
// markers track panning and 3D orbit.
function syncCentroidOverlay() {
  drawDensityBar();   // cheap: it repaints only when the box or the scale actually changed
  if (!centroidCtx) return;
  const co = $("centroidOverlay");
  sizeOverlays();   // the box may have changed since the last draw — see sizeOverlays
  centroidCtx.clearRect(0, 0, co.clientWidth, co.clientHeight);
  if (landView === "slices") return;   // the scatter is hidden behind the slice grid
  const cen = layers.find((l) => l.kind === "centroid" && l.visible);
  if (!cen || !cen.indices) return;
  const land = activeLand();
  if (!land || !land.projectData) return;
  const cc = centroidCoords();
  const dataOf = (i) => axesOf(i < L.n ? (activeCoords() || L.pca)[i] : (cc[i] || []));

  // Morph path that runs through centroids: the base renderers can't draw segments to a
  // virtual point, so we draw the whole trajectory here, colored to match the centroid.
  const traj = hidePathPicks ? [] : [state.current, ...state.path].filter((v) => v >= 0);
  if (traj.length > 1 && traj.some((i) => i >= L.n)) {
    const tp = land.projectData(traj.map(dataOf));
    const pcol = editingPathLayer ? editingPathLayer.color : (indexColor[state.current] || "#f472b6");
    centroidCtx.lineJoin = "round"; centroidCtx.lineCap = "round";
    const stroke = (color, w) => {
      centroidCtx.strokeStyle = color; centroidCtx.lineWidth = w;
      centroidCtx.beginPath(); let started = false;
      for (const p of tp) { if (!p) { started = false; continue; } if (!started) { centroidCtx.moveTo(p[0], p[1]); started = true; } else centroidCtx.lineTo(p[0], p[1]); }
      centroidCtx.stroke();
    };
    stroke("rgba(15,23,42,0.5)", 4.5);   // dark halo so the line reads on light backgrounds
    stroke(pcol, 2.5);
  }

  // Centroid markers, the current one highlighted (its ring, since the base cloud can't).
  const idxs = cen.indices.filter((i) => cc[i]);
  const proj = land.projectData(idxs.map((i) => axesOf(cc[i])));
  for (let k = 0; k < proj.length; k++) {
    const p = proj[k];
    if (p) drawCentroidMarker(centroidCtx, p[0], p[1], indexColor[idxs[k]] || cen.color || "#ffffff", idxs[k] === state.current);
  }
}
function drawCentroidMarker(ctx, x, y, hex, current) {
  const rad = current ? 8 : 6;
  ctx.beginPath(); ctx.arc(x, y, rad, 0, Math.PI * 2); ctx.fillStyle = hex; ctx.fill();
  ctx.lineWidth = 3; ctx.strokeStyle = "rgba(15,23,42,0.85)"; ctx.stroke();     // dark halo reads on light bg
  ctx.beginPath(); ctx.arc(x, y, rad, 0, Math.PI * 2); ctx.lineWidth = 1.5; ctx.strokeStyle = "rgba(255,255,255,0.95)"; ctx.stroke();
  if (current) { ctx.beginPath(); ctx.arc(x, y, rad + 4, 0, Math.PI * 2); ctx.lineWidth = 2; ctx.strokeStyle = "#2dd4bf"; ctx.stroke(); }   // selection halo
}

// One kernel-density estimate, shared by everything that needs it.
//
// Binned: drop the points into a uniform grid, blur it with three box passes (a gaussian
// to within a few percent), then read each point back out of its own cell. O(N + G)
// rather than the O(N^2) all-pairs count it replaces — that scan was ~1e12 distance tests
// at a million points and simply hung the tab.
//
// It used to run twice at two resolutions: once at 64^3 to tint the points and drive the
// glow, once at 96^3 for the Slices view. One estimate at one resolution now serves the
// point tint, every 3D density style, the ray-marched volume and the slices — which is
// what makes the shell / free-energy / volumetric styles nearly free.
//
// Returns, all keyed to the current embedding, axes and bandwidth:
//   values     per point, density normalised to its own max
//   quantiles  per point, the share of particles inside the smallest highest-density
//              region containing it — a statement about the data, not about the colormap
//   normals    per point, -grad(rho) normalised, so splats can be lit
//   grid       { nx, ny, nz, data (uint8, x fastest), lo, hi } for slices and the volume
const DENS_GRID = 96;          // cells along the longest axis, for an isotropic cloud
const DENS_GRID_MAX = 256;     // ...raised for anisotropic ones, which are far cheaper per axis.
                               // 256 is WebGL2's guaranteed MAX_3D_TEXTURE_SIZE, so the grid can
                               // always be uploaded as the ray-march's volume texture.
const DENS_CELL_BUDGET = 1e6;  // ceiling on total cells (~8 MB of ping-pong float grids)
const DENS_QBINS = 4096;       // histogram bins for the quantile pass (O(N), not O(N log N))

let densCache = null;
const densCacheKey = () => [state.embedding, state.x, state.y, state.z, state.densBandwidth,
  (activeCoords() || []).length].join("|");
const invalidateDensity = () => { densCache = null; };

function landscapeDensity() {
  const key = densCacheKey();
  if (densCache && densCache.key === key) return densCache;
  const c = activeCoords();
  if (!c || !c.length) return null;

  const n = c.length, ax = [state.x, state.y, state.z];
  const mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
  const P = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) { const v = c[i][ax[k]] ?? 0; P[i * 3 + k] = v; if (v < mn[k]) mn[k] = v; if (v > mx[k]) mx[k] = v; }
  const ext = [0, 1, 2].map((k) => (mx[k] - mn[k]) || 1);
  const maxExt = Math.max(ext[0], ext[1], ext[2]);

  // Cells stay cubic: both the slice view and the ray-march read this grid as geometry,
  // and a stretched voxel would distort them. But a strongly anisotropic cloud fills only
  // a sliver of the G^3 box — a latent space with PC1 at 98% of the variance uses well
  // under 1% of it — so it can afford a far finer grid than an isotropic one for the same
  // memory. Step G up while the cell count stays inside the budget; that is what gives the
  // minor axes enough cells to carry any structure at all.
  const cellsAt = (g) => {
    const cell = maxExt / g;
    return ext.reduce((count, e) => count * Math.max(1, Math.min(g, Math.ceil(e / cell))), 1);
  };
  // Finest cubic grid that fits the budget, never coarser than the isotropic default.
  let G = DENS_GRID;
  for (let g = DENS_GRID_MAX; g > DENS_GRID; g -= 32) {
    if (cellsAt(g) <= DENS_CELL_BUDGET) { G = g; break; }
  }

  const h = maxExt / G;
  const dim = ext.map((e) => Math.max(1, Math.min(G, Math.ceil(e / h))));
  const [nx, ny, nz] = dim;
  const grid = new Float32Array(nx * ny * nz);   // (ix*ny+iy)*nz+iz
  const cellOf = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const ix = Math.min(nx - 1, Math.max(0, (P[i * 3] - mn[0]) / h | 0));
    const iy = Math.min(ny - 1, Math.max(0, (P[i * 3 + 1] - mn[1]) / h | 0));
    const iz = Math.min(nz - 1, Math.max(0, (P[i * 3 + 2] - mn[2]) / h | 0));
    const cell = (ix * ny + iy) * nz + iz;
    cellOf[i] = cell; grid[cell] += 1;
  }

  // Bandwidth per axis, each scaled to that axis's own spread rather than to the bounding
  // box diagonal. One radius taken from the diagonal is an isotropic kernel, and in a
  // latent space where PC1 carries 98% of the variance it is wider than the PC2 and PC3
  // extents put together — the estimate then goes genuinely flat on those axes, which is
  // why the ray-marched volume used to render as a featureless slab. Scaling per axis is
  // also the standard choice: a diagonal bandwidth matrix matched to each dimension.
  const frac = 0.04 + state.densBandwidth * 0.02;
  const rads = ext.map((e) => Math.max(1, Math.round((frac * e) / h / 2)));

  // Three separable box passes. Each is a running prefix sum, so the cost does not grow
  // with the radius.
  let src = grid, dst = new Float32Array(grid.length);
  for (let pass = 0; pass < 3; pass++) {
    for (let axis = 0; axis < 3; axis++) { boxBlurAxis(src, dst, dim, axis, rads[axis]); const t = src; src = dst; dst = t; }
  }

  let max = 0;
  const values = new Float32Array(n);
  for (let i = 0; i < n; i++) { const v = src[cellOf[i]]; values[i] = v; if (v > max) max = v; }
  if (max > 0) for (let i = 0; i < n; i++) values[i] /= max;

  // Quantiles by histogram rather than by sorting: bin the densities, take the suffix sum,
  // and a point's quantile is the share of points binned at or above it. O(N + B) instead
  // of O(N log N), which at a million points is the difference between instant and a stall.
  const hist = new Int32Array(DENS_QBINS);
  const binOf = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const b = Math.min(DENS_QBINS - 1, (values[i] * (DENS_QBINS - 1)) | 0);
    binOf[i] = b; hist[b]++;
  }
  let acc = 0;
  const above = new Float32Array(DENS_QBINS);
  for (let b = DENS_QBINS - 1; b >= 0; b--) { acc += hist[b]; above[b] = acc / n; }
  const quantiles = new Float32Array(n);
  for (let i = 0; i < n; i++) quantiles[i] = above[binOf[i]];

  // Surface normal per point: -grad(rho) by central differences on the blurred grid, with
  // clamped borders. This is what lets a cloud of splats be lit instead of merely tinted.
  const sx = ny * nz, sy = nz;
  const normals = new Float32Array(n * 3);
  const cl = (v, hi) => (v < 0 ? 0 : v >= hi ? hi - 1 : v);
  for (let i = 0; i < n; i++) {
    const cell = cellOf[i];
    const iz = cell % nz, iy = ((cell - iz) / nz) % ny, ix = ((cell - iz) / nz - iy) / ny;
    const at = (jx, jy, jz) => src[cl(jx, nx) * sx + cl(jy, ny) * sy + cl(jz, nz)];
    const gx = at(ix + 1, iy, iz) - at(ix - 1, iy, iz);
    const gy = at(ix, iy + 1, iz) - at(ix, iy - 1, iz);
    const gz = at(ix, iy, iz + 1) - at(ix, iy, iz - 1);
    const len = Math.hypot(gx, gy, gz);
    if (len > 1e-9) { normals[i * 3] = -gx / len; normals[i * 3 + 1] = -gy / len; normals[i * 3 + 2] = -gz / len; }
  }

  // The slice viewer and the 3D texture both want x fastest; boxBlurAxis works on the
  // (ix*ny+iy)*nz+iz layout, so remap on the way out.
  const inv = max > 0 ? 255 / max : 0;
  const data = new Uint8Array(nx * ny * nz);
  for (let ix = 0; ix < nx; ix++) for (let iy = 0; iy < ny; iy++) for (let iz = 0; iz < nz; iz++)
    data[(iz * ny + iy) * nx + ix] = src[(ix * ny + iy) * nz + iz] * inv;

  densCache = {
    key, values, quantiles, normals,
    grid: { nx, ny, nz, data, lo: [mn[0], mn[1], mn[2]], hi: [mn[0] + nx * h, mn[1] + ny * h, mn[2] + nz * h] },
  };
  return densCache;
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
  // The path takes the current centroid's cluster color (so a morph built from centroids
  // reads in their color, matching the decoded map). A trajectory that touches a centroid
  // is drawn by the centroid overlay instead — the base renderers can't reach virtual
  // points — so hand them an empty path to avoid a broken half-line.
  const col = editingPathLayer ? editingPathLayer.color : (indexColor[state.current] || "#f472b6");
  const baseTraj = traj.some((i) => i >= L.n) ? [] : traj;
  scatter.select(state.current); scatter.setPath(pathPts); scatter.setSelection(state.selection);
  scatter.setPathOrder(baseTraj, col);
  if (cloud) { cloud.select(state.current); cloud.setPath(pathPts); cloud.setSelection(state.selection); cloud.setPathOrder(baseTraj, hexToRgb(col)); }
  syncCentroidOverlay();   // path through / marker on centroids follows the selection
}

async function onSelect(i, opts) {
  opts = opts || {};
  // A real click/tap carries coords: resolve it into a base point or a centroid marker,
  // honoring pick-restrict. Programmatic calls (layer loading, path edits) pass no coords
  // and select `i` directly.
  if (opts.cx != null) {
    i = resolvePick(i, opts.cx, opts.cy);
    if (i < 0) return;
  }
  if (playing) setPlaying(false);              // editing the path stops playback
  hidePathPicks = false;                       // manual interaction -> show path picks again
  if (opts.shift) {
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

function onLevel(level, dragging) {
  state.sigma = hist.sigma;
  if (state.render === "slices") return;   // slices show raw density, not a contour
  if (state.render !== "surface" && volRenderer && curVolMeta) { volRenderer.setThreshold(normThreshold()); return; }
  // With a morph on screen the contour has to move the *reference* the frames are contoured
  // to, not the frames — see recontourSeries. Re-decoding the selected state here (what this
  // used to do, unconditionally) painted a static map over the running animation, and the
  // frames themselves never read sigma at all, so the slider appeared to do nothing.
  if (state.timeline.length && (playing || state.frame > 0)) { liveSeriesContour(!!dragging); return; }
  liveDecode(!!dragging);
}

// Re-aim a morph's contour.
//
// A morph is contoured at *constant enclosed mass*, not at sigma: as the latent moves, the
// decoded density's mean and std drift, so holding sigma fixed would make the map appear to
// breathe and would ruin the shape comparison. `state.refMass` is that reference — the mass
// the selected state encloses at the chosen sigma — and every frame is then contoured to
// enclose it. Which is precisely why dragging the contour did nothing during a morph: the
// frames are contoured to a mass, and the mass never moved.
//
// So the slider moves the reference. Every cached frame's contour changes with it, hence the
// cache flush; and clearFrameCache() cancels the running prefetcher (it bumps fillToken), so
// it has to be restarted or playback would fall back to decoding on demand.
async function recontourSeries(draft) {
  if (state.current < 0) return;
  try {
    const res = await fetch(`./api/mass?index=${state.current}`
                          + `&sigma=${state.sigma.toFixed(3)}&size=${PLAY_GRID}`);
    if (!res.ok) return;
    const { mass } = await res.json();
    if (!(mass > 0)) return;
    state.refMass = mass;
    clearFrameCache();                  // every frame's contour — and so its cache key — moved
    if (playing) prefetch(true);        // ...and the flush stopped the prefetcher
    else scheduleWarm();                // paused: refill in the background at the new contour
    await renderFrame(state.frame, playing ? true : draft);
  } catch { /* a superseded drag frame: the next pointermove supersedes it anyway */ }
}
const liveSeriesContour = makeLive((draft) => recontourSeries(draft));

// makeLive comes from viewerbase.js (single-flight, waits rather than aborts —
// see the note there); the draft resolutions below are what make coalescing cheap.
const liveDecode = makeLive((draft) => decode(draft));
const liveFrame = makeLive((i, draft) => renderFrame(i, draft));

// Draft resolutions. Rather than mesh the full grid, ask the server for a strided one; it
// scales the result by the stride, so a draft occupies the same [-1,1] cube as the full
// surface and refines in place instead of jumping. Boxes at or below these sizes are not
// strided at all, so a small map is unaffected.
//
// Two sizes, because the two jobs want different things. A contour drag ends on a still
// you will look at, so it stays fairly fine. Playback needs a *new* frame every 33 ms —
// and a morph frame is a fresh latent, so unlike a contour it can never be served from a
// cache — so it trades detail for rate, which motion hides anyway. Measured on a 320³ box:
//   full 470 ms (2 fps) · 160³ 99 ms (10 fps) · 107³ 60 ms (17 fps) · 80³ 46 ms (22 fps)
const DRAFT_GRID = 160;   // contour drag / scrub
const PLAY_GRID = 128;    // morph playback (-> a 107³ grid at a 320³ box)

// Drafts are smoothed too, lightly. They used to be sent raw, which was backwards: a
// coarse grid facets *more*, so the draft is exactly where relaxing the surface earns its
// keep — and leaving it off meant the map visibly changed character when you released the
// slider or paused.
//
// The cap is not a speed compromise. Almost none of the cost of Taubin is the iterating:
// building the mesh's adjacency and recomputing the normals afterwards are O(edges) and
// paid once for any k > 0, while each iteration is two sparse matvecs (~1 ms on a draft).
// Measured on a 320³ box: smooth=0 -> 66 ms, smooth=1 -> 81 ms, smooth=6 -> 85 ms. So one
// tick costs essentially what six do; the choice is on-or-off. The cap is about *looks*:
// Taubin relaxes vertices by a fixed amount in mesh units, and a draft's triangles are
// bigger, so the same k smooths a draft harder than the full surface. Two keeps the draft
// looking like what you get when you let go.
const DRAFT_SMOOTH = 2;
const draftSmooth = () => Math.min(state.smooth, DRAFT_SMOOTH);

let decodedIndex = -1;    // the state whose volume the server has already decoded

async function decode(draft = false) {
  if (state.render === "slices") { loadSlices(); return; }   // slices ignore contour/mesh
  if (state.current < 0) return;
  const token = ++decodeToken;
  // Only a change of *state* runs the network. A contour change re-meshes a volume the
  // server already holds (Decoder.volume caches it), so say what is really happening.
  const fresh = state.current !== decodedIndex;
  status(fresh ? "Decoding map…" : "Contouring…", "busy");
  try {
    const q = `index=${state.current}&sigma=${state.sigma.toFixed(3)}`
            + (draft ? `&smooth=${draftSmooth()}&size=${DRAFT_GRID}` : `&smooth=${state.smooth}`);
    const res = await fetch(`./api/decode?${q}`);
    if (!res.ok) throw new Error(await res.text());
    const m = parseMesh(await res.arrayBuffer());
    if (token !== decodeToken) return;
    decodedIndex = state.current;
    // A draft's mass is only an estimate (it is measured on the strided grid), and refMass
    // is what constant-mass morphs are keyed off — so only ever take it from a full mesh.
    if (!draft) state.refMass = m.mass;
    renderer.setMesh(m.positions, m.normals, m.indices);
    renderer.setColor(colorForIndex(state.current));   // centroids carry their cluster color
    if (state.render !== "surface") { loadVolume(state.current); return; }
    if (!m.indices.length) { renderer.clear(); status("No surface at this contour — lower the threshold.", "warn"); return; }
    const tris = (m.indices.length / 3).toLocaleString();
    status(draft ? `Contouring · σ ${state.sigma.toFixed(1)} · draft`
                 : `${tris} triangles · σ ${state.sigma.toFixed(1)} · level ${m.level.toFixed(3)}`);
  } catch (err) {
    if (token === decodeToken) status("Decode failed: " + err.message, "err");
  }
}

// ---- volume ray-marching / orthogonal-slice mode ----
function setRenderMode(mode) {
  if (mode !== state.render) clearFrameCache();   // surface<->volume payloads differ
  const surf = mode === "surface";
  const isSlices = mode === "slices";
  // Three volume styles now share the ray-marcher canvas: MIP, glow and the shaded
  // isosurface. Only the style passed to volRenderer.setMode tells them apart.
  const isVol = mode === "mip" || mode === "emission" || mode === "iso";
  // Surface and the ray-marcher live on different canvases, each with its own camera;
  // the slice view has none. Hand the outgoing 3D camera over so the map does not jump
  // back to a default view when returning from slices to a 3D mode.
  const from = state.render === "surface" ? renderer : state.render === "slices" ? null : volRenderer;
  const camera = from ? from.getCamera() : null;
  state.render = mode; $("rendermode").value = mode;
  $("gl").style.display = surf ? "" : "none";
  $("glvol").style.display = isVol ? "" : "none";
  $("slices").style.display = isSlices ? "grid" : "none";
  $("lightlbl").style.display = surf ? "" : "none";
  $("smoothlbl").style.display = surf ? "" : "none";
  $("volctrls").style.display = isVol ? "" : "none";
  $("slicectrls").style.display = isSlices ? "" : "none";
  $("colorlbl").style.display = isSlices ? "none" : "";
  $("histStrip").style.display = isSlices ? "none" : "";   // contour is meaningless for raw slices
  if (isVol && !volRenderer) {
    try { volRenderer = createVolume($("glvol")); volRenderer.setColor(hexToRgb($("color").value)); volRenderer.setBackground(hexToRgb($("bg").value)); volRenderer.setGamma(state.vGamma); volRenderer.setBrightness(state.vBright); }
    catch (err) { status("Volume rendering unavailable: " + err.message, "err"); setRenderMode("surface"); return; }
  }
  if (isSlices && !slices) {
    try {
      slices = createSlices($("slices"));
      slices.setBackground($("bg").value); slices.setColormap($("slicemap").value);
      slices.setContrast(+$("slicelo").value, +$("slicehi").value); slices.setCrosshair($("slicecross").checked);
    } catch (err) { status("Slices unavailable: " + err.message, "err"); setRenderMode("surface"); return; }
  }
  if (isVol) volRenderer.setMode(mode);   // iso keeps volume.js's default "standard" lighting: this viewer has no lighting control for it
  const to = surf ? renderer : isVol ? volRenderer : null;
  if (to && to !== from && camera) to.setCamera(camera);
  if (isSlices) { refreshSlices(); return; }
  if (state.timeline.length && state.frame > 0) renderFrame(state.frame, false);
  else if (state.current >= 0) decode();
}

// ---- orthogonal slices of the decoded map (right pane) ----
// Land on the active morph frame when one is loaded (so a paused/looping morph keeps
// its plane), otherwise slice the currently selected state.
function refreshSlices() {
  if (state.timeline.length && state.frame > 0) renderFrame(state.frame, false);
  else loadSlices();
}
async function loadSlices() {
  if (!slices) return;
  if (state.current < 0) { status("Pick a state to slice its decoded map.", "warn"); return; }
  const token = ++decodeToken;
  status("Loading slices…", "busy");
  try {
    const res = await fetch(`./api/grid?index=${state.current}&size=160`);
    if (!res.ok) throw new Error(await res.text());
    const g = parseGrid(await res.arrayBuffer());
    if (token !== decodeToken) return;
    slices.setVolume(g);
    status(`Slices ${g.nx}×${g.ny}×${g.nz} · state #${state.current}`);
  } catch (err) { if (token === decodeToken) status("Slice load failed: " + err.message, "err"); }
}

// ---- landscape orthogonal-slice view (left pane) ----
// Slices with a gray-ish default; the slice viewer only knows a handful of maps, so map
// the density colormap onto the nearest one it supports (else viridis).
const SLICE_MAPS = new Set(["gray", "viridis", "magma", "inferno", "hot"]);
const sliceColormap = () => (SLICE_MAPS.has(state.densMap) ? state.densMap : "viridis");

// Swap the landscape pane between the point cloud and an orthogonal-slice density map.
function setLandView(mode) {
  landView = mode === "slices" ? "slices" : "points";
  const on = landView === "slices";
  $("landSlicesBtn").classList.toggle("active", on);
  const is3 = is3d();
  $("scatter").style.display = on ? "none" : (is3 ? "none" : "");
  $("scatter3d").style.display = on ? "none" : (is3 ? "" : "none");
  $("centroidOverlay").style.display = on ? "none" : "";
  $("lassoCanvas").style.display = on ? "none" : "";
  $("landSlices").style.display = on ? "grid" : "none";
  if (on) {
    if (lassoArmed) setLassoArmed(false);        // the point tools have no meaning over slices
    if (pathDrawArmed) setPathDrawArmed(false);
    if (!landSlices) {
      try {
        landSlices = createSlices($("landSlices"));
        landSlices.setBackground($("bg").value); landSlices.setColormap(sliceColormap()); landSlices.setCrosshair(true);
      } catch (err) { status("Slices unavailable: " + err.message, "err"); setLandView("points"); return; }
    }
    loadLandSlices();
  } else {
    syncCentroidOverlay();
  }
  drawDensityBar();   // the slice grid carries its own scale; the cloud's bar does not belong on it
}

// Bin the landscape density onto the current X/Y/Z axes and show it as orthogonal cuts.
function loadLandSlices() {
  if (!landSlices) return;
  const d = landscapeDensity();
  if (!d) { status("No points to build a density field.", "warn"); return; }
  const g = d.grid;
  landSlices.setColormap(sliceColormap());
  landSlices.setVolume(g);
  status(`Landscape density slices ${g.nx}×${g.ny}×${g.nz} · ${axisLabel(state.x)} · ${axisLabel(state.y)} · ${axisLabel(state.z)}`);
}

// Grid resolution for the ray-marcher, per style: an isosurface shows every voxel of the
// grid it was cut from, and 96³ — smooth enough for MIP and glow, which integrate along the
// ray — comes out blobby. Volume payloads are cached by URL, so the size is part of the key.
// The /api/grid query for the current render style. MIP, glow and the slice frames keep the
// 96³ uint8 payload they always had. Iso shades the grid rather than integrating along it,
// so it asks for float16 at the decoder's native box where the renderer can hold it (the
// backend clamps `size` to the box, so 384 means "native, no downsampling"), and falls back
// to 256³ uint8 + the de-banding blur where it cannot. The query string is part of the frame
// cache key, so a style switch never reuses the wrong payload.
const ISO_GRID_U8 = 256, ISO_MAX = 384;
const isoF16 = () => !!(volRenderer && volRenderer.hasF16);
function volQuery() {
  if (state.render !== "iso") return "&size=96";
  return isoF16() ? `&size=${ISO_MAX}&bits=16` : `&size=${ISO_GRID_U8}`;
}

// The blur exists only to hide 8-bit banding, so a float16 grid must never see it: it would
// throw away exactly the fine features 16 bits were fetched for. Tagged so a cached frame
// payload is smoothed once and only once.
function isoSmooth(g) {
  if (state.render === "iso" && !g.f16 && !g.smoothed) {
    g.data = smoothGrid(g.data, g.nx, g.ny, g.nz);
    g.smoothed = true;
  }
  return g;
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
    const res = await fetch(`./api/grid?index=${index}${volQuery()}`);
    if (!res.ok) throw new Error(await res.text());
    const g = isoSmooth(parseGrid(await res.arrayBuffer()));
    if (token !== decodeToken) return;
    curVolMeta = g;
    volRenderer.setVolume(g.nx, g.ny, g.nz, g.data, g.boxLo, g.boxHi);
    volRenderer.setThreshold(normThreshold());
    volRenderer.setColor(colorForIndex(index));
    status(`Volume ${g.nx}×${g.ny}×${g.nz} · ${state.render === "emission" ? "glow" : state.render === "iso" ? "shaded iso" : "MIP"}`);
  } catch (err) { if (token === decodeToken) status("Volume load failed: " + err.message, "err"); }
}

// ---- morph / volume series as a scrubbable timeline ----
// Build the ordered list of [a,b,t] frames for the current path (forward then back,
// so playback loops cleanly and the scrubber covers a full there-and-back). Cheap —
// no network — so it's safe to call on every path edit.
// A morph's cost is (path nodes − 1) × frames-per-segment — the size of the landscape has
// nothing to do with it. Two limits keep that from running away, because a series taken
// from a cluster layer can hand us hundreds or thousands of states:
//
// MAX_TIMELINE bounds the unique frames, so that a morph is slow once and then plays at the
// full display rate for ever, rather than slow for ever (never fully cached, so every cycle
// re-decodes whatever was evicted: see the eviction note above).
//
// It is a frame *count*, though, and the budget it is meant to respect is in *bytes* — so it
// cannot actually guarantee the fit, and this comment used to claim it did ("~1.5 MB per
// draft mesh, 360 frames is ~540 MB"). A draft mesh costs roughly 180 bytes per cell² of the
// *strided* grid, and that grid is a sawtooth in the model's box (PLAY_GRID divides it, so
// the stride jumps): ~1.2 MB/frame at a 160 box, ~2.1 MB at 320, ~3.0 MB at 256 or 384. So
// 360 frames is ~400 MB at one box and ~1 GB at another, and only boxes with a strided grid
// of ≤111 cells fit. checkTimelineFits() measures the real figure once frames start landing
// and says so, rather than letting the morph quietly thrash for ever.
//
// MAX_NODES bounds the *waypoints*: every hop needs at least its endpoint, so a path with
// more nodes than this could not be drawn in MAX_TIMELINE frames at all. Over the limit we
// resample the node list at even arc length (keeping the ends) — a "tour of the tour".
const MAX_TIMELINE = 360;
const MAX_NODES = 120;

// Latent-space hop lengths, memoised on the path itself (rebuilt on every path edit).
let segCache = { key: "", lengths: null };
async function segmentLengths(path) {
  const key = path.join(",");
  if (segCache.key === key && segCache.lengths) return segCache.lengths;
  try {
    const res = await fetch("./api/segments", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ indices: path }),
    });
    if (!res.ok) throw new Error(await res.text());
    const { lengths } = await res.json();
    segCache = { key, lengths };
    return lengths;
  } catch {
    // Equal hops: exactly the old behaviour, so a failure here degrades rather than breaks.
    const lengths = path.slice(1).map(() => 1);
    segCache = { key, lengths };
    return lengths;
  }
}

// Split `total` frames across the hops in proportion to how far each travels, giving every
// hop at least `MIN_SEG` (its two endpoints) so no waypoint is skipped. Largest-remainder
// rounding, so the parts sum to `total` exactly.
const MIN_SEG = 2;
function allocate(lengths, total) {
  const n = lengths.length;
  const sum = lengths.reduce((a, b) => a + b, 0);
  const spare = Math.max(0, total - MIN_SEG * n);
  if (!(sum > 0)) return lengths.map(() => MIN_SEG + Math.floor(spare / n));
  const raw = lengths.map((L) => MIN_SEG + (spare * L) / sum);
  const out = raw.map(Math.floor);
  let left = total - out.reduce((a, b) => a + b, 0);
  const order = raw.map((v, i) => [i, v - Math.floor(v)]).sort((a, b) => b[1] - a[1]);
  for (let k = 0; left > 0; k++, left--) out[order[k % n][0]]++;
  return out;
}

// Resample a too-long node list at even arc length, always keeping the first and last.
function thinPath(path, lengths, keep) {
  const cum = [0];
  for (const L of lengths) cum.push(cum[cum.length - 1] + L);
  const total = cum[cum.length - 1];
  if (!(total > 0)) {                       // degenerate: fall back to even index spacing
    return Array.from({ length: keep }, (_, i) =>
      path[Math.round((i * (path.length - 1)) / (keep - 1))]);
  }
  const out = [];
  for (let i = 0; i < keep; i++) {
    const s = (total * i) / (keep - 1);
    let j = 0;
    while (j < cum.length - 1 && cum[j + 1] < s) j++;
    // Take the *nearer* of the hop's two endpoints. Taking its start (the obvious thing)
    // silently drops the final state: at the last sample s sits at the end of the last hop,
    // whose start is the second-to-last node — so the morph would never arrive.
    const near = (j + 1 < cum.length && Math.abs(cum[j + 1] - s) < Math.abs(cum[j] - s)) ? j + 1 : j;
    out.push(path[Math.min(near, path.length - 1)]);
  }
  return out.filter((v, i) => i === 0 || v !== out[i - 1]);   // collapse any duplicates
}

async function buildTimeline() {
  clearFrameCache();   // the frame list changed -> any cached frames are stale
  let path = trajectory();
  if (path.length < 2) { state.timeline = []; state.frame = 0; $("scrub").max = "0"; $("scrub").value = "0"; $("framelbl").textContent = "—"; return false; }

  let lengths = await segmentLengths(path);
  let note = "";
  if (path.length > MAX_NODES) {
    const before = path.length;
    path = thinPath(path, lengths, MAX_NODES);
    lengths = await segmentLengths(path);
    note = `${before} states thinned to ${path.length}`;
  }

  const perSeg = Math.max(2, Math.min(120, +$("frames").value || 36));
  const nominal = perSeg * (path.length - 1);
  const total = Math.min(nominal, MAX_TIMELINE);
  const counts = allocate(lengths, total);

  // Frames are placed by arc length, so the map advances at a constant speed through latent
  // space. Giving every hop the same count (what this used to do) made the morph lurch —
  // crawling through closely-spaced states and racing through distant ones — which is a
  // smoothness problem entirely separate from the frame rate, and it bites hardest on a
  // cluster series, whose representatives are unevenly spaced almost by construction.
  //
  // Each hop after the first starts at i=1: its i=0 is the previous hop's endpoint, the very
  // same latent, and emitting it again would decode the waypoint twice under two cache keys.
  const steps = [];
  for (let s = 0; s < path.length - 1; s++) {
    const a = path[s], b = path[s + 1], k = Math.max(2, counts[s]);
    for (let i = (s === 0 ? 0 : 1); i < k; i++) steps.push([a, b, i / (k - 1)]);
  }
  state.timeline = steps.concat(steps.slice(0, -1).reverse());   // there-and-back reuses the keys
  state.frame = Math.min(state.frame, state.timeline.length - 1);
  $("scrub").max = String(state.timeline.length - 1);
  $("scrub").value = String(state.frame);
  $("framelbl").textContent = `${state.frame + 1}/${state.timeline.length}`;
  // The status line gets overwritten the moment playback starts, so the explanation also
  // lives on the frame counter, where it stays.
  const capped = nominal > MAX_TIMELINE || note;
  $("framelbl").title = capped
    ? `${steps.length} unique frames over ${path.length} states`
      + (note ? ` · ${note}` : "")
      + (nominal > MAX_TIMELINE ? ` · capped from ${nominal}` : "")
      + " — bounded so the whole morph fits the frame cache and plays back smoothly."
    : `${steps.length} unique frames over ${path.length} states, spaced by latent-space distance.`;
  if (capped) status($("framelbl").title, "warn");
  scheduleWarm();     // decode it now, while nothing is waiting on it
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
const frameCache = new Map();      // key -> { surface, payload, bytes, idx }
const frameInflight = new Map();   // key -> Promise (single-flight per key)
let cacheBytes = 0;
let fillToken = 0;                  // bumped to cancel an in-flight prefetch
let warmTimer = null;               // pending pre-Play warm (see scheduleWarm)
let evictions = 0;                  // >0 means the timeline does not fit the budget

// Prefetch shape. The server serialises decodes at 2 slots, so 2 in flight saturates it;
// more would only make each one slower (they contend for CPU and memory bandwidth).
const LOOKAHEAD = 64;         // frames kept ready ahead of the playhead
const LOW_WATER = 8;          // below this the buffer is draining -> pace to the server
const FETCH_CONCURRENCY = 2;

// Rolling measure of what the server actually sustains, in ms per delivered frame. Used to
// pace the *first* pass: better to play smoothly at 12 fps than to stutter at a nominal 30.
//
// Only intervals during which we were actually decoding count. Averaging the whole window
// end-to-end (what this used to do) folds in every gap where the prefetcher had nothing to
// fetch, so the measured "rate" climbed without bound the longer playback ran — and since
// `pace = max(delay, rate)`, a morph that had gone quiet because it was fully cached ended
// up throttling itself to a number that described idleness, not the server. The prefetcher
// also clears the window outright when it finds the look-ahead already cached.
const doneAt = [];
const RATE_IDLE_GAP = 4000;   // ms; longer than any plausible draft frame => we were idle
function decodeRateMs() {
  if (doneAt.length < 3) return 0;
  let sum = 0, n = 0;
  for (let i = 1; i < doneAt.length; i++) {
    const dt = doneAt[i] - doneAt[i - 1];
    if (dt > RATE_IDLE_GAP) continue;
    sum += dt; n++;
  }
  return n ? sum / n : 0;
}

let fitWarned = false;   // one cache-overflow warning per timeline (see checkTimelineFits)

function clearFrameCache() {
  fillToken++;                     // cancel any running prefetch (its frames are stale)
  clearTimeout(warmTimer);         // ...and any warm still pending for the old timeline
  frameCache.clear(); frameInflight.clear(); cacheBytes = 0; doneAt.length = 0;
  fitWarned = false; evictions = 0;
}

// Once enough frames have landed to measure a real bytes/frame, check whether the whole
// timeline can actually stay resident, and say so once if it cannot. See the MAX_TIMELINE
// note: the cap is a frame count, the budget is bytes, and which of the two binds depends on
// the model's box. Without this the morph just never stops re-decoding and never explains why.
function checkTimelineFits() {
  if (fitWarned || frameCache.size < 8 || !state.timeline.length) return;
  // Count the cache keys the playhead will actually ask for, rather than assuming one per
  // timeline slot. Two things collapse that count, and both are easy to get wrong: the
  // there-and-back mirror means positions i and n-1-i are the same latent under the same
  // key, and at step > 1 the playhead skips slots entirely. A 361-slot morph at the default
  // 60 fps (step 2) is 91 decodes, not 361 and not 181 — assuming otherwise would warn about
  // an overflow four times bigger than the real one.
  const n = state.timeline.length;
  const { step } = playCadence();
  const keys = new Set();
  for (const i of upcoming(state.frame, step, Math.ceil(n / Math.max(1, step)))) {
    const req = frameRequest(i, true);
    if (req) keys.add(req.key);
  }
  const unique = keys.size || 1;
  const perFrame = cacheBytes / frameCache.size;
  const need = perFrame * unique;
  if (need <= CACHE_BUDGET) return;
  fitWarned = true;
  const mb = (v) => Math.round(v / (1024 * 1024));
  const msg = `This morph needs ~${mb(need)} MB of decoded frames (${unique} unique × `
    + `~${(perFrame / (1024 * 1024)).toFixed(1)} MB) but the cache holds ${mb(CACHE_BUDGET)} MB, `
    + `so every loop will re-decode part of it. Lower “frames” to keep it fully cached.`;
  status(msg, "warn");
  $("framelbl").title = msg;
}

// Evict down to the budget, dropping the frame furthest AHEAD of the playhead first.
//
// This replaces an LRU, which is the worst possible policy for a loop: the least-recently
// used frame is precisely the one the playhead is about to come round to again, so a
// timeline that overflowed the budget evicted exactly what it was about to need and hit
// approximately never — a long morph (a cluster layer) stayed slow on every cycle, for
// ever. Evicting what we'll want *last* is near-optimal for a cyclic scan.
//
// "When we'll want it" has to account for the mirror. The timeline is there-and-back
// (buildTimeline), so `[a,b,t]` at position i is the same latent — and therefore the same
// cache key, the same single entry — as position n-1-i. Ranking on `e.idx` alone, the
// position that happened to insert it, got this exactly backwards for the whole return leg:
// a frame the playhead was about to reach again was scored as the furthest away and thrown
// out first. Score each entry by whichever of its two positions comes up soonest.
function evictTo(budget) {
  if (cacheBytes <= budget) return;
  const n = state.timeline.length || 1;
  const here = state.frame;
  const cyc = (d) => ((d % n) + n) % n;        // JS % keeps the sign; we want the cycle
  const untilNeeded = (e) => Math.min(cyc(e.idx - here), cyc((n - 1 - e.idx) - here));
  const ranked = [...frameCache.entries()]
    .map(([k, e]) => [k, e, untilNeeded(e)])
    .sort((x, y) => y[2] - x[2]);              // furthest ahead in playback order first
  for (const [k, e, dist] of ranked) {
    if (cacheBytes <= budget) return;
    if (dist === 0) continue;                  // never evict the frame on screen
    frameCache.delete(k); cacheBytes -= e.bytes; evictions++;
  }
}

// The decode request (query + cache key) for timeline frame i, or null.
// `draft` asks the server for a strided, lightly-smoothed mesh — see DRAFT_GRID.
function frameRequest(i, draft) {
  if (!state.timeline.length) return null;
  i = Math.max(0, Math.min(state.timeline.length - 1, i | 0));
  const [a, b, t] = state.timeline[i];
  const mass = state.refMass >= 0 ? state.refMass : null;
  const at = `index=${a}&index2=${b}&t=${t.toFixed(4)}`;
  const level = mass != null ? `&mass=${mass}` : `&sigma=${state.sigma.toFixed(3)}`;
  if (state.render === "surface") {
    const q = at + (draft ? `&smooth=${draftSmooth()}&size=${PLAY_GRID}` : `&smooth=${state.smooth}`) + level;
    return { surface: true, url: `./api/decode?${q}`, key: `s|${q}` };
  }
  const q = `${at}${volQuery()}`;
  return { surface: false, url: `./api/grid?${q}`, key: `v|${q}` };
}

// Decode (or reuse) frame i -> { surface, payload }. Caches the parsed result and
// coalesces concurrent requests for the same key, so the prefetcher and the render loop
// never fetch the same frame twice.
function decodeFrame(i, draft) {
  const req = frameRequest(i, draft);
  if (!req) return Promise.resolve(null);
  const hit = frameCache.get(req.key);
  // Re-stamp the position it was wanted at. `idx` drives eviction (evictTo), and a shared
  // mirror entry that was inserted on the outbound leg would otherwise keep that stale
  // position for ever. Cheap belt-and-braces alongside the mirror-aware scoring.
  if (hit) { hit.idx = i; return Promise.resolve(hit); }
  const flight = frameInflight.get(req.key);
  if (flight) return flight;
  const p = (async () => {
    const buf = await (await fetch(req.url)).arrayBuffer();
    const payload = req.surface ? parseMesh(buf) : isoSmooth(parseGrid(buf));
    const bytes = req.surface
      ? payload.positions.byteLength + payload.normals.byteLength + payload.indices.byteLength
      : payload.data.byteLength;
    const entry = { surface: req.surface, payload, bytes, idx: i };
    frameCache.set(req.key, entry); cacheBytes += bytes;
    checkTimelineFits();            // measured, before eviction pins cacheBytes to the budget
    evictTo(CACHE_BUDGET);
    doneAt.push(performance.now());
    if (doneAt.length > 12) doneAt.shift();
    return entry;
  })();
  p.finally(() => { if (frameInflight.get(req.key) === p) frameInflight.delete(req.key); });
  frameInflight.set(req.key, p);
  return p;
}

// The exact sequence of frames the play loop will visit from `from`, in order.
//
// This has to agree with togglePlay's advance *exactly*, and it used not to. The loop adds
// `step` and snaps to 0 on overflow; the prefetcher and the buffer probe both walked
// `(frame + d*step) % n` instead. Those are not the same walk: the there-and-back timeline
// makes n odd (n = 2m-1), and step is 2 at the default 60 fps, so the modular walk lands on
// the odd indices — which the playhead never displays. The prefetcher spent up to LOOKAHEAD
// decodes per cycle on frames that were never shown, while bufferedAhead reported those same
// never-shown frames as "not buffered" near every wrap, tripping `draining` and stuttering
// the morph at the same point on every loop, for ever, even with the whole timeline cached.
//
// One generator, so the three consumers cannot drift apart again.
function* upcoming(from, step, count) {
  const n = state.timeline.length;
  let i = from;
  for (let d = 0; d < count; d++) {
    i += step;
    if (i >= n) i = 0;      // exactly what the play loop does
    yield i;
  }
}

// How many frames the playhead can advance through, right now, without waiting.
function bufferedAhead(step, draft) {
  if (!state.timeline.length) return 0;
  let have = 0;
  for (const i of upcoming(state.frame, step, LOOKAHEAD)) {
    const req = frameRequest(i, draft);
    if (!req || !frameCache.has(req.key)) return have;
    have++;
  }
  return LOOKAHEAD;
}

// Keep a window of frames ready *ahead* of the playhead, nearest first, with a couple of
// requests in flight.
//
// The old filler walked the timeline one request deep, in lockstep with the playhead —
// and its premise ("use the idle time during the fps sleeps") was false: when a frame
// costs a second and the frame period is 33 ms, `sleep(delay - elapsed)` is zero. There is
// no idle time. All it achieved was a second concurrent request on a server that does not
// parallelise, making every frame slower for no look-ahead at all.
//
// `mode` is "play" — hold a LOOKAHEAD window ahead of a moving playhead, for as long as
// playback runs — or "warm": fill the whole timeline once, from wherever the playhead is
// parked, and return when there is nothing left to fetch.
//
// Warm mode is the point of the whole exercise. The complete frame list is known the moment
// buildTimeline resolves, which is normally seconds before anyone presses Play — and that
// window was the only one in which decoding is free, because nothing is waiting on it. The
// loop used to be gated on `playing` and refused to use it, so every Play began by paying
// for frames it could already have had sitting in the cache.
const WARM_MAX_FAILS = 6;     // consecutive errors before an unattended warm gives up
async function prefetch(draft, mode = "play") {
  const my = ++fillToken;
  const warming = mode === "warm";
  let fails = 0;
  while (my === fillToken && (playing || warming)) {
    const n = state.timeline.length;
    if (!n) return;
    // Two ways an unattended warm must stop, neither of which applies to playback (where a
    // human is watching and can hit Stop):
    //  - the timeline does not fit the budget, so every further frame evicts one we just
    //    decoded and the loop would churn the server for ever. checkTimelineFits() has
    //    already told the user; there is nothing more to gain here.
    //  - the server is failing. A frame that errors is never cached, so the next pass asks
    //    for it again — that is an infinite retry unless we count.
    if (warming && (evictions > 0 || fails >= WARM_MAX_FAILS)) return;
    const { step } = playCadence();
    // Warming covers one full cycle of the playhead; playback only needs the window ahead.
    const span = warming ? Math.ceil(n / Math.max(1, step)) : LOOKAHEAD;
    const jobs = [];
    for (const i of upcoming(state.frame, step, span)) {
      if (frameInflight.size + jobs.length >= FETCH_CONCURRENCY) break;
      const req = frameRequest(i, draft);
      if (!req || frameCache.has(req.key) || frameInflight.has(req.key)) continue;
      jobs.push(decodeFrame(i, draft).then(() => { fails = 0; }, () => { fails += 1; }));
    }
    // Window full (or the server is busy): wait for something to change rather than spin.
    // Nothing to fetch and nothing in flight means the span is fully cached, so the
    // decode-rate window below is now stale idle time — drop it (see decodeRateMs).
    if (!jobs.length) {
      if (!frameInflight.size) {
        doneAt.length = 0;
        if (warming) return;           // whole timeline resident: the warm is done
      }
      await sleep(30); continue;
    }
    await Promise.race(jobs);          // as soon as one lands, re-evaluate the window
  }
}

// Start decoding a morph before Play is pressed.
//
// Debounced, and that is not optional: updatePathLabel fires buildTimeline un-awaited on
// every shift-click, and a decode cannot be cancelled once the server route starts (it holds
// its slot across the whole request), so warming on each click would strand work on paths
// that were abandoned mid-edit. Waiting for the path to settle costs nothing — the user is
// still choosing states — and keeps abandoned work to at most FETCH_CONCURRENCY frames.
const WARM_DEBOUNCE_MS = 600;
function scheduleWarm() {
  clearTimeout(warmTimer);
  warmTimer = setTimeout(() => {
    if (!playing && state.timeline.length) prefetch(true, "warm");
  }, WARM_DEBOUNCE_MS);
}

// Render one timeline frame in the current render mode (mass-constant for surfaces).
async function renderFrame(i, draft = false) {
  if (!state.timeline.length) return;
  i = Math.max(0, Math.min(state.timeline.length - 1, i | 0));
  state.frame = i; $("scrub").value = String(i); $("framelbl").textContent = `${i + 1}/${state.timeline.length}`;
  $("stop").disabled = !(playing || i > 0);
  const [a, b, t] = state.timeline[i];
  const col = lerpColor(colorForIndex(a), colorForIndex(b), t);   // smooth cluster-color transition
  const tok = ++decodeToken;
  try {
    const entry = await decodeFrame(i, draft);
    if (tok !== decodeToken || !entry) return;   // a newer frame superseded this one
    if (entry.surface) {
      const m = entry.payload;
      if (m.indices.length) renderer.setMesh(m.positions, m.normals, m.indices);
      renderer.setColor(col);
    } else if (state.render === "slices") {
      if (slices) slices.setVolume(entry.payload);   // same dims across frames → view is preserved
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
  if (!state.timeline.length && !(await buildTimeline())) { status("Build a path first (shift-click states).", "warn"); return; }
  setPlaying(true);
  // Playback runs on draft meshes: a morph frame is a *new* latent every time, so nothing
  // is ever cached server-side and the full-resolution mesh cannot be made cheap. The
  // draft is ~15x cheaper, and the frame you settle on is re-rendered at full resolution
  // when you pause (below).
  prefetch(true);                              // keep a window ready ahead of the playhead
  // No preroll. It used to block until 24 frames were buffered — and since it issued no
  // fetches itself, that was ~24 *serial* decodes of "Preparing morph…" before a single
  // pixel moved. With the timeline warmed at build time the frames are usually already
  // here; when they are not, waiting for 24 of them only makes the stall longer and moves
  // it earlier. Await the one frame we are about to show, and start.
  //
  // Showing state.frame (rather than advancing first) also fixes the opening jump: the loop
  // begins with `frame + step`, so a fresh morph used to open on timeline[2] at the default
  // 60 fps, already part-way through the first hop. On resume this re-renders the frame
  // that is on screen as a draft — which is what playback shows anyway.
  if (playing) await renderFrame(state.frame, true);
  while (playing) {
    const { delay, step } = playCadence();
    let next = state.frame + step;
    if (next >= state.timeline.length) next = 0;     // loop (wrap to start)
    const t0 = performance.now();
    await renderFrame(next, true);
    if (!playing) break;
    // Pace to the buffer. With a healthy look-ahead we hold the display rate; while the
    // prefetcher is still catching up we drop to the rate the server actually sustains,
    // so the first pass is smooth-but-slower rather than jerky. Once the timeline is
    // cached the buffer never drains and it runs at full speed.
    const rate = decodeRateMs();
    const draining = bufferedAhead(step, true) < LOW_WATER && rate > 0;
    const pace = draining ? Math.max(delay, rate) : delay;
    status(draining
      ? `Playing… frame ${state.frame + 1}/${state.timeline.length} (buffering)`
      : `Playing… frame ${state.frame + 1}/${state.timeline.length}`, "busy");
    await sleep(Math.max(0, pace - (performance.now() - t0)));
  }
  fillToken++;                                 // stop the prefetcher
  if (!recording) {
    status(`Paused at frame ${state.frame + 1}/${state.timeline.length}.`);
    renderFrame(state.frame, false);           // settle on the full-resolution surface
    scheduleWarm();                            // keep filling for the next Play
  }
}

function stopPlayback() {
  setPlaying(false); recording = false;
  fillToken++;
  if (state.timeline.length) renderFrame(0, false);
  status("Stopped.");
}

async function recordSeries() {
  if (!state.timeline.length && !(await buildTimeline())) { status("Build a path first (shift-click states).", "warn"); return; }
  const active = state.render === "surface" ? renderer : volRenderer;
  if (!active || !active.captureStream) { status("Recording unavailable for this view.", "err"); return; }
  let rec = null;
  try { rec = startCanvasRecording(active, 30); }
  catch (err) { status("Recording unavailable: " + err.message, "err"); return; }
  recording = true; setPlaying(true);
  // Recording stays on full-resolution frames — an exported movie is the one place the
  // draft would be a real loss — but it now prefetches ahead of itself instead of decoding
  // strictly in lockstep, so the capture stalls less between frames.
  prefetch(false);
  await renderFrame(0, false);
  const { delay: recDelay, step: recStep } = playCadence();   // record what playback shows
  for (let i = recStep; i < state.timeline.length && playing; i += recStep) { await renderFrame(i, false); status(`Recording… ${i + 1}/${state.timeline.length}`, "busy"); await sleep(recDelay); }
  fillToken++;
  setPlaying(false); recording = false;
  downloadBlob(await rec.stop(), "morph.webm");
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

// Match the overlay canvases' backing stores to their CSS boxes.
//
// This has to be re-checked before every draw, not only on window resize. The overlays
// are stretched to their box by CSS (`.canvas-wrap > canvas { inset: 0 }`), so when the
// box changes without the *window* changing — clustering adds k+1 rows to the layers
// rail, which widens it and shrinks the canvas beside it — the backing store keeps its
// old pixel size and the browser rescales the bitmap into the new box. Everything drawn
// on it then lands off by that scale factor, which is why the centroid markers sat
// beside their clusters right after "Cluster" and snapped into place on the next window
// resize. The scatter renderer already re-syncs itself this way on every frame
// (scatter.js) — that's why the points were always right and only the overlay was off.
function sizeOverlays() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  for (const [id, ctx] of [["lassoCanvas", lctx], ["centroidOverlay", centroidCtx]]) {
    const c = $(id); if (!c) continue;
    const w = Math.max(1, Math.floor(c.clientWidth * dpr)), h = Math.max(1, Math.floor(c.clientHeight * dpr));
    // Only on a real change: assigning width/height clears the canvas, and this now runs
    // per render (an in-progress lasso path would blink away every frame otherwise).
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    if (ctx) ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
}
function sizeLasso() {
  sizeOverlays();
  syncCentroidOverlay();
  drawDensityBar();
}
function setupLasso() {
  const lc = $("lassoCanvas"); lctx = lc.getContext("2d");
  centroidCtx = $("centroidOverlay").getContext("2d");
  sizeLasso();
  window.addEventListener("resize", sizeLasso);
  // Layout changes that no resize event reports: the layers rail growing, the pane
  // divider being dragged, the panel collapsing. Watching the boxes themselves catches
  // all of them. Safe from feedback loops — the canvases are absolutely positioned and
  // sized by CSS, so writing width/height cannot change their layout.
  if (window.ResizeObserver) {
    const ro = new ResizeObserver(() => sizeLasso());
    ro.observe(lc); ro.observe($("centroidOverlay"));
  }
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
    if (is3d() && cloud) cloud.orbitZoom(e);
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
  const scope = pickScopeSet();   // Feature 3: only consider points in the checked layers
  let best = -1, bestD = Infinity;
  for (let i = 0; i < proj.length; i++) { const p = proj[i]; if (!p) continue; if (scope && !scope.has(i)) continue; const dd = (p[0] - sx) ** 2 + (p[1] - sy) ** 2; if (dd < bestD) { bestD = dd; best = i; } }
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
  const scope = pickScopeSet();   // Feature 3: lasso only grabs points in the checked layers
  const enclosed = [];
  for (let i = 0; i < proj.length; i++) if (proj[i] && (!scope || scope.has(i)) && pointInPoly(proj[i], poly)) enclosed.push(i);
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
  $("layerSelRow").style.display = layers.length ? "" : "none";
  layers.forEach((ly) => {
    const isCurve = ly.kind === "curve";
    const editing = editingPathLayer === ly || editCurve === ly;
    const inScope = state.pickRestrict && ly.selected && ly.indices && ly.indices.length > 0;
    const li = document.createElement("li"); li.className = "layer-item" + (ly.visible ? "" : " hidden") + (editing ? " editing" : "") + (ly.kind === "path" || isCurve ? " path" : "") + (inScope ? " in-scope" : "");
    const chk = document.createElement("input"); chk.type = "checkbox"; chk.className = "lsel"; chk.checked = !!ly.selected; chk.title = "Select layer (bulk show / hide · pick-restrict scope)";
    chk.addEventListener("change", () => { ly.selected = chk.checked; syncSelAll(); if (state.pickRestrict) { renderLayers(); status(pickScopeMsg()); } });
    const eye = iconBtn(ly.visible ? "◉" : "○", "Show / hide", () => { ly.visible = !ly.visible; renderLayers(); if (state.colorBy === "layers") applyColor(); });
    const sw = document.createElement("input"); sw.type = "color"; sw.className = "swatch"; sw.value = ly.color; sw.title = "Layer color";
    sw.addEventListener("input", (e) => { ly.color = e.target.value; if (state.colorBy === "layers") applyColor(); if (isCurve) refreshFreeCurves(); if (editingPathLayer === ly) syncSelection(); });
    const name = document.createElement("span"); name.className = "lname"; name.textContent = (ly.kind === "path" ? "↳ " : isCurve ? "∿ " : "") + ly.name; name.title = "Click to rename";
    name.addEventListener("click", () => { const v = prompt("Layer name", ly.name); if (v) { ly.name = v.trim() || ly.name; renderLayers(); } });
    const cnt = document.createElement("span"); cnt.className = "lcount"; cnt.textContent = `${isCurve ? ly.ctrl.length : ly.indices.length}`;
    li.append(chk, eye, sw, name, cnt);
    if (ly.kind === "path") li.append(iconBtn(editing ? "✓" : "✎", editing ? "Finish editing this path" : "Edit waypoints (shift-click to add/remove)", () => editPathLayer(ly)));
    if (isCurve) li.append(iconBtn(editing ? "✓" : "✎", editing ? "Finish editing this curve" : "Edit the curve (drag handles · alt-click delete · tap to insert)", () => editCurveLayer(ly)));
    li.append(iconBtn("▶", isCurve ? "Morph along this curve (snaps to nearest states)" : ly.kind === "path" ? "Morph along this path" : "Morph through this layer as a volume series", () => playLayerSeries(ly)));
    if (exportAvailable && !isCurve) li.append(iconBtn("⤓", "Save layer (metadata + average volume) to the project", () => saveLayer(ly)));
    li.append(iconBtn("✕", "Delete layer", () => removeLayer(ly)));
    ul.appendChild(li);
  });
  syncSelAll();
  refreshFreeCurves();
  syncCentroidOverlay();   // reflect centroid-layer visibility toggles immediately
}

// ---- bulk layer visibility + pick-scope (Features 1–3) ----
const checkedLayers = () => layers.filter((l) => l.selected);

// Feature 1: one button flips every layer on / off. If any are visible, hide all;
// otherwise show all — so a single press always changes something visible.
function toggleAllLayers() {
  const anyVisible = layers.some((l) => l.visible);
  layers.forEach((l) => (l.visible = !anyVisible));
  renderLayers();
  if (state.colorBy === "layers") applyColor();
}

// Feature 2: show / hide the checked layers. With none checked it falls back to
// acting on every layer, so the buttons are never dead.
function setCheckedVisible(v) {
  const target = checkedLayers();
  (target.length ? target : layers).forEach((l) => (l.visible = v));
  renderLayers();
  if (state.colorBy === "layers") applyColor();
}

// Keep the master "select" checkbox in sync (checked / indeterminate / clear).
function syncSelAll() {
  const box = $("selAllLayers");
  if (!box) return;
  const on = layers.filter((l) => l.selected).length;
  box.checked = layers.length > 0 && on === layers.length;
  box.indeterminate = on > 0 && on < layers.length;
}

// Feature 3: the union of indices in the checked layers (base states *and* centroids —
// both are selectable), or null when the restriction is off / nothing is checked
// (→ no restriction). Centroids (index >= L.n) aren't in the base cloud, so the renderer
// can't pick them; resolvePick() hit-tests their markers app-side instead.
function pickScopeSet() {
  if (!state.pickRestrict || !L) return null;
  const s = new Set();
  for (const l of checkedLayers()) if (l.indices) for (const i of l.indices) s.add(i);
  return s.size ? s : null;
}
function pickScopeMsg() {
  const s = pickScopeSet();
  if (!s) return "Pick-restrict on — check one or more layers to limit picking to their points.";
  return `Picking limited to ${s.size} point(s) in the checked layers.`;
}
// Pick filter handed to the scatter/cloud renderers: their base-point click/hover picking
// consults this on every pick, so pick-restrict limits which *base* points are selectable.
// Centroids are handled separately in resolvePick(). Reads the live scope each time.
const pickAllowed = (i) => { const s = pickScopeSet(); return !s || s.has(i); };

// Resolve a click at client (cx,cy) into a selectable index. The renderer hands us the
// nearest in-scope BASE point (or -1). We additionally hit-test the visible, in-scope
// Centroids — which the renderers can't pick at all — and let a click on a centroid
// marker win. This is what makes centroid picking (and "pick only Centroids") work.
const activeCanvas = () => (is3d() && cloud ? $("scatter3d") : $("scatter"));
function screenPosOf(idx) {
  const vec = idx < L.n ? (activeCoords() || L.pca)[idx] : centroidCoords()[idx];
  if (!vec) return null;
  return (activeLand().projectData([axesOf(vec)]) || [null])[0];
}
const CENTROID_HIT2 = 12 * 12;   // CSS px²; a click this close to a marker selects it
function resolvePick(baseIdx, cx, cy) {
  const canvas = activeCanvas();
  if (!canvas) return baseIdx;
  const r = canvas.getBoundingClientRect();
  const px = cx - r.left, py = cy - r.top;
  const scope = pickScopeSet();
  const cen = layers.find((l) => l.kind === "centroid" && l.visible);
  let nc = null;   // nearest in-scope centroid
  const cc = centroidCoords();
  if (cen) for (const idx of cen.indices) {
    if (!cc[idx] || (scope && !scope.has(idx))) continue;
    const p = screenPosOf(idx); if (!p) continue;
    const d = (p[0] - px) ** 2 + (p[1] - py) ** 2;
    if (!nc || d < nc.d) nc = { idx, d };
  }
  if (nc && nc.d <= CENTROID_HIT2) return nc.idx;   // clicked on a centroid marker
  if (baseIdx < 0) return nc ? nc.idx : -1;         // no in-scope base point (e.g. centroids-only)
  if (!nc) return baseIdx;
  const bp = screenPosOf(baseIdx);
  const bd = bp ? (bp[0] - px) ** 2 + (bp[1] - py) ** 2 : Infinity;
  return nc.d < bd ? nc.idx : baseIdx;             // whichever is nearer the click
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
      delete indexColor[ly.centroid];
      for (const m of Object.values(centroidVecs)) delete m[ly.centroid];
      if (!cen.indices.length) layers.splice(layers.indexOf(cen), 1);
    }
  }
  layers.splice(layers.indexOf(ly), 1);
  renderLayers(); refreshFreeCurves(); if (state.colorBy === "layers") applyColor();
}

const coordOf = (i) => (i >= L.n ? (centroidCoords()[i] ? centroidCoords()[i][state.x] : 0) : (activeCoords() || L.pca)[i][state.x]);
const orderAlongAxis = (idx) => [...idx].sort((a, b) => coordOf(a) - coordOf(b));

// Centroid coordinates for the *active* embedding, fetched once per embedding and kept.
// The server returns them in cluster order, so coords[j] is the virtual index base + j —
// the same order the centroid layer was built in. A no-op once they're cached, and when
// there is no clustering yet.
async function ensureCentroidCoords() {
  if (!clusterQuery || centroidVecs[state.embedding]) return;
  try {
    const d = await (await fetch(`./api/centroid_coords?embed=${state.embedding}&${clusterQuery.q}`)).json();
    const m = {};
    d.coords.forEach((v, j) => { m[clusterQuery.base + j] = v; });
    centroidVecs[state.embedding] = m;
  } catch (err) {
    // Leave the map absent rather than falling back to another embedding's coordinates:
    // no marker beats a marker drawn in the wrong units.
    status("Could not place the centroids in this view: " + err, "warn");
  }
}

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
  const q = seeded ? `seeds=${seeds.join(",")}` : `k=${k}`;
  const url = `./api/kmeans?${q}`;
  const label = seeded ? `seeded KMeans (${seeds.length} picks)` : `KMeans (k=${k})`;
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
  indexColor = {}; centroidVecs = {}; clusterQuery = { q, base: d.base };
  const groups = Array.from({ length: d.k }, () => []);
  for (let i = 0; i < kLabels.length; i++) { const g = groups[kLabels[i]]; if (g) g.push(i); }
  // Tag each cluster with its centroid's virtual index so deleting the cluster
  // can drop the matching point from the Centroids layer.
  groups.forEach((idxs, j) => layers.push({ name: `Cluster ${j + 1}`, color: clusterColor(j), indices: idxs, visible: true, kind: "cluster", centroid: d.base + j }));
  const cents = [];
  const pcaMap = {};
  for (let j = 0; j < d.k; j++) { const idx = d.base + j; cents.push(idx); indexColor[idx] = clusterColor(j); pcaMap[idx] = d.centroid_pca[j]; }
  centroidVecs.pca = pcaMap;
  await ensureCentroidCoords();   // the active view may not be PCA
  layers.push({ name: "Centroids", color: "#ffffff", indices: cents, visible: true, kind: "centroid" });
  $("colorby").value = "layers"; state.colorBy = "layers"; toggleColorWraps();
  renderLayers(); applyColor();
  status(`${seeded ? "Seeded KMeans" : "KMeans"} k=${d.k}: ${d.k} cluster layers + a centroid layer (▶ to morph between clusters).`);
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
  // Explicitly, before playing: a cluster series can arrive with hundreds of states, and
  // buildTimeline is what thins it and prices the hops. (updatePathLabel fires it too, but
  // does not await it — and togglePlay must not race a half-built timeline.)
  await buildTimeline();
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
  let blob = await src.snapshot({ scale, transparent });
  // The bar lives on a separate overlay canvas, so the renderer's snapshot knows nothing
  // about it. Composite it in here, at the export scale, so the PNG carries its own scale.
  if (blob && name === "landscape" && densityBarVisible()) blob = await stampDensityBar(blob, scale);
  if (blob) downloadBlob(blob, `${name}.png`);
}
async function stampDensityBar(blob, scale) {
  try {
    const img = await createImageBitmap(blob);
    const cv = document.createElement("canvas");
    cv.width = img.width; cv.height = img.height;
    const ctx = cv.getContext("2d");
    ctx.drawImage(img, 0, 0);
    paintDensityBar(ctx, img.width, img.height, scale);
    img.close();
    return await new Promise((r) => cv.toBlob(r, "image/png"));
  } catch (err) {
    console.warn("colour bar not stamped:", err.message);
    return blob;   // an image without its legend still beats no image
  }
}
function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a"); a.href = url; a.download = filename; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

function hexToRgb(hex) { const n = parseInt(hex.slice(1), 16); return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255]; }

boot();
