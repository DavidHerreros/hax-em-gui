// Map (.mrc) viewer controller — one map, or several in one scene. Reuses the same
// renderers as the conformational-landscape app — the marching-cubes surface (gl.js),
// the volume ray-marcher (volume.js) and the contour histogram (histogram.js) — plus an
// orthogonal-slice view (slices.js). No latents, no network, no landscape: it just
// inspects the volumes the backend loaded from disk.
//
// With more than one map the ChimeraX arrangement applies: a model panel toggles each
// map's visibility, every map carries its own colour and its own contour, and one of
// them is *active* — the one the contour histogram, the colour picker and the .mrc
// download act on. Surface mode draws them all together; volume mode composites every
// visible map (up to the renderer's slot count) in one ray march, each at its own
// contour and colour. Slice mode shows the active map alone — overlaying several
// slice stacks honestly needs per-map transfer functions it does not have.
import { createRenderer } from "./gl.js";
import { createVolume } from "./volume.js";
import { createHistogram } from "./histogram.js";
import { createSlices } from "./slices.js";
import { parseColorMesh, parseGrid, parseMesh } from "./wire.js";
import { latestWins, makeLive } from "./viewerbase.js";

const $ = (id) => document.getElementById(id);
const THEME_BG = { dark: "#040610", light: "#e7ecf3" };

// Guarded construction: a WebGL2 failure used to throw at module load, leaving the
// page stuck on "Loading…" with the reason only in the console (app.js already
// guarded this; mrc.js had drifted).
let renderer, volRenderer, slices, hist;
try {
  renderer = createRenderer($("gl"));       // surface meshes (all visible maps)
  volRenderer = createVolume($("glvol"));   // MIP / glow ray-march (active map)
  slices = createSlices($("slices"));       // orthogonal cuts (active map)
  hist = createHistogram($("hist"), onLevel);
} catch (err) {
  status("WebGL2 unavailable: " + err.message, "err");
  throw err;
}

// A lost WebGL context (GPU reset, driver update, the tab backgrounded too long)
// used to leave a silently frozen canvas. There is no rebuilding all the GPU state
// worth its complexity here — say what happened and offer the one real fix.
for (const id of ["gl", "glvol"]) {
  $(id).addEventListener("webglcontextlost", (e) => {
    e.preventDefault();
    status("WebGL context lost — reloading the viewer…", "err");
    setTimeout(() => location.reload(), 800);
  });
}

const state = { render: "surface", active: 0 };
let info = null;        // /api/info
let models = [];        // per-map view state, parallel to info.models
let railSetup = false;  // setupRail binds pointer listeners; must run once even though
                        // loadInfo now reruns after map arithmetic grows the scene

function status(msg, cls = "") {
  const el = $("status"); el.textContent = msg; el.title = msg;
  el.className = "status" + (cls ? " " + cls : "");
}

const active = () => models[state.active];
const visibleModels = () => models.filter((m) => m.visible);

// ---- theme (mirrors app.js: ?theme= on load + postMessage on toggle) ----
function applyTheme(theme) {
  theme = theme === "light" ? "light" : "dark";
  document.documentElement.dataset.theme = theme;
  const bg = THEME_BG[theme];
  $("bg").value = bg;
  const rgb = hexToRgb(bg);
  renderer.setBackground(rgb); volRenderer.setBackground(rgb); slices.setBackground(bg);
  // The gizmo's label halo tracks the theme; before startGizmo runs this is a no-op.
  try { gizmoInvalidate(); } catch { /* module still loading */ }
}
applyTheme(new URLSearchParams(location.search).get("theme"));
window.addEventListener("message", (e) => { if (e.data && e.data.type === "hax-theme") applyTheme(e.data.theme); });

// ---- persistence ---------------------------------------------------------
// (a) Per-map memory, keyed by the file's path: reopening a map lands on the contour,
//     colour and smoothing you last chose for it, in any scene. LRU-capped.
// (g) Per-scene session state, keyed by the set of files: render mode, visibility,
//     active map, background and camera survive closing and reopening the viewer.
const MAP_PREFS_KEY = "hax_mrc_map_prefs";
const MAP_PREFS_CAP = 300;
function readJSON(key) { try { return JSON.parse(localStorage.getItem(key) || "null"); } catch { return null; } }
function writeJSON(key, v) { try { localStorage.setItem(key, JSON.stringify(v)); } catch { /* storage off/full */ } }

function saveMapPref(m) {
  const all = readJSON(MAP_PREFS_KEY) || {};
  delete all[m.filename];                      // reinsert: object order doubles as LRU order
  all[m.filename] = { sigma: m.sigma, color: m.color, smooth: m.smooth };
  const keys = Object.keys(all);
  for (let i = 0; i < keys.length - MAP_PREFS_CAP; i++) delete all[keys[i]];
  writeJSON(MAP_PREFS_KEY, all);
}

const sceneKey = () => "hax_mrc_scene:" + models.map((m) => m.filename).slice().sort().join("|");
let sceneRestored = false;   // suppress saves while the restore itself fires the handlers
let sceneSaveTimer = null;
function schedSaveScene() {
  if (!sceneRestored) return;
  clearTimeout(sceneSaveTimer);
  sceneSaveTimer = setTimeout(saveScene, 400);
}
function saveScene() {
  if (!models.length || !sceneRestored) return;
  const target = camera3d(state.render);
  const cam = target ? target.getCamera() : camera;
  writeJSON(sceneKey(), {
    mode: state.render,
    active: state.active,
    visible: models.map((m) => m.visible),
    bg: $("bg").value,
    // Typed arrays serialize as objects; force plain arrays so setState can .slice().
    camera: cam ? { R: Array.from(cam.R), dist: cam.dist, offset: Array.from(cam.offset), pivot: Array.from(cam.pivot) } : null,
  });
}
// The camera has no change hook, so the leave events are what capture its final state.
window.addEventListener("pagehide", saveScene);
document.addEventListener("visibilitychange", () => { if (document.hidden) saveScene(); });

// ---- data loaders --------------------------------------------------------
async function loadInfo() {
  info = await (await fetch("./api/info")).json();
  const prefs = readJSON(MAP_PREFS_KEY) || {};
  models = info.models.map((m, i) => ({
    ...m,
    visible: true,
    color: (prefs[m.filename] && prefs[m.filename].color) || m.color,
    sigma: (prefs[m.filename] && Number.isFinite(prefs[m.filename].sigma)) ? prefs[m.filename].sigma : 2.5,
    smooth: (prefs[m.filename] && Number.isFinite(prefs[m.filename].smooth)) ? prefs[m.filename].smooth : 6,
    surface: null,     // the sigma the loaded mesh was cut at, so it isn't refetched
  }));
  $("title").textContent = info.title || "Map";
  const multi = models.length > 1;
  // The rail always exists (contour/display/clip live there); only the sections
  // that need several maps hide for a lone one.
  $("secModels").hidden = !multi;
  $("modelsCompute").hidden = !multi;
  if (multi) { buildModelPanel(); fillComputeSelects(); }
  if (!railSetup) { setupRail(); railSetup = true; }
  // Maps on different grids cannot be blended voxel-by-voxel, and the backend refuses
  // to; don't offer controls that can only fail.
  $("morph").hidden = !(multi && info.morph_shape);
  $("paintRow").hidden = !(multi && info.morph_shape);
  $("linklbl").hidden = !multi;
  if (!$("morph").hidden) updateMorphUI();
  setActive(0);
  document.title = "Hax — " + (multi ? `${models.length} maps` : models[0].name);
}

// Everything that depends on *which* map is active: the header labels, the colour
// picker, and the contour histogram (each map has its own density range, so the
// histogram is refetched rather than rescaled).
async function setActive(i) {
  state.active = i;
  const m = active();
  $("sel").textContent = m.name;
  // Voxel count, plus the physical size when the header carried a voxel size
  // (1.0 is the server's "unknown" sentinel — don't dress voxels up as Å).
  const dims = m.shape.slice().reverse();
  $("meta").textContent = dims.join("×")
    + (m.voxel_size && m.voxel_size !== 1
       ? ` · ${m.voxel_size.toFixed(2)} Å/vox · ${dims.map((n) => (n * m.voxel_size).toFixed(0)).join("×")} Å`
       : "");
  $("color").value = m.color;
  // Colour and smoothing are per map, so the shared controls have to follow the
  // active one — otherwise they show the previous map's settings and the next nudge
  // silently applies them to this one.
  $("smooth").value = String(m.smooth);
  $("sigVal").textContent = m.sigma.toFixed(2) + " σ";
  if (models.length > 1) markActiveRow();
  schedSaveScene();
  await loadHistogram();
  if (state.render === "mip" || state.render === "emission") loadVolume();
  else if (state.render === "slices") loadSlices();
}

async function loadHistogram() {
  const m = active();
  const h = await (await fetch(`./api/histogram?m=${m.index}`)).json();
  hist.setData(h, h.mean + m.sigma * h.std);
}

async function fetchGrid(m, size) {
  const buf = await (await fetch(`./api/grid?m=${m}&size=${size}`)).arrayBuffer();
  return parseGrid(buf);
}

// One contour costs a full marching-cubes pass over the whole volume plus a multi-MB mesh
// download, and the histogram fires this on every pointermove. Left unthrottled a single
// drag queues one such request per pixel crossed; each reads the *current* level, so
// coalescing converges on the level you released at.
//
// Even coalesced, a full pass is ~750 ms and ~21 MB at a 320³ box — steady, but not live.
// So while the pointer is down we ask for a draft: a strided grid, lightly smoothed (~80 ms,
// ~3.8 MB), and take the full mesh once on release. The server scales the draft by its
// stride, so it sits in the same [-1,1] cube and refines in place rather than jumping.
// The draft size has to be relative to the map, not a constant: `_downsample` strides by
// ceil(box/size), so a fixed 160 leaves a 128³ map completely undownsampled and the
// "draft" costs exactly what the full pass costs. Half the box edge gives a stride of 2
// (an 8x cheaper grid) on anything big enough to need it, and the floor stops small maps
// being reduced to mush for no saving.
const draftSize = () => Math.max(64, Math.round((info ? info.full_size : 320) / 2));
// Drafts carry light smoothing too: a coarse grid facets more, and Taubin's cost is
// almost all fixed setup (one tick costs what six do), so leaving it off bought nothing
// and made the surface change character when you let go of the slider. See app.js.
const DRAFT_SMOOTH = 2;

// ChimeraX's "hide dust": components smaller than dust.size voxels are dropped
// server-side before marching cubes. Changing it invalidates every cached surface.
const dust = { on: false, size: 200 };
const dustParam = () => (dust.on && dust.size > 1 ? `&dust=${Math.round(dust.size)}` : "");

// Contour one map. Only the map being adjusted is refetched — the others are already on
// the GPU and their contours have not changed.
async function surfaceOnce(idx, draft) {
  const m = models[idx];
  if (!m) return;
  status(models.length > 1 ? `Contouring ${m.name}…` : "Contouring…", "busy");
  try {
    const q = `m=${m.index}&sigma=${m.sigma.toFixed(3)}${dustParam()}`
            + (draft ? `&smooth=${Math.min(m.smooth, DRAFT_SMOOTH)}&size=${draftSize()}` : `&smooth=${m.smooth}`);
    const buf = await (await fetch(`./api/decode?${q}`)).arrayBuffer();
    if (state.render !== "surface") return;   // mode switched while we were waiting
    const mesh = parseMesh(buf);
    renderer.setModelMesh(m.index, mesh.positions, mesh.normals, mesh.indices);
    if (painted === idx) { painted = null; $("paintClear").hidden = true; }   // plain mesh = paint gone
    renderer.setModelColor(m.index, hexToRgb(m.color));
    renderer.setModelVisible(m.index, m.visible);
    if (!draft) m.surface = m.sigma;
    if (!mesh.indices.length) { status(`Empty surface at this contour (${m.name})`, "warn"); return; }
    reportSurface();
  } catch (err) { status("Contour failed: " + err.message, "err"); }
}
const loadSurface = makeLive(surfaceOnce);

function reportSurface() {
  const shown = visibleModels().length;
  status(models.length > 1
    ? `Surface · ${shown}/${models.length} shown · ${active().name} σ ${active().sigma.toFixed(2)}`
    : `Surface · σ ${active().sigma.toFixed(2)}`);
}

// Contour every visible map that has not been contoured yet (on open, and whenever a
// map is switched back on). Sequential: each is a full marching-cubes pass server-side,
// and firing twenty at once would just queue them behind each other with less feedback.
async function loadSurfaces() {
  if (morph.showing) return;   // the maps are hidden; the morph owns the scene
  for (const m of visibleModels()) {
    if (m.surface === m.sigma) continue;
    await surfaceOnce(models.indexOf(m), false);
    if (state.render !== "surface") return;
  }
  reportSurface();
}

// Which model (by list index) each renderer slot holds; live contour/colour changes
// address slots through this instead of re-uploading textures.
let volSlots = [];

// The normalized threshold a map's own sigma corresponds to over its own grid range.
function thrFor(m) {
  const g = m.gridVol;
  if (!g) return 0.45;
  const level = m.mean + m.sigma * (m.std || 1);
  return Math.max(0, Math.min(1, (level - g.vmin) / ((g.vmax - g.vmin) || 1)));
}

const volGuard = latestWins();
async function loadVolume() {
  if (morph.showing) return;     // the morph owns the volume canvas
  // Every visible map rides in one composited march, each at its own contour and
  // colour; with nothing visible, fall back to the active map so the canvas is
  // never silently empty.
  let show = visibleModels();
  if (!show.length) show = [active()];
  const dropped = Math.max(0, show.length - volRenderer.maxSlots);
  if (dropped) show = show.slice(0, volRenderer.maxSlots);
  status("Loading volume…", "busy");
  const tok = volGuard.take();   // changing the shown set mid-download: newest wins
  try {
    for (const m of show) {
      if (!m.gridVol) m.gridVol = await fetchGrid(m.index, 96);
      if (volGuard.stale(tok)) return;
    }
    volSlots = show.map((m) => models.indexOf(m));
    volRenderer.setSlots(show.map((m) => {
      const g = m.gridVol;
      return { nx: g.nx, ny: g.ny, nz: g.nz, data: g.data, lo: g.boxLo, hi: g.boxHi,
               color: hexToRgb(m.color), threshold: thrFor(m) };
    }));
    const g0 = show[0].gridVol;
    status(show.length > 1
      ? `Volume · ${show.length} maps composited`
        + (dropped ? ` · ${dropped} hidden (max ${volRenderer.maxSlots})` : "")
      : `Volume ${g0.nx}×${g0.ny}×${g0.nz} · ${show[0].name}`);
  } catch (err) { status("Volume load failed: " + err.message, "err"); }
}

const sliceGuard = latestWins();
async function loadSlices() {
  const m = active();
  status("Loading slices…", "busy");
  const tok = sliceGuard.take();
  try {
    if (!m.gridSlice) m.gridSlice = await fetchGrid(m.index, 192);
    if (sliceGuard.stale(tok)) return;
    // Å per *grid* voxel: the slice grid is stride-downsampled, so scale the header
    // voxel size by the stride. Set before the volume so the legend can use it.
    const g = m.gridSlice;
    const stride = Math.max(1, Math.round(Math.max(...m.shape) / Math.max(g.nx, g.ny, g.nz)));
    slices.setScale(m.voxel_size && m.voxel_size !== 1 ? m.voxel_size * stride : 0);
    slices.setVolume(m.gridSlice);
    status(`Slices ${m.gridSlice.nx}×${m.gridSlice.ny}×${m.gridSlice.nz} · ${m.name}`
           + (models.length > 1 ? " · active map only" : ""));
  } catch (err) { status("Slice load failed: " + err.message, "err"); }
}

// contour level (absolute) -> ray-marcher threshold in [0,1] over the grid range
function normThreshold() {
  const g = active().gridVol;
  if (!g) return 0.45;
  return (hist.level - g.vmin) / ((g.vmax - g.vmin) || 1);
}

// "link": one contour gesture drives every visible map at the same sigma (each
// map's own mean/std, so the level is statistically equivalent, not identical).
let linkContour = false;

function onLevel(level, dragging) {
  const m = active();
  m.sigma = (level - m.mean) / (m.std || 1);
  $("sigVal").textContent = m.sigma.toFixed(2) + " σ";
  if (state.render === "surface") {
    loadSurface(state.active, !!dragging);   // the active map tracks the drag live
    if (linkContour && !dragging) {
      for (const other of visibleModels()) {
        if (other !== m) { other.sigma = m.sigma; other.surface = null; saveMapPref(other); }
      }
      loadSurfaces();                        // full pass over the rest on release
    }
  } else if (state.render === "mip" || state.render === "emission") {
    const slot = volSlots.indexOf(state.active);
    if (slot >= 0) volRenderer.setSlotThreshold(slot, normThreshold());
    else volRenderer.setThreshold(normThreshold());   // morph / fallback: slot 0
  }
  if (!dragging) { saveMapPref(m); refreshRowSigmas(); }   // release: this is the level you chose
}

// Per-row σ readouts in the model panel, refreshed on any settled level change.
function refreshRowSigmas() {
  if (models.length <= 1) return;
  for (const row of $("modelsList").children) {
    const m = models[Number(row.dataset.index)];
    const el = row.querySelector(".model-sig");
    if (m && el) el.textContent = m.sigma.toFixed(2) + "σ";
  }
}

// ---- model panel (a resizable right rail, ChimeraX-style) ----------------
const RAIL_KEY = "hax_mrc_models_rail";

// Show/hide the rail. `.rail-off` on the stage also hides the drag handle, which has to
// sit before the rail in the markup and so cannot be reached by a sibling selector.
function setRail(open) {
  const rail = $("models");
  rail.classList.toggle("collapsed", !open);
  $("stage").classList.toggle("rail-off", !open || rail.hidden);
  $("modelsReopen").hidden = open || rail.hidden;
  try { localStorage.setItem(RAIL_KEY, JSON.stringify({ open, width: rail.style.width || "" })); } catch { /* storage off */ }
  // The canvas is sized by flex, so its backing store has to be recomputed.
  window.dispatchEvent(new Event("resize"));
}

function setupRail() {
  const rail = $("models");
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(RAIL_KEY) || "null"); } catch { /* ignore */ }
  if (saved && saved.width) rail.style.width = saved.width;
  setRail(!saved || saved.open !== false);

  $("modelsCollapse").addEventListener("click", () => setRail(false));
  $("modelsReopen").addEventListener("click", () => setRail(true));

  // Drag the divider. Same handle idiom as the landscape viewer's layers rail; the
  // resize event on every move is what keeps the WebGL backing store in step.
  const handle = $("modelsResize");
  let dragging = false;
  handle.addEventListener("pointerdown", (e) => {
    dragging = true; handle.classList.add("dragging");
    handle.setPointerCapture(e.pointerId); e.preventDefault();
  });
  handle.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    // Measured from the rail's right edge, so the panel follows the cursor rather than
    // inverting when the drag crosses its own left edge.
    const right = rail.getBoundingClientRect().right;
    rail.style.width = `${Math.max(150, Math.min(460, right - e.clientX))}px`;
    window.dispatchEvent(new Event("resize"));
  });
  handle.addEventListener("pointerup", (e) => {
    dragging = false; handle.classList.remove("dragging");
    try { handle.releasePointerCapture(e.pointerId); } catch { /* already gone */ }
    setRail(true);   // persists the new width
  });
}

function buildModelPanel() {
  const list = $("modelsList");
  list.textContent = "";
  models.forEach((m, i) => {
    const row = document.createElement("div");
    row.className = "model-row";
    row.dataset.index = String(i);

    const see = document.createElement("input");
    see.type = "checkbox";
    see.checked = m.visible;
    see.title = "Show this map";
    see.addEventListener("click", (e) => e.stopPropagation());
    see.addEventListener("change", () => setVisible(i, see.checked));

    const swatch = document.createElement("input");
    swatch.type = "color";
    swatch.className = "model-color";
    swatch.value = m.color;
    swatch.title = "Colour of this map";
    swatch.addEventListener("click", (e) => e.stopPropagation());
    swatch.addEventListener("input", () => setColor(i, swatch.value));

    const name = document.createElement("span");
    name.className = "model-name";
    name.textContent = m.name;
    name.title = m.filename;

    // Each map's own contour, at a glance — the panel is where "which level is
    // which map at?" gets answered.
    const sig = document.createElement("span");
    sig.className = "model-sig";
    sig.textContent = m.sigma.toFixed(2) + "σ";

    // Solo: show only this map; click again to bring the others back. The single
    // most-used ChimeraX gesture when comparing many maps.
    const solo = document.createElement("button");
    solo.className = "model-solo";
    solo.textContent = "S";
    solo.title = "Show only this map (click again to show all)";
    solo.addEventListener("click", (e) => { e.stopPropagation(); toggleSolo(i); });

    row.append(see, swatch, name, sig, solo);

    // Morph membership, and the order it will run in. Click order is the sequence, so
    // the same maps can be morphed forwards, backwards, or out and back, without any
    // separate ordering UI. Absent entirely when the maps are on different grids and
    // there is no honest morph to offer.
    if (info.morph_shape) {
      const at = morph.seq.indexOf(i);
      const ord = document.createElement("button");
      ord.className = "model-ord" + (at >= 0 ? " on" : "");
      ord.textContent = at >= 0 ? String(at + 1) : "+";
      ord.title = at >= 0 ? `Step ${at + 1} of the morph — click to remove`
                          : "Add to the morph (click order sets the sequence)";
      ord.addEventListener("click", (e) => { e.stopPropagation(); toggleMorphMember(i); });
      row.appendChild(ord);
    }
    // Clicking anywhere else on the row makes this the map the contour, colour picker
    // and volume/slice views act on.
    row.addEventListener("click", () => setActive(i));
    list.appendChild(row);
  });
  markActiveRow();
  updateModelCnt();
}

function updateModelCnt() {
  $("modelCnt").textContent = `${visibleModels().length}/${models.length}`;
}

function markActiveRow() {
  for (const row of $("modelsList").children) {
    row.classList.toggle("active", Number(row.dataset.index) === state.active);
  }
}

function toggleSolo(i) {
  const alreadySolo = models[i].visible && visibleModels().length === 1;
  models.forEach((m, k) => {
    m.visible = alreadySolo ? true : k === i;
    renderer.setModelVisible(m.index, m.visible);
    const row = $("modelsList").children[k];
    if (row) row.querySelector("input[type=checkbox]").checked = m.visible;
  });
  if (!alreadySolo) setActive(i);            // the lone visible map is the one you act on
  if (state.render === "surface") loadSurfaces();
  else if (inVolumeMode()) loadVolume();     // the composited set changed
  else reportSurface();
  updateModelCnt();
  schedSaveScene();
}

const inVolumeMode = () => state.render === "mip" || state.render === "emission";

function setVisible(i, on) {
  models[i].visible = on;
  renderer.setModelVisible(models[i].index, on);
  // A map switched on for the first time has no surface yet.
  if (on && state.render === "surface") loadSurfaces();
  else if (inVolumeMode()) loadVolume();     // the composited set changed either way
  else reportSurface();
  updateModelCnt();
  schedSaveScene();
}

function setColor(i, hex) {
  models[i].color = hex;
  renderer.setModelColor(models[i].index, hexToRgb(hex));
  if (i === state.active) $("color").value = hex;
  const slot = volSlots.indexOf(i);
  if (slot >= 0) volRenderer.setSlotColor(slot, hexToRgb(hex));
  saveMapPref(models[i]);
}

// ---- morph ---------------------------------------------------------------
// A morph is a linear blend of two density grids — what ChimeraX's `vop morph` does.
// It is honest and cheap, and it is a cross-fade, not a trajectory: density fades out
// where a domain was and in where it went, rather than the domain travelling. Recovering
// real motion would need a displacement field between the maps, which is a different
// (and much slower) computation; this does not pretend to it.
//
// The two render modes reach the same frames by opposite routes:
//   volume  — both grids go to the GPU once and the shader blends them, so scrubbing
//             and playback run at screen rate with nothing to prepare.
//   surface — every frame needs its own marching-cubes pass (~58 ms draft, ~190 ms full
//             at 128³), far too slow to do live, so Prepare fetches the whole timeline
//             into a cache and playback just uploads meshes.
const MORPH_ID = "__morph";   // renderer model id the morph frames are drawn as

const morph = {
  seq: [],           // model indices, in the order they were picked
  perLeg: 24,        // frames between consecutive maps
  mass: true,        // constant enclosed mass rather than constant sigma
  quality: "draft",  // "draft" -> draftSize(), "full" -> the whole grid
  frame: 0,
  playing: false,
  timer: null,
  cache: null,       // surface mode: parsed mesh per frame
  cacheKey: "",      // what `cache` was built for; a mismatch means Prepare again
  loadedPair: "",    // which (a,b) textures are on the GPU; uploads happen per leg, not per frame
  refMass: -1,       // enclosed mass of frame 0, held constant across the morph
  showing: false,    // is the morph on screen (as opposed to the maps)?
};

const morphLegs = () => Math.max(0, morph.seq.length - 1);
const morphFrames = () => (morphLegs() ? morphLegs() * morph.perLeg + 1 : 0);
// Everything a prepared cache depends on. Changing any of it invalidates the cache.
const morphKey = () => [morph.seq.join(","), morph.perLeg, morph.mass, morph.quality,
                        active().smooth, models[morph.seq[0]] ? models[morph.seq[0]].sigma : 0].join("|");

// Frame k -> which pair, and how far between them.
function frameSpec(k) {
  const legs = morphLegs();
  const leg = Math.min(Math.floor(k / morph.perLeg), legs - 1);
  return { a: morph.seq[leg], b: morph.seq[leg + 1], t: (k - leg * morph.perLeg) / morph.perLeg };
}

function toggleMorphMember(i) {
  const at = morph.seq.indexOf(i);
  if (at >= 0) morph.seq.splice(at, 1); else morph.seq.push(i);
  morph.cache = null;
  stopMorph();
  updateMorphUI();
  buildModelPanel();   // the order badges moved
}

function updateMorphUI() {
  const ok = morphLegs() > 0;
  $("morphSeq").textContent = ok
    ? morph.seq.map((i) => models[i].name.replace(/\.[^.]+$/, "")).join(" → ")
    : "pick maps in Models above";
  $("morphSeq").classList.toggle("muted", !ok);
  const ready = ok && (state.render !== "surface" || (morph.cache && morph.cacheKey === morphKey()));
  $("morphPrepare").disabled = !ok;
  // Volume mode has nothing to prepare, so say so rather than leaving a live-looking
  // button that does almost nothing.
  $("morphPrepare").textContent = state.render === "surface" ? "Prepare" : "Load";
  $("morphPlay").disabled = !ready;
  $("morphRecord").disabled = !ready;
  // The .mrc export computes its frames server-side, so it only needs a sequence.
  $("morphExport").disabled = !ok;
  $("morphScrub").disabled = !ready;
  $("morphScrub").max = String(Math.max(0, morphFrames() - 1));
}

// Draw frame k. In volume mode this is two uniform writes; in surface mode it is a
// cache lookup, both of which are fast enough to scrub with.
async function showMorphFrame(k) {
  if (!morphLegs()) return;
  morph.frame = Math.max(0, Math.min(morphFrames() - 1, k));
  $("morphScrub").value = String(morph.frame);
  const { a, b, t } = frameSpec(morph.frame);
  if (state.render === "surface") {
    const mesh = morph.cache && morph.cache[morph.frame];
    if (!mesh) return;
    renderer.setModelMesh(MORPH_ID, mesh.positions, mesh.normals, mesh.indices);
    renderer.setModelColor(MORPH_ID, hexToRgb(models[morph.seq[0]].color));
  } else {
    const ga = models[a].gridVol, gb = models[b].gridVol;
    if (!ga || !gb) return;
    // Upload the 3D textures only when the frame crosses into a new leg — inside
    // a leg every frame is the same two textures at a different mix, and
    // re-uploading both per frame throttled playback to the PCIe bus.
    const pair = `${a}|${b}`;
    if (morph.loadedPair !== pair) {
      volRenderer.setVolume(ga.nx, ga.ny, ga.nz, ga.data, ga.boxLo, ga.boxHi);
      volRenderer.setVolumeB(gb.nx, gb.ny, gb.nz, gb.data);
      morph.loadedPair = pair;
    }
    volRenderer.setMix(t);
  }
}

// Hide the maps and show the morph instead — they occupy the same space, and drawing
// both would just be the endpoints ghosting through every frame.
function beginMorph() {
  morph.showing = true;
  for (const m of models) renderer.setModelVisible(m.index, false);
}
function stopMorph() {
  pauseMorph();
  morph.loadedPair = "";   // whatever happens next re-uploads its own textures
  if (!morph.showing) return;
  morph.showing = false;
  renderer.dropModel(MORPH_ID);
  volRenderer.setVolumeB(0, 0, 0, null);
  for (const m of models) renderer.setModelVisible(m.index, m.visible);
  if (state.render !== "surface") loadVolume();
}

const MORPH_BUDGET_MB = 512;
const MORPH_BUDGET_BYTES = MORPH_BUDGET_MB * 1024 * 1024;

async function prepareMorph() {
  if (!morphLegs()) return;
  const btn = $("morphPrepare");
  btn.disabled = true;
  try {
    if (state.render !== "surface") {
      // Volume mode: just make sure every map in the sequence is on hand as a grid.
      for (let i = 0; i < morph.seq.length; i++) {
        const m = models[morph.seq[i]];
        status(`Loading ${m.name} (${i + 1}/${morph.seq.length})…`, "busy");
        if (!m.gridVol) m.gridVol = await fetchGrid(m.index, 96);
      }
      beginMorph();
      // The morph runs single-slot; give slot 0 the sequence's colour and contour.
      volRenderer.setColor(hexToRgb(models[morph.seq[0]].color));
      volRenderer.setThreshold(thrFor(models[morph.seq[0]]));
      await showMorphFrame(morph.frame);
      status(`Morph ready · ${morph.seq.length} maps · blended on the GPU`);
      updateMorphUI();
      return;
    }
    // Surface mode: contour every frame up front. Frame 0 also fixes the reference
    // mass, so the whole series is cut at one enclosed mass rather than one sigma.
    const n = morphFrames();
    const cache = new Array(n);
    const smooth = active().smooth;
    const size = morph.quality === "draft" ? `&size=${draftSize()}` : "";
    const sigma = models[morph.seq[0]].sigma;
    const first = await fetchMorphMesh(morph.seq[0], morph.seq[1], 0, `sigma=${sigma.toFixed(3)}`, smooth, size);
    cache[0] = first;
    morph.refMass = first.mass;
    // Byte budget on the cached meshes: the cache is what makes playback instant, but
    // unbounded it OOMs the tab (n frames x multi-MB full-quality meshes). Frame 0
    // prices the series before the long grind; the running total catches variation.
    let cachedBytes = first.bytes;
    if (first.bytes * n > MORPH_BUDGET_BYTES) {
      throw new Error(`~${Math.round(first.bytes * n / (1024 * 1024))} MB of meshes for ${n} frames — ` +
                      `switch Quality to draft or shorten the sequence (budget ${MORPH_BUDGET_MB} MB)`);
    }
    const level = morph.mass ? `mass=${morph.refMass}` : `sigma=${sigma.toFixed(3)}`;
    for (let k = 1; k < n; k++) {
      const { a, b, t } = frameSpec(k);
      status(`Preparing morph… ${k + 1}/${n}`, "busy");
      cache[k] = await fetchMorphMesh(a, b, t, level, smooth, size);
      cachedBytes += cache[k].bytes;
      if (cachedBytes > MORPH_BUDGET_BYTES) {
        throw new Error(`morph cache passed ${MORPH_BUDGET_MB} MB at frame ${k + 1}/${n} — ` +
                        "switch Quality to draft or shorten the sequence");
      }
    }
    morph.cache = cache;
    morph.cacheKey = morphKey();
    beginMorph();
    await showMorphFrame(0);
    const mb = cache.reduce((s, m) => s + m.bytes, 0) / (1024 * 1024);
    status(`Morph ready · ${n} frames · ${mb.toFixed(0)} MB cached`);
  } catch (err) {
    status("Morph preparation failed: " + err.message, "err");
  } finally {
    btn.disabled = false;
    updateMorphUI();
  }
}

async function fetchMorphMesh(a, b, t, level, smooth, size) {
  const url = `./api/morph/decode?a=${models[a].index}&b=${models[b].index}&t=${t.toFixed(5)}&${level}&smooth=${smooth}${size}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error((await res.text()).slice(0, 200));
  const buf = await res.arrayBuffer();
  return { ...parseMesh(buf), bytes: buf.byteLength };
}

function playMorph() {
  if (morph.playing || !morphLegs()) return;
  morph.playing = true;
  $("morphPlay").textContent = "❚❚";
  const step = () => {
    let next = morph.frame + 1;
    if (next >= morphFrames()) next = 0;      // loop, as ChimeraX's morph does
    showMorphFrame(next);
  };
  morph.timer = setInterval(step, 1000 / Math.max(1, +$("morphFps").value || 30));
}
function pauseMorph() {
  if (morph.timer) { clearInterval(morph.timer); morph.timer = null; }
  morph.playing = false;
  $("morphPlay").textContent = "▶";
}

// Record one pass as a video. The frames are already prepared (surface) or on the GPU
// (volume), so the capture runs at the requested rate instead of at the speed of the
// server — which is the whole reason Prepare exists.
async function recordMorph() {
  if (!morphLegs()) return;
  const target = state.render === "surface" ? renderer : volRenderer;
  if (!target.captureStream) { status("Recording is not available for this view.", "err"); return; }
  const fps = Math.max(1, +$("morphFps").value || 30);
  let rec;
  const chunks = [];
  try {
    const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp9") ? "video/webm;codecs=vp9" : "video/webm";
    rec = new MediaRecorder(target.captureStream(fps), { mimeType: mime });
    rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
    rec.start();
  } catch (err) { status("Recording unavailable: " + err.message, "err"); return; }

  pauseMorph();
  const btn = $("morphRecord");
  btn.disabled = true;
  const n = morphFrames();
  await showMorphFrame(0);
  for (let k = 1; k < n; k++) {
    await showMorphFrame(k);
    status(`Recording… ${k + 1}/${n}`, "busy");
    await new Promise((r) => setTimeout(r, 1000 / fps));
  }
  rec.onstop = () => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob(chunks, { type: "video/webm" }));
    a.download = "morph.webm";
    a.click();
    URL.revokeObjectURL(a.href);
    status(`Recording saved · ${n} frames at ${fps} fps`);
  };
  rec.stop();
  btn.disabled = false;
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
  // The morph is prepared per render mode (cached meshes vs uploaded textures), so a
  // mode switch ends it rather than showing a stale or half-built one.
  stopMorph();
  morph.cache = null;
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

  // The segmented switch tracks the mode; volume styles both light the same segment.
  const seg = isVolume ? "volume" : mode;
  for (const b of $("modeSeg").querySelectorAll("button")) {
    b.classList.toggle("on", b.dataset.mode === seg);
  }
  if (isVolume) $("volStyle").value = mode;
  // Contextual controls: only the current mode's rows exist on screen.
  show($("dispSurface"), isSurface);
  show($("dispVolume"), isVolume);
  show($("dispSlices"), isSlices);
  show($("colorlbl"), !isSlices);
  $("secContour").hidden = isSlices;   // contour is meaningless for raw slices
  $("secClip").hidden = isSlices;      // the slab cuts the 3D views only

  if (isSurface) { renderer.render(); loadSurfaces(); }
  else if (isVolume) { volRenderer.setMode(mode === "emission" ? "emission" : "mip"); loadVolume(); }
  else { loadSlices(); }
  // Morphing is a 3D notion; orthogonal slices have nothing to animate.
  $("morph").hidden = !(models.length > 1 && info && info.morph_shape) || isSlices;
  if (!$("morph").hidden) updateMorphUI();
  schedSaveScene();
}

// ---- controls ------------------------------------------------------------
// Mode: a segmented switch. "Volume" resolves to whichever style (MIP / glow) the
// Display section last had, so the segment stays a three-way choice.
let volStyle = "mip";
$("modeSeg").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-mode]");
  if (!btn) return;
  const seg = btn.dataset.mode;
  setMode(seg === "volume" ? volStyle : seg);
});
$("volStyle").addEventListener("change", (e) => {
  volStyle = e.target.value === "emission" ? "emission" : "mip";
  if (state.render === "mip" || state.render === "emission") setMode(volStyle);
});

// Export menu: the four ways out of the viewer, one dropdown.
$("exportBtn").addEventListener("click", (e) => {
  e.stopPropagation();
  $("exportMenu").classList.toggle("open");
});
document.addEventListener("click", (e) => {
  if (!e.target.closest("#exportMenu")) $("exportMenu").classList.remove("open");
});
// Choosing an export closes the menu; changing scale/transparency keeps it open.
for (const id of ["saveMap", "mrc", "exportMesh"]) {
  $(id).addEventListener("click", () => $("exportMenu").classList.remove("open"));
}

// Collapsible sections, open state remembered globally (a workflow preference,
// not a property of one scene).
const SEC_KEY = "hax_mrc_sections";
(function setupSections() {
  const saved = readJSON(SEC_KEY) || {};
  for (const sec of document.querySelectorAll("#models .sec")) {
    const k = sec.dataset.sec;
    if (k in saved) sec.classList.toggle("open", !!saved[k]);
    sec.querySelector(".sec-h").addEventListener("click", () => {
      sec.classList.toggle("open");
      const cur = readJSON(SEC_KEY) || {};
      cur[k] = sec.classList.contains("open");
      writeJSON(SEC_KEY, cur);
      // Canvases inside (the contour histogram) size themselves from their client
      // box, which was 0 while collapsed — give them a layout pass to redraw.
      if (cur[k]) window.dispatchEvent(new Event("resize"));
    });
  }
})();

$("light").addEventListener("change", (e) => renderer.setLighting(e.target.value));
$("smooth").addEventListener("input", (e) => {
  active().smooth = +e.target.value;
  active().surface = null;                     // the cached mesh was cut at the old smoothing
  if (state.render === "surface") loadSurface(state.active, false);
  saveMapPref(active());
});
$("vgamma").addEventListener("input", (e) => volRenderer.setGamma(+e.target.value));
$("vbright").addEventListener("input", (e) => volRenderer.setBrightness(+e.target.value));
$("slicemap").addEventListener("change", (e) => slices.setColormap(e.target.value));
$("slicelo").addEventListener("input", () => slices.setContrast(+$("slicelo").value, +$("slicehi").value));
$("slicehi").addEventListener("input", () => slices.setContrast(+$("slicelo").value, +$("slicehi").value));
$("slicecross").addEventListener("change", (e) => slices.setCrosshair(e.target.checked));
$("sliceReset").addEventListener("click", () => slices.resetZoom());
$("color").addEventListener("input", (e) => {
  setColor(state.active, e.target.value);
  const row = $("modelsList").children[state.active];
  if (row) row.querySelector(".model-color").value = e.target.value;
});
$("bg").addEventListener("input", (e) => { const rgb = hexToRgb(e.target.value); renderer.setBackground(rgb); volRenderer.setBackground(rgb); slices.setBackground(e.target.value); schedSaveScene(); });
$("reset").addEventListener("click", () => { camera = null; renderer.resetView(); volRenderer.resetView(); slices.resetZoom(); });
$("mrc").addEventListener("click", () => { window.location.href = `./api/mrc?m=${active().index}`; });
$("saveMap").addEventListener("click", saveMap);
$("morphFrames").addEventListener("change", (e) => {
  morph.perLeg = Math.max(2, Math.min(120, +e.target.value || 24));
  e.target.value = String(morph.perLeg);
  morph.cache = null; stopMorph(); updateMorphUI();
});
$("morphMass").addEventListener("change", (e) => { morph.mass = e.target.checked; morph.cache = null; updateMorphUI(); });
$("morphQuality").addEventListener("change", (e) => { morph.quality = e.target.value; morph.cache = null; updateMorphUI(); });
$("morphPrepare").addEventListener("click", prepareMorph);
$("morphPlay").addEventListener("click", () => (morph.playing ? pauseMorph() : playMorph()));
$("morphScrub").addEventListener("input", (e) => { pauseMorph(); showMorphFrame(+e.target.value); });
$("morphRecord").addEventListener("click", recordMorph);
$("morphExport").addEventListener("click", () => {
  if (!morphLegs()) return;
  const idx = morph.seq.map((i) => models[i].index).join(",");
  window.location.href = `./api/morph/export?seq=${idx}&per=${morph.perLeg}`;
});
$("morphClear").addEventListener("click", () => { morph.seq = []; morph.cache = null; stopMorph(); updateMorphUI(); buildModelPanel(); });
$("modelsAll").addEventListener("click", () => setAllVisible(true));
$("modelsNone").addEventListener("click", () => setAllVisible(false));
$("exportMesh").addEventListener("click", exportMesh);
$("viewFront").addEventListener("click", () => setStandardView("front"));
$("viewTop").addEventListener("click", () => setStandardView("top"));
$("viewSide").addEventListener("click", () => setStandardView("side"));
$("linkContour").addEventListener("change", (e) => { linkContour = e.target.checked; });
$("dustOn").addEventListener("change", (e) => {
  dust.on = e.target.checked;
  $("dustSize").hidden = !dust.on;
  for (const m of models) m.surface = null;   // every cached mesh was cut without/with dust
  if (state.render === "surface") loadSurfaces();
});
$("dustSize").addEventListener("change", (e) => {
  dust.size = Math.max(1, +e.target.value || 200);
  if (!dust.on) return;
  for (const m of models) m.surface = null;
  if (state.render === "surface") loadSurfaces();
});
$("morphAll").addEventListener("click", () => {
  morph.seq = models.map((m, i) => (m.visible ? i : -1)).filter((i) => i >= 0);
  morph.cache = null;
  stopMorph();
  updateMorphUI();
  buildModelPanel();
});
$("pdbBtn").addEventListener("click", togglePdb);
$("clipChips").addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");
  if (!chip) return;
  clipAx = chip.dataset.ax;
  for (const c of $("clipChips").children) c.classList.toggle("on", c === chip);
  applyClip();
});
$("clipLo").addEventListener("input", applyClip);
$("clipHi").addEventListener("input", applyClip);
$("computeGo").addEventListener("click", runCompute);
$("paintGo").addEventListener("click", paintDiff);
$("paintClear").addEventListener("click", clearPaint);

function setAllVisible(on) {
  models.forEach((m, i) => {
    m.visible = on;
    renderer.setModelVisible(m.index, on);
    const row = $("modelsList").children[i];
    if (row) row.querySelector("input[type=checkbox]").checked = on;
  });
  if (on && state.render === "surface") loadSurfaces();
  else if (inVolumeMode()) loadVolume();
  else reportSurface();
  updateModelCnt();
  schedSaveScene();
}

async function saveMap() {
  const scale = +$("dpi").value, transparent = $("transparent").checked;
  const r = state.render === "surface" ? renderer
    : state.render === "slices" ? slices : volRenderer;
  const blob = await r.snapshot({ scale, transparent });
  if (!blob) return;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  const stem = models.length > 1 ? "maps" : active().name.replace(/\.[^.]+$/, "");
  a.download = `${stem}_${state.render}.png`;
  a.click();
  URL.revokeObjectURL(a.href);
}

function hexToRgb(hex) { const n = parseInt(hex.slice(1), 16); return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255]; }

// ---- standard orientations + axes gizmo ----------------------------------
// Column-major rotation matrices: front = identity; top rotates the model +90°
// about X (its +Y face toward the camera); side −90° about Y (+X face forward).
const VIEW_R = {
  front: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
  top:   [1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1],
  side:  [0, 0, 1, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1],
};

function setStandardView(which) {
  const target = camera3d(state.render);
  if (!target) return;                       // slices have no 3D camera
  const cam = target.getCamera();
  target.setCamera({ ...cam, R: VIEW_R[which].slice(), offset: [0, 0, 0], pivot: [0, 0, 0] });
}

// Small orientation tripod in the corner, redrawn only when the orientation changes.
// Rendered at devicePixelRatio (the CSS box is 64px; the backing store is scaled) so
// the lines stay crisp, and toggleable — some people find any overlay noise.
const GIZMO_KEY = "hax_mrc_gizmo";
let gizmoOn = localStorage.getItem(GIZMO_KEY) !== "off";
let gizmoInvalidate = () => {};
function startGizmo() {
  const cv = $("gizmo");
  const ctx = cv.getContext("2d");
  const CSS = 64, c = CSS / 2, len = c - 12;
  const dpr = Math.min(window.devicePixelRatio || 1, 3);
  cv.width = CSS * dpr; cv.height = CSS * dpr;
  ctx.scale(dpr, dpr);
  let lastKey = "";
  gizmoInvalidate = () => { lastKey = ""; };
  const AXES = [["X", "#e15b5b", 0], ["Y", "#3fae5a", 4], ["Z", "#4a86e8", 8]];
  function draw(R) {
    ctx.clearRect(0, 0, CSS, CSS);
    // Text halo in the background colour so labels read over any map.
    const halo = document.documentElement.dataset.theme === "light"
      ? "rgba(231,236,243,0.9)" : "rgba(4,6,16,0.8)";
    // Back-to-front so nearer axes draw over farther ones.
    const order = AXES.map(([lbl, col, o]) => ({ lbl, col, x: R[o], y: R[o + 1], z: R[o + 2] }))
                      .sort((a, b) => a.z - b.z);
    for (const a of order) {
      const dim = 0.5 + 0.5 * (a.z + 1) / 2;
      const tx = c + a.x * len, ty = c - a.y * len;
      ctx.globalAlpha = dim;
      ctx.strokeStyle = a.col; ctx.lineWidth = 2; ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(c, c); ctx.lineTo(tx, ty); ctx.stroke();
      ctx.fillStyle = a.col;
      ctx.beginPath(); ctx.arc(tx, ty, 2.5, 0, 2 * Math.PI); ctx.fill();
      // The label sits past the tip; a foreshortened axis keeps its label on the dot.
      const r2 = Math.hypot(a.x, a.y);
      const lx = c + a.x * (len + (r2 > 0.35 ? 7 : 0)), ly = c - a.y * (len + (r2 > 0.35 ? 7 : 0));
      ctx.font = "bold 9px system-ui, sans-serif";
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.lineWidth = 3; ctx.strokeStyle = halo; ctx.strokeText(a.lbl, lx, ly);
      ctx.fillText(a.lbl, lx, ly);
    }
    ctx.globalAlpha = 1;
  }
  (function tick() {
    const target = gizmoOn ? camera3d(state.render) : null;
    cv.style.display = target ? "" : "none";
    if (target) {
      const R = target.getCamera().R;
      const key = R[0].toFixed(3) + R[1].toFixed(3) + R[2].toFixed(3)
                + R[4].toFixed(3) + R[5].toFixed(3) + R[6].toFixed(3)
                + R[8].toFixed(3) + R[9].toFixed(3) + R[10].toFixed(3);
      if (key !== lastKey) { lastKey = key; draw(R); }
    }
    requestAnimationFrame(tick);
  })();
}
$("gizmoOn").checked = gizmoOn;
$("gizmoOn").addEventListener("change", (e) => {
  gizmoOn = e.target.checked;
  try { localStorage.setItem(GIZMO_KEY, gizmoOn ? "on" : "off"); } catch { /* storage off */ }
});
startGizmo();

// ---- mesh export (binary STL, in Å) --------------------------------------
// The mesh comes back in the shared [-1,1] cube; undo that mapping (see
// meshing.mesh_at: model = (voxel*step + off - S/2)/(S/2)) to get voxel
// coordinates, then Å via the header's voxel size and origin — so the STL lands
// on the same coordinates the map has in ChimeraX.
async function exportMesh() {
  const m = active();
  const btn = $("exportMesh");
  btn.disabled = true;
  status("Building surface for export…", "busy");
  try {
    const buf = await (await fetch(`./api/decode?m=${m.index}&sigma=${m.sigma.toFixed(3)}&smooth=${m.smooth}`)).arrayBuffer();
    const mesh = parseMesh(buf);
    if (!mesh.indices.length) { status("Empty surface at this contour — nothing to export", "warn"); return; }
    const S = info.full_size, half = S / 2;
    const vs = m.voxel_size || 1;
    // per-axis voxel offsets that centred this map in the common cube, spatial order
    const off = [(S - m.shape[2]) / 2, (S - m.shape[1]) / 2, (S - m.shape[0]) / 2];
    const org = m.origin || [0, 0, 0];
    const p = mesh.positions;
    const nTri = mesh.indices.length / 3;
    const out = new DataView(new ArrayBuffer(84 + nTri * 50));
    new Uint8Array(out.buffer).set(new TextEncoder().encode(`hax-em ${m.name} sigma=${m.sigma.toFixed(2)}`).slice(0, 79), 0);
    out.setUint32(80, nTri, true);
    let o = 84;
    const ang = (i, a) => (p[i * 3 + a] * half + half - off[a]) * vs + org[a];
    for (let t = 0; t < nTri; t++) {
      const [i0, i1, i2] = [mesh.indices[t * 3], mesh.indices[t * 3 + 1], mesh.indices[t * 3 + 2]];
      const v0 = [ang(i0, 0), ang(i0, 1), ang(i0, 2)], v1 = [ang(i1, 0), ang(i1, 1), ang(i1, 2)], v2 = [ang(i2, 0), ang(i2, 1), ang(i2, 2)];
      const ux = v1[0] - v0[0], uy = v1[1] - v0[1], uz = v1[2] - v0[2];
      const wx = v2[0] - v0[0], wy = v2[1] - v0[1], wz = v2[2] - v0[2];
      let nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx;
      const ln = Math.hypot(nx, ny, nz) || 1; nx /= ln; ny /= ln; nz /= ln;
      for (const v of [[nx, ny, nz], v0, v1, v2]) {
        out.setFloat32(o, v[0], true); out.setFloat32(o + 4, v[1], true); out.setFloat32(o + 8, v[2], true);
        o += 12;
      }
      out.setUint16(o, 0, true); o += 2;
    }
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([out.buffer], { type: "model/stl" }));
    a.download = `${m.name.replace(/\.[^.]+$/, "")}_sigma${m.sigma.toFixed(1)}.stl`;
    a.click();
    URL.revokeObjectURL(a.href);
    status(`Exported ${nTri.toLocaleString()} triangles (STL, Å)`);
  } catch (err) { status("Mesh export failed: " + err.message, "err"); }
  finally { btn.disabled = false; }
}

// ---- clipping (slab along one axis, surface + volume) --------------------
let clipAx = "";   // "", "0", "1", "2" — driven by the chips in the Clip section
function applyClip() {
  const axis = clipAx;
  $("clipRanges").style.display = axis === "" ? "none" : "";
  if (axis === "") {
    volRenderer.setClip(null, null); renderer.setClip(null, null);
    $("clipState").textContent = "off";
    return;
  }
  const a = +axis;
  let lo = +$("clipLo").value / 100, hi = +$("clipHi").value / 100;
  if (hi < lo + 0.01) hi = Math.min(1, lo + 0.01);
  $("clipState").textContent = "XYZ"[a] + ` ${Math.round(lo * 100)}–${Math.round(hi * 100)}%`;
  // Volume renderer clips in texture space [0,1]; the surface renderer in the
  // model-space [-1,1] cube. Same fractions, two coordinate frames.
  const tLo = [0, 0, 0], tHi = [1, 1, 1];
  tLo[a] = lo; tHi[a] = hi;
  volRenderer.setClip(tLo, tHi);
  const mLo = [-1.001, -1.001, -1.001], mHi = [1.001, 1.001, 1.001];
  mLo[a] = lo * 2 - 1; mHi[a] = hi * 2 - 1;
  renderer.setClip(mLo, mHi);
}

// ---- map arithmetic ------------------------------------------------------
function fillComputeSelects() {
  for (const id of ["computeA", "computeB"]) {
    const sel = $(id);
    sel.textContent = "";
    models.forEach((m, i) => {
      const opt = document.createElement("option");
      opt.value = String(m.index);
      opt.textContent = m.name.replace(/\.[^.]+$/, "");
      sel.appendChild(opt);
    });
  }
  if (models.length > 1) $("computeB").selectedIndex = 1;
}

async function runCompute() {
  const btn = $("computeGo");
  btn.disabled = true;
  status("Computing…", "busy");
  try {
    const res = await fetch(`./api/compute?op=${$("computeOp").value}&a=${$("computeA").value}&b=${$("computeB").value}`,
                            { method: "POST" });
    if (!res.ok) throw new Error(((await res.json()).detail) || res.statusText);
    const made = await res.json();
    await loadInfo();            // the scene grew a model (or two): rebuild panel + selects
    if (made.pair) {
      // Two-sided difference: the standard colouring — green where A gained
      // density over B, red where it lost it.
      setColor(made.pair[0], "#51cf66");
      setColor(made.pair[1], "#ff6b6b");
    }
    setMode(state.render);
    setActive(made.index);       // show the result
    status(made.pair ? `Added ${made.name} — green = gained, red = lost` : `Added ${made.name}`);
  } catch (err) { status("Compute failed: " + err.message, "err"); }
  finally { btn.disabled = false; }
}

// ---- paint by difference (ChimeraX's `color sample` on A − B) ------------
// The surface of A, coloured by the local value of A − B: red where density was
// lost, near-white where nothing changed, blue where it was gained — the
// continuous difference-map reading, on one surface instead of two lobes.
let painted = null;   // list index of the map wearing a painted surface, or null

async function paintDiff() {
  const a = +$("computeA").value, b = +$("computeB").value;
  if (a === b) { status("Pick two different maps (A and B above) to paint their difference.", "warn"); return; }
  const ma = models[a], mb = models[b];
  if (!ma || !mb) return;
  const btn = $("paintGo");
  btn.disabled = true;
  status(`Painting ${ma.name} by (A − B)…`, "busy");
  try {
    const res = await fetch(`./api/paint?m=${ma.index}&other=${mb.index}`
                            + `&sigma=${ma.sigma.toFixed(3)}&smooth=${ma.smooth}`);
    if (!res.ok) throw new Error(((await res.json()).detail) || res.statusText);
    const mesh = parseColorMesh(await res.arrayBuffer());
    if (!mesh.indices.length) { status("Empty surface at this contour — nothing to paint", "warn"); return; }
    renderer.setModelMesh(ma.index, mesh.positions, mesh.normals, mesh.indices, mesh.colors);
    // The painted mesh IS the surface at this sigma — mark it current so the next
    // loadSurfaces pass does not replace it with a plain one. Any later contour
    // change refetches plain and thereby clears the paint (see surfaceOnce).
    ma.surface = ma.sigma;
    painted = a;
    $("paintClear").hidden = false;
    if (state.render !== "surface") setMode("surface");
    else if (a !== state.active) setActive(a);
    const r = mesh.prange;
    status(`${ma.name} painted by (${ma.name.replace(/\.[^.]+$/, "")} − ${mb.name.replace(/\.[^.]+$/, "")})`
           + ` · range ±${Math.abs(r) >= 0.01 ? r.toFixed(2) : r.toExponential(1)}`
           + " · red = lost · blue = gained");
  } catch (err) { status("Paint failed: " + err.message, "err"); }
  finally { btn.disabled = false; }
}

function clearPaint() {
  if (painted === null) return;
  const i = painted;
  painted = null;
  $("paintClear").hidden = true;
  models[i].surface = null;
  if (state.render === "surface") surfaceOnce(i, false);
}

// ---- PDB backbone overlay ------------------------------------------------
const pdb = { loaded: null };   // { name, chainIds: [] }
const PDB_COLORS = ["#ffd166", "#7fb3ff", "#8fe08f", "#ff9e7a", "#e79aff", "#6fdccf", "#ff8fa8", "#c8c8c8"];

// Choose a server-side file. Embedded in the GUI, this opens the app's real file
// browser (viewerWindow.js answers over postMessage); standalone (the viewer's own
// port, no parent) it falls back to a plain prompt. Resolves to a path or null.
function pickServerFile() {
  if (window.parent === window) {
    return Promise.resolve(window.prompt("Server path of the PDB file to overlay (e.g. /data/model.pdb):"));
  }
  return new Promise((resolve) => {
    const token = Math.random().toString(36).slice(2);
    const onMsg = (e) => {
      const d = e.data;
      if (!d || d.type !== "hax-picked-file" || d.token !== token) return;
      window.removeEventListener("message", onMsg);
      resolve(d.path || null);
    };
    window.addEventListener("message", onMsg);
    window.parent.postMessage({ type: "hax-pick-file", token }, "*");
  });
}

async function togglePdb() {
  if (pdb.loaded) {
    for (const id of pdb.loaded.chainIds) renderer.dropLineModel(id);
    pdb.loaded = null;
    $("pdbBtn").textContent = "PDB…";
    status("PDB overlay removed");
    return;
  }
  const path = await pickServerFile();
  if (!path) return;
  status("Loading PDB…", "busy");
  try {
    const res = await fetch(`./api/pdb?path=${encodeURIComponent(path)}`);
    if (!res.ok) throw new Error(((await res.json()).detail) || res.statusText);
    const data = await res.json();
    // Å -> the shared model cube, via the ACTIVE map's voxel size and origin (that
    // is the frame a fitted model is deposited in).
    const m = active();
    const S = info.full_size, half = S / 2;
    const vs = m.voxel_size || 1;
    const off = [(S - m.shape[2]) / 2, (S - m.shape[1]) / 2, (S - m.shape[0]) / 2];
    const org = m.origin || [0, 0, 0];
    const toModel = (x, a) => ((x - org[a]) / vs + off[a] - half) / half;
    const chainIds = [];
    let nSeg = 0;
    data.chains.forEach((chain, ci) => {
      const pts = chain.xyz;
      const segs = [];
      for (let i = 0; i + 5 < pts.length; i += 3) {
        // break the trace at chain gaps (successive CA >6 Å apart is not bonded)
        const d = Math.hypot(pts[i + 3] - pts[i], pts[i + 4] - pts[i + 1], pts[i + 5] - pts[i + 2]);
        if (d > 6.0) continue;
        segs.push(toModel(pts[i], 0), toModel(pts[i + 1], 1), toModel(pts[i + 2], 2),
                  toModel(pts[i + 3], 0), toModel(pts[i + 4], 1), toModel(pts[i + 5], 2));
      }
      if (!segs.length) return;
      const id = `__pdb_${chain.id}`;
      renderer.setLineModel(id, new Float32Array(segs), hexToRgb(PDB_COLORS[ci % PDB_COLORS.length]));
      chainIds.push(id);
      nSeg += segs.length / 6;
    });
    if (!chainIds.length) throw new Error("no traceable backbone found");
    pdb.loaded = { name: data.name, chainIds, path };
    $("pdbBtn").textContent = "PDB ✕";
    // Fit-in-map: density at the backbone atoms of the ACTIVE map, in σ units.
    // Advisory only, so a failure (atoms outside the box, etc.) does not block
    // the overlay — the message explains itself either way.
    let fitTxt = "";
    try {
      const fres = await fetch(`./api/fit?path=${encodeURIComponent(path)}&m=${m.index}`);
      if (fres.ok) {
        const f = await fres.json();
        fitTxt = ` · fit in ${m.name}: ⟨σ⟩ ${f.mean_sigma}, ${Math.round(f.frac_above_1sigma * 100)}% of atoms > 1σ`;
      } else fitTxt = ` · fit: ${(((await fres.json()).detail) || "unavailable")}`;
    } catch { /* score is optional */ }
    status(`${data.name} · ${data.chains.length} chain${data.chains.length > 1 ? "s" : ""} · ${nSeg.toLocaleString()} segments`
           + fitTxt
           + (state.render !== "surface" ? " · shown in Surface mode" : ""));
  } catch (err) { status("PDB overlay failed: " + err.message, "err"); }
}

// ---- boot ----------------------------------------------------------------
(async function boot() {
  try {
    await loadInfo();
    // (g) Restore the last session for this exact set of files: visibility, active
    // map, render mode, background and camera. Anything malformed is just skipped.
    const saved = readJSON(sceneKey());
    if (saved) {
      if (Array.isArray(saved.visible) && saved.visible.length === models.length) {
        models.forEach((m, i) => {
          m.visible = saved.visible[i] !== false;
          renderer.setModelVisible(m.index, m.visible);
          const row = $("modelsList").children[i];
          if (row) row.querySelector("input[type=checkbox]").checked = m.visible;
        });
      }
      if (saved.bg && /^#[0-9a-f]{6}$/i.test(saved.bg)) {
        $("bg").value = saved.bg;
        const rgb = hexToRgb(saved.bg);
        renderer.setBackground(rgb); volRenderer.setBackground(rgb); slices.setBackground(saved.bg);
      }
      if (Number.isInteger(saved.active) && saved.active >= 0 && saved.active < models.length) {
        await setActive(saved.active);
      }
    }
    const mode = saved && ["surface", "mip", "emission", "slices"].includes(saved.mode)
      ? saved.mode : "surface";
    setMode(mode);
    if (saved && saved.camera && Array.isArray(saved.camera.R)) {
      const target = camera3d(mode);
      if (target) target.setCamera(saved.camera);
      camera = saved.camera;   // carried across future mode switches too
    }
    sceneRestored = true;      // from here on, changes are worth saving
  } catch (err) {
    status("Failed to load map: " + err.message, "err");
  }
})();
