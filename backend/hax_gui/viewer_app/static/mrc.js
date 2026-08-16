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
import { createVolume, smoothGrid } from "./volume.js";
import { createHistogram } from "./histogram.js";
import { createSlices } from "./slices.js";
import { parseColorMesh, parseGrid, parseMesh } from "./wire.js";
import { latestWins, makeLive } from "./viewerbase.js";
import { createMcEngine } from "./mcengine.js";
import { massCurve, massAbove, levelForMass, blendedMassCurve } from "./mccore.js";

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
  syncBudgetInputs();   // the GPU-derived default is only knowable once info is in
  const prefs = readJSON(MAP_PREFS_KEY) || {};
  // Reloading the scene (map arithmetic grows it) must not undo the visibility
  // you set up — carry it over by file; only genuinely new models default to on.
  const prevVis = new Map(models.map((m) => [m.filename, m.visible]));
  models = info.models.map((m, i) => ({
    ...m,
    visible: prevVis.has(m.filename) ? prevVis.get(m.filename) : true,
    color: (prefs[m.filename] && prefs[m.filename].color) || m.color,
    sigma: (prefs[m.filename] && Number.isFinite(prefs[m.filename].sigma)) ? prefs[m.filename].sigma : 2.5,
    smooth: (prefs[m.filename] && Number.isFinite(prefs[m.filename].smooth)) ? prefs[m.filename].smooth : 6,
    surface: null,     // the sigma the loaded mesh was cut at, so it isn't refetched
  }));
  $("title").textContent = info.title || "Map";
  for (const m of models) renderer.setModelVisible(m.index, m.visible);
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
  // The slider is shared with iso's field smoothing, so a map switch that moves it has to
  // move the renderer with it — otherwise the control and the picture disagree.
  if (state.render === "iso") volRenderer.setSmoothing(ISO_SIGMA_PER_STEP * m.smooth);
  $("sigVal").textContent = m.sigma.toFixed(2) + " σ";
  if (models.length > 1) markActiveRow();
  schedSaveScene();
  await loadHistogram();
  if (inVolumeMode()) loadVolume();
  else if (state.render === "slices") loadSlices();
}

async function loadHistogram() {
  const m = active();
  const h = await (await fetch(`./api/histogram?m=${m.index}`)).json();
  hist.setData(h, h.mean + m.sigma * h.std);
}

// `signal` lets a superseded run stop its download. A grid is 65 MB at a native box and the
// browser only opens six connections to a host, so a run that has already been overtaken goes
// on occupying one of them for seconds — the newest-wins guards drop its payload, but only
// after it has finished arriving, which is exactly the wrong order.
async function fetchGrid(m, size, bits = 8, signal) {
  const url = `./api/grid?m=${m}&size=${size}` + (bits === 16 ? "&bits=16" : "");
  const g = parseGrid(await (await fetch(url, { signal })).arrayBuffer());
  g.size = size;   // the cached grid has to remember what it was cut at (see volWant)
  return g;
}
// Did this reject because someone cancelled it? Then the run that cancelled it has already
// decided what happens next, and there is nothing to report.
const aborted = (err) => !!err && err.name === "AbortError";

// What the current render style wants out of /api/grid.
//
// MIP and glow integrate along the ray: 96³ uint8 is already smooth for them, and it always
// was. An isosurface instead *shades* the grid, so both its resolution and its quantisation
// are visible — 96³ is blobby, and 8-bit plateaus terrace it. Where the renderer can hold
// float16 textures, iso therefore takes the map at its native box (capped at 384) in 16 bits
// and shades it as it is. Where it cannot, it falls back to exactly what it did before:
// 256³ uint8 with the de-banding blur (which costs real features — it exists only to hide
// the quantisation, so 16-bit grids must NOT go through it).
const ISO_GRID_U8 = 256;      // fallback: as fine as the uint8 payload is worth taking
const ISO_MAX = 384;          // backend cap for 16-bit requests
// A morph keeps EVERY map of the sequence resident on the GPU at once, so unlike a single map
// it has to be priced before it is fetched. Short sequences fit at native resolution and
// should have it — capping them at 256 was why an iso morph looked coarse next to a surface
// one, and it also threw away the native grids single-map viewing had already downloaded.
// Long sequences do not fit and are capped, which is a real loss of detail but the honest
// alternative to a driver refusing the allocation halfway through.
const ISO_MORPH_MAX = 256;                 // the capped size, when the budget says so
// How much VRAM one morph may hold, and how much RAM the surface morph's mesh cache may.
//
// Neither is knowable from inside the page: WebGL will not say how much video memory is free,
// and the browser will not say how much RAM it can spare (navigator.deviceMemory is capped at
// 8 and rounded to powers of two, which is worse than a stated guess). So: the server's card,
// if it reported one, else a fixed number — and in both cases the user can overrule it, which
// is the only thing that is correct when the browser and the server are different machines.
const ISO_POOL_FALLBACK_MB = 768, MORPH_CACHE_FALLBACK_MB = 512;
const BUDGET_KEY = "hax_mrc_budgets";      // global preference, not per scene
const budgetPrefs = () => readJSON(BUDGET_KEY) || {};
function isoPoolBudgetMB() {
  const override = budgetPrefs().pool;
  if (Number.isFinite(override) && override > 0) return override;
  // Half of what the card has free: a morph is not the only thing on the GPU (the maps
  // themselves, the browser's own compositor, whatever else the machine is running).
  const g = info && info.gpu_mem;
  if (g && Number.isFinite(g.free_mb)) return Math.max(512, Math.min(4096, Math.round(0.5 * g.free_mb)));
  return ISO_POOL_FALLBACK_MB;
}
function morphBudgetMB() {
  const override = budgetPrefs().cache;
  return (Number.isFinite(override) && override > 0) ? override : MORPH_CACHE_FALLBACK_MB;
}
const morphBudgetBytes = () => morphBudgetMB() * 1024 * 1024;
// Bytes a map costs in the pool: the f16 density grid plus its R8 opacity volume and mips
// (1 + 1/8 + 1/64 + ... = 1.15). Mirrors the estimate buildMorphPool reports.
const poolBytesFor = (box) => box ** 3 * (2 + 1.15);
// Does the whole sequence fit at native resolution?
function poolFitsNative() {
  let bytes = 0;
  for (const i of morph.seq) bytes += poolBytesFor(Math.min(ISO_MAX, Math.max(...models[i].shape)));
  const budget = isoPoolBudgetMB();
  return { fits: bytes <= budget * 1024 * 1024, mb: Math.round(bytes / (1024 * 1024)), budget };
}
function volWant(m, cap) {
  // Surface mode wants the same payload iso does: it is the grid the browser contours from
  // (see localContourGrid), and sharing it means switching between the two modes costs no
  // download at all.
  if (state.render !== "iso" && state.render !== "surface") return { size: 96, bits: 8 };
  if (!volRenderer.hasF16) return { size: ISO_GRID_U8, bits: 8 };
  return { size: Math.min(cap || ISO_MAX, Math.max(...m.shape)), bits: 16 };
}
// A 128³ stand-in, fetched only for a map that has no grid at all yet: ~4 MB and a fraction of
// a second, against ~65 MB and several seconds for a native one. It lives in its OWN field, so
// the cache-by-size logic above, the identity skip in the renderer and the morph pool are all
// completely unaware of it; it is dropped the moment the real grid lands.
const ISO_DRAFT = 128;
async function volDraft(m, signal) {
  if (m.gridVolDraft) return m.gridVolDraft;
  m.gridVolDraft = volRenderer.hasF16 ? await fetchGrid(m.index, ISO_DRAFT, 16, signal)
                                      : await fetchGrid(m.index, 96, 8, signal);
  return m.gridVolDraft;
}
// Is the cached grid the payload the current style wants — same size AND same sample type?
function volGridReady(m) {
  const w = volWant(m);
  return !!m.gridVol && m.gridVol.size === w.size && !!m.gridVol.f16 === (w.bits === 16);
}

// One download per (map, payload), however many callers ask for it.
//
// Two callers wanting the same grid at the same moment is now the normal case rather than a
// corner: the volume loader fetches what it needs to draw while the prefetch queue
// independently decides the same map is worth having resident. Both used to find the cache
// stale and both used to download 65 MB, and last writer won — wasteful, but harmless,
// because the two payloads were identical.
//
// Cancellation is what makes that no longer harmless: one caller giving up must not take the
// other caller's download with it. So the fetch is shared and it is abandoned only when every
// caller that asked for it has walked away. The prefetch queue asks WITHOUT a signal, which is
// the whole statement "this download is never wasted" — it counts as a consumer that never
// leaves, so a grid the queue wants finishes arriving no matter who else loses interest.
function sharedGridFetch(m, want, signal) {
  const key = `${want.size}:${want.bits}`;
  let live = m.gridPending;
  // A record whose controller has already fired is doomed — its rejection has simply not been
  // delivered yet — so a caller arriving in that window starts a fresh fetch rather than
  // joining one that is on its way to throwing at it.
  if (!live || live.key !== key || live.ctrl.signal.aborted) {
    const ctrl = new AbortController();
    live = m.gridPending = { key, ctrl, wanted: 0, promise: null };
    // Clear the slot on settle, but only if it is still this fetch: a later caller wanting a
    // different payload has already replaced it, and must not be un-registered by an older one.
    const done = () => { if (m.gridPending === live) m.gridPending = null; };
    live.promise = fetchGrid(m.index, want.size, want.bits, ctrl.signal)
      .then((g) => { done(); return g; }, (err) => { done(); throw err; });
  }
  live.wanted++;
  if (signal) {
    const leave = () => { if (--live.wanted <= 0) live.ctrl.abort(); };
    if (signal.aborted) leave();                                  // superseded before it even asked
    else signal.addEventListener("abort", leave, { once: true });
  }
  return live.promise;
}

// The cached grid is reusable only if it is the payload the current style wants — same size
// AND same sample type. The smoothing tag stops a uint8 grid from being blurred twice, which
// would soften it for good.
//
// An aborted fetch rejects out of here without touching m.gridVol, so a cancelled run leaves
// the map exactly as it found it — with its previous grid, or with none.
async function volGrid(m, cap, signal) {
  const want = volWant(m, cap);
  const cached = m.gridVol;
  if (!cached || cached.size !== want.size || !!cached.f16 !== (want.bits === 16)) {
    m.gridVol = await sharedGridFetch(m, want, signal);
  }
  const g = m.gridVol;
  if (state.render === "iso" && !g.f16 && !g.smoothed) {
    g.data = smoothGrid(g.data, g.nx, g.ny, g.nz);
    g.smoothed = true;
  }
  return g;
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
// the GPU and their contours have not changed. A painted map re-paints instead of
// falling back to a plain mesh: the colouring belongs to the map, not to one level,
// so contour and smoothing changes keep it (the paint endpoint contours + samples in
// one pass, so this costs the same as a plain decode). Hide-dust is the one option a
// painted surface ignores — the paint endpoint has no dust pass.
// Contour in the browser, off the grid this map already has resident. This is the whole
// point of the local path: no request, no marching-cubes queue behind other maps, no
// multi-megabyte mesh over the network — a drag becomes a sequence of local marches that the
// worker pool can actually keep up with, and several maps march at once because the pool is
// several threads.
async function contourLocal(m, g, draft) {
  status(models.length > 1 ? `Contouring ${m.name}…` : "Contouring…", "busy");
  const key = mcKey(m, g);
  if (!mc.has(key)) mc.store(key, g);
  const opts = { level: thrFor(m, g),                       // the same normalized contour iso uses
                 smooth: draft ? Math.min(m.smooth, DRAFT_SMOOTH) : m.smooth,
                 stride: draft ? 2 : 1 };
  let res = await mc.contour(key, opts);
  // The engine's memory budget can evict a field between requests, and a worker that died
  // takes its fields with it; the payload is still here either way, so put it back and ask
  // once more rather than falling back to the server for a grid we already hold.
  if (res && res.miss) { mc.store(key, g); res = await mc.contour(key, opts); }
  if (!res || res.miss) return;                             // superseded by a newer contour
  if (res.error) { status("Contour failed: " + res.error, "err"); return; }
  if (state.render !== "surface") return;                   // mode switched while we marched
  renderer.setModelMesh(m.index, res.positions, res.normals, res.indices);
  renderer.setModelColor(m.index, hexToRgb(m.color));
  renderer.setModelVisible(m.index, m.visible);
  if (!draft) m.surface = m.sigma;                          // only a full mesh settles the map
  if (!res.indices.length) { status(`Empty surface at this contour (${m.name})`, "warn"); return; }
  reportSurface();
}

// Fetch the grid a map would be contoured from, in the background: the first show still goes
// through the server (nothing is resident yet), but every contour after the grid lands is
// local — which is the moment the contour slider stops being a request-per-drag and becomes
// live.
//
// Queued, one at a time, and never while a contouring pass is running. Firing these off as
// each map finished contouring — the obvious way, and what round 11 did — put ten 65 MB grid
// downloads onto the same connection pool and the same server as the contour requests that
// had not happened yet, so showing ten maps got SLOWER than before there was any prefetch at
// all. The maps on screen come first; the grids are what makes the *next* interaction fast,
// and they can wait for the screen to settle.
const prefetchQ = [];
let prefetchRunning = false;
let surfaceRuns = 0;            // how many loadSurfaces passes are currently live
let volumeRuns = 0;             // ...and loadVolume passes, which own the connection just as much
// Is a resident grid worth having at all right now? volWant only asks for the payload the
// local contourer and the shaded-iso mode use while one of those is on screen; in MIP or glow
// it asks for a 96³ stand-in, and draining the queue there would REPLACE a map's native grid
// with a coarse one. So the queue holds its contents and waits for a mode that wants them.
const prefetchWorth = () => state.render === "surface" || state.render === "iso";
// Nothing on screen is loading: the queue may have the connection.
const prefetchIdle = () => !surfaceRuns && !volumeRuns;
function queuePrefetch(m) {
  if (!mc.available || dust.on || m.gridFetching || volGridReady(m)) return;
  if (!prefetchQ.includes(m)) prefetchQ.push(m);
  if (prefetchIdle()) runPrefetch();
}
// Every map at once, visible ones first. The first contour of a map used to be what queued it,
// so a map you had not touched yet always went through the server once before it could ever be
// local, and a map you had merely switched off never became local at all — even though
// switching it back on is exactly the interaction this is meant to make instant. Queuing costs
// nothing: the gates below decide when any of it actually runs.
function queueAllPrefetch() {
  if (!prefetchWorth()) return;
  for (const m of visibleModels()) queuePrefetch(m);
  for (const m of models) if (!m.visible) queuePrefetch(m);
}
async function runPrefetch() {
  if (prefetchRunning) return;
  prefetchRunning = true;
  try {
    while (prefetchQ.length) {
      // A contouring pass or a volume load has the connection; leave the rest queued for its
      // end. Same for a mode that does not want these payloads — the queue keeps its contents
      // and queueAllPrefetch restarts it when the user comes back to a mode that does.
      if (!prefetchIdle() || !prefetchWorth()) return;
      const m = prefetchQ.shift();
      // Skip anything that stopped being worth fetching while it waited. Being HIDDEN is not
      // one of those things: the grid a map needs is the same whether it is on screen or not,
      // and having it ready is what makes switching the map on instant rather than a download.
      if (!m || dust.on || volGridReady(m) || m.gridFetching) continue;
      m.gridFetching = true;
      try {
        const g = await volGrid(m);
        // Hand it to the worker pool while nothing is waiting on it. The first local contour
        // would otherwise open with a 65 MB structured clone and a ~100 ms decode on the
        // worker's thread — paid here, in the idle stretch this queue exists for, it is free.
        if (g && mc.available && !dust.on) { const key = mcKey(m, g); if (!mc.has(key)) mc.store(key, g); }
      } catch { /* the server path keeps working */ }
      finally { m.gridFetching = false; }
    }
  } finally { prefetchRunning = false; }
}

// `signal` belongs to the loadSurfaces pass this call is part of, and is undefined for the
// single-map paths (the contour drag, and the one-map repaint after map arithmetic): those
// coalesce through makeLive instead, and a mesh they asked for is a mesh they still want.
async function surfaceOnce(idx, draft, signal) {
  const m = models[idx];
  if (!m) return;
  const local = localContourGrid(m);
  if (local) return contourLocal(m, local, draft);
  status(models.length > 1 ? `Contouring ${m.name}…` : "Contouring…", "busy");
  try {
    const detail = draft ? `&smooth=${Math.min(m.smooth, DRAFT_SMOOTH)}&size=${draftSize()}`
                         : `&smooth=${m.smooth}`;
    let mesh;
    if (painted && painted.m === idx && models[painted.other]) {
      const res = await fetch(`./api/paint?m=${m.index}&other=${models[painted.other].index}`
                              + `&sigma=${m.sigma.toFixed(3)}${detail}`, { signal });
      if (!res.ok) throw new Error(((await res.json()).detail) || res.statusText);
      mesh = parseColorMesh(await res.arrayBuffer());
    } else {
      const buf = await (await fetch(`./api/decode?m=${m.index}&sigma=${m.sigma.toFixed(3)}`
                                     + `${dustParam()}${detail}`, { signal })).arrayBuffer();
      mesh = parseMesh(buf);
    }
    if (state.render !== "surface") return;   // mode switched while we were waiting
    renderer.setModelMesh(m.index, mesh.positions, mesh.normals, mesh.indices, mesh.colors);
    renderer.setModelColor(m.index, hexToRgb(m.color));
    renderer.setModelVisible(m.index, m.visible);
    if (!draft) m.surface = m.sigma;
    if (!mesh.indices.length) { status(`Empty surface at this contour (${m.name})`, "warn"); return; }
    reportSurface();
    // The map is on screen; queue its grid so the next contour of it is local. The queue does
    // not run while a contouring pass is live, so this cannot compete with the maps still
    // waiting to appear.
    if (!draft) queuePrefetch(m);
  } catch (err) {
    if (aborted(err)) return;      // a newer pass took over and cancelled this download
    status("Contour failed: " + err.message, "err");
  }
}
const loadSurface = makeLive(surfaceOnce);

function reportSurface() {
  const shown = visibleModels().length;
  status(models.length > 1
    ? `Surface · ${shown}/${models.length} shown · ${active().name} σ ${active().sigma.toFixed(2)}`
    : `Surface · σ ${active().sigma.toFixed(2)}`);
}

// Newest-run-wins for the two-pass contouring below: the refine pass is long enough that a
// visibility change can easily land in the middle of it, and a stale run must not go on
// pushing meshes for a set of maps that is no longer on screen.
//
// The guard alone only stops a stale run from PAINTING. Its downloads carried on to the end —
// a full mesh is ~21 MB — holding one of the six connections the browser will open to this
// host and delaying the pass the user is actually waiting for. The controller cancels them
// outright: one per run, aborted by whichever run takes the token next.
const surfGuard = latestWins();
let surfAbort = null;

// Contour every visible map that has not been contoured yet (on open, and whenever a map is
// switched back on). Two quite different jobs, split by whether the map's grid is already in
// the browser.
//
// Maps that still have to go through the server are contoured DRAFT FIRST, then refined, and
// both passes are sequential: each request is a marching-cubes pass server-side and firing
// twenty at once just queues them behind each other with less feedback. The split is what
// fixes the wait there: a full pass is ~750 ms and ~21 MB per map at a 320³ box, so ten maps
// meant many seconds of "Contouring…" with an unchanged canvas, while a draft is ~80 ms and
// ~4 MB. The second pass replaces each draft in place (the server scales drafts into the same
// cube, which is what makes them refine without jumping — the contour drag has relied on that
// since it was written).
//
// Maps whose grid is resident take neither. They contour in the worker pool, which is several
// threads and no network at all, so there is nothing to serialise them for and nothing a draft
// would buy: they all go at once, at full resolution, and land as they finish. The two groups
// also run alongside each other, because they contend for nothing — one is cores, the other is
// the connection.
//
// Only the full mesh stamps m.surface, so an interrupted refine simply retries next time.
async function loadSurfaces() {
  if (morph.showing) return;   // the maps are hidden; the morph owns the scene
  const todo = visibleModels().filter((m) => m.surface !== m.sigma);
  if (!todo.length) { reportSurface(); return; }
  const tok = surfGuard.take();
  if (surfAbort) surfAbort.abort();             // the pass we just superseded stops downloading
  surfAbort = new AbortController();
  const signal = surfAbort.signal;
  const live = () => state.render === "surface" && !surfGuard.stale(tok);
  surfaceRuns++;                                // pauses the background grid prefetch
  try {
    const resident = [], viaServer = [];
    for (const m of todo) (localContourGrid(m) ? resident : viaServer).push(m);
    let done = 0;
    const localPass = Promise.all(resident.map(async (m) => {
      if (!live()) return;
      await surfaceOnce(models.indexOf(m), false);
      done++;
      // contourLocal writes its own "Contouring <name>…" as each starts, so this only has to
      // keep a count of the ones that have landed.
      if (live() && resident.length > 1) status(`Contouring… (${done}/${resident.length})`, "busy");
    }));
    const serverPass = (async () => {
      for (const m of viaServer) {                // pass 1: something on screen, fast
        await surfaceOnce(models.indexOf(m), true, signal);
        if (!live()) return;
      }
      for (const m of viaServer) {                // pass 2: the real thing, in place
        if (m.surface === m.sigma) continue;      // a live single-map load got there first
        await surfaceOnce(models.indexOf(m), false, signal);
        if (!live()) return;
      }
    })();
    await Promise.all([localPass, serverPass]);
    if (!live()) return;
    reportSurface();
  } finally {
    // Whatever happened, the screen is as settled as this run will make it: let the queued
    // grids have the connection now.
    surfaceRuns--;
    if (prefetchIdle()) runPrefetch();
  }
}

// Which model (by list index) each renderer slot holds, INDEXED BY SLOT: volSlots[s] is the
// model in slot s, or -1 for an empty slot. Live contour/colour changes address slots through
// this instead of re-uploading textures.
//
// Slot assignment is sticky. It used to be `show.map(...)`, which repacked the slots in
// visibility order, so hiding one map of three moved the other two into different slots and
// forced a full re-upload (and re-blur, and re-bake) of maps that had not changed at all —
// seconds of work to toggle a checkbox. Keeping a map where it already is makes the renderer's
// identity check (same array, same dims, same blur) hit, and the toggle costs nothing.
let volSlots = [];

// The normalized threshold a map's own sigma corresponds to over its own grid range. The
// grid is a parameter because a map on screen as a draft must be contoured against the
// DRAFT's range, not against a grid it does not have yet.
function thrFor(m, g = m.gridVol) {
  if (!g) return 0.45;
  const level = m.mean + m.sigma * (m.std || 1);
  return Math.max(0, Math.min(1, (level - g.vmin) / ((g.vmax - g.vmin) || 1)));
}

// The signed difference grid of the painted pair, cached on the painted state
// (it dies with "clear"). Symmetric range: uint8 128 = zero difference.
async function fetchPaintGrid(size) {
  const url = `./api/paint/grid?m=${models[painted.m].index}&other=${models[painted.other].index}&size=${size}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(((await res.json()).detail) || res.statusText);
  return parseGrid(await res.arrayBuffer());
}

// Is the painted difference what this mode should show? Only when the painted map
// is exactly what is on screen — mixing a signed field into a composite of
// ordinary maps would be nonsense.
const paintedAlone = () => !!painted && visibleModels().length === 1 && visibleModels()[0] === models[painted.m];

// Place the maps that should be on screen into renderer slots, keeping every map that is
// already resident exactly where it is and filling the gaps with the newcomers. Returns an
// array of length maxSlots holding model indices, -1 where empty.
function assignSlots(want) {
  const n = volRenderer.maxSlots;
  const next = new Array(n).fill(-1);
  for (let sIdx = 0; sIdx < n; sIdx++) {
    const cur = volSlots[sIdx];
    if (cur >= 0 && want.includes(cur)) next[sIdx] = cur;   // still wanted: do not move it
  }
  for (const mi of want) {
    if (next.includes(mi)) continue;
    const free = next.indexOf(-1);
    if (free < 0) break;                                     // more maps than slots (caller capped it)
    next[free] = mi;
  }
  return next;
}

// Local contouring. The engine owns the worker pool and decides which worker holds what;
// `mcKey` is the identity of a decoded field inside it — model, resolution and sample type —
// so a grid refetched at another size is a different field and can never be mistaken for the
// old one.
const mc = createMcEngine();
const mcKey = (m, g) => `m${m.index}:${g.nx}x${g.ny}x${g.nz}:${g.f16 ? 16 : 8}`;
// Can this map be contoured in the browser? It needs the worker, a resident grid at the
// resolution the volume modes use, and none of the two features only the server has: dust
// removal (a connected-component pass over the whole volume) and painting (sampling a second
// map onto the surface).
function localContourGrid(m) {
  if (!mc.available || dust.on) return null;
  if (painted && models[painted.m] === m) return null;
  return volGridReady(m) ? m.gridVol : null;
}

const volGuard = latestWins();
// The downloads the newest volume run owns, so a superseded one can be told to stop. Volume
// grids are the biggest thing this viewer pulls — 65 MB each, several per run — and until now
// a run that had already been overtaken went on pulling all of them and only had its payloads
// discarded at the guard, which is to say after they had cost everything they were going to.
let volAbort = null;
// Take the newest-wins token and the run's cancellation together, so the two can never drift
// apart: whoever takes the token is by definition the run whose downloads should survive.
function volRun() {
  if (volAbort) volAbort.abort();
  volAbort = new AbortController();
  return { tok: volGuard.take(), signal: volAbort.signal };
}
async function loadVolume() {
  if (morph.showing) return;     // the morph owns the volume canvas
  if (paintedAlone()) return loadDiffVolume();
  volRenderer.setDiverging(false);
  // Every visible map rides in one composited march, each at its own contour and
  // colour; with nothing visible, fall back to the active map so the canvas is
  // never silently empty.
  let show = visibleModels();
  if (!show.length) show = [active()];
  const dropped = Math.max(0, show.length - volRenderer.maxSlots);
  if (dropped) show = show.slice(0, volRenderer.maxSlots);
  status("Loading volume…", "busy");
  const { tok, signal } = volRun();   // changing the shown set mid-download: newest wins
  // Push whatever each map currently has — its real grid if it has arrived, otherwise its
  // draft. Sticky slots plus the renderer's identity check mean a map whose grid did not
  // change costs nothing here, so this is cheap to call repeatedly as the fetches land.
  const pushSlots = () => {
    volSlots = assignSlots(show.map((m) => models.indexOf(m)));
    // Indexed by slot, with holes: an empty slot is `null`, not a shorter array, so the maps
    // that stayed keep their textures.
    volRenderer.setSlots(volSlots.map((mi) => {
      if (mi < 0) return null;
      const m = models[mi], g = m.gridVol || m.gridVolDraft;
      if (!g) return null;
      return { nx: g.nx, ny: g.ny, nz: g.nz, data: g.data, lo: g.boxLo, hi: g.boxHi,
               color: hexToRgb(m.color), threshold: thrFor(m, g) };
    }));
  };
  volumeRuns++;                  // pauses the background grid prefetch, as a contour pass does
  try {
    const need = show.filter((m) => !volGridReady(m));
    // Draft pass, iso only: a map with nothing to show at all gets a 128³ stand-in first, so
    // ten maps appear in about a second between them instead of after every native grid has
    // crossed the network. MIP and glow fetch 96³ anyway — a draft would be the same request.
    if (state.render === "iso") {
      const blank = need.filter((m) => !m.gridVol);
      for (let i = 0; i < blank.length; i++) {
        status(`Loading volume… draft ${i + 1}/${blank.length}`, "busy");
        await volDraft(blank[i], signal);
        if (volGuard.stale(tok)) return;
      }
      if (blank.length) pushSlots();
    }
    // Then the real grids, one at a time, each pushed as it arrives: the wait is visible as
    // maps sharpening one by one rather than as a status line.
    for (let i = 0; i < need.length; i++) {
      const m = need[i];
      status(need.length > 1 ? `Loading volume… ${i + 1}/${need.length}` : "Loading volume…", "busy");
      await volGrid(m, undefined, signal);
      if (volGuard.stale(tok)) return;
      delete m.gridVolDraft;      // the real thing is resident; ~4 MB of stand-in is dead weight
      pushSlots();
    }
    if (!need.length) pushSlots();
    const g0 = show[0].gridVol || show[0].gridVolDraft;
    if (!g0) return;
    status(show.length > 1
      ? `Volume · ${show.length} maps composited`
        + (dropped ? ` · ${dropped} hidden (max ${volRenderer.maxSlots})` : "")
      : `Volume ${g0.nx}×${g0.ny}×${g0.nz} · ${show[0].name}`);
  } catch (err) {
    if (!aborted(err)) status("Volume load failed: " + err.message, "err");
  } finally {
    volumeRuns--;
    if (prefetchIdle()) runPrefetch();
  }
}

// Volume rendering of the painted difference: the signed A − B grid ray-marched
// two-sided — red where density was lost, blue where it was gained, the same
// reading as the painted surface. The contour threshold does not apply here.
async function loadDiffVolume() {
  status("Loading difference volume…", "busy");
  // Through volRun even though the difference grid is a small one and is not itself
  // cancellable: taking the token is what supersedes an ordinary volume load, and that load's
  // native grids are exactly the downloads worth stopping.
  const { tok } = volRun();
  try {
    if (!painted.gridVol) painted.gridVol = await fetchPaintGrid(96);
    if (volGuard.stale(tok) || !painted) return;
    const g = painted.gridVol;
    volSlots = [painted.m];      // setVolume drives slot 0 alone; the rest are inactive
    volRenderer.setVolume(g.nx, g.ny, g.nz, g.data, g.boxLo, g.boxHi);
    volRenderer.setDiverging(true);
    status(`Difference volume · ${models[painted.m].name} − ${models[painted.other].name}`
           + " · red = lost · blue = gained");
  } catch (err) { status("Difference volume failed: " + err.message, "err"); }
}

const sliceGuard = latestWins();
async function loadSlices() {
  const m = active();
  const diffMode = !!painted && state.active === painted.m;
  status(diffMode ? "Loading difference slices…" : "Loading slices…", "busy");
  const tok = sliceGuard.take();
  try {
    let g;
    if (diffMode) {
      // Slices of the signed A − B grid, on a symmetric range so mid-gray = zero;
      // force the diverging colormap so red = lost / blue = gained reads at once.
      if (!painted.gridSlice) painted.gridSlice = await fetchPaintGrid(192);
      if (sliceGuard.stale(tok) || !painted) return;
      g = painted.gridSlice;
      $("slicemap").value = "diverging";
      slices.setColormap("diverging");
    } else {
      if (!m.gridSlice) m.gridSlice = await fetchGrid(m.index, 192);
      if (sliceGuard.stale(tok)) return;
      g = m.gridSlice;
    }
    // Å per *grid* voxel: the slice grid is stride-downsampled, so scale the header
    // voxel size by the stride. Set before the volume so the legend can use it.
    const stride = Math.max(1, Math.round(Math.max(...m.shape) / Math.max(g.nx, g.ny, g.nz)));
    slices.setScale(m.voxel_size && m.voxel_size !== 1 ? m.voxel_size * stride : 0);
    slices.setVolume(g);
    status(diffMode
      ? `Difference slices · ${models[painted.m].name} − ${models[painted.other].name} · red = lost · blue = gained`
      : `Slices ${g.nx}×${g.ny}×${g.nz} · ${m.name}` + (models.length > 1 ? " · active map only" : ""));
  } catch (err) { status("Slice load failed: " + err.message, "err"); }
}

// ---- constant-mass contours for a morph, computed client-side --------------------------
//
// The surface morph contours every frame with meshing.level_for_mass, so the enclosed mass is
// held constant and the shape comparison is fair; the volume morph never did, which is why
// its surface visibly breathed as the blend crossed a map whose variance differed. These are
// the same arithmetic, on the grid the GPU is actually showing.
//
// "Mass" is the server's enclosed_mass: the SUM OF RAW VOXEL VALUES at or above the level —
// not a voxel count, not sum(v - level). Mirrored deliberately, so the volume morph and the
// surface morph of the same maps are cut at the same contours.
// The mass arithmetic itself lives in mccore.js, because the contouring worker needs the very
// same functions and two copies of a numerical routine is one copy too many.

// contour level (absolute) -> ray-marcher threshold in [0,1] over the grid range
function normThreshold() {
  const g = active().gridVol || active().gridVolDraft;   // a draft on screen is still contourable
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
  // A morph owns the scene while it is showing, and the slider retargets IT rather than the
  // map underneath — the maps are hidden, so driving them would be a gesture with no picture.
  // The gesture's sigma is what carries over; which map the histogram is drawn from does not
  // matter, because a morph is cut relative to its own first map (see applyMorphSigma).
  if (morph.showing) {
    if (!retargetMorph(m.sigma, !!dragging) && inVolumeMode()) {
      // A STREAMING volume morph has no pool — it draws the pair through slot 0 — so its
      // contour is still just that threshold, exactly as it was. A server-cached SURFACE morph
      // has finished meshes and no field to re-cut them from, so its contour stays frozen at
      // Prepare, also exactly as it was; there is nothing here to do about it.
      volRenderer.setThreshold(normThreshold());
    }
  } else if (state.render === "surface") {
    loadSurface(state.active, !!dragging);   // the active map tracks the drag live
    if (linkContour && !dragging) {
      for (const other of visibleModels()) {
        if (other !== m) { other.sigma = m.sigma; other.surface = null; saveMapPref(other); }
      }
      loadSurfaces();                        // full pass over the rest on release
    }
  } else if (inVolumeMode()) {
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

// The three volume styles, all drawn by the ray-marcher on the same canvas.
const isVolumeStyle = (mode) => mode === "mip" || mode === "emission" || mode === "iso";
const inVolumeMode = () => isVolumeStyle(state.render);

// Show exactly this set of maps (list indices) — what compute and paint do with
// their result: a derived map is what you asked to see, and leaving the inputs
// on just buries it under them. The panel checkboxes bring anything back.
function showOnly(indices) {
  const keep = new Set(indices);
  models.forEach((m, i) => {
    m.visible = keep.has(i);
    renderer.setModelVisible(m.index, m.visible);
    const row = $("modelsList").children[i];
    if (row) row.querySelector("input[type=checkbox]").checked = m.visible;
  });
  updateModelCnt();
  if (state.render === "surface") loadSurfaces();
  else if (inVolumeMode()) loadVolume();
  schedSaveScene();
}

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
  pool: null,        // iso: per-map contours for the resident GPU pool, or null if streaming
  local: null,       // surface: contour every frame in the worker, instead of caching meshes
  refMass: -1,       // enclosed mass of frame 0, held constant across the morph
  showing: false,    // is the morph on screen (as opposed to the maps)?
};

const morphLegs = () => Math.max(0, morph.seq.length - 1);
const morphFrames = () => (morphLegs() ? morphLegs() * morph.perLeg + 1 : 0);
// Everything a prepared cache depends on. Changing any of it invalidates the cache.
//
// Every sequence map's sigma is in here, not just map 0's. In constant-mass mode map 0's is the
// only one that moves the series, but constant-sigma mode cuts each map at its own level, so a
// contour change anywhere in the sequence makes the cached meshes wrong — and the contour is
// something the slider can now move DURING a morph, which is what turned that from a case that
// could not arise into one that does.
const morphKey = () => [morph.seq.join(","), morph.perLeg, morph.mass, morph.quality,
                        active().smooth,
                        morph.seq.map((i) => (models[i] ? models[i].sigma : 0)).join(",")].join("|");

// Frame k -> which pair, and how far between them.
function frameSpec(k) {
  const legs = morphLegs();
  const leg = Math.min(Math.floor(k / morph.perLeg), legs - 1);
  // `leg` is also the pool index of the near map (the pool is built in morph.seq order), which
  // a repeated map in the sequence makes ambiguous if you go looking for it by model index.
  return { a: morph.seq[leg], b: morph.seq[leg + 1], t: (k - leg * morph.perLeg) / morph.perLeg, leg };
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
  // Playback needs prepared frames in surface mode, as it always did — morph.local alone
  // only means "grids are resident, so the scrubber can contour on the fly while the cache
  // builds", which is not something to play at.
  const ready = ok && (state.render !== "surface" || (morph.cache && morph.cacheKey === morphKey()));
  const scrubbable = ready || (state.render === "surface" && !!morph.local && morph.showing);
  $("morphPrepare").disabled = !ok;
  // Volume mode has nothing to prepare, so say so rather than leaving a live-looking
  // button that does almost nothing.
  $("morphPrepare").textContent = state.render === "surface" ? "Prepare" : "Load";
  $("morphPlay").disabled = !ready;
  $("morphRecord").disabled = !ready;
  // The .mrc export computes its frames server-side, so it only needs a sequence.
  $("morphExport").disabled = !ok;
  $("morphScrub").disabled = !scrubbable;
  $("morphScrub").max = String(Math.max(0, morphFrames() - 1));
}

// Draw frame k. In volume mode this is two uniform writes; in surface mode it is a
// cache lookup, both of which are fast enough to scrub with.
async function showMorphFrame(k) {
  if (!morphLegs()) return;
  morph.frame = Math.max(0, Math.min(morphFrames() - 1, k));
  $("morphScrub").value = String(morph.frame);
  const { a, b, t, leg } = frameSpec(morph.frame);
  if (state.render === "surface") {
    // The cache is what makes playback smooth: prepared frames go up at the timer's rate,
    // where contouring on the fly runs at whatever rate the machine manages (a few frames a
    // second) and visibly stutters. On-the-fly is only the stand-in for a scrub before the
    // cache exists.
    const mesh = morph.cache && morph.cache[morph.frame];
    if (mesh) {
      renderer.setModelMesh(MORPH_ID, mesh.positions, mesh.normals, mesh.indices);
      renderer.setModelColor(MORPH_ID, hexToRgb(models[morph.seq[0]].color));
      return;
    }
    if (morph.local) await localMorphFrame(leg, t);
    return;
  }
  if (morph.pool) {
    // The whole sequence is already on the GPU: a frame is two binds and three uniforms.
    // bScale/bOff carry the far map's samples into the near map's normalized range (each grid
    // is normalized to its own vmin/vmax, so a raw blend of two maps with different ranges is
    // not a density blend at all), and the contour is interpolated between the two maps' own
    // mass-matched levels — exact at both ends, so a leg boundary cannot pop.
    const P = morph.pool, ia = leg, ib = leg + 1;
    const span = Math.max(P.vmax[ia] - P.vmin[ia], 1e-9);
    const scale = (P.vmax[ib] - P.vmin[ib]) / span;
    const off = (P.vmin[ib] - P.vmin[ia]) / span;
    // The contour for this frame: read off the leg's solved constant-mass curve where there
    // is one (piecewise-linear between the nine solved points), and only fall back to the
    // straight line between the endpoint levels for constant-sigma morphs, where there is
    // nothing to solve.
    let thrFrame;
    const row = P.curves && P.curves[ia];
    if (row) {
      const x = t * (row.length - 1), i0 = Math.min(row.length - 2, Math.floor(x)), f = x - i0;
      thrFrame = row[i0] * (1 - f) + row[i0 + 1] * f;
    } else {
      thrFrame = P.thr[ia] * (1 - t) + (P.thr[ib] * scale + off) * t;
    }
    volRenderer.setMorphLeg(ia, ib, scale, off, thrFrame);
    volRenderer.setMorphThreshold(thrFrame);
    volRenderer.setMix(t);                 // the one call that redraws
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

// The mesh for one frame of a locally contoured morph: the blend of the leg's two resident
// fields, contoured in the worker holding them. Used twice — to build the playback cache in
// Prepare, and to answer the scrubber while that cache is still being built.
//
// `channel` decides what supersedes what. The scrubber leaves it at the default, so a tick
// issued while an older one is still marching wins; a Prepare pass contouring several legs at
// once gives each leg its own, so concurrent legs do not cancel each other.
// `pair` overrides which field keys this leg blends, and exists for the parallel Prepare: a leg
// in run r > 0 reads its first map through an ALIASED key so its run's fields can sit in a
// worker of their own (see planLegRuns). Everything else — the scrubber, a single-run Prepare —
// leaves it undefined and gets the sequence's own keys.
async function localMorphMesh(leg, t, stride, smooth, channel, pair) {
  const L = morph.local;
  if (!L) return null;
  {
    const ia = leg, ib = leg + 1;
    // The far map's samples carried into the near map's normalized range — the same
    // correction the ray-marcher applies with uBScale/uBOff, so the surface morph and the iso
    // morph of one pair show the same density.
    const span = Math.max(L.vmax[ia] - L.vmin[ia], 1e-9);
    const scale = (L.vmax[ib] - L.vmin[ib]) / span;
    const off = (L.vmin[ib] - L.vmin[ia]) / span;
    const req = { keyA: pair ? pair.a : L.keys[ia], keyB: pair ? pair.b : L.keys[ib],
                  t, scale, off, smooth, stride };
    if (channel) req.channel = channel;
    // Constant mass is solved on the blended field itself, per frame — what the server path
    // does with level_for_mass, and what the GPU morph can only approximate.
    if (morph.mass && L.refMass > 0) req.massTarget = L.refMass;
    else req.level = L.thr[ia] * (1 - t) + (L.thr[ib] * scale + off) * t;
    let res = await mc.morphFrame(req);
    // A miss means the field is not in the worker the engine sent this to — either the budget
    // evicted it, or this leg's two fields ended up apart and a blend cannot cross that line.
    // Re-storing the absent one `near` its partner fixes both: it puts the pair back together
    // and this leg's remaining frames go straight through.
    //
    // The key that missed may be an alias, so the grid to re-store it from is found by which
    // SIDE of the request it was rather than by looking the key up in the sequence — an alias
    // appears nowhere in L.keys, and the map it stands for is the one this leg blends from.
    if (res && res.miss) {
      const near = res.miss === req.keyA ? req.keyB : req.keyA;
      const src = models[morph.seq[res.miss === req.keyA ? ia : ib]].gridVol;
      if (src) mc.store(res.miss, src, { near });
      res = await mc.morphFrame(req);
    }
    // A null reply means a newer request on the morph channel superseded this one — during a
    // Prepare that can only be the scrubber, so ask once more rather than losing the frame.
    if (res === null) res = await mc.morphFrame(req);
    if (!res || res.miss || res.error) return null;
    return res;
  }
}

// The scrubber's path while no cache exists yet: contour on the fly, and drop ticks that
// arrive while the worker is still marching rather than queueing them.
// Bumped by anything that abandons a Prepare, so the frame loop can stop between frames
// instead of marching to the end of a sequence nobody is waiting for.
let morphPrepGen = 0;
let morphPreparing = false;      // a Prepare loop owns the worker's morph channel
let localFrameBusy = false;
// `draft` forces the cheap cut regardless of the Quality setting: it is what a contour drag
// previews with, where a frame has to land inside a pointermove rather than look final.
async function localMorphFrame(leg, t, draft) {
  // While Prepare is contouring the series, the scrubber does not also contour. It no longer
  // has to be for channel reasons — Prepare gives each leg its own — but a scrub frame would
  // still take a worker away from the pass building the cache, and the cache is what the
  // slider is waiting for. The slider still moves; it is moments away.
  if (!morph.local || localFrameBusy || morphPreparing) return;
  localFrameBusy = true;
  try {
    const full = models[morph.seq[0]].smooth;
    const res = await localMorphMesh(leg, t, (draft || morph.quality === "draft") ? 2 : 1,
                                     draft ? Math.min(full, DRAFT_SMOOTH) : full);
    if (!res) return;
    if (state.render !== "surface" || !morph.showing) return;
    renderer.setModelMesh(MORPH_ID, res.positions, res.normals, res.indices);
    renderer.setModelColor(MORPH_ID, hexToRgb(models[morph.seq[0]].color));
  } finally { localFrameBusy = false; }
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
  // Before the showing check, deliberately: a pool built by a Prepare that never reached the
  // screen would otherwise hold its textures for the life of the page. The pool owns its own
  // textures, so tearing it down leaves the slot textures exactly as the maps left them.
  if (morph.pool) { volRenderer.setMorphPool(null); morph.pool = null; }
  morph.local = null;              // the worker keeps its fields; only this binding goes
  morphPrepGen++;                  // stop any Prepare still contouring frames
  if (!morph.showing) return;
  morph.showing = false;
  renderer.dropModel(MORPH_ID);
  volRenderer.setVolumeB(0, 0, 0, null);
  for (const m of models) renderer.setModelVisible(m.index, m.visible);
  if (state.render !== "surface") loadVolume();
}

// Put the whole sequence on the GPU and work out what contour each map should be cut at.
//
// Two problems solved in one pass. The contour: a morph held at one sigma breathes, because
// blending two maps lowers the variance and sigma tracks it — so each map gets the level that
// encloses the same MASS as the first map does at its chosen contour (the same rule the
// surface morph uses, via meshing.level_for_mass). The illumination: each map's opacity
// pyramid is baked at that map's own contour, so the ambient shading is consistent from map to
// map instead of popping when the leg changes.
//
// Returns null if the GPU could not hold the pool, and the caller keeps the streaming path.
async function buildMorphPool() {
  const grids = morph.seq.map((i) => models[i].gridVol);
  if (grids.some((g) => !g)) return null;
  const first = grids[0];
  status("Matching contours across the sequence…", "busy");
  const curves = [];
  for (let i = 0; i < grids.length; i++) {
    // Yield between maps: this is ~100 ms of arithmetic each, and the status line is the
    // only sign the viewer has not hung.
    status(`Matching contours (${i + 1}/${grids.length})…`, "busy");
    await new Promise((r) => setTimeout(r, 0));
    curves.push(massCurve(grids[i]));
  }
  const thr0 = thrFor(models[morph.seq[0]]);
  const refMass = massAbove(curves[0], thr0);
  const thr = grids.map((g, i) => (i === 0 ? thr0
    : morph.mass ? levelForMass(curves[i], refMass)   // same enclosed mass as map 0
                 : thrFor(models[morph.seq[i]])));    // constant sigma: each map's own level
  // The contour ACROSS each leg, at constant enclosed mass.
  //
  // Interpolating between the two endpoint levels — what this used to do — is only right at
  // the endpoints: a blend of two maps has lower variance than either, so the level that
  // encloses the reference mass dips mid-leg, and holding the straight line instead made the
  // surface visibly gain and lose density halfway through every leg. So solve it properly at
  // eight interior points and interpolate the solution rather than the endpoints.
  //
  // Subsampled at stride 4: a mass curve is a statistic over tens of millions of voxels and
  // every 64th of them settles it (measured in mccore.test.mjs: a stride-4 curve picks the
  // same level to within a histogram bin). The endpoints are NOT recomputed from the
  // subsample — they are the exact per-map levels above — because a leg boundary is where the
  // two legs must agree exactly or the surface steps.
  // The CURVES themselves are kept, not just the levels read off them. A mass curve is a
  // property of a FIELD, not of the target mass you invert it at — so keeping them turns a
  // later contour change from "recompute the whole thing" (seconds, and the reason the contour
  // was frozen at Prepare) into a handful of levelForMass lookups. See retargetMorphLevels.
  //
  // The bill is small enough not to think about: a curve is MASS_BINS+1 float64s, 8.2 KB, and a
  // ten-map sequence keeps ten map curves plus nine legs of seven interior curves — about
  // 0.5 MB against the hundreds of megabytes the pool's textures already hold.
  const CURVE_N = 8;
  let curveLevels = null;
  const legCurves = [], legScale = [], legOff = [];
  for (let leg = 0; leg + 1 < grids.length; leg++) {
    const span = Math.max(grids[leg].vmax - grids[leg].vmin, 1e-9);
    legScale.push((grids[leg + 1].vmax - grids[leg + 1].vmin) / span);
    legOff.push((grids[leg + 1].vmin - grids[leg].vmin) / span);
  }
  if (morph.mass && grids.length > 1) {
    curveLevels = [];
    for (let leg = 0; leg + 1 < grids.length; leg++) {
      const gA = grids[leg], gB = grids[leg + 1];
      const sameGrid = gA.nx === gB.nx && gA.ny === gB.ny && gA.nz === gB.nz;
      const scale = legScale[leg], off = legOff[leg];
      const ends = [thr[leg], thr[leg + 1] * scale + off];      // A-space, exact
      const row = new Array(CURVE_N + 1);
      row[0] = ends[0]; row[CURVE_N] = ends[1];
      // Null for a leg with nothing to invert — a mismatched-grid pair — which is exactly
      // where the loop below falls back to the straight line, and exactly where a retarget
      // has to fall back to it too.
      const rowCurves = sameGrid ? [] : null;
      for (let k = 1; k < CURVE_N; k++) {
        // Two maps on different grids cannot be blended voxel-wise at all (the server refuses
        // to morph them); the straight line is all there is to say for such a pair.
        if (!sameGrid) { row[k] = ends[0] * (1 - k / CURVE_N) + ends[1] * (k / CURVE_N); continue; }
        status(`Matching contours… leg ${leg + 1}/${grids.length - 1} (${k}/${CURVE_N - 1})`, "busy");
        await new Promise((r) => setTimeout(r, 0));
        const t = k / CURVE_N;
        const curve = blendedMassCurve(gA, gB, t, scale, off, 4);
        rowCurves.push(curve);
        const lvl = levelForMass(curve, refMass);
        // A saturated solve means the subsample lost the mass entirely (a narrow feature
        // missed by the stride). Fall back to the straight line for that sample rather than
        // collapsing the contour to zero.
        row[k] = (lvl <= 0 || lvl >= 1) ? ends[0] * (1 - t) + ends[1] * t : lvl;
      }
      curveLevels.push(row);
      legCurves.push(rowCurves);
    }
  }
  const entries = grids.map((g, i) => ({
    nx: g.nx, ny: g.ny, nz: g.nz, data: g.data, lo: g.boxLo, hi: g.boxHi,
    vmin: g.vmin, vmax: g.vmax, thr: thr[i],
  }));
  status("Uploading the sequence…", "busy");
  if (!volRenderer.setMorphPool(entries)) { volRenderer.setMorphPool(null); return null; }
  // density bytes + the R8 opacity volume and its mip chain (1 + 1/8 + 1/64 + ... ≈ 1.15)
  const bytes = grids.reduce((acc, g) => acc + g.nx * g.ny * g.nz * ((g.f16 ? 2 : 1) + 1.15), 0);
  return { thr, curves: curveLevels, box: Math.max(first.nx, first.ny, first.nz),
           vmin: grids.map((g) => g.vmin), vmax: grids.map((g) => g.vmax),
           // Everything a retarget needs, so it never has to touch a grid again.
           mapCurves: curves, legCurves, legScale, legOff, curveN: CURVE_N,
           mb: Math.round(bytes / (1024 * 1024)) };
}



// How many legs of a Prepare may contour at once, and which legs each of those runs owns.
//
// Legs only run in parallel if their fields are in DIFFERENT workers, and adjacent legs share a
// map — which is why round 1's `near` retries pulled a whole sequence onto one worker and got
// no parallelism out of it. Contiguous runs of legs fix that for the price of one duplicated
// field per extra run: run r's first map is also run r−1's last, so only that one map needs a
// second copy (under an aliased key), and every other field stays where it is.
//
// R is bounded by three things and takes the smallest. The threads there are to run on. The
// legs there are to split. And memory, twice over: `2 x fieldBytes` is the floor for a run to
// hold a pair at all, and the second term is the one that matters for a long sequence — the
// budget has to hold every map PLUS one extra per additional run, so a sequence that already
// fills the budget gets R = 1 rather than a Prepare that spends itself evicting and re-decoding.
//
// Pure arithmetic, no globals: the harness lifts it verbatim.
function planLegRuns(legs, nmaps, fieldBytes, workers, budgetBytes) {
  const bytes = Math.max(1, fieldBytes | 0);
  const pairFits = Math.floor(budgetBytes / (2 * bytes));
  const spare = Math.floor(budgetBytes / bytes) - nmaps;   // fields we can afford to duplicate
  const R = Math.max(1, Math.min(workers, legs, pairFits, 1 + Math.max(0, spare)));
  const runs = [];
  let first = 0;
  for (let r = 0; r < R; r++) {
    const len = Math.ceil((legs - first) / (R - r));       // near-equal, longest first
    runs.push({ first, last: first + len - 1 });
    first += len;
  }
  return { R, runs };
}

// The field key each leg blends from, per leg, given a run plan. Run 0 uses the sequence's own
// keys throughout. Run r > 0 reads its FIRST map through an alias and everything after it
// plainly — which is what makes the runs' key sets disjoint: run r owns maps
// (first .. last+1], plus an aliased copy of `first`, and its neighbour owns `first` plainly.
function legKeyPlan(runs, keys) {
  const alias = [];      // the duplicate fields this plan needs, and which map each stands for
  const perLeg = [], runKeys = [];
  runs.forEach((run, r) => {
    const keyAt = (i) => {
      if (r === 0 || i !== run.first) return keys[i];
      const a = `${keys[i]}#run${r}`;
      if (!alias.some((x) => x.key === a)) alias.push({ key: a, seq: i });
      return a;
    };
    // Maps first..last+1 are what this run reads; the first of them is its anchor.
    const mine = [];
    for (let i = run.first; i <= run.last + 1; i++) mine.push({ seq: i, key: keyAt(i) });
    runKeys.push(mine);
    for (let leg = run.first; leg <= run.last; leg++) {
      perLeg[leg] = { a: keyAt(leg), b: keyAt(leg + 1), run: r };
    }
  });
  return { perLeg, alias, runKeys };
}

// ---- live contour during a morph -----------------------------------------
//
// Both morph paths used to freeze the contour at Prepare, because moving it meant redoing the
// work Prepare had just done: inverting a mass curve per map and per interior sample of every
// leg. What makes it live instead is that a mass curve belongs to the FIELD, not to the target
// mass you invert it at — so Prepare keeps the curves (buildMorphPool above, and the local
// prep below) and a contour gesture is only the inversions, which are a binary search over
// 1024 bins each.
//
// The normalized range a retarget must speak in is the one the morph was BUILT against, not
// whatever the map's grid says today: the pool's textures and the workers' fields hold that
// normalization, and a grid refetched at another size since Prepare would otherwise shift
// every level. thrFor reads nothing but vmin/vmax, so handing it the stored pair is enough.
const morphRange = (S, i) => ({ vmin: S.vmin[i], vmax: S.vmax[i] });

// The sigma of a morph gesture applies to the SEQUENCE, not to whichever map the histogram
// happens to be showing. In constant-mass mode only map 0 has a sigma that means anything —
// every other map is cut at whatever level encloses map 0's mass. In constant-sigma mode the
// mode's whole claim is "each map at its own level", so one gesture moves them all together,
// which is what linkContour does for still maps.
function applyMorphSigma(sigma) {
  models[morph.seq[0]].sigma = sigma;
  if (!morph.mass) for (const i of morph.seq) models[i].sigma = sigma;
}

// Iso morph: retarget the resident pool in place, this frame.
//
// The cost is the reason this can run on every pointermove rather than on release. A ten-map
// sequence — the longest anyone builds — is 10 inversions over the map curves plus 9 legs x 7
// interior points: 73 walks down a 1024-bin curve, measured at 2.7 us for the lot, or about
// 370 complete retargets per millisecond. Nothing here touches a grid, uploads a texture or
// allocates; that is what keeping the curves bought.
function retargetMorphLevels(sigma) {
  const P = morph.pool;
  if (!P || !P.mapCurves) return;
  applyMorphSigma(sigma);
  const thr0 = thrFor(models[morph.seq[0]], morphRange(P, 0));
  const refMass = massAbove(P.mapCurves[0], thr0);
  for (let i = 0; i < P.thr.length; i++) {
    P.thr[i] = i === 0 ? thr0
      : morph.mass ? levelForMass(P.mapCurves[i], refMass)      // same enclosed mass as map 0
                   : thrFor(models[morph.seq[i]], morphRange(P, i));
  }
  // The per-leg rows, rebuilt exactly as buildMorphPool builds them: exact endpoints from the
  // new per-map levels, interior points re-inverted on the curves it kept, and the same
  // saturated-solve fallback to the straight line.
  if (P.curves) {
    for (let leg = 0; leg < P.curves.length; leg++) {
      const row = P.curves[leg], N = row.length - 1;
      const e0 = P.thr[leg], e1 = P.thr[leg + 1] * P.legScale[leg] + P.legOff[leg];
      row[0] = e0; row[N] = e1;
      const cs = P.legCurves[leg];
      for (let k = 1; k < N; k++) {
        const t = k / N, lerp = e0 * (1 - t) + e1 * t;
        if (!cs) { row[k] = lerp; continue; }        // mismatched grids: nothing to invert
        const lvl = levelForMass(cs[k - 1], refMass);
        row[k] = (lvl <= 0 || lvl >= 1) ? lerp : lvl;
      }
    }
  }
  // The silhouette, now: showMorphFrame reads P.thr/P.curves and issues the leg binding and
  // the threshold, so the new contour is on screen this frame whether the morph is playing or
  // parked. The pool's opacity pyramids follow on their own debounce (see volume.js).
  showMorphFrame(morph.frame);
  volRenderer.setMorphPoolThresholds(P.thr);
}

// Surface morph: preview at draft quality while the pointer is down, rebuild the cache on
// release. There is no equivalent of the pool's "two uniforms" here — every frame is a march —
// so the honest thing is to show one frame cheaply during the gesture and pay for the series
// once the level has settled.
//
// Only the LOCAL path can do this at all. A server-cached morph (hide-dust on, or a map whose
// grid never became resident) has its frames as finished meshes from the server and no field
// to re-cut them from, so its contour stays frozen at Prepare exactly as it always was.
function retargetLocalMorph(sigma, dragging) {
  const L = morph.local;
  if (!L) return;
  applyMorphSigma(sigma);
  if (morph.mass && L.mapCurves) {
    // The worker solves each frame's level from massTarget on the blended field itself, so the
    // reference mass is the only thing a retarget has to move.
    L.refMass = massAbove(L.mapCurves[0], thrFor(models[morph.seq[0]], morphRange(L, 0)));
  } else {
    for (let i = 0; i < L.thr.length; i++) L.thr[i] = thrFor(models[morph.seq[i]], morphRange(L, i));
  }
  // The prepared meshes were cut at the old level: they are wrong now, and showMorphFrame
  // prefers them over anything live. Dropping the cache puts the morph back into the state it
  // is in during a first Prepare — the scrubber contours on the fly, playback waits — which is
  // exactly what this is.
  if (morph.cache || morphPreparing) {
    morph.cache = null;
    morph.cacheKey = "";
    if (morphPreparing) morphPrepGen++;    // a Prepare still filling the old level: abandon it
    updateMorphUI();
  }
  if (dragging) {
    const { t, leg } = frameSpec(morph.frame);
    localMorphFrame(leg, t, true);         // draft: stride 2 and the draft smoothing clamp
    return;
  }
  prepareMorph();                          // release: rebuild the series in the background
}

// A contour gesture while a morph is on screen. Returns false when the morph on screen is one
// whose contour cannot move, so the caller knows nothing happened.
function retargetMorph(sigma, dragging) {
  if (morph.pool) retargetMorphLevels(sigma);
  else if (morph.local && state.render === "surface") retargetLocalMorph(sigma, dragging);
  else return false;
  // The gesture moved the sequence's sigmas, so the sequence's preferences are what to save —
  // the still-map path saves the one map it moved, and this is the same rule.
  if (!dragging) for (const i of morph.seq) saveMapPref(models[i]);
  return true;
}

async function prepareMorph() {
  if (!morphLegs()) return;
  const btn = $("morphPrepare");
  btn.disabled = true;
  try {
    if (state.render !== "surface") {
      // Volume mode: just make sure every map in the sequence is on hand as a grid. In iso the
      // whole sequence then goes onto the GPU at once (see setMorphPool), so what it costs
      // decides what resolution it gets: a sequence that fits keeps the native grids (which
      // single-map iso viewing has usually already downloaded, so Prepare fetches nothing at
      // all), and only one that does not gets capped.
      const isoMorph = state.render === "iso";
      const fetchSeq = async (cap) => {
        for (let i = 0; i < morph.seq.length; i++) {
          const m = models[morph.seq[i]];
          status(`Loading ${m.name} (${i + 1}/${morph.seq.length})…`, "busy");
          await volGrid(m, cap);
        }
      };
      if (isoMorph) {
        const fit = poolFitsNative();
        if (!fit.fits) {
          status(`Sequence needs ~${fit.mb} MB at full resolution — loading at ${ISO_MORPH_MAX}³ `
                 + `to fit GPU memory (budget ${fit.budget} MB, adjustable under Morph)…`, "busy");
        }
        await fetchSeq(fit.fits ? undefined : ISO_MORPH_MAX);
        morph.pool = await buildMorphPool();
        // The estimate said it would fit and the driver disagreed: take the capped grids and
        // try once more before giving up on the pool entirely. The native grids stay in the
        // per-map cache only until the smaller fetch replaces them (cache-by-size keeps one
        // payload per map), so a later single-map view re-fetches at native — a download, not
        // a correctness problem.
        if (!morph.pool && fit.fits) {
          status("The GPU would not hold the sequence at full resolution — retrying at "
                 + `${ISO_MORPH_MAX}³…`, "busy");
          await fetchSeq(ISO_MORPH_MAX);
          morph.pool = await buildMorphPool();
        }
      } else {
        await fetchSeq(undefined);
        morph.pool = null;
      }
      beginMorph();
      // The morph runs single-slot; give slot 0 the sequence's colour and contour.
      // A leftover diverging flag from a painted diff would mis-read the maps.
      volRenderer.setDiverging(false);
      volRenderer.setColor(hexToRgb(models[morph.seq[0]].color));
      if (!morph.pool) volRenderer.setThreshold(thrFor(models[morph.seq[0]]));
      await showMorphFrame(morph.frame);
      status(`Morph ready · ${morph.seq.length} maps · blended on the GPU`
             + (morph.pool ? ` · ${morph.pool.box}³ · ${morph.pool.mb} MB resident`
                             + ` · ${morph.mass ? "constant mass" : "constant σ"}` : ""));
      updateMorphUI();
      return;
    }
    // Surface mode, local path: every map's grid goes into the worker pool and every frame is
    // contoured there, with no server involvement at all. The frames are still CACHED — the
    // cache is what lets playback run at the timer's rate, and contouring per tick instead
    // (the first version of this) generated a few frames a second and looked exactly like
    // that. What the local path removes is the network: a 24-frame morph of three maps was
    // two dozen multi-megabyte mesh downloads and as many marching-cubes jobs queued on one
    // server, and is now a few grid fetches and some arithmetic in a background thread.
    if (mc.available && !dust.on && !painted) {
      const keys = [], vmin = [], vmax = [], thr = [];
      for (let i = 0; i < morph.seq.length; i++) {
        const m = models[morph.seq[i]];
        status(`Loading ${m.name} (${i + 1}/${morph.seq.length})…`, "busy");
        await volGrid(m);
        const g = m.gridVol;
        if (!g) { keys.length = 0; break; }
        const key = mcKey(m, g);
        if (!mc.has(key)) mc.store(key, g);
        keys.push(key); vmin.push(g.vmin); vmax.push(g.vmax); thr.push(thrFor(m, g));
      }
      if (keys.length === morph.seq.length) {
        // Every map's mass curve, not only map 0's. Map 0's alone would answer "what mass are
        // we holding constant"; keeping them all is what lets a contour drag during the morph
        // re-solve the whole sequence without touching a grid again (see retargetLocalMorph).
        // ~100 ms per map, once, against a Prepare that is already seconds of marching — and
        // only in constant-mass mode, because constant sigma inverts nothing.
        let mapCurves = null;
        if (morph.mass) {
          mapCurves = [];
          for (let i = 0; i < morph.seq.length; i++) {
            status(`Matching contours (${i + 1}/${morph.seq.length})…`, "busy");
            await new Promise((r) => setTimeout(r, 0));
            mapCurves.push(massCurve(models[morph.seq[i]].gridVol));
          }
        }
        // The mass every frame is cut to: map 0's, at its own contour, in raw units — the
        // same reference meshing.level_for_mass is given server-side.
        const refMass = morph.mass ? massAbove(mapCurves[0], thr[0]) : 0;
        morph.local = { keys, vmin, vmax, thr, refMass, mapCurves };
        morph.cache = null;
        // The morph goes on screen before the cache exists, so the scrubber has something to
        // move through while it builds — on-the-fly frames, at whatever rate the worker
        // manages. Playback waits for the cache (see updateMorphUI).
        beginMorph();
        await showMorphFrame(morph.frame);

        // Then contour every frame up front, exactly as the server path does, for the same
        // reason: prepared meshes play back at the timer's rate. Contouring per tick instead
        // (what round 11 shipped) generated 4-8 frames a second and looked it.
        const n = morphFrames();
        const stride = morph.quality === "draft" ? 2 : 1;
        const smooth = models[morph.seq[0]].smooth;
        const gen = ++morphPrepGen;
        const cache = new Array(n);
        let cachedBytes = 0, landed = 0, abandoned = false;
        // The frames of one leg, in order, per leg. Legs are what parallelises here: a leg's
        // frames all blend the same two fields, so they belong to whichever worker holds that
        // pair and have to stay in order (they also share a channel, and sequential awaits are
        // what keeps newest-wins from cancelling a frame that is already marching). Different
        // legs have no such tie, so they can go at once.
        //
        // Whether they actually do is a placement question, and it used to answer itself badly:
        // adjacent legs share a map, so the miss-and-restore in localMorphMesh pulled every pair
        // together and a long sequence gathered onto ONE worker, leaving the concurrency with
        // nothing to spread. Contiguous RUNS of legs fix it — each run gets its own worker and
        // its own copy of the single map it shares with its neighbour (planLegRuns prices that
        // against the engine's budget), and the fields are placed deliberately below instead of
        // being discovered by a miss.
        const g0 = models[morph.seq[0]].gridVol;
        const { R, runs } = planLegRuns(morphLegs(), morph.seq.length,
                                        g0.nx * g0.ny * g0.nz * 4,
                                        mc.workers, mc.budgetBytes);
        const { perLeg: legKeys, alias, runKeys } = legKeyPlan(runs, keys);
        // Place each run's fields together, run by run: the anchor goes wherever there is most
        // room — which is a different worker for each run, because the runs placed before it
        // have already filled the others — and the rest of the run follows it. Doing this up
        // front rather than letting the first frame of each leg discover it with a miss saves a
        // round trip per leg and, more to the point, is what decides that two runs land in two
        // different workers instead of converging on one.
        for (const mine of runKeys) {
          const anchor = mine[0].key;
          for (const { seq, key } of mine) {
            mc.store(key, models[morph.seq[seq]].gridVol, key === anchor ? undefined : { near: anchor });
          }
        }
        const byLeg = new Map();
        for (let k = 0; k < n; k++) {
          const { t, leg } = frameSpec(k);
          if (!byLeg.has(leg)) byLeg.set(leg, []);
          byLeg.get(leg).push({ k, t });
        }
        morphPreparing = true;
        status(`Preparing morph… 0/${n}`, "busy");
        try {
          await Promise.all([...byLeg].map(async ([leg, frames]) => {
            try {
              for (const { k, t } of frames) {
                // Abandoned: the mode changed, the morph was stopped, or another Prepare started.
                if (gen !== morphPrepGen || state.render !== "surface" || !morph.showing) { abandoned = true; return; }
                const res = await localMorphMesh(leg, t, stride, smooth, `morph:${leg}`, legKeys[leg]);
                if (!res) throw new Error("contouring was interrupted");
                cache[k] = { positions: res.positions, normals: res.normals, indices: res.indices,
                             bytes: res.positions.byteLength + res.normals.byteLength + res.indices.byteLength };
                cachedBytes += cache[k].bytes;
                status(`Preparing morph… ${++landed}/${n}`, "busy");
                // The same byte budget the server path enforces, and for the same reason: n frames
                // of multi-MB meshes will OOM the tab. Frame 0 prices the series before the grind.
                if (k === 0 && cache[0].bytes * n > morphBudgetBytes()) {
                  throw new Error(`~${Math.round(cache[0].bytes * n / (1024 * 1024))} MB of meshes for ${n} frames — `
                                  + `switch Quality to draft or shorten the sequence (budget ${morphBudgetMB()} MB)`);
                }
                if (cachedBytes > morphBudgetBytes()) {
                  throw new Error(`morph cache passed ${morphBudgetMB()} MB at frame ${k + 1}/${n} — `
                                  + "switch Quality to draft or shorten the sequence");
                }
              }
            } catch (err) {
              // One leg giving up ends the pass, so tell the others: bumping the generation is
              // the same signal a stopped morph sends, and they check it between frames.
              morphPrepGen++;
              throw err;
            }
          }));
        } finally {
          morphPreparing = false;
          // The duplicates were scaffolding for this pass — they exist only so two runs could
          // contour at once — and each is a whole field. Drop them however the pass ended, or a
          // Prepare that failed halfway would leave a map's worth of memory resident under a
          // key nothing will ever ask for again. The scrubber uses the sequence's own keys, so
          // nothing after this needs them.
          for (const a of alias) mc.drop(a.key);
        }
        if (abandoned) return;
        morph.cache = cache;
        morph.cacheKey = morphKey();
        await showMorphFrame(morph.frame);
        status(`Morph ready · ${n} frames · ${(cachedBytes / (1024 * 1024)).toFixed(0)} MB cached`
               + " · contoured in the browser" + (R > 1 ? ` · ${R} threads` : "")
               + (morph.mass ? " · constant mass" : " · constant σ"));
        updateMorphUI();
        return;
      }
      morph.local = null;   // a grid did not arrive: fall through to the server-cached path
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
    if (first.bytes * n > morphBudgetBytes()) {
      throw new Error(`~${Math.round(first.bytes * n / (1024 * 1024))} MB of meshes for ${n} frames — ` +
                      `switch Quality to draft or shorten the sequence (budget ${morphBudgetMB()} MB)`);
    }
    const level = morph.mass ? `mass=${morph.refMass}` : `sigma=${sigma.toFixed(3)}`;
    for (let k = 1; k < n; k++) {
      const { a, b, t } = frameSpec(k);
      status(`Preparing morph… ${k + 1}/${n}`, "busy");
      cache[k] = await fetchMorphMesh(a, b, t, level, smooth, size);
      cachedBytes += cache[k].bytes;
      if (cachedBytes > morphBudgetBytes()) {
        throw new Error(`morph cache passed ${morphBudgetMB()} MB at frame ${k + 1}/${n} — ` +
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
const camera3d = (mode) => (mode === "surface" ? renderer : isVolumeStyle(mode) ? volRenderer : null);
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
  const isVolume = isVolumeStyle(mode);
  // Iso is its own top-level segment now, and the only volume style that is not ray-integrated:
  // the γ / brightness / style rows do nothing to a shaded surface, so they belong to the other
  // two alone.
  const isIso = mode === "iso";
  const isRayVol = isVolume && !isIso;
  const isSlices = mode === "slices";

  show($("gl"), isSurface);
  show($("glvol"), isVolume);
  $("slices").style.display = isSlices ? "grid" : "none";

  const to = camera3d(mode);
  if (to && to !== from && camera) to.setCamera(camera);

  // The segmented switch tracks the mode; MIP and glow share the "volume" segment, iso has
  // its own.
  const seg = isRayVol ? "volume" : mode;
  for (const b of $("modeSeg").querySelectorAll("button")) {
    b.classList.toggle("on", b.dataset.mode === seg);
  }
  if (isRayVol) $("volStyle").value = mode;   // the select holds mip/emission only
  // Contextual controls: only the current mode's rows exist on screen.
  // Lighting drives the surface renderer and the shaded-iso mode alike, so its row belongs to
  // both and to neither of the ray-integrated styles. Smoothing likewise — Taubin on the mesh,
  // a Gaussian on the field — but only where the renderer can actually blur (see hasSmoothing).
  show($("lightRow"), isSurface || isIso);
  show($("smoothRow"), isSurface || (isIso && volRenderer.hasSmoothing));
  showOccRow(mode);
  show($("dispVolume"), isRayVol);
  show($("dispSlices"), isSlices);
  show($("colorlbl"), !isSlices);
  $("secContour").hidden = isSlices;   // contour is meaningless for raw slices
  $("secClip").hidden = isSlices;      // the slab cuts the 3D views only

  if (isSurface) { renderer.render(); loadSurfaces(); }
  else if (isVolume) {
    volRenderer.setMode(mode);
    // Iso is shaded, so it needs the lighting preset and the smoothing the sliders are sitting
    // on — either may have been changed while a surface was on screen, or restored from a
    // previous session.
    if (isIso) {
      volRenderer.setLighting($("light").value);
      volRenderer.setSmoothing(ISO_SIGMA_PER_STEP * +$("smooth").value);
    }
    loadVolume();
  }
  else { loadSlices(); }
  // Entering a mode that wants resident grids: queue every map's. This is also how they get
  // queued at page OPEN — boot's whole job ends in a setMode, so by the time this line runs the
  // model list is loaded and the first render pass has been kicked off one line above. The
  // gates in runPrefetch are what keep the queue behind that paint rather than in front of it.
  // A no-op in the modes that do not want these payloads.
  queueAllPrefetch();
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
  // Only re-enter if one of the two styles this select owns is on screen: iso is a segment of
  // its own now, and switching it out from under a hidden select would be a surprise.
  if (state.render === "mip" || state.render === "emission") setMode(volStyle);
});

// Export menu: the four ways out of the viewer, one dropdown.
$("exportBtn").addEventListener("click", (e) => {
  e.stopPropagation();
  const menu = $("exportMenu");
  menu.classList.toggle("open");
  if (menu.classList.contains("open")) {
    // Bound it against where the button actually sits — the hosting window can be a
    // tiny floating pane, so the right-anchored menu can run past either edge.
    menu.style.right = "";
    menu.style.maxHeight = "";
    const r = menu.getBoundingClientRect();
    if (r.left < 8) {
      const shift = Math.min(8 - r.left, window.innerWidth - 8 - r.right);
      menu.style.right = (-shift) + "px";
    }
    menu.style.maxHeight = Math.max(120, window.innerHeight - r.top - 10) + "px";
  }
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

// The occlusion slider only means something where there are ambient shadows: the shaded-iso
// style under the two presets that cast them. Standard is flat ambient, so it hides the row.
function showOccRow(mode) {
  const lit = $("light").value;
  show($("occRow"), mode === "iso" && (lit === "smooth" || lit === "raytraced"));
}
// One lighting choice for both shaded views: the mesh surface and the ray-cast iso.
$("light").addEventListener("change", (e) => {
  renderer.setLighting(e.target.value);
  volRenderer.setLighting(e.target.value);
  showOccRow(state.render);          // the preset decides whether occlusion is adjustable
});
$("occgain").addEventListener("input", (e) => volRenderer.setOcclusionGain(+e.target.value));
// Slider steps -> Gaussian sigma in voxels for iso. 0.15 per step puts the 0-20 slider on
// 0..3 voxels (the renderer clamps at 2.5), which lands the default 6 at ~0.9 voxels — about
// where the field stops looking pixelated without the surface starting to melt. A taste
// constant, like the Taubin count it shares the slider with.
const ISO_SIGMA_PER_STEP = 0.15;
$("smooth").addEventListener("input", (e) => {
  // In iso the slider smooths the density field the march hits (and so its gradients); the
  // map's own Taubin setting is left alone, since no mesh is being cut here.
  if (state.render === "iso") { volRenderer.setSmoothing(ISO_SIGMA_PER_STEP * +e.target.value); return; }
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
// The two memory budgets. The inputs always SHOW the number in force — the stored override if
// there is one, otherwise whatever the default resolved to — so the nvidia-smi-derived figure
// is visible rather than implicit. Clearing an input drops the override and the default comes
// back. Neither is applied retroactively: a prepared morph keeps whatever it was built with.
function syncBudgetInputs() {
  const p = budgetPrefs();
  $("memCache").value = Number.isFinite(p.cache) && p.cache > 0 ? String(p.cache) : String(morphBudgetMB());
  $("memPool").value = Number.isFinite(p.pool) && p.pool > 0 ? String(p.pool) : String(isoPoolBudgetMB());
}
function onBudgetInput(which, el, lo, hi) {
  const raw = el.value.trim();
  const next = budgetPrefs();
  if (!raw) delete next[which];                      // cleared: back to the default
  else next[which] = Math.max(lo, Math.min(hi, Math.round(+raw) || lo));
  writeJSON(BUDGET_KEY, next);
  syncBudgetInputs();
}
$("memCache").addEventListener("change", (e) => onBudgetInput("cache", e.target, 128, 8192));
$("memPool").addEventListener("change", (e) => onBudgetInput("pool", e.target, 256, 8192));

$("morphFrames").addEventListener("change", (e) => {
  morph.perLeg = Math.max(2, Math.min(120, +e.target.value || 24));
  e.target.value = String(morph.perLeg);
  morph.cache = null; stopMorph(); updateMorphUI();
});
// Both change what a frame IS, so both drop whatever was prepared — the cached meshes on the
// server path, and the local path's reference mass with them.
$("morphMass").addEventListener("change", (e) => { morph.mass = e.target.checked; morph.cache = null; morph.local = null; updateMorphUI(); });
$("morphQuality").addEventListener("change", (e) => { morph.quality = e.target.value; morph.cache = null; morph.local = null; updateMorphUI(); });
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
  // Clicking "view" re-aims the cut at the current line of sight, including when it is
  // already the selected chip — that re-aim is the only way to move a plane that is
  // deliberately welded to the model once captured.
  if (clipAx === "v") captureViewNormal();
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

// ---- clipping (slab along one direction, surface + volume) ---------------
let clipAx = "";        // "", "0", "1", "2", "v" — driven by the chips in the Clip section
let clipViewN = null;   // model-space normal captured the last time "view" was clicked

// The line of sight, in the coordinates the meshes and grids are held in. orbit.js keeps
// the eye on world +Z looking down -Z and hangs the rotation on the model, so the model-
// space view direction is Rᵀ·[0,0,-1] = [-R[2], -R[6], -R[10]] (the same expression
// orbit's own axisRay uses). It points away from the camera, so dot(p, n) grows with
// depth and the "lo" slider is always the plane nearer the viewer.
//
// Capturing it freezes the plane into the model: rotate afterwards and the cut turns with
// the map, which is the whole point of cutting "along the view" rather than dragging a
// screen-aligned plane. Re-click the chip to re-aim it at where you are looking now.
function captureViewNormal() {
  const R = (state.render === "surface" ? renderer : volRenderer).getCamera().R;
  const v = [-R[2], -R[6], -R[10]];
  const len = Math.hypot(v[0], v[1], v[2]) || 1;
  clipViewN = [v[0] / len, v[1] / len, v[2] / len];
}

function applyClip() {
  const axis = clipAx;
  $("clipRanges").style.display = axis === "" ? "none" : "";
  if (axis === "") {
    volRenderer.setClip(null, 0, 0); renderer.setClip(null, 0, 0);
    $("clipState").textContent = "off";
    return;
  }
  let n, support;
  if (axis === "v") {
    if (!clipViewN) captureViewNormal();
    n = clipViewN;
    // How far the [-1,1] cube reaches along n — its support function, |nx|+|ny|+|nz|.
    // Sweeping the slider over ±that is what makes the ends of the range clear the model
    // whichever way the plane is tilted, exactly as ±1 does for a coordinate axis.
    support = Math.abs(n[0]) + Math.abs(n[1]) + Math.abs(n[2]);
  } else {
    n = [0, 0, 0]; n[+axis] = 1; support = 1;
  }
  let lo = +$("clipLo").value / 100, hi = +$("clipHi").value / 100;
  if (hi < lo + 0.01) hi = Math.min(1, lo + 0.01);
  $("clipState").textContent = (axis === "v" ? "view" : "XYZ"[+axis])
    + ` ${Math.round(lo * 100)}–${Math.round(hi * 100)}%`;
  // Both renderers cut against the same model-space plane pair now, so they get the same
  // numbers. A slider parked at an end is pushed a hair past the cube so the outermost
  // vertices and voxels are not shaved off by rounding.
  let d0 = support * (2 * lo - 1), d1 = support * (2 * hi - 1);
  if (lo <= 0) d0 -= 0.002 * support;
  if (hi >= 1) d1 += 0.002 * support;
  renderer.setClip(n, d0, d1);
  volRenderer.setClip(n, d0, d1);
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
    showOnly(made.pair || [made.index]);   // the result alone; the inputs would bury it
    setActive(made.index);
    status(made.pair ? `Added ${made.name} — green = gained, red = lost` : `Added ${made.name}`);
  } catch (err) { status("Compute failed: " + err.message, "err"); }
  finally { btn.disabled = false; }
}

// ---- paint by difference (ChimeraX's `color sample` on A − B) ------------
// The surface of A, coloured by the local value of A − B: red where density was
// lost, near-white where nothing changed, blue where it was gained — the
// continuous difference-map reading, on one surface instead of two lobes.
let painted = null;   // { m, other } list indices while a map wears a painted surface

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
    // loadSurfaces pass does not refetch it. From here on surfaceOnce re-paints on
    // every contour/smoothing change; only "clear" (or painting another pair)
    // returns this map to a plain surface.
    ma.surface = ma.sigma;
    painted = { m: a, other: b };
    $("paintClear").hidden = false;
    showOnly([a]);               // the painted surface alone — B overlapping it would hide the colours
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
  if (!painted) return;
  const i = painted.m;
  painted = null;                    // cached diff grids die with it
  $("paintClear").hidden = true;
  models[i].surface = null;
  if (state.render === "surface") surfaceOnce(i, false);
  else if (inVolumeMode()) { volRenderer.setDiverging(false); loadVolume(); }
  else if (state.render === "slices") loadSlices();
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
    const mode = saved && ["surface", "mip", "emission", "iso", "slices"].includes(saved.mode)
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
