// Map (.mrc) viewer controller — one map, or several in one scene. Reuses the same
// renderers as the conformational-landscape app — the marching-cubes surface (gl.js),
// the volume ray-marcher (volume.js) and the contour histogram (histogram.js) — plus an
// orthogonal-slice view (slices.js). No latents, no network, no landscape: it just
// inspects the volumes the backend loaded from disk.
//
// With more than one map the ChimeraX arrangement applies: a model panel toggles each
// map's visibility, every map carries its own colour and its own contour, and one of
// them is *active* — the one the contour histogram, the colour picker and the .mrc
// download act on. Surface mode draws them all together. Volume and slice modes show
// the active map alone: compositing several ray-marched volumes correctly (depth
// ordering, per-map transfer functions) is a different renderer, not a loop, and
// showing a plausible-but-wrong overlay would be worse than showing one map honestly.
import { createRenderer } from "./gl.js";
import { createVolume } from "./volume.js";
import { createHistogram } from "./histogram.js";
import { createSlices } from "./slices.js";

const $ = (id) => document.getElementById(id);
const THEME_BG = { dark: "#040610", light: "#e7ecf3" };

const renderer = createRenderer($("gl"));       // surface meshes (all visible maps)
const volRenderer = createVolume($("glvol"));   // MIP / glow ray-march (active map)
const slices = createSlices($("slices"));       // orthogonal cuts (active map)
const hist = createHistogram($("hist"), onLevel);

const state = { render: "surface", active: 0 };
let info = null;        // /api/info
let models = [];        // per-map view state, parallel to info.models

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
}
applyTheme(new URLSearchParams(location.search).get("theme"));
window.addEventListener("message", (e) => { if (e.data && e.data.type === "hax-theme") applyTheme(e.data.theme); });

// ---- data loaders --------------------------------------------------------
async function loadInfo() {
  info = await (await fetch("./api/info")).json();
  models = info.models.map((m, i) => ({
    ...m,
    visible: true,
    color: m.color,
    sigma: 2.5,
    smooth: 6,
    surface: null,     // the sigma the loaded mesh was cut at, so it isn't refetched
  }));
  $("title").textContent = info.title || "Map";
  const multi = models.length > 1;
  $("models").hidden = !multi;
  if (multi) { buildModelPanel(); setupRail(); }
  else $("stage").classList.add("rail-off");
  // Maps on different grids cannot be blended voxel-by-voxel, and the backend refuses
  // to; don't offer a control that can only fail.
  $("morph").hidden = !(multi && info.morph_shape);
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
  $("meta").textContent = m.shape.slice().reverse().join("×");
  $("color").value = m.color;
  // Colour and smoothing are per map, so the shared controls have to follow the
  // active one — otherwise they show the previous map's settings and the next nudge
  // silently applies them to this one.
  $("smooth").value = String(m.smooth);
  if (models.length > 1) markActiveRow();
  await loadHistogram();
  if (state.render === "mip" || state.render === "emission") loadVolume();
  else if (state.render === "slices") loadSlices();
}

async function loadHistogram() {
  const m = active();
  const h = await (await fetch(`./api/histogram?m=${m.index}`)).json();
  hist.setData(h, h.mean + m.sigma * h.std);
}

// int32 nverts, nfaces | float32 level, mass, vmax | positions, normals, indices.
// `mass` is the enclosed density above the contour: the morph holds it constant across
// the series, so it has to come back out of the header.
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

function parseGrid(buf) {
  const dv = new DataView(buf);
  const nx = dv.getInt32(0, true), ny = dv.getInt32(4, true), nz = dv.getInt32(8, true);
  const vmin = dv.getFloat32(12, true), vmax = dv.getFloat32(16, true), mean = dv.getFloat32(20, true), std = dv.getFloat32(24, true);
  const boxLo = [dv.getFloat32(28, true), dv.getFloat32(32, true), dv.getFloat32(36, true)];
  const boxHi = [dv.getFloat32(40, true), dv.getFloat32(44, true), dv.getFloat32(48, true)];
  const data = new Uint8Array(buf, 52, nx * ny * nz);
  return { nx, ny, nz, vmin, vmax, mean, std, boxLo, boxHi, data };
}

async function fetchGrid(m, size) {
  const buf = await (await fetch(`./api/grid?m=${m}&size=${size}`)).arrayBuffer();
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

// Contour one map. Only the map being adjusted is refetched — the others are already on
// the GPU and their contours have not changed.
async function surfaceOnce(idx, draft) {
  const m = models[idx];
  if (!m) return;
  status(models.length > 1 ? `Contouring ${m.name}…` : "Contouring…", "busy");
  try {
    const q = `m=${m.index}&sigma=${m.sigma.toFixed(3)}`
            + (draft ? `&smooth=${Math.min(m.smooth, DRAFT_SMOOTH)}&size=${draftSize()}` : `&smooth=${m.smooth}`);
    const buf = await (await fetch(`./api/decode?${q}`)).arrayBuffer();
    if (state.render !== "surface") return;   // mode switched while we were waiting
    const mesh = parseMesh(buf);
    renderer.setModelMesh(m.index, mesh.positions, mesh.normals, mesh.indices);
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

async function loadVolume() {
  const m = active();
  status("Loading volume…", "busy");
  try {
    if (!m.gridVol) m.gridVol = await fetchGrid(m.index, 96);
    volRenderer.setVolume(m.gridVol.nx, m.gridVol.ny, m.gridVol.nz, m.gridVol.data, m.gridVol.boxLo, m.gridVol.boxHi);
    volRenderer.setColor(hexToRgb(m.color));
    volRenderer.setThreshold(normThreshold());
    status(`Volume ${m.gridVol.nx}×${m.gridVol.ny}×${m.gridVol.nz} · ${m.name}`
           + (models.length > 1 ? " · active map only" : ""));
  } catch (err) { status("Volume load failed: " + err.message, "err"); }
}

async function loadSlices() {
  const m = active();
  status("Loading slices…", "busy");
  try {
    if (!m.gridSlice) m.gridSlice = await fetchGrid(m.index, 192);
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

function onLevel(level, dragging) {
  const m = active();
  m.sigma = (level - m.mean) / (m.std || 1);
  if (state.render === "surface") loadSurface(state.active, !!dragging);
  else if (state.render === "mip" || state.render === "emission") volRenderer.setThreshold(normThreshold());
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

    row.append(see, swatch, name);

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
}

function markActiveRow() {
  for (const row of $("modelsList").children) {
    row.classList.toggle("active", Number(row.dataset.index) === state.active);
  }
}

function setVisible(i, on) {
  models[i].visible = on;
  renderer.setModelVisible(models[i].index, on);
  // A map switched on for the first time has no surface yet.
  if (on && state.render === "surface") loadSurfaces(); else reportSurface();
}

function setColor(i, hex) {
  models[i].color = hex;
  renderer.setModelColor(models[i].index, hexToRgb(hex));
  if (i === state.active) {
    $("color").value = hex;
    volRenderer.setColor(hexToRgb(hex));
  }
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
    : "pick maps in the panel →";
  $("morphSeq").classList.toggle("muted", !ok);
  const ready = ok && (state.render !== "surface" || (morph.cache && morph.cacheKey === morphKey()));
  $("morphPrepare").disabled = !ok;
  // Volume mode has nothing to prepare, so say so rather than leaving a live-looking
  // button that does almost nothing.
  $("morphPrepare").textContent = state.render === "surface" ? "Prepare" : "Load";
  $("morphPlay").disabled = !ready;
  $("morphRecord").disabled = !ready;
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
    volRenderer.setVolume(ga.nx, ga.ny, ga.nz, ga.data, ga.boxLo, ga.boxHi);
    volRenderer.setVolumeB(gb.nx, gb.ny, gb.nz, gb.data);
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
  if (!morph.showing) return;
  morph.showing = false;
  renderer.dropModel(MORPH_ID);
  volRenderer.setVolumeB(0, 0, 0, null);
  for (const m of models) renderer.setModelVisible(m.index, m.visible);
  if (state.render !== "surface") loadVolume();
}

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
    const level = morph.mass ? `mass=${morph.refMass}` : `sigma=${sigma.toFixed(3)}`;
    for (let k = 1; k < n; k++) {
      const { a, b, t } = frameSpec(k);
      status(`Preparing morph… ${k + 1}/${n}`, "busy");
      cache[k] = await fetchMorphMesh(a, b, t, level, smooth, size);
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

  show($("lightlbl"), isSurface);
  show($("smoothlbl"), isSurface);
  show($("volctrls"), isVolume);
  show($("slicectrls"), isSlices);
  show($("colorlbl"), !isSlices);
  show($("histStrip"), !isSlices);   // contour is meaningless for raw slices

  if (isSurface) { renderer.render(); loadSurfaces(); }
  else if (isVolume) { volRenderer.setMode(mode === "emission" ? "emission" : "mip"); loadVolume(); }
  else { loadSlices(); }
  // Morphing is a 3D notion; orthogonal slices have nothing to animate.
  $("morph").hidden = !(models.length > 1 && info && info.morph_shape) || isSlices;
  if (!$("morph").hidden) updateMorphUI();
}

// ---- controls ------------------------------------------------------------
$("rendermode").addEventListener("change", (e) => setMode(e.target.value));
$("light").addEventListener("change", (e) => renderer.setLighting(e.target.value));
$("smooth").addEventListener("input", (e) => {
  active().smooth = +e.target.value;
  active().surface = null;                     // the cached mesh was cut at the old smoothing
  if (state.render === "surface") loadSurface(state.active, false);
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
$("bg").addEventListener("input", (e) => { const rgb = hexToRgb(e.target.value); renderer.setBackground(rgb); volRenderer.setBackground(rgb); slices.setBackground(e.target.value); });
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
$("morphClear").addEventListener("click", () => { morph.seq = []; morph.cache = null; stopMorph(); updateMorphUI(); buildModelPanel(); });
$("modelsAll").addEventListener("click", () => setAllVisible(true));
$("modelsNone").addEventListener("click", () => setAllVisible(false));

function setAllVisible(on) {
  models.forEach((m, i) => {
    m.visible = on;
    renderer.setModelVisible(m.index, on);
    const row = $("modelsList").children[i];
    if (row) row.querySelector("input[type=checkbox]").checked = on;
  });
  if (on && state.render === "surface") loadSurfaces(); else reportSurface();
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

// ---- boot ----------------------------------------------------------------
(async function boot() {
  try {
    await loadInfo();
    setMode("surface");
  } catch (err) {
    status("Failed to load map: " + err.message, "err");
  }
})();
