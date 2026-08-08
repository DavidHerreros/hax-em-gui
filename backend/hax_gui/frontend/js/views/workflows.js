// Visual workflow builder: a node-graph canvas where program nodes are wired
// output->input (mapping to the job input-binding system) and run as a DAG.
// Hand-written vanilla JS; SVG for the wires, absolutely-positioned divs for nodes.
//
// Features: horizontal/vertical layout, auto-layout into dependency tiers,
// drag-and-drop add + drop-to-connect, per-node + selection runs with live
// per-node status, predict<->train parameter inheritance on wiring, file-browser
// pickers for path args, and "import a project as a workflow".
import { h, clear, mount } from "../h.js";
import { poller } from "../poller.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { navigate } from "../router.js";
import { store, getState } from "../store.js";
import { errorBox } from "./ui.js";
import { visibleFields } from "./formFields.js";
import { paramForm } from "./paramForm.js";
import { nodeLabelChips, loadLabels, labelsByIds, labelChip } from "./labels.js";
import { openContextMenu } from "./contextMenu.js";
import { promptModal } from "./promptModal.js";
import { confirmCascade, downstreamNodes, nodeAsDependent } from "./cascadeModal.js";
import { pickProject } from "./pickProject.js";
import { loadProgramGroups, isModed } from "../programGroups.js";
import { layeredLayout } from "./wfLayout.js";

const LAST_WF_KEY = "hax_gui_last_workflow";   // reopen this workflow on next visit
const ORIENT_KEY = "hax_gui_wf_orientation";   // the orientation last chosen by the user

// A workflow stores its own orientation, so this is only the default for a canvas that
// carries none: a brand-new one, or a graph imported from elsewhere.
function rememberedOrientation() {
  try { return localStorage.getItem(ORIENT_KEY) === "vertical" ? "vertical" : "horizontal"; }
  catch { return "horizontal"; }
}
function rememberOrientation(v) {
  try { localStorage.setItem(ORIENT_KEY, v); } catch { /* storage unavailable — non-fatal */ }
}

const FOCUS_KEY = "hax_gui_wf_focus";          // focus mode on? (a view preference, off by default)
function rememberedFocus() {
  try { return localStorage.getItem(FOCUS_KEY) === "1"; } catch { return false; }
}
function rememberFocus(on) {
  try { localStorage.setItem(FOCUS_KEY, on ? "1" : "0"); } catch { /* storage unavailable — non-fatal */ }
}

// Where the canvas was left, per workflow id ("" = the unsaved canvas): {sl, st, zoom}.
// Module scope, not view scope — the router builds a fresh view on every visit, so
// anything held on the instance dies with it. Deliberately in memory only: it is
// ephemeral view state, so a full page reload starts centred again.
const VIEWPORTS = new Map();

// Point the *next* visit to this page at `id`. Used by callers that create a
// workflow elsewhere (the getting-started recipe) and then navigate here: the view
// already reopens whatever was left open, so this is all the handoff that's needed.
export function rememberWorkflowToOpen(id) {
  try { localStorage.setItem(LAST_WF_KEY, id); } catch { /* storage unavailable */ }
}

const NODE_W = 196;     // node width (px)
const HEADER_H = 34;    // node header height
const ROW_H = 26;       // per-input-port row height
const CONTENT_W = 4000; // scrollable canvas content size (matches the wires SVG)
const CONTENT_H = 3000;
const SVG_NS = "http://www.w3.org/2000/svg";

// Concrete status colors for the overview map. Canvas 2D takes only resolved colors,
// and these are semantic (green succeeded / red failed) rather than themed, so unlike
// the map's ink they are literals rather than read off the computed style.
const MINI_STATUS = { saved: "#a78bfa", pending: "#94a3b8", queued: "#f59e0b", running: "#38bdf8", succeeded: "#22c55e", failed: "#ef4444", blocked: "#f59e0b", cancelling: "#f59e0b", cancelled: "#94a3b8", interrupted: "#f59e0b", idle: "#94a3b8" };

// Programs that expose a train/predict mode (derived from the backend, see
// programGroups.js) — wiring their model output into another node of the same
// program inherits the form parameters (mode aside).

// Same grouping/order as the right-menu program launcher.
const CATEGORY_ORDER = ["Preprocessing", "Networks", "Post-processing", "Metrics", "Viewers", "Other"];

const STATUS_COLOR = {
  saved: "#a78bfa", pending: "var(--muted)", running: "var(--accent)",
  succeeded: "#22c55e", failed: "var(--err)", blocked: "var(--warn)",
  cancelling: "var(--warn)", cancelled: "var(--muted)", interrupted: "var(--warn)",
  queued: "var(--warn)",
  idle: "var(--muted)", skipped: "var(--muted)",
};
// Any status not in the map must still get a concrete color, or SVG strokes /
// inline styles become "undefined" and render invisibly (e.g. a wire from a
// 'cancelling' node would silently disappear).
const statusColor = (st) => STATUS_COLOR[st] || "#64748b";

function svgEl(tag, attrs = {}) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

// Trace a rounded-rectangle path on a 2D context (for the minimap node tiles).
function roundRectPath(ctx, x, y, w, h, r) {
  r = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

export function WorkflowsView() {
  let catalog = [];
  let byName = {};
  let graph = { nodes: [], edges: [], orientation: rememberedOrientation() };
  let wf = null;            // saved workflow record
  let selected = null;      // primary selected node id (drives the inspector)
  const selectedSet = new Set();   // multi-selection (run / move together)
  let runState = null;      // {nodeId: {job_id, status, error}}
  let dirty = false;
  let poll = null;
  const schemaCache = {};
  let drag = null;          // active node drag / wiring / pan gesture
  let nodeSeq = 1, edgeSeq = 1;
  let focusMode = rememberedFocus();   // dim everything not wired to the selection?
  let minimapOn = true;     // overview window visible?
  let mmCollapsed = false;  // overview shrunk to its title pill?
  let mmXform = null;       // last minimap transform (for click-to-navigate)
  let mmView = null;        // last viewport rect drawn, in minimap px (for grab-testing)
  let mmPulse = null;       // rAF handle: only alive while a step is running

  const horiz = () => graph.orientation !== "vertical";

  const root = h("div", { class: "view workflows-view" });

  // ── top bar ──
  // Three zones instead of one flat row of ~20 buttons: the workflow (open / name /
  // a ⋮ menu for file actions), the view (one View menu + the zoom readout), and the
  // actions. The action cluster is rebuilt by syncActions() because it depends on run
  // state and selection: while a run is live it offers Stop and "Queue selected"
  // (steps appended to the live queue); otherwise a Run split button. Queued steps
  // start as soon as their device has a free slot, so several can be running at once.
  const nameInput = h("input", { class: "wf-name", placeholder: "Workflow name", value: "" });
  nameInput.addEventListener("input", () => { dirty = true; });
  const wfSelect = h("select", { class: "wf-select", title: "Open a saved workflow" });
  wfSelect.addEventListener("change", () => openWorkflow(wfSelect.value || null));
  const statusEl = h("span", { class: "wf-status muted" }, "");
  const zoomLabel = h("button", { class: "btn small wf-zoom", title: "Reset zoom to 100%", onclick: resetZoom }, "100%");
  const saveBtn = h("button", { class: "btn", onclick: save }, icon("save", 15), "Save");
  const actionsEl = h("div", { class: "wf-actions" });
  const wfMenuBtn = h("button", { class: "btn small", title: "Workflow actions",
    onclick: (e) => openContextMenu(e.currentTarget, wfMenuItems()) }, icon("more", 15));
  const viewMenuBtn = h("button", { class: "btn small", title: "View & layout",
    onclick: (e) => openContextMenu(e.currentTarget, viewMenuItems()) }, icon("sliders", 14), "View");

  // Steps run concurrently, capped per device: a step's GPU (or the CPU) is the resource
  // it contends for, so branches on different GPUs already run side by side at 1. Raising
  // this stacks that many steps on the *same* card — nothing checks free VRAM, so it is a
  // deliberate "I know these two fit" setting.
  const parSel = h("select", { class: "btn small wf-par",
    title: "How many steps may run at once on any one device.\n\n1 — one step per GPU. Independent branches on different GPUs still run in parallel.\n2+ — stack that many steps on the same GPU. Only do this if you know they fit in its memory; nothing here checks." },
    ...[1, 2, 3, 4].map((n) => h("option", { value: String(n) }, n === 1 ? "1 / device" : `${n} / device`)));
  parSel.addEventListener("change", setParallelism);

  const topbar = h("div", { class: "wf-topbar" },
    h("div", { class: "wf-zone" }, h("strong", null, "Workflow"), wfSelect, nameInput, parSel, wfMenuBtn),
    h("div", { class: "wf-zone" }, viewMenuBtn, zoomLabel),
    h("span", { class: "wf-spacer" }),
    statusEl, actionsEl);

  // Menus are built on open, so they always read the current workflow / layout state.
  function wfMenuItems() {
    const items = [
      { label: "New workflow", icon: icon("plus", 15), onClick: () => openWorkflow(null) },
      { label: "Open project as workflow", icon: icon("box", 15), onClick: openProject },
    ];
    if (!(wf && wf.project_id)) items.push({ label: "Save as project", icon: icon("save", 15), onClick: saveAsProject });
    if (wf && wf.project_id) items.push({ label: "Reconnect step inputs", icon: icon("link", 15), onClick: relinkInputs });
    if (wf) items.push({ label: "Duplicate to project…", icon: icon("copy", 15), onClick: duplicateToProject });
    if (wf) items.push({ label: "Delete workflow", icon: icon("trash", 15), danger: true, onClick: removeWorkflow });
    return items;
  }

  // Repair links dropped by an edit-and-rerun: the backend re-derives a step's input
  // bindings from the literal paths its args froze in, then we reload so the recovered
  // edges show. Only reconnects unambiguous matches; a hand-typed path is left alone.
  async function relinkInputs() {
    if (!(wf && wf.project_id)) { flash("Open a project workflow first.", "warn"); return; }
    flash("Reconnecting step inputs…", "busy");
    try {
      const r = await api.relinkProject(wf.project_id);
      if (r.count) { await openWorkflow(wf.id); flash(`Reconnected ${r.count} input${r.count === 1 ? "" : "s"}.`, "muted"); }
      else flash("No inputs to reconnect — every wired input is already linked.", "muted");
    } catch (err) { flash("Reconnect failed: " + (err.detail || err.message), "err"); }
  }

  // Clone this whole pipeline into another project (a fresh, standalone copy). Internal
  // step dependencies travel with it; external data inputs are cleared to set there.
  async function duplicateToProject() {
    if (!wf) return;
    const target = await pickProject({
      title: "Duplicate workflow to project", okText: "copy here", excludeId: wf.project_id,
      hint: "Creates a self-contained copy of this pipeline in the chosen project. Step dependencies are kept; external data inputs are cleared for you to set before running.",
    });
    if (!target) return;
    try {
      const copy = await api.copyWorkflow(wf.id, target.id, null);
      store.openProject({ id: target.id, name: target.name });   // switch context so the copy lists
      await refreshList();
      await openWorkflow(copy.id);
      flash(`Copied to “${target.name}”.`, "muted");
    } catch (err) {
      flash(err && err.detail ? (typeof err.detail === "string" ? err.detail : "Could not copy workflow.") : "Could not copy workflow.", "err");
    }
  }
  function viewMenuItems() {
    const items = [
      { label: "Tidy — arrange by dependency", icon: icon("sliders", 15), onClick: () => { autoLayout(); dirty = true; } },
      { label: horiz() ? "Switch to vertical" : "Switch to horizontal", icon: icon(horiz() ? "chevronRight" : "arrowRight", 15), onClick: toggleOrientation },
      { label: "Center in view", icon: icon("orbit", 15), onClick: centerView },
      { label: minimapOn ? "Hide overview map" : "Show overview map", icon: icon("grid", 15), onClick: toggleMinimap },
      { label: focusMode ? "Focus mode: on" : "Focus mode: off", icon: icon(focusMode ? "eye" : "eyeOff", 15), onClick: toggleFocusMode },
      { label: "Pin this layout", icon: icon("bookmark", 15), onClick: saveLayout },
    ];
    if (graph.savedLayout && graph.savedLayout.pos) {
      items.push({ label: "Restore pinned layout", icon: icon("restore", 15), onClick: restoreLayout });
    }
    return items;
  }

  // A primary action joined to a caret holding its variant — same pattern as the job page.
  function splitButton(main, items) {
    return h("div", { class: "btn-split" },
      h("button", { class: "btn split-main", title: main.title || "", onclick: main.onClick }, icon(main.icon, 15), main.label),
      h("button", { class: "btn split-caret", title: "More…", onclick: (e) => { e.stopPropagation(); openContextMenu(e.currentTarget, items); } },
        h("span", { class: "split-chevron" }, icon("chevronUp", 13))));
  }

  // Persist the per-device cap on its own (not via save()), so changing it never drags
  // an unsaved canvas along with it. The scheduler re-reads it every tick (~2s), so
  // raising it mid-run starts the queued steps without waiting for anything to finish.
  async function setParallelism() {
    const n = Number(parSel.value) || 1;
    if (!wf) return;                       // unsaved workflow: applied when it is created
    try {
      wf = await api.updateWorkflow(wf.id, { max_per_device: n });
      flash(n === 1
        ? "One step per device — branches on different GPUs still run in parallel."
        : `Up to ${n} steps per device. Make sure they fit in one GPU's memory.`, "muted");
    } catch (err) {
      parSel.value = String((wf && wf.max_per_device) || 1);
      flash("Could not change the step limit: " + (err.detail || err.message), "err");
    }
  }

  function syncActions() {
    parSel.value = String((wf && wf.max_per_device) || 1);
    parSel.disabled = !wf;
    clear(actionsEl);
    const running = !!(wf && wf.status === "running");
    const n = selectedSet.size;
    if (n) {   // selection tools only exist while there *is* a selection
      actionsEl.append(
        h("span", { class: "wf-selcount" }, `${n} selected`),
        h("button", { class: "btn secondary small", title: "Duplicate the selected step(s) as new, un-run steps (Ctrl/Cmd+D)", onclick: () => duplicateNodes([...selectedSet]) }, icon("copy", 14)),
        h("button", { class: "btn secondary small", title: "Remove the selected step(s) (Delete / Backspace). Save to persist.", onclick: () => removeNodes([...selectedSet]) }, icon("x", 14)));
    }
    actionsEl.appendChild(saveBtn);
    if (running) {
      if (n) actionsEl.appendChild(h("button", { class: "btn secondary", title: "Queue the selected step(s) onto the running workflow — each starts as soon as its dependencies are met and its device is free",
        onclick: () => runNodes([...selectedSet]) }, icon("plus", 15), "Queue selected"));
      actionsEl.appendChild(h("button", { class: "btn secondary", style: { color: "var(--warn)", borderColor: "var(--warn)" },
        title: "Stop the run: cancel every step running now and dequeue the rest", onclick: stopRun }, icon("ban", 15), "Stop"));
    } else {
      actionsEl.appendChild(splitButton(
        { icon: "play", label: "Run all", title: "Run the whole workflow", onClick: () => runNodes(null) },
        [{ label: "Run selected steps", icon: icon("play", 15), onClick: () => runNodes([...selectedSet]) }]));
    }
  }

  // ── body: palette | canvas (+minimap) | inspector ──
  // Both side panels are resizable (drag the divider) and collapsible (chevron in
  // the panel bar; a thin reopen tab brings it back), like the main-window menus.
  let paletteW = 200, inspectorW = 264;
  const paletteInner = h("div", { class: "wf-pane-inner" });
  const palette = h("div", { class: "wf-palette", style: { width: paletteW + "px" } },
    h("div", { class: "wf-pane-bar" },
      h("span", { class: "wf-pane-title" }, "Programs"),
      h("button", { class: "wf-pane-collapse", title: "Hide programs", onclick: () => setPalette(false) }, icon("chevronLeft", 16))),
    paletteInner);

  // The content (wires + nodes) lives in a fixed CONTENT_W×CONTENT_H layer that is
  // CSS-scaled by `zoom`; a sizer of the scaled size gives the scroll container its
  // extent. Node coords stay in content units — only the view scales.
  let zoom = 1;
  const ZOOM_MIN = 0.2, ZOOM_MAX = 2.5;
  const svg = svgEl("svg", { class: "wf-wires", width: String(CONTENT_W), height: String(CONTENT_H) });
  const nodesLayer = h("div", { class: "wf-nodes" });
  const contentLayer = h("div", { class: "wf-content", style: { width: CONTENT_W + "px", height: CONTENT_H + "px" } }, svg, nodesLayer);
  const sizer = h("div", { class: "wf-sizer" }, contentLayer);
  const canvas = h("div", { class: "wf-canvas" }, sizer);
  // ── overview panel ──
  // A titled card rather than a bare canvas: the header carries the one number that
  // is hard to read off a 200px map (how many steps, and how many are live) and the
  // collapse control, so the map itself stays pure graph.
  const mmCanvas = h("canvas", { class: "wf-mm-canvas" });
  const mmCount = h("span", { class: "wf-mm-count" }, "");
  const mmCollapse = h("button", { class: "wf-mm-btn", title: "Collapse overview",
    onclick: (e) => { e.stopPropagation(); setMinimapCollapsed(!mmCollapsed); } }, icon("chevronUp", 13));
  const mmBar = h("div", { class: "wf-mm-bar", title: "Overview" },
    icon("grid", 12), h("span", { class: "wf-mm-title" }, "Overview"), mmCount, mmCollapse);
  // Clicking the bar while collapsed reopens it — the whole pill is the target, not
  // just the 13px chevron.
  mmBar.addEventListener("click", () => { if (mmCollapsed) setMinimapCollapsed(false); });
  const minimap = h("div", { class: "wf-mm" }, mmBar, mmCanvas);
  const canvasWrap = h("div", { class: "wf-canvas-wrap" }, canvas, minimap);
  // Rubber-band selection rectangle. Lives inside contentLayer (content units), so
  // it scales/scrolls with the nodes; it survives renderNodes() (which only clears
  // nodesLayer). Shown only while a Shift/Ctrl-drag marquee is in progress.
  const marqueeEl = h("div", { class: "wf-marquee", style: { display: "none" } });
  contentLayer.appendChild(marqueeEl);

  function applyZoom() {
    sizer.style.width = CONTENT_W * zoom + "px";
    sizer.style.height = CONTENT_H * zoom + "px";
    contentLayer.style.transform = `scale(${zoom})`;
    updateZoomLabel();
  }
  function updateZoomLabel() { zoomLabel.textContent = Math.round(zoom * 100) + "%"; }
  // Zoom to `next`, keeping the content point under (mx,my) (canvas-local px) fixed.
  function zoomTo(next, mx, my) {
    next = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, next));
    if (next === zoom) return;
    const cx = (canvas.scrollLeft + mx) / zoom, cy = (canvas.scrollTop + my) / zoom;
    zoom = next; applyZoom();
    canvas.scrollLeft = cx * zoom - mx;
    canvas.scrollTop = cy * zoom - my;
    drawMinimap();
    rememberViewport();   // a zoom at the scroll extremes moves no scrollbar, so record it here too
  }
  function resetZoom() { zoomTo(1, canvas.clientWidth / 2, canvas.clientHeight / 2); }
  applyZoom();
  // Mouse wheel zooms (toward the cursor) instead of scrolling.
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    zoomTo(zoom * Math.exp(-e.deltaY * 0.0015), e.clientX - rect.left, e.clientY - rect.top);
  }, { passive: false });

  const inspBody = h("div", { class: "wf-pane-inner wf-inspector-body" });
  const inspector = h("div", { class: "wf-inspector", style: { width: inspectorW + "px" } },
    h("div", { class: "wf-pane-bar" },
      h("button", { class: "wf-pane-collapse", title: "Hide inspector", onclick: () => setInspector(false) }, icon("chevronRight", 16)),
      h("span", { class: "wf-pane-title" }, "Inspector")),
    inspBody);

  const paletteRz = h("div", { class: "wf-rz", title: "Drag to resize" });
  const inspectorRz = h("div", { class: "wf-rz", title: "Drag to resize" });
  paletteRz.addEventListener("pointerdown", (e) => startResize(e, "palette"));
  inspectorRz.addEventListener("pointerdown", (e) => startResize(e, "inspector"));
  const reopenLeft = h("button", { class: "wf-reopen left hidden", title: "Show programs", onclick: () => setPalette(true) }, icon("chevronRight", 16));
  const reopenRight = h("button", { class: "wf-reopen right hidden", title: "Show inspector", onclick: () => setInspector(true) }, icon("chevronLeft", 16));

  const body = h("div", { class: "wf-body" }, palette, paletteRz, canvasWrap, inspectorRz, inspector, reopenLeft, reopenRight);
  root.append(topbar, body);

  function setPalette(open) {
    palette.classList.toggle("collapsed", !open);
    paletteRz.classList.toggle("hidden", !open);
    reopenLeft.classList.toggle("hidden", open);
    if (open) palette.style.width = paletteW + "px";
    drawMinimap();
  }
  function setInspector(open) {
    inspector.classList.toggle("collapsed", !open);
    inspectorRz.classList.toggle("hidden", !open);
    reopenRight.classList.toggle("hidden", open);
    if (open) inspector.style.width = inspectorW + "px";
    drawMinimap();
  }
  function startResize(e, which) {
    e.preventDefault();
    const startX = e.clientX, pw = paletteW, iw = inspectorW;
    document.body.style.cursor = "col-resize";
    function mv(ev) {
      if (which === "palette") { paletteW = Math.max(150, Math.min(400, pw + (ev.clientX - startX))); palette.style.width = paletteW + "px"; }
      else { inspectorW = Math.max(200, Math.min(520, iw - (ev.clientX - startX))); inspector.style.width = inspectorW + "px"; }
      drawMinimap();
    }
    function up() { window.removeEventListener("pointermove", mv); window.removeEventListener("pointerup", up); document.body.style.cursor = ""; }
    window.addEventListener("pointermove", mv);
    window.addEventListener("pointerup", up);
  }

  // Empty-canvas drag pans the view; a Shift/Ctrl/Cmd-drag instead draws a marquee
  // to box-select nodes. A plain click without movement clears the selection.
  canvas.addEventListener("pointerdown", (e) => {
    if (e.target === canvas || e.target === nodesLayer || e.target === svg || e.target === sizer || e.target === contentLayer) {
      e.preventDefault();
      if (e.shiftKey || e.ctrlKey || e.metaKey) {
        const p = canvasPoint(e);
        // Ctrl/Cmd adds to the current selection; Shift alone replaces it.
        const add = e.ctrlKey || e.metaKey;
        drag = { kind: "marquee", x0: p.x, y0: p.y, x1: p.x, y1: p.y, base: add ? new Set(selectedSet) : new Set() };
        updateMarquee();
        canvas.classList.add("marquee-select");
      } else {
        drag = { kind: "pan", sx: e.clientX, sy: e.clientY, sl: canvas.scrollLeft, st: canvas.scrollTop, moved: false };
        canvas.classList.add("panning");
      }
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
    }
  });
  canvas.addEventListener("scroll", () => { drawMinimap(); rememberViewport(); });
  // Double-click a step -> open the job it ran. Delegated to the canvas and resolved by
  // hit-testing rather than bound per node: pressing a node re-renders the nodes layer,
  // so the two clicks land on different elements and the dblclick is dispatched to
  // their common ancestor (this canvas), never to the node itself.
  canvas.addEventListener("dblclick", (e) => {
    const under = document.elementFromPoint(e.clientX, e.clientY);
    if (under && under.closest(".wf-node-x, .wf-node-run, .wf-port")) return;   // own gestures
    const id = elementNodeAt(e.clientX, e.clientY);
    if (!id) return;
    e.preventDefault();               // suppress the double-click text selection
    openJobFor(id);
  });
  // The panel is sized in CSS (and shrinks with the canvas on narrow screens), so the
  // backing store has to follow it rather than be set once.
  if (window.ResizeObserver) new ResizeObserver(() => drawMinimap()).observe(mmCanvas);
  // The minimap drives the viewport by dragging, not just by clicking: press inside
  // the viewport rectangle to grab it where you touched it (no jump), press outside
  // to recentre there and keep dragging from that point. Both are the same gesture,
  // so a click is simply a drag of zero length.
  mmCanvas.addEventListener("pointerdown", (e) => {
    e.stopPropagation(); e.preventDefault();
    if (!mmXform) return;
    const r = mmCanvas.getBoundingClientRect();
    const px = e.clientX - r.left, py = e.clientY - r.top;
    const inside = mmView && px >= mmView.x && px <= mmView.x + mmView.w
                           && py >= mmView.y && py <= mmView.y + mmView.h;
    // Offset from the viewport centre, so grabbing an edge doesn't snap it to the cursor.
    const offX = inside ? px - (mmView.x + mmView.w / 2) : 0;
    const offY = inside ? py - (mmView.y + mmView.h / 2) : 0;
    const move = (ev) => {
      const mx = ev.clientX - r.left - offX, my = ev.clientY - r.top - offY;
      scrollToContent(mmXform.gx0 + (mx - mmXform.ox) / mmXform.s,
                      mmXform.gy0 + (my - mmXform.oy) / mmXform.s);
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      mmCanvas.classList.remove("dragging");
    };
    mmCanvas.classList.add("dragging");
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    move(e);
  });
  // Wheel over the map zooms the canvas about the viewport centre rather than
  // scrolling the page behind it.
  mmCanvas.addEventListener("wheel", (e) => {
    e.preventDefault(); e.stopPropagation();
    zoomTo(zoom * Math.exp(-e.deltaY * 0.0015), canvas.clientWidth / 2, canvas.clientHeight / 2);
  }, { passive: false });

  // drag-and-drop a palette program onto the canvas (drop on a node => auto-wire)
  canvas.addEventListener("dragover", (e) => { if (drag?.kind !== "node") { e.preventDefault(); e.dataTransfer.dropEffect = "copy"; canvas.classList.add("wf-drop"); } });
  canvas.addEventListener("dragleave", (e) => { if (e.target === canvas) canvas.classList.remove("wf-drop"); });
  canvas.addEventListener("drop", (e) => {
    e.preventDefault(); canvas.classList.remove("wf-drop");
    const program = e.dataTransfer.getData("text/plain");
    if (!program || !byName[program]) return;
    const overNode = elementNodeAt(e.clientX, e.clientY);   // existing node under the drop, if any
    const p = canvasPoint(e);
    const n = addNode(program, { x: Math.max(0, p.x - NODE_W / 2), y: Math.max(0, p.y - HEADER_H / 2) });
    if (overNode && overNode !== n.id) autoConnect(overNode, n.id);
  });

  // ── data load ──
  async function boot() {
    try {
      catalog = await api.workflowCatalog();
      byName = Object.fromEntries(catalog.map((p) => [p.name, p]));
    } catch (err) { mount(palette, errorBox(err)); return; }
    renderPalette(); syncActions();
    loadLabels().catch(() => {});
    loadProgramGroups().catch(() => {});
    await refreshList();
    // Reopen the workflow left open last time (if it still exists), so entering the
    // page resumes where you were instead of a blank canvas. Falls back to empty.
    const last = localStorage.getItem(LAST_WF_KEY);
    if (last && wfList.some((w) => w.id === last)) openWorkflow(last);
    else openWorkflow(null);
  }

  let wfList = [];   // last-fetched saved-workflow list (drives the reopen check)
  async function refreshList() {
    const pid = getState().activeProjectId;
    try { wfList = await api.workflows(pid); } catch { wfList = []; }
    clear(wfSelect);
    wfSelect.appendChild(h("option", { value: "" }, "— saved workflows —"));
    for (const w of wfList) wfSelect.appendChild(h("option", { value: w.id }, w.name || "Untitled"));
    if (wf) wfSelect.value = wf.id;
  }

  function renderPalette() {
    clear(paletteInner);
    paletteInner.appendChild(h("div", { class: "muted small", style: { padding: "8px 10px 6px" } }, "Click or drag onto the canvas."));
    // Group by program category, same order as the right-menu launcher.
    const byCat = {};
    for (const p of catalog) (byCat[p.category || "Other"] ??= []).push(p);
    const cats = Object.keys(byCat).sort((a, b) => ((CATEGORY_ORDER.indexOf(a) + 1) || 99) - ((CATEGORY_ORDER.indexOf(b) + 1) || 99));
    for (const cat of cats) {
      paletteInner.appendChild(h("div", { class: "wf-pal-cat" }, cat));
      for (const p of byCat[cat]) {
        const producesTag = p.produces && p.produces.length ? "out: " + (p.produces.includes("latents") ? "latents" : p.produces[0]) : "";
        const item = h("button", { class: "wf-pal-item", title: p.description || p.name, draggable: "true", onclick: () => addNode(p.name) },
          h("span", { class: "mono" }, p.name),
          producesTag ? h("span", { class: "wf-pal-out muted small" }, producesTag) : null);
        item.addEventListener("dragstart", (e) => { e.dataTransfer.setData("text/plain", p.name); e.dataTransfer.effectAllowed = "copy"; });
        paletteInner.appendChild(item);
      }
    }
    paletteInner.appendChild(h("div", { class: "wf-legend" },
      ...["saved", "succeeded", "running", "failed", "blocked"].map((s) =>
        h("span", { class: "wf-leg" }, h("span", { class: "wf-leg-dot", style: { background: STATUS_COLOR[s] } }), s))));
  }

  // ── graph mutation ──
  function addNode(program, pos) {
    const x = pos ? pos.x : 60 + (graph.nodes.length % 4) * 60 + canvas.scrollLeft;
    const y = pos ? pos.y : 60 + (graph.nodes.length % 6) * 40 + canvas.scrollTop;
    const id = "n" + (nodeSeq++);
    const node = { id, program, args: {}, gpu: "0", name: "", labels: [], x, y };
    graph.nodes.push(node);
    selectOnly(id); dirty = true;
    renderNodes(); renderInspector();
    return node;
  }
  // The job backing a node, if any: project-canvas nodes carry `job_id`; a
  // standalone workflow that has run maps its nodes to jobs via runState.
  const jobIdOf = (id) => nodeById(id)?.job_id || (runState && runState[id] && runState[id].job_id) || null;

  // Double-clicking a step opens the job page for the job it ran — live stdout/stderr,
  // progress, outputs. A step that has never run has no job to open yet. The canvas is
  // saved on the way out: leaving the view discards unsaved edits, and the job page is
  // a navigation away, not a panel.
  async function openJobFor(id) {
    const jid = jobIdOf(id);
    if (!jid) { flash("That step hasn't run yet — no job to open.", "warn"); return; }
    if (dirty && wf) await save();
    navigate("/jobs/" + jid);
  }

  // Remove every step in `ids` (and any edges touching them) at once, and delete
  // the jobs those steps produced (DB record + output/log folder). Confirms first —
  // this permanently discards outputs and can't be undone.
  //
  // The removal is saved for you: leaving it as an unsaved canvas edit meant a step
  // whose job was already deleted came back the next time the page was opened.
  // A step whose job *couldn't* be deleted (it is still running) is put back instead,
  // so the canvas keeps matching what actually exists.
  async function removeNodes(ids) {
    const set = new Set((ids || []).filter(Boolean));
    if (!set.size) { flash("Select step(s) to remove.", "muted"); return; }
    const withJob = [...set].map((id) => [id, jobIdOf(id)]).filter(([, jid]) => jid);
    const jobIds = withJob.map(([, jid]) => jid);
    const n = set.size;
    let msg = n > 1 ? `Remove ${n} selected steps from the canvas?` : "Remove this step from the canvas?";
    if (jobIds.length) msg += `\n\nThis also permanently deletes ${jobIds.length} associated job${jobIds.length > 1 ? "s" : ""} — their output folders and logs — and cannot be undone.`;
    if ((n > 1 || jobIds.length) && !confirm(msg)) return;

    // keep what we drop, so a step whose job survives can be restored below
    const cutNodes = graph.nodes.filter((nd) => set.has(nd.id));
    const cutEdges = graph.edges.filter((e) => set.has(e.from) || set.has(e.to));
    const cutState = {};
    for (const id of set) if (runState && runState[id]) cutState[id] = runState[id];

    // drop the boxes + their wiring from the canvas
    graph.nodes = graph.nodes.filter((nd) => !set.has(nd.id));
    graph.edges = graph.edges.filter((e) => !set.has(e.from) && !set.has(e.to));
    for (const id of set) { selectedSet.delete(id); if (runState) delete runState[id]; }
    if (set.has(selected)) selected = selectedSet.size ? [...selectedSet][0] : null;
    dirty = true; renderNodes(); renderInspector();

    // delete the backing jobs (best-effort: an active job can't be deleted, so it
    // is reported rather than silently skipped)
    let failed = 0;
    if (jobIds.length) {
      const results = await Promise.allSettled(withJob.map(([, jid]) => api.deleteJob(jid)));
      const kept = new Set(withJob.filter((_, i) => results[i].status === "rejected").map(([id]) => id));
      failed = kept.size;
      if (kept.size) {                       // put the un-deletable steps back
        for (const nd of cutNodes) if (kept.has(nd.id)) graph.nodes.push(nd);
        const present = new Set(graph.nodes.map((nd) => nd.id));
        for (const e of cutEdges) {
          if (present.has(e.from) && present.has(e.to) && !graph.edges.some((x) => x.id === e.id)) graph.edges.push(e);
        }
        if (runState) for (const id of kept) if (cutState[id]) runState[id] = cutState[id];
        renderNodes(); renderInspector();
      }
    }

    const removed = n - failed;
    if (wf && removed) await save();          // make the removal outlive this page visit
    const noun = removed === 1 ? "step" : "steps";
    if (!removed) flash(`Nothing removed — ${failed === 1 ? "that step's job is" : "those steps' jobs are"} still running (cancel first).`, "warn");
    else if (failed) flash(`Removed ${removed} ${noun}; ${failed} could not be removed — their job is still running (cancel it first).`, "warn");
    else if (jobIds.length) flash(`Removed ${removed} ${noun} and deleted ${jobIds.length - failed} job(s).`, "muted");
    else flash(wf ? `Removed ${removed} ${noun}.` : `Removed ${removed} ${noun} — Save to persist.`, "muted");
  }

  // "<base> copy", then "<base> copy 2", "<base> copy 3", … : a copy of a copy re-uses the
  // base instead of stacking suffixes ("Name copy copy copy"), and the number is the first
  // one free among `taken`, so duplicating any of the siblings lands on the same series.
  function copyName(name, taken) {
    const trimmed = name.trim();
    const base = trimmed.replace(/\s+copy(\s+\d+)?$/i, "").trim() || trimmed;
    let candidate = `${base} copy`;
    for (let i = 2; taken.has(candidate); i++) candidate = `${base} copy ${i}`;
    return candidate;
  }

  // Duplicate node(s) as fresh, un-run steps: copy program/args/gpu/name/labels at
  // an offset and preserve edges internal to the set, leaving the copies unlinked
  // (no job_id) so they become new jobs only when the workflow is run. The copies
  // live in the graph and persist on Save.
  function duplicateNodes(ids) {
    const src = [...new Set(ids || [])].map(nodeById).filter(Boolean);
    if (!src.length) { flash("Select a step to duplicate.", "muted"); return; }
    const idMap = new Map(), made = [];
    // every name on the canvas, grown as we go so a multi-step duplicate does not
    // hand the same "… copy" to two of its own copies
    const taken = new Set(graph.nodes.map((n) => (n.name || "").trim()).filter(Boolean));
    for (const n of src) {
      const id = "n" + (nodeSeq++);
      idMap.set(n.id, id);
      const name = n.name ? copyName(n.name, taken) : "";
      if (name) taken.add(name);
      graph.nodes.push({
        id, program: n.program,
        args: JSON.parse(JSON.stringify(n.args || {})),   // deep copy so edits don't bleed back
        gpu: n.gpu || "0",
        name,
        labels: [...(n.labels || [])],
        x: (n.x || 0) + 40, y: (n.y || 0) + 40,
      });
      made.push(id);
    }
    // re-create only the wiring whose both ends are in the duplicated set
    const inSet = new Set(src.map((n) => n.id));
    for (const e of graph.edges.slice()) {
      if (inSet.has(e.from) && inSet.has(e.to)) {
        graph.edges.push({ id: "e" + (edgeSeq++), from: idMap.get(e.from), to: idMap.get(e.to), field: e.field, kind: e.kind });
      }
    }
    selectedSet.clear(); for (const id of made) selectedSet.add(id);
    selected = made[made.length - 1];
    dirty = true; renderNodes(); renderInspector();
    flash(made.length === 1 ? "Step duplicated — Save to keep it." : `${made.length} steps duplicated — Save to keep them.`, "muted");
  }
  const isMultiField = (program, field) => (byName[program]?.bindable_multi || []).includes(field);

  function addEdge(fromId, toId, field, kind) {
    if (fromId === toId) return;
    const dstProg = nodeById(toId)?.program;
    if (isMultiField(dstProg, field)) {
      // multi-valued input (e.g. FlexConsensus --input_space): keep all sources,
      // just avoid wiring the exact same source twice.
      if (graph.edges.some((e) => e.to === toId && e.field === field && e.from === fromId)) return;
    } else {
      graph.edges = graph.edges.filter((e) => !(e.to === toId && e.field === field));  // one source per field
    }
    graph.edges.push({ id: "e" + (edgeSeq++), from: fromId, to: toId, field, kind });
    inheritParams(fromId, toId, kind);
    dirty = true; renderNodes(); renderInspector();
  }
  function removeEdge(id) { graph.edges = graph.edges.filter((e) => e.id !== id); dirty = true; renderNodes(); renderInspector(); }

  // When a node's model output feeds another node of the same moded program
  // (train -> predict), copy the source's parameters into the target (mode aside),
  // since a predict run usually matches its training run apart from --mode.
  function inheritParams(fromId, toId, kind) {
    const src = nodeById(fromId), dst = nodeById(toId);
    if (!src || !dst || src.program !== dst.program) return;
    if (!isModed(dst.program) || !String(kind || "").endsWith("_model")) return;
    const wiredFields = new Set(graph.edges.filter((e) => e.to === toId).map((e) => e.field));
    for (const [k, v] of Object.entries(src.args || {})) {
      if (k === "mode" || wiredFields.has(k)) continue;
      if (!(k in dst.args)) dst.args[k] = v;
    }
    if (!("mode" in dst.args)) dst.args.mode = "predict";
    if (!("mode" in src.args)) src.args.mode = "train";
  }

  // A node's output kinds, narrowed by its mode: a moded network yields its trained
  // *_model checkpoint only in `train`, and its predict metadata (latents/particles)
  // only in `predict`. Mirrors adapters.job_produces on the backend, so a train node
  // can't be wired as a latents/particles source, nor a predict node as a model.
  function nodeProduces(node) {
    let all = (node && byName[node.program]?.produces) || [];
    const args = (node && node.args) || {};
    // A displacement field exists only under --transport_mass, so a plain HetSIREN node
    // must not offer the port reconstruct_motion_corrected needs. Mirrors job_produces.
    if (!args.transport_mass) all = all.filter((k) => k !== "transport_motion_model");
    const mode = args.mode;
    if (mode === "predict") return all.filter((k) => !k.endsWith("_model"));
    if (mode === "train") return all.filter((k) => k.endsWith("_model"));
    return all;
  }

  // Auto-wire a freshly dropped node `b` to the node `a` it landed on: prefer
  // a -> b (a's output into b's input), else b -> a, on the first matching kind.
  function autoConnect(aId, bId) {
    const a = nodeById(aId), b = nodeById(bId);
    if (!a || !b) return;
    const aOut = nodeProduces(a), bOut = nodeProduces(b);
    const aIn = inputsOf(a.program), bIn = inputsOf(b.program);
    let hit = bIn.find(([, kind]) => aOut.includes(kind));
    if (hit) return addEdge(aId, bId, hit[0], hit[1]);
    hit = aIn.find(([, kind]) => bOut.includes(kind));
    if (hit) { addEdge(bId, aId, hit[0], hit[1]); return; }
    flash(`No compatible ports between ${a.program} and ${b.program}.`);
  }

  const nodeById = (id) => graph.nodes.find((n) => n.id === id);
  const inputsOf = (program) => Object.entries((byName[program] || {}).bindable || {});
  const LABEL_ROW_H = 22;   // height of one wrapped row of label chips
  const nodeLabels = (n) => labelsByIds(n.labels);
  // Estimate how many rows the label chips wrap into within NODE_W so the node
  // grows to fit them (instead of clipping). Deterministic — same on every
  // render — so the computed nodeHeight keeps the output port aligned.
  const labelRowCount = (labels) => {
    const avail = NODE_W - 16;                 // strip inner width (8px padding each side)
    let rows = 1, cur = 0;
    for (const l of labels) {
      const nameW = Math.min(96, (l.name || "").length * 5.3);
      const w = 26 + nameW;                    // dot + gaps + padding + border + name
      if (cur === 0) cur = w;
      else if (cur + 4 + w <= avail) cur += 4 + w;
      else { rows += 1; cur = w; }
    }
    return rows;
  };
  const nodeHeight = (n) => {
    const labels = nodeLabels(n);
    return HEADER_H + Math.max(1, inputsOf(n.program).length) * ROW_H + 8
      + (labels.length ? labelRowCount(labels) * LABEL_ROW_H + 8 : 0);
  };

  // Port positions (canvas coords) — orientation-aware.
  function outPort(n) {
    return horiz() ? { x: n.x + NODE_W, y: n.y + HEADER_H / 2 }
                   : { x: n.x + NODE_W / 2, y: n.y + nodeHeight(n) };
  }
  function inPort(n, idx, count) {
    if (horiz()) return { x: n.x, y: n.y + HEADER_H + idx * ROW_H + ROW_H / 2 };
    const c = Math.max(1, count);
    return { x: n.x + (NODE_W * (idx + 1)) / (c + 1), y: n.y };
  }

  // ── selection ──
  function selectOnly(id) { selectedSet.clear(); if (id) selectedSet.add(id); selected = id; }
  function toggleSel(id) { if (selectedSet.has(id)) selectedSet.delete(id); else selectedSet.add(id); selected = selectedSet.has(id) ? id : ([...selectedSet][0] || null); }

  // ── focus mode ──
  // What the selection is wired to: the selected steps, everything one hop up- or
  // downstream, and the edges between them. Null when focus mode is off or nothing is
  // selected — i.e. nothing to pick out, so the canvas renders normally.
  function focusSets() {
    if (!focusMode || !selectedSet.size) return null;
    const nodes = new Set(selectedSet), edges = new Set();
    for (const e of graph.edges) {
      if (!selectedSet.has(e.from) && !selectedSet.has(e.to)) continue;
      edges.add(e.id); nodes.add(e.from); nodes.add(e.to);
    }
    return { nodes, edges };
  }
  function toggleFocusMode() {
    focusMode = !focusMode;
    rememberFocus(focusMode);
    renderNodes();
    flash(focusMode
      ? "Focus mode on — selecting a step dims everything it isn't wired to."
      : "Focus mode off.", "muted");
  }

  // ── rendering ──
  function renderNodes() {
    clear(nodesLayer);
    const focus = focusSets();
    for (const n of graph.nodes) {
      const el = nodeEl(n);
      // three tiers: the selection (already ringed by .sel/.primary), the steps it is
      // wired to, and everything else.
      if (focus) el.classList.add(!focus.nodes.has(n.id) ? "dim" : (selectedSet.has(n.id) ? "focused" : "near"));
      nodesLayer.appendChild(el);
    }
    redrawWires(null, focus);
    drawMinimap();
    syncActions();   // the action cluster follows the selection and the run state
  }

  function portEl(cls, originX, originY, pos, data) {
    const el = h("span", { class: "wf-port abs " + cls, style: { left: pos.x - originX - 6 + "px", top: pos.y - originY - 6 + "px" } });
    Object.assign(el.dataset, data);
    return el;
  }

  function nodeEl(n) {
    const st = runState && runState[n.id] ? runState[n.id].status : null;
    // Its inputs were re-made after it ran, so what it holds no longer matches them.
    const stale = !!(runState && runState[n.id] && runState[n.id].stale);
    const running = wf && wf.status === "running";
    const el = h("div", {
      class: "wf-node" + (selectedSet.has(n.id) ? " sel" : "") + (selected === n.id ? " primary" : "") + (stale ? " stale" : ""),
      style: { left: n.x + "px", top: n.y + "px", width: NODE_W + "px", height: nodeHeight(n) + "px" },
    });
    el.dataset.node = n.id;
    if (st) el.style.borderColor = STATUS_COLOR[st] || "var(--border)";
    // the dblclick that opens this step's job is delegated to the canvas
    if (jobIdOf(n.id)) el.title = "Double-click to open this step's job (log, outputs)";
    // header (drag handle + select + run + remove)
    const head = h("div", { class: "wf-node-head" },
      h("span", { class: "wf-node-title mono", title: n.program }, n.name || n.program),
      stale ? h("span", { class: "wf-node-stale", title: "A step this one depends on has been re-run since — these outputs are out of date. Run it again to refresh them." }, icon("refresh", 11)) : null,
      st ? h("span", { class: "wf-node-st", style: { color: statusColor(st) } }, st) : null,
      h("button", { class: "wf-node-run", title: running ? "Queue this step onto the running workflow — it starts when its device is free" : "Run this step", onclick: (e) => { e.stopPropagation(); runNodes([n.id]); } }, icon(running ? "plus" : "play", 12)),
      h("button", { class: "wf-node-run", title: "Duplicate this step (un-run copy)", onclick: (e) => { e.stopPropagation(); duplicateNodes([n.id]); } }, icon("copy", 12)),
      h("button", { class: "wf-node-x", title: "Remove step (deletes its job + outputs)", onclick: (e) => { e.stopPropagation(); removeNodes([n.id]); } }, "×"));
    head.addEventListener("pointerdown", (e) => startNodeDrag(e, n));
    el.appendChild(head);
    // input ports (label rows in the body + an absolutely-placed dot per port)
    const ins = inputsOf(n.program);
    ins.forEach(([field, kind], i) => {
      const wiredAll = graph.edges.filter((e) => e.to === n.id && e.field === field);
      const multi = isMultiField(n.program, field);
      const label = field + (multi && wiredAll.length ? ` ×${wiredAll.length}` : "");
      const row = h("div", { class: "wf-port-row" + (horiz() ? " in" : " v") },
        h("span", { class: "wf-port-label", title: `${field} (${kind})${multi ? " — accepts multiple" : ""}` }, label));
      el.appendChild(row);
      el.appendChild(portEl("in" + (wiredAll.length ? " wired" : "") + (multi ? " multi" : ""), n.x, n.y, inPort(n, i, ins.length),
        { node: n.id, field, kind }));
    });
    // named label chips at the bottom of the box (colored to match each label)
    const labels = nodeLabels(n);
    if (labels.length) {
      el.appendChild(h("div", { class: "wf-node-labels-strip" }, ...labels.map((l) => labelChip(l))));
    }
    // single output port
    if (nodeProduces(n).length) {
      const op = portEl("out", n.x, n.y, outPort(n), { node: n.id, out: "1" });
      op.addEventListener("pointerdown", (e) => startWire(e, n));
      el.appendChild(op);
    }
    return el;
  }

  // `fs` is the focus state of this wire: "hot" (wired to the selection), "dim" (not),
  // or null (focus mode idle).
  function wirePath(p1, p2, edge, temp = false, fs = null) {
    let d;
    if (horiz()) {
      const dx = Math.max(40, Math.abs(p2.x - p1.x) * 0.5);
      d = `M ${p1.x} ${p1.y} C ${p1.x + dx} ${p1.y}, ${p2.x - dx} ${p2.y}, ${p2.x} ${p2.y}`;
    } else {
      const dy = Math.max(40, Math.abs(p2.y - p1.y) * 0.5);
      d = `M ${p1.x} ${p1.y} C ${p1.x} ${p1.y + dy}, ${p2.x} ${p2.y - dy}, ${p2.x} ${p2.y}`;
    }
    const st = edge && runState && runState[edge.from] ? runState[edge.from].status : null;
    const color = temp ? "var(--accent)" : statusColor(st);
    const path = svgEl("path", {
      d, fill: "none", "stroke-width": temp ? 2 : (fs === "hot" ? 4 : 2.5),
      stroke: color,
      "stroke-dasharray": temp ? "5 4" : "",
      // A focused wire keeps its status color and thickens; the rest fade back far
      // enough to read as background without disappearing.
      "stroke-opacity": fs === "dim" ? 0.15 : 1,
    });
    if (fs === "hot") {
      // the glow is drawn in the wire's own status color: `currentColor` in the
      // stylesheet resolves against this, which the stroke attribute can't provide
      path.setAttribute("class", "wf-wire-hot");
      path.style.color = color;
    }
    if (edge && !temp) {
      path.style.cursor = "pointer"; path.style.pointerEvents = "stroke";
      path.addEventListener("click", () => { if (confirm("Remove this connection?")) removeEdge(edge.id); });
    }
    return path;
  }

  function redrawWires(tempTo, focus = focusSets()) {
    clear(svg);
    // With focus on, draw the dimmed wires first so the focused ones sit on top of them
    // (SVG has no z-index — paint order is document order).
    const edges = focus
      ? [...graph.edges].sort((a, b) => (focus.edges.has(a.id) ? 1 : 0) - (focus.edges.has(b.id) ? 1 : 0))
      : graph.edges;
    for (const e of edges) {
      const s = nodeById(e.from), d = nodeById(e.to);
      if (!s || !d) continue;
      const ins = inputsOf(d.program);
      const idx = Math.max(0, ins.findIndex(([f]) => f === e.field));
      svg.appendChild(wirePath(outPort(s), inPort(d, idx, ins.length), e, false,
        focus ? (focus.edges.has(e.id) ? "hot" : "dim") : null));
    }
    if (drag && drag.kind === "wire" && tempTo) svg.appendChild(wirePath(outPort(nodeById(drag.from)), tempTo, null, true));
  }

  // ── gestures (node drag + wiring) ──
  function canvasPoint(e) {
    const r = canvas.getBoundingClientRect();
    return { x: (e.clientX - r.left + canvas.scrollLeft) / zoom, y: (e.clientY - r.top + canvas.scrollTop) / zoom };
  }
  function elementNodeAt(clientX, clientY) {
    const tgt = document.elementFromPoint(clientX, clientY);
    const el = tgt && tgt.closest && tgt.closest(".wf-node");
    return el ? el.dataset.node : null;
  }
  function startNodeDrag(e, n) {
    if (e.target.closest(".wf-node-x") || e.target.closest(".wf-node-run")) return;
    e.preventDefault();
    if (e.shiftKey || e.ctrlKey || e.metaKey) toggleSel(n.id);
    else if (!selectedSet.has(n.id)) selectOnly(n.id);
    else selected = n.id;
    renderNodes(); renderInspector();
    const p = canvasPoint(e);
    // move every selected node together
    const group = (selectedSet.has(n.id) ? [...selectedSet] : [n.id]).map((id) => nodeById(id)).filter(Boolean);
    drag = { kind: "node", id: n.id, ox: p.x, oy: p.y, start: group.map((g) => ({ id: g.id, x: g.x, y: g.y })) };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  }
  function startWire(e, n) {
    e.preventDefault(); e.stopPropagation();
    drag = { kind: "wire", from: n.id };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  }
  // ── rubber-band (marquee) selection ──
  function marqueeRect() {
    const d = drag;
    return { x0: Math.min(d.x0, d.x1), y0: Math.min(d.y0, d.y1), x1: Math.max(d.x0, d.x1), y1: Math.max(d.y0, d.y1) };
  }
  function updateMarquee() {
    const r = marqueeRect();
    marqueeEl.style.display = "block";
    marqueeEl.style.left = r.x0 + "px"; marqueeEl.style.top = r.y0 + "px";
    marqueeEl.style.width = (r.x1 - r.x0) + "px"; marqueeEl.style.height = (r.y1 - r.y0) + "px";
  }
  function applyMarquee() {
    const r = marqueeRect();
    selectedSet.clear();
    for (const id of drag.base) selectedSet.add(id);
    for (const n of graph.nodes) {                    // any node overlapping the box
      if (n.x < r.x1 && n.x + NODE_W > r.x0 && n.y < r.y1 && n.y + nodeHeight(n) > r.y0) selectedSet.add(n.id);
    }
    selected = selectedSet.has(selected) ? selected : (selectedSet.size ? [...selectedSet][0] : null);
    // Live class toggle (cheaper than a full renderNodes on every move).
    for (const el of nodesLayer.querySelectorAll(".wf-node")) el.classList.toggle("sel", selectedSet.has(el.dataset.node));
  }

  function onMove(e) {
    if (!drag) return;
    if (drag.kind === "marquee") {
      const p = canvasPoint(e);
      drag.x1 = p.x; drag.y1 = p.y;
      updateMarquee(); applyMarquee();
      return;
    }
    if (drag.kind === "pan") {
      canvas.scrollLeft = drag.sl - (e.clientX - drag.sx);
      canvas.scrollTop = drag.st - (e.clientY - drag.sy);
      if (Math.abs(e.clientX - drag.sx) + Math.abs(e.clientY - drag.sy) > 3) drag.moved = true;
      drawMinimap();
      return;
    }
    const p = canvasPoint(e);
    if (drag.kind === "node") {
      const ddx = p.x - drag.ox, ddy = p.y - drag.oy;
      for (const s of drag.start) {
        const node = nodeById(s.id); if (!node) continue;
        node.x = Math.max(0, s.x + ddx); node.y = Math.max(0, s.y + ddy);
        const el = nodesLayer.querySelector(`.wf-node[data-node="${node.id}"]`);
        if (el) { el.style.left = node.x + "px"; el.style.top = node.y + "px"; }
      }
      // Ports are computed from node data (outPort/inPort), so only the wires
      // and the minimap need repainting — the old full renderNodes() rebuilt
      // every node element at pointer-event rate, which made dragging a
      // 15-node graph visibly lag. Throttled to one repaint per frame.
      scheduleDragRedraw();
    } else {
      redrawWires(p);
    }
  }
  let dragRaf = 0;
  function scheduleDragRedraw() {
    if (dragRaf) return;
    dragRaf = requestAnimationFrame(() => { dragRaf = 0; redrawWires(); drawMinimap(); });
  }
  function onUp(e) {
    window.removeEventListener("pointermove", onMove);
    window.removeEventListener("pointerup", onUp);
    if (drag && drag.kind === "marquee") {
      applyMarquee();
      marqueeEl.style.display = "none";
      canvas.classList.remove("marquee-select");
      renderNodes(); renderInspector();   // finalize .primary highlight + inspector
      drag = null; return;
    }
    if (drag && drag.kind === "pan") {
      canvas.classList.remove("panning");
      if (!drag.moved) { selected = null; selectedSet.clear(); renderNodes(); renderInspector(); }  // a plain click clears the selection
      drag = null; return;
    }
    if (drag && drag.kind === "wire") {
      const tgt = document.elementFromPoint(e.clientX, e.clientY);
      const srcProduces = nodeProduces(nodeById(drag.from));
      const exact = tgt && tgt.closest && tgt.closest(".wf-port.in");
      if (exact && exact.dataset.node && exact.dataset.node !== drag.from) {
        // dropped right on an input dot
        if (srcProduces.includes(exact.dataset.kind)) addEdge(drag.from, exact.dataset.node, exact.dataset.field, exact.dataset.kind);
        else flash(`That step can't provide "${exact.dataset.kind}".`);
      } else {
        // forgiving: dropped anywhere on a node -> wire to its first compatible
        // input (the 11px port dots are otherwise easy to miss)
        const nodeEl = tgt && tgt.closest && tgt.closest(".wf-node");
        const toId = nodeEl && nodeEl.dataset.node;
        if (toId && toId !== drag.from) {
          const hit = inputsOf(nodeById(toId).program).find(([, k]) => srcProduces.includes(k));
          if (hit) addEdge(drag.from, toId, hit[0], hit[1]);
          else flash("That step has no input this output can feed.");
        }
      }
      redrawWires();
    } else if (drag && drag.kind === "node") {
      dirty = true;
    }
    drag = null;
  }

  // Persist a step rename: save the graph, and let the server carry the new name onto
  // the job that step has already run (it holds the authoritative node->job link, which
  // the canvas cannot: during a run the client keeps its own copy of the graph, so a
  // node's job_id here may be missing, or left over from an earlier run). A step that
  // has never run has no job yet — its name reaches one when it launches.
  async function commitNodeName(n, value) {
    if (!wf) return;                       // unsaved canvas: goes with the first save
    await save();
    const saved = ((wf.graph && wf.graph.nodes) || []).find((x) => x.id === n.id);
    const jid = (saved && saved.job_id) || (runState && runState[n.id] && runState[n.id].job_id);
    if (jid) store.renameViewerForJob(jid, value.trim());
  }

  // ── inspector (per-node args) ──
  function renderInspector() {
    clear(inspBody);
    const n = selected && nodeById(selected);
    if (!n) {
      inspBody.appendChild(h("div", { class: "wf-insp-empty muted" },
        "Select a step to edit its parameters. Wire a step's output port into another step's input port to pass data. Shift-click to multi-select. Double-click a step that has run to open its job log."));
      return;
    }
    inspBody.appendChild(h("div", { class: "wf-insp-head" }, h("strong", null, n.program)));
    const nameI = h("input", { class: "wf-field", placeholder: "step name (optional)", value: n.name || "" });
    nameI.addEventListener("input", () => { n.name = nameI.value; dirty = true; const t = nodesLayer.querySelector(`.wf-node[data-node="${n.id}"] .wf-node-title`); if (t) t.textContent = n.name || n.program; });
    // A step that has already run owns a job, and that job holds the name the Jobs
    // view shows — and the one a project canvas reads back on every load. Renaming
    // only the node would be undone by that re-sync unless the step were run again,
    // so commit the new name to the job as well (same as the label chips below).
    nameI.addEventListener("change", () => commitNodeName(n, nameI.value));
    const gpuI = h("input", { class: "wf-field", placeholder: "GPU (e.g. 0, empty = CPU)", value: n.gpu ?? "0" });
    gpuI.addEventListener("input", () => { n.gpu = gpuI.value; dirty = true; });
    inspBody.appendChild(h("label", { class: "wf-lbl" }, "Name", nameI));
    inspBody.appendChild(h("label", { class: "wf-lbl" }, "GPU", gpuI));
    inspBody.appendChild(h("label", { class: "wf-lbl" }, "Labels",
      nodeLabelChips(n, {
        // Persist to the linked job whether the link is on the node (project
        // canvas) or only in the run mapping (a standalone workflow that has run).
        jobId: n.job_id || (runState && runState[n.id] && runState[n.id].job_id),
        onChange: () => { dirty = true; const el = nodesLayer.querySelector(`.wf-node[data-node="${n.id}"]`); if (el) renderNodes(); },
      })));

    const schema = schemaCache[n.program];
    if (!schema) { loadSchema(n.program).then(() => { if (selected === n.id) renderInspector(); }); inspBody.appendChild(h("div", { class: "muted small" }, "Loading parameters…")); return; }
    const bindable = byName[n.program]?.bindable || {};
    // Same grouped/searchable form as the launcher, at compact density. A node
    // stores only the arguments that differ from the program's defaults, so a
    // default changed in hax-em is picked up on the next run — hence the sparse
    // `set` that deletes rather than storing an empty value.
    inspBody.appendChild(paramForm(schema, visibleFields(schema), {
      density: "compact",
      get: (name) => n.args[name],
      set: (name, v, f) => {
        const numeric = f.type === "integer" || f.type === "number";
        if (v === "" || v == null) delete n.args[name];
        else n.args[name] = numeric && v !== "" ? Number(v) : v;
        dirty = true;
      },
      // A wired input is fed by the graph, so it shows its source(s) instead of
      // an editable control.
      override: (f) => {
        const wired = graph.edges.filter((e) => e.to === n.id && e.field === f.name);
        if (!wired.length) return null;
        return h("div", { class: "wf-wired" }, ...wired.map((w) => {
          const src = nodeById(w.from);
          return h("div", { class: "wf-wired-chip" }, icon("link", 12),
            h("span", { style: { flex: "1" } }, `from ${src ? (src.name || src.program) : "?"} · ${w.kind}`),
            h("button", { class: "wf-chip-x", title: "Remove connection", onclick: () => removeEdge(w.id) }, "×"));
        }), isMultiField(n.program, f.name)
          ? h("p", { class: "field-help" }, "Wire more sources to add to this input.")
          : null);
      },
      tag: (f) => {
        if (!(f.name in bindable)) return null;
        return isMultiField(n.program, f.name) ? "wirable, multi" : "wirable";
      },
    }).el);
  }

  function loadSchema(program) {
    if (schemaCache[program]) return Promise.resolve(schemaCache[program]);
    return api.program(program).then((s) => (schemaCache[program] = s)).catch(() => null);
  }

  // ── layout ──
  // The orientation belongs to the graph, so it only outlives this page visit if it is
  // written back — leaving the view and returning reloads the workflow from the server.
  // Save it right away rather than waiting for an explicit Save: the toggle re-lays out
  // every node anyway, and an unsaved canvas loses its edits on navigation regardless,
  // so saving here only ever preserves work. The choice is also remembered locally, to
  // seed canvases that have no orientation of their own (including unsaved ones, which
  // have no record to save into).
  async function toggleOrientation() {
    graph.orientation = horiz() ? "vertical" : "horizontal";
    rememberOrientation(graph.orientation);
    autoLayout(); dirty = true;
    if (wf) await save();
  }

  // Arrange nodes into dependency tiers, one independent subworkflow at a time, with
  // every step centred between the ones it feeds off. The algorithm lives in
  // wfLayout.js and is orientation-agnostic: it works in (main, cross), and all this
  // has to say is which screen axis each of those is and how big a node is across.
  function autoLayout() {
    const PAD = 40, GAP_MAIN = 100, GAP_CROSS = 26, GAP_COMP = 72;
    const H = horiz();
    const laid = layeredLayout(graph.nodes, graph.edges, {
      // Horizontal spreads nodes down the screen, where their heights differ; vertical
      // spreads them across, where every node is one width.
      crossSize: (n) => (H ? nodeHeight(n) : NODE_W),
      mainStep: H ? NODE_W + GAP_MAIN : HEADER_H + 3 * ROW_H + 70,
      pad: PAD, gapCross: GAP_CROSS, gapComp: GAP_COMP,
    });
    if (!laid) { flash("Can't auto-layout: the graph has a cycle.", "err"); return; }
    for (const n of graph.nodes) {
      // `cross` is a centre; nodes are positioned by their top-left corner.
      if (H) { n.x = laid.main[n.id]; n.y = laid.cross[n.id] - nodeHeight(n) / 2; }
      else { n.y = laid.main[n.id]; n.x = laid.cross[n.id] - NODE_W / 2; }
    }
    // shift the laid-out graph to the middle of the scrollable content so there's
    // room to pan on every side, then center the viewport on it
    const b = graphBBox();
    if (b) {
      const offx = Math.round(CONTENT_W / 2 - b.cx), offy = Math.round(CONTENT_H / 2 - b.cy);
      for (const n of graph.nodes) { n.x = Math.max(0, n.x + offx); n.y = Math.max(0, n.y + offy); }
    }
    renderNodes();
    centerView();
  }
  // ── view: center, pan, minimap ──
  function graphBBox() {
    if (!graph.nodes.length) return null;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const n of graph.nodes) {
      x0 = Math.min(x0, n.x); y0 = Math.min(y0, n.y);
      x1 = Math.max(x1, n.x + NODE_W); y1 = Math.max(y1, n.y + nodeHeight(n));
    }
    return { x0, y0, x1, y1, cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, w: x1 - x0, h: y1 - y0 };
  }
  // ── viewport memory (pan + zoom, across visits to this page) ──
  // Recorded continuously while the canvas is scrolled/zoomed, and once more on the way
  // out, so returning to the page resumes the exact view instead of re-centring.
  function rememberViewport() {
    if (!canvas.clientWidth) return;              // not laid out yet — nothing to record
    VIEWPORTS.set(wf ? wf.id : "", { sl: canvas.scrollLeft, st: canvas.scrollTop, zoom });
  }
  // Restore the view remembered for `id`; false if there is none (caller centres instead).
  // Zoom is applied first: it resizes the scroll content, which is what the scroll
  // offsets are clamped against.
  function restoreViewport(id) {
    const v = VIEWPORTS.get(id || "");
    if (!v || !canvas.clientWidth) return false;
    zoom = v.zoom; applyZoom();
    canvas.scrollLeft = v.sl; canvas.scrollTop = v.st;
    drawMinimap();
    return true;
  }

  // Scroll so the graph's bounding box sits in the middle of the viewport (the
  // content is scaled by `zoom`, so content coords map to scroll via *zoom).
  function centerView() {
    const b = graphBBox(); if (!b) return;
    canvas.scrollLeft = Math.max(0, Math.min(CONTENT_W * zoom - canvas.clientWidth, b.cx * zoom - canvas.clientWidth / 2));
    canvas.scrollTop = Math.max(0, Math.min(CONTENT_H * zoom - canvas.clientHeight, b.cy * zoom - canvas.clientHeight / 2));
    drawMinimap();
  }
  function toggleMinimap() { minimapOn = !minimapOn; minimap.classList.toggle("hidden", !minimapOn); drawMinimap(); }
  function setMinimapCollapsed(v) {
    mmCollapsed = v;
    minimap.classList.toggle("collapsed", v);
    mmCollapse.title = v ? "Show overview" : "Collapse overview";
    drawMinimap();
  }
  // Centre the viewport on a content-space point, clamped to the scrollable extent.
  function scrollToContent(cx, cy) {
    canvas.scrollLeft = Math.max(0, Math.min(CONTENT_W * zoom - canvas.clientWidth, cx * zoom - canvas.clientWidth / 2));
    canvas.scrollTop = Math.max(0, Math.min(CONTENT_H * zoom - canvas.clientHeight, cy * zoom - canvas.clientHeight / 2));
    drawMinimap();
  }

  // Canvas 2D can't resolve `var(--x)`, but it can be handed the *computed* value —
  // so the map follows the light/dark theme instead of hardcoding one palette.
  function themeInk() {
    const cs = getComputedStyle(minimap);
    const v = (name, fallback) => (cs.getPropertyValue(name).trim() || fallback);
    return {
      line: v("--border", "#1e293b"),
      muted: v("--muted", "#94a3b8"),
      text: v("--text", "#f1f5f9"),
      accent: v("--accent", "#2dd4bf"),
      // Derived from the ground's luminance rather than matched against a known hex,
      // so re-tuning the palette can't silently flip the scrim to the wrong one.
      dark: luminance(v("--bg", "#020617")) < 0.5,
    };
  }
  // Rough perceptual lightness of a #rgb / #rrggbb token; 0.5 on anything unparseable
  // is harmless — the scrim only picks between two tints.
  function luminance(css) {
    const m = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(css);
    if (!m) return 0.5;
    const hex = m[1].length === 3 ? m[1].replace(/./g, (c) => c + c) : m[1];
    const n = parseInt(hex, 16);
    return (0.299 * ((n >> 16) & 255) + 0.587 * ((n >> 8) & 255) + 0.114 * (n & 255)) / 255;
  }

  // Recompute the summary in the panel header. Cheap enough to call on every draw.
  function updateMinimapCount() {
    const n = graph.nodes.length;
    const live = runState ? graph.nodes.filter((x) => runState[x.id] && runState[x.id].status === "running").length : 0;
    mmCount.textContent = live ? `${n} · ${live} running` : `${n} step${n === 1 ? "" : "s"}`;
    mmCount.classList.toggle("live", live > 0);
  }

  function drawMinimap() {
    schedulePulse();          // before the early-outs, so hiding the panel stops the loop
    if (!minimapOn) return;
    updateMinimapCount();
    if (mmCollapsed) { mmXform = null; mmView = null; return; }
    const W = mmCanvas.clientWidth, H = mmCanvas.clientHeight;
    if (!W || !H) return;
    const dpr = window.devicePixelRatio || 1;
    // Only resize the backing store when it actually changed: assigning width/height
    // clears the canvas and reallocates, and this runs on every scroll frame.
    const bw = Math.round(W * dpr), bh = Math.round(H * dpr);
    if (mmCanvas.width !== bw || mmCanvas.height !== bh) { mmCanvas.width = bw; mmCanvas.height = bh; }
    const ctx = mmCanvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const ink = themeInk();
    const b = graphBBox();
    if (!b) {
      mmXform = null; mmView = null;
      ctx.fillStyle = ink.muted; ctx.globalAlpha = 0.65;
      ctx.font = "11px system-ui, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText("No steps yet", W / 2, H / 2);
      ctx.globalAlpha = 1;
      return;
    }
    // Fit the graph with a margin proportional to the graph itself, so a two-node
    // workflow isn't blown up to fill the panel and a large one still breathes.
    const pad = Math.max(40, Math.max(b.w, b.h) * 0.06);
    const gx0 = b.x0 - pad, gy0 = b.y0 - pad, gw = b.w + 2 * pad, gh = b.h + 2 * pad;
    const s = Math.min(W / gw, H / gh);
    const ox = (W - gw * s) / 2, oy = (H - gh * s) / 2;
    mmXform = { gx0, gy0, ox, oy, s };
    const tx = (x) => ox + (x - gx0) * s, ty = (y) => oy + (y - gy0) * s;

    // Node footprints in map px, kept to a legible minimum. Real proportions (not
    // squares) are what makes the map recognisable as the graph you just left.
    const MIN_W = 9, MIN_H = 5;
    const boxes = new Map();
    for (const n of graph.nodes) {
      const w = Math.max(MIN_W, NODE_W * s), hh = Math.max(MIN_H, nodeHeight(n) * s);
      boxes.set(n.id, { x: tx(n.x + NODE_W / 2) - w / 2, y: ty(n.y + nodeHeight(n) / 2) - hh / 2, w, h: hh });
    }

    // Wires — the same bezier shape as the canvas, so the map reads as the same
    // picture. Anchored on the box edges the ports live on, not the centres.
    ctx.lineCap = "round";
    for (const e of graph.edges) {
      const a = boxes.get(e.from), c = boxes.get(e.to);
      if (!a || !c) continue;
      const p1 = horiz() ? { x: a.x + a.w, y: a.y + a.h / 2 } : { x: a.x + a.w / 2, y: a.y + a.h };
      const p2 = horiz() ? { x: c.x, y: c.y + c.h / 2 } : { x: c.x + c.w / 2, y: c.y };
      const st = runState && runState[e.from] ? runState[e.from].status : null;
      ctx.strokeStyle = (st && MINI_STATUS[st]) || ink.muted;
      ctx.globalAlpha = st ? 0.75 : 0.4;
      ctx.lineWidth = 1.25;
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      if (horiz()) {
        const dx = Math.max(6, Math.abs(p2.x - p1.x) * 0.5);
        ctx.bezierCurveTo(p1.x + dx, p1.y, p2.x - dx, p2.y, p2.x, p2.y);
      } else {
        const dy = Math.max(6, Math.abs(p2.y - p1.y) * 0.5);
        ctx.bezierCurveTo(p1.x, p1.y + dy, p2.x, p2.y - dy, p2.x, p2.y);
      }
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // Steps — a translucent status wash inside a solid status edge. At this size a
    // flat fill loses the outline against a same-hue wire, and a pure outline is too
    // faint to scan; the pair keeps both the colour and the shape readable.
    const phase = mmPulse !== null ? (performance.now() % 1600) / 1600 : 0;
    for (const n of graph.nodes) {
      const bx = boxes.get(n.id);
      const st = runState && runState[n.id] ? runState[n.id].status : null;
      const col = (st && MINI_STATUS[st]) || ink.muted;
      const r = Math.min(3, bx.h / 2.5);
      // A running step breathes, which is the one thing worth animating here: it is
      // exactly what you glance at the map to check.
      if (st === "running") {
        const grow = 2.5 + 2.5 * Math.sin(phase * Math.PI * 2);   // 0..5, never inset
        ctx.globalAlpha = 0.28;
        ctx.fillStyle = col;
        roundRectPath(ctx, bx.x - grow, bx.y - grow, bx.w + 2 * grow, bx.h + 2 * grow, r + grow);
        ctx.fill();
        ctx.globalAlpha = 1;
      }
      roundRectPath(ctx, bx.x, bx.y, bx.w, bx.h, r);
      ctx.globalAlpha = st ? 0.55 : 0.3;
      ctx.fillStyle = col; ctx.fill();
      ctx.globalAlpha = 1;
      ctx.lineWidth = 1.25; ctx.strokeStyle = col; ctx.stroke();
      // Selection reads as a ring in the text colour — never a status hue, so it
      // can't be mistaken for a state.
      if (selectedSet.has(n.id)) {
        roundRectPath(ctx, bx.x - 1.75, bx.y - 1.75, bx.w + 3.5, bx.h + 3.5, r + 1.75);
        ctx.lineWidth = 1.25; ctx.strokeStyle = ink.text; ctx.globalAlpha = 0.85; ctx.stroke();
        ctx.globalAlpha = 1;
      }
    }

    // Viewport — everything outside it is scrimmed, so the visible region reads as a
    // lit window rather than one more rectangle among the wires.
    const vx = tx(canvas.scrollLeft / zoom), vy = ty(canvas.scrollTop / zoom);
    const vw = (canvas.clientWidth / zoom) * s, vh = (canvas.clientHeight / zoom) * s;
    mmView = { x: vx, y: vy, w: vw, h: vh };
    ctx.fillStyle = ink.dark ? "rgba(2,6,23,0.55)" : "rgba(15,23,42,0.18)";
    ctx.beginPath();
    ctx.rect(0, 0, W, H);
    ctx.rect(vx, vy, vw, vh);
    ctx.fill("evenodd");
    ctx.strokeStyle = ink.accent; ctx.lineWidth = 1.5;
    roundRectPath(ctx, vx + 0.75, vy + 0.75, Math.max(2, vw - 1.5), Math.max(2, vh - 1.5), 3);
    ctx.stroke();
  }

  // Drive the running-step pulse only while something is running — a permanent rAF
  // loop behind an idle panel is pure battery burn.
  function schedulePulse() {
    // root.isConnected stops the loop when the router swaps the view out: this view
    // has no teardown hook, so the check has to live here.
    const live = minimapOn && !mmCollapsed && runState && root.isConnected
      && graph.nodes.some((n) => runState[n.id] && runState[n.id].status === "running");
    if (live && mmPulse === null) {
      // The pulse is a slow 1.6 s glow: ~10 repaints/s renders it just as
      // smoothly as the old every-frame loop, at a sixth of the work — this
      // used to repaint the full minimap at 60 fps for a multi-hour run.
      let last = 0;
      const tick = (t) => {
        mmPulse = requestAnimationFrame(tick);
        if (t - last >= 100) { last = t; drawMinimap(); }
      };
      mmPulse = requestAnimationFrame(tick);
    } else if (!live && mmPulse !== null) {
      cancelAnimationFrame(mmPulse); mmPulse = null;
    }
  }

  // ── open a project as a live workflow ──
  async function openProject() {
    const pid = getState().activeProjectId;
    if (!pid) { flash("Open a project first (Projects) to open it as a workflow.", "err"); return; }
    flash("Opening project…", "busy");
    try {
      const opened = await api.workflowFromProject(pid);
      await refreshList();
      await openWorkflow(opened.id);
      flash(`Opened "${wf ? wf.name : "project"}" as a live workflow — edits & runs change the project.`, "muted");
    } catch (err) { flash("Open failed: " + (err.detail || err.message), "err"); }
  }

  // Create a new project backed by this workflow (so edits/runs populate it).
  async function saveAsProject() {
    const name = await promptModal({ title: "Save as new project", label: "Project name", value: nameInput.value.trim(), placeholder: "my-project", okText: "Create project" });
    if (!name) return;
    await save();             // ensure the workflow record exists first
    if (!wf) return;
    try {
      wf = await api.workflowSaveAsProject(wf.id, name);
      nameInput.value = wf.name || name;
      await refreshList();
      flash(`Saved as project "${name}" — this workflow now drives it.`, "muted");
    } catch (err) { flash("Couldn't create project: " + (err.detail || err.message), "err"); }
  }

  // ── pinned layout (save / restore node positions) ──
  // Independent of "Tidy": pins the user's own arrangement so they can return to it.
  async function saveLayout() {
    if (!graph.nodes.length) { flash("Add some steps first.", "warn"); return; }
    graph.savedLayout = {
      orientation: graph.orientation,
      pos: Object.fromEntries(graph.nodes.map((n) => [n.id, { x: n.x, y: n.y, job: n.job_id || null }])),
    };
    if (wf) await save();      // persist the pin
    flash(`Pinned the current layout (${graph.nodes.length} positions).`, "muted");
  }
  function restoreLayout() {
    const sl = graph.savedLayout;
    if (!sl || !sl.pos) { flash("No pinned layout yet — pin one first.", "warn"); return; }
    if (sl.orientation && sl.orientation !== graph.orientation) { graph.orientation = sl.orientation; }
    const known = [], unknown = [];
    for (const n of graph.nodes) {
      // match by node id, falling back to job id (project re-syncs can renumber)
      const p = sl.pos[n.id] || (n.job_id ? Object.values(sl.pos).find((q) => q.job && q.job === n.job_id) : null);
      if (p) { n.x = p.x; n.y = p.y; known.push(n); } else unknown.push(n);
    }
    // Nodes added since the pin aren't in it — stack them just to the right of the
    // restored arrangement so they're visible and don't overlap it.
    if (unknown.length) {
      const refs = known.length ? known : graph.nodes.filter((n) => !unknown.includes(n));
      const x1 = Math.max(0, ...(refs.length ? refs : graph.nodes).map((n) => n.x + NODE_W));
      let y = refs.length ? Math.min(...refs.map((n) => n.y)) : 40;
      for (const n of unknown) { n.x = x1 + 80; n.y = y; y += nodeHeight(n) + 26; }
    }
    dirty = true; renderNodes(); centerView();
    flash(unknown.length ? `Restored the pinned layout; placed ${unknown.length} new node(s) alongside it.` : "Restored the pinned layout.", "muted");
  }

  // ── persistence + run ──
  function graphSnapshot() { return { nodes: graph.nodes, edges: graph.edges, orientation: graph.orientation, savedLayout: graph.savedLayout || null }; }
  function flash(msg, kind = "warn") { statusEl.className = "wf-status " + kind; statusEl.textContent = msg; }

  async function save() {
    const pid = getState().activeProjectId;
    const name = nameInput.value.trim() || "Untitled workflow";
    try {
      if (wf) wf = await api.updateWorkflow(wf.id, { name, graph: graphSnapshot() });
      else wf = await api.createWorkflow({ project_id: pid, name, graph: graphSnapshot() });
      dirty = false; rememberWorkflow(wf.id); flash("Saved.", "muted"); await refreshList(); wfSelect.value = wf.id; syncActions();
    } catch (err) { flash("Save failed: " + (err.detail || err.message), "err"); }
    return wf;
  }

  // ids === null -> run the whole workflow; an array -> run that subset only.
  async function runNodes(ids) {
    if (!(wf && wf.project_id) && !getState().activeProjectId) { flash("Open a project first (Projects), or use “Save as project” — steps run into a project.", "err"); return; }
    if (!graph.nodes.length) { flash("Add at least one step.", "warn"); return; }
    if (ids && !ids.length) { flash("Select one or more steps first.", "warn"); return; }
    // Running a subset re-makes those steps under everything they feed, so ask what
    // becomes of the steps downstream. Running the *whole* graph needs no question:
    // every one of them is being re-run anyway.
    //
    // Steps that never ran are listed too, even though they have nothing to lose —
    // "queue" runs them as well, and a list that didn't mention them would understate
    // what the button does. But if *none* of the downstream steps has ever run there is
    // no warning to give, so the dialog is skipped and the run behaves as it always did.
    let cascade = null;
    if (ids) {
      const down = downstreamNodes(graph, ids).map((n) => nodeAsDependent(n, runState));
      // Steps in this selection that are still executing. Relaunching one means stopping
      // it, and the alternative — letting it run on — is precisely what makes an edited
      // parameter look like it was applied when the old run finishes on the old settings.
      const busy = ids.map((id) => nodeById(id)).filter(Boolean)
        .map((n) => nodeAsDependent(n, runState)).filter((d) => d.active);
      cascade = await confirmCascade(down.some((d) => d.status) ? down : [], {
        restarting: busy,
        title: ids.length === 1 ? "Re-run this step?" : `Re-run ${ids.length} steps?`,
        lead: ids.length === 1
          ? "Its run folder is rebuilt from scratch — the current outputs and log are replaced."
          : "Their run folders are rebuilt from scratch — the current outputs and logs are replaced.",
      });
      if (!cascade) return;
    }
    // Save first — the scheduler reads the graph from the server, so a step that only
    // exists on the canvas has to be persisted before it can be queued.
    const wasRunning = !!(wf && wf.status === "running");
    await save();
    if (!wf) return;
    try {
      wf = await api.runWorkflow(wf.id, ids || undefined, cascade);
      runState = wf.run_state || null;
      // The queued count is read back from the server: with cascade "queue" it also took
      // on the steps downstream, which is more than were asked for here.
      const queued = runState ? Object.values(runState).filter((s) => s.status === "pending" || s.status === "running").length : (ids || graph.nodes).length;
      flash(wasRunning
        ? `Queued onto the running workflow — ${queued} step(s) pending or running.`
        : (ids ? `Running ${queued} step(s)…` : "Running…"), "busy");
      renderNodes();
      startPolling();
    } catch (err) {
      const errs = err.detail && err.detail.errors ? err.detail.errors : [err.detail || err.message];
      flash("Cannot run: " + errs.join("; "), "err");
    }
  }

  // Halt the run: the backend cancels every step executing now and dequeues the rest.
  async function stopRun() {
    if (!wf) return;
    try {
      wf = await api.stopWorkflow(wf.id);
      runState = wf.run_state || null;
      stopPolling();
      renderNodes();
      flash("Stopped — the running steps were cancelled and the queue cleared.", "warn");
    } catch (err) { flash("Stop failed: " + (err.detail || err.message), "err"); }
  }

  // Adopt a server graph into local state (keeping the id counters ahead).
  function adoptGraph(g) {
    graph = { nodes: g.nodes || [], edges: g.edges || [], orientation: g.orientation || graph.orientation || rememberedOrientation(), savedLayout: g.savedLayout || null };
    nodeSeq = Math.max(1, ...graph.nodes.map((n) => +(n.id.replace(/\D/g, "")) || 0)) + 1;
    edgeSeq = Math.max(1, ...graph.edges.map((e) => +((e.id || "").replace(/\D/g, "")) || 0)) + 1;
  }

  function startPolling() {
    stopPolling();
    poll = poller(async () => {
      if (!wf) return stopPolling();
      try {
        const w = await api.workflow(wf.id);
        wf = w; runState = w.run_state || null;
        // when a run settles, adopt the server graph so live node<->job links and
        // statuses (project workflows write job ids back) are reflected.
        // NOT while `dirty`: adoptGraph replaces the node objects the open
        // inspector is writing into, so a run finishing mid-edit silently
        // discarded the user's keystrokes on the next Save. The user's graph
        // wins until they save or reopen.
        if (w.status !== "running" && w.graph && !dirty) { adoptGraph(w.graph); renderInspector(); }
        renderNodes();
        const counts = countStatuses();
        flash(`${w.status} — ${counts}`, w.status === "running" ? "busy" : (w.status === "succeeded" ? "muted" : "err"));
        if (w.status !== "running") stopPolling();
      } catch {
        // Transient fetch failure (server restart, blip): keep polling — the
        // old stopPolling() here froze node statuses on the first hiccup with
        // no retry and no indication.
      }
    }, 2000, { immediate: false });
  }
  function stopPolling() { if (poll) { poll.stop(); poll = null; } }
  function countStatuses() {
    if (!runState) return "";
    const c = {};
    for (const s of Object.values(runState)) c[s.status] = (c[s.status] || 0) + 1;
    return Object.entries(c).map(([k, v]) => `${v} ${k}`).join(", ");
  }

  async function openWorkflow(id) {
    stopPolling();
    // Park the outgoing workflow's view before `wf` changes. Skipped for an empty
    // canvas, so the first open of a freshly built view doesn't record its 0,0 start.
    if (graph.nodes.length) rememberViewport();
    selectedSet.clear();
    if (!id) {
      wf = null; graph = { nodes: [], edges: [], orientation: rememberedOrientation() }; runState = null; selected = null; nameInput.value = "";
      wfSelect.value = ""; rememberWorkflow(null);
      renderNodes(); restoreViewport("");
      renderInspector(); syncActions(); flash("New workflow.", "muted"); return;
    }
    try {
      wf = await api.workflow(id);
      adoptGraph(wf.graph || {});
      runState = wf.run_state || null; selected = null; nameInput.value = wf.name || ""; wfSelect.value = wf.id;
      rememberWorkflow(wf.id);
      // freshly imported / never-laid-out graphs have all-zero coords — tidy once
      if (graph.nodes.length && graph.nodes.every((n) => !n.x && !n.y)) autoLayout();
      else { renderNodes(); if (!restoreViewport(wf.id)) centerView(); }
      renderInspector();
      syncActions();          // the run controls and the per-device cap follow the workflow
      flash(wf.status && wf.status !== "idle" ? `${wf.status} — ${countStatuses()}` : "Loaded.", "muted");
      if (wf.status === "running") startPolling();
    } catch (err) {
      rememberWorkflow(null);   // stale/removed target — don't keep trying to reopen it
      flash("Load failed: " + (err.detail || err.message), "err");
    }
  }

  function rememberWorkflow(id) {
    try { if (id) localStorage.setItem(LAST_WF_KEY, id); else localStorage.removeItem(LAST_WF_KEY); }
    catch { /* storage unavailable — non-fatal */ }
  }

  async function removeWorkflow() {
    if (!wf) return openWorkflow(null);
    if (!confirm(`Delete workflow "${wf.name}"?`)) return;
    try { const id = wf.id; await api.deleteWorkflow(id); VIEWPORTS.delete(id); await refreshList(); openWorkflow(null); }
    catch (err) { flash("Delete failed: " + (err.detail || err.message), "err"); }
  }

  // Keyboard shortcuts (ignored while typing in a field): Ctrl/Cmd+D duplicates
  // the selection, Delete / Backspace removes it.
  function onKey(e) {
    const t = e.target;
    if (t && /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName)) return;
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "d") {
      e.preventDefault();
      duplicateNodes([...selectedSet]);
    } else if ((e.key === "Delete" || e.key === "Backspace") && selectedSet.size) {
      e.preventDefault();
      removeNodes([...selectedSet]);
    }
  }
  document.addEventListener("keydown", onKey);

  boot();
  // cleanup runs before the view is detached, so the canvas is still laid out here and
  // its final scroll/zoom can be recorded.
  return { el: root, cleanup: () => { rememberViewport(); stopPolling(); document.removeEventListener("keydown", onKey); } };
}
