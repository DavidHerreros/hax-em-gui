// Visual workflow builder: a node-graph canvas where program nodes are wired
// output->input (mapping to the job input-binding system) and run as a DAG.
// Hand-written vanilla JS; SVG for the wires, absolutely-positioned divs for nodes.
//
// Features: horizontal/vertical layout, auto-layout into dependency tiers,
// drag-and-drop add + drop-to-connect, per-node + selection runs with live
// per-node status, predict<->train parameter inheritance on wiring, file-browser
// pickers for path args, and "import a project as a workflow".
import { h, clear, mount } from "../h.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { store, getState } from "../store.js";
import { errorBox } from "./ui.js";
import { visibleFields } from "./formFields.js";
import { openFilePicker } from "./filePicker.js";
import { nodeLabelChips, loadLabels, labelsByIds } from "./labels.js";
import { promptModal } from "./promptModal.js";

const NODE_W = 196;     // node width (px)
const HEADER_H = 34;    // node header height
const ROW_H = 26;       // per-input-port row height
const CONTENT_W = 4000; // scrollable canvas content size (matches the wires SVG)
const CONTENT_H = 3000;
const SVG_NS = "http://www.w3.org/2000/svg";

// Concrete colors for the minimap (canvas 2D can't read CSS custom properties).
const MINI_STATUS = { saved: "#a78bfa", pending: "#94a3b8", running: "#38bdf8", succeeded: "#22c55e", failed: "#ef4444", blocked: "#f59e0b", cancelled: "#94a3b8", idle: "#94a3b8" };

// Programs that expose a train/predict mode — wiring their model output into
// another node of the same program inherits the form parameters (mode aside).
const MODED_PROGRAMS = new Set(["hetsiren", "zernike3deep", "reconsiren"]);

// Same grouping/order as the right-menu program launcher.
const CATEGORY_ORDER = ["Networks", "Post-processing", "Metrics", "Viewers", "Other"];

const STATUS_COLOR = {
  saved: "#a78bfa", pending: "var(--muted)", running: "var(--accent)",
  succeeded: "#22c55e", failed: "var(--err)", blocked: "var(--warn)",
  cancelled: "var(--muted)", interrupted: "var(--warn)", idle: "var(--muted)", skipped: "var(--muted)",
};

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
  let graph = { nodes: [], edges: [], orientation: "horizontal" };
  let wf = null;            // saved workflow record
  let selected = null;      // primary selected node id (drives the inspector)
  const selectedSet = new Set();   // multi-selection (run / move together)
  let runState = null;      // {nodeId: {job_id, status, error}}
  let dirty = false;
  let poll = null;
  const schemaCache = {};
  let drag = null;          // active node drag / wiring / pan gesture
  let nodeSeq = 1, edgeSeq = 1;
  let minimapOn = true;     // overview window visible?
  let mmXform = null;       // last minimap transform (for click-to-navigate)

  const horiz = () => graph.orientation !== "vertical";

  const root = h("div", { class: "view workflows-view" });

  // ── top bar ──
  const nameInput = h("input", { class: "wf-name", placeholder: "Workflow name", value: "" });
  nameInput.addEventListener("input", () => { dirty = true; });
  const wfSelect = h("select", { class: "wf-select", title: "Open a saved workflow" });
  wfSelect.addEventListener("change", () => openWorkflow(wfSelect.value || null));
  const statusEl = h("span", { class: "wf-status muted" }, "");
  const orientBtn = h("button", { class: "btn small", title: "Toggle horizontal / vertical layout", onclick: toggleOrientation });
  const layoutBtn = h("button", { class: "btn small", title: "Auto-arrange into dependency levels", onclick: () => { autoLayout(); dirty = true; } }, icon("sliders", 14), "Tidy");
  const centerBtn = h("button", { class: "btn small", title: "Center the workflow in view", onclick: centerView }, icon("orbit", 14), "Center");
  const mmBtn = h("button", { class: "btn small", title: "Toggle the overview minimap", onclick: toggleMinimap }, icon("grid", 14), "Map");
  const importBtn = h("button", { class: "btn small", title: "Open the active project as a live workflow (edits & runs change the project)", onclick: openProject }, icon("box", 14), "Open project");
  const asProjBtn = h("button", { class: "btn small", title: "Create a new project backed by this workflow", onclick: saveAsProject }, icon("plus", 14), "Save as project");
  const saveBtn = h("button", { class: "btn", onclick: save }, icon("save", 15), "Save");
  const runSelBtn = h("button", { class: "btn secondary", title: "Run the selected step(s) only", onclick: () => runNodes([...selectedSet]) }, icon("play", 15), "Run selected");
  const runBtn = h("button", { class: "btn", title: "Run the whole workflow", onclick: () => runNodes(null) }, icon("play", 15), "Run all");
  const delBtn = h("button", { class: "btn", title: "Delete this workflow", onclick: removeWorkflow }, icon("trash", 15));
  const topbar = h("div", { class: "wf-topbar" },
    h("strong", null, "Workflow"), wfSelect,
    h("button", { class: "btn small", onclick: () => openWorkflow(null) }, "+ New"),
    importBtn, asProjBtn, nameInput, orientBtn, layoutBtn, centerBtn, mmBtn,
    h("span", { style: { flex: "1" } }), statusEl, saveBtn, runSelBtn, runBtn, delBtn);

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
  const minimap = h("canvas", { class: "wf-minimap", title: "Overview — click to jump" });
  const canvasWrap = h("div", { class: "wf-canvas-wrap" }, canvas, minimap);

  function applyZoom() {
    sizer.style.width = CONTENT_W * zoom + "px";
    sizer.style.height = CONTENT_H * zoom + "px";
    contentLayer.style.transform = `scale(${zoom})`;
  }
  applyZoom();
  // Mouse wheel zooms (toward the cursor) instead of scrolling.
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const cx = (canvas.scrollLeft + mx) / zoom, cy = (canvas.scrollTop + my) / zoom;   // content point under cursor
    const next = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, zoom * Math.exp(-e.deltaY * 0.0015)));
    if (next === zoom) return;
    zoom = next; applyZoom();
    canvas.scrollLeft = cx * zoom - mx;   // keep that content point under the cursor
    canvas.scrollTop = cy * zoom - my;
    drawMinimap();
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

  // empty-canvas drag pans the view (and a click without movement clears selection)
  canvas.addEventListener("pointerdown", (e) => {
    if (e.target === canvas || e.target === nodesLayer || e.target === svg || e.target === sizer || e.target === contentLayer) {
      e.preventDefault();
      drag = { kind: "pan", sx: e.clientX, sy: e.clientY, sl: canvas.scrollLeft, st: canvas.scrollTop, moved: false };
      canvas.classList.add("panning");
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
    }
  });
  canvas.addEventListener("scroll", drawMinimap);
  minimap.addEventListener("pointerdown", (e) => {
    e.stopPropagation();
    if (!mmXform) return;
    const r = minimap.getBoundingClientRect();
    const cx = mmXform.gx0 + (e.clientX - r.left - mmXform.ox) / mmXform.s;
    const cy = mmXform.gy0 + (e.clientY - r.top - mmXform.oy) / mmXform.s;
    canvas.scrollLeft = Math.max(0, cx * zoom - canvas.clientWidth / 2);
    canvas.scrollTop = Math.max(0, cy * zoom - canvas.clientHeight / 2);
    drawMinimap();
  });

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
    renderPalette(); syncOrientBtn();
    mmBtn.classList.toggle("on", minimapOn);
    loadLabels().catch(() => {});
    await refreshList();
    openWorkflow(null);
  }

  async function refreshList() {
    const pid = getState().activeProjectId;
    let items = [];
    try { items = await api.workflows(pid); } catch { items = []; }
    clear(wfSelect);
    wfSelect.appendChild(h("option", { value: "" }, "— saved workflows —"));
    for (const w of items) wfSelect.appendChild(h("option", { value: w.id }, w.name || "Untitled"));
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
  function removeNode(id) {
    graph.nodes = graph.nodes.filter((n) => n.id !== id);
    graph.edges = graph.edges.filter((e) => e.from !== id && e.to !== id);
    selectedSet.delete(id);
    if (selected === id) selected = selectedSet.size ? [...selectedSet][0] : null;
    dirty = true; renderNodes(); renderInspector();
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
    if (!MODED_PROGRAMS.has(dst.program) || !String(kind || "").endsWith("_model")) return;
    const wiredFields = new Set(graph.edges.filter((e) => e.to === toId).map((e) => e.field));
    for (const [k, v] of Object.entries(src.args || {})) {
      if (k === "mode" || wiredFields.has(k)) continue;
      if (!(k in dst.args)) dst.args[k] = v;
    }
    if (!("mode" in dst.args)) dst.args.mode = "predict";
    if (!("mode" in src.args)) src.args.mode = "train";
  }

  // Auto-wire a freshly dropped node `b` to the node `a` it landed on: prefer
  // a -> b (a's output into b's input), else b -> a, on the first matching kind.
  function autoConnect(aId, bId) {
    const a = nodeById(aId), b = nodeById(bId);
    if (!a || !b) return;
    const aOut = byName[a.program]?.produces || [], bOut = byName[b.program]?.produces || [];
    const aIn = inputsOf(a.program), bIn = inputsOf(b.program);
    let hit = bIn.find(([, kind]) => aOut.includes(kind));
    if (hit) return addEdge(aId, bId, hit[0], hit[1]);
    hit = aIn.find(([, kind]) => bOut.includes(kind));
    if (hit) { addEdge(bId, aId, hit[0], hit[1]); return; }
    flash(`No compatible ports between ${a.program} and ${b.program}.`);
  }

  const nodeById = (id) => graph.nodes.find((n) => n.id === id);
  const inputsOf = (program) => Object.entries((byName[program] || {}).bindable || {});
  const nodeHeight = (n) => HEADER_H + Math.max(1, inputsOf(n.program).length) * ROW_H + 8;

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

  // ── rendering ──
  function renderNodes() {
    clear(nodesLayer);
    for (const n of graph.nodes) nodesLayer.appendChild(nodeEl(n));
    redrawWires();
    drawMinimap();
  }

  function portEl(cls, originX, originY, pos, data) {
    const el = h("span", { class: "wf-port abs " + cls, style: { left: pos.x - originX - 6 + "px", top: pos.y - originY - 6 + "px" } });
    Object.assign(el.dataset, data);
    return el;
  }

  function nodeEl(n) {
    const st = runState && runState[n.id] ? runState[n.id].status : null;
    const running = wf && wf.status === "running";
    const el = h("div", {
      class: "wf-node" + (selectedSet.has(n.id) ? " sel" : "") + (selected === n.id ? " primary" : ""),
      style: { left: n.x + "px", top: n.y + "px", width: NODE_W + "px", height: nodeHeight(n) + "px" },
    });
    el.dataset.node = n.id;
    if (st) el.style.borderColor = STATUS_COLOR[st] || "var(--border)";
    // header (drag handle + select + run + remove)
    const labelDots = labelsByIds(n.labels);
    const head = h("div", { class: "wf-node-head" },
      h("span", { class: "wf-node-title mono", title: n.program }, n.name || n.program),
      labelDots.length ? h("span", { class: "wf-node-labels" }, ...labelDots.map((l) => h("span", { class: "wf-node-ldot", style: { background: l.color }, title: l.name }))) : null,
      st ? h("span", { class: "wf-node-st", style: { color: STATUS_COLOR[st] } }, st) : null,
      h("button", { class: "wf-node-run", title: "Run this step", disabled: running, onclick: (e) => { e.stopPropagation(); runNodes([n.id]); } }, icon("play", 12)),
      h("button", { class: "wf-node-x", title: "Remove step", onclick: (e) => { e.stopPropagation(); removeNode(n.id); } }, "×"));
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
    // single output port
    if ((byName[n.program]?.produces || []).length) {
      const op = portEl("out", n.x, n.y, outPort(n), { node: n.id, out: "1" });
      op.addEventListener("pointerdown", (e) => startWire(e, n));
      el.appendChild(op);
    }
    return el;
  }

  function wirePath(p1, p2, edge, temp = false) {
    let d;
    if (horiz()) {
      const dx = Math.max(40, Math.abs(p2.x - p1.x) * 0.5);
      d = `M ${p1.x} ${p1.y} C ${p1.x + dx} ${p1.y}, ${p2.x - dx} ${p2.y}, ${p2.x} ${p2.y}`;
    } else {
      const dy = Math.max(40, Math.abs(p2.y - p1.y) * 0.5);
      d = `M ${p1.x} ${p1.y} C ${p1.x} ${p1.y + dy}, ${p2.x} ${p2.y - dy}, ${p2.x} ${p2.y}`;
    }
    const st = edge && runState && runState[edge.from] ? runState[edge.from].status : null;
    const path = svgEl("path", {
      d, fill: "none", "stroke-width": temp ? 2 : 2.5,
      stroke: temp ? "var(--accent)" : (st ? STATUS_COLOR[st] : "#64748b"),
      "stroke-dasharray": temp ? "5 4" : "",
    });
    if (edge && !temp) {
      path.style.cursor = "pointer"; path.style.pointerEvents = "stroke";
      path.addEventListener("click", () => { if (confirm("Remove this connection?")) removeEdge(edge.id); });
    }
    return path;
  }

  function redrawWires(tempTo) {
    clear(svg);
    for (const e of graph.edges) {
      const s = nodeById(e.from), d = nodeById(e.to);
      if (!s || !d) continue;
      const ins = inputsOf(d.program);
      const idx = Math.max(0, ins.findIndex(([f]) => f === e.field));
      svg.appendChild(wirePath(outPort(s), inPort(d, idx, ins.length), e));
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
  function onMove(e) {
    if (!drag) return;
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
      // ports move with the node; cheapest correct redraw is a full re-render
      renderNodes();
    } else {
      redrawWires(p);
    }
  }
  function onUp(e) {
    window.removeEventListener("pointermove", onMove);
    window.removeEventListener("pointerup", onUp);
    if (drag && drag.kind === "pan") {
      canvas.classList.remove("panning");
      if (!drag.moved) { selected = null; selectedSet.clear(); renderNodes(); renderInspector(); }  // a plain click clears the selection
      drag = null; return;
    }
    if (drag && drag.kind === "wire") {
      const tgt = document.elementFromPoint(e.clientX, e.clientY);
      const port = tgt && tgt.closest && tgt.closest(".wf-port.in");
      if (port && port.dataset.node && port.dataset.node !== drag.from) {
        const srcProduces = byName[nodeById(drag.from).program]?.produces || [];
        if (srcProduces.includes(port.dataset.kind)) addEdge(drag.from, port.dataset.node, port.dataset.field, port.dataset.kind);
        else flash(`That step can't provide "${port.dataset.kind}".`);
      }
      redrawWires();
    } else if (drag && drag.kind === "node") {
      dirty = true;
    }
    drag = null;
  }

  // ── inspector (per-node args) ──
  function renderInspector() {
    clear(inspBody);
    const n = selected && nodeById(selected);
    if (!n) {
      inspBody.appendChild(h("div", { class: "wf-insp-empty muted" },
        "Select a step to edit its parameters. Wire a step's output port into another step's input port to pass data. Shift-click to multi-select."));
      return;
    }
    inspBody.appendChild(h("div", { class: "wf-insp-head" }, h("strong", null, n.program)));
    const nameI = h("input", { class: "wf-field", placeholder: "step name (optional)", value: n.name || "" });
    nameI.addEventListener("input", () => { n.name = nameI.value; dirty = true; const t = nodesLayer.querySelector(`.wf-node[data-node="${n.id}"] .wf-node-title`); if (t) t.textContent = n.name || n.program; });
    const gpuI = h("input", { class: "wf-field", placeholder: "GPU (e.g. 0, empty = CPU)", value: n.gpu ?? "0" });
    gpuI.addEventListener("input", () => { n.gpu = gpuI.value; dirty = true; });
    inspBody.appendChild(h("label", { class: "wf-lbl" }, "Name", nameI));
    inspBody.appendChild(h("label", { class: "wf-lbl" }, "GPU", gpuI));
    inspBody.appendChild(h("label", { class: "wf-lbl" }, "Labels",
      nodeLabelChips(n, { onChange: () => { dirty = true; const el = nodesLayer.querySelector(`.wf-node[data-node="${n.id}"]`); if (el) renderNodes(); } })));

    const schema = schemaCache[n.program];
    if (!schema) { loadSchema(n.program).then(() => { if (selected === n.id) renderInspector(); }); inspBody.appendChild(h("div", { class: "muted small" }, "Loading parameters…")); return; }
    const bindable = byName[n.program]?.bindable || {};
    inspBody.appendChild(h("div", { class: "wf-insp-sub muted small" }, "Parameters"));
    for (const f of visibleFields(schema)) {
      const wiredAll = graph.edges.filter((e) => e.to === n.id && e.field === f.name);
      const multi = isMultiField(n.program, f.name);
      if (wiredAll.length) {
        const chips = wiredAll.map((w) => {
          const src = nodeById(w.from);
          return h("div", { class: "wf-wired-chip" }, icon("link", 12),
            h("span", { style: { flex: "1" } }, `from ${src ? (src.name || src.program) : "?"} · ${w.kind}`),
            h("button", { class: "wf-chip-x", title: "Remove connection", onclick: () => removeEdge(w.id) }, "×"));
        });
        inspBody.appendChild(h("label", { class: "wf-lbl" },
          f.name + (multi ? ` (${wiredAll.length})` : "") + (multi ? "  — wire more sources to add" : ""), ...chips));
        continue;
      }
      inspBody.appendChild(h("label", { class: "wf-lbl", title: f.help || "" },
        f.name + (f.required ? " *" : "") + (f.name in bindable ? (multi ? "  (wirable, multi)" : "  (wirable)") : ""), argInput(n, f)));
    }
  }

  function argInput(n, f) {
    const cur = n.args[f.name];
    const set = (v) => { if (v === "" || v == null) delete n.args[f.name]; else n.args[f.name] = v; dirty = true; };
    const choices = f.choices || f.enum;
    const t = (f.type || "string").toLowerCase();
    if (t === "boolean" || t === "bool") {
      const cb = h("input", { type: "checkbox" }); cb.checked = cur ?? !!f.default;
      cb.addEventListener("change", () => set(cb.checked));
      return h("span", { class: "wf-bool" }, cb);
    }
    if (Array.isArray(choices) && choices.length) {
      const sel = h("select", { class: "wf-field" }, h("option", { value: "" }, "— default —"),
        ...choices.map((c) => h("option", { value: String(c) }, String(c))));
      sel.value = cur != null ? String(cur) : "";
      sel.addEventListener("change", () => set(sel.value));
      return sel;
    }
    const numeric = ["integer", "int", "number", "float"].includes(t);
    const inp = h("input", { class: "wf-field", type: numeric ? "number" : "text",
      placeholder: f.default != null ? String(f.default) : "", value: cur != null ? String(cur) : "" });
    inp.addEventListener("input", () => set(numeric && inp.value !== "" ? Number(inp.value) : inp.value));
    // file/dir parameters get a "browse the server" button (same picker as the form)
    if (f.widget === "path") {
      const browse = h("button", { class: "wf-browse btn-icon", type: "button", title: "Browse server",
        onclick: () => openFilePicker(f.path_kind || "file", (p) => { inp.value = p; set(p); }) }, icon("folderOpen", 15));
      return h("div", { class: "wf-path-row" }, inp, browse);
    }
    return inp;
  }

  function loadSchema(program) {
    if (schemaCache[program]) return Promise.resolve(schemaCache[program]);
    return api.program(program).then((s) => (schemaCache[program] = s)).catch(() => null);
  }

  // ── layout ──
  function toggleOrientation() {
    graph.orientation = horiz() ? "vertical" : "horizontal";
    syncOrientBtn(); autoLayout(); dirty = true;
  }
  function syncOrientBtn() {
    clear(orientBtn);
    orientBtn.append(icon(horiz() ? "arrowRight" : "chevronRight", 14), horiz() ? "Horizontal" : "Vertical");
  }

  // Arrange nodes into longest-path dependency tiers along the layout axis.
  function autoLayout() {
    const order = topoLocal();
    if (!order) { flash("Can't auto-layout: the graph has a cycle.", "err"); return; }
    const depth = {};
    for (const n of graph.nodes) depth[n.id] = 0;
    for (const id of order) for (const e of graph.edges) if (e.from === id) depth[e.to] = Math.max(depth[e.to] || 0, (depth[id] || 0) + 1);
    const tiers = {};
    for (const n of graph.nodes) (tiers[depth[n.id] || 0] ||= []).push(n);
    const PAD = 40, GAP_MAIN = 100, GAP_CROSS = 26;
    Object.keys(tiers).map(Number).sort((a, b) => a - b).forEach((d) => {
      let cross = PAD;
      for (const n of tiers[d]) {
        if (horiz()) { n.x = PAD + d * (NODE_W + GAP_MAIN); n.y = cross; cross += nodeHeight(n) + GAP_CROSS; }
        else { n.y = PAD + d * (HEADER_H + 3 * ROW_H + 70); n.x = cross; cross += NODE_W + GAP_CROSS; }
      }
    });
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
  function topoLocal() {
    const ids = graph.nodes.map((n) => n.id);
    const indeg = Object.fromEntries(ids.map((i) => [i, 0]));
    const adj = Object.fromEntries(ids.map((i) => [i, []]));
    for (const e of graph.edges) if (adj[e.from] && e.to in indeg) { adj[e.from].push(e.to); indeg[e.to]++; }
    const q = ids.filter((i) => indeg[i] === 0), order = [];
    while (q.length) { const i = q.shift(); order.push(i); for (const j of adj[i]) if (--indeg[j] === 0) q.push(j); }
    return order.length === ids.length ? order : null;
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
  // Scroll so the graph's bounding box sits in the middle of the viewport (the
  // content is scaled by `zoom`, so content coords map to scroll via *zoom).
  function centerView() {
    const b = graphBBox(); if (!b) return;
    canvas.scrollLeft = Math.max(0, Math.min(CONTENT_W * zoom - canvas.clientWidth, b.cx * zoom - canvas.clientWidth / 2));
    canvas.scrollTop = Math.max(0, Math.min(CONTENT_H * zoom - canvas.clientHeight, b.cy * zoom - canvas.clientHeight / 2));
    drawMinimap();
  }
  function toggleMinimap() { minimapOn = !minimapOn; minimap.classList.toggle("hidden", !minimapOn); mmBtn.classList.toggle("on", minimapOn); drawMinimap(); }

  function drawMinimap() {
    if (!minimapOn) return;
    const W = minimap.clientWidth, H = minimap.clientHeight;
    if (!W || !H) return;
    const dpr = window.devicePixelRatio || 1;
    minimap.width = Math.round(W * dpr); minimap.height = Math.round(H * dpr);
    const ctx = minimap.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const b = graphBBox();
    if (!b) { mmXform = null; return; }
    const pad = 60;
    const gx0 = b.x0 - pad, gy0 = b.y0 - pad, gw = b.w + 2 * pad, gh = b.h + 2 * pad;
    const s = Math.min(W / gw, H / gh);
    const ox = (W - gw * s) / 2, oy = (H - gh * s) / 2;
    mmXform = { gx0, gy0, ox, oy, s };
    const tx = (x) => ox + (x - gx0) * s, ty = (y) => oy + (y - gy0) * s;
    // edges — connect tile centers (the tiles are small squares centered on each
    // node), so wires visibly join the boxes; tiles are drawn after, on top.
    const center = (n) => ({ x: n.x + NODE_W / 2, y: n.y + nodeHeight(n) / 2 });
    ctx.strokeStyle = "#64748b"; ctx.lineWidth = 1.25;
    for (const e of graph.edges) {
      const sN = nodeById(e.from), dN = nodeById(e.to); if (!sN || !dN) continue;
      const p1 = center(sN), p2 = center(dN);
      ctx.beginPath(); ctx.moveTo(tx(p1.x), ty(p1.y)); ctx.lineTo(tx(p2.x), ty(p2.y)); ctx.stroke();
    }
    // nodes — small rounded squares colored by run status. Size from the node's
    // *smaller* dimension (its height) so stacked tier neighbours never overlap
    // (nodes are much wider than tall, and stack closer vertically than NODE_W).
    for (const n of graph.nodes) {
      const st = runState && runState[n.id] ? runState[n.id].status : null;
      const col = (st && MINI_STATUS[st]) || "#94a3b8";
      const side = Math.max(7, Math.min(NODE_W, nodeHeight(n)) * s * 0.82);
      const cx = tx(n.x + NODE_W / 2) - side / 2, cy = ty(n.y + nodeHeight(n) / 2) - side / 2;
      roundRectPath(ctx, cx, cy, side, side, Math.max(2, side * 0.28));
      ctx.fillStyle = col; ctx.fill();
      ctx.lineWidth = 1; ctx.strokeStyle = "rgba(0,0,0,0.35)"; ctx.stroke();
    }
    // current viewport rectangle (scroll is in scaled px → /zoom for content coords)
    ctx.strokeStyle = "#2dd4bf"; ctx.lineWidth = 1.5;
    ctx.strokeRect(tx(canvas.scrollLeft / zoom), ty(canvas.scrollTop / zoom), (canvas.clientWidth / zoom) * s, (canvas.clientHeight / zoom) * s);
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
      syncProjectBtn();
      flash(`Saved as project "${name}" — this workflow now drives it.`, "muted");
    } catch (err) { flash("Couldn't create project: " + (err.detail || err.message), "err"); }
  }

  // "Save as project" is only offered for a draft workflow not yet bound to one.
  function syncProjectBtn() { asProjBtn.style.display = wf && wf.project_id ? "none" : ""; }

  // ── persistence + run ──
  function graphSnapshot() { return { nodes: graph.nodes, edges: graph.edges, orientation: graph.orientation }; }
  function flash(msg, kind = "warn") { statusEl.className = "wf-status " + kind; statusEl.textContent = msg; }

  async function save() {
    const pid = getState().activeProjectId;
    const name = nameInput.value.trim() || "Untitled workflow";
    try {
      if (wf) wf = await api.updateWorkflow(wf.id, { name, graph: graphSnapshot() });
      else wf = await api.createWorkflow({ project_id: pid, name, graph: graphSnapshot() });
      dirty = false; flash("Saved.", "muted"); await refreshList(); wfSelect.value = wf.id; syncProjectBtn();
    } catch (err) { flash("Save failed: " + (err.detail || err.message), "err"); }
    return wf;
  }

  // ids === null -> run the whole workflow; an array -> run that subset only.
  async function runNodes(ids) {
    if (!(wf && wf.project_id) && !getState().activeProjectId) { flash("Open a project first (Projects), or use “Save as project” — steps run into a project.", "err"); return; }
    if (!graph.nodes.length) { flash("Add at least one step.", "warn"); return; }
    if (ids && !ids.length) { flash("Select one or more steps first.", "warn"); return; }
    await save();
    if (!wf) return;
    try {
      wf = await api.runWorkflow(wf.id, ids || undefined);
      flash(ids ? `Running ${ids.length} step(s)…` : "Running…", "busy");
      startPolling();
    } catch (err) {
      const errs = err.detail && err.detail.errors ? err.detail.errors : [err.detail || err.message];
      flash("Cannot run: " + errs.join("; "), "err");
    }
  }

  // Adopt a server graph into local state (keeping the id counters ahead).
  function adoptGraph(g) {
    graph = { nodes: g.nodes || [], edges: g.edges || [], orientation: g.orientation || graph.orientation || "horizontal" };
    nodeSeq = Math.max(1, ...graph.nodes.map((n) => +(n.id.replace(/\D/g, "")) || 0)) + 1;
    edgeSeq = Math.max(1, ...graph.edges.map((e) => +((e.id || "").replace(/\D/g, "")) || 0)) + 1;
  }

  function startPolling() {
    stopPolling();
    poll = setInterval(async () => {
      if (!wf) return stopPolling();
      try {
        const w = await api.workflow(wf.id);
        wf = w; runState = w.run_state || null;
        // when a run settles, adopt the server graph so live node<->job links and
        // statuses (project workflows write job ids back) are reflected
        if (w.status !== "running" && w.graph) { adoptGraph(w.graph); if (!dirty) renderInspector(); }
        renderNodes();
        const counts = countStatuses();
        flash(`${w.status} — ${counts}`, w.status === "running" ? "busy" : (w.status === "succeeded" ? "muted" : "err"));
        if (w.status !== "running") stopPolling();
      } catch { stopPolling(); }
    }, 2000);
  }
  function stopPolling() { if (poll) { clearInterval(poll); poll = null; } }
  function countStatuses() {
    if (!runState) return "";
    const c = {};
    for (const s of Object.values(runState)) c[s.status] = (c[s.status] || 0) + 1;
    return Object.entries(c).map(([k, v]) => `${v} ${k}`).join(", ");
  }

  async function openWorkflow(id) {
    stopPolling();
    selectedSet.clear();
    if (!id) {
      wf = null; graph = { nodes: [], edges: [], orientation: "horizontal" }; runState = null; selected = null; nameInput.value = "";
      wfSelect.value = ""; syncOrientBtn(); syncProjectBtn(); renderNodes(); renderInspector(); flash("New workflow.", "muted"); return;
    }
    try {
      wf = await api.workflow(id);
      adoptGraph(wf.graph || {});
      runState = wf.run_state || null; selected = null; nameInput.value = wf.name || ""; wfSelect.value = wf.id;
      syncOrientBtn(); syncProjectBtn();
      // freshly imported / never-laid-out graphs have all-zero coords — tidy once
      if (graph.nodes.length && graph.nodes.every((n) => !n.x && !n.y)) autoLayout();
      else { renderNodes(); centerView(); }
      renderInspector();
      flash(wf.status && wf.status !== "idle" ? `${wf.status} — ${countStatuses()}` : "Loaded.", "muted");
      if (wf.status === "running") startPolling();
    } catch (err) { flash("Load failed: " + (err.detail || err.message), "err"); }
  }

  async function removeWorkflow() {
    if (!wf) return openWorkflow(null);
    if (!confirm(`Delete workflow "${wf.name}"?`)) return;
    try { await api.deleteWorkflow(wf.id); await refreshList(); openWorkflow(null); }
    catch (err) { flash("Delete failed: " + (err.detail || err.message), "err"); }
  }

  boot();
  return { el: root, cleanup: () => stopPolling() };
}
