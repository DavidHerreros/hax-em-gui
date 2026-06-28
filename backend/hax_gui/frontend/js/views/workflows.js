// Visual workflow builder: a node-graph canvas where program nodes are wired
// output->input (mapping to the job input-binding system) and run as a DAG.
// Hand-written vanilla JS; SVG for the wires, absolutely-positioned divs for nodes.
import { h, clear, mount } from "../h.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { store, getState } from "../store.js";
import { errorBox } from "./ui.js";
import { visibleFields } from "./formFields.js";

const NODE_W = 196;     // node width (px)
const HEADER_H = 34;    // node header height
const ROW_H = 26;       // per-input-port row height
const SVG_NS = "http://www.w3.org/2000/svg";

const STATUS_COLOR = {
  pending: "var(--muted)", running: "var(--accent)", succeeded: "#22c55e",
  failed: "var(--err)", blocked: "var(--warn)", skipped: "var(--muted)",
};

function svgEl(tag, attrs = {}) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

export function WorkflowsView() {
  let catalog = [];
  let byName = {};
  let graph = { nodes: [], edges: [] };
  let wf = null;            // saved workflow record
  let selected = null;      // selected node id
  let runState = null;      // {nodeId: {job_id, status, error}}
  let dirty = false;
  let poll = null;
  const schemaCache = {};
  let drag = null;          // active node drag / wiring gesture
  let nodeSeq = 1, edgeSeq = 1;

  const root = h("div", { class: "view workflows-view" });

  // ── top bar ──
  const nameInput = h("input", { class: "wf-name", placeholder: "Workflow name", value: "" });
  nameInput.addEventListener("input", () => { dirty = true; });
  const wfSelect = h("select", { class: "wf-select", title: "Open a saved workflow" });
  wfSelect.addEventListener("change", () => openWorkflow(wfSelect.value || null));
  const statusEl = h("span", { class: "wf-status muted" }, "");
  const saveBtn = h("button", { class: "btn", onclick: save }, icon("save", 15), "Save");
  const runBtn = h("button", { class: "btn", onclick: run }, icon("play", 15), "Run");
  const delBtn = h("button", { class: "btn", title: "Delete this workflow", onclick: removeWorkflow }, icon("trash", 15));
  const topbar = h("div", { class: "wf-topbar" },
    h("strong", null, "Workflow"), wfSelect,
    h("button", { class: "btn small", onclick: () => openWorkflow(null) }, "+ New"),
    nameInput, h("span", { style: { flex: "1" } }), statusEl, saveBtn, runBtn, delBtn);

  // ── body: palette | canvas | inspector ──
  const palette = h("div", { class: "wf-palette" });
  const svg = svgEl("svg", { class: "wf-wires", width: "4000", height: "3000" });
  const nodesLayer = h("div", { class: "wf-nodes" });
  const canvas = h("div", { class: "wf-canvas" }, svg, nodesLayer);
  const inspector = h("div", { class: "wf-inspector" });
  const body = h("div", { class: "wf-body" }, palette, canvas, inspector);
  root.append(topbar, body);

  // clicking empty canvas clears the selection
  canvas.addEventListener("pointerdown", (e) => {
    if (e.target === canvas || e.target === nodesLayer || e.target === svg) { selected = null; renderNodes(); renderInspector(); }
  });

  // ── data load ──
  async function boot() {
    try {
      catalog = await api.workflowCatalog();
      byName = Object.fromEntries(catalog.map((p) => [p.name, p]));
    } catch (err) { mount(palette, errorBox(err)); return; }
    renderPalette();
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
    clear(palette);
    palette.appendChild(h("div", { class: "wf-palette-head" }, "Programs"));
    palette.appendChild(h("div", { class: "muted small", style: { padding: "0 10px 6px" } }, "Click to add a step."));
    for (const p of catalog) {
      const producesTag = p.produces && p.produces.length ? "out: " + (p.produces.includes("latents") ? "latents" : p.produces[0]) : "";
      palette.appendChild(h("button", { class: "wf-pal-item", title: p.description || p.name, onclick: () => addNode(p.name) },
        h("span", { class: "mono" }, p.name),
        producesTag ? h("span", { class: "wf-pal-out muted small" }, producesTag) : null));
    }
  }

  // ── graph mutation ──
  function addNode(program) {
    const x = 60 + (graph.nodes.length % 4) * 60 + canvas.scrollLeft;
    const y = 60 + (graph.nodes.length % 6) * 40 + canvas.scrollTop;
    const id = "n" + (nodeSeq++);
    graph.nodes.push({ id, program, args: {}, gpu: "0", name: "", x, y });
    selected = id; dirty = true;
    renderNodes(); renderInspector();
  }
  function removeNode(id) {
    graph.nodes = graph.nodes.filter((n) => n.id !== id);
    graph.edges = graph.edges.filter((e) => e.from !== id && e.to !== id);
    if (selected === id) selected = null;
    dirty = true; renderNodes(); renderInspector();
  }
  function addEdge(fromId, toId, field, kind) {
    if (fromId === toId) return;
    graph.edges = graph.edges.filter((e) => !(e.to === toId && e.field === field));   // one source per field
    graph.edges.push({ id: "e" + (edgeSeq++), from: fromId, to: toId, field, kind });
    dirty = true; renderNodes(); renderInspector();
  }
  function removeEdge(id) { graph.edges = graph.edges.filter((e) => e.id !== id); dirty = true; renderNodes(); renderInspector(); }

  const nodeById = (id) => graph.nodes.find((n) => n.id === id);
  const inputsOf = (program) => Object.entries((byName[program] || {}).bindable || {});
  const nodeHeight = (n) => HEADER_H + Math.max(1, inputsOf(n.program).length) * ROW_H + 8;
  const outPort = (n) => ({ x: n.x + NODE_W, y: n.y + HEADER_H / 2 });
  const inPort = (n, idx) => ({ x: n.x, y: n.y + HEADER_H + idx * ROW_H + ROW_H / 2 });

  // ── rendering ──
  function renderNodes() {
    clear(nodesLayer);
    for (const n of graph.nodes) nodesLayer.appendChild(nodeEl(n));
    redrawWires();
  }

  function nodeEl(n) {
    const st = runState && runState[n.id] ? runState[n.id].status : null;
    const el = h("div", {
      class: "wf-node" + (selected === n.id ? " sel" : ""),
      style: { left: n.x + "px", top: n.y + "px", width: NODE_W + "px", height: nodeHeight(n) + "px" },
    });
    el.dataset.node = n.id;
    if (st) el.style.borderColor = STATUS_COLOR[st] || "var(--border)";
    // header (drag handle + select)
    const head = h("div", { class: "wf-node-head" },
      h("span", { class: "wf-node-title mono", title: n.program }, n.name || n.program),
      st ? h("span", { class: "wf-node-st", style: { color: STATUS_COLOR[st] } }, st) : null,
      h("button", { class: "wf-node-x", title: "Remove step", onclick: (e) => { e.stopPropagation(); removeNode(n.id); } }, "×"));
    head.addEventListener("pointerdown", (e) => startNodeDrag(e, n));
    el.appendChild(head);
    // input ports
    inputsOf(n.program).forEach(([field, kind], i) => {
      const wired = graph.edges.find((e) => e.to === n.id && e.field === field);
      const port = h("span", { class: "wf-port in" + (wired ? " wired" : "") });
      port.dataset.node = n.id; port.dataset.field = field; port.dataset.kind = kind;
      const row = h("div", { class: "wf-port-row in" }, port,
        h("span", { class: "wf-port-label", title: `${field} (${kind})` }, field));
      el.appendChild(row);
    });
    // single output port
    if ((byName[n.program]?.produces || []).length) {
      const op = h("span", { class: "wf-port out" });
      op.dataset.node = n.id; op.dataset.out = "1";
      op.addEventListener("pointerdown", (e) => startWire(e, n));
      el.appendChild(op);
    }
    return el;
  }

  function wirePath(p1, p2, edge, temp = false) {
    const dx = Math.max(40, Math.abs(p2.x - p1.x) * 0.5);
    const d = `M ${p1.x} ${p1.y} C ${p1.x + dx} ${p1.y}, ${p2.x - dx} ${p2.y}, ${p2.x} ${p2.y}`;
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
      const idx = Math.max(0, inputsOf(d.program).findIndex(([f]) => f === e.field));
      svg.appendChild(wirePath(outPort(s), inPort(d, idx), e));
    }
    if (drag && drag.kind === "wire" && tempTo) svg.appendChild(wirePath(outPort(nodeById(drag.from)), tempTo, null, true));
  }

  // ── gestures (node drag + wiring) ──
  function canvasPoint(e) {
    const r = canvas.getBoundingClientRect();
    return { x: e.clientX - r.left + canvas.scrollLeft, y: e.clientY - r.top + canvas.scrollTop };
  }
  function startNodeDrag(e, n) {
    if (e.target.closest(".wf-node-x")) return;
    e.preventDefault();
    selected = n.id; renderInspector();
    const p = canvasPoint(e);
    drag = { kind: "node", id: n.id, dx: p.x - n.x, dy: p.y - n.y };
    nodesLayer.querySelectorAll(".wf-node").forEach((el) => el.classList.toggle("sel", el.dataset.node === n.id));
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
    const p = canvasPoint(e);
    if (drag.kind === "node") {
      const n = nodeById(drag.id); if (!n) return;
      n.x = Math.max(0, p.x - drag.dx); n.y = Math.max(0, p.y - drag.dy);
      const el = nodesLayer.querySelector(`.wf-node[data-node="${n.id}"]`);
      if (el) { el.style.left = n.x + "px"; el.style.top = n.y + "px"; }
      redrawWires();
    } else {
      redrawWires(p);
    }
  }
  function onUp(e) {
    window.removeEventListener("pointermove", onMove);
    window.removeEventListener("pointerup", onUp);
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
    clear(inspector);
    const n = selected && nodeById(selected);
    if (!n) {
      inspector.appendChild(h("div", { class: "wf-insp-empty muted" },
        "Select a step to edit its parameters. Wire a step's output port (right) into another step's input port (left) to pass data."));
      return;
    }
    inspector.appendChild(h("div", { class: "wf-insp-head" }, h("strong", null, n.program)));
    const nameI = h("input", { class: "wf-field", placeholder: "step name (optional)", value: n.name || "" });
    nameI.addEventListener("input", () => { n.name = nameI.value; dirty = true; const t = nodesLayer.querySelector(`.wf-node[data-node="${n.id}"] .wf-node-title`); if (t) t.textContent = n.name || n.program; });
    const gpuI = h("input", { class: "wf-field", placeholder: "GPU (e.g. 0, empty = CPU)", value: n.gpu ?? "0" });
    gpuI.addEventListener("input", () => { n.gpu = gpuI.value; dirty = true; });
    inspector.appendChild(h("label", { class: "wf-lbl" }, "Name", nameI));
    inspector.appendChild(h("label", { class: "wf-lbl" }, "GPU", gpuI));

    const schema = schemaCache[n.program];
    if (!schema) { api.program(n.program).then((s) => { schemaCache[n.program] = s; if (selected === n.id) renderInspector(); }).catch(() => {}); inspector.appendChild(h("div", { class: "muted small" }, "Loading parameters…")); return; }
    const bindable = byName[n.program]?.bindable || {};
    inspector.appendChild(h("div", { class: "wf-insp-sub muted small" }, "Parameters"));
    for (const f of visibleFields(schema)) {
      const wired = graph.edges.find((e) => e.to === n.id && e.field === f.name);
      if (wired) {
        const src = nodeById(wired.from);
        inspector.appendChild(h("label", { class: "wf-lbl" }, f.name,
          h("div", { class: "wf-wired-chip" }, icon("link", 12), `from ${src ? (src.name || src.program) : "?"} · ${wired.kind}`)));
        continue;
      }
      inspector.appendChild(h("label", { class: "wf-lbl", title: f.help || "" },
        f.name + (f.required ? " *" : "") + (f.name in bindable ? "  (wirable)" : ""), argInput(n, f)));
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
    return inp;
  }

  // ── persistence + run ──
  function graphSnapshot() { return { nodes: graph.nodes, edges: graph.edges }; }
  function flash(msg, kind = "warn") { statusEl.className = "wf-status " + kind; statusEl.textContent = msg; }

  async function save() {
    const pid = getState().activeProjectId;
    const name = nameInput.value.trim() || "Untitled workflow";
    try {
      if (wf) wf = await api.updateWorkflow(wf.id, { name, graph: graphSnapshot() });
      else wf = await api.createWorkflow({ project_id: pid, name, graph: graphSnapshot() });
      dirty = false; flash("Saved.", "muted"); await refreshList(); wfSelect.value = wf.id;
    } catch (err) { flash("Save failed: " + (err.detail || err.message), "err"); }
    return wf;
  }

  async function run() {
    if (!getState().activeProjectId) { flash("Open a project first (Projects) — workflow steps run into it.", "err"); return; }
    if (!graph.nodes.length) { flash("Add at least one step.", "warn"); return; }
    await save();
    if (!wf) return;
    try { wf = await api.runWorkflow(wf.id); flash("Running…", "busy"); startPolling(); }
    catch (err) {
      const errs = err.detail && err.detail.errors ? err.detail.errors : [err.detail || err.message];
      flash("Cannot run: " + errs.join("; "), "err");
    }
  }

  function startPolling() {
    stopPolling();
    poll = setInterval(async () => {
      if (!wf) return stopPolling();
      try {
        const w = await api.workflow(wf.id);
        runState = w.run_state || null; renderNodes();
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
    if (!id) {
      wf = null; graph = { nodes: [], edges: [] }; runState = null; selected = null; nameInput.value = "";
      wfSelect.value = ""; renderNodes(); renderInspector(); flash("New workflow.", "muted"); return;
    }
    try {
      wf = await api.workflow(id);
      graph = wf.graph && wf.graph.nodes ? { nodes: wf.graph.nodes, edges: wf.graph.edges || [] } : { nodes: [], edges: [] };
      // keep id counters ahead of loaded ids
      nodeSeq = Math.max(1, ...graph.nodes.map((n) => +(n.id.replace(/\D/g, "")) || 0)) + 1;
      edgeSeq = Math.max(1, ...graph.edges.map((e) => +((e.id || "").replace(/\D/g, "")) || 0)) + 1;
      runState = wf.run_state || null; selected = null; nameInput.value = wf.name || ""; wfSelect.value = wf.id;
      renderNodes(); renderInspector();
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
