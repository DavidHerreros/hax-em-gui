// Shared job-label UI: colored chips + an anchored picker that assigns labels to
// a job and lets you create / recolor / rename / delete labels inline. A tiny
// module-level cache keeps the palette in sync across the Jobs, Project and
// Workflow views (refresh after any mutation, notify subscribers).
import { h, clear } from "../h.js";
import { icon } from "../icons.js";
import { api } from "../api.js";

const PALETTE = ["#ef4444", "#f59e0b", "#eab308", "#22c55e", "#14b8a6", "#3b82f6", "#8b5cf6", "#ec4899", "#64748b", "#0ea5e9"];

let _cache = null;                 // [{id,name,color,count}]
const _subs = new Set();

export function getLabelsCache() { return _cache || []; }
export function onLabelsChanged(fn) { _subs.add(fn); return () => _subs.delete(fn); }
function _notify() { for (const fn of _subs) { try { fn(_cache); } catch { /* ignore */ } } }

export async function loadLabels(force = false) {
  if (_cache && !force) return _cache;
  _cache = await api.labels();
  return _cache;
}
async function _refresh() { _cache = await api.labels(); _notify(); return _cache; }

const labelById = (id) => getLabelsCache().find((l) => l.id === id);
export function labelsByIds(ids) { return (ids || []).map(labelById).filter(Boolean); }

// One read-only chip. `onRemove` (optional) renders a × to detach it.
export function labelChip(label, onRemove) {
  if (!label) return null;
  return h("span", { class: "lbl-chip", style: { "--lc": label.color }, title: label.name },
    h("span", { class: "lbl-dot" }),
    h("span", { class: "lbl-name" }, label.name),
    onRemove ? h("button", { class: "lbl-x", title: "Remove label", onclick: (e) => { e.preventDefault(); e.stopPropagation(); onRemove(label); } }, "×") : null);
}

// A job's chips. `job.labels` is the attached list. Pass `editable` + `onChange`
// to add a "+" button that opens the picker (and updates `job.labels` in place).
export function jobLabelChips(job, { editable = false, onChange } = {}) {
  const wrap = h("span", { class: "lbl-chips" });
  const render = () => {
    clear(wrap);
    for (const l of job.labels || []) wrap.appendChild(labelChip(l, editable ? (lab) => detach(lab) : null));
    if (editable) {
      wrap.appendChild(h("button", {
        class: "lbl-add", title: "Add label",
        onclick: (e) => { e.preventDefault(); e.stopPropagation(); openLabelPicker(e.currentTarget, job, () => { render(); onChange && onChange(job); }); },
      }, icon("plus", 12)));
    }
  };
  async function detach(lab) {
    const ids = (job.labels || []).filter((l) => l.id !== lab.id).map((l) => l.id);
    try { const r = await api.setJobLabels(job.id, ids); job.labels = r.labels; render(); onChange && onChange(job); } catch { /* ignore */ }
  }
  render();
  return wrap;
}

// A node's label chips (workflow builder). `node.labels` is a list of label ids;
// resolves them against the cache and offers a "+" to pick (no server call —
// the labels are applied to the job when the node runs).
export function nodeLabelChips(node, { onChange } = {}) {
  const wrap = h("span", { class: "lbl-chips" });
  const render = () => {
    clear(wrap);
    for (const l of labelsByIds(node.labels)) wrap.appendChild(labelChip(l, (lab) => {
      node.labels = (node.labels || []).filter((id) => id !== lab.id); render(); onChange && onChange();
    }));
    wrap.appendChild(h("button", {
      class: "lbl-add", title: "Add label",
      onclick: (e) => { e.preventDefault(); e.stopPropagation(); openNodeLabelPicker(e.currentTarget, node, () => { render(); onChange && onChange(); }); },
    }, icon("plus", 12)));
  };
  render();
  return wrap;
}

// Anchored popover with an adapter so it serves both a job (persist via API) and
// a workflow node (mutate node.labels locally). Includes inline create + manage.
function openPicker(anchorEl, adapter, onChange) {
  const listEl = h("div", { class: "lblpick-list" });
  const newName = h("input", { class: "lblpick-new", placeholder: "New label…", maxLength: 40 });
  let newColor = PALETTE[Math.floor(Math.random() * PALETTE.length)];
  const swatch = h("button", { class: "lblpick-swatch", title: "Pick color", style: { background: newColor } });
  swatch.onclick = (e) => { e.preventDefault(); openColorMenu(swatch, newColor, (c) => { newColor = c; swatch.style.background = c; }); };
  const addBtn = h("button", { class: "btn small", onclick: createNew }, icon("plus", 13), "Create");

  const menu = h("div", { class: "lblpick" },
    h("div", { class: "lblpick-head" }, "Labels"),
    listEl,
    h("div", { class: "lblpick-create" }, swatch, newName, addBtn));
  const overlay = h("div", { class: "ctx-overlay", onclick: close }, menu);
  document.body.appendChild(overlay);
  document.addEventListener("keydown", onKey, true);
  position();
  renderList();

  function selectedSet() { return new Set(adapter.selected()); }

  async function renderList() {
    await loadLabels();
    clear(listEl);
    const labels = getLabelsCache();
    const sel = selectedSet();
    if (!labels.length) listEl.appendChild(h("div", { class: "muted small", style: { padding: "6px 4px" } }, "No labels yet — create one below."));
    for (const l of labels) {
      const on = sel.has(l.id);
      listEl.appendChild(h("div", { class: "lblpick-row" + (on ? " on" : "") },
        h("button", { class: "lblpick-toggle", onclick: () => toggle(l) },
          h("span", { class: "lbl-dot", style: { "--lc": l.color, background: l.color } }),
          h("span", { class: "lbl-name" }, l.name),
          on ? icon("check", 14) : null),
        h("button", { class: "lblpick-edit", title: "Edit label", onclick: (e) => { e.stopPropagation(); openLabelEditor(e.currentTarget, l, async () => { await _refresh(); renderList(); onChange && onChange(); }); } }, icon("pencil", 13))));
    }
  }
  async function toggle(l) {
    const sel = selectedSet();
    if (sel.has(l.id)) sel.delete(l.id); else sel.add(l.id);
    try { await adapter.setSelected([...sel]); onChange && onChange(); renderList(); } catch { /* ignore */ }
  }
  async function createNew() {
    const name = newName.value.trim();
    if (!name) { newName.focus(); return; }
    try {
      const l = await api.createLabel(name, newColor);
      await _refresh();
      await adapter.setSelected([...selectedSet(), l.id]);
      newName.value = "";
      onChange && onChange();
      renderList();
    } catch { /* ignore */ }
  }
  function position() {
    const r = anchorEl.getBoundingClientRect();
    const mw = menu.offsetWidth || 240, mh = menu.offsetHeight || 0;
    const left = Math.max(8, Math.min(r.left, window.innerWidth - mw - 8));
    let top = r.bottom + 4;
    if (top + mh > window.innerHeight - 8) top = Math.max(8, r.top - mh - 4);
    menu.style.left = left + "px"; menu.style.top = top + "px";
  }
  function close() { overlay.remove(); document.removeEventListener("keydown", onKey, true); }
  function onKey(e) { if (e.key === "Escape") { e.stopPropagation(); close(); } }
}

// Anchored popover: toggle this job's labels (persisted to the server).
export function openLabelPicker(anchorEl, job, onChange) {
  openPicker(anchorEl, {
    selected: () => (job.labels || []).map((l) => l.id),
    setSelected: async (ids) => { const r = await api.setJobLabels(job.id, ids); job.labels = r.labels; },
  }, () => onChange && onChange(job));
}

// Anchored popover for a workflow node (labels stored locally as ids on the node).
export function openNodeLabelPicker(anchorEl, node, onChange) {
  openPicker(anchorEl, {
    selected: () => node.labels || [],
    setSelected: (ids) => { node.labels = ids; },
  }, onChange);
}

// Editor popover for a single label: rename, recolor, delete.
function openLabelEditor(anchorEl, label, onDone) {
  const nameI = h("input", { class: "lblpick-new", value: label.name, maxLength: 40 });
  const swatches = h("div", { class: "lblpick-swatches" },
    ...PALETTE.map((c) => h("button", { class: "lblpick-sw" + (c === label.color ? " on" : ""), style: { background: c }, title: c,
      onclick: async () => { await api.updateLabel(label.id, { color: c }); await onDone(); close(); } })));
  const save = h("button", { class: "btn small", onclick: async () => { await api.updateLabel(label.id, { name: nameI.value }); await onDone(); close(); } }, icon("check", 13), "Save");
  const del = h("button", { class: "btn small ghost", style: { color: "var(--err)" }, onclick: async () => { if (!confirm(`Delete label "${label.name}"? It will be removed from all jobs.`)) return; await api.deleteLabel(label.id); await onDone(); close(); } }, icon("trash", 13));
  const menu = h("div", { class: "lblpick lblpick-editor" },
    h("div", { class: "lblpick-head" }, "Edit label"),
    nameI, swatches, h("div", { class: "lblpick-create" }, save, h("span", { style: { flex: "1" } }), del));
  const overlay = h("div", { class: "ctx-overlay", onclick: close }, menu);
  document.body.appendChild(overlay);
  document.addEventListener("keydown", onKey, true);
  const r = anchorEl.getBoundingClientRect();
  const mw = menu.offsetWidth || 240;
  menu.style.left = Math.max(8, Math.min(r.left, window.innerWidth - mw - 8)) + "px";
  menu.style.top = (r.bottom + 4) + "px";
  function close() { overlay.remove(); document.removeEventListener("keydown", onKey, true); }
  function onKey(e) { if (e.key === "Escape") { e.stopPropagation(); close(); } }
}

function openColorMenu(anchorEl, current, onPick) {
  const menu = h("div", { class: "lblpick" },
    h("div", { class: "lblpick-swatches" },
      ...PALETTE.map((c) => h("button", { class: "lblpick-sw" + (c === current ? " on" : ""), style: { background: c },
        onclick: (e) => { e.preventDefault(); onPick(c); close(); } }))));
  const overlay = h("div", { class: "ctx-overlay", onclick: close }, menu);
  document.body.appendChild(overlay);
  const r = anchorEl.getBoundingClientRect();
  menu.style.left = r.left + "px"; menu.style.top = (r.bottom + 4) + "px";
  function close() { overlay.remove(); }
}
