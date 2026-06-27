// "Use a job's output as this input" — the chaining UI. A bindable form field
// can be fed from a previous job either by picking it from a list or by dragging
// an execution row onto the field. The chosen job is recorded in a shared
// `bindings` map ({ field: { job_id, kind, label } }) the launcher submits.
import { h, mount, clear } from "../h.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { store } from "../store.js";
import { errorBox } from "./ui.js";

export const JOB_DRAG_TYPE = "application/x-hax-job";

// Human label for an artifact kind, used in picker copy.
function prettyKind(kind) {
  if (kind === "latents") return "latents";
  if (kind && kind.endsWith("_model")) return "a saved network";
  return kind;
}

// Mark a DOM element as a draggable chaining source (an execution row).
export function makeJobDraggable(el, job) {
  el.setAttribute("draggable", "true");
  el.addEventListener("dragstart", (e) => {
    e.dataTransfer.setData(JOB_DRAG_TYPE, JSON.stringify({ id: job.id, program: job.program, name: job.name || null }));
    e.dataTransfer.setData("text/plain", job.name || job.program);
    e.dataTransfer.effectAllowed = "copy";
    el.classList.add("dragging");
  });
  el.addEventListener("dragend", () => el.classList.remove("dragging"));
}

// Build the binding control for one bindable field. Returns the control element;
// also wires `fieldEl` as a drop target. Mutates `bindings`.
export function bindingControl({ field, kind, producers, bindings, fieldEl, valueInput, onChange }) {
  const row = h("div", { class: "bind-row" });

  function setBinding(b) {
    if (b) bindings[field.name] = b; else delete bindings[field.name];
    if (valueInput) { valueInput.disabled = !!b; if (b) valueInput.value = ""; }
    render();
    if (onChange) onChange();
  }

  function render() {
    clear(row);
    const b = bindings[field.name];
    if (b) {
      row.appendChild(h("div", { class: "bind-chip" },
        icon("link", 13),
        h("span", { class: "bind-chip-name", title: b.label || b.job_id }, b.label || b.job_id),
        h("button", { class: "bind-chip-x", type: "button", title: "Unlink", onclick: () => setBinding(null) }, icon("x", 12))));
    } else {
      row.appendChild(h("button", { class: "bind-btn", type: "button",
        onclick: () => openSourcePicker(kind, (src) => setBinding({ job_id: src.id, kind, label: src.name || src.program })) },
        icon("link", 13), h("span", null, "Use a job's output")));
      row.appendChild(h("span", { class: "bind-hint muted" }, "or drag a run here"));
    }
  }

  if (fieldEl) {
    fieldEl.addEventListener("dragover", (e) => {
      if (Array.from(e.dataTransfer.types || []).includes(JOB_DRAG_TYPE)) { e.preventDefault(); fieldEl.classList.add("drop-ok"); }
    });
    fieldEl.addEventListener("dragleave", (e) => { if (!fieldEl.contains(e.relatedTarget)) fieldEl.classList.remove("drop-ok"); });
    fieldEl.addEventListener("drop", (e) => {
      const raw = e.dataTransfer.getData(JOB_DRAG_TYPE);
      fieldEl.classList.remove("drop-ok");
      if (!raw) return;
      e.preventDefault();
      let job;
      try { job = JSON.parse(raw); } catch { return; }
      if (!producers.includes(job.program)) { flash(fieldEl); return; }
      setBinding({ job_id: job.id, kind, label: job.name || job.program });
    });
  }

  render();
  return row;
}

function flash(el) {
  el.classList.add("drop-bad");
  setTimeout(() => el.classList.remove("drop-bad"), 600);
}

// Modal listing this project's jobs whose output can feed `kind`.
export function openSourcePicker(kind, onPick) {
  const proj = store.activeProject();
  const list = h("div", { class: "filelist" });
  const modal = h("div", { class: "modal", onclick: (e) => e.stopPropagation() },
    h("div", { class: "modal-head" }, h("span", null, "Use a job's output"),
      h("button", { class: "btn-icon", style: { border: "none" }, onclick: close }, icon("x", 18))),
    h("div", { class: "modal-path" }, `Compatible runs in ${proj ? proj.name : "this project"} (${kind})`),
    list);
  const overlay = h("div", { class: "overlay", onclick: close }, modal);
  document.body.appendChild(overlay);
  document.addEventListener("keydown", onKey);

  function close() { overlay.remove(); document.removeEventListener("keydown", onKey); }
  function onKey(e) { if (e.key === "Escape") close(); }

  mount(list, h("p", { class: "muted", style: { padding: "12px" } }, "Loading…"));
  if (!proj) { mount(list, h("p", { class: "muted", style: { padding: "12px" } }, "Open a project first.")); return; }
  api.projectSources(proj.id, kind)
    .then((sources) => {
      if (!sources.length) {
        mount(list, h("p", { class: "muted", style: { padding: "12px" } },
          `No compatible runs in this project yet (need a finished run that produces ${prettyKind(kind)}).`));
        return;
      }
      mount(list, h("div", null, ...sources.map((s) => h(
        "button", { class: "entry", onclick: () => { onPick(s); close(); } },
        icon("box", 16),
        h("span", { class: "nm" }, s.name || s.program),
        h("span", { class: "sz mono" }, s.program)))));
    })
    .catch((err) => mount(list, errorBox(err)));
}
