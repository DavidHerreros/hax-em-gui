// Jobs list with light polling. The detail view streams live instead.
import { h, mount, clear } from "../h.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { errorBox, skeleton } from "./ui.js";
import { openContextMenu } from "./contextMenu.js";
import { promptModal } from "./promptModal.js";
import { jobLabelChips, loadLabels } from "./labels.js";

export function badge(status) {
  const animated = status === "running" || status === "pending";
  return h("span", { class: "badge " + status }, animated ? h("span", { class: "dot" }) : null, status);
}

export function JobsView() {
  const root = h("div", { class: "view" }, h("h1", { style: { marginBottom: "16px" } }, "Jobs"));
  const filterBar = h("div", { class: "jobs-filters" });
  const body = h("div");
  root.appendChild(filterBar);
  root.appendChild(body);
  mount(body, skeleton("260px"));

  const filters = { project: "", program: "", status: "", label: "" };
  const projectSel = filterSelect("All projects", (v) => { filters.project = v; renderBody(); });
  const programSel = filterSelect("All programs", (v) => { filters.program = v; renderBody(); });
  const statusSel = filterSelect("All statuses", (v) => { filters.status = v; renderBody(); });
  const labelSel = filterSelect("All labels", (v) => { filters.label = v; renderBody(); });
  const clearBtn = h("button", { class: "btn ghost filter-clear", title: "Clear filters", onclick: () => {
    filters.project = filters.program = filters.status = filters.label = "";
    projectSel.value = programSel.value = statusSel.value = labelSel.value = "";
    renderBody();
  } }, icon("x", 14), "Clear");
  filterBar.append(
    h("div", { class: "filter-group" }, icon("box", 14), projectSel),
    h("div", { class: "filter-group" }, icon("sliders", 14), programSel),
    h("div", { class: "filter-group" }, icon("list", 14), statusSel),
    h("div", { class: "filter-group" }, icon("more", 14), labelSel),
    clearBtn);

  let allJobs = [];
  let stop = false;
  let lastSig = null;
  let animatedOnce = false;

  function renderBody() {
    let content;
    if (!allJobs.length) content = h("p", { class: "muted" }, "No jobs yet. Pick a program from the right menu.");
    else {
      const jobs = allJobs.filter((j) =>
        (!filters.project || projectKey(j) === filters.project) &&
        (!filters.program || j.program === filters.program) &&
        (!filters.status || j.status === filters.status) &&
        (!filters.label || (j.labels || []).some((l) => l.id === filters.label)));
      content = jobs.length ? groupedList(jobs) : h("p", { class: "muted" }, "No jobs match the current filters.");
    }
    mount(body, content);
    // Animate the row entrance only on the first paint; later refreshes (every 3s)
    // rebuild the list, and re-animating them makes the list visibly "jump".
    body.classList.toggle("no-anim", animatedOnce);
    animatedOnce = true;
  }

  function syncFilters() {
    syncOptions(projectSel, distinct(allJobs.map(projectKey)), filters, "project");
    syncOptions(programSel, distinct(allJobs.map((j) => j.program)), filters, "program");
    syncOptions(statusSel, distinct(allJobs.map((j) => j.status)), filters, "status");
    syncLabelOptions();
  }

  // The label filter maps id -> name (a job carries label objects, not strings).
  function syncLabelOptions() {
    const seen = new Map();
    for (const j of allJobs) for (const l of j.labels || []) seen.set(l.id, l.name);
    const sig = JSON.stringify([...seen]);
    if (labelSel._sig === sig) return;
    labelSel._sig = sig;
    if (filters.label && !seen.has(filters.label)) filters.label = "";
    clear(labelSel);
    labelSel.appendChild(h("option", { value: "" }, "All labels"));
    for (const [id, name] of [...seen].sort((a, b) => a[1].localeCompare(b[1])))
      labelSel.appendChild(h("option", { value: id, selected: id === filters.label }, name));
    labelSel.value = filters.label;
  }

  async function refresh() {
    try {
      const jobs = await api.jobs();
      if (stop) return;
      const sig = jobsSignature(jobs);
      if (sig === lastSig) return; // nothing changed -> don't rebuild (avoids flicker)
      lastSig = sig;
      allJobs = jobs;
      syncFilters();
      renderBody();
    } catch (err) {
      if (!stop) mount(body, errorBox(err));
    }
  }
  loadLabels().catch(() => {});
  refresh();
  const timer = setInterval(refresh, 3000);

  return { el: root, cleanup: () => { stop = true; clearInterval(timer); } };
}

function projectKey(job) { return job.project_name || "(no project)"; }
function distinct(arr) { return [...new Set(arr)].sort((a, b) => String(a).localeCompare(String(b))); }

function filterSelect(allLabel, onChange) {
  const sel = h("select", { class: "input filter-select", onchange: (e) => onChange(e.target.value) },
    h("option", { value: "" }, allLabel));
  sel._allLabel = allLabel;
  return sel;
}

// Rebuild a select's options only when the value set changes, preserving (or
// clearing) the current selection.
function syncOptions(sel, values, filters, key) {
  const want = JSON.stringify(values);
  if (sel._sig === want) return;
  sel._sig = want;
  if (filters[key] && !values.includes(filters[key])) filters[key] = "";
  clear(sel);
  sel.appendChild(h("option", { value: "" }, sel._allLabel));
  for (const v of values) sel.appendChild(h("option", { value: v, selected: v === filters[key] }, v));
  sel.value = filters[key];
}

// Group jobs under their project (alphabetical; "(no project)" last).
function groupedList(jobs) {
  const groups = new Map();
  for (const j of jobs) {
    const k = projectKey(j);
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(j);
  }
  const keys = [...groups.keys()].sort((a, b) =>
    a === "(no project)" ? 1 : b === "(no project)" ? -1 : a.localeCompare(b));
  return h("div", null, ...keys.map((k) => {
    const items = groups.get(k);
    return h("div", { class: "jobs-group" },
      h("div", { class: "jobs-group-head" },
        icon("box", 14),
        h("span", { class: "jobs-group-name" }, k),
        h("span", { class: "jobs-group-count" }, String(items.length))),
      h("div", null, ...items.map((j, i) => row(j, i))));
  }));
}

// A change signature for a job list: re-render only when ids/statuses/names change.
export function jobsSignature(jobs) {
  return jobs.map((j) => `${j.id}:${j.status}:${j.name || ""}:${(j.labels || []).map((l) => l.id).join(",")}`).join("|");
}

// What to show as a run's title: its custom name, falling back to the program.
export function jobDisplayName(job) {
  return job.name || job.program;
}

// Shared "⋮" options menu for a job row (Open / Rename / Delete).
export function jobMenuButton(job, { onRenamed, onDeleted } = {}) {
  const btn = h("button", {
    class: "row-menu", title: "Options",
    onclick: (e) => { e.preventDefault(); e.stopPropagation(); open(e.currentTarget); },
  }, icon("more", 16));

  function open(anchor) {
    const items = [];
    if (job.status === "saved") items.push({ label: "Run", icon: icon("play", 15), onClick: run });
    items.push(
      { label: "Open", icon: icon("arrowRight", 15), onClick: () => { location.hash = "#/jobs/" + job.id; } },
      { label: "Rename", icon: icon("pencil", 15), onClick: rename },
    );
    if (!job.active) items.push({ label: "Delete", icon: icon("trash", 15), danger: true, onClick: del });
    openContextMenu(anchor, items);
  }
  async function run() {
    try { await api.startJob(job.id); location.hash = "#/jobs/" + job.id; }
    catch (err) { alert(err && err.detail ? (typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail)) : "Failed to start job."); }
  }
  async function rename() {
    const name = await promptModal({ title: "Rename run", label: "Run name", value: job.name || "", placeholder: job.program, okText: "Rename" });
    if (name === null) return;
    try { const u = await api.renameJob(job.id, name); job.name = u.name; if (onRenamed) onRenamed(u); } catch { /* ignore */ }
  }
  async function del() {
    if (!window.confirm("Delete this job and its run folder (outputs + log)? This cannot be undone.")) return;
    try { await api.deleteJob(job.id); if (onDeleted) onDeleted(); } catch { /* ignore */ }
  }
  return btn;
}

// Display label for a job's device, or null if unspecified.
export function gpuLabel(gpu) {
  if (gpu == null || gpu === "") return null;
  return gpu === "cpu" ? "CPU" : "GPU " + gpu;
}

function row(job, i) {
  const title = h("span", { class: "prog" }, jobDisplayName(job));
  const a = h(
    "a",
    { class: "job-row", href: "#/jobs/" + job.id, style: { animationDelay: `${i * 0.02}s` } },
    badge(job.status),
    title,
    job.name ? h("span", { class: "prog-sub mono" }, job.program) : null,
    jobLabelChips(job, { editable: true }),
    gpuLabel(job.gpu) ? h("span", { class: "muted", style: { fontSize: "12px" } }, gpuLabel(job.gpu)) : null,
    h("span", { class: "time" }, new Date(job.created_at).toLocaleString()),
    jobMenuButton(job, { onRenamed: (u) => { title.textContent = jobDisplayName(u); }, onDeleted: () => a.remove() })
  );
  return a;
}
