// A reusable "runs grouped by project, with filters" browser. The caller supplies
// `renderRow(job)` (which owns each row's click action); this handles the project
// filter, program filter and a name/folder search, and groups the results under each
// project — mirroring the Jobs page so the Metrics and Viewer run lists read the same.
import { h, mount, clear } from "../h.js";
import { icon } from "../icons.js";

export function projectKey(job) { return job.project_name || "(no project)"; }
export function folderName(p) { const s = (p || "").replace(/\/+$/, "").split("/"); return s[s.length - 1] || "run"; }
const distinct = (arr) => [...new Set(arr)].sort((a, b) => String(a).localeCompare(String(b)));

export function runBrowser({ renderRow, emptyText = "No runs." }) {
  const filters = { project: "", program: "", q: "" };
  let jobs = [];

  const projectSel = mkSelect("All projects", (v) => { filters.project = v; render(); });
  const programSel = mkSelect("All programs", (v) => { filters.program = v; render(); });
  const search = h("input", { class: "input filter-select", placeholder: "Search name / folder…",
    oninput: (e) => { filters.q = e.target.value.trim().toLowerCase(); render(); } });
  const clearBtn = h("button", { class: "btn ghost filter-clear", title: "Clear filters",
    onclick: () => { filters.project = filters.program = filters.q = ""; projectSel.value = programSel.value = ""; search.value = ""; render(); } },
    icon("x", 14), "Clear");

  const bar = h("div", { class: "jobs-filters" },
    h("div", { class: "filter-group" }, icon("box", 14), projectSel),
    h("div", { class: "filter-group" }, icon("sliders", 14), programSel),
    h("div", { class: "filter-group" }, icon("search", 14), search),
    clearBtn);
  const body = h("div");
  const el = h("div", null, bar, body);

  const matches = (j) =>
    (!filters.project || projectKey(j) === filters.project) &&
    (!filters.program || j.program === filters.program) &&
    (!filters.q || `${j.name || ""} ${j.program || ""} ${folderName(j.workdir)}`.toLowerCase().includes(filters.q));

  function render() {
    const shown = jobs.filter(matches);
    mount(body, shown.length ? grouped(shown, renderRow)
      : h("p", { class: "muted small", style: { padding: "8px 2px" } }, jobs.length ? "No runs match the filters." : emptyText));
  }

  // Replace the run set (e.g. after a refresh); keeps the current filter selections.
  function setJobs(list) {
    jobs = list || [];
    syncOptions(projectSel, distinct(jobs.map(projectKey)), filters, "project");
    syncOptions(programSel, distinct(jobs.map((j) => j.program)), filters, "program");
    render();
  }
  // Show an arbitrary node in the list area (loading / error), leaving the bar in place.
  function setStatus(node) { mount(body, node); }

  return { el, body, setJobs, setStatus };
}

function grouped(jobs, renderRow) {
  const groups = new Map();
  for (const j of jobs) {
    const k = projectKey(j);
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(j);
  }
  const keys = [...groups.keys()].sort((a, b) => (a === "(no project)" ? 1 : b === "(no project)" ? -1 : a.localeCompare(b)));
  return h("div", null, ...keys.map((k) => h("div", { class: "jobs-group" },
    h("div", { class: "jobs-group-head" },
      icon("box", 14),
      h("span", { class: "jobs-group-name" }, k),
      h("span", { class: "jobs-group-count" }, String(groups.get(k).length))),
    h("div", null, ...groups.get(k).map(renderRow)))));
}

function mkSelect(allLabel, onChange) {
  const sel = h("select", { class: "input filter-select", onchange: (e) => onChange(e.target.value) },
    h("option", { value: "" }, allLabel));
  sel._allLabel = allLabel;
  return sel;
}

// Rebuild a select's options only when the value set changes, preserving (or clearing)
// the current selection — same helper the Jobs page uses.
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
