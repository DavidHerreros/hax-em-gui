// The job tree: one rail shared by the Jobs, Metrics and 3D-viewer pages.
//
// Scoped to the active project (a switcher sits at the top; "All projects" is an
// explicit opt-in), it draws the project's runs as a lineage tree built from their
// input bindings — a run sits under the run whose output it consumed — so "what fed
// what" is the structure of the list rather than something to work out from names.
// Rows are one line (status dot, name, label dots, progress); the rest lives in the
// hover card (jobTip.js) and in the row's ⋮ menu (the same one the old list had).
//
// The rail is rebuilt on every route change (a selection on the Jobs page is a
// route), so its filters, collapsed nodes and scroll offset live at module level
// and the last job list is cached to paint instantly before the refresh lands.
import { h, mount, clear } from "../h.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { store, getState, subscribe } from "../store.js";
import { poller } from "../poller.js";
import { onJobEvent } from "../events.js";
import { errorBox } from "./ui.js";
import { jobMenuButton, jobDisplayName, jobsSignature, onJobRenamed } from "./jobs.js";
import { makeJobDraggable } from "./inputBinding.js";
import { attachJobTip, hideJobTip } from "./jobTip.js";
import { loadLabels } from "./labels.js";
import { navigate, currentPath } from "../router.js";
import { openContextMenu } from "./contextMenu.js";
import { createProjectPrompt, projectMenuItems } from "./projectOverview.js";

const STATUS_ORDER = ["running", "queued", "pending", "cancelling", "succeeded", "failed", "cancelled", "interrupted", "saved"];
const STATUS_LABEL = { running: "Running", queued: "Queued", pending: "Starting", cancelling: "Cancelling", succeeded: "Done", failed: "Failed",
                       cancelled: "Cancelled", interrupted: "Interrupted", saved: "Draft" };

// Survives route re-renders; see the header comment.
const memo = {
  q: "", status: new Set(), program: new Set(), label: new Set(), stale: false,
  all: false, collapsed: new Set(), scroll: 0,
  jobs: null, jobsKey: null,          // last list + what it was for ("all" | project id)
  projects: null,
};

function parentOf(job, ids) {
  const b = job.bindings || {};
  for (const k of Object.keys(b)) {
    const pid = b[k] && b[k].job_id;
    if (pid && pid !== job.id && ids.has(pid)) return pid;
  }
  return null;
}

// roots + children map, oldest first so a tree reads top-down like the pipeline it
// records. A job whose parent chain never reaches a root (a cycle, which the backend
// does not produce but nothing here should hang on) is appended as a root.
function buildTree(jobs) {
  const byTime = (a, b) => String(a.created_at || "").localeCompare(String(b.created_at || ""));
  const sorted = [...jobs].sort(byTime);
  const ids = new Set(sorted.map((j) => j.id));
  const kids = new Map();
  const roots = [];
  for (const j of sorted) {
    const p = parentOf(j, ids);
    if (p) { if (!kids.has(p)) kids.set(p, []); kids.get(p).push(j); }
    else roots.push(j);
  }
  return { roots, kids, all: sorted };
}

// opts:
//   isSelected(job) -> bool
//   onSelect(job)               click on a row
//   usable(job) -> bool         false dims the row (this page cannot open it)
//   emptyText                   shown when the project has nothing usable
export function jobTree(opts = {}) {
  // Callbacks are swappable so a page can keep one tree across its own route changes
  // (see acquireTree in jobs.js) and only tell it what "selected" means now.
  const cfg = { isSelected: null, onSelect: null, usable: () => true, emptyText: "No runs in this project yet.", ...opts };
  const root = h("aside", { class: "jt" });
  const head = h("div", { class: "jt-head" });
  const list = h("div", { class: "jt-list" });
  root.append(head, list);

  let jobs = memo.jobs || [];
  let stop = false;
  let lastSig = memo.jobs ? sigOf(memo.jobs) : null;

  // ── header: project switcher, search, filter, scope ──
  // The name opens the project's overview page; the chevron opens the switcher.
  const projName = h("span", { class: "jt-proj-name" }, "…");
  const projBtn = h("button", { class: "jt-proj", title: "Project overview", onclick: () => navigate("/project") }, projName);
  const switchBtn = h("button", { class: "jt-switch", title: "Switch project" }, icon("chevronUp", 14));
  const projWrap = h("div", { class: "jt-pop-anchor jt-projrow" }, projBtn, switchBtn);
  switchBtn.onclick = (e) => { e.stopPropagation(); togglePop(projWrap, buildProjectMenu); };

  const search = h("input", { class: "jt-search-in", type: "search", placeholder: "Filter by name, program or run id", value: memo.q,
    oninput: (e) => { memo.q = e.target.value.trim().toLowerCase(); renderList(); } });
  const filterBtn = h("button", { class: "jt-ibtn", title: "Filter by status, program or label" }, icon("sliders", 14));
  const filterBadge = h("span", { class: "jt-badge", hidden: true });
  filterBtn.appendChild(filterBadge);
  const filterWrap = h("div", { class: "jt-pop-anchor" }, filterBtn);
  filterBtn.onclick = (e) => { e.stopPropagation(); togglePop(filterWrap, buildFilterMenu); };

  const scopeThis = h("button", { class: "jt-scope", onclick: () => setScope(false) }, "This project");
  const scopeAll = h("button", { class: "jt-scope", onclick: () => setScope(true) }, "All projects");
  const count = h("span", { class: "jt-count" });

  head.append(
    projWrap,
    h("div", { class: "jt-tools" }, h("label", { class: "jt-search" }, icon("search", 14), search), filterWrap),
    h("div", { class: "jt-sub" }, scopeThis, scopeAll, h("span", { class: "ml-auto" }), count));

  // ── popovers (project switcher / filters) ──
  let pop = null;
  function closePop() { if (pop) { pop.remove(); pop = null; document.removeEventListener("mousedown", onDocDown, true); } }
  function onDocDown(e) { if (pop && !pop.contains(e.target)) closePop(); }
  function togglePop(anchor, build) {
    if (pop && pop.parentNode === anchor) { closePop(); return; }
    closePop();
    pop = h("div", { class: "jt-pop" });
    anchor.appendChild(pop);
    build(pop);
    document.addEventListener("mousedown", onDocDown, true);
  }

  // Switching projects keeps the user on the view they are in (Metrics, Viewer,
  // Workflows…): the tree re-reads the new project's runs on the store change. The
  // one page that cannot survive the switch is a run detail, since that run belongs
  // to the old project — fall back to the project overview from there.
  function switchProject(p) {
    store.openProject(p);
    if (currentPath().startsWith("/jobs/")) navigate("/project");
  }

  function buildProjectMenu(el) {
    const ps = memo.projects || [];
    const active = getState().activeProjectId;
    const q = h("input", { class: "jt-search-in jt-pop-search", placeholder: "Switch project…" });
    const rows = h("div", { class: "jt-pop-rows" });
    const draw = () => {
      clear(rows);
      const needle = q.value.trim().toLowerCase();
      const shown = ps.filter((p) => !needle || p.name.toLowerCase().includes(needle));
      for (const p of shown) {
        rows.appendChild(h("div", { class: "jt-pop-row" + (p.id === active ? " on" : ""),
          onclick: () => { closePop(); switchProject(p); } },
          h("span", { class: "jt-pop-row-name" }, p.name),
          p.runs != null ? h("span", { class: "jt-count" }, `${p.runs}`) : null,
          h("button", { class: "row-menu", title: "Project options",
            onclick: (e) => { e.stopPropagation(); closePop(); openContextMenu(e.currentTarget, projectMenuItems(p, { onChanged: loadProjects })); } }, icon("more", 14))));
      }
      if (!shown.length) rows.appendChild(h("div", { class: "muted small", style: { padding: "6px 8px" } }, ps.length ? "No project matches." : "No projects yet."));
    };
    q.oninput = draw;
    draw();
    el.append(q, rows,
      h("div", { class: "jt-pop-foot" },
        h("button", { class: "jt-pop-row", onclick: async () => { closePop(); if (await createProjectPrompt()) { loadProjects(); navigate("/project"); } } },
          icon("plus", 13), h("span", null, "New project…"))));
    setTimeout(() => q.focus(), 0);
  }

  function buildFilterMenu(el) {
    const opt = (on, label, cb) => h("button", { class: "jt-fopt" + (on ? " on" : ""), onclick: () => { cb(); rebuildPop(); renderHead(); renderList(); } },
      h("span", { class: "jt-chk" }, on ? icon("check", 10) : null), label);
    const toggle = (set, v) => (set.has(v) ? set.delete(v) : set.add(v));
    const rebuildPop = () => { clear(el); buildFilterMenu(el); };
    const statuses = STATUS_ORDER.filter((s) => jobs.some((j) => j.status === s));
    const programs = [...new Set(jobs.map((j) => j.program))].sort();
    const labels = new Map();
    for (const j of jobs) for (const l of j.labels || []) labels.set(l.id, l);
    el.append(h("h4", null, "Status"), ...statuses.map((s) => opt(memo.status.has(s), h("span", { class: "row", style: { gap: "7px" } }, h("i", { class: "jt-dot " + s }), STATUS_LABEL[s] || s), () => toggle(memo.status, s))));
    el.append(opt(memo.stale, h("span", { class: "row", style: { gap: "7px" } }, icon("refresh", 11), "Stale only"), () => { memo.stale = !memo.stale; }));
    if (programs.length) el.append(h("h4", null, "Program"), ...programs.map((p) => opt(memo.program.has(p), h("span", { class: "mono" }, p), () => toggle(memo.program, p))));
    if (labels.size) el.append(h("h4", null, "Label"), ...[...labels.values()].map((l) => opt(memo.label.has(l.id),
      h("span", { class: "row", style: { gap: "7px" } }, h("i", { class: "jt-ldot", style: { background: l.color || "var(--faint)" } }), l.name), () => toggle(memo.label, l.id))));
    if (filterCount()) el.append(h("button", { class: "jt-fclear", onclick: () => { memo.status.clear(); memo.program.clear(); memo.label.clear(); memo.stale = false; closePop(); renderHead(); renderList(); } }, "Clear filters"));
  }

  function filterCount() { return memo.status.size + memo.program.size + memo.label.size + (memo.stale ? 1 : 0); }
  // The run folder is `<program>_<job id>` (jobs.py), so a search matches the id at its
  // end and anything printed in a log or a path — the shortest way to find "that run".
  const folderOf = (j) => (j.workdir || "").replace(/\/+$/, "").split("/").pop() || "";
  function matches(j) {
    if (memo.q && !`${j.name || ""} ${j.program} ${j.id} ${folderOf(j)}`.toLowerCase().includes(memo.q)) return false;
    if (memo.status.size && !memo.status.has(j.status)) return false;
    if (memo.program.size && !memo.program.has(j.program)) return false;
    if (memo.label.size && !(j.labels || []).some((l) => memo.label.has(l.id))) return false;
    if (memo.stale && !j.stale) return false;
    return true;
  }

  function setScope(all) {
    if (memo.all === all) return;
    memo.all = all;
    lastSig = null;
    renderHead();
    refresh(true);
  }

  // ── data ──
  function sigOf(list) {
    // jobsSignature covers id/status/stale/name/progress/time; bindings and labels
    // shape the tree and the dots, so they are folded in here.
    return jobsSignature(list) + "#" + list.map((j) => `${Object.keys(j.bindings || {}).length}:${(j.labels || []).map((l) => l.id).join(",")}`).join("|");
  }
  async function refresh(force) {
    const pid = getState().activeProjectId;
    const key = memo.all ? "all" : pid;
    try {
      let list;
      if (memo.all) list = await api.jobs(500);
      else if (pid) list = await api.projectJobs(pid);
      else list = [];
      if (stop) return;
      // The project endpoint's rows carry no project_name; the all-jobs rows do, but
      // fall back to the project list for anything that arrived without one.
      const names = new Map((memo.projects || []).map((p) => [p.id, p.name]));
      const active = store.activeProject();
      for (const j of list) if (!j.project_name) j.project_name = names.get(j.project_id) || (!memo.all && active ? active.name : null);
      const sig = sigOf(list);
      if (!force && sig === lastSig && key === memo.jobsKey) return;
      lastSig = sig;
      jobs = list;
      memo.jobs = list; memo.jobsKey = key;
      renderList();
    } catch (err) {
      lastSig = null;
      if (!stop) mount(list, errorBox(err));
    }
  }
  async function loadProjects() {
    try { memo.projects = await api.projects(); } catch { memo.projects = memo.projects || []; }
    if (!stop) renderHead();
  }

  // ── render ──
  function renderHead() {
    const p = store.activeProject();
    projName.textContent = p ? p.name : "No project open";
    projBtn.disabled = !p;
    switchBtn.classList.toggle("on", memo.all);
    scopeThis.classList.toggle("on", !memo.all);
    scopeAll.classList.toggle("on", memo.all);
    const n = filterCount();
    filterBadge.hidden = !n;
    filterBadge.textContent = String(n);
    filterBtn.classList.toggle("on", n > 0);
  }

  // Rebuilding the list fires a scroll event (an emptied box scrolls to 0) that would
  // clobber the remembered offset; ignore scroll while rebuilding, then put the offset
  // back — once now, once on the next frame for the case where the rail is rebuilt
  // before the router has attached it to the document (setting scrollTop on a
  // detached element does nothing).
  let rebuilding = false;
  function restoreScroll() {
    list.scrollTop = memo.scroll;
    requestAnimationFrame(() => { rebuilding = true; list.scrollTop = memo.scroll; rebuilding = false; });
  }
  function renderList() {
    hideJobTip();
    rebuilding = true;
    clear(list);
    try { renderListBody(); } finally { rebuilding = false; }
    restoreScroll();
  }
  function renderListBody() {
    const pid = getState().activeProjectId;
    if (!memo.all && !pid) {
      list.appendChild(h("div", { class: "jt-empty" },
        "No project open.", h("br"),
        h("button", { class: "jt-link", onclick: (e) => { e.stopPropagation(); togglePop(projWrap, buildProjectMenu); } }, "Choose a project"),
        " or ", h("button", { class: "jt-link", onclick: async () => { if (await createProjectPrompt()) { loadProjects(); navigate("/project"); } } }, "create one"), "."));
      count.textContent = "";
      return;
    }
    const shown = jobs.filter(matches);
    count.textContent = jobs.length ? `${shown.length} of ${jobs.length}` : "";
    if (!jobs.length) { list.appendChild(h("div", { class: "jt-empty" }, cfg.emptyText)); return; }
    if (!shown.length) { list.appendChild(h("div", { class: "jt-empty" }, "Nothing matches these filters.")); return; }

    if (memo.all) {
      const groups = new Map();
      for (const j of jobs) {
        const k = j.project_name || "(no project)";
        if (!groups.has(k)) groups.set(k, []);
        groups.get(k).push(j);
      }
      const keys = [...groups.keys()].sort((a, b) => (a === "(no project)" ? 1 : b === "(no project)" ? -1 : a.localeCompare(b)));
      for (const k of keys) {
        const key = "proj:" + k;
        const open = !memo.collapsed.has(key);
        const nodes = treeNodes(groups.get(k));
        if (!nodes.length) continue;
        list.appendChild(h("div", { class: "jt-group" },
          h("button", { class: "jt-group-head", onclick: () => { open ? memo.collapsed.add(key) : memo.collapsed.delete(key); renderList(); } },
            h("span", { class: "jt-tw" + (open ? " open" : "") }, icon("chevronRight", 11)), h("span", null, k),
            h("span", { class: "jt-count" }, String(groups.get(k).length))),
          open ? h("div", null, ...nodes) : null));
      }
    } else {
      for (const n of treeNodes(jobs)) list.appendChild(n);
    }
  }

  // The tree keeps a node whose own row is filtered out when a descendant matches,
  // so a filtered view still shows where the matching run sits.
  function treeNodes(set) {
    const { roots, kids, all } = buildTree(set);
    // Reachability ignores collapse: a collapsed node's descendants are hidden, not orphans.
    const reached = new Set();
    const reach = (j) => { if (!reached.has(j.id)) { reached.add(j.id); (kids.get(j.id) || []).forEach(reach); } };
    roots.forEach(reach);
    const subtreeMatch = (j) => matches(j) || (kids.get(j.id) || []).some(subtreeMatch);
    const node = (j, last) => {
      const children = (kids.get(j.id) || []).filter(subtreeMatch);
      const open = !memo.collapsed.has(j.id);
      return h("div", { class: "jt-node" + (last ? " last" : "") },
        row(j, children.length, open),
        children.length && open ? h("div", { class: "jt-kids" }, ...children.map((c, i) => node(c, i === children.length - 1))) : null);
    };
    const out = roots.filter(subtreeMatch).map((r, i, arr) => node(r, i === arr.length - 1));
    for (const j of all) if (!reached.has(j.id) && subtreeMatch(j)) { reach(j); out.push(node(j, true)); }
    return out;
  }

  function row(j, nKids, open) {
    const sel = cfg.isSelected && cfg.isSelected(j);
    const ok = matches(j) && cfg.usable(j);
    const p = j.progress;
    const el = h("div", {
      class: "jt-row" + (sel ? " sel" : "") + (ok ? "" : " dim"), "data-id": j.id,
      onclick: () => cfg.onSelect && cfg.onSelect(j),
    },
      h("span", { class: "jt-tw" + (nKids ? (open ? " open" : "") : " leaf"),
        onclick: (e) => { e.stopPropagation(); open ? memo.collapsed.add(j.id) : memo.collapsed.delete(j.id); renderList(); } }, icon("chevronRight", 11)),
      h("i", { class: "jt-dot " + j.status + (j.stale ? " stale" : ""), title: j.stale ? "stale" : j.status }),
      h("span", { class: "jt-name" }, jobDisplayName(j)),
      (j.labels || []).length ? h("span", { class: "jt-ldots" }, ...j.labels.slice(0, 4).map((l) => h("i", { class: "jt-ldot", style: { background: l.color || "var(--faint)" }, title: l.name }))) : null,
      p ? h("span", { class: "jt-pct" }, `${Math.round(p.pct)}%`) : h("span", { class: "jt-prog mono" }, j.name ? j.program : ""),
      jobMenuButton(j, { onDeleted: () => refresh(true) }),
      p ? h("span", { class: "jt-bar" }, h("i", { style: { width: Math.max(2, p.pct) + "%" } })) : null);
    makeJobDraggable(el, j);          // still a binding source for the launcher
    attachJobTip(el, () => jobs.find((x) => x.id === j.id) || j);
    return el;
  }

  list.addEventListener("scroll", () => { if (!rebuilding && list.isConnected) memo.scroll = list.scrollTop; });

  // ── lifecycle ──
  renderHead();
  if (memo.jobs && memo.jobsKey === (memo.all ? "all" : getState().activeProjectId)) renderList();
  else mount(list, h("div", { class: "jt-empty muted" }, "Loading…"));
  loadProjects();
  loadLabels().catch(() => {});
  const p = poller(() => refresh(false), 3000);
  const offEvents = onJobEvent(() => p.kick());
  // A rename from anywhere: patch the cached job and its row in place (no rebuild, so
  // nothing moves), and realign the signature so the next poll does not rebuild either.
  const offRename = onJobRenamed((jid, name) => {
    const j = jobs.find((x) => x.id === jid);
    if (!j) return;
    j.name = name;
    const nm = list.querySelector(`.jt-row[data-id="${jid}"] .jt-name`);
    if (nm) nm.textContent = jobDisplayName(j);
    const pg = list.querySelector(`.jt-row[data-id="${jid}"] .jt-prog`);
    if (pg) pg.textContent = j.name ? j.program : "";
    lastSig = sigOf(jobs);
  });
  let lastPid = getState().activeProjectId;
  const unsub = subscribe(() => {
    loadProjects();
    const pid = getState().activeProjectId;
    if (pid !== lastPid) { lastPid = pid; memo.collapsed.clear(); memo.scroll = 0; lastSig = null; renderHead(); p.kick(); }
  });

  return {
    el: root,
    // Repaint rows from the cached list (a selection changed without a route change).
    repaint: renderList,
    // Swap what selection / usability mean (a reused tree on a new route).
    setOptions(o) { Object.assign(cfg, o); },
    // A detached element forgets its scroll offset; call once it is back in the document.
    onAttached: restoreScroll,
    kick: () => p.kick(),
    jobs: () => jobs,
    get stopped() { return stop; },
    cleanup() { stop = true; p.stop(); offEvents(); unsub(); offRename(); closePop(); hideJobTip(); },
  };
}
