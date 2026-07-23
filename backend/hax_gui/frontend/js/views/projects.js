// Projects section: tabs of open projects + a CryoSPARC-style project view
// listing that project's runs (executions).
import { h, mount, clear, append } from "../h.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { store, getState, subscribe } from "../store.js";
import { badge, jobsSignature, gpuLabel, jobDisplayName, jobMenuButton } from "./jobs.js";
import { errorBox, formatSize } from "./ui.js";
import { ProjectManager } from "./projectManager.js";
import { makeJobDraggable } from "./inputBinding.js";
import { jobLabelChips, loadLabels } from "./labels.js";

export function ProjectsView() {
  const root = h("div", { class: "view projects-view" });
  const manager = ProjectManager();
  const workspace = h("div", { class: "project-workspace" });
  root.appendChild(manager.el);
  root.appendChild(workspace);

  const tabbar = h("div", { class: "tabbar" });
  const bodyWrap = h("div", { class: "tab-body" });
  workspace.appendChild(tabbar);
  workspace.appendChild(bodyWrap);

  let bodyCleanup = null;
  let lastKey = null;

  function renderTabs() {
    clear(tabbar);
    const { openProjects, activeProjectId } = getState();
    for (const p of openProjects) {
      const tab = h("div", { class: "tab" + (p.id === activeProjectId ? " active" : ""), onclick: () => store.setActiveProject(p.id) },
        h("span", { class: "tab-label" }, p.name),
        h("button", { class: "tab-close", title: "Close", onclick: (e) => { e.stopPropagation(); store.closeProject(p.id); } }, icon("x", 14)));
      tabbar.appendChild(tab);
    }
    tabbar.appendChild(h("button", { class: "tab-add", title: "Open or create project", onclick: openProjectChooser }, icon("plus", 16)));
  }

  function renderBody() {
    if (bodyCleanup) { bodyCleanup(); bodyCleanup = null; }
    clear(bodyWrap);
    const proj = store.activeProject();
    if (!proj) {
      bodyWrap.appendChild(emptyState());
      return;
    }
    const panel = projectPanel(proj);
    bodyWrap.appendChild(panel.el);
    bodyCleanup = panel.cleanup;
  }

  function emptyState() {
    return h("div", { class: "empty-state" },
      icon("box", 40),
      h("p", { style: { fontSize: "18px", marginTop: "16px" } }, "No project open"),
      h("p", { class: "muted", style: { marginTop: "4px" } }, "Pick a project card on the left to open it here, or create a new one."));
  }

  // Re-render tabs on any change, but rebuild the body only when the set of open
  // projects or the active one changes (so polling/scroll isn't reset needlessly).
  const rerenderAll = () => {
    renderTabs();
    const { openProjects, activeProjectId } = getState();
    const key = activeProjectId + "|" + openProjects.map((p) => p.id).join(",");
    if (key !== lastKey) { lastKey = key; renderBody(); }
  };
  const unsub = subscribe(rerenderAll);
  rerenderAll();

  return { el: root, cleanup: () => { unsub(); manager.cleanup(); if (bodyCleanup) bodyCleanup(); } };
}

function projectPanel(project) {
  const el = h("div", { class: "project-panel" });
  el.appendChild(h("div", { class: "project-header" },
    h("div", null,
      h("h1", null, project.name),
      project.path ? h("p", { class: "subtitle mono small" }, project.path) : null),
    h("div", { class: "ml-auto run-hint-inline" }, icon("arrowRight", 16), "Pick a program in the right menu to run it here")));

  // Disk usage: the project folder's total, plus a size per run once measured. Walked
  // on demand and cached server-side, so opening the project costs one measurement and
  // revisiting it costs nothing; the refresh forces a fresh walk.
  const usageEl = h("div", { class: "project-usage muted small" });
  el.appendChild(usageEl);
  let sizes = new Map();          // job id -> bytes, filled by the usage call

  function renderUsage(u, state) {
    clear(usageEl);
    if (state) { append(usageEl, [icon("box", 13), h("span", null, state)]); return; }
    const runs = u.jobs.length;
    append(usageEl, [                       // h.js's append, which drops the null slots
      icon("box", 13),
      h("strong", { class: "mono" }, formatSize(u.bytes)),
      h("span", null, ` across ${runs} run${runs === 1 ? "" : "s"}`),
      u.other_bytes ? h("span", { title: "In the project folder but outside any run folder — project metadata, and anything a deleted run left behind" },
        ` · ${formatSize(u.other_bytes)} outside runs`) : null,
      // The SSD scratch is a rebuildable cache of the particle stack shared by the
      // project's runs. It lives outside the project folder, so it is reported beside
      // the total rather than inside it — and it is usually the biggest thing to reclaim.
      u.scratch && u.scratch.exists ? h("span", { title: u.scratch.path },
        ` · SSD scratch ${formatSize(u.scratch.bytes)}`) : null,
      h("button", { class: "btn-icon", title: "Re-measure disk usage",
        onclick: () => loadUsage(true) }, icon("refresh", 13))]);
  }
  async function loadUsage(refresh) {
    renderUsage(null, "measuring disk usage…");
    try {
      const u = await api.projectUsage(project.id, refresh);
      if (stop) return;
      sizes = new Map(u.jobs.map((j) => [j.id, j.bytes]));
      renderUsage(u);
      lastSig = null;             // rebuild the run list so the per-run sizes show up
      refreshJobs();
    } catch (err) {
      if (!stop) renderUsage(null, "disk usage unavailable: " + (err.detail || err.message));
    }
  }

  el.appendChild(h("h2", { class: "section", style: { marginTop: "8px" } }, "Executions"));
  const execWrap = h("div");
  el.appendChild(execWrap);

  let stop = false;
  let lastSig = null;
  let animatedOnce = false;
  loadLabels().catch(() => {});
  async function refreshJobs() {
    try {
      const jobs = await api.projectJobs(project.id);
      if (stop) return;
      const sig = jobsSignature(jobs);
      if (sig === lastSig) return; // unchanged -> skip rebuild (avoids flicker)
      lastSig = sig;
      mount(execWrap, jobs.length ? execList(jobs, sizes) : h("p", { class: "muted" }, "No runs yet."));
      // Animate rows only on first paint; later refreshes rebuild the list and
      // re-animating them (e.g. after adding a label) makes it visibly "jump".
      execWrap.classList.toggle("no-anim", animatedOnce);
      animatedOnce = true;
    } catch (err) {
      if (!stop) mount(execWrap, errorBox(err));
    }
  }
  refreshJobs();
  loadUsage(false);
  const timer = setInterval(refreshJobs, 3000);

  return { el, cleanup: () => { stop = true; clearInterval(timer); } };
}

function execList(jobs, sizes) {
  return h("div", null, ...jobs.map((j, i) => {
    const title = h("span", { class: "exec-name" }, jobDisplayName(j));
    const a = h(
      "a", { class: "exec-row", href: "#/jobs/" + j.id, title: "Drag onto a compatible input to chain it", style: { animationDelay: `${i * 0.02}s` } },
      h("span", { class: "exec-grip", title: "Drag as input" }, icon("more", 14)),
      badge(j.status),
      title,
      h("span", { class: "exec-prog mono" }, j.program),
      h("span", { class: "exec-folder mono" }, folderName(j.workdir)),
      // Only once the project's usage has been measured — the job list refreshes every
      // few seconds and must never trigger a folder walk of its own.
      sizes && sizes.has(j.id)
        ? h("span", { class: "exec-size mono", title: "Disk used by this run's folder" }, formatSize(sizes.get(j.id)))
        : null,
      jobLabelChips(j, { editable: true }),
      gpuLabel(j.gpu) ? h("span", { class: "muted small" }, gpuLabel(j.gpu)) : null,
      h("span", { class: "exec-time" }, new Date(j.created_at).toLocaleString()),
      jobMenuButton(j, { onRenamed: (u) => { title.textContent = jobDisplayName(u); }, onDeleted: () => a.remove() }));
    makeJobDraggable(a, j);
    return a;
  }));
}

function folderName(workdir) {
  if (!workdir) return "";
  const parts = workdir.replace(/\/+$/, "").split("/");
  return parts[parts.length - 1] || workdir;
}

// ---- project chooser modal ----
export function openProjectChooser() {
  const list = h("div", { class: "filelist" });
  const nameInput = h("input", { class: "input", placeholder: "New project name" });
  const createBtn = h("button", { class: "btn", onclick: create }, icon("plus", 16), "Create");

  const modal = h("div", { class: "modal", onclick: (e) => e.stopPropagation() },
    h("div", { class: "modal-head" }, h("span", null, "Open or create project"),
      h("button", { class: "btn-icon", style: { border: "none" }, onclick: close }, icon("x", 18))),
    h("div", { class: "modal-path", style: { gap: "8px" } }, nameInput, createBtn),
    list);
  const overlay = h("div", { class: "overlay", onclick: close }, modal);
  document.body.appendChild(overlay);
  document.addEventListener("keydown", onKey);

  function close() { overlay.remove(); document.removeEventListener("keydown", onKey); }
  function onKey(e) { if (e.key === "Escape") close(); }

  function pick(p) { store.openProject(p); close(); }

  async function create() {
    const name = nameInput.value.trim();
    if (!name) return;
    createBtn.disabled = true;
    try { pick(await api.createProject(name)); }
    catch (err) { mount(list, errorBox(err)); createBtn.disabled = false; }
  }

  mount(list, h("p", { class: "muted", style: { padding: "12px" } }, "Loading…"));
  api.projects()
    .then((ps) => {
      if (!ps.length) return mount(list, h("p", { class: "muted", style: { padding: "12px" } }, "No projects yet. Create one above."));
      mount(list, h("div", null, ...ps.map((p) => h(
        "button", { class: "entry", onclick: () => pick(p) },
        icon("box", 16), h("span", { class: "nm" }, p.name),
        h("span", { class: "sz mono" }, p.id)))));
    })
    .catch((err) => mount(list, errorBox(err)));
}
