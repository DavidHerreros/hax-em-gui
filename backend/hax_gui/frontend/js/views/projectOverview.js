// The project itself, shown on the Project page when no run is selected: what it
// is (name, folder, age, run count), what it weighs on disk (total, outside-run
// leftovers, SSD scratch, and the largest runs), and the project-level actions
// (rename, storage dashboard, delete). Creating and switching projects lives in
// the job tree's switcher; this is the page you land on after either.
import { h, mount, clear, append } from "../h.js";
import { icon } from "../icons.js";
import { toastError } from "../toast.js";
import { confirmModal } from "./confirmModal.js";
import { promptModal } from "./promptModal.js";
import { api } from "../api.js";
import { store } from "../store.js";
import { errorBox, formatSize } from "./ui.js";
import { openContextMenu } from "./contextMenu.js";
import { openStorageModal } from "./storageModal.js";
import { badge } from "./jobs.js";

// ---- project-level actions, shared with the tree's switcher menu ----

export async function createProjectPrompt() {
  const name = await promptModal({ title: "New project", label: "Project name", placeholder: "e.g. ribosome_80S", okText: "Create" });
  if (name === null || !name.trim()) return null;
  try {
    const p = await api.createProject(name.trim());
    store.openProject(p);
    return p;
  } catch (err) { toastError(err, "Failed to create project."); return null; }
}

export async function renameProjectPrompt(p) {
  const name = await promptModal({ title: "Rename project", label: "Project name", value: p.name, okText: "Rename" });
  if (name === null) return null;
  const trimmed = name.trim();
  if (!trimmed || trimmed === p.name) return null;
  try {
    const updated = await api.renameProject(p.id, trimmed);
    store.renameOpenProject(p.id, updated.name);
    return updated;
  } catch (err) { toastError(err, "Failed to rename project."); return null; }
}

export async function deleteProjectConfirm(p) {
  if (!(await confirmModal({ title: "Delete project?", danger: true, okText: "Delete project",
      message: `“${p.name}” and ALL of its runs — outputs and logs — will be removed.\nThis cannot be undone.` }))) return false;
  try { await api.deleteProject(p.id); store.closeProject(p.id); return true; }
  catch (err) { toastError(err, "Failed to delete project."); return false; }
}

export function projectMenuItems(p, { onChanged } = {}) {
  return [
    { label: "Rename", icon: icon("pencil", 15), onClick: async () => { if (await renameProjectPrompt(p)) onChanged?.(); } },
    { label: "Storage…", icon: icon("box", 15), onClick: () => openStorageModal(p) },
    { label: "Delete", icon: icon("trash", 15), danger: true, onClick: async () => { if (await deleteProjectConfirm(p)) onChanged?.(); } },
  ];
}

// ---- the overview panel ----

export function projectOverview(project) {
  const el = h("div", { class: "po" });
  let stop = false;
  let proj = { ...project };

  const title = h("h1", { class: "po-title" }, proj.name);
  const head = h("div", { class: "po-head" },
    h("div", { class: "po-titleblock" },
      h("div", { class: "jd-titlerow" }, title,
        h("button", { class: "row-menu", title: "Rename project", onclick: async () => { const u = await renameProjectPrompt(proj); if (u) { proj.name = u.name; title.textContent = u.name; } } }, icon("pencil", 14))),
      proj.path ? h("div", { class: "po-path mono", title: proj.path }, proj.path) : null),
    h("div", { class: "jd-actions" },
      h("button", { class: "btn secondary", title: "Where the bytes are, run by run, and what is reclaimable", onclick: () => openStorageModal(proj) }, icon("box", 15), "Storage"),
      h("button", { class: "btn secondary jd-more", title: "More actions",
        onclick: (e) => { e.stopPropagation(); openContextMenu(e.currentTarget, projectMenuItems(proj)); } }, icon("more", 16))));

  const hint = h("div", { class: "po-hint" }, icon("arrowRight", 14), "Pick a program in the right-hand menu to start a run in this project. Runs appear in the tree on the left.");

  const facts = h("dl", { class: "jd-dl po-facts" });
  const usageBox = h("div", { class: "po-usage" });
  el.append(head, hint, h("h2", { class: "section", style: { marginTop: "26px" } }, "Disk usage"), usageBox);
  el.insertBefore(facts, el.children[2]);

  function renderFacts(jobs) {
    clear(facts);
    const row = (k, ...v) => facts.append(h("dt", null, k), h("dd", null, ...v));
    if (proj.created_at) row("Created", new Date(proj.created_at).toLocaleString());
    if (jobs) {
      const by = {};
      for (const j of jobs) by[j.status] = (by[j.status] || 0) + 1;
      const parts = Object.entries(by).sort((a, b) => b[1] - a[1]).map(([s, n]) => `${n} ${s}`);
      row("Runs", jobs.length ? `${jobs.length} · ${parts.join(", ")}` : "none yet");
    }
    row("Project id", h("span", { class: "mono small muted" }, proj.id));
  }

  function renderUsage(u, state) {
    clear(usageBox);
    if (state) { append(usageBox, [h("div", { class: "muted small" }, state)]); return; }
    const runs = u.jobs.length;
    const line = h("div", { class: "po-usage-line" },
      h("strong", { class: "mono" }, formatSize(u.bytes)),
      h("span", { class: "muted" }, ` across ${runs} run${runs === 1 ? "" : "s"}`),
      u.other_bytes ? h("span", { class: "muted", title: "In the project folder but outside any run folder — project metadata, and anything a deleted run left behind" },
        ` · ${formatSize(u.other_bytes)} outside runs`) : null,
      // The SSD scratch is a rebuildable cache of the particle stack shared by the
      // project's runs. It lives outside the project folder, so it is reported beside
      // the total rather than inside it — and it is usually the biggest thing to reclaim.
      u.scratch && u.scratch.exists ? h("span", { class: "muted", title: u.scratch.path }, ` · SSD scratch ${formatSize(u.scratch.bytes)}`) : null,
      h("button", { class: "row-menu", title: "Re-measure disk usage", onclick: () => loadUsage(true) }, icon("refresh", 13)));
    usageBox.appendChild(line);
    const top = [...u.jobs].sort((a, b) => (b.bytes || 0) - (a.bytes || 0)).slice(0, 8);
    if (top.length) {
      const max = top[0].bytes || 1;
      usageBox.appendChild(h("div", { class: "po-runs" }, ...top.map((j) => h("a", { class: "po-run", href: "#/jobs/" + j.id, title: "Open this run" },
        h("span", { class: "po-run-name" }, j.name || j.program || j.id),
        j.status ? badge(j.status) : null,
        h("span", { class: "po-run-bar" }, h("i", { style: { width: Math.max(2, 100 * (j.bytes || 0) / max) + "%" } })),
        h("span", { class: "po-run-size mono" }, formatSize(j.bytes || 0))))));
      if (u.jobs.length > top.length) usageBox.appendChild(h("div", { class: "muted small", style: { marginTop: "6px" } },
        `Largest ${top.length} of ${u.jobs.length} runs — the Storage button lists them all.`));
    }
  }
  async function loadUsage(refresh) {
    renderUsage(null, refresh ? "measuring…" : "reading sizes…");
    try { const u = await api.projectUsage(proj.id, refresh); if (!stop) renderUsage(u); }
    catch (err) { if (!stop) renderUsage(null, "disk usage unavailable: " + (err.detail || err.message)); }
  }

  let lastJobs = null;
  renderFacts(null);
  api.projectJobs(proj.id).then((jobs) => { if (!stop) { lastJobs = jobs; renderFacts(jobs); } }).catch(() => {});
  // The store only carries id/name/path; created_at and the like come from the server.
  api.project(proj.id).then((p) => { if (!stop && p) { proj = { ...proj, ...p }; renderFacts(lastJobs); } }).catch(() => {});
  loadUsage(false);

  return { el, cleanup: () => { stop = true; } };
}

// ---- project chooser modal (Home's getting-started step) ----
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
