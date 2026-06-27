// Live job view: streams the log over WebSocket into a terminal renderer
// (ANSI colors + progress lines), with cancel / rerun / delete actions.
import { h, mount, clear } from "../h.js";
import { icon } from "../icons.js";
import { api, subscribeLog } from "../api.js";
import { navigate, reload } from "../router.js";
import { store } from "../store.js";
import { errorBox } from "./ui.js";
import { badge, jobDisplayName } from "./jobs.js";
import { promptModal } from "./promptModal.js";
import { createTerminal } from "./terminal.js";

export function JobDetailView(params) {
  const id = params.id;
  const root = h("div", { class: "view" });

  const head = h("div", { class: "row", style: { marginBottom: "8px" } });
  const progress = h("div", { class: "progress" });
  const cmd = h("pre", { class: "cmdline" }, "…");
  const wd = h("div", { class: "muted small", style: { margin: "-8px 0 16px" } });
  const actionErr = h("div", { style: { marginBottom: "12px" } });
  const consoleEl = h("div", { class: "console" });

  root.appendChild(h("a", { class: "back", href: "#/jobs" }, icon("arrowLeft", 16), "Jobs"));
  root.appendChild(head);
  root.appendChild(progress);
  root.appendChild(cmd);
  root.appendChild(wd);
  root.appendChild(actionErr);
  root.appendChild(consoleEl);

  const term = createTerminal(consoleEl);
  let status = "pending";
  let job = null;

  function renderHead() {
    clear(head);
    const isNamed = job && job.name;
    head.appendChild(h("h1", { style: { fontSize: "20px", color: "var(--accent)", fontFamily: "JetBrains Mono, monospace" } }, job ? jobDisplayName(job) : "…"));
    if (job) head.appendChild(h("button", { class: "row-menu", title: "Rename run", onclick: rename }, icon("pencil", 15)));
    if (isNamed) head.appendChild(h("span", { class: "muted mono", style: { fontSize: "12px" } }, job.program));
    head.appendChild(badge(status));
    if (job && job.project_name) head.appendChild(h("span", { class: "proj-chip" }, icon("box", 12), job.project_name));
    const running = status === "running" || status === "pending";
    if (job && job.return_code != null && !running) {
      head.appendChild(h("span", { class: "muted", style: { fontSize: "12px" } }, "exit " + job.return_code));
    }
    if (job && status === "saved") {
      head.appendChild(h("div", { class: "row", style: { marginLeft: "auto", gap: "8px" } },
        h("button", { class: "btn", onclick: startSaved, title: "Launch this saved job now" }, icon("play", 15), "Run"),
        viewable(job) ? viewerBtn() : null,
        h("button", { class: "btn secondary", onclick: editRerun, title: "Open the form preloaded with these inputs to edit and save/launch a new run" }, icon("copy", 15), "Edit"),
        h("button", { class: "btn ghost", style: { color: "var(--err)" }, onclick: del }, icon("trash", 15), "Delete")));
    } else if (running && job) {
      head.appendChild(h("div", { class: "row", style: { marginLeft: "auto", gap: "8px" } },
        viewable(job) ? viewerBtn() : null,
        h("button", { class: "btn ghost", style: { color: "var(--warn)" }, onclick: () => api.cancelJob(id).catch(() => {}) }, icon("ban", 15), "Cancel")));
    } else if (job) {
      head.appendChild(h("div", { class: "row", style: { marginLeft: "auto", gap: "8px" } },
        viewable(job) ? viewerBtn() : null,
        h("button", { class: "btn secondary", onclick: editRerun, title: "Open the form preloaded with these inputs to edit and launch a new run" }, icon("copy", 15), "Edit & rerun"),
        h("button", { class: "btn secondary", onclick: rerun, title: "Re-launch immediately with the same inputs" }, icon("play", 15), "Rerun"),
        h("button", { class: "btn ghost", style: { color: "var(--err)" }, onclick: del }, icon("trash", 15), "Delete")));
    }
    renderProgress(running);
  }

  function editRerun() {
    store.editFrom(job);
    navigate("/projects");
  }

  function viewable(j) {
    const a = (j && j.args) || {};
    return (a.z_space || a.latents || a.latents_file) && (a.pickled_nn || a.reload);
  }
  async function openViewer() {
    clear(actionErr);
    try { store.openViewer(await api.viewerFromJob(id)); navigate("/viewer"); }
    catch (err) { mount(actionErr, errorBox(err)); }
  }
  const viewerBtn = () => h("button", { class: "btn secondary", onclick: openViewer, title: "Explore this run's conformational landscape in the embedded WebGL viewer" }, icon("box", 15), "3D viewer");

  async function startSaved() {
    clear(actionErr);
    try {
      const j = await api.startJob(id);
      job = j; status = j.status;
      term.reset();
      connectLog();      // re-open the stream now that the job is running
      renderHead();
    } catch (err) { mount(actionErr, errorBox(err)); }
  }

  async function rename() {
    if (!job) return;
    const name = await promptModal({ title: "Rename run", label: "Run name", value: job.name || "", placeholder: job.program, okText: "Rename" });
    if (name === null) return;
    try { const u = await api.renameJob(id, name); job = { ...job, name: u.name }; renderHead(); }
    catch (err) { mount(actionErr, errorBox(err)); }
  }

  async function rerun() {
    clear(actionErr);
    // Rerun replaces this job in place (same id), so re-render the current route
    // to reset the log/state instead of navigating (the hash is unchanged).
    try { await api.rerunJob(id); reload(); }
    catch (err) { mount(actionErr, errorBox(err)); }
  }

  async function del() {
    if (!window.confirm("Delete this job and its run folder (outputs + log)? This cannot be undone.")) return;
    clear(actionErr);
    try { await api.deleteJob(id); navigate("/jobs"); }
    catch (err) { mount(actionErr, errorBox(err)); }
  }

  function renderProgress(running) {
    clear(progress);
    if (status === "saved") return; // not started yet — nothing to show
    if (running) {
      progress.appendChild(h("div", { class: "sweep" }));
    } else {
      const kind = status === "succeeded" ? "succeeded" : status === "failed" ? "failed" : "other";
      progress.appendChild(h("div", { class: "bar " + kind, style: { width: "100%" } }));
    }
  }

  // Fetch job metadata (command + program name), then stream the log.
  api.job(id)
    .then((j) => {
      job = j; status = j.status;
      cmd.textContent = j.command.join(" ");
      if (j.cwd) wd.textContent = "Working directory: " + j.cwd;
      renderHead();
    })
    .catch((err) => mount(root, errorBox(err)));

  let closeLog = null;
  function connectLog() {
    if (closeLog) closeLog();
    closeLog = subscribeLog(id, (m) => {
      if (m.type === "snapshot") { status = m.status; term.reset(); if (m.data) term.write(m.data); renderHead(); }
      else if (m.type === "chunk") { term.write(m.data); }
      else if (m.type === "status") {
        status = m.status;
        if (m.status !== "running") api.job(id).then((j) => { job = j; renderHead(); }).catch(() => renderHead());
        else renderHead();
      }
    });
  }
  connectLog();

  return { el: root, cleanup: () => { if (closeLog) closeLog(); } };
}
