// Project page: the shared job tree on the left; on the right the selected run
// (/jobs/:id) or, with none selected, the project itself (/project). Row helpers
// used by every run list live here too.
import { h, mount } from "../h.js";
import { icon } from "../icons.js";
import { toastError } from "../toast.js";
import { confirmModal } from "./confirmModal.js";
import { api } from "../api.js";
import { openContextMenu } from "./contextMenu.js";
import { promptModal } from "./promptModal.js";
import { pickProject } from "./pickProject.js";
import { openFileBrowser } from "./filePicker.js";
import { store, getState, subscribe } from "../store.js";
import { navigate } from "../router.js";
import { jobTree } from "./jobTree.js";
import { jobWorkspace, stageEmpty } from "./jobWorkspace.js";
import { jobDetailPanel } from "./jobDetail.js";
import { projectOverview, createProjectPrompt } from "./projectOverview.js";

export function badge(status) {
  const animated = status === "running" || status === "pending";
  return h("span", { class: "badge " + status }, animated ? h("span", { class: "dot" }) : null, status);
}

// Sits next to the status badge on a run whose inputs were re-made after it finished:
// its outputs are real, they just no longer correspond to anything that exists. Returns
// null for a healthy run so callers can drop it straight into a child list.
export function staleBadge(job) {
  if (!job || !job.stale) return null;
  return h("span", { class: "badge stale", title: "A run this one depends on has been re-run since. These outputs came from the previous version of that input — re-run this step to bring them up to date." },
    icon("refresh", 11), "stale");
}

// ── when a run happened ──
// `created_at` is set once, when the row is first inserted, and a rerun deliberately
// keeps it (jobs.py: that is what holds the row's place in these lists). So on anything
// that has been re-run it names a run that no longer exists — which is why these lists
// used to claim a job "ran" days before the outputs sitting in its folder were written.
// The run's own clock is started_at / finished_at, reset by every launch path.
//
// Returns { label, stamp } naming the timestamp that actually describes what you are
// looking at, or nulls when there is nothing to show.
export function jobTime(job) {
  if (!job) return { label: null, stamp: null };
  if (job.active) return job.started_at ? { label: "started", stamp: job.started_at } : { label: "queued", stamp: job.created_at };
  if (job.finished_at) return { label: "finished", stamp: job.finished_at };
  // Started but no end recorded: the machine went down under it, or it is a saved row
  // that has never run at all (both timestamps cleared on replace).
  if (job.started_at) return { label: "started", stamp: job.started_at };
  return { label: "created", stamp: job.created_at };
}

// How long the run took, once it has both ends. Null while it is still going: these
// lists only rebuild when something changes, so a "2h 14m so far" would sit there
// frozen and quietly lie.
export function jobDuration(job) {
  if (!job || job.active || !job.started_at || !job.finished_at) return null;
  const ms = new Date(job.finished_at) - new Date(job.started_at);
  if (!(ms >= 0)) return null;                       // clock skew, or a malformed stamp
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s`;
  const hrs = Math.floor(m / 60);
  if (hrs < 24) return `${hrs}h ${m % 60}m`;
  return `${Math.floor(hrs / 24)}d ${hrs % 24}h`;
}

// The row shows one time; the tooltip shows the whole life of the run, so "why does this
// say finished 10:04 when I launched it yesterday" is answerable without leaving the list.
export function jobTimeTitle(job) {
  if (!job) return "";
  const at = (t) => (t ? new Date(t).toLocaleString() : "—");
  const lines = [`Created: ${at(job.created_at)}`, `Started: ${at(job.started_at)}`];
  if (!job.active) lines.push(`Finished: ${at(job.finished_at)}`);
  const dur = jobDuration(job);
  if (dur) lines.push(`Duration: ${dur}`);
  if (job.created_at && job.started_at && job.created_at !== job.started_at) {
    lines.push("", "Created is when this step was first added; a rerun keeps it so the row stays put.");
  }
  return lines.join("\n");
}

// "~2h 10m left" from a seconds estimate. Coarse on purpose: an ETA that ticks
// every second promises a precision the estimate does not have.
export function fmtEta(s) {
  if (!(s > 0)) return null;
  if (s < 90) return "~1m left";
  const m = Math.round(s / 60);
  if (m < 90) return `~${m}m left`;
  const hr = Math.floor(m / 60);
  if (hr < 48) return `~${hr}h ${m % 60}m left`;
  return `~${Math.round(hr / 24)}d left`;
}

// Coarse "how long ago" for the jobs-list card, deliberately imprecise like fmtEta:
// a live clock next to a 3s poll would tick out of sync with the row's actual age.
export function relTime(stamp) {
  if (!stamp) return null;
  const s = (Date.now() - new Date(stamp)) / 1000;
  if (s < 90) return "just now";
  const m = Math.round(s / 60);
  if (m < 90) return `${m}m ago`;
  const hr = Math.round(m / 60);
  if (hr < 48) return `${hr}h ago`;
  const d = Math.round(hr / 24);
  if (d < 14) return `${d}d ago`;
  return new Date(stamp).toLocaleDateString();
}

// Line-1 right-hand time for the jobs-list card: relative, with the same full-history
// tooltip as jobTimeEl. "finished" drops its own label since the relative stamp alone
// already reads as past tense.
export function jobRelTimeEl(job) {
  const { label, stamp } = jobTime(job);
  const rel = relTime(stamp);
  return h("span", { class: "jr-when", title: jobTimeTitle(job) },
    rel ? (label === "finished" ? rel : `${label} ${rel}`) : "—");
}

// The one-line form used in every run list: "finished 25/07/2026, 14:32:10 · 2h 14m".
export function jobTimeEl(job, cls = "time") {
  const { label, stamp } = jobTime(job);
  const dur = jobDuration(job);
  return h("span", { class: cls, title: jobTimeTitle(job) },
    stamp ? `${label} ${new Date(stamp).toLocaleString()}` : "—",
    dur ? h("span", { class: "time-dur" }, " · " + dur) : null);
}

export function JobsView(params = {}) {
  const id = params.id || null;
  if (id) store.setLastJob(id);
  store.setProjectRoute(id ? `/jobs/${id}` : "/project");   // the nav's "Project" link returns here
  const tree = jobTree({
    isSelected: (j) => j.id === id,
    onSelect: (j) => { if (j.id !== id) navigate(`/jobs/${j.id}`); },
  });
  const stage = h("div", { class: "jw-stage" });
  let detail = null;
  function showProject() {
    detail?.cleanup?.();
    const proj = store.activeProject();
    detail = proj ? projectOverview(proj) : null;
    mount(stage, detail ? detail.el : stageEmpty("No project open",
      h("span", null, "Choose one from the switcher on the left, or ",
        h("button", { class: "jt-link", onclick: () => createProjectPrompt() }, "create a project"), ".")));
  }
  if (id) { detail = jobDetailPanel(id); stage.appendChild(detail.el); }
  else showProject();
  // With no run selected the page is the project: a project switch repaints the stage
  // in place (the tree follows the store on its own).
  let lastPid = getState().activeProjectId;
  const unsub = subscribe(() => { const pid = getState().activeProjectId; if (!id && pid !== lastPid) { lastPid = pid; showProject(); } });
  const el = h("div", { class: "view" }, jobWorkspace(tree.el, stage));
  return { el, cleanup: () => { unsub(); tree.cleanup(); detail?.cleanup?.(); } };
}

// Copy this run's recipe into another project: pick a destination, then open the New
// Run form there preloaded with the program + parameters (data inputs cleared, to be
// set in the target project). Nothing runs until the user saves/launches it there.
export async function copyJobToProject(job) {
  const target = await pickProject({
    title: "Copy run to project", okText: "copy here", excludeId: job.project_id,
    hint: "Opens the New Run form in the chosen project, preloaded with this run's program and settings. Data inputs are cleared for you to set there.",
  });
  if (!target) return;
  try {
    const pf = await api.jobCopyPrefill(job.id);
    store.prefillProgram({ program: pf.program, values: pf.values, gpu: pf.gpu, name: pf.name,
                           projectId: target.id, projectName: target.name });
  } catch (err) {
    toastError(err, "Could not copy this run.");
  }
}





// A change signature for a job list: re-render only when ids/statuses/names change.
// Signature to decide whether a background refresh needs to rebuild the list.
// Labels are deliberately excluded: they're edited inline and applied to the job
// object locally (the chip updates immediately), so a label change shouldn't force
// a full list rebuild — which otherwise causes a visible ~refresh-tick "jump".
export function jobsSignature(jobs) {
  // `stale` is in here because it changes with no status change of its own: a run goes
  // stale when a *different* run is re-made, and without it the badge would only appear
  // the next time something else about the list moved.
  // Whole-percent progress is in the signature so a training run's bar advances
  // on the poll, without sub-percent noise rebuilding the list constantly.
  // The card's line 1 now prints a relative time ("2h ago"); it has to be in here too,
  // or a poll that changes nothing else lets that label go stale until something else moves.
  return jobs.map((j) => `${j.id}:${j.status}:${j.stale ? "s" : ""}:${j.name || ""}:` +
                         `${j.progress ? Math.round(j.progress.pct) : ""}:${relTime(jobTime(j).stamp)}`).join("|");
}

// What to show as a run's title: its custom name, falling back to the program.
export function jobDisplayName(job) {
  return job.name || job.program;
}

// Two-click compare across the whole list (works across projects and pages):
// "Select for compare" marks a run, "Compare with <it>" on any other run opens
// the diff view. One pending mark, module-wide — sweeps are the main customer.
let comparePick = null;   // { id, label }

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
      { label: "Copy to project…", icon: icon("copy", 15), onClick: copyToProject },
    );
    if (comparePick && comparePick.id !== job.id) {
      items.push({ label: `Compare with ${comparePick.label}`, icon: icon("grid", 15),
        onClick: () => { const other = comparePick.id; comparePick = null; location.hash = `#/compare/${other}/${job.id}`; } });
    }
    items.push(comparePick && comparePick.id === job.id
      ? { label: "Cancel compare", icon: icon("x", 15), onClick: () => { comparePick = null; } }
      : { label: "Select for compare", icon: icon("grid", 15),
          onClick: () => { comparePick = { id: job.id, label: jobDisplayName(job) }; } });
    // Straight into the run folder, without going through the detail view first.
    // A saved-but-never-run job has no folder yet.
    if (job.workdir) items.push({ label: "Output folder…", icon: icon("folderOpen", 15), onClick: () => openFileBrowser(job.workdir) });
    if (!job.active) items.push({ label: "Delete", icon: icon("trash", 15), danger: true, onClick: del });
    openContextMenu(anchor, items);
  }
  const copyToProject = () => copyJobToProject(job);
  async function run() {
    try { await api.startJob(job.id); location.hash = "#/jobs/" + job.id; }
    catch (err) { toastError(err, "Failed to start job."); }
  }
  async function rename() {
    const name = await promptModal({ title: "Rename run", label: "Run name", value: job.name || "", placeholder: job.program, okText: "Rename" });
    if (name === null) return;
    try { const u = await api.renameJob(job.id, name); job.name = u.name; store.renameViewerForJob(job.id, u.name); if (onRenamed) onRenamed(u); } catch { /* ignore */ }
  }
  async function del() {
    if (!(await confirmModal({ title: "Delete run?", danger: true, okText: "Delete run",
        message: "This job and its run folder — outputs and log — will be removed.\nThis cannot be undone." }))) return;
    try { await api.deleteJob(job.id); if (onDeleted) onDeleted(); } catch { /* ignore */ }
  }
  return btn;
}

// Display label for a job's device, or null if unspecified.
export function gpuLabel(gpu) {
  if (gpu == null || gpu === "") return null;
  return gpu === "cpu" ? "CPU" : "GPU " + gpu;
}

