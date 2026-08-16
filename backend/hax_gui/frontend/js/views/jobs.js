// Jobs list with light polling. The detail view streams live instead.
import { h, mount, clear } from "../h.js";
import { icon } from "../icons.js";
import { toastError } from "../toast.js";
import { confirmModal } from "./confirmModal.js";
import { api } from "../api.js";
import { errorBox, skeleton } from "./ui.js";
import { openContextMenu } from "./contextMenu.js";
import { promptModal } from "./promptModal.js";
import { pickProject } from "./pickProject.js";
import { openFileBrowser } from "./filePicker.js";
import { jobLabelChips, loadLabels } from "./labels.js";
import { store } from "../store.js";
import { poller } from "../poller.js";
import { onJobEvent } from "../events.js";

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

export function JobsView() {
  const root = h("div", { class: "view" }, h("h1", { style: { marginBottom: "16px" } }, "Jobs"));
  const filterBar = h("div", { class: "jobs-filters" });
  const body = h("div");
  root.appendChild(filterBar);
  root.appendChild(body);
  mount(body, skeleton("260px"));

  const filters = { name: "", project: "", program: "", status: "", label: "" };
  const nameInput = h("input", { class: "input filter-search", type: "search", placeholder: "Filter by name…",
    oninput: (e) => { filters.name = e.target.value.trim().toLowerCase(); renderBody(); } });
  const projectSel = filterSelect("All projects", (v) => { filters.project = v; renderBody(); });
  const programSel = filterSelect("All programs", (v) => { filters.program = v; renderBody(); });
  const statusSel = filterSelect("All statuses", (v) => { filters.status = v; renderBody(); });
  const labelSel = filterSelect("All labels", (v) => { filters.label = v; renderBody(); });
  const clearBtn = h("button", { class: "btn ghost filter-clear", title: "Clear filters", onclick: () => {
    filters.name = filters.project = filters.program = filters.status = filters.label = "";
    nameInput.value = "";
    projectSel.value = programSel.value = statusSel.value = labelSel.value = "";
    renderBody();
  } }, icon("x", 14), "Clear");
  filterBar.append(
    h("div", { class: "filter-group" }, icon("search", 14), nameInput),
    h("div", { class: "filter-group" }, icon("box", 14), projectSel),
    h("div", { class: "filter-group" }, icon("sliders", 14), programSel),
    h("div", { class: "filter-group" }, icon("list", 14), statusSel),
    h("div", { class: "filter-group" }, icon("more", 14), labelSel),
    clearBtn);

  let allJobs = [];
  let stop = false;
  let lastSig = null;
  let animatedOnce = false;

  // A rename mutates the job object in place (which lives in `allJobs`), so realign
  // the signature to what we already show — otherwise the next 3s poll sees the name
  // change, rebuilds the whole list, and the scroll jumps back to the top.
  const hooks = { afterLocalRename() { lastSig = jobsSignature(allJobs); } };

  function renderBody() {
    let content;
    if (!allJobs.length) content = h("p", { class: "muted" }, "No jobs yet. Pick a program from the right menu.");
    else {
      const jobs = allJobs.filter((j) =>
        (!filters.name || (j.name || "").toLowerCase().includes(filters.name) || j.program.toLowerCase().includes(filters.name)) &&
        (!filters.project || projectKey(j) === filters.project) &&
        (!filters.program || j.program === filters.program) &&
        (!filters.status || j.status === filters.status) &&
        (!filters.label || (j.labels || []).some((l) => l.id === filters.label)));
      content = jobs.length ? groupedList(jobs, hooks) : h("p", { class: "muted" }, "No jobs match the current filters.");
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
      lastSig = null; // next successful poll must repaint over the error box
      if (!stop) mount(body, errorBox(err));
    }
  }
  loadLabels().catch(() => {});
  const p = poller(refresh, 3000);
  const offEvents = onJobEvent(() => p.kick());   // push beats the 3 s tick

  return { el: root, cleanup: () => { stop = true; p.stop(); offEvents(); } };
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
function groupedList(jobs, hooks) {
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
      h("div", null, ...items.map((j, i) => row(j, i, hooks))));
  }));
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
  // Copy this run's recipe into another project: pick a destination, then open the New
  // Run form there preloaded with the program + parameters (data inputs cleared, to be
  // set in the target project). Nothing runs until the user saves/launches it there.
  async function copyToProject() {
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

function row(job, i, hooks) {
  const title = h("span", { class: "prog" }, jobDisplayName(job));
  const line1 = h("div", { class: "jr-line1" },
    badge(job.status),
    staleBadge(job),
    title,
    jobRelTimeEl(job),
    jobMenuButton(job, { onRenamed: (u) => { title.textContent = jobDisplayName(u); hooks?.afterLocalRename?.(); }, onDeleted: () => a.remove() }));

  const dur = jobDuration(job);
  const p = job.progress;
  const progressText = p ? [p.epochs ? `Epoch ${p.epoch}/${p.epochs}` : null, `${Math.round(p.pct)}%`, fmtEta(p.eta_s)]
    .filter(Boolean).join(" · ") : null;
  const segs = [
    job.name ? h("span", { class: "jr-seg prog-sub mono" }, job.program) : null,
    gpuLabel(job.gpu) ? h("span", { class: "jr-seg" }, gpuLabel(job.gpu)) : null,
    dur ? h("span", { class: "jr-seg" }, dur) : null,
    progressText ? h("span", { class: "jr-seg" }, progressText) : null,
  ].filter(Boolean);
  if ((job.labels || []).length) segs.push(jobLabelChips(job, { max: 3 }));
  const line2 = segs.length ? h("div", { class: "jr-line2" }, ...segs) : null;

  const a = h(
    "a",
    { class: "job-row", href: "#/jobs/" + job.id, style: { animationDelay: `${Math.min(i, 30) * 0.02}s` } },
    line1,
    line2,
    p ? h("div", { class: "jr-bar" }, h("div", { class: "jr-fill", style: { width: Math.max(2, p.pct) + "%" } })) : null
  );
  return a;
}
