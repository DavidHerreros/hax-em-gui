// Global app state with a tiny pub/sub. Holds the open project tabs, the active
// project, open metrics tabs, the right-menu program selection, and the
// collapse state of the two side menus. Persisted bits survive reloads.

const PERSIST_KEY = "hax_gui_state";

const state = {
  leftCollapsed: false,
  pmCollapsed: false,   // project-manager (flash-card) rail collapsed?
  rightOpen: true,
  openProjects: [],     // [{id, name}]
  activeProjectId: null,
  openMetrics: [],      // [{id, logdir, title, jobId}] — native metrics tabs, not persisted
  activeMetricsId: null,
  openViewers: [],      // [{id, title, url}] — WebGL viewer sessions, not persisted
  activeViewerId: null,
  selectedProgram: null, // program name shown in the right menu (null = catalog list)
  prefill: null,        // { program, values, gpu } to preload a form from a past job
  formNonce: 0,         // bumped whenever the launcher form should rebuild
};

(function restore() {
  try {
    const saved = JSON.parse(localStorage.getItem(PERSIST_KEY) || "{}");
    Object.assign(state, {
      leftCollapsed: !!saved.leftCollapsed,
      pmCollapsed: !!saved.pmCollapsed,
      rightOpen: saved.rightOpen !== false,
      openProjects: Array.isArray(saved.openProjects) ? saved.openProjects : [],
      activeProjectId: saved.activeProjectId ?? null,
    });
  } catch { /* ignore */ }
})();

function persist() {
  localStorage.setItem(PERSIST_KEY, JSON.stringify({
    leftCollapsed: state.leftCollapsed,
    pmCollapsed: state.pmCollapsed,
    rightOpen: state.rightOpen,
    openProjects: state.openProjects,
    activeProjectId: state.activeProjectId,
  }));
}

const listeners = new Set();
function emit() { listeners.forEach((fn) => fn(state)); }

export function subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn); }
export function getState() { return state; }

export const store = {
  toggleLeft() { state.leftCollapsed = !state.leftCollapsed; persist(); emit(); },
  togglePM() { state.pmCollapsed = !state.pmCollapsed; persist(); emit(); },
  toggleRight() { state.rightOpen = !state.rightOpen; persist(); emit(); },
  setRightOpen(v) { state.rightOpen = v; persist(); emit(); },

  openProject(p) {
    if (!state.openProjects.some((x) => x.id === p.id)) state.openProjects.push({ id: p.id, name: p.name, path: p.path });
    state.activeProjectId = p.id;
    persist(); emit();
  },
  closeProject(id) {
    state.openProjects = state.openProjects.filter((x) => x.id !== id);
    if (state.activeProjectId === id) state.activeProjectId = state.openProjects.at(-1)?.id ?? null;
    persist(); emit();
  },
  setActiveProject(id) { state.activeProjectId = id; persist(); emit(); },
  // Reflect a rename in the open tab (and persisted state) without reopening.
  renameOpenProject(id, name) {
    const p = state.openProjects.find((x) => x.id === id);
    if (p) { p.name = name; persist(); }
    emit();
  },
  activeProject() { return state.openProjects.find((x) => x.id === state.activeProjectId) ?? null; },

  // Metrics tabs are keyed by log directory, so opening the same run twice focuses
  // the tab that is already there instead of stacking duplicates.
  openMetrics({ logdir, title, jobId }) {
    const existing = state.openMetrics.find((x) => x.id === logdir);
    if (existing) { if (jobId) existing.jobId = jobId; }
    else state.openMetrics.push({ id: logdir, logdir, title: title || logdir, jobId: jobId || null });
    state.activeMetricsId = logdir;
    emit();
  },
  closeMetrics(id) {
    state.openMetrics = state.openMetrics.filter((x) => x.id !== id);
    if (state.activeMetricsId === id) state.activeMetricsId = state.openMetrics.at(-1)?.id ?? null;
    emit();
  },
  setActiveMetrics(id) { state.activeMetricsId = id; emit(); },

  openViewer(inst) {
    if (!state.openViewers.some((x) => x.id === inst.id)) state.openViewers.push(inst);
    state.activeViewerId = inst.id;
    emit();
  },
  closeViewer(id) {
    state.openViewers = state.openViewers.filter((x) => x.id !== id);
    if (state.activeViewerId === id) state.activeViewerId = state.openViewers.at(-1)?.id ?? null;
    emit();
  },
  setActiveViewer(id) { state.activeViewerId = id; emit(); },
  // Reflect a job rename in any open viewer tab launched from that job. Keeps a
  // " · suffix" (e.g. a FlexConsensus space) intact, replacing only the name part.
  renameViewerForJob(jobId, name) {
    let changed = false;
    for (const v of state.openViewers) {
      if (v.jobId !== jobId) continue;
      const sep = (v.title || "").indexOf(" · ");
      v.title = sep >= 0 ? name + v.title.slice(sep) : name;
      changed = true;
    }
    if (changed) emit();
  },

  selectProgram(name) { state.selectedProgram = name; state.rightOpen = true; state.formNonce++; persist(); emit(); },
  clearProgram() { state.selectedProgram = null; state.formNonce++; emit(); },

  // Open the launcher form for `job`'s program, preloaded with its inputs, so the
  // user can edit and re-run it. `jobId` is carried so Save/Run reuse the same job
  // (edit in place) instead of creating a duplicate. Targets the job's project.
  // `opts.resume` turns this into "edit & continue": Save/Run reuse the same run
  // folder (not wiped) so the program resumes from the checkpoint it left there, with
  // the edited args. Plain edit (no opts) replaces the run for a fresh start.
  editFrom(job, opts = {}) {
    state.prefill = { program: job.program, values: { ...(job.args || {}) }, gpu: job.gpu || "",
                      jobId: job.id, resume: !!opts.resume };
    state.selectedProgram = job.program;
    state.rightOpen = true;
    state.formNonce++;
    if (job.project_id) this.openProject({ id: job.project_id, name: job.project_name || job.project_id });
    else { persist(); emit(); }
  },
  clearPrefill() { state.prefill = null; }, // consumed by the launcher; no emit

  // Open the launcher for `program` preloaded with `values` (an assistant
  // proposal). No jobId → a NEW job: the user completes any required fields and
  // launches it themselves. Nothing runs from this call.
  prefillProgram({ program, values, projectId, projectName }) {
    state.prefill = { program, values: { ...(values || {}) }, gpu: "" };
    state.selectedProgram = program;
    state.rightOpen = true;
    state.formNonce++;
    if (projectId) this.openProject({ id: projectId, name: projectName || projectId });
    else { persist(); emit(); }
  },
};
