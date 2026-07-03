// Global app state with a tiny pub/sub. Holds the open project tabs, the active
// project, open TensorBoard tabs, the right-menu program selection, and the
// collapse state of the two side menus. Persisted bits survive reloads.

const PERSIST_KEY = "hax_gui_state";

const state = {
  leftCollapsed: false,
  pmCollapsed: false,   // project-manager (flash-card) rail collapsed?
  rightOpen: true,
  openProjects: [],     // [{id, name}]
  activeProjectId: null,
  openTBs: [],          // [{id, logdir, url}] — server-side, not persisted
  activeTBId: null,
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

  openTB(inst) {
    if (!state.openTBs.some((x) => x.id === inst.id)) state.openTBs.push(inst);
    state.activeTBId = inst.id;
    emit();
  },
  closeTB(id) {
    state.openTBs = state.openTBs.filter((x) => x.id !== id);
    if (state.activeTBId === id) state.activeTBId = state.openTBs.at(-1)?.id ?? null;
    emit();
  },
  setActiveTB(id) { state.activeTBId = id; emit(); },

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

  selectProgram(name) { state.selectedProgram = name; state.rightOpen = true; state.formNonce++; persist(); emit(); },
  clearProgram() { state.selectedProgram = null; state.formNonce++; emit(); },

  // Open the launcher form for `job`'s program, preloaded with its inputs, so the
  // user can edit and re-run it. `jobId` is carried so Save/Run reuse the same job
  // (edit in place) instead of creating a duplicate. Targets the job's project.
  editFrom(job) {
    state.prefill = { program: job.program, values: { ...(job.args || {}) }, gpu: job.gpu || "", jobId: job.id };
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
