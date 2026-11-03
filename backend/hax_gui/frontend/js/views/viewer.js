// 3D viewer launcher page: the job tree on the left with viewable runs bright
// (a network checkpoint plus latents, or a finished FlexConsensus run), and the
// sessions on the right. The viewers themselves (and mrc / metadata viewers)
// render in the shared floating window (viewerWindow.js), so clicking a run pops
// it over the page as a movable/resizable tab rather than living on this route.
import { h, mount, clear } from "../h.js";
import { icon } from "../icons.js";
import { store, getState, subscribe } from "../store.js";
import { errorBox } from "./ui.js";
import { launchViewerForJob } from "./viewerLaunch.js";
import { loadProgramGroups, isPredictLatents } from "../programGroups.js";
import { jobDisplayName } from "./jobs.js";
import { jobTree } from "./jobTree.js";
import { jobWorkspace, stageEmpty } from "./jobWorkspace.js";

// A job is viewable if it has a network checkpoint plus latents — either a
// latents input, or a finished predict run whose output provides them (which
// programs is derived from the backend, see programGroups.js). A finished
// FlexConsensus run is viewable too: the user picks a consensus space and its
// decoder is auto-detected.
export function viewable(job) {
  const a = job.args || {};
  if (job.program === "flexconsensus") return job.status === "succeeded";
  const reload = a.pickled_nn || a.reload;
  if (!reload) return false;
  if (a.z_space || a.latents || a.latents_file) return true;
  return isPredictLatents(job.program) && job.status === "succeeded";
}

export function ViewerView() {
  let launching = null;   // job id while its session is starting
  const tree = jobTree({
    isSelected: (j) => getState().openViewers.some((v) => v.jobId === j.id && v.id === getState().activeViewerId),
    onSelect: (j) => (viewable(j) ? launch(j) : explain(j)),
    usable: viewable,
    emptyText: "No runs in this project yet.",
  });

  const note = h("div");
  const sessions = h("div");
  const stage = h("div", { class: "jw-stage" }, note, sessions);
  const root = h("div", { class: "view viewer-view" }, jobWorkspace(tree.el, stage));

  function explain(j) {
    mount(note, h("div", { class: "jw-note" },
      h("strong", null, jobDisplayName(j)),
      j.status === "saved" ? " has not run yet." :
      j.program === "flexconsensus" ? " opens here once it has finished." :
      " has no landscape to show: the viewer needs a network checkpoint plus latents (a predict run, or a run given a latents file)."));
  }

  async function launch(job) {
    if (launching) return;
    launching = job.id;
    store.setLastJob(job.id);
    clear(note);
    mount(note, h("div", { class: "muted small", style: { marginBottom: "12px" } }, "Starting the viewer for ", h("strong", null, jobDisplayName(job)), "…"));
    try {
      await launchViewerForJob(job);   // opens in the floating window (handles the FlexConsensus space chooser)
      clear(note);
    } catch (err) {
      mount(note, errorBox(err));
    } finally { launching = null; tree.repaint(); }
  }

  // The open sessions, so the page itself says what is running even when the
  // floating window is minimised or popped out.
  function renderSessions() {
    const { openViewers, activeViewerId } = getState();
    clear(sessions);
    if (!openViewers.length) {
      sessions.appendChild(stageEmpty("Select a run", "Viewable runs are the bright ones in the tree. The viewer opens in a floating window you can move, resize or pop out."));
      return;
    }
    sessions.appendChild(h("h2", { class: "section" }, "Open sessions"));
    for (const v of openViewers) {
      sessions.appendChild(h("button", { class: "jw-session" + (v.id === activeViewerId ? " on" : ""),
        onclick: () => store.setActiveViewer(v.id) },
        icon("orbit", 14), h("span", { class: "jw-session-title" }, v.title), v.subtitle ? h("span", { class: "mono faint small" }, v.subtitle) : null,
        h("button", { class: "tab-close", title: "Close", onclick: (e) => { e.stopPropagation(); store.closeViewer(v.id); } }, icon("x", 14))));
    }
    tree.repaint();
  }

  const unsub = subscribe(renderSessions);
  loadProgramGroups().then(() => tree.repaint()).catch(() => {});
  renderSessions();
  return { el: root, cleanup() { unsub(); tree.cleanup(); } };
}
