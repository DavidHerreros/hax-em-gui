// 3D viewer launcher page: lists runs with a conformational landscape and opens
// them. The viewers themselves (and mrc / metadata viewers) render in the shared
// floating window (viewerWindow.js), so opening one pops it over the current page
// as a movable/resizable tab rather than living on this route.
import { h, mount } from "../h.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { errorBox } from "./ui.js";
import { jobLabelChips } from "./labels.js";
import { launchViewerForJob } from "./viewerLaunch.js";
import { loadProgramGroups, isPredictLatents } from "../programGroups.js";

// A job is viewable if it has a network checkpoint plus latents — either a
// latents input, or a finished predict run whose output provides them (which
// programs is derived from the backend, see programGroups.js). A finished
// FlexConsensus run is viewable too: the user picks a consensus space and its
// decoder is auto-detected.
function viewable(job) {
  const a = job.args || {};
  if (job.program === "flexconsensus") return job.status === "succeeded" ? true : null;
  const reload = a.pickled_nn || a.reload;
  if (!reload) return null;
  if (a.z_space || a.latents || a.latents_file) return true;
  return isPredictLatents(job.program) && job.status === "succeeded" ? true : null;
}

export function ViewerView() {
  const root = h("div", { class: "view viewer-view" });
  root.appendChild(h("header", { class: "page-head" }, h("h1", null, "3D viewer"),
    h("p", { class: "subtitle" }, "Explore a conformational landscape in WebGL: pick a state, see its decoded map. Sessions open in a floating window you can move and resize.")));

  const runsBox = h("div");
  root.appendChild(runsBox);

  function loadRuns() {
    mount(runsBox, h("p", { class: "muted small", style: { margin: "8px 0" } }, "Loading runs…"));
    Promise.all([loadProgramGroups().catch(() => {}), api.jobs()])
      .then(([, jobs]) => mount(runsBox, renderRuns(jobs.filter((j) => viewable(j)))))
      .catch((err) => mount(runsBox, errorBox(err)));
  }

  function renderRuns(jobs) {
    const header = h("div", { class: "row section-head" },
      h("h2", { class: "section", style: { margin: 0 } }, "Viewable runs"),
      h("button", { class: "btn-icon", title: "Refresh", style: { marginLeft: "auto" }, onclick: loadRuns }, icon("refresh", 15)));
    if (!jobs.length) {
      return h("div", null, header,
        h("p", { class: "muted small" }, "None yet. Open the viewer from an ", h("span", { class: "mono" }, "annotate_space"), " run, or from a finished network ", h("span", { class: "mono" }, "predict"), " run (its latents + network), then refresh."));
    }
    return h("div", null, header, ...jobs.map((j) => {
      const btn = h("button", { class: "run-row", onclick: () => launch(j, btn) },
        h("span", { class: "run-prog mono" }, j.name || j.program),
        j.project_name ? h("span", { class: "proj-chip" }, icon("box", 12), j.project_name) : null,
        h("span", { class: "run-folder mono" }, j.program),
        (j.labels && j.labels.length) ? jobLabelChips(j) : null,
        h("span", { class: "run-time" }, new Date(j.created_at).toLocaleString()),
        h("span", { class: "run-open" }, icon("box", 14), "Open viewer"));
      return btn;
    }));
  }

  async function launch(job, btn) {
    if (btn) { btn.disabled = true; btn.lastChild.textContent = "Starting…"; }
    try {
      await launchViewerForJob(job);   // opens in the floating window (handles the FlexConsensus space chooser)
    } catch (err) {
      mount(runsBox, errorBox(err));
    } finally {
      if (btn) { btn.disabled = false; btn.lastChild.textContent = "Open viewer"; }
    }
  }

  loadRuns();
  return { el: root };
}
