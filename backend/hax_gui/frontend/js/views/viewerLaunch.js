// Shared "open the WebGL viewer for a job" helper. For FlexConsensus runs it
// first lets the user choose which consensus output space to view (each space's
// decoder network is auto-detected on the backend). Used by the Viewer view and
// the job detail view so the behaviour is identical.
import { h } from "../h.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { store } from "../store.js";

// Opens the viewer for `job` (calls store.openViewer). Returns the session, or
// null if the user cancelled the FlexConsensus space chooser.
export async function launchViewerForJob(job) {
  if (job.program === "flexconsensus") {
    const { spaces } = await api.flexconsensusSpaces(job.id);
    const viewables = spaces.filter((s) => s.viewable);
    let index;
    if (viewables.length === 1 && spaces.length === 1) index = viewables[0].index;
    else {
      index = await chooseSpace(spaces);
      if (index == null) return null;
    }
    const v = await api.viewerFromJob(job.id, index);
    store.openViewer({ ...v, jobId: job.id });   // jobId links the tab back to the job for renaming
    return v;
  }
  const v = await api.viewerFromJob(job.id);
  store.openViewer({ ...v, jobId: job.id });   // jobId links the tab back to the job for renaming
  return v;
}

// Modal listing the consensus spaces; resolves to the chosen index, or null.
function chooseSpace(spaces) {
  return new Promise((resolve) => {
    const rows = spaces.map((s) => h("button", {
      class: "btn", disabled: !s.viewable,
      style: { justifyContent: "space-between", width: "100%" },
      title: s.viewable ? `decoder: ${s.decoder_program}` : (s.reason || "not viewable"),
      onclick: () => done(s.index),
    },
      h("span", { class: "mono" }, s.name),
      h("span", { class: "muted small" }, s.viewable ? `decoder: ${s.decoder_program}` : (s.reason || ""))));

    const modal = h("div", { class: "modal", onclick: (e) => e.stopPropagation() },
      h("div", { class: "modal-head" }, h("span", null, "Choose a consensus space"),
        h("button", { class: "btn-icon", style: { border: "none" }, onclick: () => done(null) }, icon("x", 18))),
      h("div", { class: "prompt-body", style: { display: "flex", flexDirection: "column", gap: "6px", minWidth: "320px" } },
        h("div", { class: "muted small" }, "Each space is decoded with the network auto-detected from its input method."),
        ...rows),
      h("div", { class: "modal-foot" }, h("button", { class: "btn ghost", onclick: () => done(null) }, "Cancel")));

    const overlay = h("div", { class: "overlay", onclick: () => done(null) }, modal);
    document.body.appendChild(overlay);
    document.addEventListener("keydown", onKey);
    function onKey(e) { if (e.key === "Escape") done(null); }
    function done(v) { overlay.remove(); document.removeEventListener("keydown", onKey); resolve(v); }
  });
}
