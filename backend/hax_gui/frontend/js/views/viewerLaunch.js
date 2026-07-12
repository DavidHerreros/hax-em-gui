// Shared "open the WebGL viewer for a job" helper. For FlexConsensus runs it
// first lets the user choose which consensus output space to view (each space's
// decoder network is auto-detected on the backend). Used by the Viewer view and
// the job detail view so the behaviour is identical.
import { h } from "../h.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { store } from "../store.js";

// Opens the viewer for `job` (calls store.openViewer). Returns the session, or
// null if the user cancelled one of the choosers.
export async function launchViewerForJob(job) {
  let index;
  if (job.program === "flexconsensus") {
    const { spaces } = await api.flexconsensusSpaces(job.id);
    const viewables = spaces.filter((s) => s.viewable);
    if (viewables.length === 1 && spaces.length === 1) index = viewables[0].index;
    else {
      index = await chooseSpace(spaces);
      if (index == null) return null;
    }
  }
  const gpu = await pickViewerGpu();
  if (gpu == null) return null;
  const v = await api.viewerFromJob(job.id, index, gpu);
  store.openViewer({ ...v, jobId: job.id });   // jobId links the tab back to the job for renaming
  return v;
}

// Which device the viewer backend loads the network and decodes maps on.
// Resolves to "0" / "cpu" / "" (leave it to JAX), or null if the user cancelled.
//
// The device is fixed when the viewer subprocess starts and cannot be changed
// afterwards, so it has to be settled here. We only ask when there is a real
// choice to make: a device pinned in Settings wins outright, and a machine with
// fewer than two GPUs has nothing to choose between — so the common case stays
// exactly as many clicks as it was.
async function pickViewerGpu() {
  const [settings, gpus] = await Promise.all([
    api.getSettings().catch(() => ({})),
    api.gpus().catch(() => []),
  ]);
  const pinned = (settings.viewer_gpu || "").trim();
  if (pinned) return pinned;
  if (!gpus || gpus.length < 2) return "";
  return chooseGpu(gpus);
}

// Modal listing the GPUs (with what is already running on them, so a busy card is
// visible before you pick it); resolves to the chosen device, or null.
function chooseGpu(gpus) {
  return new Promise((resolve) => {
    const row = (value, label, detail) => h("button", {
      class: "btn", style: { justifyContent: "space-between", width: "100%" },
      onclick: () => done(value),
    }, h("span", { class: "mono" }, label), h("span", { class: "muted small" }, detail));

    const gb = (mb) => (mb == null ? null : `${(mb / 1024).toFixed(1)}`);
    const rows = gpus.map((g) => {
      const used = gb(g.memory_used_mb), total = gb(g.memory_total_mb);
      const detail = [
        used && total ? `${used}/${total} GB used` : null,
        g.utilization_pct != null ? `${g.utilization_pct}% busy` : null,
      ].filter(Boolean).join(" · ");
      return row(String(g.index), `GPU ${g.index} · ${g.name}`, detail);
    });
    rows.push(row("cpu", "CPU", "no GPU — slower, but never runs out of memory"));

    const modal = h("div", { class: "modal", onclick: (e) => e.stopPropagation() },
      h("div", { class: "modal-head" }, h("span", null, "Choose a device"),
        h("button", { class: "btn-icon", style: { border: "none" }, onclick: () => done(null) }, icon("x", 18))),
      h("div", { class: "prompt-body", style: { display: "flex", flexDirection: "column", gap: "6px", minWidth: "340px" } },
        h("div", { class: "muted small" }, "The viewer loads the network and decodes maps here. Pick one in Settings to stop being asked."),
        ...rows),
      h("div", { class: "modal-foot" }, h("button", { class: "btn ghost", onclick: () => done(null) }, "Cancel")));

    const overlay = h("div", { class: "overlay", onclick: () => done(null) }, modal);
    document.body.appendChild(overlay);
    document.addEventListener("keydown", onKey);
    function onKey(e) { if (e.key === "Escape") done(null); }
    function done(v) { overlay.remove(); document.removeEventListener("keydown", onKey); resolve(v); }
  });
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
