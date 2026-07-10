// Metrics: pick a run that has TensorBoard events (auto-listed) or enter a log
// directory manually, then view it. Multiple sessions open as tabs; iframes stay
// mounted across tab switches so each keeps its state.
import { h, mount, clear } from "../h.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { store, getState, subscribe } from "../store.js";
import { errorBox } from "./ui.js";
import { openFilePicker } from "./filePicker.js";
import { jobLabelChips } from "./labels.js";

export function MetricsView() {
  const root = h("div", { class: "view metrics-view" });

  let logdir = "";
  const input = h("input", { class: "input", placeholder: "server log directory (manual)", oninput: (e) => (logdir = e.target.value) });
  const browse = h("button", { class: "btn-icon", title: "Browse server", onclick: () => openFilePicker("directory", (p) => { input.value = p; logdir = p; }, input.value) }, icon("folderOpen", 16));
  const openBtn = h("button", { class: "btn", onclick: () => launch(logdir.trim()) }, icon("plus", 18), "Open");
  const launchErr = h("div");

  root.appendChild(h("header", { class: "page-head" }, h("h1", null, "Metrics"),
    h("p", { class: "subtitle" }, "Open TensorBoard for any run. Sessions appear as tabs below.")));

  // -- available runs (jobs with TensorBoard events) -------------------
  const runsBox = h("div", { style: { marginBottom: "20px" } });
  root.appendChild(runsBox);

  function loadRuns() {
    mount(runsBox, h("p", { class: "muted small", style: { margin: "8px 0" } }, "Loading available runs…"));
    api.metricsJobs()
      .then((jobs) => mount(runsBox, renderRuns(jobs)))
      .catch((err) => mount(runsBox, errorBox(err)));
  }

  function renderRuns(jobs) {
    const header = h("div", { class: "row section-head" },
      h("h2", { class: "section", style: { margin: 0 } }, "Available runs"),
      h("button", { class: "btn-icon", title: "Refresh", style: { marginLeft: "auto" }, onclick: loadRuns }, icon("refresh", 15)));
    if (!jobs.length) {
      return h("div", null, header, h("p", { class: "muted small" }, "No runs with metrics yet. Train a model, then refresh."));
    }
    return h("div", null, header, ...jobs.map((j) => h("button", { class: "run-row", onclick: () => launch(j.workdir) },
      h("span", { class: "run-prog mono" }, j.program),
      j.project_name ? h("span", { class: "proj-chip" }, icon("box", 12), j.project_name) : null,
      h("span", { class: "run-folder mono" }, folderName(j.workdir)),
      (j.labels && j.labels.length) ? jobLabelChips(j) : null,
      h("span", { class: "run-time" }, new Date(j.created_at).toLocaleString()),
      h("span", { class: "run-open" }, icon("chart", 14), "Open"))));
  }

  // -- manual launch ----------------------------------------------------
  root.appendChild(h("h2", { class: "section" }, "Or open a folder manually"));
  root.appendChild(h("div", { class: "launch-row" }, h("div", { class: "path-row", style: { flex: "1" } }, input, browse), openBtn));
  root.appendChild(launchErr);

  const tabbar = h("div", { class: "tabbar", style: { marginTop: "20px" } });
  const framesWrap = h("div", { class: "frames-wrap" });
  root.appendChild(tabbar);
  root.appendChild(framesWrap);

  const frames = new Map();

  async function launch(dir) {
    if (!dir) { mount(launchErr, h("div", { style: { marginTop: "12px" } }, errorBox("Choose a log directory first."))); return; }
    clear(launchErr);
    openBtn.disabled = true;
    try { store.openTB(await api.startTensorBoard(dir)); }
    catch (err) { mount(launchErr, h("div", { style: { marginTop: "12px" } }, errorBox(err))); }
    finally { openBtn.disabled = false; }
  }

  function sync() {
    const { openTBs, activeTBId } = getState();
    clear(tabbar);
    for (const tb of openTBs) {
      tabbar.appendChild(h("div", { class: "tab" + (tb.id === activeTBId ? " active" : ""), onclick: () => store.setActiveTB(tb.id) },
        h("span", { class: "tab-label" }, folderName(tb.logdir)),
        h("button", { class: "tab-close", title: "Close", onclick: (e) => { e.stopPropagation(); store.closeTB(tb.id); } }, icon("x", 14))));
    }
    for (const tb of openTBs) {
      if (!frames.has(tb.id)) {
        const f = h("iframe", { class: "tb-frame", src: tb.url, title: "TensorBoard " + tb.id });
        frames.set(tb.id, f);
        framesWrap.appendChild(f);
      }
    }
    for (const id of [...frames.keys()]) {
      if (!openTBs.some((t) => t.id === id)) { frames.get(id).remove(); frames.delete(id); }
    }
    for (const [id, f] of frames) f.style.display = id === activeTBId ? "block" : "none";

    if (!openTBs.length) mount(framesWrap, h("p", { class: "muted", style: { padding: "24px 0" } }, "No TensorBoard sessions open. Open a run above."));
    else if (framesWrap.querySelector("p")) framesWrap.querySelector("p").remove();
  }

  const unsub = subscribe(sync);
  loadRuns();
  sync();
  return { el: root, cleanup: unsub };
}

function folderName(p) {
  const parts = (p || "").replace(/\/+$/, "").split("/");
  return parts[parts.length - 1] || "TensorBoard";
}
