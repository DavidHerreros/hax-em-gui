// Metrics: pick a run that has event files (auto-listed) or enter a log directory
// manually, then read it. Each run opens as a tab, rendered natively by metricsPanel —
// the GUI parses the same tfevents/projector files TensorBoard would, so there is no
// iframe. Real TensorBoard is still one click away inside each panel.
import { h, mount, clear } from "../h.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { store, getState, subscribe } from "../store.js";
import { errorBox } from "./ui.js";
import { openFilePicker } from "./filePicker.js";
import { jobLabelChips } from "./labels.js";
import { metricsPanel } from "./metricsPanel.js";
import { runBrowser, folderName, metricsTabInfo } from "./runBrowser.js";
import { jobTimeEl } from "./jobs.js";

export function MetricsView() {
  const root = h("div", { class: "view metrics-view" });

  let logdir = "";
  const input = h("input", { class: "input", placeholder: "server log directory (manual)", oninput: (e) => (logdir = e.target.value) });
  const browse = h("button", { class: "btn-icon", title: "Browse server", onclick: () => openFilePicker("directory", (p) => { input.value = p; logdir = p; }, input.value) }, icon("folderOpen", 16));
  const openBtn = h("button", { class: "btn", onclick: () => open(logdir.trim()) }, icon("plus", 18), "Open");
  const launchErr = h("div");

  root.appendChild(h("header", { class: "page-head" }, h("h1", null, "Metrics"),
    h("p", { class: "subtitle" }, "Training curves, image panels and the latent space, read straight from a run's logs.")));

  // -- available runs (jobs with event files), grouped by project + filterable -------
  const runsBox = h("div", { style: { marginBottom: "20px" } });
  root.appendChild(runsBox);

  const browser = runBrowser({
    renderRow: (j) => h("button", { class: "run-row", onclick: () => open(j.workdir, j) },
      h("span", { class: "run-prog mono" }, j.name || j.program),
      j.name ? h("span", { class: "run-sub mono" }, j.program) : null,
      h("span", { class: "run-folder mono" }, folderName(j.workdir)),
      (j.labels && j.labels.length) ? jobLabelChips(j) : null,
      jobTimeEl(j, "run-time"),
      h("span", { class: "run-open" }, icon("chart", 14), "Open")),
    emptyText: "No runs with metrics yet. Train a model, then refresh.",
  });
  mount(runsBox, h("div", null,
    h("div", { class: "row section-head" },
      h("h2", { class: "section", style: { margin: 0 } }, "Available runs"),
      h("button", { class: "btn-icon", title: "Refresh", style: { marginLeft: "auto" }, onclick: loadRuns }, icon("refresh", 15))),
    browser.el));

  function loadRuns() {
    browser.setStatus(h("p", { class: "muted small", style: { margin: "8px 0" } }, "Loading available runs…"));
    api.metricsJobs()
      .then((jobs) => browser.setJobs(jobs))
      .catch((err) => browser.setStatus(errorBox(err)));
  }

  // -- manual launch ----------------------------------------------------
  root.appendChild(h("h2", { class: "section" }, "Or open a folder manually"));
  root.appendChild(h("div", { class: "launch-row" }, h("div", { class: "path-row", style: { flex: "1" } }, input, browse), openBtn));
  root.appendChild(launchErr);

  const tabbar = h("div", { class: "tabbar", style: { marginTop: "20px" } });
  const panelsWrap = h("div", { class: "mx-wrap" });
  root.appendChild(tabbar);
  root.appendChild(panelsWrap);

  const panels = new Map();   // logdir -> panel instance (kept mounted so state survives tab switches)
  let shownId = null;         // which panel is currently on screen

  function open(dir, job) {
    if (!dir) { mount(launchErr, h("div", { style: { marginTop: "12px" } }, errorBox("Choose a log directory first."))); return; }
    clear(launchErr);
    const { title, subtitle } = metricsTabInfo(dir, job);
    store.openMetrics({ logdir: dir, title, subtitle, jobId: job ? job.id : null });
  }

  function close(id) {
    const p = panels.get(id);
    if (p) { p.destroy(true); p.el.remove(); panels.delete(id); }
    store.closeMetrics(id);
  }

  function sync() {
    const { openMetrics, activeMetricsId } = getState();
    clear(tabbar);
    for (const m of openMetrics) {
      // Two lines, capped width: a run called "… copy copy copy" glued to its folder is
      // wider than the tab bar, and truncating it to one line hides exactly the part
      // (the run id) that tells two of them apart.
      tabbar.appendChild(h("div", {
        class: "tab mx-tab" + (m.id === activeMetricsId ? " active" : ""),
        onclick: () => store.setActiveMetrics(m.id),
        title: `${m.title}\n${m.logdir}`,
      },
        h("span", { class: "mx-tabcol" },
          h("span", { class: "mx-tabname" }, m.title),
          m.subtitle ? h("span", { class: "mx-tabsub mono" }, m.subtitle) : null),
        h("button", { class: "tab-close", title: "Close", onclick: (e) => { e.stopPropagation(); close(m.id); } }, icon("x", 14))));
    }
    for (const m of openMetrics) {
      if (!panels.has(m.id)) {
        const p = metricsPanel({ logdir: m.logdir, jobId: m.jobId });
        panels.set(m.id, p);
        panelsWrap.appendChild(p.el);
      }
    }
    for (const [id, p] of [...panels]) {
      if (!openMetrics.some((m) => m.id === id)) { p.destroy(true); p.el.remove(); panels.delete(id); }
    }
    for (const [id, p] of panels) p.el.style.display = id === activeMetricsId ? "" : "none";
    // Nudge the newly revealed panel to redraw: until this point its canvases had no
    // width to draw into. (The renderers also self-heal via ResizeObserver; this just
    // makes the switch immediate rather than a frame late.)
    if (activeMetricsId !== shownId) {
      shownId = activeMetricsId;
      panels.get(activeMetricsId)?.onShow?.();
    }

    const empty = panelsWrap.querySelector(".mx-none");
    if (!openMetrics.length && !empty) {
      panelsWrap.appendChild(h("p", { class: "muted mx-none", style: { padding: "24px 0" } }, "No run open. Pick one above."));
    } else if (openMetrics.length && empty) empty.remove();
  }

  const unsub = subscribe(sync);
  loadRuns();
  sync();
  return {
    el: root,
    cleanup() {
      unsub();
      for (const p of panels.values()) p.destroy();   // stop pollers / free object URLs
      panels.clear();
    },
  };
}
