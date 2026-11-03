// Metrics: the job tree on the left (runs without event files dimmed), the
// selected run's curves / image panels / latent space on the right, rendered
// natively by metricsPanel — the GUI parses the same tfevents/projector files
// TensorBoard would, so there is no iframe. Picking a run replaces the one on screen
// unless that one is pinned (the tab's pin): panels are heavy and stay mounted while
// open, so only the runs kept on purpose are. A log directory that is not a run can
// still be opened by hand.
import { h, mount, clear } from "../h.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { store, getState, subscribe } from "../store.js";
import { errorBox } from "./ui.js";
import { openFilePicker } from "./filePicker.js";
import { metricsPanel } from "./metricsPanel.js";
import { metricsTabInfo } from "./runBrowser.js";
import { onJobEvent } from "../events.js";
import { jobTree } from "./jobTree.js";
import { jobWorkspace, stageEmpty } from "./jobWorkspace.js";

export function MetricsView() {
  // Which runs have event files: the list endpoint does not say, so ask once and
  // again whenever a job changes state (a training writes its first events early).
  let withMetrics = new Set();
  let byId = new Map();          // job id -> the metrics-jobs row (workdir + names)

  const tree = jobTree({
    isSelected: (j) => { const m = getState().activeMetricsId; return !!m && byId.get(j.id)?.workdir === m; },
    onSelect: (j) => { if (withMetrics.has(j.id)) open(j.workdir, j); else showNoMetrics(j); },
    usable: (j) => withMetrics.has(j.id),
    emptyText: "No runs in this project yet.",
  });

  // ── manual open: a log directory that is not a run of this project ──
  let logdir = "";
  const input = h("input", { class: "input", placeholder: "server log directory", oninput: (e) => (logdir = e.target.value) });
  const browse = h("button", { class: "btn-icon", title: "Browse server", onclick: () => openFilePicker("directory", (p) => { input.value = p; logdir = p; }, input.value) }, icon("folderOpen", 16));
  const openBtn = h("button", { class: "btn secondary", onclick: () => open(logdir.trim()) }, icon("plus", 16), "Open");
  const manual = h("div", { class: "jw-manual" },
    h("div", { class: "muted small", style: { marginBottom: "6px" } }, "Or open a log directory by hand"),
    h("div", { class: "launch-row" }, h("div", { class: "path-row", style: { flex: "1" } }, input, browse), openBtn));
  const launchErr = h("div");

  const tabbar = h("div", { class: "tabbar mx-tabs", hidden: true });
  const panelsWrap = h("div", { class: "mx-wrap" });
  const noteBox = h("div");
  const stage = h("div", { class: "jw-stage" }, tabbar, noteBox, launchErr, panelsWrap, manual);
  const root = h("div", { class: "view metrics-view" }, jobWorkspace(tree.el, stage));

  const panels = new Map();   // logdir -> panel instance (kept mounted so state survives switches)
  let shownId = null;

  function open(dir, job) {
    if (!dir) { mount(launchErr, h("div", { style: { marginBottom: "12px" } }, errorBox("Choose a log directory first."))); return; }
    clear(launchErr); clear(noteBox);
    const { title, subtitle } = metricsTabInfo(dir, job);
    if (job) store.setLastJob(job.id);
    store.openMetrics({ logdir: dir, title, subtitle, jobId: job ? job.id : null });
  }
  function showNoMetrics(j) {
    mount(noteBox, h("div", { class: "jw-note" },
      h("strong", null, j.name || j.program), j.status === "saved" ? " has not run yet." : " has not written any metrics — nothing to plot."));
  }
  function close(id) {
    const p = panels.get(id);
    if (p) { p.destroy(true); p.el.remove(); panels.delete(id); }
    store.closeMetrics(id);
  }

  function sync() {
    const { openMetrics, activeMetricsId } = getState();
    clear(tabbar);
    tabbar.hidden = !openMetrics.length;
    for (const m of openMetrics) {
      tabbar.appendChild(h("div", {
        class: "tab two-line" + (m.id === activeMetricsId ? " active" : ""),
        onclick: () => store.setActiveMetrics(m.id),
        title: `${m.title}\n${m.logdir}`,
      },
        h("span", { class: "tab-lines" },
          h("span", { class: "tab-main" }, m.title),
          m.subtitle ? h("span", { class: "tab-sub mono" }, m.subtitle) : null),
        h("button", { class: "tab-pin" + (m.pinned ? " on" : ""),
          title: m.pinned ? "Pinned: stays open while you browse other runs. Click to unpin."
                          : "Pin: keep this run open while you open others (otherwise the next run you pick replaces it)",
          onclick: (e) => { e.stopPropagation(); store.togglePinMetrics(m.id); } }, icon("bookmark", 13)),
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
    const empty = panelsWrap.querySelector(".jw-empty");
    if (!openMetrics.length && !empty) {
      panelsWrap.appendChild(stageEmpty("Select a run", "Runs with training curves are the bright ones in the tree; the rest are dimmed."));
    } else if (openMetrics.length && empty) empty.remove();
    tree.repaint();
  }

  async function loadMetricsJobs() {
    try {
      const rows = await api.metricsJobs();
      withMetrics = new Set(rows.map((j) => j.id));
      byId = new Map(rows.map((j) => [j.id, j]));
      tree.repaint();
      // First visit with nothing open: land on the run last selected elsewhere, if it has metrics.
      const last = getState().lastJobId;
      if (!getState().openMetrics.length && last && byId.has(last)) open(byId.get(last).workdir, byId.get(last));
    } catch (err) { mount(launchErr, errorBox(err)); }
  }

  const unsub = subscribe(sync);
  const offEvents = onJobEvent(() => loadMetricsJobs());
  loadMetricsJobs();
  sync();
  return {
    el: root,
    cleanup() {
      unsub(); offEvents(); tree.cleanup();
      for (const p of panels.values()) p.destroy();   // stop pollers / free object URLs
      panels.clear();
    },
  };
}
