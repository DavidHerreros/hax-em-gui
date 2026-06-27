// 3D viewer: launch the WebGL conformational-landscape viewer for a run that has
// latents + a trained network, embedded as iframe tabs (sessions persist across
// tab switches, like Metrics).
import { h, mount, clear } from "../h.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { store, getState, subscribe } from "../store.js";
import { errorBox } from "./ui.js";

// A job is viewable if it carries both a latents file and a network checkpoint.
function viewable(job) {
  const a = job.args || {};
  const latents = a.z_space || a.latents || a.latents_file;
  const reload = a.pickled_nn || a.reload;
  return latents && reload ? { latents, reload } : null;
}

export function ViewerView() {
  const root = h("div", { class: "view viewer-view" });
  root.appendChild(h("header", { class: "page-head" }, h("h1", null, "3D viewer"),
    h("p", { class: "subtitle" }, "Explore a conformational landscape in WebGL: pick a state, see its decoded map. Sessions open as tabs below.")));

  const runsBox = h("div", { style: { marginBottom: "20px" } });
  root.appendChild(runsBox);

  const tabbar = h("div", { class: "tabbar", style: { marginTop: "8px" } });
  const framesWrap = h("div", { class: "frames-wrap" });
  root.appendChild(tabbar);
  root.appendChild(framesWrap);
  const frames = new Map();

  // Mirror the GUI light/dark theme into each viewer iframe (initial ?theme= + live postMessage).
  const themeOf = () => (document.documentElement.dataset.theme === "light" ? "light" : "dark");
  const withTheme = (url) => url + (url.includes("?") ? "&" : "?") + "theme=" + themeOf();
  function onTheme(e) {
    const theme = e.detail === "light" ? "light" : "dark";
    for (const f of frames.values()) { try { f.contentWindow.postMessage({ type: "hax-theme", theme }, "*"); } catch (_) {} }
  }
  window.addEventListener("hax-theme", onTheme);

  function loadRuns() {
    mount(runsBox, h("p", { class: "muted small", style: { margin: "8px 0" } }, "Loading runs…"));
    api.jobs()
      .then((jobs) => mount(runsBox, renderRuns(jobs.filter((j) => viewable(j)))))
      .catch((err) => mount(runsBox, errorBox(err)));
  }

  function renderRuns(jobs) {
    const header = h("div", { class: "row section-head" },
      h("h2", { class: "section", style: { margin: 0 } }, "Viewable runs"),
      h("button", { class: "btn-icon", title: "Refresh", style: { marginLeft: "auto" }, onclick: loadRuns }, icon("refresh", 15)));
    if (!jobs.length) {
      return h("div", null, header,
        h("p", { class: "muted small" }, "None yet. Launch annotate_space (or any run) with both a latents file and a ", h("span", { class: "mono" }, "--pickled_nn"), " network, then refresh."));
    }
    return h("div", null, header, ...jobs.map((j) => {
      const btn = h("button", { class: "run-row", onclick: () => launch(j, btn) },
        h("span", { class: "run-prog mono" }, j.name || j.program),
        j.project_name ? h("span", { class: "proj-chip" }, icon("box", 12), j.project_name) : null,
        h("span", { class: "run-folder mono" }, j.program),
        h("span", { class: "run-time" }, new Date(j.created_at).toLocaleString()),
        h("span", { class: "run-open" }, icon("box", 14), "Open viewer"));
      return btn;
    }));
  }

  async function launch(job, btn) {
    if (btn) { btn.disabled = true; btn.lastChild.textContent = "Starting…"; }
    try {
      store.openViewer(await api.viewerFromJob(job.id));
    } catch (err) {
      mount(runsBox, errorBox(err));
    } finally {
      if (btn) { btn.disabled = false; btn.lastChild.textContent = "Open viewer"; }
    }
  }

  function sync() {
    const { openViewers, activeViewerId } = getState();
    clear(tabbar);
    for (const v of openViewers) {
      tabbar.appendChild(h("div", { class: "tab" + (v.id === activeViewerId ? " active" : ""), onclick: () => store.setActiveViewer(v.id) },
        h("span", { class: "tab-label" }, v.title || "Viewer"),
        h("button", { class: "tab-close", title: "Close", onclick: (e) => { e.stopPropagation(); closeViewer(v.id); } }, icon("x", 14))));
    }
    for (const v of openViewers) {
      if (!frames.has(v.id)) {
        const f = h("iframe", { class: "tb-frame", src: withTheme(v.url), title: "Viewer " + v.id });
        frames.set(v.id, f);
        framesWrap.appendChild(f);
      }
    }
    for (const id of [...frames.keys()]) {
      if (!openViewers.some((v) => v.id === id)) { frames.get(id).remove(); frames.delete(id); }
    }
    for (const [id, f] of frames) f.style.display = id === activeViewerId ? "block" : "none";

    if (!openViewers.length) mount(framesWrap, h("p", { class: "muted", style: { padding: "24px 0" } }, "No viewer open. Open a run above."));
    else if (framesWrap.querySelector("p")) framesWrap.querySelector("p").remove();
  }

  function closeViewer(id) {
    api.stopViewer(id).catch(() => {});
    store.closeViewer(id);
  }

  const unsub = subscribe(sync);
  loadRuns();
  sync();
  return { el: root, cleanup: () => { unsub(); window.removeEventListener("hax-theme", onTheme); } };
}
