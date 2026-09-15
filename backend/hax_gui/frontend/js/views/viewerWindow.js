// Floating, movable, resizable window that hosts every open viewer (mrc maps,
// metadata tables, and the 3D WebGL landscape) as tabs. Mounted once in the app
// shell so it persists across route navigation — opening a viewer pops it over
// whatever page you're on instead of navigating away.
//
// The iframes live in a persistent Map and are only ever shown/hidden (never
// detached), so dragging / resizing / switching tabs never reloads a session.
// Drag & resize use pointer capture (+ a body class disabling iframe pointer
// events) so the gesture keeps tracking even while the cursor is over an iframe.
import { h, clear } from "../h.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { store, getState, subscribe } from "../store.js";
import { announceJobRename } from "./jobs.js";
import { promptModal } from "./promptModal.js";
import { openFilePicker } from "./filePicker.js";
import { toast } from "../toast.js";

const GEO_KEY = "hax_gui_viewer_window";
const MIN_W = 380, MIN_H = 260;

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

function loadGeom() {
  const def = { width: 900, height: 620, left: Math.max(12, window.innerWidth - 924), top: 84 };
  try {
    const g = JSON.parse(localStorage.getItem(GEO_KEY) || "null");
    if (g && ["width", "height", "left", "top"].every((k) => typeof g[k] === "number")) return g;
  } catch { /* fall through */ }
  return def;
}

export function mountViewerWindow() {
  const geom = loadGeom();
  let maximized = false, minimized = false;
  let knownIds = new Set();
  const frames = new Map();
  const popped = new Map();

  const tabs = h("div", { class: "vwin-tabs" });
  const body = h("div", { class: "vwin-body" });
  const btnMin = h("button", { class: "vwin-btn", title: "Minimize", onclick: () => { minimized = !minimized; render(); } }, "–");
  const btnMax = h("button", { class: "vwin-btn", title: "Maximize", onclick: () => { maximized = !maximized; applyGeom(); render(); } }, "▢");
  const btnClose = h("button", { class: "vwin-btn", title: "Close all viewers", onclick: closeAll }, icon("x", 16));
  const bar = h("div", { class: "vwin-bar" },
    h("span", { class: "vwin-title" }, "Viewers"), h("span", { class: "vwin-spacer" }), btnMin, btnMax, btnClose);
  // Invisible resize handles on every edge and corner (the cursor signals them).
  const RESIZE_DIRS = ["n", "s", "e", "w", "ne", "nw", "se", "sw"];
  const resizeHandles = RESIZE_DIRS.map((dir) => h("div", { class: "vwin-rz vwin-rz-" + dir }));
  const win = h("div", { class: "vwin" }, bar, tabs, body, ...resizeHandles);
  document.body.appendChild(win);

  // ── theme: mirror the GUI light/dark into each viewer iframe ──
  const themeOf = () => (document.documentElement.dataset.theme === "light" ? "light" : "dark");
  const withTheme = (url) => url + (url.includes("?") ? "&" : "?") + "theme=" + themeOf();
  window.addEventListener("hax-theme", (e) => {
    const theme = e.detail === "light" ? "light" : "dark";
    for (const f of frames.values()) { try { f.contentWindow.postMessage({ type: "hax-theme", theme }, "*"); } catch { /* cross-origin/none */ } }
    for (const w of popped.values()) { try { w.postMessage({ type: "hax-theme", theme }, "*"); } catch { /* window gone */ } }
  });

  // ── file picking on behalf of a viewer iframe (e.g. the PDB overlay) ──
  // The viewer apps are separate pages with no access to the GUI's file browser,
  // so they ask over postMessage and get the chosen server path back. Only our
  // own iframes are answered; the reply goes to whichever frame asked.
  window.addEventListener("message", (e) => {
    const d = e.data;
    if (!d || d.type !== "hax-pick-file" || !e.source) return;
    if (![...frames.values()].some((f) => f.contentWindow === e.source) && ![...popped.values()].includes(e.source)) return;
    const reply = (path) => { try { e.source.postMessage({ type: "hax-picked-file", token: d.token, path }, "*"); } catch { /* frame gone */ } };
    openFilePicker("file", reply, d.startPath || undefined);
  });

  function applyGeom() {
    if (maximized) {
      win.style.left = win.style.top = "12px";
      win.style.width = "calc(100vw - 24px)"; win.style.height = "calc(100vh - 24px)";
      return;
    }
    geom.width = Math.max(MIN_W, geom.width);
    geom.height = Math.max(MIN_H, geom.height);
    geom.left = clamp(geom.left, 0, window.innerWidth - 120);
    geom.top = clamp(geom.top, 0, window.innerHeight - 44);
    win.style.left = geom.left + "px"; win.style.top = geom.top + "px";
    win.style.width = geom.width + "px"; win.style.height = geom.height + "px";
  }
  const saveGeom = () => { try { localStorage.setItem(GEO_KEY, JSON.stringify(geom)); } catch { /* storage off */ } };

  // ── drag (title bar) & resize (corner), pointer-captured so iframes don't
  //    swallow the gesture ──
  function gesture(el, onDown) {
    el.addEventListener("pointerdown", (e) => {
      if (e.button !== 0 || e.target.closest(".vwin-btn")) return;
      const step = onDown(e);
      if (!step) return;
      e.preventDefault();
      el.setPointerCapture(e.pointerId);
      document.body.classList.add("vwin-dragging");
      const move = (ev) => step(ev);
      const up = () => {
        el.releasePointerCapture(e.pointerId);
        el.removeEventListener("pointermove", move); el.removeEventListener("pointerup", up);
        document.body.classList.remove("vwin-dragging");
        saveGeom();
      };
      el.addEventListener("pointermove", move); el.addEventListener("pointerup", up);
    });
  }
  gesture(bar, (e) => {
    if (maximized) return null;
    const sx = e.clientX, sy = e.clientY, l = geom.left, t = geom.top;
    return (ev) => { geom.left = l + ev.clientX - sx; geom.top = t + ev.clientY - sy; applyGeom(); };
  });
  // Resize from any edge/corner: e/s grow width/height; w/n also move the top-left
  // corner so the opposite edge stays put (min-size honoured on all sides).
  resizeHandles.forEach((el, i) => {
    const dir = RESIZE_DIRS[i];
    gesture(el, (e) => {
      if (maximized) return null;
      const sx = e.clientX, sy = e.clientY, l0 = geom.left, t0 = geom.top, w0 = geom.width, h0 = geom.height;
      return (ev) => {
        const dx = ev.clientX - sx, dy = ev.clientY - sy;
        if (dir.includes("e")) geom.width = Math.max(MIN_W, w0 + dx);
        if (dir.includes("s")) geom.height = Math.max(MIN_H, h0 + dy);
        if (dir.includes("w")) { const nw = Math.max(MIN_W, w0 - dx); geom.left = l0 + (w0 - nw); geom.width = nw; }
        if (dir.includes("n")) { const nh = Math.max(MIN_H, h0 - dy); geom.top = t0 + (h0 - nh); geom.height = nh; }
        applyGeom();
      };
    });
  });
  window.addEventListener("resize", () => { if (win.style.display !== "none") applyGeom(); });

  // ── pop-out: each viewer is already a standalone same-origin page (that's
  // what makes it embeddable as an iframe), so the same URL opened with
  // window.open is simply the same page living in a real OS window instead
  // of our tab strip — useful for dragging a viewer to another monitor.
  // There's no reliable cross-window "closed" event, so a single poll (only
  // running while something is popped) notices when the user hits the OS
  // close button and re-docks it. Re-docking recreates the iframe from
  // scratch, so its client-side state resets, but the viewer session itself
  // lives on the server and the fresh iframe just reconnects to it.
  let popPoll = null;
  function startPopPoll() {
    if (popPoll) return;
    popPoll = setInterval(() => {
      let changed = false;
      for (const [id, w] of popped) if (w.closed) { popped.delete(id); changed = true; }
      if (!popped.size) { clearInterval(popPoll); popPoll = null; }
      if (changed) render();
    }, 1000);
  }
  function popOut(v) {
    const w = window.open(withTheme(v.url), "hax-viewer-" + v.id,
      `popup=yes,width=${geom.width},height=${geom.height},left=${geom.left + 40},top=${geom.top + 40}`);
    if (!w) { toast("Pop-out blocked by the browser — allow popups for this site.", { kind: "error" }); return; }
    popped.set(v.id, w);
    const f = frames.get(v.id);
    if (f) { f.remove(); frames.delete(v.id); }
    startPopPoll();
    if (getState().activeViewerId === v.id) {
      const next = getState().openViewers.find((x) => x.id !== v.id && !popped.has(x.id));
      if (next) { store.setActiveViewer(next.id); return; } // setActiveViewer emits → render()
    }
    render();
  }
  function reDock(id) {
    const w = popped.get(id);
    popped.delete(id);
    if (w && !w.closed) { try { w.close(); } catch { /* already gone */ } }
    render();
  }
  // If the GUI tab itself goes away, popped-out windows would otherwise be
  // left running with no way to manage or close them — take them down too.
  window.addEventListener("pagehide", () => { for (const w of popped.values()) { try { w.close(); } catch { /* already gone */ } } });

  function closeOne(id) {
    const v = getState().openViewers.find((x) => x.id === id);
    const stop = v && v.kind === "mrc" ? api.stopMrcViewer : v && v.kind === "md" ? api.stopMdViewer : api.stopViewer;
    stop(id).catch(() => {});
    const w = popped.get(id);
    if (w) { popped.delete(id); try { w.close(); } catch { /* already gone */ } }
    store.closeViewer(id);
  }
  function closeAll() { for (const v of [...getState().openViewers]) closeOne(v.id); }

  // Rename the run behind a viewer tab (double-click its label). Renames the job on
  // the server and updates the tab title; the Jobs list picks it up on its next poll.
  // A viewer's title is "<run or file> · <what of it>" — the map's file name, the
  // consensus space, "3 maps". That is already the shape of the two-line tab the metrics
  // run tabs use, so split on the same separator the rename path splits on. A landscape
  // opened from a job carries no qualifier, and falls back to the subtitle the launcher
  // attached (program · run id).
  function tabLines(v) {
    const title = v.title || "Viewer";
    const sep = title.indexOf(" · ");
    if (sep >= 0) return { main: title.slice(0, sep), sub: title.slice(sep + 3) };
    return { main: title, sub: v.subtitle || "" };
  }

  async function renameTab(v) {
    if (!v.jobId) return;
    const sep = (v.title || "").indexOf(" · ");
    const current = sep >= 0 ? v.title.slice(0, sep) : (v.title || "");
    const name = await promptModal({ title: "Rename run", label: "Run name", value: current, okText: "Rename" });
    if (name === null || !name) return;
    try { const u = await api.renameJob(v.jobId, name); announceJobRename(v.jobId, u.name); }
    catch { /* ignore */ }
  }

  function render() {
    const { openViewers, activeViewerId } = getState();
    if (!openViewers.length) { win.style.display = "none"; knownIds = new Set(); return; }

    // A newly opened viewer un-minimizes and raises the window.
    if (openViewers.some((v) => !knownIds.has(v.id))) minimized = false;
    knownIds = new Set(openViewers.map((v) => v.id));

    win.style.display = "flex";
    win.classList.toggle("vwin-min", minimized);
    win.classList.toggle("vwin-max", maximized);
    btnMin.textContent = minimized ? "▔" : "–";
    btnMin.title = minimized ? "Restore" : "Minimize";
    btnMax.textContent = maximized ? "❐" : "▢";
    btnMax.title = maximized ? "Restore" : "Maximize";
    applyGeom();

    clear(tabs);
    for (const v of openViewers) {
      const isPopped = popped.has(v.id);
      const { main, sub } = tabLines(v);
      const label = h("span", {
        class: "tab-lines",
        title: (v.jobId ? "Double-click to rename the run\n" : "") + (v.title || "Viewer"),
        ondblclick: v.jobId ? (e) => { e.stopPropagation(); renameTab(v); } : null,
      },
        h("span", { class: "tab-main" }, main),
        sub ? h("span", { class: "tab-sub mono" }, sub) : null);
      tabs.appendChild(h("div", {
        class: "tab two-line" + (isPopped ? " popped" : v.id === activeViewerId ? " active" : ""),
        onclick: () => isPopped ? popped.get(v.id)?.focus() : store.setActiveViewer(v.id),
      },
        label,
        h("button", { class: "tab-pop", title: isPopped ? "Bring back into the GUI" : "Open in a separate window", onclick: (e) => { e.stopPropagation(); isPopped ? reDock(v.id) : popOut(v); } }, icon("external", 14)),
        h("button", { class: "tab-close", title: "Close", onclick: (e) => { e.stopPropagation(); closeOne(v.id); } }, icon("x", 14))));
    }
    for (const v of openViewers) {
      if (!frames.has(v.id) && !popped.has(v.id)) {
        const f = h("iframe", { class: "vwin-frame", src: withTheme(v.url), title: "Viewer " + v.id });
        frames.set(v.id, f); body.appendChild(f);
      }
    }
    for (const id of [...frames.keys()]) if (!knownIds.has(id)) { frames.get(id).remove(); frames.delete(id); }
    for (const [id, f] of frames) f.style.display = id === activeViewerId ? "block" : "none";
  }

  subscribe(render);
  render();
}
