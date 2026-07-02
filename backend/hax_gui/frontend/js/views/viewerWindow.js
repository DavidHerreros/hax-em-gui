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

  const tabs = h("div", { class: "vwin-tabs" });
  const body = h("div", { class: "vwin-body" });
  const btnMin = h("button", { class: "vwin-btn", title: "Minimize", onclick: () => { minimized = !minimized; render(); } }, "–");
  const btnMax = h("button", { class: "vwin-btn", title: "Maximize", onclick: () => { maximized = !maximized; applyGeom(); render(); } }, "▢");
  const btnClose = h("button", { class: "vwin-btn", title: "Close all viewers", onclick: closeAll }, icon("x", 16));
  const bar = h("div", { class: "vwin-bar" },
    h("span", { class: "vwin-title" }, "Viewers"), h("span", { class: "vwin-spacer" }), btnMin, btnMax, btnClose);
  const resizeH = h("div", { class: "vwin-resize", title: "Drag to resize" });
  const win = h("div", { class: "vwin" }, bar, tabs, body, resizeH);
  document.body.appendChild(win);

  // ── theme: mirror the GUI light/dark into each viewer iframe ──
  const themeOf = () => (document.documentElement.dataset.theme === "light" ? "light" : "dark");
  const withTheme = (url) => url + (url.includes("?") ? "&" : "?") + "theme=" + themeOf();
  window.addEventListener("hax-theme", (e) => {
    const theme = e.detail === "light" ? "light" : "dark";
    for (const f of frames.values()) { try { f.contentWindow.postMessage({ type: "hax-theme", theme }, "*"); } catch { /* cross-origin/none */ } }
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
  gesture(resizeH, (e) => {
    if (maximized) return null;
    const sx = e.clientX, sy = e.clientY, w = geom.width, ht = geom.height;
    return (ev) => { geom.width = Math.max(MIN_W, w + ev.clientX - sx); geom.height = Math.max(MIN_H, ht + ev.clientY - sy); applyGeom(); };
  });
  window.addEventListener("resize", () => { if (win.style.display !== "none") applyGeom(); });

  function closeOne(id) {
    const v = getState().openViewers.find((x) => x.id === id);
    const stop = v && v.kind === "mrc" ? api.stopMrcViewer : v && v.kind === "md" ? api.stopMdViewer : api.stopViewer;
    stop(id).catch(() => {});
    store.closeViewer(id);
  }
  function closeAll() { for (const v of [...getState().openViewers]) closeOne(v.id); }

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
      tabs.appendChild(h("div", { class: "tab" + (v.id === activeViewerId ? " active" : ""), onclick: () => store.setActiveViewer(v.id) },
        h("span", { class: "tab-label" }, v.title || "Viewer"),
        h("button", { class: "tab-close", title: "Close", onclick: (e) => { e.stopPropagation(); closeOne(v.id); } }, icon("x", 14))));
    }
    for (const v of openViewers) {
      if (!frames.has(v.id)) {
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
