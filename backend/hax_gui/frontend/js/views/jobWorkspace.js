// Two-pane shell for the pages that work on one run at a time: the job tree on
// the left, the page's own content on the right, with a draggable divider whose
// width is remembered. Jobs, Metrics and the 3D viewer all mount into this.
import { h } from "../h.js";

const WIDTH_KEY = "hax_gui_jobtree_w";
const MIN = 220, MAX = 600, DEFAULT = 300;

function savedWidth() {
  try { const w = parseInt(localStorage.getItem(WIDTH_KEY), 10); if (w >= MIN && w <= MAX) return w; } catch { /* ignore */ }
  return DEFAULT;
}

export function jobWorkspace(rail, stage) {
  const divider = h("div", { class: "jw-divider", title: "Drag to resize · double-click to reset" });
  const el = h("div", { class: "jw", style: { "--jw-rail": savedWidth() + "px" } }, rail, divider, stage);

  function setWidth(w) {
    w = Math.max(MIN, Math.min(MAX, Math.round(w)));
    el.style.setProperty("--jw-rail", w + "px");
    try { localStorage.setItem(WIDTH_KEY, String(w)); } catch { /* ignore */ }
  }
  divider.addEventListener("dblclick", () => setWidth(DEFAULT));
  divider.addEventListener("mousedown", (e) => {
    e.preventDefault();
    const left = el.getBoundingClientRect().left;
    divider.classList.add("drag");
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    const move = (ev) => setWidth(ev.clientX - left);
    const up = () => {
      divider.classList.remove("drag");
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  });
  return el;
}

// The right pane's "nothing selected" state.
export function stageEmpty(title, text) {
  return h("div", { class: "jw-empty" }, h("div", null, h("h2", null, title), text ? h("p", null, text) : null));
}
