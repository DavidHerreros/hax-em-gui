// The assistant chat as a window that stays on top of the other tabs: floating
// (draggable) or docked to the right edge, minimised to a pill that says when
// something waits for you. It mounts the same view as the Assistant tab, so the
// conversation carries over; on the Assistant tab itself the window steps aside.
import { h, clear, append } from "../h.js";
import { icon } from "../icons.js";
import { navigate, currentPath } from "../router.js";

const KEY = "hax_gui_assistant_float";
const st = { on: false, min: false, docked: false, x: null, y: null };
try { Object.assign(st, JSON.parse(localStorage.getItem(KEY) || "{}")); } catch { /* storage off */ }
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(st)); } catch { /* storage off */ } };

let chat = null;        // { View, activity, subscribe } from the assistant view, set on mount
let win = null, body = null, pill = null, view = null;
let lastOther = "/";    // where to go back to when the chat is floated from its own tab
const onAssistantTab = () => currentPath().startsWith("/assistant");

export function setFloating(on) {
  st.on = on;
  if (on) st.min = false;
  save();
  if (on && onAssistantTab()) navigate(lastOther);   // the route change shows the window
  else sync();
}

function place() {
  if (st.docked || st.x == null) { win.style.left = win.style.top = ""; return; }
  const x = Math.max(0, Math.min(st.x, window.innerWidth - win.offsetWidth));
  const y = Math.max(0, Math.min(st.y, window.innerHeight - 60));
  win.style.left = x + "px";
  win.style.top = y + "px";
}

function renderPill() {
  const a = chat.activity();
  clear(pill);
  append(pill, [icon("chat", 18), "Assistant",
    a.pending ? h("span", { class: "chat-badge ask" }, `${a.pending} to review`)
      : a.streaming ? h("span", { class: "chat-pill-note" }, a.streaming > 1 ? `${a.streaming} answering…` : "answering…")
      : a.unread ? h("span", { class: "chat-pill-note" }, "new reply") : null]);
}

function sync() {
  if (!win) return;
  const show = st.on && !onAssistantTab();
  const open = show && !st.min;
  pill.hidden = !(show && st.min);
  win.hidden = !open;
  win.classList.toggle("docked", st.docked);
  document.body.classList.toggle("chat-docked", open && st.docked);
  if (open && !view) {
    view = chat.View({}, { compact: true, headerExtra: controls() });
    body.replaceChildren(view.el);
    drag(view.header);
  } else if (!open && view) {
    view.cleanup();
    view = null;
    body.replaceChildren();
  }
  if (open) place();
  renderPill();
}

function controls() {
  const btn = (name, title, fn) => h("button", { class: "vwin-btn", title, "aria-label": title, onclick: fn }, icon(name, 15));
  return [
    btn("dock", "Dock to the side, or float", () => { st.docked = !st.docked; save(); sync(); }),
    btn("expand", "Open in the Assistant tab", () => navigate("/assistant")),
    btn("minus", "Minimise", () => { st.min = true; save(); sync(); }),
    btn("x", "Close the floating chat", () => setFloating(false)),
  ];
}

function drag(handle) {
  handle.addEventListener("pointerdown", (e) => {
    if (st.docked || e.button !== 0 || e.target.closest("button, select, input")) return;
    const r = win.getBoundingClientRect();
    const dx = e.clientX - r.left, dy = e.clientY - r.top;
    const move = (ev) => { st.x = ev.clientX - dx; st.y = ev.clientY - dy; place(); };
    const up = () => {
      handle.removeEventListener("pointermove", move);
      handle.removeEventListener("pointerup", up);
      save();
    };
    handle.setPointerCapture(e.pointerId);
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", up);
  });
}

export function mountAssistantFloat(assistant) {
  if (win) return;
  chat = assistant;
  body = h("div", { class: "chat-float-body" });
  win = h("section", { class: "chat-float", "aria-label": "Assistant", hidden: true }, body);
  pill = h("button", { class: "chat-pill", hidden: true, onclick: () => { st.min = false; save(); sync(); } });
  document.body.append(win, pill);
  chat.subscribe(renderPill);
  if (!onAssistantTab()) lastOther = currentPath();
  // Deferred: the router has then rendered (or its guard has put the old hash back).
  window.addEventListener("hashchange", () => setTimeout(() => {
    if (!onAssistantTab()) lastOther = currentPath();
    sync();
  }, 0));
  window.addEventListener("resize", () => { if (view) place(); });
  sync();
}
