// Non-blocking toast notifications, replacing the browser-native alert() calls
// that froze the tab and looked like a different program. Bottom-right stack,
// auto-dismiss (errors linger longer), click to dismiss immediately.
import { h } from "./h.js";
import { icon } from "./icons.js";

let stack = null;

function ensureStack() {
  if (!stack) {
    stack = h("div", { class: "toast-stack" });
    document.body.appendChild(stack);
  }
  return stack;
}

const ICONS = { error: "ban", success: "check", info: "chat" };

export function toast(message, { kind = "info", timeout } = {}) {
  const ms = timeout ?? (kind === "error" ? 8000 : 4000);
  const el = h("div", { class: `toast ${kind}`, onclick: () => dismiss() },
    icon(ICONS[kind] || "chat", 15),
    h("span", { class: "toast-msg" }, String(message)));
  ensureStack().appendChild(el);
  let timer = setTimeout(dismiss, ms);
  function dismiss() {
    if (!el.isConnected) return;
    clearTimeout(timer);
    el.classList.add("out");
    setTimeout(() => el.remove(), 180);
  }
  return dismiss;
}

// The message an API error deserves: its detail when it has one, generic otherwise.
export function errText(err, fallback = "Something went wrong.") {
  if (!err) return fallback;
  const d = err.detail;
  if (typeof d === "string" && d) return d;
  if (d) { try { return JSON.stringify(d); } catch { /* fall through */ } }
  return err.message || fallback;
}

export const toastError = (err, fallback) => toast(errText(err, fallback), { kind: "error" });
