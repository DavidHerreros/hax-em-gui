// In-app confirmation modal replacing window.confirm(): styled like the rest of
// the GUI, Escape cancels, Enter confirms, danger actions get a red button.
// Returns a Promise<boolean>. Also used by router.js's nav guard, which reverts
// the hash and asks asynchronously (see the comment there).
import { h } from "../h.js";
import { icon } from "../icons.js";

export function confirmModal({ title = "Are you sure?", message = "", okText = "Confirm",
                               cancelText = "Cancel", danger = false } = {}) {
  return new Promise((resolve) => {
    const okBtn = h("button", { class: "btn" + (danger ? " danger" : ""), onclick: () => done(true) }, okText);
    const onKey = (e) => {
      e.stopPropagation();
      if (e.key === "Enter") { e.preventDefault(); done(true); }
      else if (e.key === "Escape") { e.preventDefault(); done(false); }
    };
    const modal = h("div", { class: "modal prompt-modal", onclick: (e) => e.stopPropagation(), onkeydown: onKey },
      h("div", { class: "modal-head" }, h("span", null, title),
        h("button", { class: "btn-icon", style: { border: "none" }, onclick: () => done(false) }, icon("x", 18))),
      message ? h("div", { class: "prompt-body", style: { whiteSpace: "pre-wrap" } }, message) : null,
      h("div", { class: "modal-foot", style: { gap: "8px" } },
        h("button", { class: "btn ghost", onclick: () => done(false) }, cancelText), okBtn));
    const overlay = h("div", { class: "overlay", onclick: () => done(false) }, modal);
    document.body.appendChild(overlay);
    document.addEventListener("keydown", onDocKey);
    queueMicrotask(() => okBtn.focus());

    function onDocKey(e) { if (e.key === "Escape") done(false); }
    function done(v) {
      overlay.remove();
      document.removeEventListener("keydown", onDocKey);
      resolve(v);
    }
  });
}
