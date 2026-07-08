// Small modal asking for a single text value. Returns a Promise that resolves to
// the entered string, or null if cancelled. Dependency-free.
import { h } from "../h.js";
import { icon } from "../icons.js";

export function promptModal({ title, label, value = "", placeholder = "", okText = "Save" } = {}) {
  return new Promise((resolve) => {
    const input = h("input", { class: "input", value, placeholder });

    // Keep every keystroke inside the modal: commit on Enter, cancel on Escape, and
    // stop it from reaching the view underneath. Otherwise global view shortcuts
    // (workflows Ctrl+D / Delete, other close-on-key handlers) would steal keys —
    // including Ctrl+C / Ctrl+V — and dismiss the popup while you're editing.
    const onModalKey = (e) => {
      e.stopPropagation();
      if (e.key === "Enter") { e.preventDefault(); done(input.value); }
      else if (e.key === "Escape") { e.preventDefault(); done(null); }
    };

    const modal = h("div", { class: "modal prompt-modal", onclick: (e) => e.stopPropagation(), onkeydown: onModalKey },
      h("div", { class: "modal-head" }, h("span", null, title || "Edit"),
        h("button", { class: "btn-icon", style: { border: "none" }, onclick: () => done(null) }, icon("x", 18))),
      h("div", { class: "prompt-body" },
        label ? h("label", { class: "field-label" }, label) : null,
        input),
      h("div", { class: "modal-foot", style: { gap: "8px" } },
        h("button", { class: "btn ghost", onclick: () => done(null) }, "Cancel"),
        h("button", { class: "btn", onclick: () => done(input.value) }, okText)));

    const overlay = h("div", { class: "overlay", onclick: () => done(null) }, modal);
    document.body.appendChild(overlay);
    // Escape fallback for the rare case focus has left the modal subtree (in-modal
    // Escape is handled above and stops there, so this never double-fires).
    document.addEventListener("keydown", onDocKey);
    queueMicrotask(() => { input.focus(); input.select(); });

    function onDocKey(e) { if (e.key === "Escape") done(null); }
    function done(v) {
      overlay.remove();
      document.removeEventListener("keydown", onDocKey);
      resolve(v == null ? null : v.trim());
    }
  });
}
