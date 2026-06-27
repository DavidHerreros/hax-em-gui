// Small modal asking for a single text value. Returns a Promise that resolves to
// the entered string, or null if cancelled. Dependency-free.
import { h } from "../h.js";
import { icon } from "../icons.js";

export function promptModal({ title, label, value = "", placeholder = "", okText = "Save" } = {}) {
  return new Promise((resolve) => {
    const input = h("input", { class: "input", value, placeholder,
      onkeydown: (e) => { if (e.key === "Enter") done(input.value); } });

    const modal = h("div", { class: "modal prompt-modal", onclick: (e) => e.stopPropagation() },
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
    document.addEventListener("keydown", onKey);
    queueMicrotask(() => { input.focus(); input.select(); });

    function onKey(e) { if (e.key === "Escape") done(null); }
    function done(v) {
      overlay.remove();
      document.removeEventListener("keydown", onKey);
      resolve(v == null ? null : v.trim());
    }
  });
}
