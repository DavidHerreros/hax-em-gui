// Small modal that asks the user to pick a destination project. Resolves to the
// chosen project object ({ id, name, ... }), or null if cancelled. Dependency-free,
// styled like promptModal. Used by the "Copy to project" / "Duplicate to project"
// actions on jobs and workflows.
import { h } from "../h.js";
import { icon } from "../icons.js";
import { api } from "../api.js";

export function pickProject({ title = "Copy to project", okText, excludeId = null, hint = "" } = {}) {
  return new Promise((resolve) => {
    const list = h("div", { class: "pick-list" }, h("p", { class: "muted small" }, "Loading projects…"));

    const modal = h("div", { class: "modal prompt-modal", onclick: (e) => e.stopPropagation(),
        onkeydown: (e) => { e.stopPropagation(); if (e.key === "Escape") { e.preventDefault(); done(null); } } },
      h("div", { class: "modal-head" }, h("span", null, title),
        h("button", { class: "btn-icon", style: { border: "none" }, onclick: () => done(null) }, icon("x", 18))),
      h("div", { class: "prompt-body" },
        hint ? h("p", { class: "muted small", style: { marginTop: 0 } }, hint) : null,
        list),
      h("div", { class: "modal-foot", style: { gap: "8px" } },
        h("button", { class: "btn ghost", onclick: () => done(null) }, "Cancel")));

    const overlay = h("div", { class: "overlay", onclick: () => done(null) }, modal);
    document.body.appendChild(overlay);
    document.addEventListener("keydown", onDocKey);

    (async () => {
      let projects = [];
      try { projects = await api.projects(); } catch { /* handled below */ }
      const choices = (projects || []).filter((p) => p.id !== excludeId);
      list.textContent = "";
      if (!choices.length) {
        list.append(h("p", { class: "muted small" },
          "No other project to copy into. Create one from the Projects page first."));
        return;
      }
      for (const p of choices) {
        list.append(h("button", { class: "pick-item", onclick: () => done(p) },
          icon("box", 15),
          h("span", { class: "pick-name" }, p.name || p.id),
          okText ? h("span", { class: "muted small" }, okText) : null));
      }
    })();

    function onDocKey(e) { if (e.key === "Escape") done(null); }
    function done(v) {
      overlay.remove();
      document.removeEventListener("keydown", onDocKey);
      resolve(v || null);
    }
  });
}
