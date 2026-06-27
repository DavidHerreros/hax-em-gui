// Server-side file browser modal (sandboxed to the backend's file_roots).
import { h, mount } from "../h.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { errorBox } from "./ui.js";

export function openFilePicker(kind, onSelect) {
  let current; // current directory listing

  const list = h("div", { class: "filelist" });
  const pathLabel = h("span", { class: "cur" }, "…");
  const upBtn = h("button", { class: "btn-icon", onclick: () => load(current && current.parent) }, icon("chevronUp", 16));

  const foot =
    kind === "directory"
      ? h(
          "div",
          { class: "modal-foot" },
          h("button", { class: "btn", onclick: () => current && finish(current.path) }, icon("check", 16), "Use this folder")
        )
      : null;

  const modal = h(
    "div",
    { class: "modal", onclick: (e) => e.stopPropagation() },
    h(
      "div",
      { class: "modal-head" },
      h("span", null, "Select " + kind),
      h("button", { class: "btn-icon", style: { border: "none" }, onclick: close }, icon("x", 18))
    ),
    h("div", { class: "modal-path" }, upBtn, pathLabel),
    list,
    foot
  );

  const overlay = h("div", { class: "overlay", onclick: close }, modal);
  document.body.appendChild(overlay);

  function close() { overlay.remove(); document.removeEventListener("keydown", onKey); }
  function onKey(e) { if (e.key === "Escape") close(); }
  document.addEventListener("keydown", onKey);

  function finish(path) { close(); onSelect(path); }

  function load(path) {
    mount(list, h("p", { class: "muted", style: { padding: "12px" } }, "Loading…"));
    api
      .files(path || undefined)
      .then((data) => {
        current = data;
        pathLabel.textContent = data.path;
        upBtn.disabled = !data.parent;
        const entries = data.entries.filter((e) => (kind === "directory" ? e.is_dir : true));
        if (!entries.length) return mount(list, h("p", { class: "muted", style: { padding: "12px" } }, "Empty."));
        mount(
          list,
          h("div", null, ...entries.map((e) => entryRow(e)))
        );
      })
      .catch((err) => mount(list, errorBox(err)));
  }

  function entryRow(e) {
    return h(
      "button",
      {
        class: "entry",
        onclick: () => (e.is_dir ? load(e.path) : finish(e.path)),
        ondblclick: () => kind === "directory" && e.is_dir && finish(e.path),
      },
      icon(e.is_dir ? "folder" : "file", 16),
      h("span", { class: "nm" + (e.is_dir ? " folder" : "") }, e.name),
      e.size != null ? h("span", { class: "sz" }, formatSize(e.size)) : null
    );
  }

  load(undefined);
}

function formatSize(bytes) {
  const units = ["B", "KB", "MB", "GB", "TB"];
  let n = bytes, i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(n < 10 && i > 0 ? 1 : 0)} ${units[i]}`;
}
