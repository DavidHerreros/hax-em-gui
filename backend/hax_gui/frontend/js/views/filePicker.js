// Server-side file browser modal (sandboxed to the backend's file_roots).
import { h, mount } from "../h.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { store } from "../store.js";
import { errorBox } from "./ui.js";

const MAP_RX = /\.(mrc|map|vol)$/i;
const MD_RX = /\.(xmd|star|cs)$/i;

export function openFilePicker(kind, onSelect) {
  let current;            // current directory listing
  let showHidden = false; // hide dotfiles by default

  const list = h("div", { class: "filelist" });
  const pathLabel = h("span", { class: "cur" }, "…");
  const upBtn = h("button", { class: "btn-icon", onclick: () => load(current && current.parent) }, icon("chevronUp", 16));
  const hiddenToggle = h("label", { class: "fp-hidden", title: "Show hidden files (dotfiles)" },
    h("input", { type: "checkbox", onchange: (e) => { showHidden = e.target.checked; renderList(); } }),
    "Hidden");

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
    h("div", { class: "modal-path" }, upBtn, pathLabel, hiddenToggle),
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
        renderList();
      })
      .catch((err) => mount(list, errorBox(err)));
  }

  // Render the current listing, hiding dotfiles unless "Hidden" is on. Toggling
  // re-renders in place (no re-fetch).
  function renderList() {
    if (!current) return;
    const entries = current.entries.filter((e) =>
      (showHidden || !e.hidden) && (kind === "directory" ? e.is_dir : true));
    if (!entries.length) {
      const hiddenN = current.entries.filter((e) => e.hidden && (kind === "directory" ? e.is_dir : true)).length;
      const msg = !showHidden && hiddenN ? `Empty (${hiddenN} hidden).` : "Empty.";
      return mount(list, h("p", { class: "muted", style: { padding: "12px" } }, msg));
    }
    mount(list, h("div", null, ...entries.map((e) => entryRow(e))));
  }

  function entryRow(e) {
    const row = h(
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
    // Volume and metadata files get an inline "inspect" action that opens the
    // matching standalone viewer without selecting the file into the current field.
    if (!e.is_dir && MAP_RX.test(e.name)) {
      const inspect = h("button",
        { class: "btn-icon", title: "Inspect this map (3D / orthogonal slices)", onclick: (ev) => { ev.stopPropagation(); inspectFile(e.path, "map"); } },
        icon("box", 15));
      return h("div", { class: "entry-map" }, row, inspect);
    }
    if (!e.is_dir && MD_RX.test(e.name)) {
      const inspect = h("button",
        { class: "btn-icon", title: "Browse this metadata as a table (with plots)", onclick: (ev) => { ev.stopPropagation(); inspectFile(e.path, "md"); } },
        icon("grid", 15));
      return h("div", { class: "entry-map" }, row, inspect);
    }
    return row;
  }

  async function inspectFile(path, kind) {
    try {
      store.openViewer(kind === "md" ? await api.mdViewer(path) : await api.mrcViewer(path));
      close();   // opens in the floating viewer window
    } catch (err) { mount(list, errorBox(err)); }
  }

  load(undefined);
}

function formatSize(bytes) {
  const units = ["B", "KB", "MB", "GB", "TB"];
  let n = bytes, i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(n < 10 && i > 0 ? 1 : 0)} ${units[i]}`;
}
