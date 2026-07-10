// Server-side file browser modal (sandboxed to the backend's file_roots).
import { h, mount } from "../h.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { store } from "../store.js";
import { errorBox } from "./ui.js";

const MAP_RX = /\.(mrc|map|vol)$/i;
const MD_RX = /\.(xmd|star|cs)$/i;

export function openFilePicker(kind, onSelect) {
  let current;            // last *successfully* listed directory
  let ok = false;         // is `current` what the modal is showing right now?
  let showHidden = false; // hide dotfiles by default

  const list = h("div", { class: "filelist" });
  const pathLabel = h("span", { class: "cur" }, "…");
  // After a failed listing, "up" returns to the last good directory (the one the
  // failing entry was clicked in); otherwise it walks to the parent as usual.
  const upBtn = h("button", { class: "btn-icon", onclick: () => load(current && (ok ? current.parent : current.path)) }, icon("chevronUp", 16));
  const hiddenCheck = h("input", { type: "checkbox", onchange: (e) => { showHidden = e.target.checked; renderList(); } });
  const hiddenToggle = h("label", { class: "fp-hidden", title: "Show hidden files (dotfiles)" }, hiddenCheck, "Hidden");

  // Creating folders only makes sense when picking one; a file picker stays read-only.
  const newDir = kind === "directory" ? newDirRow() : null;

  const useBtn =
    kind === "directory"
      ? h("button", { class: "btn", onclick: () => ok && current && finish(current.path) }, icon("check", 16), "Use this folder")
      : null;
  const foot = useBtn ? h("div", { class: "modal-foot" }, useBtn) : null;

  const modal = h(
    "div",
    { class: "modal", onclick: (e) => e.stopPropagation() },
    h(
      "div",
      { class: "modal-head" },
      h("span", null, "Select " + kind),
      h("button", { class: "btn-icon", style: { border: "none" }, onclick: close }, icon("x", 18))
    ),
    h("div", { class: "modal-path" }, upBtn, pathLabel, newDir && newDir.toggle, hiddenToggle),
    newDir && newDir.row,
    list,
    foot
  );

  const overlay = h("div", { class: "overlay", onclick: close }, modal);
  document.body.appendChild(overlay);

  function close() { overlay.remove(); document.removeEventListener("keydown", onKey); }
  function onKey(e) { if (e.key === "Escape") close(); }
  document.addEventListener("keydown", onKey);

  function finish(path) { close(); onSelect(path); }

  // Nothing may be selected or created unless the modal is showing a real listing.
  function setUsable(on) {
    ok = on;
    if (useBtn) useBtn.disabled = !on;
    if (newDir) newDir.toggle.disabled = !on;
  }

  function load(path) {
    if (newDir) newDir.show(false);
    setUsable(false);
    mount(list, h("p", { class: "muted", style: { padding: "12px" } }, "Loading…"));
    api
      .files(path || undefined)
      .then((data) => {
        current = data;
        pathLabel.textContent = data.path;
        upBtn.disabled = !data.parent;
        setUsable(true);
        renderList();
      })
      .catch((err) => {
        // `current` still holds the last good directory so "up" can go back — but the
        // modal is no longer showing it, so it must not be selectable. Naming the
        // attempted path here keeps the label honest about what actually failed.
        if (path) pathLabel.textContent = path;
        upBtn.disabled = !current;
        mount(list, errorBox(err));
      });
  }

  // Toggle button for the path bar, plus the name row it expands beneath it.
  // The new folder is created inside whatever directory is currently listed.
  function newDirRow() {
    const err = h("span", { class: "fp-err" });
    const input = h("input", {
      class: "input",
      placeholder: "New folder name",
      onkeydown: (e) => {
        if (e.key === "Enter") { e.preventDefault(); create(); }
        // Escape cancels the row; don't let it bubble up and close the modal.
        else if (e.key === "Escape") { e.stopPropagation(); show(false); }
      },
    });
    const createBtn = h("button", { class: "btn", onclick: create }, icon("check", 16), "Create");
    const row = h("div", { class: "fp-newdir", hidden: true },
      input,
      createBtn,
      h("button", { class: "btn-icon", title: "Cancel", onclick: () => show(false) }, icon("x", 16)),
      err);
    const toggle = h("button", { class: "btn-icon", title: "Create a new folder here", onclick: () => show(row.hidden) }, icon("folderPlus", 16));
    toggle.disabled = true;   // enabled once a directory has listed

    function show(on) {
      row.hidden = !on;
      err.textContent = "";
      if (on) { input.value = ""; input.focus(); }
    }

    async function create() {
      const name = input.value.trim();
      if (!name || !current) return;
      createBtn.disabled = true;
      err.textContent = "";
      try {
        current = await api.mkdir(current.path, name);
        // A dotted name would otherwise vanish straight behind the "Hidden" filter.
        if (name.startsWith(".") && !showHidden) showHidden = hiddenCheck.checked = true;
        show(false);
        renderList();
      } catch (e) {
        err.textContent = e.message || String(e);
      } finally {
        createBtn.disabled = false;
      }
    }

    return { row, toggle, show };
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
