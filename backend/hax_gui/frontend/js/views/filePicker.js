// Server-side file browser modal (sandboxed to the backend's file_roots).
import { h, mount } from "../h.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { store } from "../store.js";
import { errorBox } from "./ui.js";

const MAP_RX = /\.(mrc|map|vol)$/i;
const MD_RX = /\.(xmd|star|cs)$/i;

// `startPath` opens the modal at a folder (or at a file's folder) instead of at the
// first root — pass a field's current value so Browse resumes where the user left off.
export function openFilePicker(kind, onSelect, startPath) {
  let current;            // last *successfully* listed directory
  let ok = false;         // is `current` what the modal is showing right now?
  let showHidden = false; // hide dotfiles by default

  const list = h("div", { class: "filelist" });
  // Editable: type or paste a path and press Enter to jump straight there. Pasting a
  // *file* path lands in its folder with that file scrolled into view.
  const pathInput = h("input", {
    class: "cur", spellcheck: "false", autocomplete: "off",
    title: "Type or paste a path, then press Enter",
    onfocus: () => pathInput.select(),
    onkeydown: (e) => {
      e.stopPropagation();   // keep Escape (and typing) away from the modal's handler
      if (e.key === "Enter") { const v = pathInput.value.trim(); if (v) load(v); }
      else if (e.key === "Escape") { pathInput.value = (current && current.path) || ""; pathInput.blur(); }
    },
  });
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
    h("div", { class: "modal-path" }, upBtn, pathInput, newDir && newDir.toggle, hiddenToggle),
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

  // `fallback` is for the opening listing only: a start path that no longer exists
  // (or sits outside the roots) should quietly open somewhere sensible, whereas a
  // path the user navigated to or typed must report why it failed.
  function load(path, fallback) {
    if (newDir) newDir.show(false);
    setUsable(false);
    mount(list, h("p", { class: "muted", style: { padding: "12px" } }, "Loading…"));
    api
      .files(path || undefined, fallback)
      .then((data) => {
        current = data;
        pathInput.value = data.path;
        upBtn.disabled = !data.parent;
        // A named dotfile would otherwise sit behind the "Hidden" filter, unfindable.
        if (data.file && data.file.split("/").pop().startsWith(".") && !showHidden) {
          showHidden = hiddenCheck.checked = true;
        }
        setUsable(true);
        renderList();
      })
      .catch((err) => {
        // `current` still holds the last good directory so "up" can go back — but the
        // modal is no longer showing it, so it must not be selectable. Naming the
        // attempted path here keeps the label honest about what actually failed.
        if (path) pathInput.value = path;
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
    const rows = entries.map((e) => entryRow(e));
    mount(list, h("div", null, ...rows));

    // The listing was reached by naming a file (a pasted path, or a field's value):
    // point at it rather than making the user hunt down the list.
    const i = current.file ? entries.findIndex((e) => e.path === current.file) : -1;
    if (i >= 0) {
      rows[i].classList.add("hit");
      rows[i].scrollIntoView({ block: "center" });
    }
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

  load(startPath && String(startPath).trim() ? String(startPath).trim() : undefined, true);
}

function formatSize(bytes) {
  const units = ["B", "KB", "MB", "GB", "TB"];
  let n = bytes, i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(n < 10 && i > 0 ? 1 : 0)} ${units[i]}`;
}
