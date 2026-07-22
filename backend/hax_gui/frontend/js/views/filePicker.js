// Server-side file browser modal (sandboxed to the backend's file_roots).
import { h, mount, clear } from "../h.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { store } from "../store.js";
import { errorBox } from "./ui.js";

const MAP_RX = /\.(mrc|map|vol)$/i;
const MD_RX = /\.(xmd|star|cs)$/i;

// Which standalone viewer a file opens in, or null if it isn't viewable.
const viewerKind = (name) => (MAP_RX.test(name) ? "map" : MD_RX.test(name) ? "md" : null);

// Browse-only: no field to select into, so the modal exists purely to open things.
// Used by "Output folder" on a run — see jobDetail.js.
export function openFileBrowser(startPath) {
  return openFilePicker("explore", null, startPath);
}

// `kind` is "file" / "directory" (pick a value for a form field) or "explore"
// (browse-only). `startPath` opens the modal at a folder — or at a file's folder —
// instead of at the first root, so Browse resumes where the user left off.
export function openFilePicker(kind, onSelect, startPath) {
  const explore = kind === "explore";
  let current;            // last *successfully* listed directory
  let ok = false;         // is `current` what the modal is showing right now?
  let showHidden = false; // hide dotfiles by default
  // Maps and metadata ticked for opening, path -> "map" | "md". Deliberately survives
  // navigation, so a set can be gathered from several folders before opening.
  const picked = new Map();

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
  // Opening several files at once. Hidden until something is ticked, so the modal looks
  // exactly as it did before for anyone who doesn't use it.
  const pickCount = h("span", { class: "muted small" });
  const clearPick = h("button", { class: "btn ghost small", onclick: () => { picked.clear(); refreshFoot(); renderList(); } }, "Clear all");
  const openBtn = h("button", { class: "btn", onclick: openPicked }, icon("box", 16), h("span", null, "Open"));
  const foot = h("div", { class: "modal-foot" }, pickCount, h("span", { class: "ml-auto" }), clearPick, openBtn, useBtn);
  // Failures from opening go here rather than over the listing — losing the folder you
  // were working through in order to read an error message is its own small disaster.
  const pickErr = h("div", { class: "fp-openerr" });

  // "Select all in this folder", above the listing. Hidden when the folder holds
  // nothing that can be opened, which is most folders — it should not be a permanent
  // strip of chrome.
  const selectAllBox = h("input", {
    type: "checkbox", class: "fp-pick",
    onclick: (e) => {
      const on = e.target.checked;
      tickable.forEach((_, at) => setPicked(at, on));
      anchor = -1;   // a bulk change is not a sensible anchor for a later shift-range
      refreshSelectAll();
      refreshFoot();
    },
  });
  const selectAllLabel = h("span", null, "");
  const selectAllWrap = h("label", { class: "fp-selectall", hidden: true }, selectAllBox, selectAllLabel);

  const modal = h(
    "div",
    { class: "modal", onclick: (e) => e.stopPropagation() },
    h(
      "div",
      { class: "modal-head" },
      h("span", null, explore ? "Browse files" : "Select " + kind),
      h("button", { class: "btn-icon", style: { border: "none" }, onclick: close }, icon("x", 18))
    ),
    h("div", { class: "modal-path" }, upBtn, pathInput, newDir && newDir.toggle, hiddenToggle),
    newDir && newDir.row,
    pickErr,
    selectAllWrap,
    list,
    foot
  );
  refreshFoot();

  const overlay = h("div", { class: "overlay", onclick: close }, modal);
  document.body.appendChild(overlay);

  function close() { overlay.remove(); document.removeEventListener("keydown", onKey); }
  function onKey(e) { if (e.key === "Escape") close(); }
  document.addEventListener("keydown", onKey);

  // Browse-only has nothing to select into, so every path that could reach here is
  // already gated on `explore` — the guard is so a future caller can't turn that into
  // a crash.
  function finish(path) { close(); if (onSelect) onSelect(path); }

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

  // The tickable entries of the folder on screen, in display order. Range selection and
  // "all" both work over this, so both mean "as listed" — a shift-range never reaches
  // past the "Hidden" filter into rows you cannot see.
  let tickable = [];       // [{ path, kind, box }]
  let anchor = -1;         // index in `tickable` of the last box clicked without shift

  // Render the current listing, hiding dotfiles unless "Hidden" is on. Toggling
  // re-renders in place (no re-fetch).
  function renderList() {
    if (!current) return;
    const entries = current.entries.filter((e) =>
      (showHidden || !e.hidden) && (kind === "directory" ? e.is_dir : true));
    tickable = [];
    anchor = -1;
    if (!entries.length) {
      const hiddenN = current.entries.filter((e) => e.hidden && (kind === "directory" ? e.is_dir : true)).length;
      const msg = !showHidden && hiddenN ? `Empty (${hiddenN} hidden).` : "Empty.";
      refreshSelectAll();
      return mount(list, h("p", { class: "muted", style: { padding: "12px" } }, msg));
    }
    const rows = entries.map((e) => entryRow(e));
    mount(list, h("div", null, ...rows));
    refreshSelectAll();

    // The listing was reached by naming a file (a pasted path, or a field's value):
    // point at it rather than making the user hunt down the list.
    const i = current.file ? entries.findIndex((e) => e.path === current.file) : -1;
    if (i >= 0) {
      rows[i].classList.add("hit");
      rows[i].scrollIntoView({ block: "center" });
    }
  }

  // Tick or untick one entry, and paint its box.
  function setPicked(at, on) {
    const t = tickable[at];
    if (!t) return;
    if (on) picked.set(t.path, t.kind); else picked.delete(t.path);
    t.box.checked = on;
  }

  // A click on a tick box. Shift extends from the last plain click, the way a file
  // manager does it: everything between the anchor and here takes the state you just
  // gave this one, so shift-clicking an unticked box in a ticked run clears the run.
  function onTick(at, e) {
    const on = e.target.checked;
    if (e.shiftKey && anchor >= 0 && anchor !== at) {
      const [lo, hi] = at < anchor ? [at, anchor] : [anchor, at];
      for (let k = lo; k <= hi; k++) setPicked(k, on);
    } else {
      setPicked(at, on);
      anchor = at;   // only a plain click moves the anchor, so a range can be re-shifted
    }
    refreshSelectAll();
    refreshFoot();
  }

  // The "all" box above the list: ticked when every tickable entry here is picked,
  // indeterminate when only some. It acts on this folder only — picks made elsewhere
  // are none of its business.
  function refreshSelectAll() {
    const n = tickable.length;
    selectAllWrap.hidden = !n;
    if (!n) return;
    const on = tickable.filter((t) => picked.has(t.path)).length;
    selectAllBox.checked = on === n;
    selectAllBox.indeterminate = on > 0 && on < n;
    const maps = tickable.filter((t) => t.kind === "map").length;
    const parts = [];
    if (maps) parts.push(`${maps} map${maps > 1 ? "s" : ""}`);
    if (n - maps) parts.push(`${n - maps} metadata`);
    selectAllLabel.textContent = `Select all in this folder — ${parts.join(" · ")}`;
  }

  function entryRow(e) {
    const vk = e.is_dir ? null : viewerKind(e.name);
    // Browse-only has no field to fill, so a click on a viewable file opens it and a
    // click on anything else does nothing rather than silently "selecting" it.
    const activate = e.is_dir ? () => load(e.path)
      : explore ? (vk ? () => inspectFile(e.path, vk) : null)
      : () => finish(e.path);
    const row = h(
      "button",
      {
        class: "entry" + (activate ? "" : " inert"),
        disabled: activate ? null : true,
        onclick: activate,
        ondblclick: () => kind === "directory" && e.is_dir && finish(e.path),
      },
      icon(e.is_dir ? "folder" : "file", 16),
      h("span", { class: "nm" + (e.is_dir ? " folder" : "") }, e.name),
      e.size != null ? h("span", { class: "sz" }, formatSize(e.size)) : null
    );
    if (!vk) return row;
    // Volume and metadata files get a tick box (open several at once, each in its own
    // tab) and an inline action that opens just this one.
    // `click` rather than `change`: it is the only one carrying shiftKey, and by the
    // time it fires the box already holds its new state.
    const at = tickable.length;
    const box = h("input", {
      type: "checkbox", class: "fp-pick", checked: picked.has(e.path) ? "" : null,
      title: "Open several files together · shift-click to select a range",
      onclick: (ev) => { ev.stopPropagation(); onTick(at, ev); },
    });
    tickable.push({ path: e.path, kind: vk, box });
    const inspect = h("button",
      {
        class: "btn-icon",
        title: vk === "map" ? "Inspect this map (3D / orthogonal slices)" : "Browse this metadata as a table (with plots)",
        onclick: (ev) => { ev.stopPropagation(); inspectFile(e.path, vk); },
      },
      icon(vk === "map" ? "box" : "grid", 15));
    return h("div", { class: "entry-map" }, box, row, inspect);
  }

  function refreshFoot() {
    const n = picked.size;
    const maps = [...picked.values()].filter((v) => v === "map").length;
    const parts = [];
    if (maps) parts.push(`${maps} map${maps > 1 ? "s" : ""}`);
    if (n - maps) parts.push(`${n - maps} metadata`);
    pickCount.textContent = n ? `${parts.join(" · ")} selected` : "";
    openBtn.hidden = clearPick.hidden = !n;
    openBtn.lastChild.textContent = n > 1 ? `Open ${n}` : "Open";
    // Nothing to offer and no folder to accept: the bar is just a grey stripe.
    foot.hidden = !n && !useBtn;
  }

  async function inspectFile(path, kind) {
    try {
      store.openViewer(kind === "md" ? await api.mdViewer(path) : await api.mrcViewer(path));
      close();   // opens in the floating viewer window, which this modal would cover
    } catch (err) { mount(pickErr, errorBox(err)); }
  }

  // Open what was ticked. Maps go into ONE viewer — a single scene with a model panel
  // you can toggle and morph through — because that is the reason to select several of
  // them. Tables have nothing to share, so they get a tab each.
  //
  // Sequential on purpose: every viewer is a subprocess the backend spawns and
  // health-checks, so several at once are no faster and can trip the startup timeout.
  async function openPicked() {
    if (!picked.size) return;
    clear(pickErr);
    openBtn.disabled = clearPick.disabled = true;
    const maps = [...picked.entries()].filter(([, k]) => k === "map").map(([p]) => p);
    const tables = [...picked.entries()].filter(([, k]) => k !== "map").map(([p]) => p);
    // One job per viewer to be started: the maps collectively, then each table.
    const jobs = (maps.length ? [{ paths: maps, kind: "map" }] : [])
      .concat(tables.map((p) => ({ paths: [p], kind: "md" })));
    const failed = [];
    for (let i = 0; i < jobs.length; i++) {
      const { paths, kind } = jobs[i];
      pickCount.textContent = `Opening ${i + 1}/${jobs.length}…`;
      try {
        store.openViewer(kind === "md" ? await api.mdViewer(paths[0])
          : paths.length > 1 ? await api.mrcViewerMulti(paths, `${paths.length} maps`)
          : await api.mrcViewer(paths[0]));
        // So a retry after a partial failure doesn't reopen what already worked.
        for (const p of paths) picked.delete(p);
      } catch (err) {
        // Name the files while there are few enough to be worth reading: the maps open
        // as one viewer, so "2 maps failed" would leave you guessing which.
        const names = paths.map((p) => p.split("/").pop());
        const what = names.length > 3 ? `${names.length} maps` : names.join(", ");
        failed.push(`${what}: ${err && err.message ? err.message : err}`);
      }
    }
    openBtn.disabled = clearPick.disabled = false;
    // All good: get out of the way so the viewer window is visible. Otherwise stay put,
    // still showing what didn't open and with only those files still ticked.
    if (!failed.length) return close();
    mount(pickErr, errorBox(new Error(`${failed.length} of ${jobs.length} could not be opened — ${failed.join("; ")}`)));
    refreshFoot();
    renderList();
  }

  load(startPath && String(startPath).trim() ? String(startPath).trim() : undefined, true);
}

function formatSize(bytes) {
  const units = ["B", "KB", "MB", "GB", "TB"];
  let n = bytes, i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(n < 10 && i > 0 ? 1 : 0)} ${units[i]}`;
}
