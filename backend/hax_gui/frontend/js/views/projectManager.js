// Project-manager rail: a collapsible lateral panel of project "flash cards".
// Clicking a card opens the project as a tab in the workspace; each card's
// three-dot menu can rename or delete the project. Lives inside ProjectsView.
import { h, mount, clear } from "../h.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { store, getState, subscribe } from "../store.js";
import { errorBox, formatSize } from "./ui.js";
import { poller } from "../poller.js";
import { onJobEvent } from "../events.js";
import { openContextMenu } from "./contextMenu.js";

export function ProjectManager() {
  const root = h("div", { class: "project-manager" });

  // Collapsed strip: a thin rail with an expand button + vertical label.
  const rail = h("div", { class: "pm-rail" },
    h("button", { class: "pm-rail-toggle", title: "Show projects", onclick: () => store.togglePM() }, icon("chevronRight", 16)),
    h("div", { class: "pm-vlabel" }, "PROJECTS"));

  // Expanded body.
  const nameInput = h("input", { class: "input pm-input", placeholder: "New project…",
    onkeydown: (e) => { if (e.key === "Enter") create(); } });
  const addBtn = h("button", { class: "pm-add", title: "Create project", onclick: create }, icon("plus", 16));
  const cards = h("div", { class: "pm-cards" });
  // Disk usage is opt-in: measuring means walking every project folder, which on
  // shared storage is worth asking for rather than doing behind the user's back. Once
  // measured the sizes stay on the cards (the server caches them for a couple of
  // minutes, so a second click is instant).
  const diskBtn = h("button", { class: "pm-collapse", title: "Show how much disk each project uses",
    onclick: measureDisk }, icon("box", 16));
  const body = h("div", { class: "pm-body" },
    h("div", { class: "pm-head" },
      h("span", { class: "pm-title" }, "Projects"),
      diskBtn,
      h("button", { class: "pm-collapse", title: "Collapse", onclick: () => store.togglePM() }, icon("chevronLeft", 16))),
    h("div", { class: "pm-new" }, nameInput, addBtn),
    cards);

  root.append(rail, body);

  let projects = [];
  let sig = null;
  let renamingId = null;
  let stop = false;
  let diskById = new Map();      // project id -> bytes, once measured

  async function measureDisk() {
    diskBtn.disabled = true;
    diskBtn.title = "Measuring…";
    try {
      // Re-measure on an explicit second click; the first one takes whatever is
      // already cached so it comes straight back.
      const rows = await api.projectsUsage(diskById.size > 0);
      if (stop) return;
      diskById = new Map(rows.map((r) => [r.id, r.bytes]));
      renderCards();
    } catch (err) {
      if (!stop) mount(cards, errorBox(err));
    } finally {
      diskBtn.disabled = false;
      diskBtn.title = "Re-measure how much disk each project uses";
    }
  }

  async function load() {
    try {
      const list = await api.projects();
      if (stop) return;
      // A rename input is open: never rebuild the cards under it (the poll used
      // to wipe the user's text every 5 s — the old condition was inverted).
      // sig is dropped so the first poll after the rename closes repaints.
      if (renamingId !== null) { sig = null; return; }
      const s = list.map((p) => `${p.id}:${p.name}:${p.runs}`).join("|");
      if (s === sig) return; // unchanged -> avoid flicker
      sig = s; projects = list; renderCards();
    } catch (err) {
      sig = null; // next successful poll must repaint over the error box
      if (!stop) mount(cards, errorBox(err));
    }
  }

  function renderCards() {
    clear(cards);
    if (!projects.length) {
      cards.appendChild(h("p", { class: "pm-empty muted" }, "No projects yet. Create one above to get started."));
      return;
    }
    const activeId = getState().activeProjectId;
    for (const p of projects) cards.appendChild(card(p, p.id === activeId));
  }

  function card(p, active) {
    const el = h("div", { class: "pm-card" + (active ? " active" : ""), "data-id": p.id, onclick: () => open(p) });

    let nameEl;
    if (renamingId === p.id) {
      nameEl = h("input", { class: "input pm-rename", value: p.name,
        onclick: (e) => e.stopPropagation(),
        onkeydown: (e) => {
          if (e.key === "Enter") commitRename(p, e.target.value);
          else if (e.key === "Escape") { renamingId = null; renderCards(); }
        },
        onblur: (e) => commitRename(p, e.target.value) });
      // focus after mount
      queueMicrotask(() => { nameEl.focus(); nameEl.select(); });
    } else {
      nameEl = h("span", { class: "pm-name", title: p.name }, p.name);
    }

    const menuBtn = h("button", { class: "pm-menu", title: "Options",
      onclick: (e) => { e.stopPropagation(); openMenu(e.currentTarget, p); } }, icon("more", 16));

    const runs = p.runs || 0;
    el.append(
      h("div", { class: "pm-card-top" }, icon("box", 16), nameEl, menuBtn),
      h("div", { class: "pm-meta" },
        h("span", null, `${runs} ${runs === 1 ? "run" : "runs"}`),
        p.created_at ? h("span", { class: "pm-sep" }, "·") : null,
        p.created_at ? h("span", null, new Date(p.created_at).toLocaleDateString()) : null,
        diskById.has(p.id) ? h("span", { class: "pm-sep" }, "·") : null,
        diskById.has(p.id) ? h("span", { class: "mono", title: "Disk used by this project folder" }, formatSize(diskById.get(p.id))) : null),
      p.path ? h("div", { class: "pm-path mono", title: p.path }, p.path) : null);
    return el;
  }

  function open(p) {
    if (renamingId === p.id) return; // ignore clicks while editing this card's name
    store.openProject(p);
  }

  function openMenu(anchor, p) {
    openContextMenu(anchor, [
      { label: "Open", icon: icon("arrowRight", 15), onClick: () => store.openProject(p) },
      { label: "Rename", icon: icon("pencil", 15), onClick: () => { renamingId = p.id; renderCards(); } },
      { label: "Delete", icon: icon("trash", 15), danger: true, onClick: () => remove(p) },
    ]);
  }

  async function commitRename(p, value) {
    const name = (value || "").trim();
    renamingId = null;
    if (!name || name === p.name) { renderCards(); return; }
    try {
      const updated = await api.renameProject(p.id, name);
      p.name = updated.name;
      store.renameOpenProject(p.id, updated.name);
      sig = null; // force next poll to redraw
      renderCards();
    } catch (err) {
      mount(cards, errorBox(err));
    }
  }

  async function remove(p) {
    if (!window.confirm(`Delete project “${p.name}” and ALL of its runs (outputs + logs)?\nThis cannot be undone.`)) return;
    try {
      await api.deleteProject(p.id);
      store.closeProject(p.id);
      projects = projects.filter((x) => x.id !== p.id);
      sig = null;
      renderCards();
    } catch (err) {
      alert(err && err.detail ? (typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail)) : "Failed to delete project.");
    }
  }

  async function create() {
    const name = nameInput.value.trim();
    if (!name) return;
    addBtn.disabled = true;
    try {
      const p = await api.createProject(name);
      nameInput.value = "";
      store.openProject(p);
      sig = null;
      await load();
    } catch (err) {
      alert(err && err.detail ? (typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail)) : "Failed to create project.");
    } finally {
      addBtn.disabled = false;
    }
  }

  // React to store: collapse state + active-card highlight (without rebuilding
  // cards, so a rename input keeps focus).
  function syncStore() {
    root.classList.toggle("collapsed", getState().pmCollapsed);
    const activeId = getState().activeProjectId;
    for (const el of cards.querySelectorAll(".pm-card")) el.classList.toggle("active", el.getAttribute("data-id") === activeId);
  }
  const unsub = subscribe(syncStore);
  syncStore();

  const p = poller(load, 5000);
  const offEvents = onJobEvent(() => p.kick());   // push beats the 5 s tick

  return { el: root, cleanup: () => { stop = true; p.stop(); unsub(); offEvents(); } };
}
