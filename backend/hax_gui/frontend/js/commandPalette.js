// Ctrl-K command palette: views, programs, projects and recent runs from one
// keystroke. Sources are fetched fresh on open (they're all cached server-side
// or tiny), matched with a simple subsequence scorer, driven entirely by
// keyboard: type to filter, ↑/↓ to move, Enter to go, Esc to close.
import { h, clear } from "./h.js";
import { icon } from "./icons.js";
import { api } from "./api.js";
import { navigate } from "./router.js";
import { store } from "./store.js";

const VIEWS = [
  { label: "Home", hint: "view", icon: "home", go: () => navigate("/") },
  { label: "Projects", hint: "view", icon: "box", go: () => navigate("/projects") },
  { label: "Jobs", hint: "view", icon: "list", go: () => navigate("/jobs") },
  { label: "Metrics", hint: "view", icon: "chart", go: () => navigate("/metrics") },
  { label: "3D viewer", hint: "view", icon: "orbit", go: () => navigate("/viewer") },
  { label: "Workflows", hint: "view", icon: "workflow", go: () => navigate("/workflows") },
  { label: "Settings", hint: "view", icon: "gear", go: () => navigate("/settings") },
];

// Subsequence match with a crude quality score: earlier + tighter matches win,
// prefix matches win harder. Returns -1 for no match.
function score(query, text) {
  const q = query.toLowerCase(), t = text.toLowerCase();
  if (!q) return 0;
  if (t.startsWith(q)) return 1000 - t.length;
  const at = t.indexOf(q);
  if (at >= 0) return 500 - at - t.length * 0.1;
  let ti = 0, gaps = 0;
  for (const ch of q) {
    const found = t.indexOf(ch, ti);
    if (found < 0) return -1;
    gaps += found - ti;
    ti = found + 1;
  }
  return 100 - gaps - t.length * 0.1;
}

let overlay = null;

export function openPalette() {
  if (overlay) return;   // already open
  let items = [...VIEWS];
  let filtered = items;
  let sel = 0;

  const input = h("input", { class: "input palette-input", placeholder: "Jump to a view, program, project or run…" });
  const list = h("div", { class: "palette-list" });
  const modal = h("div", { class: "modal palette", onclick: (e) => e.stopPropagation() }, input, list);
  overlay = h("div", { class: "overlay palette-overlay", onclick: close }, modal);
  document.body.appendChild(overlay);
  input.focus();

  function close() {
    if (!overlay) return;
    overlay.remove();
    overlay = null;
    document.removeEventListener("keydown", onDocKey, true);
  }

  function run(item) { close(); item.go(); }

  function renderList() {
    clear(list);
    sel = Math.max(0, Math.min(sel, filtered.length - 1));
    filtered.slice(0, 12).forEach((it, i) => {
      list.appendChild(h("div", { class: "palette-item" + (i === sel ? " sel" : ""),
        onclick: () => run(it),
        onpointermove: () => { if (sel !== i) { sel = i; renderList(); } } },
        icon(it.icon, 15),
        h("span", { class: "palette-label" }, it.label),
        h("span", { class: "muted small" }, it.hint)));
    });
    if (!filtered.length) list.appendChild(h("div", { class: "muted small", style: { padding: "10px 12px" } }, "No matches."));
  }

  function refilter() {
    const q = input.value.trim();
    filtered = items
      .map((it) => ({ it, s: score(q, it.label + " " + it.hint) }))
      .filter((x) => x.s >= 0)
      .sort((a, b) => b.s - a.s)
      .map((x) => x.it);
    sel = 0;
    renderList();
  }

  input.addEventListener("input", refilter);
  function onDocKey(e) {
    if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); close(); }
    else if (e.key === "ArrowDown") { e.preventDefault(); sel = Math.min(sel + 1, Math.min(filtered.length, 12) - 1); renderList(); }
    else if (e.key === "ArrowUp") { e.preventDefault(); sel = Math.max(sel - 1, 0); renderList(); }
    else if (e.key === "Enter") { e.preventDefault(); if (filtered[sel]) run(filtered[sel]); }
  }
  document.addEventListener("keydown", onDocKey, true);
  renderList();

  // Async sources appended as they land; the current filter re-applies each time.
  api.programs().then((ps) => {
    items.push(...ps.map((p) => ({
      label: p.name, hint: "program", icon: "rocket",
      go: () => store.selectProgram(p.name),
    })));
    refilter();
  }).catch(() => {});
  api.projects().then((ps) => {
    items.push(...ps.map((p) => ({
      label: p.name, hint: "project", icon: "box",
      go: () => { store.openProject(p); navigate("/projects"); },
    })));
    refilter();
  }).catch(() => {});
  api.jobs().then((js) => {
    items.push(...js.slice(0, 30).map((j) => ({
      label: (j.name || j.program) + (j.project_name ? ` · ${j.project_name}` : ""),
      hint: `run · ${j.status}`, icon: "play",
      go: () => navigate(`/jobs/${j.id}`),
    })));
    refilter();
  }).catch(() => {});
}

export function initPalette() {
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
      e.preventDefault();
      openPalette();
    }
  });
}
