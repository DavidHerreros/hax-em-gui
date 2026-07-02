// App bootstrap: collapsible left sidebar + main outlet + collapsible right
// program launcher. Routing for the top-level sections; tabs live in the views.
import { h, mount } from "./h.js";
import { icon } from "./icons.js";
import { defineRoutes, setOutlet, onRoute, startRouter } from "./router.js";
import { getToken, setToken } from "./api.js";
import { store, getState, subscribe } from "./store.js";
import { buildLauncher } from "./views/launcher.js";
import { startUpdates, updateBadge } from "./updates.js";
import { loadProgramGroups } from "./programGroups.js";
import { mountViewerWindow } from "./views/viewerWindow.js";
import { HomeView } from "./views/home.js";
import { ProjectsView } from "./views/projects.js";
import { SettingsView } from "./views/settings.js";
import { MetricsView } from "./views/metrics.js";
import { ViewerView } from "./views/viewer.js";
import { JobsView } from "./views/jobs.js";
import { JobDetailView } from "./views/jobDetail.js";
import { WorkflowsView } from "./views/workflows.js";

const DOCS_URL = "https://davidherreros.github.io/hax-em-docs/";
const THEME_KEY = "hax_gui_theme";

const NAV = [
  { path: "/", label: "Home", icon: "home", match: (p) => p === "/" },
  { path: "/projects", label: "Projects", icon: "box", match: (p) => p.startsWith("/projects") },
  { path: "/jobs", label: "Jobs", icon: "list", match: (p) => p.startsWith("/jobs") },
  { path: "/metrics", label: "Metrics", icon: "chart", match: (p) => p.startsWith("/metrics") },
  { path: "/viewer", label: "3D viewer", icon: "orbit", match: (p) => p.startsWith("/viewer") },
  { path: "/workflows", label: "Workflows", icon: "workflow", match: (p) => p.startsWith("/workflows") },
];

// ---- theme ----
function initTheme() { document.documentElement.dataset.theme = localStorage.getItem(THEME_KEY) || "dark"; }
function currentTheme() { return document.documentElement.dataset.theme === "light" ? "light" : "dark"; }
function makeThemeToggle() {
  const render = () => {
    const dark = currentTheme() === "dark";
    btn.replaceChildren(icon(dark ? "sun" : "moon"), labelSpan(dark ? "Light mode" : "Dark mode"));
  };
  const btn = h("button", { class: "side-item theme-toggle", onclick: () => {
    const next = currentTheme() === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem(THEME_KEY, next);
    window.dispatchEvent(new CustomEvent("hax-theme", { detail: next }));
    render();
  } });
  render();
  return btn;
}

function labelSpan(text) { return h("span", { class: "side-label" }, text); }

function buildSidebar() {
  const navLinks = NAV.map((n) => h("a", { href: "#" + n.path, "data-path": n.path, class: "side-item" }, icon(n.icon), labelSpan(n.label)));

  const collapseBtn = h("button", { class: "collapse-btn", title: "Collapse menu", onclick: () => store.toggleLeft() }, icon("chevronLeft", 18));
  subscribe(() => collapseBtn.replaceChildren(icon(getState().leftCollapsed ? "chevronRight" : "chevronLeft", 18)));

  const settingsLink = h("a", { href: "#/settings", "data-path": "/settings", class: "side-item" }, icon("gear"), labelSpan("Settings"));
  const tokenBtn = h("button", { class: "side-item", onclick: () => {
    const t = window.prompt("Access token (leave empty if the server has none):", getToken());
    if (t !== null) { setToken(t.trim()); location.reload(); }
  } }, icon("cpu"), labelSpan("Access token"));
  const docsLink = h("a", { href: DOCS_URL, target: "_blank", rel: "noreferrer", class: "side-item docs" }, icon("book"), labelSpan("Documentation"));

  const sidebar = h("aside", { class: "sidebar" },
    h("div", { class: "brand" },
      h("img", { src: "/assets/logo_small.png", alt: "Hax" }),
      h("span", { class: "side-label brand-name" }, "Hax"),
      updateBadge(),
      collapseBtn),
    h("nav", { class: "nav" }, ...navLinks),
    h("div", { class: "spacer" }),
    h("div", { class: "side-footer" }, settingsLink, makeThemeToggle(), tokenBtn, docsLink));

  // active-state on route change
  const allLinks = [...navLinks, settingsLink];
  onRoute((path) => {
    for (const a of allLinks) {
      const item = [...NAV, { path: "/settings", match: (p) => p.startsWith("/settings") }].find((x) => x.path === a.getAttribute("data-path"));
      a.classList.toggle("active", item.match(path));
    }
  });
  return sidebar;
}

function buildRightMenu() {
  const railToggle = h("button", { class: "rail-toggle", title: "Toggle programs menu", onclick: () => store.toggleRight() }, icon("chevronRight", 18));
  const rail = h("div", { class: "launcher-rail" }, railToggle, h("div", { class: "rail-vlabel" }, "PROGRAMS"));
  const panel = h("div", { class: "launcher-panel" }, buildLauncher());
  const menu = h("aside", { class: "launcher" }, rail, panel);

  const syncRail = () => railToggle.firstChild.replaceWith(icon(getState().rightOpen ? "chevronRight" : "chevronLeft", 18));
  subscribe(syncRail);
  syncRail();
  return menu;
}

initTheme();
const appEl = h("div", { class: "app" });
const outlet = h("div", { class: "container" });
appEl.appendChild(buildSidebar());
appEl.appendChild(h("main", { class: "main" }, outlet));
appEl.appendChild(buildRightMenu());
mount(document.getElementById("app"), appEl);
setOutlet(outlet);

// reflect collapse state as classes on the shell
function applyShellClasses() {
  appEl.classList.toggle("left-collapsed", getState().leftCollapsed);
  appEl.classList.toggle("right-closed", !getState().rightOpen);
}
subscribe(applyShellClasses);
applyShellClasses();

defineRoutes([
  { path: "/", view: HomeView },
  { path: "/projects", view: ProjectsView },
  { path: "/jobs", view: JobsView },
  { path: "/jobs/:id", view: JobDetailView },
  { path: "/metrics", view: MetricsView },
  { path: "/viewer", view: ViewerView },
  { path: "/settings", view: SettingsView },
  { path: "/workflows", view: WorkflowsView },
]);

startRouter();
startUpdates();
loadProgramGroups();   // derive program groupings (moded / predict-latents) from the backend
mountViewerWindow();   // floating window that hosts all opened viewers as tabs
