// App bootstrap: collapsible left sidebar + main outlet + collapsible right
// program launcher. Routing for the top-level sections; tabs live in the views.
import { h, mount } from "./h.js";
import { icon } from "./icons.js";
import { defineRoutes, setOutlet, onRoute, startRouter, currentPath, reload, navigate } from "./router.js";
import { api, ApiError, getToken, setToken } from "./api.js";
import { store, getState, subscribe } from "./store.js";
import { buildLauncher } from "./views/launcher.js";
import { startUpdates, updateBadge } from "./updates.js";
import { startNotifications } from "./notify.js";
import { loadProgramGroups } from "./programGroups.js";
import { poller } from "./poller.js";
import { mountViewerWindow } from "./views/viewerWindow.js";
import { initPalette } from "./commandPalette.js";
import { promptModal } from "./views/promptModal.js";
import { HomeView } from "./views/home.js";
import { SettingsView } from "./views/settings.js";
import { MetricsView } from "./views/metrics.js";
import { ViewerView } from "./views/viewer.js";
import { JobsView } from "./views/jobs.js";
import { CompareView } from "./views/compare.js";
import { WorkflowsView } from "./views/workflows.js";
import { AssistantView } from "./views/assistant.js";

const DOCS_URL = "https://davidherreros.github.io/hax-em-docs/";
const THEME_KEY = "hax_gui_theme";

const NAV = [
  { path: "/", label: "Home", icon: "home", match: (p) => p === "/" },
  { path: "/project", label: "Project", icon: "box", match: (p) => p.startsWith("/project") || p.startsWith("/jobs") },
  { path: "/metrics", label: "Metrics", icon: "chart", match: (p) => p.startsWith("/metrics") },
  { path: "/viewer", label: "3D viewer", icon: "orbit", match: (p) => p.startsWith("/viewer") },
  { path: "/workflows", label: "Workflows", icon: "workflow", match: (p) => p.startsWith("/workflows") },
];

// The AI assistant is an optional, separately-installed package. Probed in the
// background — this used to be a top-level await, so a slow or down backend
// held the entire module (a blank page) hostage to one request. When the probe
// lands, the route and nav item appear; until then the GUI is simply without it.
function probeAssistant() {
  api.assistantStatus().then((s) => {
    if (!s.available) return;
    defineRoutes([{ path: "/assistant", view: AssistantView }]);
    const nav = document.querySelector(".sidebar .nav");
    if (nav) nav.appendChild(h("a", { href: "#/assistant", "data-path": "/assistant", class: "side-item" },
                              icon("chat"), labelSpan("Assistant")));
    if (currentPath().startsWith("/assistant")) reload();   // deep link opened early
  }).catch(() => { /* offline or endpoint missing: stay hidden */ });
}

// ---- theme ----
function initTheme() {
  const stored = localStorage.getItem(THEME_KEY);
  const preferred = window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  document.documentElement.dataset.theme = stored || preferred;
}
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

// Shared by the sidebar "Access token" button and the startup auth gate below:
// prompt for a token, validate it against the server before committing (a user
// who pastes a token from the wrong machine used to get stuck in a silent
// paste → reload → modal-reappears loop), then store it and reload. Resolves
// false iff the user cancelled (the caller never sees true — a reload follows).
async function promptForToken(message) {
  let label = message;
  let prefill = getToken();
  for (;;) {
    const t = await promptModal({ title: "Access token", okText: "Apply", label, value: prefill });
    if (t === null) return false;
    const status = await api.authCheckWith(t);
    if (status === 401) {
      // Re-prompt in place, with what they typed still in the box, instead of
      // storing a token we already know is wrong.
      label = "That token was rejected by the server. " + message;
      prefill = t;
      continue;
    }
    // status 0 (network failure) or anything else (200, or a 404 from an older
    // server without this endpoint) — apply it. Backend-down is the offline
    // bar's business, not this gate's.
    setToken(t);
    location.reload();
    return true;
  }
}

function buildSidebar() {
  const navLinks = NAV.map((n) => h("a", { href: "#" + n.path, "data-path": n.path, class: "side-item" }, icon(n.icon), labelSpan(n.label)));

  const collapseBtn = h("button", { class: "collapse-btn", title: "Collapse menu", onclick: () => store.toggleLeft() }, icon("chevronLeft", 18));
  subscribe(() => collapseBtn.replaceChildren(icon(getState().leftCollapsed ? "chevronRight" : "chevronLeft", 18)));

  const settingsLink = h("a", { href: "#/settings", "data-path": "/settings", class: "side-item" }, icon("gear"), labelSpan("Settings"));
  const tokenBtn = h("button", { class: "side-item",
    onclick: () => promptForToken("Bearer token for this server (leave empty if it has none):"),
  }, icon("cpu"), labelSpan("Access token"));
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
  { path: "/project", view: JobsView },
  // Old bookmarks: the Projects page and the Jobs list are now one page.
  { path: "/projects", view: () => { navigate("/project"); return null; } },
  { path: "/jobs", view: JobsView },
  { path: "/jobs/:id", view: JobsView },
  { path: "/compare/:a/:b", view: CompareView },
  { path: "/metrics", view: MetricsView },
  { path: "/viewer", view: ViewerView },
  { path: "/settings", view: SettingsView },
  { path: "/workflows", view: WorkflowsView },
]);

startRouter();
startUpdates();
startNotifications();   // /ws/events → title badge + browser Notification
probeAssistant();

// ---- startup auth gate --------------------------------------------------
// The old failure mode: a public server needs a token, but the first the user
// hears of it is an "Unauthorized" error box when they open some view. Instead,
// probe /api/auth/check right away and prompt for the token up front — before
// any view has a chance to render an auth error.
const authBar = h("div", { class: "auth-bar", hidden: true },
                  "This server requires an access token — the API will refuse everything until one is set.",
                  h("button", { class: "btn small", onclick: () => promptForToken(authPromptMessage()) }, "Set token"));
document.body.appendChild(authBar);

function authPromptMessage() {
  const prefix = getToken() ? "The stored access token was rejected. " : "";
  return prefix + "This server requires an access token. It was printed in the terminal where the "
    + "server was started (hax-gui start) and is stored in the token file on that machine — the "
    + "machine running the server, not the one running your browser. Paste it here:";
}

(async () => {
  try {
    const h2 = await api.health();
    if (!h2.auth_required) return;   // open server: no gate, no extra request
    await api.authCheck();
  } catch (err) {
    // Network failures / non-401 errors are none of this gate's business — the
    // offline bar below already covers a backend that isn't reachable at all.
    if (!(err instanceof ApiError) || err.status !== 401) return;
    const applied = await promptForToken(authPromptMessage());
    if (!applied) authBar.hidden = false;   // cancelled: leave a persistent way back in
  }
})();

// ---- backend-gone indicator -------------------------------------------------
// A fetch that *throws* is a network failure (server down / unreachable); any
// HTTP answer — even a 401 — means the backend is there.
const offlineBar = h("div", { class: "offline-bar", hidden: true },
                     "Backend unreachable — retrying…");
document.body.appendChild(offlineBar);
poller(async () => {
  try { await api.health(); offlineBar.hidden = true; }
  catch (err) { offlineBar.hidden = err instanceof ApiError; }
}, 10000);
loadProgramGroups();   // derive program groupings (moded / predict-latents) from the backend
mountViewerWindow();   // floating window that hosts all opened viewers as tabs
initPalette();         // Ctrl-K / Cmd-K: jump to any view, program, project or run
