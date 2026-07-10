// First-run notice on the home page, nagging until the shared roots are set.
//
// Two flavours, and the difference matters:
//   * unset roots    — advisory. "Close forever" silences them server-side.
//   * missing roots  — a root that *was* configured and whose folder has since
//                      vanished. Always shown, even after "Close forever", since
//                      every run pointing at it will now fail.
//
// "Remind me later" is scoped to the server's id in sessionStorage, so it comes
// back after a server restart or in a fresh browser session, but survives a reload.
import { h, mount } from "../h.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { navigate } from "../router.js";
import { requestSettingsHighlight } from "./settings.js";

const SNOOZE_KEY = "hax_gui_setup_snoozed_for";

const LABELS = {
  projects_root: "Projects root folder",
  ssd_scratch_root: "SSD scratch folder",
};

const WHY = {
  projects_root: "projects and their runs have nowhere to live until this is set",
  ssd_scratch_root: "optional, but runs fall back to slower storage without it",
};

export function SetupNotice() {
  const slot = h("div");
  // Advisory only — never let it break the landing page.
  api.setupNotice().then((s) => render(slot, s)).catch(() => {});
  return slot;
}

function isSnoozed(serverId) {
  try { return sessionStorage.getItem(SNOOZE_KEY) === serverId; } catch { return false; }
}

function snooze(serverId) {
  try { sessionStorage.setItem(SNOOZE_KEY, serverId); } catch { /* private mode */ }
}

function render(slot, s) {
  if (!s.show || isSnoozed(s.server_id)) return;

  const gone = s.missing || [];
  const todo = (s.unset || []).filter((k) => !(s.silenced || []).includes(k));
  const urgent = gone.length > 0;

  const remindBtn = h("button", { class: "btn secondary", onclick: () => { snooze(s.server_id); mount(slot, null); } },
    icon("clock", 16), "Remind me later");

  const setBtn = h("button", { class: "btn", onclick: () => { requestSettingsHighlight([...gone, ...todo]); navigate("/settings"); } },
    icon("gear", 16), "Set");

  // Offered only when nothing is actually broken: silencing hides the unset
  // roots, but a missing folder would re-surface immediately anyway.
  const foreverBtn = urgent ? null : h("button", { class: "btn secondary", onclick: closeForever }, icon("x", 16), "Close forever");

  async function closeForever() {
    foreverBtn.disabled = true;
    try { await api.silenceSetupNotice(); mount(slot, null); }
    catch { foreverBtn.disabled = false; }
  }

  mount(slot, h("div", { class: "setup-banner" + (urgent ? " urgent" : "") },
    h("div", { class: "setup-banner-icon" }, icon(urgent ? "ban" : "gear", 20)),
    h("div", { class: "setup-banner-body" },
      h("h3", null, urgent
        ? (gone.length === 1 ? "A configured folder is missing" : "Configured folders are missing")
        : "Finish setting up Hax"),
      h("p", { class: "setup-lead" }, urgent
        ? "These folders are set in Settings but no longer exist on disk. Runs that point at them will fail."
        : "Hax needs to know where to keep your projects before you can run anything."),
      h("ul", { class: "setup-list" },
        ...gone.map((k) => h("li", null,
          h("strong", null, LABELS[k]), " — missing: ", h("code", null, (s.values || {})[k] || "?"))),
        ...todo.map((k) => h("li", null,
          h("strong", null, LABELS[k]), " — not set; ", WHY[k]))),
      h("div", { class: "setup-actions" }, setBtn, remindBtn, foreverBtn))));
}
