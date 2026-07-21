// "Carry on where you left off?" — for workflow runs the server was restarted out from
// under, whose next step therefore never started.
//
// This is deliberately *not* shown for every restart. A workflow whose step was still
// running is resumed silently at startup (nothing new is launched — the scheduler simply
// re-adopts the job, which now survives the server; see jobs.py), and one that finished
// while the server was down just settles. So a routine GUI update never raises this.
//
// It appears only when resuming would actually *launch* a training with nobody watching:
// the step that was running finished while we were down, and the next one is sitting
// ready. That is the moment worth asking about — not "was the server down for long",
// which is only a proxy for it, and a leaky one (a training happily survives an overnight
// outage; another can be OOM-killed during a five-second restart). See
// workflows.py `_plan_resume`.
import { h, mount } from "../h.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { navigate } from "../router.js";

export function ResumeNotice() {
  const slot = h("div");
  // Advisory: never let it break the page it sits on.
  api.resumableWorkflows().then((list) => render(slot, list)).catch(() => {});
  return slot;
}

function ago(iso) {
  const t = Date.parse(iso);
  if (!Number.isFinite(t)) return "";
  const mins = Math.max(0, Math.round((Date.now() - t) / 60000));
  if (mins < 60) return `${mins} min ago`;
  const hrs = Math.round(mins / 60);
  if (hrs < 48) return `${hrs} h ago`;
  return `${Math.round(hrs / 24)} days ago`;
}

function render(slot, list) {
  if (!list || !list.length) return;

  const rows = new Map();
  const listEl = h("ul", { class: "setup-list" });

  const drop = (id) => {
    rows.get(id)?.remove();
    rows.delete(id);
    if (!rows.size) mount(slot, null);       // last one decided: the banner is done
  };

  for (const w of list) {
    const resumeBtn = h("button", { class: "btn", onclick: async () => {
      resumeBtn.disabled = true;
      try { await api.resumeWorkflow(w.id); drop(w.id); navigate("/workflows"); }
      catch { resumeBtn.disabled = false; }
    } }, icon("play", 14), "Resume");

    const dismissBtn = h("button", { class: "btn secondary", onclick: async () => {
      dismissBtn.disabled = true;
      try { await api.dismissWorkflowResume(w.id); drop(w.id); }
      catch { dismissBtn.disabled = false; }
    } }, icon("x", 14), "Dismiss");

    const li = h("li", { class: "resume-row" },
      h("div", { class: "resume-what" },
        h("strong", null, w.name || "Untitled workflow"),
        h("span", { class: "muted" },
          ` — ${w.done} of ${w.total} steps finished, ${w.remaining} still queued`),
        w.updated_at ? h("span", { class: "muted" }, ` · last active ${ago(w.updated_at)}`) : null),
      h("div", { class: "resume-actions" }, resumeBtn, dismissBtn));
    rows.set(w.id, li);
    listEl.appendChild(li);
  }

  mount(slot, h("div", { class: "setup-banner" },
    h("div", { class: "setup-banner-icon" }, icon("workflow", 20)),
    h("div", { class: "setup-banner-body" },
      h("h3", null, list.length === 1
        ? "A workflow run was interrupted"
        : `${list.length} workflow runs were interrupted`),
      h("p", { class: "setup-lead" },
        "The server stopped between steps, so the next step never started. Resuming carries "
        + "on from there — the steps that already finished keep their results and are not run again."),
      listEl)));
}
