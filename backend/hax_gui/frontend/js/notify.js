// Finished/failed-run notifications: title badge always, browser Notification
// when the tab is hidden and permission was granted. Without this a user had to
// keep the tab open and poll by eye to learn a 12-hour training died.
import { api } from "./api.js";
import { onJobEvent } from "./events.js";
import { h } from "./h.js";
import { toast, toastError } from "./toast.js";

const TERMINAL = {
  succeeded: "finished",
  failed: "FAILED",
  interrupted: "was interrupted",
};

let pending = 0;
const baseTitle = document.title;

function updateTitle() {
  document.title = pending > 0 ? `(${pending}) ${baseTitle}` : baseTitle;
}

// A watch alert: the finding, its measured evidence, and the assistant's proposal.
// Apply only appears when the autonomy level asks for confirmation; the server
// re-derives the changes, the click only names the finding being accepted.
function watchToast(m) {
  const body = h("div", null,
    h("div", null, m.title),
    m.evidence ? h("div", { class: "toast-detail" }, m.evidence) : null,
    m.proposal ? h("div", { class: "toast-detail" }, m.proposal.text) : null,
    m.blocked ? h("div", { class: "toast-detail" }, `Not acted on: ${m.blocked}`) : null);
  let dismiss = () => {};
  if (m.needs_confirmation) {
    const apply = h("button", { class: "btn small" }, "Apply");
    apply.addEventListener("click", async (e) => {
      e.stopPropagation();
      apply.disabled = true;
      try {
        const r = await api.assistantWatchApply(m.job_id, m.alert);
        toast(`${m.name || "run"} ${r.message}`, { kind: r.applied ? "success" : "info", timeout: 10000 });
        dismiss();
      } catch (err) {
        apply.disabled = false;
        toastError(err, "Could not apply the proposal.");
      }
    });
    body.appendChild(h("div", { class: "toast-actions" }, apply,
      h("span", { class: "muted" }, "or click to dismiss")));
  }
  dismiss = toast(body, { kind: m.level === "warn" ? "error" : "info",
                          timeout: m.needs_confirmation ? 0 : (m.proposal ? 30000 : 12000) });
}

export function startNotifications() {
  // Permission can only be requested from a user gesture; piggyback on the
  // first click. No prompt spam: the browser remembers the answer.
  const askOnce = () => {
    if ("Notification" in window && Notification.permission === "default") {
      Notification.requestPermission().catch(() => { /* user dismissed */ });
    }
    window.removeEventListener("pointerdown", askOnce);
  };
  window.addEventListener("pointerdown", askOnce);

  // Coming back to the tab clears the badge — the user is looking now.
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) { pending = 0; updateTitle(); }
  });

  onJobEvent((m) => {
    if (m.kind === "watch") {
      watchToast(m);
      if ("Notification" in window && Notification.permission === "granted") {
        try {
          const body = m.evidence ? `${m.title}\n${m.evidence}` : m.title;
          const n = new Notification("hax-em watch", { body, tag: `watch-${m.job_id}-${m.alert}` });
          n.onclick = () => { window.focus(); location.hash = `#/jobs/${m.job_id}`; n.close(); };
        } catch { /* notification API unavailable */ }
      }
      return;
    }
    if (m.kind !== "job" || !(m.status in TERMINAL)) return;
    if (!document.hidden) return;             // visible tab: the list updates in view
    pending += 1;
    updateTitle();
    if ("Notification" in window && Notification.permission === "granted") {
      const label = m.name || m.program || "job";
      const body = m.status === "failed" && m.return_code != null
        ? `exit code ${m.return_code}` : "";
      try {
        const n = new Notification(`hax-em: ${label} ${TERMINAL[m.status]}`,
                                   { body, tag: `hax-job-${m.job_id}` });
        n.onclick = () => { window.focus(); location.hash = `#/jobs/${m.job_id}`; n.close(); };
      } catch { /* notification constructor blocked (e.g. insecure context) */ }
    }
  });
}
