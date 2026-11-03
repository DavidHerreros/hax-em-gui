// Finished/failed-run notifications: title badge always, browser Notification
// when the tab is hidden and permission was granted. Without this a user had to
// keep the tab open and poll by eye to learn a 12-hour training died.
import { onJobEvent } from "./events.js";
import { toast } from "./toast.js";

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
      toast(m.title, { kind: m.level === "warn" ? "error" : "info", timeout: 12000 });
      if ("Notification" in window && Notification.permission === "granted") {
        try {
          const n = new Notification("hax-em watch", { body: m.title, tag: `watch-${m.job_id}-${m.alert}` });
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
