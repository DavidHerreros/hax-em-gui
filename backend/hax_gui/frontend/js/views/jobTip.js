// Hover card for a run: what the one-line tree row leaves out (program, when it
// ran, how long, on which device, progress / exit code / error). One floating
// element for the whole page, shown after a short delay so sweeping the mouse
// down a list does not flash a card per row.
import { h } from "../h.js";
import { jobTime, jobDuration, fmtEta, gpuLabel } from "./jobs.js";

const DELAY_MS = 350;
let timer = null;
let tip = null;

export function hideJobTip() {
  clearTimeout(timer);
  timer = null;
  if (tip) { tip.remove(); tip = null; }
}

function build(job) {
  const at = (t) => (t ? new Date(t).toLocaleString() : null);
  const rows = [["Program", h("span", { class: "mono" }, job.program)]];
  const { label, stamp } = jobTime(job);
  if (stamp) rows.push([label[0].toUpperCase() + label.slice(1), at(stamp)]);
  if (job.started_at && job.finished_at && !job.active) rows.push(["Started", at(job.started_at)]);
  const dur = jobDuration(job);
  if (dur) rows.push(["Duration", dur]);
  const gpu = gpuLabel(job.gpu);
  if (gpu) rows.push(["Device", gpu]);
  const p = job.progress;
  if (p) rows.push(["Progress", [p.epochs ? `epoch ${p.epoch} of ${p.epochs}` : null, `${Math.round(p.pct)}%`, fmtEta(p.eta_s)].filter(Boolean).join(" · ")]);
  if (job.return_code != null && !job.active) rows.push(["Exit code", String(job.return_code)]);
  if (job.stale) rows.push(["Stale", "an input was re-run after this finished"]);
  if (job.project_name) rows.push(["Project", job.project_name]);
  const folder = (job.workdir || "").replace(/\/+$/, "").split("/").pop();
  if (folder) rows.push(["Folder", h("span", { class: "mono", style: { wordBreak: "break-all" } }, folder)]);
  return h("div", { class: "job-tip" },
    h("div", { class: "job-tip-title" }, job.name || job.program),
    h("dl", null, ...rows.map(([k, v]) => [h("dt", null, k), h("dd", null, v)])));
}

function place(el, anchor) {
  const b = anchor.getBoundingClientRect();
  const w = el.offsetWidth, hgt = el.offsetHeight;
  let x = b.right + 8, y = b.top - 6;
  if (x + w > window.innerWidth - 8) x = Math.max(8, b.left);
  if (y + hgt > window.innerHeight - 8) y = Math.max(8, window.innerHeight - hgt - 8);
  el.style.left = x + "px";
  el.style.top = y + "px";
}

// Wire `el` to show the card for `getJob()` on hover. The getter is read at show
// time, so a row that is rebuilt with fresher data shows the fresher data.
export function attachJobTip(el, getJob) {
  el.addEventListener("mouseenter", () => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      hideJobTip();
      const job = getJob();
      if (!job || !el.isConnected) return;
      tip = build(job);
      document.body.appendChild(tip);
      place(tip, el);
      tip.classList.add("show");
    }, DELAY_MS);
  });
  el.addEventListener("mouseleave", hideJobTip);
  el.addEventListener("mousedown", hideJobTip);
}
