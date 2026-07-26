// "This will invalidate the steps below" — the dialog every rerun goes through.
//
// Re-running a step rebuilds its run folder, and everything downstream reads its inputs
// from that folder, so those runs stop matching anything that exists (backend: cascade.py).
// They are never silently destroyed: their outputs and logs stay put, flagged stale, and
// the user picks whether to re-run them now or later.
//
// Returns a Promise of { cascade: "stale" | "queue", cancel_running: boolean }, or null
// if the rerun was called off. With nothing downstream there is nothing to decide, so no
// dialog is shown and the default answer comes straight back.
import { h, clear } from "../h.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { badge } from "./jobs.js";

// The last choice, reused as the default. Sticky within the session only: it is a
// preference about how you work, not something worth persisting past a reload, and a
// destructive-ish default that survives forever is exactly the kind that surprises you.
let lastChoice = "stale";

// Everything `ids` feed in `graph`, transitively — the canvas' own copy of what the
// backend computes from bindings. Both agree because the canvas is saved before it runs.
export function downstreamNodes(graph, ids) {
  const start = new Set(ids);
  const seen = new Set(ids);
  const queue = [...ids];
  while (queue.length) {
    const cur = queue.shift();
    for (const e of graph.edges || []) {
      if (e.from === cur && !seen.has(e.to)) { seen.add(e.to); queue.push(e.to); }
    }
  }
  return (graph.nodes || []).filter((n) => seen.has(n.id) && !start.has(n.id));
}

// Ask about the runs downstream of one job (Jobs / job detail views).
export async function confirmJobCascade(jobId, opts = {}) {
  let deps = [];
  try { deps = (await api.jobDependents(jobId)).jobs || []; }
  catch { deps = []; }          // can't enumerate them: don't block the rerun over it
  return confirmCascade(deps, opts);
}

// `dependents`: [{ id?, label, program, status, active, stale }] — the canvas builds
// these from its nodes, the job views from the API.
export function confirmCascade(dependents, opts = {}) {
  const {
    title = "Re-run this step?",
    verb = "Re-run",
    // What the rerun does to *this* step's own outputs, said plainly up front.
    lead = "Its run folder is rebuilt from scratch — the current outputs and log are replaced.",
  } = opts;

  if (!dependents.length) return Promise.resolve({ cascade: lastChoice, cancel_running: false });

  const running = dependents.filter((d) => d.active);
  const n = dependents.length;

  return new Promise((resolve) => {
    let choice = lastChoice;
    let cancelOk = !running.length;

    const okBtn = h("button", { class: "btn", onclick: () => done({ cascade: choice, cancel_running: !!running.length }) },
      icon("play", 15), verb);

    const runningRow = running.length ? h("label", { class: "cascade-cancel" },
      h("input", {
        type: "checkbox",
        onchange: (e) => { cancelOk = e.currentTarget.checked; syncOk(); },
      }),
      h("span", null, `Also stop the ${running.length} downstream run${running.length === 1 ? "" : "s"} still executing`),
      h("span", { class: "muted" }, " — they are reading the folder this rerun deletes, so they cannot be left running.")) : null;

    function syncOk() { okBtn.disabled = !cancelOk; }

    const options = h("div", { class: "cascade-opts" },
      optionRow("stale", "Leave them as they are, flagged stale",
        "Their outputs and logs are kept so you can still look at them, marked out of date. Re-run them yourself whenever you like."),
      optionRow("queue", "Queue them to re-run automatically",
        "Each one re-runs as soon as the step it depends on succeeds, in order, one per GPU. Their current outputs are replaced when their turn comes."));

    function optionRow(value, label, hint) {
      const input = h("input", { type: "radio", name: "cascade-mode", value, checked: choice === value,
        onchange: () => { choice = value; } });
      return h("label", { class: "cascade-opt" }, input,
        h("div", null, h("div", { class: "cascade-opt-label" }, label), h("div", { class: "muted" }, hint)));
    }

    const modal = h("div", { class: "modal cascade-modal", onclick: (e) => e.stopPropagation() },
      h("div", { class: "modal-head" }, h("span", null, title),
        h("button", { class: "btn-icon", style: { border: "none" }, onclick: () => done(null) }, icon("x", 18))),
      h("div", { class: "cascade-body" },
        h("p", { class: "muted", style: { marginTop: "0" } }, lead),
        h("p", null, `${n} later step${n === 1 ? "" : "s"} ${n === 1 ? "reads" : "read"} its output. Whatever ${n === 1 ? "it has" : "they have"} already produced stops matching anything that exists:`),
        h("div", { class: "cascade-list" }, ...dependents.map(depRow)),
        runningRow,
        h("div", { class: "cascade-q" }, `What should happen to ${n === 1 ? "it" : "them"}?`),
        options),
      h("div", { class: "modal-foot", style: { gap: "8px" } },
        h("button", { class: "btn ghost", onclick: () => done(null) }, "Cancel"),
        okBtn));

    syncOk();
    const overlay = h("div", { class: "overlay", onclick: () => done(null) }, modal);
    document.body.appendChild(overlay);
    document.addEventListener("keydown", onKey);

    function onKey(e) { if (e.key === "Escape") done(null); }
    function done(v) {
      overlay.remove();
      document.removeEventListener("keydown", onKey);
      if (v) lastChoice = v.cascade;
      resolve(v);
    }
  });
}

function depRow(d) {
  return h("div", { class: "cascade-item" + (d.active ? " active" : "") },
    icon("box", 14),
    h("span", { class: "nm", title: d.program }, d.label || d.program),
    h("span", { class: "mono muted" }, d.program),
    // No status means it has never run: nothing to lose, but "queue" still runs it, so
    // it belongs in the list rather than being a surprise afterwards.
    d.status ? badge(d.status) : h("span", { class: "muted", style: { fontSize: "12px" } }, "not run yet"));
}

// A canvas node in the shape the dialog wants. Its status comes from the run state,
// which is what the node itself is drawn from.
export function nodeAsDependent(node, runState) {
  const s = (runState && runState[node.id]) || {};
  return {
    id: s.job_id || null,
    label: node.name || node.program,
    program: node.program,
    status: s.status && s.status !== "idle" ? s.status : null,
    active: s.status === "running" || s.status === "pending",
    stale: false,
  };
}
