// "Fix parameters" — one window that repairs every parameter a run would trip over,
// across every step, instead of one inspector visit per broken step.
//
// A step whose arguments don't satisfy its program schema never becomes a job: the
// launch dies in build_command, so there is no run folder and no log to read the reason
// from. The canvas works out what is wrong (stepIssues in workflows.js) and hands the
// broken steps here, each split into the fields that can be typed and the ones that
// cannot — a value arriving over a wire belongs to the graph, not to a text box.
//
// Resolves to { action: "save" | "save-run", nodes: [nodeId] } — the nodes being the
// steps to launch for "save-run", and the steps actually edited for "save" — or null
// when the window was dismissed, in which case every edit it made is rolled back.
//
// Knows nothing about the canvas on purpose: each step carries its own `recheck`, and
// jumping to a step is the caller's `onGoToStep`.
import { h } from "../h.js";
import { icon } from "../icons.js";
import { paramForm } from "./paramForm.js";

const fieldOf = (schema, name) => (schema.fields || []).find((f) => f.name === name) || null;
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

/**
 * @param steps  [{ node, schema, issues, wiredIssues, recheck }]
 *   node         the graph node — edited in place through the forms
 *   schema       its program schema (`/api/programs/{name}`)
 *   issues       fixable issues, [{ field, message, source }]
 *   wiredIssues  issues no form can settle (a wired input, an argument the program
 *                dropped) — listed, never given a control
 *   recheck(node) -> the fixable issues as they stand now; drives the live counter
 * @param onGoToStep(nodeId)  select that step on the canvas
 * @param onEdit()            called on the first keystroke of every field, so the caller
 *                            can hold off anything that would replace the node objects
 *                            these forms are writing into (the canvas' poll adopts a
 *                            fresh server graph unless it is marked dirty)
 */
export function fixParamsModal({ steps, onGoToStep, onEdit }) {
  return new Promise((resolve) => {
    const total = steps.reduce((a, s) => a + s.issues.length, 0);          // fixable here
    const totalAll = steps.reduce((a, s) => a + s.wiredIssues.length, total);
    const blocked = steps.filter((s) => s.wiredIssues.length).length;
    const touched = new Set();
    // Steps still waiting on a wire cannot be launched from here whatever is typed.
    const runnable = () => steps.filter((s) => !s.wiredIssues.length).map((s) => s.node.id);
    // Cancel has to mean cancel: the forms write straight into the live graph nodes, so
    // the only way back is the copy taken before the first keystroke.
    const before = new Map(steps.map((s) => [s.node, { ...(s.node.args || {}) }]));

    const counter = h("span", { class: "fix-count" });
    const runBtn = h("button", { class: "btn", onclick: () => done("save-run") }, icon("play", 15), "Save & run");

    const body = h("div", { class: "fix-body" },
      h("p", { class: "muted", style: { margin: "0" } },
        `${plural(steps.length, "step")} cannot run as ${steps.length === 1 ? "it stands" : "they stand"} — `
        + `${plural(totalAll, "parameter")} to settle. A step that fails to build never becomes a job, `
        + "so there is no log to read the reason from."));

    for (const s of steps) {
      const sec = h("div", { class: "fix-step" },
        h("div", { class: "fix-step-head" },
          h("span", { class: "fix-step-name" }, s.node.name || s.node.program),
          h("span", { class: "mono muted" }, s.node.program),
          h("span", { class: "muted" }, s.node.gpu === "" || s.node.gpu == null ? "CPU" : "GPU " + s.node.gpu)));

      const fields = s.issues.map((i) => fieldOf(s.schema, i.field)).filter(Boolean);
      if (fields.length) {
        const form = paramForm(s.schema, fields, {
          density: "compact",
          get: (name) => (s.node.args || {})[name],
          // Same sparse storage as the inspector: a node keeps only what differs from
          // the program's own defaults, so an emptied field is deleted, not blanked.
          set: (name, v, f) => {
            const args = (s.node.args ||= {});
            const numeric = f.type === "integer" || f.type === "number";
            if (v === "" || v == null) delete args[name];
            else args[name] = numeric ? Number(v) : v;
            touched.add(s.node.id);
            if (onEdit) onEdit();
            sync();
          },
        });
        // Flagged from the start — the whole point of this window is to say what is
        // wrong, and an error inside a collapsed group would be invisible otherwise.
        // Run while the modal is still detached: setErrors scrolls its first offender
        // into view, which would otherwise leave the window opening at the last step.
        const map = {};
        for (const i of s.issues) map[i.field] = i.message;
        form.setErrors(map);
        sec.appendChild(form.el);
      }
      if (s.wiredIssues.length) sec.appendChild(rewireBlock(s));
      body.appendChild(sec);
    }

    function rewireBlock(s) {
      return h("div", { class: "fix-rewire" },
        h("div", { class: "fix-rewire-head" }, icon("link", 13),
          h("span", null, "Needs re-wiring — can't be fixed here")),
        ...s.wiredIssues.map((i) => h("div", { class: "fix-rewire-row" },
          h("span", { class: "mono" }, i.field),
          h("span", { class: "fix-msg" }, i.message),
          i.source === "launch" ? h("span", { class: "fix-tag" }, "launch") : null)),
        h("p", { class: "fix-rewire-hint muted" },
          "This input is fed by another step, so there is no value to type here: connect it to a step that "
          + "produces it, or run the step it comes from again so its output exists."),
        h("button", {
          class: "btn ghost small",
          onclick: () => { const id = s.node.id; done(touched.size ? "save" : null); if (onGoToStep) onGoToStep(id); },
        }, icon("arrowRight", 14), "Go to step"));
    }

    function sync() {
      let left = 0;
      for (const s of steps) left += (s.recheck ? s.recheck(s.node) : s.issues).length;
      counter.textContent = `${Math.max(0, total - left)} of ${total} fixed`;
      counter.classList.toggle("done", left === 0);
      const canRun = left === 0 && runnable().length > 0;
      runBtn.disabled = !canRun;
      runBtn.title = left > 0
        ? "Give every parameter above a valid value first."
        : (!runnable().length
            ? "Every step here is waiting on a wire, so there is nothing to run yet."
            : (blocked
                ? `Runs the ${plural(runnable().length, "step")} fixed here — the ${blocked} waiting on a re-wire are skipped.`
                : "Save the workflow and run these steps now."));
    }

    // Keep every keystroke inside the window (see promptModal): the canvas underneath
    // still has Ctrl+D / Delete bound, and those would fire on the selected step while
    // you are typing in here. No Enter-to-commit — Enter belongs to the inputs.
    const onModalKey = (e) => {
      e.stopPropagation();
      if (e.key === "Escape") { e.preventDefault(); done(null); }
    };

    const modal = h("div", { class: "modal fix-modal", onclick: (e) => e.stopPropagation(), onkeydown: onModalKey },
      h("div", { class: "modal-head" }, h("span", null, "Fix parameters"),
        h("button", { class: "btn-icon", style: { border: "none" }, onclick: () => done(null) }, icon("x", 18))),
      body,
      h("div", { class: "modal-foot", style: { gap: "8px" } },
        counter,
        h("span", { class: "ml-auto" }),
        h("button", { class: "btn ghost", onclick: () => done(null) }, "Cancel"),
        h("button", { class: "btn secondary", onclick: () => done("save") }, icon("save", 15), "Save changes"),
        runBtn));

    sync();
    const overlay = h("div", { class: "overlay", onclick: () => done(null) }, modal);
    document.body.appendChild(overlay);
    // Escape fallback for the case focus has left the modal subtree (in-modal Escape is
    // handled above and stops there, so this never double-fires).
    document.addEventListener("keydown", onDocKey);

    function onDocKey(e) { if (e.key === "Escape") done(null); }
    function done(action) {
      overlay.remove();
      document.removeEventListener("keydown", onDocKey);
      if (!action) for (const [node, args] of before) node.args = args;   // dismissed: put every edit back
      resolve(action ? { action, nodes: action === "save-run" ? runnable() : [...touched] } : null);
    }
  });
}
