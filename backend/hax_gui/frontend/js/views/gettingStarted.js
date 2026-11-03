// "Getting started" on the home page: a checklist that knows how far you've got,
// and empties itself as you go.
//
// It replaces four static cards that described the app in the abstract. Nothing in
// them was clickable, so every card ended exactly where the user had to act, and
// they looked identical on a first visit and a three-hundredth. Here each step
// carries one button that *performs* it, and a condition the front end can already
// answer from endpoints the rest of the GUI calls anyway — no new API:
//
//     projects root set   <- /api/setup-notice   (the setup banner's own source)
//     a project exists    <- /api/projects
//     a run was launched  <- /api/jobs           (status !== "saved")
//     a run succeeded     <- the same call
//     a workflow exists   <- /api/workflows
//
// Finished steps shrink to a ticked line; once all five are done the block collapses
// to a single row, and "Hide" removes it for good. Onboarding that never leaves stops
// being read, and it costs a returning user the top of their home page.
//
// The last step is workflows on purpose: the builder is the strongest reason to use
// the GUI over the CLI, and the page it lives behind never mentioned it.
//
// The probe is advisory throughout — every call falls back, and if all of them fail
// (server down, no token) the block renders nothing rather than telling an
// experienced user they have not started yet.
import { h, mount } from "../h.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { navigate } from "../router.js";
import { store, getState, subscribe } from "../store.js";
import { coachmark } from "../coachmark.js";
import { requestSettingsHighlight } from "./settings.js";
import { openProjectChooser } from "./projectOverview.js";
import { rememberWorkflowToOpen } from "./workflows.js";

const HIDE_KEY = "hax_gui_getting_started_hidden";

const hidden = () => { try { return localStorage.getItem(HIDE_KEY) === "1"; } catch { return false; } };
const hide = () => { try { localStorage.setItem(HIDE_KEY, "1"); } catch { /* private mode */ } };

// Each step's `done` reads the probe below; `act` is the single thing the button does.
const STEPS = [
  {
    title: "Point Hax at a folder",
    doneAs: "Projects folder set",
    desc: "Hax keeps every project and every run under one folder on the server. Set it once.",
    why: "Opens Settings with the field highlighted.",
    action: "Choose folder",
    done: (s) => s.rootSet === true,
    act() { requestSettingsHighlight(["projects_root"]); navigate("/settings"); },
  },
  {
    title: "Create a project",
    doneAs: "Project created",
    desc: "A project is a workspace: its runs, their outputs and their history stay together.",
    why: "Opens the chooser here — you don't leave the page.",
    action: "New project",
    done: (s) => s.projects > 0,
    act(refresh) {
      openProjectChooser();
      // The chooser has no completion callback, so watch the store instead: picking
      // or creating a project is exactly what `openProject` emits.
      const off = subscribe(() => { off(); refresh(); });
    },
  },
  {
    title: "Run a program",
    doneAs: "First run launched",
    desc: "Every Hax program is in the menu on the right — it starts collapsed. The form is generated from the "
      + "program itself, so it can never drift out of date.",
    why: "Required arguments are marked; the rest have working defaults.",
    action: "Show me",
    done: (s) => s.launched > 0,
    act() {
      // The one thing a checklist can't do on its own: point at the rail. Open it
      // first — but the panel animates its width over 200ms, so measuring on the next
      // frame would spotlight a half-open menu.
      const wasOpen = getState().rightOpen;
      store.setRightOpen(true);
      const mark = () => coachmark(".launcher", {
        title: "Programs live here",
        text: "This menu is collapsed by default, which is why it's easy to miss. Pick a program, fill the form, "
          + "and launch it into the project you have open.",
      });
      if (wasOpen) mark();
      else afterTransition(document.querySelector(".launcher-panel"), mark);
    },
  },
  {
    title: "See what came out",
    doneAs: "A run finished",
    desc: "Loss curves, image panels and the latent space update while the run is still training. Maps and tables "
      + "open in the 3D viewer.",
    why: "You don't have to wait for a run to end to see it going wrong.",
    action: "Open Metrics",
    done: (s) => s.succeeded > 0,
    act() { navigate("/metrics"); },
  },
  {
    title: "Chain it into a workflow",
    doneAs: "Workflow built",
    desc: "Drag programs onto a canvas and wire one run's output into the next one's input. Hax runs the chain "
      + "unattended.",
    why: "This is the part that saves you the most time.",
    action: "Open the builder",
    done: (s) => s.workflows > 0,
    act() { navigate("/workflows"); },
  },
];

// The shortest real path from a particle stack to volumes you can look at. A first
// user's blocking question is "which program do I run, and on what?" — which the old
// section never answered; it named no programs at all.
//
// Two HetSIREN nodes is not a typo: `train` learns the landscape and yields the
// checkpoint, `predict` assigns every particle its latent. Decoding needs both, which
// is why the wiring below is worth generating rather than describing.
const RECIPE = [
  { program: "preprocess_particles", args: {}, blurb: "your .star / .xmd stack → a cropped, CTF-corrected one" },
  { program: "hetsiren", args: { mode: "train" }, blurb: "learns the conformational landscape" },
  { program: "hetsiren", args: { mode: "predict" }, blurb: "gives every particle a latent" },
  { program: "decode_states_from_latents", args: {}, blurb: "turns latents back into volumes" },
];

// Run `fn` once `el` has finished animating. The timeout is not belt-and-braces: a
// transition that never starts (reduced motion, or the property didn't actually
// change) fires no event at all, and the caller would wait forever.
function afterTransition(el, fn) {
  if (!el) return requestAnimationFrame(fn);
  let done = false;
  const go = () => { if (done) return; done = true; el.removeEventListener("transitionend", go); fn(); };
  el.addEventListener("transitionend", go);
  setTimeout(go, 320);
}

export function GettingStarted() {
  const slot = h("div");
  if (hidden()) return slot;
  let shown = false;
  const refresh = () => probe().then((s) => { shown = true; render(slot, s, refresh); })
                              .catch(() => { if (!shown) mount(slot, null); });
  refresh();
  return slot;
}

// One round trip per signal, all of them advisory. `null` means "couldn't tell",
// which is treated as not-done for that step but never as a reason to hide the block
// — unless every probe failed, in which case there is nothing to say.
async function probe() {
  const [rootSet, projects, jobs, workflows] = await Promise.all([
    api.setupNotice().then((s) => !(s.unset || []).includes("projects_root")
                              && !(s.missing || []).includes("projects_root")).catch(() => null),
    api.projects().then((p) => p.length).catch(() => null),
    api.jobs().catch(() => null),
    api.workflows().then((w) => w.length).catch(() => null),
  ]);
  if (rootSet === null && projects === null && jobs === null && workflows === null) throw new Error("offline");
  return {
    rootSet,
    projects: projects || 0,
    // "saved" is a job that was filled in but never launched, so it doesn't count.
    launched: jobs ? jobs.filter((j) => j.status !== "saved").length : 0,
    succeeded: jobs ? jobs.filter((j) => j.status === "succeeded").length : 0,
    workflows: workflows || 0,
  };
}

function render(slot, state, refresh) {
  const done = STEPS.map((s) => s.done(state));
  const count = done.filter(Boolean).length;
  const all = count === STEPS.length;

  if (all) return mount(slot, collapsed(slot, state, refresh));

  const list = h("ol", { class: "gs-list" }, ...STEPS.map((s, i) => stepRow(s, i, done, refresh)));
  mount(slot, h("section", { class: "gs" },
    h("div", { class: "gs-head" },
      h("h2", null, "Getting started"),
      h("span", { class: "gs-count" }, `${count} of ${STEPS.length} done`)),
    h("p", { class: "gs-sub" }, "Five steps from an empty install to a pipeline that runs itself."),
    h("div", { class: "gs-track" }, h("div", { class: "gs-fill", style: { width: `${(100 * count) / STEPS.length}%` } })),
    list,
    recipe()));
}

function collapsed(slot, state, refresh) {
  const row = h("div", { class: "gs gs-done" },
    h("span", { class: "gs-tick" }, icon("check", 16)),
    h("span", { class: "gs-donetext" }, "Getting started — all five steps done."),
    h("button", { class: "gs-link", onclick: () => mount(slot, expanded(slot, state, refresh)) }, "Show"),
    h("button", { class: "gs-link", onclick: () => { hide(); mount(slot, null); } }, "Hide for good"));
  return row;
}

// The finished list, re-shown on request. Steps stay ticked; the recipe is gone,
// since by definition this user has already built something.
function expanded(slot, state, refresh) {
  const done = STEPS.map(() => true);
  return h("section", { class: "gs" },
    h("div", { class: "gs-head" },
      h("h2", null, "Getting started"),
      h("button", { class: "gs-link", onclick: () => mount(slot, collapsed(slot, state, refresh)) }, "Collapse")),
    h("ol", { class: "gs-list" }, ...STEPS.map((s, i) => stepRow(s, i, done, refresh))));
}

function stepRow(s, i, done, refresh) {
  const isDone = done[i];
  // "Now" is the first unfinished step: the one place a primary button belongs.
  const isNow = !isDone && done.slice(0, i).every(Boolean);
  const body = h("div", null, h("h3", null, isDone ? s.doneAs : s.title));
  if (!isDone) {
    body.appendChild(h("p", null, s.desc));
    body.appendChild(h("span", { class: "gs-why" }, s.why));
  }
  return h("li", { class: "gs-step" + (isDone ? " done" : isNow ? " now" : "") },
    h("span", { class: "gs-dot" }, isDone ? icon("check", 13) : String(i + 1)),
    body,
    isDone ? h("span", { class: "gs-tag" }, "done")
           : h("button", { class: "btn" + (isNow ? "" : " ghost"), onclick: () => s.act(refresh) }, s.action));
}

function recipe() {
  const err = h("p", { class: "gs-err", hidden: true });
  const build = h("button", { class: "btn ghost", onclick: async () => {
    build.disabled = true;
    err.hidden = true;
    try { await buildStarter(); }
    catch (e) { err.textContent = "Couldn't build it: " + (e.detail || e.message); err.hidden = false; build.disabled = false; }
  } }, icon("workflow", 15), "Build this workflow");

  return h("div", { class: "gs-recipe" },
    h("h3", null, "Not sure what to run? Start here."),
    h("p", null, "The shortest path from a particle stack to a conformational landscape — each step fed by the last."),
    h("div", { class: "gs-chain" }, ...RECIPE.flatMap((r, i) => [
      i ? h("span", { class: "gs-arrow" }, icon("arrowRight", 14)) : null,
      h("div", { class: "gs-prog" },
        h("span", { class: "gs-nm" }, r.program + (r.args.mode ? ` (${r.args.mode})` : "")),
        h("span", { class: "gs-io" }, r.blurb)),
    ])),
    h("div", { class: "gs-recipe-foot" }, build,
      h("span", { class: "gs-why" }, "Creates it as a draft on the canvas — nothing runs until you fill the arguments and press Run.")),
    err);
}

// A node's output kinds, narrowed by its mode — a moded network offers its trained
// checkpoint only in `train` and its predict artifacts (latents, particles) only in
// `predict`. This mirrors `nodeProduces` in workflows.js, which mirrors
// adapters.job_produces; wiring the recipe by hand instead would bake in an answer
// that silently goes stale the next time those tables change.
function producesOf(entry, node) {
  let all = (entry && entry.produces) || [];
  if (!node.args.transport_mass) all = all.filter((k) => k !== "transport_motion_model");
  if (node.args.mode === "predict") return all.filter((k) => !k.endsWith("_model"));
  if (node.args.mode === "train") return all.filter((k) => k.endsWith("_model"));
  return all;
}

// The recipe as a real graph. Edges are derived from the catalog's own ports rather
// than hardcoded, so a program whose inputs change here yields *fewer* edges — never
// wrong ones — and an input the recipe can't satisfy is simply left unwired for the
// user to fill in.
export function starterGraph(byName) {
  const absent = [...new Set(RECIPE.map((r) => r.program).filter((n) => !byName[n]))];
  if (absent.length) throw new Error(`this hax install has no ${absent.join(", ")}`);

  const nodes = RECIPE.map((r, i) => ({ id: "n" + i, program: r.program, args: { ...r.args },
                                        gpu: "0", name: "", labels: [], x: 0, y: 0 }));
  const edges = [];
  for (let i = 1; i < nodes.length; i++) {
    for (const [field, kind] of Object.entries(byName[nodes[i].program].bindable || {})) {
      // Nearest upstream producer wins, so `predict` reloads the train node beside it
      // rather than something further back that happens to offer the same kind.
      for (let j = i - 1; j >= 0; j--) {
        if (!producesOf(byName[nodes[j].program], nodes[j]).includes(kind)) continue;
        edges.push({ id: "e" + edges.length, from: nodes[j].id, to: nodes[i].id, field, kind });
        break;
      }
    }
  }
  // All-zero coordinates are the builder's signal to lay a graph out on open.
  return { nodes, edges, orientation: "horizontal" };
}

// Build the recipe and open it in the builder.
async function buildStarter() {
  const catalog = await api.workflowCatalog();
  const graph = starterGraph(Object.fromEntries(catalog.map((p) => [p.name, p])));
  const wf = await api.createWorkflow({
    project_id: getState().activeProjectId || null,
    name: "Starter pipeline",
    graph,
  });
  rememberWorkflowToOpen(wf.id);
  navigate("/workflows");
}
