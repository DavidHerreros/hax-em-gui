// Right-side collapsible menu: browse programs and fill their forms inline,
// then launch them into the currently active project.
import { h, mount, clear } from "../h.js";
import { icon } from "../icons.js";
import { api, ApiError } from "../api.js";
import { navigate } from "../router.js";
import { store, getState, subscribe } from "../store.js";
import { errorBox } from "./ui.js";
import { openFilePicker } from "./filePicker.js";
import { initialValues, visibleFields, fieldRow } from "./formFields.js";

const CATEGORY_ORDER = ["Networks", "Post-processing", "Metrics", "Viewers", "Other"];

export function buildLauncher() {
  const content = h("div", { class: "launcher-content" });
  let programsCache = null;
  let formCleanup = null;
  let lastNonce;

  const programs = () => programsCache || (programsCache = api.programs());

  function render() {
    if (formCleanup) { formCleanup(); formCleanup = null; }
    clear(content);
    const sel = getState().selectedProgram;
    if (!sel) renderCatalog();
    else renderProgram(sel);
  }

  // -- catalog list ----------------------------------------------------
  function renderCatalog() {
    const search = h("input", { class: "input", placeholder: "Search programs…", oninput: (e) => filter(e.target.value) });
    const list = h("div", { class: "prog-list" });
    content.appendChild(h("div", { class: "launcher-head" }, h("span", null, "Programs")));
    content.appendChild(h("div", { style: { padding: "0 14px 10px" } }, search));
    content.appendChild(list);
    mount(list, h("p", { class: "muted", style: { padding: "12px" } }, "Loading…"));

    let all = [];
    const filter = (q) => {
      const term = q.trim().toLowerCase();
      const items = all.filter((p) => !term || p.name.toLowerCase().includes(term) || p.description.toLowerCase().includes(term));
      renderGroups(list, items);
    };
    Promise.resolve(programs())
      .then((ps) => { all = ps; renderGroups(list, ps); })
      .catch((err) => mount(list, errorBox(err)));
  }

  function renderGroups(list, items) {
    clear(list);
    const byCat = {};
    for (const p of items) (byCat[p.category] ??= []).push(p);
    const cats = Object.keys(byCat).sort((a, b) => (CATEGORY_ORDER.indexOf(a) + 1 || 99) - (CATEGORY_ORDER.indexOf(b) + 1 || 99));
    for (const cat of cats) {
      list.appendChild(h("div", { class: "prog-cat" }, cat));
      for (const p of byCat[cat]) {
        list.appendChild(
          h("button", { class: "prog-item", onclick: () => store.selectProgram(p.name) },
            h("span", { class: "pi-name" }, p.name),
            h("span", { class: "pi-desc" }, p.description))
        );
      }
    }
    if (!items.length) list.appendChild(h("p", { class: "muted", style: { padding: "12px" } }, "No matches."));
  }

  // -- single program form --------------------------------------------
  function renderProgram(name) {
    const head = h("div", { class: "launcher-head" },
      h("button", { class: "btn-icon", style: { border: "none", padding: "0 8px 0 0" }, title: "Back", onclick: () => store.clearProgram() }, icon("arrowLeft", 18)),
      h("span", { class: "mono", style: { color: "var(--accent)" } }, name));
    content.appendChild(head);
    const body = h("div", { class: "launcher-body" });
    content.appendChild(body);
    mount(body, h("p", { class: "muted", style: { padding: "12px" } }, "Loading…"));

    if (name === "display_metrics") return renderMetricsLauncher(body);

    Promise.all([api.program(name), api.gpus().catch(() => [])])
      .then(([schema, gpus]) => buildForm(body, schema, gpus))
      .catch((err) => mount(body, errorBox(err)));
  }

  function buildForm(body, schema, gpus) {
    clear(body);
    const values = initialValues(schema);
    const bindings = {};       // { field: { job_id, kind, label } } — inputs fed from a job
    let runName = "";          // optional custom name for this run
    // Default to the first GPU; "cpu" forces CUDA_VISIBLE_DEVICES="" (no GPU).
    let gpu = gpus && gpus.length ? String(gpus[0].index) : "";

    // Preload from a past job ("Edit & rerun"), keeping only visible fields so the
    // auto-managed output/SSD dirs regenerate.
    const prefill = getState().prefill;
    if (prefill && prefill.program === schema.name) {
      const visible = new Set(visibleFields(schema).map((f) => f.name));
      for (const [k, v] of Object.entries(prefill.values || {})) if (visible.has(k)) values[k] = v;
      if (prefill.gpu) gpu = String(prefill.gpu);
      store.clearPrefill();
    }

    const errorEls = {};
    const formErr = h("div", { style: { padding: "0 14px" } });
    const ctx = { bindable: schema.bindable || {}, producers: schema.producers || {}, bindings };

    const fields = h("div", { class: "launcher-fields" });
    fields.appendChild(h("p", { class: "field-help", style: { margin: "0 0 12px" } }, schema.description));

    // Optional custom run name (makes executions easy to identify).
    fields.appendChild(h("div", { class: "field" },
      h("div", { class: "field-label" }, h("span", { class: "flag" }, "Run name"), h("span", { class: "ctype" }, "optional")),
      h("input", { class: "input", placeholder: "e.g. hetsiren run 1", oninput: (e) => (runName = e.target.value) })));

    if (schema.output_arg || schema.ssd_arg) {
      const note = schema.ssd_arg
        ? "Output and SSD scratch folders are created automatically inside the project."
        : "Output folder is created automatically inside the project.";
      fields.appendChild(h("p", { class: "managed-note" }, icon("folder", 14), note));
    }
    for (const f of visibleFields(schema)) fields.appendChild(fieldRow(f, values, errorEls, ctx));
    body.appendChild(formErr);
    body.appendChild(fields);

    // run bar (reflects active project)
    const gpuSel = gpus && gpus.length
      ? h("select", { class: "input", style: { width: "auto" }, onchange: (e) => (gpu = e.target.value) },
          h("option", { value: "cpu", selected: gpu === "cpu" }, "CPU"),
          ...gpus.map((g) => h("option", { value: String(g.index), selected: String(g.index) === gpu }, `GPU ${g.index}`)))
      : null;
    const runBtn = h("button", { class: "btn", style: { flex: "1" }, onclick: () => submit(true) }, icon("play", 18), h("span", null, "Run"));
    const saveBtn = h("button", { class: "btn secondary", style: { width: "100%", justifyContent: "center" }, title: "Prepare this job and save it to run later", onclick: () => submit(false) }, icon("save", 16), h("span", null, "Save job"));
    const runBar = h("div", { class: "launcher-run" });
    body.appendChild(runBar);

    function refreshRunBar() {
      clear(runBar);
      const proj = store.activeProject();
      if (!proj) {
        runBar.appendChild(h("div", { class: "run-hint" },
          h("span", { class: "muted" }, "Open a project to run programs."),
          h("button", { class: "btn secondary", onclick: () => navigate("/projects") }, "Go to Projects")));
        return;
      }
      runBtn.disabled = false;
      saveBtn.disabled = false;
      runBar.appendChild(h("div", { class: "run-target" }, "Run in ", h("strong", null, proj.name)));
      runBar.appendChild(h("div", { class: "row", style: { gap: "8px" } }, gpuSel, runBtn));
      runBar.appendChild(h("div", { class: "row", style: { marginTop: "8px" } }, saveBtn));
    }
    formCleanup = subscribe(refreshRunBar);
    refreshRunBar();

    async function submit(launch) {
      const proj = store.activeProject();
      if (!proj) return;
      clear(formErr);
      Object.values(errorEls).forEach((e) => (e.textContent = ""));
      runBtn.disabled = true;
      saveBtn.disabled = true;
      const activeBtn = launch ? runBtn : saveBtn;
      activeBtn.lastChild.textContent = launch ? "Launching…" : "Saving…";
      try {
        // Bound fields are materialised server-side from the source job, so don't
        // also send any stale literal value for them.
        const payload = { ...values };
        for (const f of Object.keys(bindings)) delete payload[f];
        await api.createProjectJob(proj.id, schema.name, payload, gpu || null, runName, bindings, launch);
        store.setActiveProject(proj.id);
        navigate("/projects");
      } catch (err) {
        if (err instanceof ApiError && err.status === 422 && err.detail && err.detail.errors) {
          for (const [k, v] of Object.entries(err.detail.errors)) if (errorEls[k]) errorEls[k].textContent = v;
        } else {
          mount(formErr, errorBox(err));
        }
      } finally {
        // The launcher lives in the app shell and isn't unmounted by navigation,
        // so always restore the buttons (success path included).
        runBtn.disabled = false;
        saveBtn.disabled = false;
        runBtn.lastChild.textContent = "Run";
        saveBtn.lastChild.textContent = "Save job";
      }
    }
  }

  // -- display_metrics: launch a TensorBoard tab ----------------------
  function renderMetricsLauncher(body) {
    clear(body);
    let logdir = "";
    const input = h("input", { class: "input", placeholder: "server log directory", oninput: (e) => (logdir = e.target.value) });
    const browse = h("button", { class: "btn-icon", title: "Browse server", onclick: () => openFilePicker("directory", (p) => { input.value = p; logdir = p; }) }, icon("folderOpen", 16));
    const openBtn = h("button", { class: "btn", style: { width: "100%", marginTop: "10px", justifyContent: "center" }, onclick: open }, icon("chart", 18), h("span", null, "Open TensorBoard"));
    const err = h("div", { style: { marginTop: "10px" } });
    body.appendChild(h("div", { class: "launcher-fields" },
      h("p", { class: "field-help", style: { margin: "0 0 10px" } }, "Open TensorBoard for a run's output directory as a new Metrics tab."),
      h("div", { class: "path-row" }, input, browse), openBtn, err));

    async function open() {
      if (!logdir.trim()) { mount(err, errorBox("Choose a log directory first.")); return; }
      openBtn.disabled = true; openBtn.lastChild.textContent = "Starting…";
      try {
        const inst = await api.startTensorBoard(logdir.trim());
        store.openTB(inst);
        navigate("/metrics");
      } catch (e) {
        mount(err, errorBox(e));
      } finally {
        openBtn.disabled = false; openBtn.lastChild.textContent = "Open TensorBoard";
      }
    }
  }

  // re-render only when the form should rebuild (program change or prefill),
  // preserving in-progress form state on unrelated store changes.
  subscribe(() => {
    if (getState().formNonce !== lastNonce) { lastNonce = getState().formNonce; render(); }
  });
  lastNonce = getState().formNonce;
  render();

  return content;
}
