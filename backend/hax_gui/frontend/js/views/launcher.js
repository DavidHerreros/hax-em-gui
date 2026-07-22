// Right-side collapsible menu: browse programs and fill their forms inline,
// then launch them into the currently active project.
import { h, mount, clear } from "../h.js";
import { icon } from "../icons.js";
import { api, ApiError } from "../api.js";
import { navigate } from "../router.js";
import { store, getState, subscribe } from "../store.js";
import { errorBox } from "./ui.js";
import { initialValues, visibleFields } from "./formFields.js";
import { paramForm } from "./paramForm.js";
import { bindingControl } from "./inputBinding.js";

const CATEGORY_ORDER = ["Preprocessing", "Networks", "Post-processing", "Metrics", "Viewers", "Other"];

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

    Promise.all([api.program(name), api.gpus().catch(() => [])])
      .then(([schema, gpus]) => buildForm(body, schema, gpus))
      .catch((err) => mount(body, errorBox(err)));
  }

  function buildForm(body, schema, gpus) {
    clear(body);
    const values = initialValues(schema);
    const bindings = {};       // { field: { job_id, kind, label } } — inputs fed from a job
    let runName = "";          // optional custom name for this run
    let recreateSsd = false;   // wipe + rebuild the shared SSD scratch before running
    // Default to the first GPU; "cpu" forces CUDA_VISIBLE_DEVICES="" (no GPU).
    let gpu = gpus && gpus.length ? String(gpus[0].index) : "";

    // The job this form edits in place (same id), so Save/Run reuse it instead of
    // creating duplicates: set when editing an existing job, and also after the
    // first Save so a following Run launches that same saved job. null = create new.
    let targetJobId = null;
    // "Edit & continue": reuse targetJobId's folder (resume from its checkpoint) rather
    // than replacing it with a fresh run.
    let resume = false;

    // Preload from a past job ("Edit" / "Edit & rerun"), keeping only visible fields
    // so the auto-managed output/SSD dirs regenerate. Editing reuses the source job.
    const prefill = getState().prefill;
    if (prefill && prefill.program === schema.name) {
      const visible = new Set(visibleFields(schema).map((f) => f.name));
      for (const [k, v] of Object.entries(prefill.values || {})) if (visible.has(k)) values[k] = v;
      if (prefill.gpu) gpu = String(prefill.gpu);
      if (prefill.jobId) targetJobId = prefill.jobId;
      resume = !!prefill.resume;
      store.clearPrefill();
    }

    const formErr = h("div", { style: { padding: "0 14px" } });
    const bindable = schema.bindable || {};

    // Everything above the search bar: what the program is, plus the run-level
    // settings that aren't program parameters at all.
    const intro = h("div", { class: "pf-intro" });
    intro.appendChild(h("p", { class: "field-help", style: { margin: "0 0 12px" } }, schema.description));

    // Optional custom run name (makes executions easy to identify).
    intro.appendChild(h("div", { class: "field" },
      h("div", { class: "field-label" }, h("span", { class: "flag" }, "Run name"), h("span", { class: "ctype" }, "optional")),
      h("input", { class: "input", placeholder: "e.g. hetsiren run 1", oninput: (e) => (runName = e.target.value) })));

    if (schema.output_arg || schema.ssd_arg) {
      const note = schema.ssd_arg
        ? "Output and SSD scratch folders are created automatically inside the project."
        : "Output folder is created automatically inside the project.";
      intro.appendChild(h("p", { class: "managed-note" }, icon("folder", 14), note));
    }
    // The SSD scratch is shared per project+program and reused across runs; offer to
    // rebuild it from scratch (applies only when a scratch root is configured).
    if (schema.ssd_arg) {
      intro.appendChild(h("label", { class: "recreate-ssd", title: "Delete the cached images of THIS dataset (mmap / array-record shards) so they are rebuilt from the current inputs. The scratch folder is shared by every program in the project — other datasets cached there are left untouched. Only applies when a scratch root is configured in Settings." },
        h("input", { type: "checkbox", onchange: (e) => (recreateSsd = e.target.checked) }),
        h("span", null, "Recreate the SSD scratch cache"),
        h("span", { class: "muted small" }, "discard this dataset's cached images and rebuild for this run")));
    }

    const form = paramForm(schema, visibleFields(schema), {
      get: (name) => values[name],
      set: (name, v) => { values[name] = v; },
      density: "comfortable",
      intro,
      // Bindable inputs get the "use a job's output" control (and become a drop
      // target for dragged runs) underneath the plain value input.
      augment: (field, control, row) => {
        const kind = bindable[field.name];
        if (!kind) return null;
        row.classList.add("bindable");
        // Path fields wrap their input in a .path-row; plain ones are the input.
        const valueInput = control.querySelector("input.input")
          || (control.tagName === "INPUT" ? control : null);
        if (bindings[field.name] && valueInput) valueInput.disabled = true;
        return bindingControl({
          field, kind,
          producers: (schema.producers && schema.producers[kind]) || [],
          bindings,
          fieldEl: row,
          valueInput,
          onChange: () => form.clearErrors(),
        });
      },
    });
    body.appendChild(formErr);
    body.appendChild(form.el);

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
      form.clearErrors();
      runBtn.disabled = true;
      saveBtn.disabled = true;
      const activeBtn = launch ? runBtn : saveBtn;
      activeBtn.lastChild.textContent = launch ? "Launching…" : "Saving…";
      try {
        // Bound fields are materialised server-side from the source job, so don't
        // also send any stale literal value for them.
        const payload = { ...values };
        for (const f of Object.keys(bindings)) delete payload[f];
        // Reuse the edited / just-saved job in place (no duplicate). The server
        // ignores a stale target (deleted / already running) and creates fresh.
        const created = await api.createProjectJob(proj.id, schema.name, payload, gpu || null, runName, bindings, launch, recreateSsd, targetJobId, resume);
        // Remember the id so a subsequent Run/Save in this same form keeps reusing
        // it (e.g. Save job → Run launches that saved job rather than a new one).
        if (created && created.id) targetJobId = created.id;
        store.setActiveProject(proj.id);
        navigate("/projects");
      } catch (err) {
        if (err instanceof ApiError && err.status === 422 && err.detail && err.detail.errors) {
          // Opens the groups holding the offending fields and scrolls to the first.
          form.setErrors(err.detail.errors);
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

  // re-render only when the form should rebuild (program change or prefill),
  // preserving in-progress form state on unrelated store changes.
  subscribe(() => {
    if (getState().formNonce !== lastNonce) { lastNonce = getState().formNonce; render(); }
  });
  lastNonce = getState().formNonce;
  render();

  return content;
}
