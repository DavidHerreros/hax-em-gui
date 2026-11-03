// Settings: the common projects root folder shared by all projects.
import { h, mount } from "../h.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { errorBox, skeleton } from "./ui.js";
import { openFilePicker } from "./filePicker.js";
import { BranchPanel } from "./branchPanel.js";

// Settings keys the setup notice can ask us to draw attention to on arrival.
let pendingHighlight = [];
export function requestSettingsHighlight(keys) {
  pendingHighlight = Array.isArray(keys) ? keys.slice() : [];
}

export function SettingsView() {
  const root = h("div", { class: "view form-container" },
    h("header", { class: "page-head" },
      h("h1", null, "Settings"),
      h("p", { class: "subtitle" }, "Configuration shared across all projects.")));
  const body = h("div");
  root.appendChild(body);
  mount(body, skeleton("160px"));

  Promise.all([api.getSettings(), api.gpus().catch(() => [])])
    .then(([s, gpus]) => { mount(body, h("div", null, form(s, gpus), BranchPanel())); applyHighlight(); })
    .catch((err) => mount(body, errorBox(err)));

  // Pulse (and focus) the fields the user was sent here to fill in.
  function applyHighlight() {
    const keys = pendingHighlight;
    pendingHighlight = [];
    const cards = keys.map((k) => body.querySelector(`#setting-${k}`)).filter(Boolean);
    if (!cards.length) return;
    cards.forEach((c) => c.classList.add("highlight"));
    cards[0].scrollIntoView({ behavior: "smooth", block: "center" });
    const first = cards[0].querySelector("input");
    if (first) first.focus({ preventScroll: true });
    setTimeout(() => cards.forEach((c) => c.classList.remove("highlight")), 2600);
  }

  function pathField(key, label, help, current, onChange) {
    let value = current || "";
    const input = h("input", { class: "input", value, placeholder: "/path/to/folder", oninput: (e) => { value = e.target.value; onChange(value); } });
    const browse = h("button", { class: "btn-icon", title: "Browse server", onclick: () => openFilePicker("directory", (p) => { input.value = p; value = p; onChange(p); }, input.value) }, icon("folderOpen", 16));
    onChange(value);
    return h("div", { class: "card", id: `setting-${key}`, style: { padding: "22px", marginBottom: "16px" } },
      h("h2", { class: "section" }, label),
      h("p", { class: "field-help", style: { margin: "0 0 12px" } }, help),
      h("div", { class: "path-row" }, input, browse));
  }

  // The device the landscape viewer pins itself to. "" = decide at launch: with two
  // or more GPUs the viewer asks, otherwise it just lets JAX choose.
  function gpuField(gpus, current, onChange) {
    const sel = h("select", { class: "input", style: { maxWidth: "420px" }, onchange: (e) => onChange(e.target.value) },
      h("option", { value: "", selected: !current }, "Ask each time"),
      ...gpus.map((g) => h("option", { value: String(g.index), selected: String(g.index) === current },
        `GPU ${g.index} — ${g.name}`)),
      h("option", { value: "cpu", selected: current === "cpu" }, "CPU (no GPU)"));
    onChange(current);
    return h("div", { class: "card", id: "setting-viewer_gpu", style: { padding: "22px", marginBottom: "16px" } },
      h("h2", { class: "section" }, "Landscape viewer device"),
      h("p", { class: "field-help", style: { margin: "0 0 12px" } },
        gpus.length
          ? "Where the conformational-landscape viewer loads the network and decodes maps. It is fixed when the "
            + "viewer opens, so pick a card with room to spare. \"Ask each time\" prompts on launch when there is "
            + "more than one GPU to choose between."
          : "No GPUs detected (nvidia-smi found none), so the viewer will decode on the CPU regardless."),
      sel);
  }

  // Where jobs execute: this machine, or a Slurm cluster. The Slurm fields only
  // show when that backend is chosen — a single-workstation user never sees them.
  function backendField(values) {
    const slurmBox = h("div", { style: { marginTop: "12px" } });
    const text = (key, label, placeholder, help) => h("div", { class: "field", style: { marginBottom: "10px" } },
      h("div", { class: "field-label" }, h("span", { class: "flag" }, label), h("span", { class: "ctype" }, "optional")),
      help ? h("p", { class: "field-help", style: { margin: "0 0 6px" } }, help) : null,
      h("input", { class: "input", value: values[key], placeholder,
                   oninput: (e) => (values[key] = e.target.value) }));
    const syncSlurm = () => { slurmBox.style.display = values.execution_backend === "slurm" ? "" : "none"; };
    const sel = h("select", { class: "input", style: { maxWidth: "420px" },
      onchange: (e) => { values.execution_backend = e.target.value; syncSlurm(); } },
      h("option", { value: "local", selected: values.execution_backend !== "slurm" }, "This machine (subprocess)"),
      h("option", { value: "slurm", selected: values.execution_backend === "slurm" }, "Slurm cluster (sbatch)"));
    slurmBox.append(
      h("p", { class: "field-help", style: { margin: "0 0 10px" } },
        "Runs are submitted with sbatch; logs stream into the GUI exactly as local runs do, and jobs "
        + "survive GUI restarts (state lives in the scheduler). The projects root must be on a filesystem "
        + "shared with the compute nodes. The per-GPU launch queue is bypassed — Slurm schedules."),
      text("slurm_partition", "partition", "e.g. gpu", ""),
      text("slurm_gres", "gres", "e.g. gpu:1", "GPUs to request per run."),
      text("slurm_time", "time limit", "e.g. 48:00:00", ""),
      text("slurm_account", "account", "", ""),
      h("div", { class: "field", style: { marginBottom: "10px" } },
        h("div", { class: "field-label" }, h("span", { class: "flag" }, "extra sbatch options"), h("span", { class: "ctype" }, "optional")),
        h("p", { class: "field-help", style: { margin: "0 0 6px" } }, "One option per line, e.g. --mem=64G"),
        h("textarea", { class: "input", rows: "2", style: { fontFamily: "monospace" },
                        oninput: (e) => (values.slurm_extra = e.target.value) }, values.slurm_extra)),
      h("div", { class: "field" },
        h("div", { class: "field-label" }, h("span", { class: "flag" }, "environment prelude"), h("span", { class: "ctype" }, "optional")),
        h("p", { class: "field-help", style: { margin: "0 0 6px" } },
          "Shell lines run on the compute node before hax — e.g. module load cuda, source activate hax."),
        h("textarea", { class: "input", rows: "3", style: { fontFamily: "monospace" },
                        oninput: (e) => (values.slurm_prelude = e.target.value) }, values.slurm_prelude)));
    syncSlurm();
    return h("div", { class: "card", id: "setting-execution_backend", style: { padding: "22px", marginBottom: "16px" } },
      h("h2", { class: "section" }, "Job execution"),
      h("p", { class: "field-help", style: { margin: "0 0 12px" } },
        "Where launched programs actually run. Applies to new launches; running jobs finish where they started."),
      sel, slurmBox);
  }

  function form(s, gpus) {
    const values = {
      projects_root: s.projects_root || "",
      ssd_scratch_root: s.ssd_scratch_root || "",
      viewer_gpu: s.viewer_gpu || "",
      execution_backend: s.execution_backend === "slurm" ? "slurm" : "local",
      slurm_partition: s.slurm_partition || "",
      slurm_gres: s.slurm_gres || "",
      slurm_time: s.slurm_time || "",
      slurm_account: s.slurm_account || "",
      slurm_extra: s.slurm_extra || "",
      slurm_prelude: s.slurm_prelude || "",
    };
    const status = h("div", { style: { marginTop: "4px" } });
    const saveBtn = h("button", { class: "btn", onclick: save }, icon("check", 18), h("span", null, "Save"));

    async function save() {
      saveBtn.disabled = true; saveBtn.lastChild.textContent = "Saving…";
      try {
        const saved = await api.putSettings({
          projects_root: values.projects_root.trim(),
          ssd_scratch_root: values.ssd_scratch_root.trim(),
          viewer_gpu: values.viewer_gpu,
          execution_backend: values.execution_backend,
          slurm_partition: values.slurm_partition.trim(),
          slurm_gres: values.slurm_gres.trim(),
          slurm_time: values.slurm_time.trim(),
          slurm_account: values.slurm_account.trim(),
          slurm_extra: values.slurm_extra,
          slurm_prelude: values.slurm_prelude,
        });
        const dev = saved.viewer_gpu ? (saved.viewer_gpu === "cpu" ? "CPU" : `GPU ${saved.viewer_gpu}`) : "ask each time";
        const exec = saved.execution_backend === "slurm" ? "Slurm" : "local";
        mount(status, h("p", { class: "field-help", style: { color: "var(--ok)" } },
          `Saved. Projects: ${saved.projects_root || "—"} · SSD scratch: ${saved.ssd_scratch_root || "—"} · Viewer: ${dev} · Execution: ${exec}`));
      } catch (err) {
        mount(status, errorBox(err));
      } finally {
        saveBtn.disabled = false; saveBtn.lastChild.textContent = "Save";
      }
    }

    return h("div", null,
      pathField("projects_root", "Projects root folder",
        "New projects are created as subfolders here. Each program run gets its own folder inside its project, holding all outputs and logs.",
        values.projects_root, (v) => (values.projects_root = v)),
      pathField("ssd_scratch_root", "SSD scratch folder",
        "Optional fast-disk scratch space. Each project gets a folder at <ssd>/<project>, shared by all its programs: the particle images are cached there once (per dataset) and reused by every run, instead of each program re-reading the original stack. Leave empty to disable — then SSD scratch paths are simply not passed to programs.",
        values.ssd_scratch_root, (v) => (values.ssd_scratch_root = v)),
      gpuField(gpus, values.viewer_gpu, (v) => (values.viewer_gpu = v)),
      backendField(values),
      h("div", { style: { marginTop: "4px" } }, saveBtn),
      status);
  }

  return root;
}
