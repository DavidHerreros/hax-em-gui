// Settings: the common projects root folder shared by all projects.
import { h, mount } from "../h.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { errorBox, skeleton } from "./ui.js";
import { openFilePicker } from "./filePicker.js";
import { BranchPanel } from "./branchPanel.js";

export function SettingsView() {
  const root = h("div", { class: "view form-container" },
    h("header", { class: "page-head" },
      h("h1", null, "Settings"),
      h("p", { class: "subtitle" }, "Configuration shared across all projects.")));
  const body = h("div");
  root.appendChild(body);
  mount(body, skeleton("160px"));

  api.getSettings()
    .then((s) => mount(body, h("div", null, form(s), BranchPanel())))
    .catch((err) => mount(body, errorBox(err)));

  function pathField(label, help, current, onChange) {
    let value = current || "";
    const input = h("input", { class: "input", value, placeholder: "/path/to/folder", oninput: (e) => { value = e.target.value; onChange(value); } });
    const browse = h("button", { class: "btn-icon", title: "Browse server", onclick: () => openFilePicker("directory", (p) => { input.value = p; value = p; onChange(p); }) }, icon("folderOpen", 16));
    onChange(value);
    return h("div", { class: "card", style: { padding: "22px", marginBottom: "16px" } },
      h("h2", { class: "section" }, label),
      h("p", { class: "field-help", style: { margin: "0 0 12px" } }, help),
      h("div", { class: "path-row" }, input, browse));
  }

  function form(s) {
    const values = { projects_root: s.projects_root || "", ssd_scratch_root: s.ssd_scratch_root || "" };
    const status = h("div", { style: { marginTop: "4px" } });
    const saveBtn = h("button", { class: "btn", onclick: save }, icon("check", 18), h("span", null, "Save"));

    async function save() {
      saveBtn.disabled = true; saveBtn.lastChild.textContent = "Saving…";
      try {
        const saved = await api.putSettings({ projects_root: values.projects_root.trim(), ssd_scratch_root: values.ssd_scratch_root.trim() });
        mount(status, h("p", { class: "field-help", style: { color: "var(--ok)" } },
          `Saved. Projects: ${saved.projects_root || "—"} · SSD scratch: ${saved.ssd_scratch_root || "—"}`));
      } catch (err) {
        mount(status, errorBox(err));
      } finally {
        saveBtn.disabled = false; saveBtn.lastChild.textContent = "Save";
      }
    }

    return h("div", null,
      pathField("Projects root folder",
        "New projects are created as subfolders here. Each program run gets its own folder inside its project, holding all outputs and logs.",
        values.projects_root, (v) => (values.projects_root = v)),
      pathField("SSD scratch folder",
        "Optional fast-disk scratch space. Each run that uses it gets a folder at <ssd>/<project>/<program>. Leave empty to disable — then SSD scratch paths are simply not passed to programs.",
        values.ssd_scratch_root, (v) => (values.ssd_scratch_root = v)),
      h("div", { style: { marginTop: "4px" } }, saveBtn),
      status);
  }

  return root;
}
