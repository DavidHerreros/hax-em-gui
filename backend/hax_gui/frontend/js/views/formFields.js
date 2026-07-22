// Schema -> a single input widget, plus the two helpers that decide which fields
// a form shows at all. Grouping, search and the advanced tier live one level up
// in paramForm.js; this file only knows how to render one parameter.
import { h } from "../h.js";
import { icon } from "../icons.js";
import { openFilePicker } from "./filePicker.js";

// Args the project manager fills in automatically (output + SSD scratch dirs),
// so they're hidden from the form.
function managedArgs(schema) {
  return new Set([schema.output_arg, schema.ssd_arg].filter(Boolean));
}

// Build the initial values map from a schema, skipping the auto-managed args.
export function initialValues(schema) {
  const managed = managedArgs(schema);
  const values = {};
  for (const f of schema.fields) {
    if (managed.has(f.name)) continue;
    if (f.default === null || f.default === undefined) continue;
    values[f.name] = Array.isArray(f.default) ? f.default.join(",") : f.default;
  }
  return values;
}

// Visible fields = all fields except the auto-managed directories.
export function visibleFields(schema) {
  const managed = managedArgs(schema);
  return schema.fields.filter((f) => !managed.has(f.name));
}

// The field's default rendered as the string a text input would hold. Used both
// as the placeholder (so a sparsely-stored workflow node still shows what it will
// run with) and to decide whether a value counts as modified.
function defaultString(field) {
  const d = field.default;
  if (d === null || d === undefined) return "";
  return Array.isArray(d) ? d.join(",") : String(d);
}

// True when the stored value differs from the program's own default.
export function isModified(field, value) {
  if (value === undefined || value === null) return false;   // sparse: unset = default
  if (field.type === "boolean") return !!value !== !!field.default;
  return String(value) !== defaultString(field);
}

// One input widget. `get()` reads the current value (may be undefined when the
// caller stores only overrides), `set(v)` writes it — coercion and sparse-vs-dense
// storage are the caller's business, not this file's.
export function fieldControl(field, get, set) {
  if (field.type === "boolean") {
    const on = () => (get() === undefined ? !!field.default : !!get());
    const btn = h("button", {
      class: "toggle" + (on() ? " on" : ""), type: "button",
      role: "switch", "aria-checked": String(on()), "aria-label": field.flag,
    }, h("span", { class: "knob" }));
    btn.onclick = () => {
      const next = !on();
      set(next);
      btn.classList.toggle("on", next);
      btn.setAttribute("aria-checked", String(next));
    };
    return btn;
  }
  if (field.type === "enum") {
    return h(
      "select",
      { class: "input", "aria-label": field.flag, onchange: (e) => set(e.target.value) },
      h("option", { value: "" }, "— select —"),
      ...(field.choices || []).map((c) => h("option", { value: c, selected: get() === c }, c))
    );
  }

  const placeholder = field.widget === "path"
    ? `server ${field.path_kind || "file"} path`
    : (defaultString(field) || (field.widget === "csv" ? "comma,separated,values" : null));

  if (field.widget === "path") {
    const input = h("input", {
      class: "input", "aria-label": field.flag, value: get() ?? "", placeholder,
      oninput: (e) => set(e.target.value),
    });
    const browse = h(
      "button",
      {
        class: "btn-icon", type: "button", title: "Browse server",
        onclick: () => openFilePicker(field.path_kind || "file", (p) => { input.value = p; set(p); }, input.value),
      },
      icon("folderOpen", 16)
    );
    return h("div", { class: "path-row" }, input, browse);
  }

  const numeric = field.type === "integer" || field.type === "number";
  return h("input", {
    class: "input",
    "aria-label": field.flag,
    type: numeric ? "number" : "text",
    step: field.type === "number" ? "any" : null,
    value: get() ?? "",
    placeholder,
    oninput: (e) => set(e.target.value),
  });
}
