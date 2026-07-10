// Shared schema-driven form widgets, used by the right-menu program launcher.
import { h } from "../h.js";
import { icon } from "../icons.js";
import { openFilePicker } from "./filePicker.js";
import { bindingControl } from "./inputBinding.js";

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

// `ctx` (optional) enables job-output chaining on bindable fields:
//   { bindable: {field: kind}, producers: {kind: [programs]}, bindings: {} }
export function fieldRow(field, values, errorEls, ctx) {
  const err = h("p", { class: "field-error" });
  errorEls[field.name] = err;
  const input = fieldInput(field, values);
  const wrap = h(
    "div",
    { class: "field" },
    h(
      "div",
      { class: "field-label" },
      h("span", { class: "flag" }, field.flag, field.required ? h("span", { class: "req" }, "*") : null),
      field.custom_type ? h("span", { class: "ctype" }, field.custom_type) : null
    ),
    input,
    field.help ? h("p", { class: "field-help" }, field.help) : null,
    err
  );

  const kind = ctx && ctx.bindable ? ctx.bindable[field.name] : null;
  if (kind) {
    wrap.classList.add("bindable");
    const valueInput = input.querySelector ? input.querySelector("input.input") : (input.tagName === "INPUT" ? input : null);
    const ctrl = bindingControl({
      field, kind,
      producers: (ctx.producers && ctx.producers[kind]) || [],
      bindings: ctx.bindings,
      fieldEl: wrap,
      valueInput,
      onChange: () => { err.textContent = ""; },
    });
    input.after(ctrl);
    if (ctx.bindings[field.name] && valueInput) valueInput.disabled = true;
  }
  return wrap;
}

function fieldInput(field, values) {
  const get = () => values[field.name];
  const set = (v) => (values[field.name] = v);

  if (field.type === "boolean") {
    const btn = h("button", { class: "toggle" + (get() ? " on" : ""), type: "button" }, h("span", { class: "knob" }));
    btn.onclick = () => { set(!get()); btn.classList.toggle("on", !!get()); };
    return btn;
  }
  if (field.type === "enum") {
    return h(
      "select",
      { class: "input", onchange: (e) => set(e.target.value) },
      h("option", { value: "" }, "— select —"),
      ...(field.choices || []).map((c) => h("option", { value: c, selected: get() === c }, c))
    );
  }
  if (field.widget === "path") {
    const input = h("input", { class: "input", value: get() ?? "", placeholder: `server ${field.path_kind || "file"} path`, oninput: (e) => set(e.target.value) });
    const browse = h(
      "button",
      { class: "btn-icon", type: "button", title: "Browse server", onclick: () => openFilePicker(field.path_kind || "file", (p) => { input.value = p; set(p); }, input.value) },
      icon("folderOpen", 16)
    );
    return h("div", { class: "path-row" }, input, browse);
  }
  const numeric = field.type === "integer" || field.type === "number";
  return h("input", {
    class: "input",
    type: numeric ? "number" : "text",
    step: field.type === "number" ? "any" : null,
    value: get() ?? "",
    placeholder: field.widget === "csv" ? "comma,separated,values" : null,
    oninput: (e) => set(e.target.value),
  });
}
