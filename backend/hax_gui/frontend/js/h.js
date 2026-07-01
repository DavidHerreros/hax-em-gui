// Minimal hyperscript helper — builds DOM nodes safely (text via textContent,
// never innerHTML unless an explicit `html` prop is passed). ~30 lines, no deps.

export function h(tag, props, ...children) {
  const el = document.createElement(tag);
  if (props) {
    for (const [k, v] of Object.entries(props)) {
      if (v == null || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "html") el.innerHTML = v; // only for trusted inline SVG
      else if (k === "style" && typeof v === "object") setStyle(el, v);
      else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2).toLowerCase(), v);
      else if (v === true) el.setAttribute(k, "");
      else el.setAttribute(k, String(v));
    }
  }
  append(el, children);
  return el;
}

// Apply a style object. CSS custom properties (`--foo`) must go through
// setProperty — assigning them as named keys on el.style is silently ignored.
function setStyle(el, style) {
  for (const [k, v] of Object.entries(style)) {
    if (v == null) continue;
    if (k.startsWith("--")) el.style.setProperty(k, v);
    else el.style[k] = v;
  }
}

export function append(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false) continue;
    el.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}

export function clear(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
}

// Replace the contents of `el` with `node` (used by the router and live views).
export function mount(el, node) {
  clear(el);
  if (node) el.appendChild(node);
}
