// Minimal vanilla terminal renderer for job logs: interprets ANSI SGR colors,
// carriage returns / backspace / tabs (so progress bars update in place), and
// erase-line. No third-party libraries. Renders into a scrollable container.

const MAX_LINES = 5000;

// One Dark-ish palette — reads well on the dark console background.
const PALETTE = {
  30: "#5c6370", 31: "#e06c75", 32: "#98c379", 33: "#e5c07b",
  34: "#61afef", 35: "#c678dd", 36: "#56b6c2", 37: "#abb2bf",
  90: "#7f848e", 91: "#ff7b86", 92: "#b5e890", 93: "#f0d399",
  94: "#7cc5ff", 95: "#e29bf0", 96: "#74d3de", 97: "#ffffff",
};

function defaultStyle() { return { fg: null, bold: false, italic: false, underline: false }; }
function styleKey(s) { return `${s.fg || ""}|${s.bold}|${s.italic}|${s.underline}`; }

function ansi256(n) {
  const base16 = [30, 31, 32, 33, 34, 35, 36, 37, 90, 91, 92, 93, 94, 95, 96, 97];
  if (n < 16) return PALETTE[base16[n]];
  if (n >= 232) { const v = 8 + (n - 232) * 10; return `rgb(${v},${v},${v})`; }
  n -= 16;
  const r = Math.floor(n / 36), g = Math.floor((n % 36) / 6), b = n % 6;
  const c = (x) => (x ? 55 + x * 40 : 0);
  return `rgb(${c(r)},${c(g)},${c(b)})`;
}

export function createTerminal(container) {
  let style = defaultStyle();
  let cur = { cells: [], col: 0 };   // current (editable) line
  let node = lineNode();
  let stick = true;

  container.replaceChildren(node);
  container.addEventListener("scroll", () => {
    stick = container.scrollHeight - container.scrollTop - container.clientHeight < 40;
  });

  function lineNode() { const d = document.createElement("div"); d.className = "tline"; return d; }

  function putChar(ch) {
    cur.cells[cur.col] = { ch, style: { ...style } };
    cur.col += 1;
  }

  function commitLine() {
    renderLine(node, cur.cells);
    node = lineNode();
    container.appendChild(node);
    cur = { cells: [], col: 0 };
    while (container.childNodes.length > MAX_LINES + 1) container.removeChild(container.firstChild);
  }

  function applySGR(params) {
    const codes = params === "" ? [0] : params.split(";").map((x) => parseInt(x || "0", 10));
    for (let k = 0; k < codes.length; k++) {
      const c = codes[k];
      if (c === 0) style = defaultStyle();
      else if (c === 1) style.bold = true;
      else if (c === 22) style.bold = false;
      else if (c === 3) style.italic = true;
      else if (c === 23) style.italic = false;
      else if (c === 4) style.underline = true;
      else if (c === 24) style.underline = false;
      else if ((c >= 30 && c <= 37) || (c >= 90 && c <= 97)) style.fg = PALETTE[c];
      else if (c === 39) style.fg = null;
      else if (c === 38 || c === 48) {
        const fg = c === 38;
        if (codes[k + 1] === 5) { if (fg) style.fg = ansi256(codes[k + 2]); k += 2; }
        else if (codes[k + 1] === 2) { if (fg) style.fg = `rgb(${codes[k + 2]},${codes[k + 3]},${codes[k + 4]})`; k += 4; }
      }
    }
  }

  function eraseLine(params) {
    const mode = parseInt(params || "0", 10);
    if (mode === 0) cur.cells.length = cur.col;                 // cursor -> end
    else if (mode === 1) for (let i = 0; i < cur.col; i++) cur.cells[i] = { ch: " ", style: defaultStyle() };
    else if (mode === 2) { cur.cells = []; cur.col = 0; }
  }

  function write(text) {
    let i = 0;
    while (i < text.length) {
      const ch = text[i];
      if (ch === "\x1b") {
        if (text[i + 1] === "[") {
          let j = i + 2, params = "";
          while (j < text.length && !(text.charCodeAt(j) >= 0x40 && text.charCodeAt(j) <= 0x7e)) { params += text[j]; j += 1; }
          const final = text[j];
          if (final === "m") applySGR(params);
          else if (final === "K") eraseLine(params);
          // other CSI (cursor moves, etc.) ignored
          i = j + 1;
        } else {
          i += 1; // stray ESC or other sequence: skip
        }
      } else if (ch === "\n") { commitLine(); i += 1; }
      else if (ch === "\r") { cur.col = 0; i += 1; }
      else if (ch === "\b") { cur.col = Math.max(0, cur.col - 1); i += 1; }
      else if (ch === "\t") { const next = (Math.floor(cur.col / 8) + 1) * 8; while (cur.col < next) putChar(" "); i += 1; }
      else if (ch < " ") { i += 1; } // other control char: skip
      else { putChar(ch); i += 1; }
    }
    renderLine(node, cur.cells);
    if (stick) container.scrollTop = container.scrollHeight;
  }

  function reset() {
    style = defaultStyle();
    cur = { cells: [], col: 0 };
    node = lineNode();
    container.replaceChildren(node);
    stick = true;
  }

  return { write, reset };
}

function renderLine(node, cells) {
  node.replaceChildren();
  let i = 0;
  while (i < cells.length) {
    const st = cells[i].style;
    const key = styleKey(st);
    let text = "";
    while (i < cells.length && styleKey(cells[i].style) === key) { text += cells[i].ch; i += 1; }
    const span = document.createElement("span");
    if (st.fg) span.style.color = st.fg;
    if (st.bold) span.style.fontWeight = "600";
    if (st.italic) span.style.fontStyle = "italic";
    if (st.underline) span.style.textDecoration = "underline";
    span.textContent = text;
    node.appendChild(span);
  }
}
