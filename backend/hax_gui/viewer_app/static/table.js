// Windowed (virtual-scroll) table for large metadata files. Only the visible rows
// are in the DOM and only their pages are fetched from the server, so a
// million-row star file scrolls smoothly without loading everything. The image
// column can be toggled between the raw path/ref (default, zero image loads) and
// a rendered thumbnail (fetched lazily per visible row).
//
// config: { columns:[names], nRows, imageCol:index|-1, getRows(start,count),
//           imageURL(row) }

const PAGE = 200;        // rows fetched per request
const OVERSCAN = 6;      // extra rows above/below the viewport
const GUTTER = 74;       // row-number column width
const COL_W = 132;       // default data-column width

export function createTable(container, cfg) {
  const { columns, nRows, imageCol = -1, getRows, imageURL } = cfg;
  let renderImages = false;
  let imgSize = 72;

  const pages = new Map();     // pageIndex -> rows[] (or "pending")
  container.classList.add("md-table");
  container.innerHTML = "";

  const header = document.createElement("div");
  header.className = "mdt-header";
  const body = document.createElement("div");
  body.className = "mdt-body";
  container.appendChild(header);
  container.appendChild(body);

  function colWidth(ci) { return ci === imageCol && renderImages ? Math.max(COL_W, imgSize + 12) : COL_W; }
  function totalWidth() { return GUTTER + columns.reduce((s, _, ci) => s + colWidth(ci), 0); }
  function rowHeight() { return renderImages ? imgSize + 12 : 30; }

  function buildHeader() {
    header.style.width = totalWidth() + "px";
    header.innerHTML = "";
    const g = cell("mdt-cell mdt-h mdt-gutter", "#"); g.style.width = GUTTER + "px";
    header.appendChild(g);
    columns.forEach((name, ci) => {
      const c = cell("mdt-cell mdt-h", name); c.style.width = colWidth(ci) + "px"; c.title = name;
      header.appendChild(c);
    });
  }

  function cell(cls, text) {
    const d = document.createElement("div");
    d.className = cls;
    if (text != null) d.textContent = text;
    return d;
  }

  function pageOf(row) { return Math.floor(row / PAGE); }

  async function ensurePages(first, last) {
    const need = [];
    for (let p = pageOf(first); p <= pageOf(last); p++) {
      if (!pages.has(p)) { pages.set(p, "pending"); need.push(p); }
    }
    for (const p of need) {
      try {
        const start = p * PAGE;
        const res = await getRows(start, Math.min(PAGE, nRows - start));
        pages.set(p, res.rows);
      } catch { pages.set(p, []); }
      render(true);   // force a repaint as each page lands (fills skeleton cells)
    }
  }

  function rowData(row) {
    const pg = pages.get(pageOf(row));
    if (!pg || pg === "pending") return null;
    return pg[row - pageOf(row) * PAGE] || null;
  }

  function fmt(v) { return v == null ? "" : String(v); }

  let scheduled = false;
  let lastKey = "";
  function render(force) {
    body.style.height = nRows * rowHeight() + "px";
    body.style.width = totalWidth() + "px";
    const rh = rowHeight();
    const top = container.scrollTop;
    const first = Math.max(0, Math.floor(top / rh) - OVERSCAN);
    const last = Math.min(nRows - 1, Math.ceil((top + container.clientHeight) / rh) + OVERSCAN);
    ensurePages(first, last);

    // Skip the rebuild (and image re-requests) when nothing visible changed.
    const key = `${first}:${last}:${renderImages}:${imgSize}`;
    if (!force && key === lastKey) return;
    lastKey = key;

    body.innerHTML = "";
    for (let r = first; r <= last; r++) {
      const rowEl = document.createElement("div");
      rowEl.className = "mdt-row";
      rowEl.style.top = r * rh + "px";
      rowEl.style.height = rh + "px";
      rowEl.style.width = totalWidth() + "px";
      const g = cell("mdt-cell mdt-gutter", String(r)); g.style.width = GUTTER + "px";
      rowEl.appendChild(g);
      const data = rowData(r);
      columns.forEach((name, ci) => {
        const c = document.createElement("div");
        c.className = "mdt-cell";
        c.style.width = colWidth(ci) + "px";
        if (ci === imageCol && renderImages && data) {
          const img = document.createElement("img");
          img.className = "mdt-img"; img.loading = "lazy";
          img.width = imgSize; img.height = imgSize;
          img.src = imageURL(r);
          img.title = fmt(data[ci]);
          c.appendChild(img);
        } else {
          const t = data ? fmt(data[ci]) : "";
          c.textContent = t; c.title = t;
          if (!data) c.classList.add("mdt-skel");
        }
        rowEl.appendChild(c);
      });
      body.appendChild(rowEl);
    }
  }

  function onScroll() {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(() => { scheduled = false; render(); });
  }
  const onResize = () => render(true);
  container.addEventListener("scroll", onScroll);
  window.addEventListener("resize", onResize);

  buildHeader();
  render(true);

  return {
    setRenderImages(on) { renderImages = !!on; buildHeader(); render(true); },
    setImageSize(px) { imgSize = px; if (renderImages) { buildHeader(); render(true); } },
    refresh() { pages.clear(); render(true); },
    destroy() { container.removeEventListener("scroll", onScroll); window.removeEventListener("resize", onResize); },
  };
}
