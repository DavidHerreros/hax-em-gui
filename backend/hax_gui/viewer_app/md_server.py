"""Standalone backend for the metadata (.xmd/.star/.cs) table viewer.

Loads a metadata file with ``XmippMetaData`` and serves it to a browser table
*lazily*: the file is parsed once (text only — never the particle images), and
the frontend pulls row windows, single-particle thumbnails and plot data on
demand. Nothing bulk-loads the images, so opening is fast and memory stays low
even for million-row star files.

    python md_server.py --md /path/particles.star --port 8053

It is also spawned per-session by the main GUI backend, which reverse-proxies it
at ``/mdviewer/<id>/`` (the frontend uses relative URLs).
"""

from __future__ import annotations

import argparse
import io
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

STATIC = Path(__file__).resolve().parent / "static"


def _cell(v):
    """JSON-safe scalar for a table cell (numbers stay numbers; NaN -> None)."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if math.isnan(f) else round(f, 6)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, (int, str, bool)):
        return v
    try:
        if pd.isna(v):
            return None
    except (ValueError, TypeError):
        pass
    return str(v)


class Meta:
    """Lazily-parsed metadata; caches the table and its column classification."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.dir = Path(path).resolve().parent
        self._md = None
        self._cols: list[str] | None = None
        self._numeric: list[str] | None = None

    @property
    def md(self):
        if self._md is None:
            from xmipp_metadata.metadata import XmippMetaData
            self._md = XmippMetaData(self.path)
        return self._md

    @property
    def table(self) -> pd.DataFrame:
        return self.md.table

    @property
    def columns(self) -> list[str]:
        if self._cols is None:
            self._cols = [str(c) for c in self.table.columns]
        return self._cols

    @property
    def numeric(self) -> list[str]:
        if self._numeric is None:
            self._numeric = [c for c in self.columns if pd.api.types.is_numeric_dtype(self.table[c])]
        return self._numeric

    @property
    def has_images(self) -> bool:
        # Base this on whether we can actually read an image (resolving relative
        # refs against the metadata's own directory), NOT on XmippMetaData's
        # `binaries` flag — that flag is False whenever the first image can't be
        # found from the process CWD, which hides the toggle for perfectly good
        # files that store image paths relative to themselves.
        if "image" not in self.columns:
            return False
        try:
            self.image(0)
            return True
        except Exception:  # noqa: BLE001
            return False

    def _resolve(self, path: str) -> str:
        """Resolve an image path, trying it as-is then relative to the metadata
        file's directory (and its parent) — the usual Xmipp/Relion conventions."""
        p = Path(path)
        if p.is_absolute():
            return str(p)
        for base in (Path.cwd(), self.dir, self.dir.parent):
            cand = base / path
            if cand.exists():
                return str(cand)
        return path

    def image(self, row: int) -> np.ndarray:
        from xmipp_metadata.image_handler import ImageHandler

        ref = str(self.table["image"].iloc[int(row)])
        if "@" in ref:
            idx_s, path = ref.split("@", 1)
            idx = int(idx_s) - 1                     # Xmipp indices are 1-based
        else:
            idx, path = 0, ref
        ih = ImageHandler(self._resolve(path))
        try:
            data = ih[idx]
        except Exception:  # noqa: BLE001 - single-image files aren't indexable
            data = ih.getData()
        img = np.squeeze(np.asarray(data, dtype=np.float32))
        if img.ndim == 3:      # a tiny stack came back — take the first
            img = img[0]
        if img.ndim != 2:
            raise ValueError(f"row {row} is not a 2D image (shape {img.shape})")
        return img


def _png_from_image(img: np.ndarray, size: int) -> bytes:
    """Contrast-stretched (2–98 percentile) grayscale PNG thumbnail."""
    from PIL import Image

    finite = img[np.isfinite(img)]
    if finite.size:
        lo, hi = np.percentile(finite, (2.0, 98.0))
    else:
        lo, hi = 0.0, 1.0
    if hi <= lo:
        hi = lo + 1.0
    u8 = np.clip((img - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    im = Image.fromarray(u8, mode="L")
    if size and max(im.size) != size:
        im = im.resize((size, size), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def build_app(meta: Meta, title: str = "Metadata") -> FastAPI:
    app = FastAPI(title="Hax metadata viewer")

    @app.middleware("http")
    async def _no_cache(request, call_next):
        resp = await call_next(request)
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return resp

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True}

    @app.get("/api/meta")
    def info() -> dict:
        # First call triggers the (one-time) parse.
        cols = meta.columns
        box = None
        if meta.has_images:
            try:
                box = int(meta.image(0).shape[-1])
            except Exception:  # noqa: BLE001 - a broken image ref shouldn't sink the viewer
                box = None
        return {
            "title": title,
            "filename": meta.path,
            "n_rows": int(len(meta.table)),
            "columns": cols,
            "numeric_columns": meta.numeric,
            "has_images": meta.has_images,
            "image_column": "image" if meta.has_images else None,
            "box": box,
        }

    @app.get("/api/rows")
    def rows(start: int = Query(0), count: int = Query(100)) -> dict:
        n = int(len(meta.table))
        start = max(0, min(start, n))
        count = max(1, min(500, count))
        end = min(n, start + count)
        cols = meta.columns
        sub = meta.table.iloc[start:end]
        data = [[_cell(v) for v in rec] for rec in sub.itertuples(index=False, name=None)]
        return {"start": start, "end": end, "n_rows": n, "columns": cols, "rows": data}

    @app.get("/api/image")
    def image(row: int = Query(...), size: int = Query(96)) -> Response:
        if not meta.has_images:
            raise HTTPException(status_code=404, detail="this metadata has no images")
        if row < 0 or row >= len(meta.table):
            raise HTTPException(status_code=404, detail="row out of range")
        try:
            img = meta.image(row)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"image read failed: {exc}")
        png = _png_from_image(img, int(max(16, min(512, size))))
        return Response(content=png, media_type="image/png",
                        headers={"Cache-Control": "max-age=3600"})

    @app.get("/api/columns")
    def columns(names: str = Query(...), max: int = Query(30000)) -> dict:
        """Decimated numeric columns for line/scatter plots. Uniform stride keeps the
        payload bounded; `indices` are the sampled row numbers (x for line plots)."""
        want = [c for c in names.split(",") if c in meta.columns]
        if not want:
            raise HTTPException(status_code=400, detail="no valid columns")
        n = int(len(meta.table))
        cap = int(max) if int(max) > 0 else n
        stride = int(np.ceil(n / cap)) if n > cap else 1
        idx = np.arange(0, n, stride)
        out = {}
        for c in want:
            col = pd.to_numeric(meta.table[c], errors="coerce").to_numpy()[idx]
            out[c] = [None if not np.isfinite(v) else round(float(v), 6) for v in col]
        return {"total": n, "stride": int(stride), "decimated": stride > 1,
                "indices": [int(i) for i in idx], "columns": out}

    @app.get("/api/histogram")
    def histogram(name: str = Query(...), bins: int = Query(60)) -> dict:
        if name not in meta.columns:
            raise HTTPException(status_code=400, detail="unknown column")
        col = pd.to_numeric(meta.table[name], errors="coerce").to_numpy()
        col = col[np.isfinite(col)]
        if col.size == 0:
            raise HTTPException(status_code=400, detail="column has no numeric values")
        counts, edges = np.histogram(col, bins=int(np.clip(bins, 4, 256)))
        return {"name": name, "counts": counts.astype(int).tolist(),
                "edges": edges.astype(float).tolist(),
                "min": float(col.min()), "max": float(col.max()),
                "mean": float(col.mean()), "std": float(col.std()), "n": int(col.size)}

    @app.get("/api/download")
    def download() -> FileResponse:
        p = Path(meta.path)
        if not p.exists():
            raise HTTPException(status_code=404, detail="source file not found")
        return FileResponse(str(p), media_type="application/octet-stream", filename=p.name)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(STATIC / "md.html"))

    if STATIC.is_dir():
        app.mount("/", StaticFiles(directory=str(STATIC), html=False), name="static")
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Metadata (.xmd/.star/.cs) table viewer backend")
    parser.add_argument("--md", required=True, help="path to a .xmd/.star/.cs metadata file")
    parser.add_argument("--title", default="Metadata")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8053)
    args = parser.parse_args()

    meta = Meta(str(Path(args.md).expanduser().resolve()))
    app = build_app(meta, title=args.title)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
