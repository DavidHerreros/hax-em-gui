"""Native metrics reader: serve a run's TensorBoard logs as JSON.

hax keeps writing plain tensorboardX event files (``JaxSummaryWriter``), and real
TensorBoard still runs against them — see ``tensorboard.py``. This module is simply a
*second reader* of the very same files, so the GUI can draw the curves, image panels
and latent space itself instead of embedding TB's UI in an iframe.

Two different things live in a hax log directory:

* ``events.out.tfevents.*`` — scalars, images (already PNG-encoded, so serving one is
  a byte passthrough) and text. Decoded with TensorBoard's own ``EventAccumulator``,
  imported lazily so the GUI's startup stays light and TB stays an optional dep.
* the projector's per-epoch **TSV files** — ``<step>/<tag>/{tensors,metadata}.tsv`` plus
  ``sprite.png``. These are plain text, so we parse them ourselves, with no protobuf
  involved. Because every request re-reads the directory, epochs logged after the page
  was opened simply appear — which is the whole reason the projector is worth replacing
  (TB reads ``projector_config.pbtxt`` once per process and can never invalidate it for
  a JAX log dir).

Layout note: ``JaxSummaryWriter`` timestamps its own subdirectory, and ``add_scalars``
creates a *further* subdirectory per key. So a "log dir" is a tree, and every directory
holding event files is one run — never assume a flat folder.
"""

from __future__ import annotations

import math
import os
import struct
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .files import resolve

_EVENT_PREFIX = "events.out.tfevents"

# Directories that can hold thousands of files but never event files. Pruning them
# keeps the tree walk cheap on a run folder full of checkpoints and .mrc volumes.
_PRUNE = {"checkpoints", "extra_files"}

# Reservoir sizes. Scalars: 0 == keep everything (they are cheap). Images are the
# expensive ones — hax's "predicted images batch" runs to ~150 KB a frame, so an
# uncapped run would pin hundreds of MB. 200 frames still plays as a smooth training
# movie, and TB's Reservoir always keeps the newest item, so a live run's current epoch
# is never the one that gets dropped.
_GUIDANCE = {
    "scalars": 0,
    "images": 200,
    "tensors": 200,
    "audio": 1,
    "histograms": 1,
    "compressedHistograms": 1,
}

# Accumulators are memory-hungry (a run with hundreds of image frames pins tens of MB),
# but this cap must stay above the number of metrics tabs anyone realistically opens:
# below it, every poll of the (n+1)-th tab evicts another tab's accumulator and forces a
# full re-parse of its event files on the next poll — which reads as "the plots take
# forever to appear". Closing a tab drops its entry, so the cache tracks what is open.
_MAX_LOGDIRS = 8
_RELOAD_EVERY = 2.0     # s — a client polling faster than this reuses the last read

_lock = threading.Lock()
_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()


class TensorBoardMissing(RuntimeError):
    """The `tensorboard` package isn't importable (it ships with hax)."""


def _accumulator_cls():
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:   # pragma: no cover - depends on the env
        raise TensorBoardMissing(
            "reading metrics needs the 'tensorboard' package (it is a hax dependency); "
            "run the GUI in the hax environment"
        ) from exc
    return EventAccumulator


# --------------------------------------------------------------------------
# run discovery + accumulator cache
# --------------------------------------------------------------------------
def _event_dirs(root: Path) -> list[Path]:
    """Every directory under ``root`` that holds event files (``root`` itself included)."""
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _PRUNE and not d.endswith("_CHECKPOINT")]
        if any(f.startswith(_EVENT_PREFIX) for f in filenames):
            out.append(Path(dirpath))
    return sorted(out)


def has_events(root: Path) -> bool:
    return bool(_event_dirs(root))


def _runs(root: Path, reload: bool = True) -> dict[str, Any]:
    """``{run_name: EventAccumulator}`` for ``root``, refreshed (incrementally)."""
    ea = _accumulator_cls()
    key = str(root)
    with _lock:
        entry = _cache.get(key)
        if entry is None:
            entry = {"runs": {}, "reloaded": {}}
            _cache[key] = entry
            while len(_cache) > _MAX_LOGDIRS:
                _cache.popitem(last=False)
        _cache.move_to_end(key)

        now = time.monotonic()
        for d in _event_dirs(root):
            name = os.path.relpath(d, root)
            acc = entry["runs"].get(name)
            if acc is None:
                acc = ea(str(d), size_guidance=_GUIDANCE)
                entry["runs"][name] = acc
                entry["reloaded"][name] = 0.0
            if reload and now - entry["reloaded"][name] > _RELOAD_EVERY:
                acc.Reload()   # picks up only what was appended since the last call
                entry["reloaded"][name] = now
        return dict(entry["runs"])


def drop(logdir: str) -> bool:
    """Forget a log dir's accumulators (called when its tab closes)."""
    with _lock:
        return _cache.pop(str(resolve(logdir)), None) is not None


def _root(logdir: str) -> Path:
    root = resolve(logdir)
    if not root.is_dir():
        raise FileNotFoundError(f"log directory not found: {root}")
    return root


def _finite(v: float) -> float | None:
    """NaN/Inf are not valid JSON — a diverged run must not break the whole payload."""
    f = float(v)
    return f if math.isfinite(f) else None


# --------------------------------------------------------------------------
# events: scalars, images, text
# --------------------------------------------------------------------------
def _text_value(acc: Any, tag: str) -> str | None:
    try:
        events = acc.Tensors(tag)
    except KeyError:
        return None
    if not events:
        return None
    vals = events[-1].tensor_proto.string_val
    return vals[0].decode("utf-8", "replace") if vals else None


def manifest(logdir: str) -> dict[str, Any]:
    """Everything the client needs to build the UI: which tags exist, of what kind,
    in which runs, and (for images/embeddings) at which steps."""
    root = _root(logdir)
    runs = _runs(root)

    scalars: dict[str, set[str]] = {}
    images: dict[str, dict[str, Any]] = {}
    texts: dict[str, str] = {}

    for name, acc in runs.items():
        tags = acc.Tags()
        for tag in tags.get("scalars", []):
            scalars.setdefault(tag, set()).add(name)
        for tag in tags.get("images", []):
            # Frames, not steps. hax logs images with no global_step, so tensorboardX
            # stamps every epoch as step 0 — indexing by step would collapse a whole
            # training run down to a single picture (and TB's own image view has the
            # same problem). Wall-clock order is what actually separates the epochs.
            ent = images.setdefault(tag, {"runs": set(), "frames": []})
            ent["runs"].add(name)
            if not ent["frames"]:
                ent["frames"] = [{"step": e.step, "wall": e.wall_time} for e in _image_events(acc, tag)]
        for tag in tags.get("tensors", []):
            # tensorboardX stores add_text as a tensor with a "/text_summary" suffix
            if not tag.endswith("/text_summary"):
                continue
            val = _text_value(acc, tag)
            if val is not None:
                texts[tag[: -len("/text_summary")]] = val

    return {
        "logdir": str(root),
        "runs": sorted(runs),
        "scalars": [{"tag": t, "runs": sorted(r)} for t, r in sorted(scalars.items())],
        "images": [{"tag": t, "runs": sorted(v["runs"]), "frames": v["frames"]}
                   for t, v in sorted(images.items())],
        "texts": texts,
        "embeddings": embeddings(root),
    }


def _decimate(points: list[list[Any]], limit: int) -> list[list[Any]]:
    """Uniform stride, first and last always kept. A long run logs tens of thousands of
    points per tag and the client re-polls the lot every few seconds; more than a couple
    of thousand is past what a 380px-wide card can draw anyway."""
    n = len(points)
    if limit <= 0 or n <= limit:
        return points
    stride = math.ceil(n / limit)
    out = points[::stride]
    if out[-1] is not points[-1]:
        out.append(points[-1])
    return out


def scalars(logdir: str, tags: list[str] | None = None, max_points: int = 2000) -> dict[str, Any]:
    """``{tag: {run: [[step, wall_time, value], ...]}}`` — the shape the plots want
    (one card per tag, one series per run, which is also how ``add_scalars`` lands)."""
    root = _root(logdir)
    out: dict[str, dict[str, list[list[float]]]] = {}
    for name, acc in _runs(root).items():
        for tag in acc.Tags().get("scalars", []):
            if tags and tag not in tags:
                continue
            points = [[e.step, e.wall_time, _finite(e.value)] for e in acc.Scalars(tag)]
            out.setdefault(tag, {})[name] = _decimate(points, max_points)
    return {"tags": out}


def _image_events(acc: Any, tag: str) -> list[Any]:
    """Image events oldest-first. The reservoir's own order isn't guaranteed once it
    starts evicting, so sort by wall time — that is the only thing separating hax's
    epochs, which all carry step 0."""
    return sorted(acc.Images(tag), key=lambda e: e.wall_time)


def image(logdir: str, run: str, tag: str, frame: int | None = None) -> bytes:
    """The PNG TensorBoard already encoded — a byte passthrough, no re-encoding.
    ``frame`` indexes the wall-clock-ordered list; out-of-range clamps to the newest."""
    root = _root(logdir)
    acc = _runs(root, reload=False).get(run)
    if acc is None:
        raise KeyError(f"unknown run: {run}")
    try:
        events = _image_events(acc, tag)
    except KeyError as exc:
        raise KeyError(f"unknown image tag: {tag}") from exc
    if not events:
        raise KeyError(f"no images for tag: {tag}")
    i = len(events) - 1 if frame is None else max(0, min(int(frame), len(events) - 1))
    return events[i].encoded_image_string


# --------------------------------------------------------------------------
# projector: per-epoch TSVs (no protobuf, re-read every request)
# --------------------------------------------------------------------------
def _step_tag(d: Path) -> tuple[int | None, str]:
    """``<step>/<tag>/tensors.tsv`` is what tensorboardX writes; tolerate the swap."""
    name, parent = d.name, d.parent.name
    if parent.isdigit():
        return int(parent), name
    if name.isdigit():
        return int(name), parent
    return None, ""


def _png_size(path: Path) -> tuple[int, int] | None:
    """Width/height straight from the IHDR chunk — no image library needed."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(24)
    except OSError:
        return None
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    w, h = struct.unpack(">II", head[16:24])
    return int(w), int(h)


def _embedding_dirs(root: Path) -> list[tuple[int, str, Path]]:
    found: list[tuple[int, str, Path]] = []
    for tsv in root.rglob("tensors.tsv"):
        step, tag = _step_tag(tsv.parent)
        if step is None:
            continue
        found.append((step, tag, tsv.parent))
    found.sort(key=lambda e: (e[1], e[0]))
    return found


def embeddings(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "step": step,
            "tag": tag,
            "metadata": (d / "metadata.tsv").exists(),
            "sprite": (d / "sprite.png").exists(),
        }
        for step, tag, d in _embedding_dirs(root)
    ]


def _find_embedding(root: Path, step: int, tag: str | None) -> Path:
    entries = _embedding_dirs(root)
    if tag:
        entries = [e for e in entries if e[1] == tag]
    if not entries:
        raise FileNotFoundError("no latent-space embeddings in this run")
    return min(entries, key=lambda e: abs(e[0] - step))[2]


def _read_metadata(path: Path) -> dict[str, Any] | None:
    """tensorboardX writes a bare column when no ``metadata_header`` was given, and a
    header row when one was. Follow the projector's own rule: multi-column => header."""
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return None
    lines = [ln for ln in lines if ln != ""]
    if not lines:
        return None
    if "\t" in lines[0]:
        columns = lines[0].split("\t")
        rows = [ln.split("\t") for ln in lines[1:]]
    else:
        columns = ["label"]
        rows = [[ln] for ln in lines]
    return {"columns": columns, "rows": rows}


def embedding(logdir: str, step: int, tag: str | None = None, max_points: int = 6000) -> dict[str, Any]:
    """One epoch of the latent space: the raw vectors (subsampled to keep the payload
    sane) plus any per-point metadata. Projection to 2D happens in the browser."""
    root = _root(logdir)
    d = _find_embedding(root, step, tag)
    step, tag = _step_tag(d)

    rows: list[list[float]] = []
    with open(d / "tensors.tsv", "r", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append([float(x) for x in line.split("\t")])
            except ValueError:
                continue
    if not rows:
        raise FileNotFoundError(f"empty latent tensors at {d}")

    n_full = len(rows)
    stride = max(1, math.ceil(n_full / max_points))
    indices = list(range(0, n_full, stride))
    points = [rows[i] for i in indices]

    meta = _read_metadata(d / "metadata.tsv")
    if meta is not None:
        meta["rows"] = [meta["rows"][i] for i in indices if i < len(meta["rows"])]

    sprite: dict[str, Any] | None = None
    size = _png_size(d / "sprite.png")
    if size is not None:
        # make_sprite lays the thumbnails out on a square grid of ceil(sqrt(n)) tiles.
        per_side = math.ceil(math.sqrt(n_full))
        sprite = {"width": size[0], "height": size[1], "per_side": per_side,
                  "tile": size[0] // per_side if per_side else 0}

    return {
        "step": step,
        "tag": tag,
        "n": len(points),
        "n_full": n_full,
        "dim": len(points[0]),
        "stride": stride,
        "indices": indices,
        "points": points,
        "meta": meta,
        "sprite": sprite,
    }


def sprite(logdir: str, step: int, tag: str | None = None) -> bytes:
    root = _root(logdir)
    path = _find_embedding(root, step, tag) / "sprite.png"
    if not path.exists():
        raise FileNotFoundError("no sprite for this epoch")
    return path.read_bytes()
