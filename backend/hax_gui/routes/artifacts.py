"""Resolution of a job's on-disk artifacts (latents, decoders, maps, tables).

Pure lookup logic shared by the viewer routes and the assistant: given a job,
find the files it produced or consumed — the latents table of a predict run,
the decoder checkpoint that pairs with an input space, the FlexConsensus
spaces, the .mrc/.star files under a run folder — without touching any viewer
session state. Extracted from app.py, where ~300 lines of this sat between
route handlers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import HTTPException

from .. import adapters
from ..config import settings
from ..jobs import manager

_MAP_SUFFIXES = (".mrc", ".map", ".vol", ".ccp4")
_MD_SUFFIXES = (".xmd", ".star", ".cs")

def _find_latent_metadata(project_dir: "Path | None", latents: str,
                          project_id: str | None = None) -> str | None:
    """Find the project's particle metadata that is row-aligned with these latents.

    The latents `.npy` was materialised from a predict run's `_latent_space` column;
    we recover that metadata (poses/CTF/latent) so layers can be exported with full
    per-particle info. Matched by `_latent_space` presence + row count + first-row value.

    The metadata almost always lives in a predict run's own folder, so those are
    scanned first (from the job records — a handful of directories) before falling
    back to a lazy walk of the whole project. The old version `sorted(rglob("*"))`-ed
    the entire project tree up front and parsed every metadata file on the way — a
    mature project holds hundreds of GB across millions of files, so opening the
    viewer hung the server for minutes. Blocking: call from a worker thread.
    """
    if project_dir is None:
        return None
    try:
        import numpy as np
        from xmipp_metadata.metadata import XmippMetaData

        lp = Path(latents)
        lat = np.loadtxt(lp) if lp.suffix == ".txt" else np.load(lp)
        lat = np.atleast_2d(np.asarray(lat, dtype=np.float32))

        seen: set[Path] = set()

        def _candidates():
            # Pass 1: predict-run folders, newest jobs first (deterministic).
            if project_id:
                for j in manager.list_for_project(project_id):
                    wd = j.get("workdir")
                    if not wd or j.get("program") not in _predict_programs():
                        continue
                    hits: list[Path] = []
                    for pat in ("*.xmd", "*.star"):
                        hits.extend(Path(wd).rglob(pat))
                    for p in sorted(hits):
                        if p not in seen:
                            seen.add(p)
                            yield p
            # Pass 2 (fallback): the rest of the project, lazily — no up-front
            # materialisation, and only suffix matches are ever touched.
            for pat in ("*.xmd", "*.star"):
                for p in project_dir.rglob(pat):
                    if p not in seen:
                        seen.add(p)
                        yield p

        for p in _candidates():
            try:
                md = XmippMetaData(str(p))
                if not md.isMetaDataLabel("latent_space") or len(md) != lat.shape[0]:
                    continue
                first = np.fromstring(str(md.getMetaDataColumns("latent_space")[0]), sep=",")
                if first.size == lat.shape[1] and np.allclose(first, lat[0], atol=1e-3):
                    return str(p)
            except Exception:  # noqa: BLE001 - skip unreadable candidates
                continue
    except Exception:  # noqa: BLE001
        return None
    return None

def _predict_programs() -> set[str]:
    return set(adapters.latents_decoder_programs())

def _load_latents_array(path: str):
    import numpy as np
    p = Path(path)
    arr = np.loadtxt(p) if p.suffix == ".txt" else np.load(p)
    return np.atleast_2d(np.asarray(arr, dtype=np.float32))

def _predict_latents_array(job: dict[str, Any]):
    """Load a predict run's latents (extracting the metadata column if needed)."""
    src = adapters.source_file_for("latents", job)
    if src is None:
        return None
    try:
        if src.suffix == ".npy":
            return _load_latents_array(str(src))
        out_dir = settings.data_dir / "viewer_cache" / "_cmp"
        produced = adapters.produce("latents", job, out_dir, job["id"])
        return _load_latents_array(produced)
    except Exception:  # noqa: BLE001
        return None

def _find_decoder(input_path: str, predict_jobs: list[dict[str, Any]]):
    """Find the predict job (and its decoder checkpoint) that produced `input_path`,
    by path containment first, then by matching latents (shape + first row)."""
    try:
        ip = Path(input_path).expanduser().resolve()
    except Exception:  # noqa: BLE001
        ip = None
    for j in predict_jobs:
        wd = j.get("workdir")
        net = j["args"].get("reload") or j["args"].get("pickled_nn")
        if wd and net and ip and str(ip).startswith(str(Path(wd).expanduser().resolve())):
            return j, str(adapters.checkpoint_dir(j["program"], net))
    import numpy as np
    try:
        lat = _load_latents_array(input_path)
    except Exception:  # noqa: BLE001
        return None, None
    for j in predict_jobs:
        net = j["args"].get("reload") or j["args"].get("pickled_nn")
        if not net:
            continue
        cand = _predict_latents_array(j)
        if cand is not None and cand.shape == lat.shape and np.allclose(cand[0], lat[0], atol=1e-3):
            return j, str(adapters.checkpoint_dir(j["program"], net))
    return None, None

def _flexconsensus_spaces(job: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve a FlexConsensus run's consensus output spaces. Each space pairs a
    consensus file (display) with its input latent space (decoded) and the
    auto-detected decoder network from the predict run that produced that input."""
    args = job.get("args", {})
    raw = args.get("input_space")
    entries = raw if isinstance(raw, list) else (str(raw).split() if raw else [])
    workdir = Path(job.get("workdir") or ".")
    pid = job.get("project_id")
    predict = [j for j in (manager.list_for_project(pid) if pid else []) if j["program"] in _predict_programs()]
    spaces = []
    for i, entry in enumerate(entries):
        entry = str(entry)
        # FlexConsensus accepts "NAME:path"; plain paths are auto-named "Input_i".
        # (Linux paths have no ':' so a ':' marks an explicit name.)
        if ":" in entry:
            name, ipath = entry.split(":", 1)
        else:
            name, ipath = f"Input_{i}", entry
        cons = workdir / f"{name}_consensus.npy"
        djob, reload = _find_decoder(ipath, predict)
        ok = bool(djob and reload and cons.exists())
        spaces.append({
            "index": i, "name": name, "input_space": ipath, "consensus": str(cons),
            "consensus_exists": cons.exists(),
            "decoder_job": djob["id"] if djob else None,
            "decoder_program": djob["program"] if djob else None,
            "reload": reload, "viewable": ok,
            "reason": None if ok else ("no consensus output found for this space" if not cons.exists()
                                       else "couldn't auto-detect the decoder network for this space"),
        })
    return spaces

def _list_job_files(job_id: str, suffixes: tuple[str, ...]) -> dict[str, Any]:
    """List a job's run-folder files whose suffix is in `suffixes` (recursive)."""
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    wd = job.get("workdir")
    if not wd:
        return {"workdir": None, "files": []}
    root = Path(wd)
    out: list[dict[str, Any]] = []
    if root.is_dir():
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in suffixes:
                try:
                    st = p.stat()
                except OSError:
                    continue
                out.append({
                    "path": str(p),
                    "name": str(p.relative_to(root)),
                    "size": st.st_size,
                    "mtime": st.st_mtime,
                })
    return {"workdir": str(root), "files": out}

def _find_latents_file(root: Path) -> str | None:
    """A .xmd/.star under ``root`` carrying a `_latent_space` column (name-hint first)."""
    mds = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in (".xmd", ".star")]
    mds.sort(key=lambda p: ("latent" not in p.name.lower(), str(p)))
    for cand in mds:
        try:
            head = cand.read_text(errors="ignore")[:20000]
        except OSError:
            continue
        if "_latent_space" in head:
            return str(cand)
    return None

def _find_sibling_latents(job: dict[str, Any]) -> tuple[str | None, str | None]:
    """Look for latents in other runs of the same project (same program first, then
    any), newest first — so attaching a *train* run (metrics+volumes) can still pull
    the landscape from the matching *predict* run. Returns (path, source_label)."""
    pid = job.get("project_id")
    if not pid:
        return None, None
    try:
        siblings = [j for j in manager.list_for_project(pid) if j.get("id") != job.get("id")]
    except Exception:  # noqa: BLE001
        return None, None
    prog = job.get("program")
    siblings.sort(key=lambda j: j.get("program") != prog)  # same-program first (stable: keeps recency)
    for sib in siblings:
        wd = sib.get("workdir")
        if not wd or not Path(wd).is_dir():
            continue
        found = _find_latents_file(Path(wd))
        if found:
            return found, (sib.get("name") or sib.get("program") or "sibling run")
    return None, None

def _resolve_run_artifacts(job: dict[str, Any]) -> dict[str, Any]:
    """Resolve a job's run folder into the artifact paths the assistant reads:
    the run dir (metrics + config), a latents metadata file, and a decoded-volume
    series directory. Best-effort — any of latents/volumes may be None. When the
    attached run has no latents of its own, they are pulled from a sibling run in
    the same project (e.g. train run ← its predict run)."""
    wd = job.get("workdir")
    run_dir = wd if wd and Path(wd).is_dir() else None
    latents: str | None = None
    latents_from: str | None = None
    volumes: str | None = None
    has_metrics = False
    if run_dir:
        root = Path(run_dir)
        # metrics = a *_metrics dir, or event files somewhere under the run folder
        has_metrics = bool(next(root.glob("*_metrics"), None)) or \
            next(root.rglob("events.out.tfevents*"), None) is not None
        latents = _find_latents_file(root)
        # volumes: the directory holding the most .mrc (≥2), preferring volume-ish names
        counts: dict[Path, int] = {}
        for p in root.rglob("*.mrc"):
            counts[p.parent] = counts.get(p.parent, 0) + 1
        series = [(d, n) for d, n in counts.items() if n >= 2]
        if series:
            def _score(item: tuple[Path, int]) -> tuple[bool, bool, int]:
                d, n = item
                nm = d.name.lower()
                # A decoded-state series with an exported deformation field wins:
                # it is a real conformational trajectory with exact correspondences,
                # unlike Intermediate_volumes (training-epoch snapshots).
                has_deform = (d / "deformation" / "deformed_points.npy").is_file()
                return (has_deform, "volume" in nm or "intermediate" in nm, n)
            volumes = str(max(series, key=_score)[0])
    if latents is None:
        latents, latents_from = _find_sibling_latents(job)
    return {
        "run_dir": run_dir,
        "latents": latents,
        "latents_from": latents_from,  # sibling run label, when latents came from one
        "volumes": volumes,
        "has_metrics": has_metrics,
        "label": f"{job.get('name') or job.get('program') or 'run'} run",
    }

def _job_arg_paths(job: dict[str, Any]) -> set[str]:
    """Resolved file paths referenced in a job's argument values. Handles plain
    strings, lists, whitespace-separated multi-values and the ``NAME:path`` form
    (e.g. FlexConsensus input_space). Used to authorise viewing a run's *inputs*."""
    out: set[str] = set()
    args = job.get("args") or {}
    for v in args.values():
        items = v if isinstance(v, list) else [v]
        for it in items:
            if not isinstance(it, (str, int, float)):
                continue
            for tok in str(it).split():
                cand = tok
                if ":" in tok:
                    after = tok.split(":", 1)[1]
                    if after and Path(after).expanduser().exists():
                        cand = after
                try:
                    p = Path(cand).expanduser()
                    if p.is_file():
                        out.add(str(p.resolve()))
                except (OSError, RuntimeError):
                    continue
    return out

def _list_job_input_files(job_id: str, suffixes: tuple[str, ...]) -> dict[str, Any]:
    """List a job's *input* files (from its args) whose suffix is in `suffixes`."""
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    out: list[dict[str, Any]] = []
    for path in sorted(_job_arg_paths(job)):
        p = Path(path)
        if p.suffix.lower() not in suffixes:
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        out.append({"path": path, "name": p.name, "size": st.st_size, "mtime": st.st_mtime})
    return {"files": out}

def _job_file(job_id: str, path: str, suffixes: tuple[str, ...]) -> tuple[dict[str, Any], Path]:
    """Validate that `path` is a file of the given kind that this job legitimately
    references — either an output inside its run folder or an input in its args
    (so no arbitrary filesystem access is granted through the from-job routes)."""
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    try:
        target = Path(path).expanduser().resolve()
    except (OSError, RuntimeError):
        raise HTTPException(status_code=400, detail="invalid path")
    wd = job.get("workdir")
    in_workdir = False
    if wd:
        root = Path(wd).resolve()
        in_workdir = root == target or root in target.parents
    if not in_workdir and str(target) not in _job_arg_paths(job):
        raise HTTPException(status_code=403, detail="path is not one of this job's inputs or outputs")
    if not target.is_file() or target.suffix.lower() not in suffixes:
        raise HTTPException(status_code=404, detail="file not found")
    return job, target
