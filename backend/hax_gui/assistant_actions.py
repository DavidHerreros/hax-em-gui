"""Deterministic backing for assistant-proposed actions (Phase 1/2).

The assistant only ever proposes *what* to do (a program + a data path); the
concrete, safety-critical parts — mapping the path to the right input argument,
filling defaults, validating, and creating the job — are done here by fixed code
using the schema/registry, never by the language model. Nothing is launched: jobs
are created as drafts (``launch=false``) for the user to review and start.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from . import registry
from .config import settings

# Candidate input-data arguments, best first. `md` (particle metadata) is the
# dominant input across hax programs and is also the launch-cwd arg.
_INPUT_ARG_CANDIDATES = ("md", "metadata", "input", "i", "particles", "star", "xmd")

_MD_SUFFIXES = (".xmd", ".star", ".cs")
_MAP_SUFFIXES = (".mrc", ".map", ".vol")


def input_arg(program: str) -> Optional[str]:
    """The program's primary input-data argument (deterministic, from schema)."""
    schema = registry.get_schema(program)
    if schema is None:
        return None
    names = {f["name"] for f in schema["fields"]}
    wd = registry.working_dir_arg(program)
    if wd and wd in names:
        return wd
    for cand in _INPUT_ARG_CANDIDATES:
        if cand in names:
            return cand
    return None


def inspect_path(path: str) -> dict[str, Any]:
    """Best-effort, deterministic facts about a data path (existence, kind, and
    a few descriptors) so a proposal can be grounded and the user can sanity-check
    it. Never raises; unknown facts are simply omitted."""
    p = Path(path).expanduser()
    out: dict[str, Any] = {"path": str(p), "exists": p.exists()}
    if not p.exists():
        return out
    if p.is_dir():
        out["kind"] = "directory"
        mrcs = list(p.glob("*.mrc"))
        mds = [q for q in p.iterdir() if q.suffix.lower() in _MD_SUFFIXES]
        out["volumes"] = len(mrcs)
        out["metadata_files"] = len(mds)
        return out
    suffix = p.suffix.lower()
    try:
        out["size"] = p.stat().st_size
    except OSError:
        pass
    if suffix in _MD_SUFFIXES:
        out["kind"] = "metadata"
        out.update(_inspect_metadata(p))
    elif suffix in _MAP_SUFFIXES:
        out["kind"] = "volume"
        out.update(_inspect_volume(p))
    else:
        out["kind"] = suffix.lstrip(".") or "file"
    return out


def _inspect_metadata(p: Path) -> dict[str, Any]:
    """Row count + notable columns from a STAR/xmd loop, read cheaply from the head."""
    facts: dict[str, Any] = {}
    try:
        head = p.read_text(errors="replace")[:200000]
    except OSError:
        return facts
    cols = [ln.split()[0].lstrip("_") for ln in head.splitlines()
            if ln.strip().startswith("_") and len(ln.split()) >= 1]
    if cols:
        facts["columns"] = cols[:40]
        facts["has_latents"] = "latent_space" in cols
        facts["has_ctf"] = any(c.startswith("ctf") for c in cols)
    return facts


def _inspect_volume(p: Path) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    try:
        import mrcfile
        with mrcfile.open(str(p), permissive=True, header_only=True) as m:
            facts["box"] = [int(m.header.nx), int(m.header.ny), int(m.header.nz)]
            try:
                facts["voxel_size"] = round(float(m.voxel_size.x), 3)
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001 - mrcfile missing or unreadable header
        pass
    return facts


def build_draft_spec(program: str, data_path: Optional[str]) -> dict[str, Any]:
    """Build a deterministic draft-job spec: registry defaults + the data path
    mapped to the program's input arg. Reports required args still unset (the
    output/ssd args are auto-filled by the job manager at create time, so they
    are excluded). Never launches anything."""
    schema = registry.get_schema(program)
    if schema is None:
        return {"program": program, "error": f"unknown program '{program}'"}
    fields = {f["name"]: f for f in schema["fields"]}
    args: dict[str, Any] = dict(registry.field_defaults(program))

    in_arg = input_arg(program)
    if data_path and in_arg:
        args[in_arg] = data_path

    auto = {registry.output_arg(program), registry.ssd_arg(program)}
    missing = [
        n for n, f in fields.items()
        if f.get("required") and n not in args and n not in auto
    ]
    return {
        "program": program,
        "input_arg": in_arg,
        "args": args,
        "missing_required": missing,
        "auto_filled": [a for a in auto if a],
    }


def actions_enabled() -> bool:
    return bool(settings.assistant_actions)
