"""Viewer routes: the landscape (latents+network) viewer, the map (.mrc) viewer
and the metadata table viewer — session lifecycles plus their reverse proxies —
and the job-artifact endpoints that feed them."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel

from .. import adapters, projects, settings_store, usage
from .. import files as filebrowser
from ..config import settings
from ..jobs import manager
from ..md_viewer import md_viewer_manager
from ..mrc_viewer import mrc_viewer_manager
from ..viewer import viewer_manager
from .artifacts import (
    _find_latent_metadata, _flexconsensus_spaces, _job_file, _list_job_files,
    _list_job_input_files,
)
from .shared import auth, proxy as _proxy

router = APIRouter()

# -- WebGL viewer (per-session subprocess, reverse-proxied) ----------------
class ViewerStart(BaseModel):
    latents: str
    reload: str
    title: str | None = None
    reduced: str | None = None
    gpu: str | None = None


@router.post("/api/viewer", dependencies=[Depends(auth)])
async def start_viewer(body: ViewerStart) -> dict[str, Any]:
    return await _ensure_viewer(body.latents, body.reload, body.title or "Conformational landscape",
                                reduced=body.reduced, gpu=body.gpu)


@router.get("/api/viewer/flexconsensus/{job_id}", dependencies=[Depends(auth)])
def flexconsensus_spaces(job_id: str) -> dict[str, Any]:
    """The consensus output spaces of a FlexConsensus run, each with the decoder
    network auto-detected from the predict run that produced its input space."""
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.get("program") != "flexconsensus":
        raise HTTPException(status_code=422, detail="not a FlexConsensus run")
    spaces = _flexconsensus_spaces(job)
    return {"spaces": [{k: v for k, v in s.items() if k != "reload"} for s in spaces]}


@router.post("/api/viewer/from-job/{job_id}", dependencies=[Depends(auth)])
async def start_viewer_from_job(job_id: str, space: int | None = None,
                                gpu: str | None = None) -> dict[str, Any]:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    program = job.get("program", "")
    args = job.get("args", {})
    # FlexConsensus: pick one consensus output space; its input space supplies the
    # full latents to decode, and the decoder network is auto-detected.
    if program == "flexconsensus":
        # Off the loop: resolving spaces can load and compare latent arrays.
        spaces = await asyncio.to_thread(_flexconsensus_spaces, job)
        if space is None:
            raise HTTPException(status_code=422,
                                detail={"choose_flexconsensus_space": True,
                                        "spaces": [{k: v for k, v in s.items() if k != "reload"} for s in spaces]})
        sp = next((s for s in spaces if s["index"] == space), None)
        if sp is None:
            raise HTTPException(status_code=422, detail="unknown consensus space")
        if not sp["viewable"]:
            raise HTTPException(status_code=422, detail=sp["reason"] or "this consensus space is not viewable")
        title = f"{job.get('name') or 'FlexConsensus'} · {sp['name']}"
        project_dir = projects.dir_for(job.get("project_id"))
        metadata = await asyncio.to_thread(
            _find_latent_metadata, project_dir, sp["input_space"], job.get("project_id"))
        return await _ensure_viewer(sp["input_space"], sp["reload"], title, metadata,
                                    str(project_dir) if project_dir else None, sp["consensus"], gpu)
    reload_raw = args.get("pickled_nn") or args.get("reload")
    # Full latents: a direct input (annotate_space), or derived from this run's
    # own output (a predict run's predicted latents).
    latents = args.get("z_space") or args.get("latents") or args.get("latents_file")
    if not latents:
        try:
            # Off the loop: extracting the latents column parses per-particle metadata.
            latents = await asyncio.to_thread(
                adapters.produce, "latents", job,
                settings.data_dir / "viewer_cache" / job_id, "z_space")
        except adapters.AdapterError:
            latents = None
    if not latents or not reload_raw:
        raise HTTPException(
            status_code=422,
            detail="this run has no latents + network to view (need latents/z_space — or a predict "
                   "run's output — plus pickled_nn/reload)",
        )
    reload = str(adapters.checkpoint_dir(program, reload_raw))
    # Optional reduced display space (e.g. a FlexConsensus consensus bound here).
    reduced = args.get("z_space_reduced")
    title = job.get("name") or job.get("program") or "Conformational landscape"
    project_dir = projects.dir_for(job.get("project_id"))
    metadata = await asyncio.to_thread(
        _find_latent_metadata, project_dir, latents, job.get("project_id"))
    return await _ensure_viewer(latents, reload, title, metadata,
                                str(project_dir) if project_dir else None, reduced, gpu)


# Programs whose predict run yields latents + a decoder — derived from the
# adapters maps (single source of truth), so new such programs need no edit here.
def _viewer_gpu(explicit: str | None) -> str | None:
    """The device to pin a viewer to: the caller's choice, else the saved default.

    None (nothing chosen, no default saved) means "don't pin": the viewer inherits
    our environment and JAX picks its own device, which is what it always did.
    """
    gpu = explicit if explicit is not None else settings_store.load().get("viewer_gpu")
    return (gpu or "").strip() or None


async def _ensure_viewer(latents: str, reload: str, title: str,
                         metadata: str | None = None, project_dir: str | None = None,
                         reduced: str | None = None, gpu: str | None = None) -> dict[str, Any]:
    try:
        return await viewer_manager.ensure(latents, reload, title, metadata, project_dir,
                                           reduced, _viewer_gpu(gpu))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (RuntimeError, TimeoutError) as exc:
        raise HTTPException(status_code=502, detail=f"viewer failed to start: {exc}")


@router.get("/api/viewer", dependencies=[Depends(auth)])
def list_viewers() -> list[dict[str, Any]]:
    return viewer_manager.list()


@router.delete("/api/viewer/{vid}", dependencies=[Depends(auth)])
def stop_viewer(vid: str) -> dict[str, bool]:
    return {"stopped": viewer_manager.stop(vid)}


@router.api_route("/viewer/{vid}", methods=["GET"])
def viewer_redirect(vid: str) -> RedirectResponse:
    return RedirectResponse(url=f"/viewer/{vid}/")


@router.api_route("/viewer/{vid}/{path:path}", methods=["GET", "POST", "OPTIONS", "HEAD"])
async def viewer_proxy(vid: str, path: str, request: Request) -> Response:
    inst = viewer_manager.get(vid)
    if inst is None:
        raise HTTPException(status_code=404, detail="unknown viewer session")
    viewer_manager.touch(vid)   # any proxied request defers the idle timeout
    # The viewer subprocess serves at root; strip the /viewer/<id> prefix.
    query = request.url.query
    url = f"http://127.0.0.1:{inst['port']}/{path}" + (f"?{query}" if query else "")
    return await _proxy(request, url)


# -- Map (.mrc) viewer: inspect any volume a job produced -------------------
_MAP_SUFFIXES = (".mrc", ".map", ".vol", ".ccp4")
_MD_SUFFIXES = (".xmd", ".star", ".cs")


@router.get("/api/jobs/{job_id}/mrcs", dependencies=[Depends(auth)])
def job_mrcs(job_id: str) -> dict[str, Any]:
    """List the volume files (.mrc/.map/.vol) inside a job's run folder, so any map
    it produced can be inspected (works while running and after it finishes)."""
    return _list_job_files(job_id, _MAP_SUFFIXES)


@router.get("/api/jobs/{job_id}/usage", dependencies=[Depends(auth)])
async def job_usage(job_id: str, refresh: bool = False) -> dict[str, Any]:
    """Disk occupied by this run's folder (outputs, checkpoints and logs).

    Measured on demand rather than carried on every job listing: it means walking the
    folder, and a job list refreshes every few seconds.
    """
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    measured = await asyncio.to_thread(usage.measure, job.get("workdir"), refresh)
    return measured or {"path": None, "bytes": 0, "files": 0, "exists": False, "measured_at": None}


@router.get("/api/jobs/{job_id}/inputs", dependencies=[Depends(auth)])
def job_inputs(job_id: str) -> dict[str, Any]:
    """Volume + metadata files a job took as *inputs* (referenced in its args), so
    they can be inspected in the same viewers as the outputs."""
    return {
        "maps": _list_job_input_files(job_id, _MAP_SUFFIXES)["files"],
        "metadata": _list_job_input_files(job_id, _MD_SUFFIXES)["files"],
    }


async def _ensure_mrc_viewer(mrc: str | list[str], title: str) -> dict[str, Any]:
    try:
        return await mrc_viewer_manager.ensure(mrc, title)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (RuntimeError, TimeoutError) as exc:
        raise HTTPException(status_code=502, detail=f"map viewer failed to start: {exc}")


class MrcViewerStart(BaseModel):
    # One map, or several to show together in one scene. `path` is kept so existing
    # single-map callers are unchanged; exactly one of the two must be given.
    path: str | None = None
    paths: list[str] | None = None
    title: str | None = None


@router.post("/api/mrcviewer", dependencies=[Depends(auth)])
async def start_mrc_viewer(body: MrcViewerStart) -> dict[str, Any]:
    """Open the map viewer for .mrc files within the sandboxed file roots.

    Several maps open as one viewer (a model panel you can toggle and morph
    through), not as several tabs."""
    wanted = body.paths if body.paths else ([body.path] if body.path else [])
    if not wanted:
        raise HTTPException(status_code=422, detail="give `path` or `paths`")
    resolved = []
    for one in wanted:
        try:
            path = filebrowser.resolve(one)   # confines to HAX_GUI_FILE_ROOTS
        except filebrowser.AccessDenied as exc:
            raise HTTPException(status_code=403, detail=str(exc))
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"file not found: {one}")
        resolved.append(str(path))
    default = (Path(resolved[0]).name if len(resolved) == 1
               else f"{len(resolved)} maps")
    return await _ensure_mrc_viewer(resolved, body.title or default)


@router.post("/api/mrcviewer/from-job/{job_id}", dependencies=[Depends(auth)])
async def start_mrc_viewer_from_job(job_id: str, path: str) -> dict[str, Any]:
    """Open the map viewer for a .mrc this job produced or used (the path must be
    one of the job's inputs or outputs — no arbitrary filesystem access here)."""
    job, target = _job_file(job_id, path, _MAP_SUFFIXES)
    title = f"{job.get('name') or job.get('program') or 'map'} · {target.name}"
    return await _ensure_mrc_viewer(str(target), title)


@router.get("/api/mrcviewer", dependencies=[Depends(auth)])
def list_mrc_viewers() -> list[dict[str, Any]]:
    return mrc_viewer_manager.list()


@router.delete("/api/mrcviewer/{vid}", dependencies=[Depends(auth)])
def stop_mrc_viewer(vid: str) -> dict[str, bool]:
    return {"stopped": mrc_viewer_manager.stop(vid)}


@router.api_route("/mrcviewer/{vid}", methods=["GET"])
def mrc_viewer_redirect(vid: str) -> RedirectResponse:
    return RedirectResponse(url=f"/mrcviewer/{vid}/")


@router.api_route("/mrcviewer/{vid}/{path:path}", methods=["GET", "POST", "OPTIONS", "HEAD"])
async def mrc_viewer_proxy(vid: str, path: str, request: Request) -> Response:
    inst = mrc_viewer_manager.get(vid)
    if inst is None:
        raise HTTPException(status_code=404, detail="unknown map-viewer session")
    mrc_viewer_manager.touch(vid)   # any proxied request defers the idle timeout
    query = request.url.query
    url = f"http://127.0.0.1:{inst['port']}/{path}" + (f"?{query}" if query else "")
    return await _proxy(request, url)


# -- Metadata (.xmd/.star/.cs) viewer: browse any table a job produced ------
async def _ensure_md_viewer(md: str, title: str) -> dict[str, Any]:
    try:
        return await md_viewer_manager.ensure(md, title)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (RuntimeError, TimeoutError) as exc:
        raise HTTPException(status_code=502, detail=f"metadata viewer failed to start: {exc}")


class MdViewerStart(BaseModel):
    path: str
    title: str | None = None


@router.get("/api/jobs/{job_id}/metadata", dependencies=[Depends(auth)])
def job_metadata(job_id: str) -> dict[str, Any]:
    """List the metadata files (.xmd/.star/.cs) inside a job's run folder."""
    return _list_job_files(job_id, _MD_SUFFIXES)


@router.post("/api/mdviewer", dependencies=[Depends(auth)])
async def start_md_viewer(body: MdViewerStart) -> dict[str, Any]:
    """Open the metadata viewer for any .xmd/.star/.cs within the file roots."""
    try:
        path = filebrowser.resolve(body.path)   # confines to HAX_GUI_FILE_ROOTS
    except filebrowser.AccessDenied as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    if not path.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return await _ensure_md_viewer(str(path), body.title or path.name)


@router.post("/api/mdviewer/from-job/{job_id}", dependencies=[Depends(auth)])
async def start_md_viewer_from_job(job_id: str, path: str) -> dict[str, Any]:
    """Open the metadata viewer for a table this job produced or used (the path must
    be one of the job's inputs or outputs)."""
    job, target = _job_file(job_id, path, _MD_SUFFIXES)
    title = f"{job.get('name') or job.get('program') or 'metadata'} · {target.name}"
    return await _ensure_md_viewer(str(target), title)


@router.get("/api/mdviewer", dependencies=[Depends(auth)])
def list_md_viewers() -> list[dict[str, Any]]:
    return md_viewer_manager.list()


@router.delete("/api/mdviewer/{vid}", dependencies=[Depends(auth)])
def stop_md_viewer(vid: str) -> dict[str, bool]:
    return {"stopped": md_viewer_manager.stop(vid)}


@router.api_route("/mdviewer/{vid}", methods=["GET"])
def md_viewer_redirect(vid: str) -> RedirectResponse:
    return RedirectResponse(url=f"/mdviewer/{vid}/")


@router.api_route("/mdviewer/{vid}/{path:path}", methods=["GET", "POST", "OPTIONS", "HEAD"])
async def md_viewer_proxy(vid: str, path: str, request: Request) -> Response:
    inst = md_viewer_manager.get(vid)
    if inst is None:
        raise HTTPException(status_code=404, detail="unknown metadata-viewer session")
    md_viewer_manager.touch(vid)   # any proxied request defers the idle timeout
    query = request.url.query
    url = f"http://127.0.0.1:{inst['port']}/{path}" + (f"?{query}" if query else "")
    return await _proxy(request, url)

