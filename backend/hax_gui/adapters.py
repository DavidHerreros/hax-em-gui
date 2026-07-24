"""Derive one program's input from a previous job's output ("job chaining").

Several hax programs consume a file that an earlier run already produced. For
example annotate_space / filter_latents / decode_states_from_latents all want a
latents ``.npy`` of shape ``(N, latent_dim)``, while:

* ``hetsiren`` / ``zernike3deep`` / ``reconsiren`` predict-runs emit those
  latents *inside* a metadata column (``latent_space``, comma-separated floats);
* ``filter_latents`` emits them as a ready ``filtered_latents.npy``.

This module knows, per "artifact kind", which jobs can produce it and how to
materialise the exact file the consumer expects. The GUI uses it to let users
pick (or drag-and-drop) a job as the input of the next program.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any


class AdapterError(Exception):
    """A job's output could not be turned into the requested input."""


# Form fields that may be fed from a previous job: {program: {field: kind}}.
# `latents` -> a (N, latent_dim) .npy; the various `*_model` kinds -> a saved
# network checkpoint directory passed straight to `--reload`; `particles` -> the
# metadata a preprocess_particles run wrote; `volume` -> a preprocessed map.
BINDABLE_FIELDS: dict[str, dict[str, str]] = {
    "annotate_space": {"z_space": "latents", "z_space_reduced": "latents", "pickled_nn": "decoder_model"},
    "filter_latents": {"latents": "latents"},
    "decode_states_from_latents": {"latents_file": "latents", "reload": "decoder_model"},
    "hetsiren": {"reload": "hetsiren_model", "md": "particles", "vol": "volume"},
    "zernike3deep": {"reload": "zernike3deep_model", "md": "particles", "vol": "volume"},
    "reconsiren": {"reload": "reconsiren_model", "md": "particles", "vol": "volume"},
    "reconsiren_het_only": {"reload": "reconsiren_het_only_model", "md": "particles", "vol": "volume"},
    # FlexConsensus consensuates many input latent spaces, so `--input_space` is a
    # *multi-valued* input: wire one latents source per program you want included.
    "flexconsensus": {"reload": "flexconsensus_model", "input_space": "latents"},
    "latent_space_deconvolution": {"reload": "deconvolver_model", "md": "particles"},
    "image_gray_scale_adjustment": {"reload": "image_adjust_model", "md": "particles", "vol": "volume"},
    "volume_gray_scale_adjustment": {"reload": "volume_adjust_model", "md": "particles", "vol": "volume"},
    "estimate_latent_covariances": {"md": "particles"},
    # `motion_correction` is a --reload in disguise: MoDART loads a trained motion
    # decoder (HetSIREN w/ transport of mass, or Zernike3Deep) to correct particles
    # while reconstructing, so it auto-wires from such a run just like a reload.
    "modart": {"md": "particles", "vol": "volume", "motion_correction": "motion_model"},
    # Same wiring as modart, minus `vol`: the Fourier-gridding reconstructor builds its
    # map from the particles alone, so it takes no reference volume. Its motion kind is
    # narrower than modart's -- see `transport_motion_model`.
    "reconstruct_motion_corrected": {"md": "particles", "motion_correction": "transport_motion_model"},
    # reconstruct_volume needs nothing but the posed particles: it reconstructs the
    # consensus map from their alignments, so it takes no reference volume to wire in.
    "reconstruct_volume": {"md": "particles"},
    "preprocess_particles": {"md": "particles"},
    "preprocess_volumes": {"vol": "volume"},
}

# Bindable fields that accept *several* sources (the program's flag is nargs="+").
# Each wired source is materialised into its own file and passed as a list.
MULTI_BINDABLE: dict[str, set[str]] = {
    "flexconsensus": {"input_space"},
    "preprocess_volumes": {"vol"},
}


def multi_bindable_fields(program: str) -> list[str]:
    """Bindable fields of `program` that accept multiple wired sources."""
    return sorted(MULTI_BINDABLE.get(program, set()))


def is_multi_bindable(program: str, field: str) -> bool:
    return field in MULTI_BINDABLE.get(program, set())

# The checkpoint subfolder each network program writes inside its run folder.
# A run only contains this when it actually trained/saved a model (e.g. a
# hetsiren *predict* run has none), so directory-existence filters sources for us.
_CHECKPOINT_DIR: dict[str, str] = {
    "hetsiren": "HetSIREN",
    "zernike3deep": "Zernike3Deep",
    "reconsiren": "ReconSIREN",
    "reconsiren_het_only": "ReconSIREN",   # het-only variant saves under the same subfolder
    "flexconsensus": "FlexConsensus",
    "latent_space_deconvolution": "deconvolver",
    "image_gray_scale_adjustment": "imageAdjustment",
    "volume_gray_scale_adjustment": "volumeAdjustment",
}

# Each `*_model` kind -> the producer programs whose checkpoint a consumer can
# reload, and how that consumer reads `--reload`:
#   "parent" -> the program does load(join(reload, "<ModelName>")) itself, so it
#               must be given the *run folder* (the checkpoint's parent);
#   "dir"    -> the program does load(reload) directly, so it must be given the
#               *checkpoint folder* itself.
# Most networks reload only their own type (and append the name); only
# decode_states_from_latents loads the checkpoint path directly, and it can
# reload any decoder network (anything exposing decode_volume).
_MODEL_KINDS: dict[str, dict[str, Any]] = {
    "hetsiren_model": {"programs": ["hetsiren"], "form": "parent"},
    "zernike3deep_model": {"programs": ["zernike3deep"], "form": "parent"},
    "reconsiren_model": {"programs": ["reconsiren"], "form": "parent"},
    "reconsiren_het_only_model": {"programs": ["reconsiren_het_only"], "form": "parent"},
    "flexconsensus_model": {"programs": ["flexconsensus"], "form": "parent"},
    "deconvolver_model": {"programs": ["latent_space_deconvolution"], "form": "parent"},
    "image_adjust_model": {"programs": ["image_gray_scale_adjustment"], "form": "parent"},
    "volume_adjust_model": {"programs": ["volume_gray_scale_adjustment"], "form": "parent"},
    "decoder_model": {"programs": ["hetsiren", "zernike3deep", "reconsiren", "reconsiren_het_only"], "form": "dir"},
    # MoDART's motion_correction loads the checkpoint path directly and calls
    # decode_field on it, which only HetSIREN and Zernike3Deep implement -- so this
    # kind is narrower than decoder_model (no reconsiren variants).
    "motion_model": {"programs": ["hetsiren", "zernike3deep"], "form": "dir"},
    # reconstruct_motion_corrected queries the displacement field at arbitrary points
    # (decode_field_at), which only HetSIREN's mass-transport decoder implements -- a
    # Zernike3Deep or fixed-grid HetSIREN checkpoint makes it exit with a UserWarning.
    # Hence a kind narrower than motion_model, further filtered on --transport_mass by
    # `job_produces` so only a run that actually trained one is offered.
    "transport_motion_model": {"programs": ["hetsiren"], "form": "dir"},
}

# Artifacts referenced *in place* (no copy): the consumer is handed the producer's
# own file. Particle metadata must not be copied out of its run folder -- it points
# at the stack sitting next to it.
#
# "unique_glob" yields a source only when the glob matches exactly one file. A
# preprocess_volumes run may emit a map *and* its mask; silently binding whichever
# sorts first would be worse than offering nothing, so a multi-volume run simply
# exposes no volume port and the user picks the file with the normal path picker.
_ARTIFACT_PRODUCERS: dict[str, list[dict[str, Any]]] = {
    "particles": [
        {
            "style": "path",
            "programs": {"preprocess_particles"},
            "globs": ["preprocessed_particles.xmd", "preprocessed_particles.star"],
        },
        {
            # A network predict run's refined metadata: the same particle stack, now
            # carrying the network's refined poses/shifts (and a latent column), so it
            # feeds anywhere a preprocessed particle set does -- reconstruct_volume,
            # modart, estimate_latent_covariances, ... Referenced in place like the
            # preprocessed set: it points at the same image stack. Each program writes
            # exactly one of these names (latents for the decoders, pose_shifts for the
            # reconstruction nets), so listing all four picks whichever the run wrote.
            "style": "path",
            "programs": {"hetsiren", "zernike3deep", "reconsiren", "reconsiren_het_only"},
            "globs": ["predicted_latents.xmd", "predicted_latents.star",
                      "predicted_pose_shifts.xmd", "predicted_pose_shifts.star"],
        },
    ],
    "volume": [
        {
            "style": "unique_glob",
            "programs": {"preprocess_volumes"},
            "globs": ["*_preprocessed.mrc"],
        },
        {
            # The consensus map a reconstruct_volume run produced -- the reference the
            # heterogeneity programs and MoDART take in `--vol`. The glob names the map
            # explicitly rather than matching `*.mrc`: a `--write_mask` run also writes
            # `consensus_mask.mrc`, and under "unique_glob" a second hit would silently
            # cost the run its volume port.
            "style": "unique_glob",
            "programs": {"reconstruct_volume"},
            "globs": ["consensus_reconstruction.mrc"],
        },
    ],
}

# Latents producers: a predict metadata column, or a ready filtered .npy.
_LATENT_PRODUCERS: list[dict[str, Any]] = [
    {
        "style": "metadata",
        "programs": {"hetsiren", "zernike3deep", "reconsiren", "reconsiren_het_only"},
        "globs": ["predicted_latents.xmd", "predicted_latents.star", "predicted_latents.*",
                  "predicted_pose_shifts.xmd", "predicted_pose_shifts.star", "predicted_pose_shifts.*"],
        "column": "latent_space",
    },
    {
        "style": "file",
        "programs": {"filter_latents"},
        "globs": ["filtered_latents.npy"],
    },
    {
        # FlexConsensus consensus space — a reduced latent space; bind it to a
        # viewer/annotate_space `z_space_reduced` (display), not to `z_space`.
        "style": "file",
        "programs": {"flexconsensus"},
        "globs": ["*_consensus.npy"],
    },
]


def bindable_fields(program: str) -> dict[str, str]:
    """{field_name: kind} for fields of `program` that accept a job as input."""
    return BINDABLE_FIELDS.get(program, {})


def _pretty_kind(kind: str) -> str:
    if kind == "latents":
        return "latents"
    if kind == "particles":
        return "a preprocessed particle set"
    if kind == "volume":
        return "a volume"
    if kind.endswith("_model"):
        return "a saved network"
    return kind


def _specs_for_kind(kind: str) -> list[dict[str, Any]]:
    if kind == "latents":
        return _LATENT_PRODUCERS
    if kind in _ARTIFACT_PRODUCERS:
        return _ARTIFACT_PRODUCERS[kind]
    if kind in _MODEL_KINDS:
        info = _MODEL_KINDS[kind]
        return [{"style": "dir", "programs": set(info["programs"]), "form": info["form"]}]
    return []


def producer_programs(kind: str) -> list[str]:
    """Program names whose output can produce `kind` (for client-side DnD checks)."""
    progs: set[str] = set()
    for spec in _specs_for_kind(kind):
        progs |= spec["programs"]
    return sorted(progs)


def latents_decoder_programs() -> list[str]:
    """Programs whose *predict* run yields latents AND a reloadable decoder network
    (so a finished run is directly viewable / decodable). Derived from the maps
    above, so a new such program is picked up automatically once registered here —
    no hardcoded program lists elsewhere to keep in sync."""
    return sorted(set(producer_programs("latents")) & set(_MODEL_KINDS["decoder_model"]["programs"]))


def produces(program: str) -> list[str]:
    """The artifact kinds a program's output can feed into another program's input
    (its workflow output ports): e.g. a predict run -> ``latents``; a trained
    network -> its ``*_model`` kinds."""
    kinds: set[str] = set()
    for spec in _LATENT_PRODUCERS:
        if program in spec["programs"]:
            kinds.add("latents")
    for kind, specs in _ARTIFACT_PRODUCERS.items():
        if any(program in spec["programs"] for spec in specs):
            kinds.add(kind)
    for kind, info in _MODEL_KINDS.items():
        if program in info["programs"]:
            kinds.add(kind)
    return sorted(kinds)


def job_produces(program: str, args: dict[str, Any] | None = None) -> list[str]:
    """The kinds a *specific* run can feed downstream, narrowed by its ``--mode``.

    ``produces(program)`` is the program's static capability, but a moded network
    only writes different artifacts per mode: in ``train`` it saves its checkpoint
    (a ``*_model`` kind) and no predict metadata; in ``predict`` it writes that
    metadata (``latents`` + ``particles``) and no checkpoint. So a *train* run must
    not be offered as a latents/particles source, nor a *predict* run as a model.
    Programs without a train/predict mode carry no such ``mode`` and are unaffected.
    """
    kinds = produces(program)
    args = args or {}
    # Only a HetSIREN trained with --transport_mass carries a displacement field, and
    # reconstruct_motion_corrected hard-fails without one. The flag is a store_true, so
    # a truthy value in the run's args is exactly "the checkpoint has a motion field".
    if not args.get("transport_mass"):
        kinds = [k for k in kinds if k != "transport_motion_model"]
    mode = args.get("mode")
    if mode == "predict":
        return [k for k in kinds if not k.endswith("_model")]
    if mode == "train":
        return [k for k in kinds if k.endswith("_model")]
    return kinds


def producers_for(program: str) -> dict[str, list[str]]:
    """{kind: [producer programs]} for every bindable field-kind of `program`."""
    kinds = set(bindable_fields(program).values())
    return {kind: producer_programs(kind) for kind in kinds}


def _match_producer(kind: str, program: str) -> dict[str, Any] | None:
    for spec in _specs_for_kind(kind):
        if program in spec["programs"]:
            return spec
    return None


def _is_loadable_checkpoint(path: Path) -> bool:
    """True only for a *complete* checkpoint dir, matching what
    ``NeuralNetworkCheckpointer.load`` reads: ``<dir>/checkpoints/{config,state}``.

    This is what makes the reload picker honest: a still-training run hasn't
    committed its final checkpoint yet, and a crashed run may have created the
    folder but never finished writing it -- neither passes this check, so only
    runs you can actually reload are offered.
    """
    cp = path / "checkpoints"
    return (cp / "config").is_file() and (cp / "state").exists()


# Programs whose training loop auto-resumes from an *intermediate* checkpoint it wrote
# inside its own output folder — restoring weights + optimizer + epoch, with no
# --reload needed (the loop keys off the folder's existence). Maps program -> the
# checkpoint and metrics subfolders it uses, plus an optional "phase_done" marker whose
# presence means the auto-resuming phase already finished (HetSIREN's inverse phase does
# NOT auto-resume, so a run past that point must not advertise "Continue").
_RESUMABLE: dict[str, dict[str, str]] = {
    "hetsiren":            {"checkpoint": "HetSIREN_CHECKPOINT",     "metrics": "HetSIREN_metrics",     "phase_done": "HetSIREN_No_Inv"},
    "zernike3deep":        {"checkpoint": "Zernike3Deep_CHECKPOINT", "metrics": "Zernike3Deep_metrics"},
    "reconsiren":          {"checkpoint": "ReconSIREN_CHECKPOINT",   "metrics": "ReconSIREN_metrics"},
    "reconsiren_het_only": {"checkpoint": "ReconSIREN_CHECKPOINT",   "metrics": "ReconSIREN_metrics"},
}


def is_resumable(program: str, workdir: str | None) -> bool:
    """True when ``program`` left a loadable intermediate checkpoint in ``workdir``
    that its own training loop will pick up if relaunched with the same output folder.

    Drives the "Continue" button, so it must not promise a resume the program won't
    actually perform: a run has to have committed its first intermediate checkpoint,
    and (HetSIREN) must not already be in its non-resuming inverse phase.
    """
    spec = _RESUMABLE.get(program)
    if not spec or not workdir:
        return False
    wd = Path(workdir)
    done = spec.get("phase_done")
    if done and (wd / done).is_dir():
        return False
    return _is_loadable_checkpoint(wd / spec["checkpoint"])


def resume_metrics_dir(program: str, workdir: str | None) -> Path | None:
    """The metrics / TensorBoard folder to clear before a continued run starts, so its
    curves restart cleanly instead of overlapping the interrupted run's. None when the
    program/workdir has none."""
    spec = _RESUMABLE.get(program)
    if not spec or not workdir:
        return None
    return Path(workdir) / spec["metrics"]


def _find_artifact(workdir: Path, spec: dict[str, Any], program: str) -> Path | None:
    if spec["style"] == "dir":
        name = _CHECKPOINT_DIR.get(program)
        if not name:
            return None
        path = workdir / name
        return path if path.is_dir() and _is_loadable_checkpoint(path) else None
    if spec["style"] == "unique_glob":
        hits = [hit for pattern in spec["globs"] for hit in sorted(workdir.glob(pattern)) if hit.is_file()]
        return hits[0] if len(hits) == 1 else None
    for pattern in spec["globs"]:
        hit = next(iter(sorted(workdir.glob(pattern))), None)
        if hit is not None and hit.is_file():
            return hit
    return None


def source_file_for(kind: str, job: dict[str, Any]) -> Path | None:
    """The output file/dir in `job` that can produce `kind`, or None."""
    workdir = job.get("workdir")
    if not workdir:
        return None
    program = job.get("program", "")
    spec = _match_producer(kind, program)
    if spec is None:
        return None
    return _find_artifact(Path(workdir), spec, program)


def reference_path(kind: str, source_job: dict[str, Any]) -> str | None:
    """The path a binding to `kind` from `source_job` resolves to **in place**.

    Mirrors :func:`produce` for the styles that reference their source without copying
    (particles/volumes/checkpoints), so a job's stored *literal* argument can be matched
    back to the run that produced it -- which is how a binding lost to an edit-and-rerun
    (the resolved path was frozen into the args, the binding dropped) is recovered.

    None for kinds materialised by copying into the *consumer's* own folder (latents):
    their stored path points inside the consumer, not the producer, so value-matching
    cannot identify the source. None too if the source hasn't produced the artifact.
    """
    spec = _match_producer(kind, source_job.get("program", ""))
    if spec is None or spec["style"] not in ("path", "unique_glob", "dir"):
        return None
    src = source_file_for(kind, source_job)
    if src is None:
        return None
    if spec["style"] == "dir":
        return str(src.parent if spec.get("form") == "parent" else src)
    return str(src)


def checkpoint_dir(program: str, reload_path: str) -> Path:
    """Resolve a directly-loadable checkpoint dir from a job's reload/pickled_nn
    value, which may already be the checkpoint or the run folder that holds it.

    annotate_space's ``pickled_nn`` is the checkpoint dir itself; a predict run's
    ``reload`` is the training run folder, so the checkpoint is ``<reload>/<Name>``.
    """
    p = Path(reload_path).expanduser()
    if _is_loadable_checkpoint(p):
        return p
    name = _CHECKPOINT_DIR.get(program)
    if name and _is_loadable_checkpoint(p / name):
        return p / name
    if p.is_dir():
        for child in sorted(p.iterdir()):
            if child.is_dir() and _is_loadable_checkpoint(child):
                return child
    return p


def compatible_sources(kind: str, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Finished jobs whose output can feed a `kind` input, newest first."""
    out = []
    for job in jobs:
        src = source_file_for(kind, job)
        if src is not None:
            out.append({
                "id": job["id"],
                "program": job["program"],
                "name": job.get("name"),
                "created_at": job.get("created_at"),
                "status": job.get("status"),
                "file": str(src),
            })
    return out


def produce(kind: str, source_job: dict[str, Any], dest_dir: Path, field: str) -> str:
    """Materialise the `kind` artifact from `source_job` into `dest_dir`.

    Returns the path of the file the consuming program should read. Raises
    AdapterError with a user-facing message on any failure.
    """
    program = source_job.get("program", "")
    spec = _match_producer(kind, program)
    if spec is None:
        raise AdapterError(f"a '{program}' run cannot provide {_pretty_kind(kind)}")
    src = source_file_for(kind, source_job)
    if src is None:
        if spec["style"] == "dir":
            hint = "did it finish training?"
        elif spec["style"] == "unique_glob":
            hint = "did it finish, and did it write exactly one volume?"
        else:
            hint = "did it finish?"
        raise AdapterError(f"the selected run hasn't produced {_pretty_kind(kind)} yet ({hint})")

    # Checkpoints are referenced in place (no copy). `src` is the checkpoint
    # folder (e.g. <run>/HetSIREN); "parent"-form consumers append the model
    # name themselves, so hand them the run folder instead to avoid <run>/X/X.
    if spec["style"] == "dir":
        return str(src.parent if spec.get("form") == "parent" else src)

    # Also referenced in place: particle metadata resolves its image stack from a
    # sibling path, and volumes are large binaries not worth duplicating per job.
    if spec["style"] in ("path", "unique_glob"):
        return str(src)

    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{field}.npy"
    if spec["style"] == "file":
        shutil.copyfile(src, out)
    else:
        _extract_latents_column(src, spec["column"], out)
    return str(out)


def _extract_latents_column(md_path: Path, column: str, out: Path) -> None:
    """Parse a string-encoded latents column from a metadata file into a .npy."""
    import numpy as np
    from xmipp_metadata.metadata import XmippMetaData

    md = XmippMetaData(str(md_path))
    if not md.isMetaDataLabel(column):
        raise AdapterError(f"'{column}' column not found in {md_path.name}")
    col = md.getMetaDataColumns(column)
    try:
        rows = [np.fromstring(str(s), sep=",") for s in col]
    except Exception as exc:  # noqa: BLE001
        raise AdapterError(f"could not parse the latents column: {exc}")
    widths = {r.size for r in rows}
    if not rows or widths == {0} or len(widths) != 1:
        raise AdapterError("the latents column is empty or has inconsistent dimensions")
    arr = np.asarray(rows, dtype=np.float32)
    np.save(out, arr)
