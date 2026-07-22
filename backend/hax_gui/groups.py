"""Parameter grouping and the advanced tier for the auto-generated forms.

A hax program can have upwards of fifty arguments (``hetsiren`` has 53), which as
one flat list is unusable in a 264-380 px side panel. Every form field therefore
carries two extra keys:

``group``
    Display group, e.g. ``"Regularisation"``. Rendered as an accordion section.
``advanced``
    Hidden behind the form's "Advanced parameters" switch.

Two sources feed those keys, in priority order:

1. **hax-em itself.** ``argparse`` already models grouping via
   ``parser.add_argument_group(title)``, and an action is a plain object so
   ``action.advanced = True`` survives to introspection time. When a program
   declares either, :mod:`hax_gui.introspect` records it and this module leaves it
   alone. This is the durable answer; see ``hax/cli/common_args.py``.
2. **The heuristic below.** hax argument names are regular enough (``log_*``,
   ``*_lambda``, ``*_anneal_*``, ``pose_refine_*``) that a short ordered rule list
   classifies almost everything correctly. This means the GUI groups every program
   today, and each program that gains real argparse groups simply stops relying on
   it. Anything unmatched lands in ``General``, which sorts near the top so a
   missed argument is conspicuous rather than buried.
"""

from __future__ import annotations

import re
from typing import Any

# Accordion order. ``General`` sits second so arguments the rules missed stay
# visible right under the inputs instead of sinking to the bottom.
GROUP_ORDER = [
    "Inputs & data",
    "General",
    "Model",
    "Training",
    "Pose refinement",
    "Regularisation",
    "Schedules",
    "Masking",
    "Reconstruction",
    "Performance",
    "Logging",
]
FALLBACK_GROUP = "General"

# First match wins, so order is significant: exact-name rules precede the broader
# suffix patterns they would otherwise be swallowed by.
_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(p), g)
    for p, g in [
        # -- inputs, on-disk artifacts and where the run reads from ------------
        (r"^(md|vol|mask|sr|ctf_type|load_images_to_ram|ssd_scratch_folder|config"
         r"|latents|latents_file|z_space|z_space_reduced|covariances|nn_path"
         r"|volumes_path|logdir|path|input_space|relative_image_paths"
         r"|server_functions_path|chimerax_binary|env_name)$", "Inputs & data"),

        # -- logging cadence ---------------------------------------------------
        (r"^log_", "Logging"),

        # -- mask construction (before the generic `mask` rule above is moot) ---
        (r"^(mask_|write_mask$)", "Masking"),

        # -- pose / shift refinement ------------------------------------------
        (r"^(pose_refine|shift_refine|refine_positions|refine_current_assignment"
         r"|do_not_use_anchor_rotations)", "Pose refinement"),

        # -- schedules and warmups (before the `_reg` rule, `*_warmup_epochs`
        #    would otherwise read as training) ---------------------------------
        (r"(_anneal_|_anneal$|_warmup_epochs$)", "Schedules"),

        # -- loss terms and regularisers --------------------------------------
        (r"(_lambda$|_reg$|_tv$|_kappa$|^kl_|^denoising_strength$|^l1_delta$"
         r"|^jacobian$|^arap_|^occupancy_)", "Regularisation"),

        # -- reconstruction / volume synthesis --------------------------------
        (r"^(correction|ctf_correction|wiener_epsilon|motion_correction"
         r"|reconstruct_halves|warp_sigma|min_coverage|do_not_learn_volume"
         r"|deconvolution_strength|no_denoise|export_deformation|no_gray_scale"
         r"|crop_box_size|new_box_size)", "Reconstruction"),

        # -- optimisation loop -------------------------------------------------
        (r"^(mode|epochs|batch_size|learning_rate|lr_schedule|reload|ema_decay"
         r"|grad_clip_norm|loss_type|dataset_split_fraction|tau)$", "Training"),

        # -- machine resources. `thr` and `n_neighbours` are deliberately absent:
        #    they mean "threads"/"neighbours" in some programs and "threshold"/
        #    "neighbourhood size" in others (``filter_latents``), so they fall to
        #    ``General`` rather than being confidently misfiled. -----------------
        (r"^(threads|device|num_workers|num_read_workers|num_write_workers"
         r"|field_batch|field_chunk)$", "Performance"),

        # -- architecture ------------------------------------------------------
        (r"^(lat_dim|num_gaussians|implicit_network|point_transformer"
         r"|transport_mass|local_reconstruction|rigid_gauge|symmetry_group"
         r"|num_components|train_inverse|L1|L2)$", "Model"),
    ]
]

# Parameters that need a reason to touch: internals, tuning knobs whose default is
# the product of experiment, and escape hatches. A *required* argument is never
# treated as advanced (see :func:`annotate`) — hiding something the program cannot
# run without would make the form silently un-submittable.
ADVANCED = {
    # architecture / experimental heads
    "transport_mass", "num_gaussians", "local_reconstruction", "implicit_network",
    "point_transformer", "rigid_gauge", "train_inverse",
    # optimiser internals
    "dataset_split_fraction", "grad_clip_norm", "ema_decay",
    # regularisers whose defaults encode a lot of tuning
    "denoising_strength", "occupancy_kappa", "occupancy_tv", "arap_rotation",
    "decoupling_lambda", "distance_preservation_lambda", "coord_reg", "jacobian",
    "l1_delta", "tv_lambda",
    # schedules
    "sigma_anneal_factor", "freq_anneal_start", "latent_warmup_epochs",
    "occupancy_warmup_epochs", "pose_refine_warmup_epochs",
    # logging internals
    "log_time_budget", "log_sync",
    # machine resources — good defaults, wrong values are a footgun
    "threads", "device", "num_workers", "num_read_workers",
    "num_write_workers", "field_batch", "field_chunk",
    # mask shaping. `write_mask` itself stays basic: it decides whether an output
    # is produced at all, and these two only take effect once it is on.
    "mask_dilate", "mask_taper", "mask_threshold", "mask_radius",
    # reconstruction internals, plus the two "skip a good default" escape hatches
    "wiener_epsilon", "warp_sigma", "min_coverage", "deconvolution_strength",
    "do_not_learn_volume", "do_not_use_anchor_rotations",
    "refine_current_assignment", "no_denoise", "no_gray_scale_calibration",
    # environment plumbing
    "config", "relative_image_paths", "chimerax_binary", "env_name", "input_space",
}
# Deliberately NOT advanced, though they look like flags: `return_ids`,
# `only_view`, `predict_value`, `predicts_value`, `also_consensus`,
# `export_deformation`, `check`, `yes`. Each changes what the program *does*
# rather than tuning how it does it, and in the smaller programs one of them is
# most of the form.


def group_for(name: str) -> str:
    """Heuristic group for an argument name (``General`` when nothing matches)."""
    for pattern, group in _RULES:
        if pattern.search(name):
            return group
    return FALLBACK_GROUP


def annotate(fields: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return ``fields`` with ``group``/``advanced`` filled in.

    Values already present (put there by :mod:`hax_gui.introspect` from real
    argparse groups / ``action.advanced``) are preserved, so annotating a program
    in hax-em progressively overrides the heuristic one argument at a time.
    """
    out = []
    for field in fields:
        group = field.get("group") or group_for(field["name"])
        advanced = field.get("advanced")
        if advanced is None:
            advanced = field["name"] in ADVANCED
        # A required argument must stay reachable without flipping a switch.
        out.append({**field, "group": group, "advanced": bool(advanced) and not field.get("required")})

    # Guard: a program whose every argument reads as advanced would render as an
    # empty form with the switch off, which is a bug rather than a preference.
    # Small programs (``update``, ``filter_latents``) are the realistic case.
    if out and not any(not f["advanced"] for f in out):
        out = [{**f, "advanced": False} for f in out]
    return out


def order_key(group: str) -> int:
    """Sort position of a group; unknown groups sort after the known ones."""
    return GROUP_ORDER.index(group) if group in GROUP_ORDER else len(GROUP_ORDER)
