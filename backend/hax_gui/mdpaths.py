"""Find the directory a metadata file's relative image references resolve against.

Xmipp/Relion metadata points at its image stacks by *relative* path, resolved
against the process working directory — which the file itself never records. The
common case is that the paths are relative to the metadata's own folder, so
launching from there works. But Scipion writes them relative to the **project
root**, several levels up::

    /data/you/ScipionUserData/projects/P/Runs/006254_Import/extra/particles.xmd
        _image  ->  000001@Runs/006254_Import/tmp/images.mrcs

Rather than guess, we read the first image reference out of the file and walk up
from the metadata's folder until the reference actually resolves. Nearest match
wins, which reproduces the folder-relative convention when it applies.
"""

from __future__ import annotations

from pathlib import Path

# The label naming the particle stack, lowercased: Xmipp `_image`, Relion
# `_rlnImageName`.
_IMAGE_LABELS = frozenset({"image", "rlnimagename"})

# Both .xmd and .star are STAR text; .cs is a binary numpy record array.
_STAR_SUFFIXES = frozenset({".xmd", ".star"})

# We stop at the first data row of the first labelled loop, so a well-formed file
# costs a few dozen lines. This only bounds a pathological one.
_MAX_SCAN_LINES = 4096


def first_image_ref(path: Path) -> str | None:
    """The first image reference in a STAR file, verbatim (``idx@stack.mrcs``).

    ``None`` when the file has no loop carrying an image label — a volume list, a
    latents table, an unreadable file. Multi-block files (Relion's ``data_optics``
    ahead of ``data_particles``) are handled by skipping loops without the label.
    """
    labels: list[str] = []
    in_loop = False
    try:
        with path.open("r", errors="replace") as fh:
            for lineno, raw in enumerate(fh):
                if lineno >= _MAX_SCAN_LINES:
                    return None
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("data_"):
                    labels, in_loop = [], False
                elif line == "loop_":
                    labels, in_loop = [], True
                elif in_loop and line.startswith("_"):
                    labels.append(line.split()[0][1:].lower())
                elif in_loop and labels:
                    # First data row of this loop. If it has no image label the
                    # loop is the wrong block; stop reading it and look further.
                    col = next((i for i, lb in enumerate(labels) if lb in _IMAGE_LABELS), None)
                    if col is None:
                        in_loop = False
                        continue
                    fields = line.split()
                    return fields[col] if col < len(fields) else None
    except OSError:
        return None
    return None


def image_base_dir(md_path: str | Path) -> Path | None:
    """Directory from which ``md_path``'s image references resolve.

    ``None`` when there is nothing to resolve (absolute refs, no image column, a
    ``.cs`` file) or when no ancestor of the metadata makes the reference exist —
    callers should fall back to the metadata's own folder.
    """
    path = Path(md_path).expanduser()
    if path.suffix.lower() not in _STAR_SUFFIXES or not path.is_file():
        return None
    ref = first_image_ref(path)
    if not ref:
        return None
    rel = ref.split("@", 1)[-1]        # drop the 1-based Xmipp stack index
    if not rel or Path(rel).is_absolute():
        return None
    start = path.resolve().parent
    for base in (start, *start.parents):
        if (base / rel).exists():
            return base
    return None
