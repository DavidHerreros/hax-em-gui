"""Per-layer exports into the owning project (subset metadata + average volume).

Extracted from server.py. The decoder is passed in; xmipp_metadata is imported
lazily so importing this module stays cheap.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation only — the decoder instance is passed in
    from server import Decoder

class Exporter:
    """Writes per-layer outputs (subset metadata + average-latent volume) into the
    owning project. Only available when the viewer is launched with a project dir
    (and, for particle metadata, a row-aligned `--metadata`)."""

    def __init__(self, decoder: Decoder, metadata: str | None, project_dir: str | None) -> None:
        self.decoder = decoder
        self.metadata_path = metadata
        self.project_dir = Path(project_dir).expanduser() if project_dir else None
        self._md = None  # lazy XmippMetaData

    @property
    def available(self) -> bool:
        return self.project_dir is not None

    def _md_table(self):
        if self._md is None and self.metadata_path:
            from xmipp_metadata.metadata import XmippMetaData
            self._md = XmippMetaData(self.metadata_path)
        return self._md

    def save_layer(self, name: str, indices: list[int], fmt: str) -> dict:
        import re
        import numpy as np
        from xmipp_metadata.image_handler import ImageHandler

        if self.project_dir is None:
            raise RuntimeError("export needs a project directory (open the viewer from a job)")
        idx = [int(i) for i in indices if 0 <= int(i) < self.decoder.total]
        if not idx:
            raise RuntimeError("layer has no valid states to save")
        safe = re.sub(r"[^\w.-]+", "_", str(name)).strip("_") or "layer"
        folder = self.project_dir / safe
        folder.mkdir(parents=True, exist_ok=True)
        files: list[str] = []

        # volume decoded from the average latent of the whole layer
        avg = np.mean([self.decoder._latent(i) for i in idx], axis=0)
        vol = self.decoder._decode_latent(avg)
        vol_name = f"{safe}_average.mrc"
        ImageHandler().write(vol, str(folder / vol_name), overwrite=True)
        files.append(vol_name)

        # subset metadata for the real particles in the layer (centroids have no row)
        real = [i for i in idx if i < self.decoder.landscape.n]
        n_particles = 0
        md = self._md_table()
        if md is not None and real:
            ext = fmt if fmt in {"xmd", "star", "cs"} else "xmd"
            from xmipp_metadata.metadata import XmippMetaData
            sub = XmippMetaData()
            # `real` addresses the landscape, which a filter renumbers; the metadata never
            # moves. This map is what keeps an exported subset pointing at the particles it
            # claims to be — it is the whole reason filtering lives here rather than in a
            # `filtered_latents.npy`, which renumbers from zero and saves no way back.
            rows = self.decoder.landscape.rows[real]
            sub.table = md.table.iloc[rows].reset_index(drop=True)
            md_name = f"{safe}.{ext}"
            sub.write(str(folder / md_name))
            files.append(md_name)
            n_particles = len(real)
        return {"folder": str(folder), "files": files, "n": len(idx), "n_particles": n_particles}

def _parse_seeds(seeds: str) -> list[int] | None:
    """"3,17,42" -> [3, 17, 42]; empty or malformed -> None (i.e. unseeded KMeans)."""
    if not seeds.strip():
        return None
    try:
        return [int(x) for x in seeds.split(",") if x.strip() != ""]
    except ValueError:
        return None
