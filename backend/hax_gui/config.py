"""Runtime configuration for the hax-em GUI backend.

Single-tenant deployment (see PLAN.md §10): one shared token behind TLS, no
per-user accounts. All paths default to sensible locations and can be overridden
with environment variables so the same code runs in dev and on the GPU server.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# The package directory; bundled assets (the SPA, the WebGL viewer, and the seed
# schema cache) live here so they resolve identically for an editable install and
# a pip/PyPI wheel (pip unpacks wheels, so __file__-relative paths are reliable).
_PKG_DIR = Path(__file__).resolve().parent


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if not raw:
        return default
    return [p for p in raw.split(os.pathsep) if p]


@dataclass
class Settings:
    # Where job metadata (SQLite) and per-job log files live.
    data_dir: Path = field(default_factory=lambda: _env_path("HAX_GUI_DATA_DIR", Path.home() / ".hax_gui"))
    # Console entry point used to launch programs (on PATH inside the hax env).
    hax_cli: str = field(default_factory=lambda: os.environ.get("HAX_PROJECT_MANAGER", "hax_project_manager"))
    # Roots the file browser is allowed to expose (path-traversal sandbox).
    file_roots: list[str] = field(
        default_factory=lambda: _env_list("HAX_GUI_FILE_ROOTS", [str(Path.home())])
    )
    # Shared auth token. Empty disables auth (dev only).
    auth_token: str = field(default_factory=lambda: os.environ.get("HAX_GUI_TOKEN", ""))
    # Whether the GUI may start a local Ollama when one isn't already running, and
    # stop that (and only that) instance on shutdown. A user-run Ollama is never
    # touched. Off with HAX_GUI_MANAGE_OLLAMA=0.
    manage_ollama: bool = field(
        default_factory=lambda: os.environ.get("HAX_GUI_MANAGE_OLLAMA", "1") not in ("0", "false", "no", "")
    )
    # Whether the assistant may propose GUI actions (create draft jobs from a data
    # path). Off by default — opt in with HAX_GUI_ASSISTANT_ACTIONS=1. Proposals
    # are always draft-only (never launched) and require the user to confirm.
    assistant_actions: bool = field(
        default_factory=lambda: os.environ.get("HAX_GUI_ASSISTANT_ACTIONS", "0") in ("1", "true", "yes")
    )
    # CORS origins for the SPA during dev (Vite default is 5173).
    cors_origins: list[str] = field(
        default_factory=lambda: _env_list("HAX_GUI_CORS_ORIGINS", ["http://localhost:5173", "http://127.0.0.1:5173"])
    )

    @property
    def db_path(self) -> Path:
        return self.data_dir / "hax_gui.db"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def schema_cache(self) -> Path:
        """Writable program-schema cache. Defaults under the data dir (never the
        installed package, which may be read-only); falls back to the shipped seed
        in :attr:`packaged_schema_cache` when this file is absent."""
        return _env_path("HAX_GUI_SCHEMA_CACHE", self.data_dir / "program_schemas.json")

    @property
    def packaged_schema_cache(self) -> Path:
        """Read-only schema cache shipped inside the package (seed/fallback)."""
        return _PKG_DIR / "schemas_cache" / "program_schemas.json"

    @property
    def frontend_dir(self) -> Path:
        """Bundled single-page-app directory served at ``/``."""
        return _PKG_DIR / "frontend"

    @property
    def viewer_server(self) -> Path:
        """Bundled WebGL viewer backend launched per session (as a subprocess)."""
        return _PKG_DIR / "viewer_app" / "server.py"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
