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
    # Folders the file browser may reach. Empty (the default) means unrestricted: it
    # can show anything the server process can read, which is what makes shared
    # storage like /data/<you>/... browsable with no setup. Set this to confine it.
    # It was never a security boundary — path-typed job arguments accept any path
    # (see command.py) — so confining it buys hygiene, not protection.
    file_roots: list[str] = field(default_factory=lambda: _env_list("HAX_GUI_FILE_ROOTS", []))
    # Where the file browser opens when no path is given. Not a restriction.
    browse_home: Path = field(default_factory=lambda: _env_path("HAX_GUI_BROWSE_HOME", Path.home()))
    # Shared auth token. Empty disables auth — allowed only on a loopback bind:
    # a public bind with no HAX_GUI_TOKEN falls back to the persisted token in
    # `token_path` (generated on first use). See `ensure_auth_for_bind`.
    auth_token: str = field(default_factory=lambda: os.environ.get("HAX_GUI_TOKEN", ""))
    # The address the server was asked to bind (set by `hax-gui start`). Unset
    # (e.g. a bare `uvicorn hax_gui.app:app`) is treated as loopback — the
    # secure default for the Host allowlist below.
    bind_host: str = field(default_factory=lambda: os.environ.get("HAX_GUI_BIND_HOST", ""))
    # Extra Host-header values to accept (comma-separated), for reverse-proxy
    # setups where the browser's Host is a public name the server can't infer.
    # "*" disables the check entirely.
    allowed_hosts: list[str] = field(
        default_factory=lambda: [h.strip() for h in os.environ.get("HAX_GUI_ALLOWED_HOSTS", "").split(",") if h.strip()]
    )
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
    # Minutes a viewer session (landscape/map/metadata) may go without a single
    # proxied request before it is stopped, releasing its port — and, for the
    # landscape viewer, its GPU memory. 0 disables the timeout.
    viewer_idle_minutes: float = field(
        default_factory=lambda: float(os.environ.get("HAX_GUI_VIEWER_IDLE_MINUTES", "120") or 0)
    )
    # CORS origins for the SPA during dev (Vite default is 5173).
    cors_origins: list[str] = field(
        default_factory=lambda: _env_list("HAX_GUI_CORS_ORIGINS", ["http://localhost:5173", "http://127.0.0.1:5173"])
    )

    @property
    def db_path(self) -> Path:
        return self.data_dir / "hax_gui.db"

    @property
    def token_path(self) -> Path:
        """Persisted auth token, generated on first public bind (0600)."""
        return self.data_dir / "token"

    @property
    def is_public_bind(self) -> bool:
        return bool(self.bind_host) and not is_loopback_host(self.bind_host)

    def load_or_create_token(self) -> str:
        """The persisted token, minted on first use. Never returns empty."""
        import secrets

        try:
            tok = self.token_path.read_text().strip()
            if tok:
                return tok
        except OSError:
            pass
        tok = secrets.token_urlsafe(32)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.token_path.write_text(tok + "\n")
        self.token_path.chmod(0o600)
        return tok

    def ensure_auth_for_bind(self) -> str | None:
        """Enforce 'no open server on a public interface'.

        On a public bind with no HAX_GUI_TOKEN, adopt the persisted token
        (creating it on first use) so the API is never exposed unauthenticated —
        the job API is arbitrary command execution as the server user. Returns a
        human-readable description of what happened, or None if nothing changed.
        """
        if self.auth_token or not self.is_public_bind:
            return None
        self.auth_token = self.load_or_create_token()
        return (f"public bind ({self.bind_host}) with no HAX_GUI_TOKEN: "
                f"using the persisted token from {self.token_path}")

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def viewer_logs_dir(self) -> Path:
        """Per-session stdout/stderr of the viewer subprocesses (see viewer_logs.py).

        Kept out of :attr:`runs_dir`, which is per-*job*: a viewer session is not a
        job, outlives none of them, and is pruned on its own schedule.
        """
        return self.data_dir / "viewer_logs"

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
        self.viewer_logs_dir.mkdir(parents=True, exist_ok=True)


def is_loopback_host(host: str) -> bool:
    """True for addresses that only the local machine can reach."""
    h = host.strip("[]").lower()
    return h in ("localhost", "::1", "") or h.startswith("127.")


settings = Settings()
