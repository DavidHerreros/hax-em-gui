"""Best-effort update detection for ``hax-em`` and ``hax-em-gui``.

Mirrors the philosophy of :mod:`hax_gui.gpus`: every probe is wrapped so it
degrades to ``kind="unknown"`` / ``error`` on any failure and never raises. It
distinguishes two kinds of user (the install layout decides, not a setting):

* **production** — a plain PyPI/wheel install. We compare the installed version
  against the latest on PyPI and suggest ``pip install -U <pkg>``.
* **devel** — an *editable* local clone (``pip install -e``). We read the source
  tree from PEP 610 ``direct_url.json`` and compare the checked-out branch with
  its upstream (``git fetch`` then ahead/behind), suggesting ``git pull``.

A ``git+<url>`` (non-editable) install lands in site-packages with no working
tree to inspect or switch branches in, so it is reported as ``kind="vcs"`` with
a reinstall command. Branch switching (a future phase) only applies to editable
clones, where ``repo`` points at a real checkout.
"""

from __future__ import annotations

import importlib.metadata as im
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import httpx

# Never let git block on an interactive credential/passphrase prompt (a private
# remote would otherwise hang the probe until the timeout kills it).
_GIT_ENV = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}

# The GUI package itself: updating it modifies the running server's own code, so
# the update must be followed by a server restart (the others don't need one).
GUI_DIST = "hax-em-gui"

# hax-em ships GPU dependencies behind CUDA extras (see hax-em's pyproject:
# `cuda12`/`cuda13`). pip records no extras for an installed dist, so we infer
# the one in use from the CUDA-variant packages actually installed — their names
# encode the CUDA major version. First match wins; order is most-specific first.
_CUDA_EXTRAS = {
    "cuda13": ("cuml-cu13", "cupy-cuda13x", "jax-cuda13-plugin", "jax-cuda13-pjrt"),
    "cuda12": ("cuml-cu12", "cupy-cuda12x", "jax-cuda12-plugin", "jax-cuda12-pjrt"),
}

# The two packages we watch. ``module`` is only a fallback hint for locating the
# source tree; the authoritative path comes from direct_url.json when editable.
# ``cuda_extras`` marks a package whose (re)install command must carry the
# detected CUDA extra so its GPU deps move in lockstep.
_PACKAGES = [
    {"dist": "hax-em", "module": "hax", "cuda_extras": True},
    {"dist": GUI_DIST, "module": "hax_gui", "cuda_extras": False},
]

_PYPI = "https://pypi.org/pypi/{dist}/json"
_NET_TIMEOUT = 8.0
_GIT_TIMEOUT = 20.0  # `git fetch` hits the network


def check_updates() -> list[dict[str, Any]]:
    """Probe every watched package. Synchronous and blocking (it does network
    I/O and may run ``git fetch``); call it from a thread/executor. Never raises.
    """
    return [_package_status(p) for p in _PACKAGES]


def status_for(name: str) -> dict[str, Any] | None:
    """Fresh status for a single watched package, or ``None`` if not watched.

    Used by the updater so the dirty/ahead-behind checks reflect the repo *now*,
    not the cached background poll. Blocking; call from a thread/executor."""
    for p in _PACKAGES:
        if p["dist"] == name:
            return _package_status(p)
    return None


def is_watched(name: str) -> bool:
    return any(p["dist"] == name for p in _PACKAGES)


def local_checkout(name: str) -> dict[str, Any] | None:
    """A local git checkout backing this install — editable *or not* — as
    ``{"repo": Path, "editable": bool}``, else ``None``.

    A non-editable local-directory install (``pip install .``) still records its
    source dir in direct_url.json, and the branch panel can manage that checkout
    (switching it needs a reinstall to take effect, hence the ``editable`` flag).
    A pure PyPI/conda install has no direct_url and returns ``None`` — the panel
    then falls back to a configured source path."""
    if not is_watched(name):
        return None
    direct = _direct_url(name)
    if not direct:
        return None
    url = direct.get("url", "")
    if not url.startswith("file://"):
        return None
    repo = Path(url[len("file://"):])
    if not (repo / ".git").exists():
        return None
    return {"repo": repo, "editable": bool(direct.get("dir_info", {}).get("editable"))}


def reinstall_command(name: str, repo: str) -> str:
    """The command to (re)install ``name`` from a non-editable checkout so a
    branch switch there takes effect, carrying the detected CUDA extra."""
    extra = detect_cuda_extra() if _uses_cuda_extras(name) else None
    return f"pip install {_spec(repo, extra)}"


def _uses_cuda_extras(name: str) -> bool:
    return any(p["dist"] == name and p.get("cuda_extras") for p in _PACKAGES)


def watched_packages() -> list[dict[str, Any]]:
    """The watched package descriptors (dist/module/cuda_extras), in order."""
    return list(_PACKAGES)


def detect_cuda_extra() -> str | None:
    """The CUDA extra (``cuda12``/``cuda13``) in use, inferred from the installed
    GPU-variant packages, or ``None`` if none are present (CPU-only install)."""
    try:
        installed = {
            (d.metadata["Name"] or "").lower() for d in im.distributions()
        }
    except Exception:
        return None
    for extra, probes in _CUDA_EXTRAS.items():
        if any(p in installed for p in probes):
            return extra
    return None


def _package_status(pkg: dict[str, Any]) -> dict[str, Any]:
    dist = pkg["dist"]
    status: dict[str, Any] = {
        "name": dist,
        "kind": "unknown",
        "current": None,
        "latest": None,
        "update_available": False,
        "command": None,
        "is_gui": dist == GUI_DIST,
        "error": None,
    }
    # The CUDA extra (if this package uses one) drives both the pypi upgrade and
    # the editable reinstall commands so GPU deps move with the package.
    extra = detect_cuda_extra() if pkg.get("cuda_extras") else None
    status["extra"] = extra
    try:
        status["current"] = im.version(dist)
    except im.PackageNotFoundError:
        status["error"] = "not installed"
        return status
    except Exception as exc:  # pragma: no cover - defensive
        status["error"] = str(exc)
        return status

    direct = _direct_url(dist)
    repo = _editable_repo(direct, pkg["module"])
    if repo is not None:
        status["kind"] = "editable"
        _fill_git(status, repo, extra)
    elif direct and direct.get("vcs_info"):
        status["kind"] = "vcs"
        url = direct.get("url", "")
        status["vcs_url"] = url or None
        status["command"] = f"pip install -U 'git+{url}'" if url else f"pip install -U {dist}"
    else:
        status["kind"] = "pypi"
        _fill_pypi(status, dist, extra)
    return status


def _spec(target: str, extra: str | None) -> str:
    """A pip target with its extra, quoted for shells that glob ``[]`` (zsh)."""
    return f"'{target}[{extra}]'" if extra else target


# -- install-kind introspection (PEP 610) ----------------------------------
def _norm(name: str | None) -> str:
    """PEP 503 name normalization (so ``hax-em`` and ``hax_em`` compare equal)."""
    return re.sub(r"[-_.]+", "-", (name or "").lower())


def _direct_url(dist: str) -> dict[str, Any] | None:
    """PEP 610 ``direct_url.json`` for ``dist``, or ``None``.

    Robust to *duplicate metadata*: a stale ``.egg-info`` left in the source tree
    (e.g. from an earlier ``setup.py develop``/``egg_info``) can shadow the real
    PEP 660 ``.dist-info`` and carries no direct_url.json — so ``distribution()``
    may hand back the wrong one and editability looks absent. Scan every metadata
    directory for the project and use whichever actually has direct_url.json."""
    target = _norm(dist)
    try:
        cands = [d for d in im.distributions() if _norm(d.metadata["Name"]) == target]
    except Exception:
        cands = []
    for d in cands:
        try:
            text = d.read_text("direct_url.json")
        except Exception:
            text = None
        if text:
            try:
                return json.loads(text)
            except (ValueError, TypeError):
                continue
    return None


def _editable_repo(direct: dict[str, Any] | None, module: str) -> Path | None:
    """Return the source tree for an *editable* install, else ``None``.

    Editable installs record ``dir_info.editable: true`` and a ``file://`` URL
    in direct_url.json. We only treat it as editable when that tree is an actual
    git checkout (so branch comparison is meaningful)."""
    if not direct:
        return None
    if not direct.get("dir_info", {}).get("editable"):
        return None
    url = direct.get("url", "")
    if not url.startswith("file://"):
        return None
    repo = Path(url[len("file://"):])
    return repo if (repo / ".git").exists() else None


# -- production: compare against PyPI ---------------------------------------
def _fill_pypi(status: dict[str, Any], dist: str, extra: str | None) -> None:
    status["command"] = f"pip install -U {_spec(dist, extra)}"
    try:
        resp = httpx.get(_PYPI.format(dist=dist), timeout=_NET_TIMEOUT)
        resp.raise_for_status()
        latest = resp.json()["info"]["version"]
    except Exception as exc:
        status["error"] = f"pypi: {exc}"
        return
    status["latest"] = latest
    if status["current"] is not None:
        status["update_available"] = _is_newer(latest, status["current"])


# -- devel: compare checkout against its upstream ---------------------------
def _fill_git(status: dict[str, Any], repo: Path, extra: str | None) -> None:
    status["repo"] = str(repo)
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    status["branch"] = branch
    status["dirty"] = bool(_git(repo, "status", "--porcelain"))
    status["command"] = f"git -C {repo} pull --ff-only"
    status["pyproject_changed"] = False
    # The command the user runs to (re)install deps after pulling — carries the
    # detected CUDA extra so hax-em's GPU stack stays consistent.
    status["dep_command"] = f"pip install -e {_spec(str(repo), extra)}"
    # How to bring the server back up after a manual update (GUI package only).
    status["start_command"] = "hax-gui start"

    # Best-effort refresh of remote-tracking refs; ignore failures (offline, etc).
    _git(repo, "fetch", "--quiet", timeout=_GIT_TIMEOUT)

    upstream = _git(repo, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    if not upstream:
        status["error"] = "no upstream tracking branch"
        return
    status["upstream"] = upstream
    counts = _git(repo, "rev-list", "--left-right", "--count", "HEAD...@{u}")
    if counts:
        try:
            ahead, behind = (int(x) for x in counts.split())
            status["ahead"] = ahead
            status["behind"] = behind
            status["update_available"] = behind > 0
            # Do the incoming commits touch pyproject.toml? If so, dependencies
            # may have changed and a plain `git pull` won't install them.
            if behind > 0:
                changed = _git(repo, "diff", "--name-only", "HEAD..@{u}", "--", "pyproject.toml")
                status["pyproject_changed"] = bool(changed)
        except ValueError:
            status["error"] = "could not parse ahead/behind"


def _git(repo: Path, *args: str, timeout: float = _GIT_TIMEOUT) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True, text=True, timeout=timeout, check=True,
            env=_GIT_ENV, stdin=subprocess.DEVNULL,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return out.stdout.strip()


# -- version comparison -----------------------------------------------------
def _is_newer(candidate: str, installed: str) -> bool:
    try:
        from packaging.version import Version

        return Version(candidate) > Version(installed)
    except Exception:
        # Fallback: numeric-tuple compare, then plain string inequality.
        def parts(v: str) -> tuple[int, ...]:
            return tuple(int(x) for x in v.split(".") if x.isdigit())

        cp, ip = parts(candidate), parts(installed)
        if cp and ip:
            return cp > ip
        return candidate != installed
