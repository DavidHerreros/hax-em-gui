"""Git branch status & switching for hax source checkouts (Phase 3).

Read-mostly by design: listing the current branch, ahead/behind, dirty flag and
the available local/remote branches is always safe and never mutates anything.
The one mutating action — :func:`switch` — runs ``git checkout`` and is refused
on a dirty tree (so the GUI never silently carries or clobbers uncommitted work)
and only accepts a branch that already exists (no arbitrary git args). It never
attempts to resolve conflicts; a failed checkout returns git's own message.

A checkout is resolved, in order, from:
  1. an **editable** install (``pip install -e``) — switching applies *live*;
  2. a **non-editable local** install (``pip install .``) — the install's source
     dir; switching needs a reinstall to take effect;
  3. a **configured** source path — for a PyPI/conda install plus a separate dev
     clone; switching that clone needs a reinstall too.
Every watched package is reported (so the panel can offer "set source folder"
for one with no checkout yet); ``available`` says whether a checkout was found.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import settings_store, updates
from .updates import GUI_DIST, _git  # reuse the hardened git runner (no prompts)


class BranchError(Exception):
    """Switch refused or failed; the message is shown to the user verbatim."""


def list_all() -> list[dict[str, Any]]:
    """Branch status for every watched package (a stub when no checkout exists)."""
    return [status(p["dist"]) for p in updates.watched_packages()]


def status(name: str) -> dict[str, Any]:
    """Branch status for one package. Always returns a dict; ``available`` is
    False (with ``configurable`` True) when no source checkout is resolved.

    Blocking (runs git); call from a thread/executor. Does *not* fetch — the
    remote-tracking refs are as fresh as the last update check; the panel offers
    an explicit refresh that fetches."""
    base: dict[str, Any] = {
        "name": name,
        "is_gui": name == GUI_DIST,
        "available": False,
        "configurable": True,
        "configured": name in settings_store.source_repos(),
        "repo": None,
        "source": None,
        "live": False,
        "error": None,
    }
    resolved = _resolve(name)
    if resolved is None:
        return base

    repo = resolved["repo"]
    base.update({
        "available": True,
        "repo": str(repo),
        "source": resolved["source"],
        "live": resolved["live"],
        "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_git(repo, "status", "--porcelain")),
        "upstream": None,
        "ahead": None,
        "behind": None,
        "branches": _branch_names(repo, _git(repo, "rev-parse", "--abbrev-ref", "HEAD")),
    })
    # Not the live install → switching only moves the clone; show how to apply.
    if not resolved["live"]:
        base["reinstall_command"] = updates.reinstall_command(name, str(repo))
    upstream = _git(repo, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    if upstream:
        base["upstream"] = upstream
        counts = _git(repo, "rev-list", "--left-right", "--count", "HEAD...@{u}")
        if counts:
            try:
                ahead, behind = (int(x) for x in counts.split())
                base["ahead"], base["behind"] = ahead, behind
            except ValueError:
                base["error"] = "could not parse ahead/behind"
    return base


def fetch(name: str) -> dict[str, Any] | None:
    """Refresh remote-tracking refs, then return the fresh status."""
    resolved = _resolve(name)
    if resolved is None:
        return None
    _git(resolved["repo"], "fetch", "--quiet", "--prune", timeout=updates._GIT_TIMEOUT)
    return status(name)


def set_source(name: str, path: str | None) -> dict[str, Any]:
    """Configure (or clear) the source clone for a package. A non-empty path must
    be an existing git checkout. Returns the fresh status."""
    if not updates.is_watched(name):
        raise BranchError(f"'{name}' is not a managed package")
    if path:
        repo = Path(path).expanduser()
        if not repo.is_dir():
            raise BranchError(f"no such folder: {repo}")
        if not (repo / ".git").exists():
            raise BranchError(f"{repo} is not a git repository")
        path = str(repo)
    settings_store.set_source_repo(name, path)
    return status(name)


def switch(name: str, branch: str) -> dict[str, Any]:
    """Check out ``branch`` in the package's checkout. Refused on a dirty tree or
    an unknown branch. Returns ``{branch, output, live, reinstall_command?}``."""
    resolved = _resolve(name)
    if resolved is None:
        raise BranchError(f"no source checkout is configured for '{name}'")
    repo = resolved["repo"]

    current = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    if branch == current:
        raise BranchError(f"already on '{branch}'")
    if bool(_git(repo, "status", "--porcelain")):
        raise BranchError("the working tree has uncommitted changes — commit or stash them first")
    # Only switch to a branch we actually listed (no arbitrary checkout args).
    valid = {b["name"] for b in _branch_names(repo, current)}
    if branch not in valid:
        raise BranchError(f"unknown branch '{branch}'")

    # `git checkout <name>` DWIMs a remote-only branch into a local tracking one.
    result = _git_run(repo, "checkout", branch)
    if result["code"] != 0:
        raise BranchError(result["output"].strip() or f"could not switch to '{branch}'")
    out = {"branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
           "output": result["output"], "live": resolved["live"]}
    if not resolved["live"]:
        out["reinstall_command"] = updates.reinstall_command(name, str(repo))
    return out


# -- repo resolution -------------------------------------------------------
def _resolve(name: str) -> dict[str, Any] | None:
    """Locate the checkout for ``name`` → ``{repo, source, live}`` or ``None``."""
    auto = updates.local_checkout(name)
    if auto is not None:
        editable = auto["editable"]
        return {"repo": auto["repo"],
                "source": "editable" if editable else "local-install",
                "live": editable}
    configured = settings_store.source_repos().get(name)
    if configured:
        repo = Path(configured).expanduser()
        if (repo / ".git").exists():
            # A separate clone; the installed package comes from elsewhere.
            return {"repo": repo, "source": "configured", "live": False}
    return None


# -- helpers ---------------------------------------------------------------
def _branch_names(repo: Path, current: str | None) -> list[dict[str, Any]]:
    """Unique switchable branch names (local + remote-tracking, ``origin/`` and
    ``HEAD`` stripped), each flagged with where it exists and whether it's current."""
    local = _git(repo, "for-each-ref", "--format=%(refname:short)", "refs/heads") or ""
    remote = _git(repo, "for-each-ref", "--format=%(refname:short)", "refs/remotes") or ""
    locals_ = {b for b in local.splitlines() if b}
    remotes = set()
    for ref in remote.splitlines():
        # Skip blanks, the per-remote HEAD symref (".../HEAD") and its bare-name
        # short form ("origin", no slash) — neither is a switchable branch.
        if not ref or ref.endswith("/HEAD") or "/" not in ref:
            continue
        remotes.add(ref.split("/", 1)[1])  # origin/devel -> devel

    names = sorted(locals_ | remotes)
    out = []
    for n in names:
        where = "both" if n in locals_ and n in remotes else ("local" if n in locals_ else "remote")
        out.append({"name": n, "current": n == current, "where": where})
    return out


def _git_run(repo: Path, *args: str) -> dict[str, Any]:
    """Run a (mutating) git command, capturing combined output and exit code."""
    import subprocess

    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True, text=True, timeout=updates._GIT_TIMEOUT,
            env=updates._GIT_ENV, stdin=subprocess.DEVNULL,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return {"code": -1, "output": str(exc)}
    return {"code": proc.returncode, "output": (proc.stdout or "") + (proc.stderr or "")}
