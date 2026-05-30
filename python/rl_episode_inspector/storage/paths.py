"""Safe resolution of episode directories.

Episode IDs come from the filesystem *and* from HTTP clients, so they are
treated as untrusted. This module is the single chokepoint that turns an
``episode_id`` into a path, rejecting anything that would escape the configured
episodes root (``../../etc/passwd`` and friends).
"""

from __future__ import annotations

import re
from pathlib import Path

# Episode IDs are restricted to a safe, filesystem-friendly charset. This alone
# rejects path traversal (no '/', no '..'), but we still re-check containment
# after resolving, in case of symlinks.
_EPISODE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class UnsafeEpisodeIdError(ValueError):
    """Raised when an episode_id is malformed or escapes the episodes root."""


def is_valid_episode_id(episode_id: str) -> bool:
    return bool(_EPISODE_ID_RE.match(episode_id)) and ".." not in episode_id


def safe_subpath(root: str | Path, relpath: str) -> Path:
    """Resolve ``root/relpath`` (which MAY contain subdirs) safely.

    Allows nested paths like ``franka/panda_link0.glb`` but rejects absolute
    paths, ``..`` traversal, and anything that resolves outside ``root``. Used to
    serve mesh assets.
    """
    if not relpath or relpath.startswith("/") or ".." in relpath.split("/"):
        raise UnsafeEpisodeIdError(f"Unsafe asset path: {relpath!r}")
    root_resolved = Path(root).resolve()
    candidate = (root_resolved / relpath).resolve()
    if root_resolved != candidate and root_resolved not in candidate.parents:
        raise UnsafeEpisodeIdError(f"Asset path escapes the assets directory: {relpath!r}")
    return candidate


def safe_episode_dir(root: str | Path, episode_id: str) -> Path:
    """Return ``root/episode_id`` only if it is well-formed and stays inside root."""
    if not is_valid_episode_id(episode_id):
        raise UnsafeEpisodeIdError(f"Invalid episode_id: {episode_id!r}")

    root_resolved = Path(root).resolve()
    candidate = (root_resolved / episode_id).resolve()
    if root_resolved != candidate and root_resolved not in candidate.parents:
        raise UnsafeEpisodeIdError(
            f"Resolved episode path escapes the episodes directory: {episode_id!r}"
        )
    return candidate
