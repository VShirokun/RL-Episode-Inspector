"""HTTP-facing safety helpers.

The backend serves files out of a configured episodes directory. ``episode_id``
arrives from the network and is therefore untrusted; this module validates it
and maps HTTP-layer failures to the right status codes. Actual path containment
is enforced in :mod:`rl_episode_inspector.storage.paths`.
"""

from __future__ import annotations

from fastapi import HTTPException

from ..storage.paths import is_valid_episode_id


def require_valid_episode_id(episode_id: str) -> str:
    """Reject malformed / path-traversal episode IDs with HTTP 400."""
    if not is_valid_episode_id(episode_id):
        raise HTTPException(status_code=400, detail="Invalid episode_id")
    return episode_id
