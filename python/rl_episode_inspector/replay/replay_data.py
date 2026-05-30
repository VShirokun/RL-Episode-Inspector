"""Build viewer-ready replay data from a stored episode.

The MVP replay is a deterministic *visual reconstruction* from recorded state
(not a physics replay). This helper resolves the viewer ``state_mapping`` into
concrete per-frame series so a (Python-side) consumer can drive a viewer. The
frontend does the equivalent in TypeScript from the ``/frames`` response.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..storage import Episode


@dataclass
class ReplayData:
    viewer_type: str
    num_frames: int
    timestamps: np.ndarray
    # viewer role (e.g. "cart_position") -> per-frame values
    state: dict[str, np.ndarray]


def build_replay_data(episode: Episode) -> ReplayData:
    meta = episode.metadata
    frames = episode.frames
    state: dict[str, np.ndarray] = {}
    for role, signal_name in meta.viewer.state_mapping.items():
        if signal_name not in frames:
            raise KeyError(
                f"viewer.state_mapping role {role!r} -> {signal_name!r} not in frames"
            )
        state[role] = np.asarray(frames[signal_name])
    return ReplayData(
        viewer_type=meta.viewer.type,
        num_frames=meta.num_frames,
        timestamps=np.asarray(frames["timestamp"]),
        state=state,
    )
