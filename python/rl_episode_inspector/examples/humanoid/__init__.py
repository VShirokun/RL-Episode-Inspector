"""Humanoid task adapter: replay AMP reference mocap (walk / run / dance).

A complex (15-body) articulation demo. The motion comes from the Isaac Lab
Humanoid-AMP reference clips, so it needs no policy and no simulator — it loads
the ``.npz`` mocap and records full-body poses via the generic recorder. This is
a stress test of the tool on a many-body robot (toward the user's humanoid work).
"""

from __future__ import annotations

from .motion_replay import HUMANOID_PARENT_BY_NAME, frame_metrics, parents_for

__all__ = ["HUMANOID_PARENT_BY_NAME", "parents_for", "frame_metrics"]
