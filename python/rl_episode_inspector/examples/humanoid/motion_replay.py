"""Pure helpers for humanoid mocap replay (no Isaac Lab, no numpy-heavy deps).

Defines the humanoid skeleton hierarchy (so the viewer draws correct bones) and
a small per-frame metric computation used to populate the reward charts with
interpretable kinematic signals (this is reference motion, not a trained policy,
so these are derived metrics rather than task rewards).
"""

from __future__ import annotations

# Kinematic parent of each body, by name (None = root). Standard humanoid tree.
HUMANOID_PARENT_BY_NAME: dict[str, str | None] = {
    "pelvis": None,
    "torso": "pelvis",
    "head": "torso",
    "right_upper_arm": "torso",
    "right_lower_arm": "right_upper_arm",
    "right_hand": "right_lower_arm",
    "left_upper_arm": "torso",
    "left_lower_arm": "left_upper_arm",
    "left_hand": "left_lower_arm",
    "right_thigh": "pelvis",
    "right_shin": "right_thigh",
    "right_foot": "right_shin",
    "left_thigh": "pelvis",
    "left_shin": "left_thigh",
    "left_foot": "left_shin",
}

# Reward-chart weights for the derived kinematic metrics (see frame_metrics).
DEFAULT_REWARD_WEIGHTS: dict[str, float] = {
    "alive": 1.0,
    "forward_velocity": 1.0,
    "root_height": 0.5,
    "footstep": 5.0,  # sparse: spikes on each foot strike
}


def parents_for(body_names: list[str]) -> list[int]:
    """Parent index per body (-1 = root), from the humanoid hierarchy by name."""
    parents: list[int] = []
    for name in body_names:
        parent_name = HUMANOID_PARENT_BY_NAME.get(name)
        if parent_name is None or parent_name not in body_names:
            parents.append(-1)
        else:
            parents.append(body_names.index(parent_name))
    return parents


def frame_metrics(
    root_pos: tuple[float, float, float],
    prev_root_pos: tuple[float, float, float] | None,
    dt: float,
) -> dict[str, float]:
    """Derived per-frame kinematic 'reward' terms (excluding the sparse footstep,
    which needs foot-height history and is computed in the rollout loop)."""
    if prev_root_pos is None:
        forward_velocity = 0.0
    else:
        forward_velocity = (root_pos[0] - prev_root_pos[0]) / dt
    return {
        "alive": 1.0,
        "forward_velocity": float(forward_velocity),
        "root_height": float(root_pos[2]),
    }
