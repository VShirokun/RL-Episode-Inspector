"""Reward decomposition for the Franka Reach "visit targets" task.

Pure functions (no Isaac Lab / torch) so they're unit-tested in standard CI.

Terms:
- ``reach_progress``  dense shaping: -distance to the current target (closer -> higher)
- ``action_smoothness`` dense penalty: -(end-effector speed)^2 (discourage jerky motion)
- ``time_penalty``    dense constant: -1 per step (encourage finishing quickly)
- ``target_reached``  **sparse**: +1 only on the frame a target is reached, else 0

The sparse term is the point of this demo: its weighted series is flat-zero with
occasional spikes, and ``reward_cumulative`` becomes a staircase — a good test for
chart rendering and seeking to rare events.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_REWARD_WEIGHTS: dict[str, float] = {
    "reach_progress": 1.0,
    "action_smoothness": 0.05,
    "time_penalty": 0.1,
    "target_reached": 10.0,
}

REWARD_TERM_NAMES = list(DEFAULT_REWARD_WEIGHTS.keys())


def compute_reward_terms(
    distance_to_target: float,
    ee_speed: float,
    reached_now: bool,
) -> dict[str, float]:
    """Raw (pre-weight) reward terms for one Reach frame."""
    return {
        "reach_progress": -float(distance_to_target),
        "action_smoothness": -float(ee_speed) * float(ee_speed),
        "time_penalty": -1.0,
        "target_reached": 1.0 if reached_now else 0.0,
    }


def load_reward_weights(path: str | Path | None) -> dict[str, float]:
    weights = dict(DEFAULT_REWARD_WEIGHTS)
    if path is None:
        return weights
    import yaml

    data = yaml.safe_load(Path(path).read_text()) or {}
    for key, value in data.items():
        if key in weights:
            weights[key] = float(value)
    return weights
