"""Cartpole reward decomposition.

Pure functions (no Isaac Lab, no torch) so they can be unit-tested in standard
CI. The same six terms are used by the fake generator
(``rl_episode_inspector.fake_data``) — kept in sync by sharing these weights via
config; the formulas are intentionally simple and interpretable.

The action penalty is computed on the *normalized* action (in [-1, 1]) rather
than the raw force in Newtons, so it stays comparable in magnitude to the other
terms regardless of ``action_scale``.
"""

from __future__ import annotations

import math
from pathlib import Path

DEFAULT_REWARD_WEIGHTS: dict[str, float] = {
    "alive": 1.0,
    "pole_upright": 2.0,
    "cart_centering": 0.5,
    "cart_velocity_penalty": 0.05,
    "pole_angular_velocity_penalty": 0.05,
    "action_penalty": 0.01,
}

REWARD_TERM_NAMES = list(DEFAULT_REWARD_WEIGHTS.keys())


def compute_reward_terms(
    cart_position: float,
    cart_velocity: float,
    pole_angle: float,
    pole_angular_velocity: float,
    action_normalized: float,
    terminated: bool,
) -> dict[str, float]:
    """Raw (pre-weight) reward terms for one Cartpole frame."""
    return {
        "alive": 0.0 if terminated else 1.0,
        "pole_upright": max(-1.0, min(1.0, math.cos(pole_angle))),
        "cart_centering": -(cart_position * cart_position),
        "cart_velocity_penalty": -(cart_velocity * cart_velocity),
        "pole_angular_velocity_penalty": -(pole_angular_velocity * pole_angular_velocity),
        "action_penalty": -(action_normalized * action_normalized),
    }


def load_reward_weights(path: str | Path | None) -> dict[str, float]:
    """Load reward weights from a YAML file, falling back to defaults."""
    weights = dict(DEFAULT_REWARD_WEIGHTS)
    if path is None:
        return weights
    import yaml

    data = yaml.safe_load(Path(path).read_text()) or {}
    for key, value in data.items():
        if key in weights:
            weights[key] = float(value)
    return weights
