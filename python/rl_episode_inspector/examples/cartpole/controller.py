"""A tiny PD balancing controller (pure, testable).

Used by the demo rollout to produce *interesting* real episodes: varying the gain
("skill") and disturbance per episode yields a spread of returns (good balancing
runs vs early falls), which is what makes best/worst/median meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Gains:
    kp: float = 12.0   # pole angle
    kd: float = 2.0    # pole angular velocity
    kx: float = 1.2    # cart position (keep it centered)
    kxd: float = 1.0   # cart velocity


def balance_action(
    cart_position: float,
    cart_velocity: float,
    pole_angle: float,
    pole_angular_velocity: float,
    gains: Gains,
) -> float:
    """Return a normalized cart force in [-1, 1] that tries to keep the pole up.

    Sign convention verified empirically against Isaac Lab's
    ``Isaac-Cartpole-Direct-v0`` (see ``--probe-sign``): a pole leaning +theta is
    corrected by a *negative* normalized force, so the feedback is negated.
    """
    u = -(
        gains.kp * pole_angle
        + gains.kd * pole_angular_velocity
        + gains.kx * cart_position
        + gains.kxd * cart_velocity
    )
    return max(-1.0, min(1.0, u))
