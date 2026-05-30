"""Waypoint sequencing for the Reach task (pure, testable).

All positions are in the Franka base frame and lie inside the env's reachable EE
workspace (x∈[0.35,0.65], y∈[-0.2,0.2], z∈[0.15,0.5]).
"""

from __future__ import annotations

import math

# A fixed sequence of reachable EE targets to visit, in order.
DEFAULT_WAYPOINTS: list[tuple[float, float, float]] = [
    (0.50, 0.00, 0.45),
    (0.60, 0.18, 0.25),
    (0.40, -0.18, 0.40),
    (0.62, -0.10, 0.18),
    (0.45, 0.18, 0.48),
    (0.55, -0.18, 0.30),
    (0.38, 0.12, 0.20),
    (0.60, 0.00, 0.45),
]

REACH_THRESHOLD = 0.04  # meters; EE within this of a target counts as "reached"


def distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b, strict=True)))


def approach(
    cmd: tuple[float, float, float],
    target: tuple[float, float, float],
    gain: float,
) -> tuple[float, float, float]:
    """Low-pass the commanded EE pose toward ``target`` (gain in (0,1])."""
    return tuple(  # type: ignore[return-value]
        c + gain * (t - c) for c, t in zip(cmd, target, strict=True)
    )


class WaypointTracker:
    """Tracks progress through a waypoint sequence and flags reach events."""

    def __init__(
        self,
        waypoints: list[tuple[float, float, float]] | None = None,
        threshold: float = REACH_THRESHOLD,
    ):
        self.waypoints = list(waypoints if waypoints is not None else DEFAULT_WAYPOINTS)
        self.threshold = threshold
        self.index = 0

    @property
    def done(self) -> bool:
        return self.index >= len(self.waypoints)

    @property
    def reached_count(self) -> int:
        return self.index

    @property
    def current_target(self) -> tuple[float, float, float]:
        # Once done, keep returning the last target (so a controller can hold).
        i = min(self.index, len(self.waypoints) - 1)
        return self.waypoints[i]

    def update(self, ee_pos: tuple[float, float, float]) -> bool:
        """Advance if the EE is within threshold of the current target.

        Returns True exactly on the frame a target is reached (the sparse event).
        """
        if self.done:
            return False
        if distance(ee_pos, self.waypoints[self.index]) <= self.threshold:
            self.index += 1
            return True
        return False
