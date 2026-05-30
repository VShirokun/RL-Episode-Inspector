"""CI-safe tests for the Franka Reach adapter's pure logic (no Isaac Lab)."""

from __future__ import annotations

from rl_episode_inspector.examples.reach.reward import (
    DEFAULT_REWARD_WEIGHTS,
    REWARD_TERM_NAMES,
    compute_reward_terms,
)
from rl_episode_inspector.examples.reach.targets import (
    DEFAULT_WAYPOINTS,
    WaypointTracker,
    approach,
    distance,
)


def test_reward_terms_and_sparse_behavior():
    far = compute_reward_terms(distance_to_target=0.3, ee_speed=0.1, reached_now=False)
    assert far["reach_progress"] == -0.3
    assert far["action_smoothness"] == -(0.1**2)
    assert far["time_penalty"] == -1.0
    assert far["target_reached"] == 0.0  # sparse: zero off-event
    on = compute_reward_terms(0.01, 0.0, reached_now=True)
    assert on["target_reached"] == 1.0  # spike only on reach
    assert set(far) == set(REWARD_TERM_NAMES)


def test_target_reached_is_heavily_weighted():
    # Sparse bonus should dominate a single frame's reward when it fires.
    assert DEFAULT_REWARD_WEIGHTS["target_reached"] >= 10.0


def test_waypoint_tracker_advances_and_flags_once():
    wps = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    t = WaypointTracker(wps, threshold=0.05)
    assert t.reached_count == 0 and not t.done
    # not yet within threshold
    assert t.update((0.5, 0.0, 0.0)) is False
    # reach first target -> flagged exactly once
    assert t.update((0.01, 0.0, 0.0)) is True
    assert t.reached_count == 1
    assert t.current_target == (1.0, 0.0, 0.0)
    # reach second -> done
    assert t.update((1.0, 0.0, 0.0)) is True
    assert t.done is True
    # further updates do not re-flag
    assert t.update((1.0, 0.0, 0.0)) is False


def test_approach_moves_toward_target_by_gain():
    cmd = approach((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), gain=0.25)
    assert abs(cmd[0] - 0.25) < 1e-9
    # repeated application converges
    for _ in range(100):
        cmd = approach(cmd, (1.0, 0.0, 0.0), gain=0.25)
    assert distance(cmd, (1.0, 0.0, 0.0)) < 1e-3


def test_default_waypoints_in_workspace():
    for x, y, z in DEFAULT_WAYPOINTS:
        assert 0.35 <= x <= 0.65
        assert -0.2 <= y <= 0.2
        assert 0.15 <= z <= 0.5
