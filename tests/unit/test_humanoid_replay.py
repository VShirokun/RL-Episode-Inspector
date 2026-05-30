"""CI-safe tests for the humanoid mocap-replay helpers (no Isaac Lab)."""

from __future__ import annotations

from rl_episode_inspector.examples.humanoid.motion_replay import frame_metrics, parents_for

BODY_NAMES = [
    "pelvis", "torso", "head",
    "right_upper_arm", "right_lower_arm", "right_hand",
    "left_upper_arm", "left_lower_arm", "left_hand",
    "right_thigh", "right_shin", "right_foot",
    "left_thigh", "left_shin", "left_foot",
]


def test_parents_form_correct_skeleton():
    parents = parents_for(BODY_NAMES)
    idx = {n: i for i, n in enumerate(BODY_NAMES)}
    assert parents[idx["pelvis"]] == -1  # root
    assert parents[idx["torso"]] == idx["pelvis"]
    assert parents[idx["head"]] == idx["torso"]
    assert parents[idx["right_lower_arm"]] == idx["right_upper_arm"]
    assert parents[idx["left_foot"]] == idx["left_shin"]
    assert parents[idx["right_thigh"]] == idx["pelvis"]
    assert len(parents) == len(BODY_NAMES)


def test_parents_unknown_body_is_root():
    assert parents_for(["mystery_link"]) == [-1]


def test_frame_metrics_forward_velocity():
    m0 = frame_metrics((0.0, 0.0, 0.9), None, dt=1 / 60)
    assert m0["forward_velocity"] == 0.0  # no previous frame
    assert m0["root_height"] == 0.9
    assert m0["alive"] == 1.0
    m1 = frame_metrics((0.1, 0.0, 0.9), (0.0, 0.0, 0.9), dt=0.1)
    assert abs(m1["forward_velocity"] - 1.0) < 1e-9  # 0.1 m / 0.1 s
