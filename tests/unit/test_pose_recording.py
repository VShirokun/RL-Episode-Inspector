"""Tests for full-body pose recording (the articulation feature)."""

from __future__ import annotations

import pytest

from rl_episode_inspector.recorder import EpisodeRecorder, RecorderError, pose_columns
from rl_episode_inspector.storage import EpisodeStore


def _recorder(tmp_path, **kw):
    return EpisodeRecorder(
        output_dir=tmp_path, task_name="Arm", dt=1 / 30,
        episode_id_prefix="arm", viewer_type="articulation3d", **kw,
    )


def _frame(rec, frame, poses):
    rec.record_frame(
        frame_index=frame, timestamp=frame / 30.0, state={}, action={},
        rewards_raw={"alive": 1.0}, reward_weights={"alive": 1.0},
        terminated=False, truncated=(frame == 2), poses=poses,
    )


def test_pose_columns_naming():
    assert pose_columns("link0") == [
        "pose_link0_px", "pose_link0_py", "pose_link0_pz",
        "pose_link0_qw", "pose_link0_qx", "pose_link0_qy", "pose_link0_qz",
    ]


def test_records_body_poses_and_structure(tmp_path):
    rec = _recorder(tmp_path)
    rec.register_bodies(["base", "link1"], parents=[-1, 0])
    rec.start_episode(episode_index=0)
    for f in range(3):
        _frame(rec, f, {
            "base": (0, 0, 0, 1, 0, 0, 0),
            "link1": (0.1 * f, 0, 0.5, 1, 0, 0, 0),
        })
    rec.end_episode(reset_reason="time_limit")

    ep = EpisodeStore(tmp_path).load_episode("arm_000000")
    # columns exist for both bodies (float32, so compare approximately)
    assert ep.frames["pose_link1_px"].tolist() == pytest.approx([0.0, 0.1, 0.2], abs=1e-6)
    assert ep.frames["pose_base_qw"].tolist() == [1.0, 1.0, 1.0]
    # structure carried into the viewer spec
    bodies = ep.metadata.viewer.bodies
    assert [b.name for b in bodies] == ["base", "link1"]
    assert bodies[1].parent == 0
    assert bodies[1].pos == ["pose_link1_px", "pose_link1_py", "pose_link1_pz"]
    # pose signals declared with kind "pose"
    pose_signals = [s for s in ep.metadata.signals if s.kind == "pose"]
    assert len(pose_signals) == 14  # 2 bodies * 7


def test_validation_passes_with_poses(tmp_path):
    rec = _recorder(tmp_path)
    rec.register_bodies(["b"], parents=[-1])
    rec.start_episode(episode_index=0)
    for f in range(3):
        _frame(rec, f, {"b": (f, 0, 0, 1, 0, 0, 0)})
    rec.end_episode(reset_reason="time_limit")
    # load_episode validates; should not raise
    EpisodeStore(tmp_path).load_episode("arm_000000")


def test_missing_poses_when_bodies_registered_raises(tmp_path):
    rec = _recorder(tmp_path)
    rec.register_bodies(["b"], parents=[-1])
    rec.start_episode(episode_index=0)
    with pytest.raises(RecorderError, match="no poses"):
        _frame(rec, 0, None)


def test_wrong_pose_length_raises(tmp_path):
    rec = _recorder(tmp_path)
    rec.register_bodies(["b"], parents=[-1])
    rec.start_episode(episode_index=0)
    with pytest.raises(RecorderError, match="7 numbers"):
        _frame(rec, 0, {"b": (0, 0, 0)})


def test_markers_and_up_axis_in_viewer_spec(tmp_path):
    from rl_episode_inspector.storage import MarkerSpec

    rec = _recorder(
        tmp_path,
        markers=[MarkerSpec(name="target", pos=["tx", "ty", "tz"], color="#0f0")],
        up_axis="z",
    )
    rec.register_bodies(["b"], parents=[-1])
    rec.start_episode(episode_index=0)
    rec.record_frame(
        frame_index=0, timestamp=0.0, state={"tx": 0.5, "ty": 0.0, "tz": 0.4},
        action={}, rewards_raw={"alive": 1.0}, reward_weights={"alive": 1.0},
        terminated=False, truncated=True, poses={"b": (0, 0, 0, 1, 0, 0, 0)},
    )
    rec.end_episode()
    v = EpisodeStore(tmp_path).load_metadata("arm_000000").viewer
    assert v.up_axis == "z"
    assert v.markers[0].name == "target"
    assert v.markers[0].pos == ["tx", "ty", "tz"]
