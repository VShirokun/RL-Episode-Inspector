"""Tracking arbitrary variables via the generic observation / debug channels."""

from __future__ import annotations

import pytest

from rl_episode_inspector.recorder import EpisodeRecorder, RecorderError
from rl_episode_inspector.storage import EpisodeStore


def _recorder(tmp_path, **kw):
    return EpisodeRecorder(
        output_dir=tmp_path, task_name="Track", dt=0.1, episode_id_prefix="tr", **kw
    )


def test_tracks_observation_and_debug_variables(tmp_path):
    rec = _recorder(
        tmp_path,
        signal_units={"ball_spin": "rev/s"},
        signal_descriptions={"ball_spin": "Ball spin rate"},
    )
    rec.start_episode(episode_index=0)
    for f in range(3):
        rec.record_frame(
            frame_index=f, timestamp=f * 0.1, state={}, action={},
            rewards_raw={"r": 1.0}, reward_weights={"r": 1.0}, truncated=(f == 2),
            observations={"ball_spin": float(f), "ball_speed": 2.0 * f},
            debug={"solver_iters": float(f)},
        )
    rec.end_episode(reset_reason="time_limit")

    ep = EpisodeStore(tmp_path).load_episode("tr_000000")
    # Columns exist for the tracked variables.
    assert ep.frames["ball_spin"].tolist() == [0.0, 1.0, 2.0]
    assert ep.frames["ball_speed"].tolist() == [0.0, 2.0, 4.0]
    assert ep.frames["solver_iters"].tolist() == [0.0, 1.0, 2.0]
    # Declared with the right kinds + carried annotations.
    by_name = {s.name: s for s in ep.metadata.signals}
    assert by_name["ball_spin"].kind == "observation"
    assert by_name["ball_speed"].kind == "observation"
    assert by_name["solver_iters"].kind == "debug"
    assert by_name["ball_spin"].unit == "rev/s"
    assert by_name["ball_spin"].description == "Ball spin rate"


def test_tracked_channels_are_optional(tmp_path):
    """Not passing observations/debug leaves them out entirely (no columns)."""
    rec = _recorder(tmp_path)
    rec.start_episode(episode_index=0)
    rec.record_frame(
        frame_index=0, timestamp=0.0, state={"x": 1.0}, action={},
        rewards_raw={"r": 1.0}, reward_weights={"r": 1.0}, truncated=True,
    )
    rec.end_episode()
    ep = EpisodeStore(tmp_path).load_episode("tr_000000")
    kinds = {s.kind for s in ep.metadata.signals}
    assert "observation" not in kinds and "debug" not in kinds


def test_tracked_key_set_must_stay_stable(tmp_path):
    rec = _recorder(tmp_path)
    rec.start_episode(episode_index=0)
    rec.record_frame(
        frame_index=0, timestamp=0.0, state={}, action={},
        rewards_raw={"r": 1.0}, reward_weights={"r": 1.0},
        observations={"ball_spin": 1.0},
    )
    with pytest.raises(RecorderError, match="observation keys changed"):
        rec.record_frame(
            frame_index=1, timestamp=0.1, state={}, action={}, truncated=True,
            rewards_raw={"r": 1.0}, reward_weights={"r": 1.0},
            observations={"ball_spin": 1.0, "extra": 2.0},
        )
