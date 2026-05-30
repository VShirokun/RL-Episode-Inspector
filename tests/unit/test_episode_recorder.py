from __future__ import annotations

import math

import pytest

from rl_episode_inspector.recorder import EpisodeRecorder, RecorderError
from rl_episode_inspector.storage import EpisodeStore

WEIGHTS = {"alive": 1.0, "effort": 0.5}


def _record_episode(recorder, episode_index, n_frames, *, terminate_last=True):
    recorder.start_episode(episode_index=episode_index, global_step=episode_index * 1000)
    for f in range(n_frames):
        terminated = terminate_last and (f == n_frames - 1)
        truncated = (not terminate_last) and (f == n_frames - 1)
        recorder.record_frame(
            frame_index=f,
            timestamp=f / 60.0,
            state={"cart_position": 0.1 * f},
            action={"action_cart_force": -0.2 * f},
            rewards_raw={"alive": 1.0, "effort": -float(f)},
            reward_weights=WEIGHTS,
            terminated=terminated,
            truncated=truncated,
        )
    return recorder.end_episode(reset_reason="done")


def _make_recorder(tmp_path):
    return EpisodeRecorder(
        output_dir=tmp_path, task_name="Cartpole", dt=1 / 60,
        episode_id_prefix="cartpole", viewer_type="cartpole",
        state_mapping={"cart_position": "cart_position"},
    )


def test_start_record_end_flow(tmp_path):
    recorder = _make_recorder(tmp_path)
    episode_id = _record_episode(recorder, 1, 5)
    assert episode_id == "cartpole_000001"

    episode = EpisodeStore(tmp_path).load_episode(episode_id)
    assert episode.metadata.num_frames == 5
    assert episode.metadata.terminated is True
    assert episode.metadata.truncated is False


def test_weighted_and_cumulative_reward(tmp_path):
    recorder = _make_recorder(tmp_path)
    _record_episode(recorder, 1, 3)
    frames = EpisodeStore(tmp_path).load_frames("cartpole_000001")
    # effort weighted = raw(-f) * 0.5
    assert frames["reward_effort_weighted"].tolist() == [0.0, -0.5, -1.0]
    # step_total = 1.0 (alive) + effort_weighted
    assert frames["reward_step_total"].tolist() == [1.0, 0.5, 0.0]
    # cumulative = running sum
    assert frames["reward_cumulative"].tolist() == [1.0, 1.5, 1.5]


def test_episode_return_matches_cumulative(tmp_path):
    recorder = _make_recorder(tmp_path)
    _record_episode(recorder, 1, 4)
    meta = EpisodeStore(tmp_path).load_metadata("cartpole_000001")
    frames = EpisodeStore(tmp_path).load_frames("cartpole_000001")
    assert math.isclose(meta.episode_return, frames["reward_cumulative"][-1], rel_tol=1e-5)


def test_multiple_episodes_in_sequence(tmp_path):
    recorder = _make_recorder(tmp_path)
    _record_episode(recorder, 1, 3)
    _record_episode(recorder, 2, 5, terminate_last=False)
    ids = EpisodeStore(tmp_path).list_episode_ids()
    assert ids == ["cartpole_000001", "cartpole_000002"]
    assert recorder.saved_count == 2
    meta2 = EpisodeStore(tmp_path).load_metadata("cartpole_000002")
    assert meta2.truncated is True and meta2.terminated is False


def test_record_before_start_raises(tmp_path):
    recorder = _make_recorder(tmp_path)
    with pytest.raises(RecorderError, match="before start_episode"):
        recorder.record_frame(
            frame_index=0, timestamp=0.0, state={}, action={},
            rewards_raw={"alive": 1.0}, reward_weights={"alive": 1.0},
            terminated=False, truncated=True,
        )


def test_end_before_start_raises(tmp_path):
    recorder = _make_recorder(tmp_path)
    with pytest.raises(RecorderError, match="before start_episode"):
        recorder.end_episode()


def test_double_start_raises(tmp_path):
    recorder = _make_recorder(tmp_path)
    recorder.start_episode(episode_index=1)
    with pytest.raises(RecorderError, match="already active"):
        recorder.start_episode(episode_index=2)


def test_empty_episode_not_saved(tmp_path):
    recorder = _make_recorder(tmp_path)
    recorder.start_episode(episode_index=1)
    assert recorder.end_episode() is None
    assert EpisodeStore(tmp_path).list_episode_ids() == []


def test_nan_reward_rejected(tmp_path):
    recorder = _make_recorder(tmp_path)
    recorder.start_episode(episode_index=1)
    with pytest.raises(ValueError, match="Non-finite"):
        recorder.record_frame(
            frame_index=0, timestamp=0.0, state={"cart_position": 0.0},
            action={"a": 0.0}, rewards_raw={"alive": float("nan")},
            reward_weights={"alive": 1.0}, terminated=False, truncated=True,
        )


def test_inf_state_rejected(tmp_path):
    recorder = _make_recorder(tmp_path)
    recorder.start_episode(episode_index=1)
    with pytest.raises(ValueError, match="Non-finite"):
        recorder.record_frame(
            frame_index=0, timestamp=0.0, state={"cart_position": float("inf")},
            action={"a": 0.0}, rewards_raw={"alive": 1.0},
            reward_weights={"alive": 1.0}, terminated=False, truncated=True,
        )


def test_max_saved_episodes(tmp_path):
    recorder = EpisodeRecorder(
        output_dir=tmp_path, task_name="Cartpole", dt=1 / 60, max_saved_episodes=1
    )
    _record_episode(recorder, 1, 3)
    assert _record_episode(recorder, 2, 3) is None
    assert len(EpisodeStore(tmp_path).list_episode_ids()) == 1
