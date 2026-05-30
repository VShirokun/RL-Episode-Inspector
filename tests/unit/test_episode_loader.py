from __future__ import annotations

import pytest

from rl_episode_inspector.replay import build_replay_data
from rl_episode_inspector.storage import EpisodeStore


def test_load_frames_slicing(populated_store):
    store, _ = populated_store
    episode_id = store.list_episode_ids()[0]
    full = store.load_frames(episode_id)
    n = len(full["frame_index"])
    sliced = store.load_frames(episode_id, start=2, end=5)
    assert len(sliced["frame_index"]) == min(3, max(0, n - 2))
    assert sliced["frame_index"].tolist() == full["frame_index"][2:5].tolist()


def test_load_frames_column_subset(populated_store):
    store, _ = populated_store
    episode_id = store.list_episode_ids()[0]
    cols = store.load_frames(episode_id, names=["frame_index", "pole_angle"])
    assert set(cols.keys()) == {"frame_index", "pole_angle"}


def test_load_frames_unknown_column(populated_store):
    store, _ = populated_store
    episode_id = store.list_episode_ids()[0]
    with pytest.raises(KeyError):
        store.load_frames(episode_id, names=["frame_index", "does_not_exist"])


def test_missing_episode(tmp_path):
    store = EpisodeStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.load_metadata("nope")


def test_build_replay_data(populated_store):
    store, _ = populated_store
    episode = store.load_episode(store.list_episode_ids()[0])
    replay = build_replay_data(episode)
    assert replay.viewer_type == "cartpole"
    assert "cart_position" in replay.state
    assert len(replay.timestamps) == replay.num_frames
