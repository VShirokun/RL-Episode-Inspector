from __future__ import annotations

import numpy as np

from rl_episode_inspector.ranking import EpisodeRanker
from rl_episode_inspector.storage import (
    EpisodeMetadata,
    EpisodeStore,
    SignalKind,
    SignalSpec,
)


def _write_episode(store: EpisodeStore, episode_id: str, episode_return: float, n: int = 3):
    terminated = np.zeros(n, dtype=bool)
    truncated = np.zeros(n, dtype=bool)
    truncated[-1] = True
    columns = {
        "frame_index": np.arange(n, dtype=np.int32),
        "timestamp": (np.arange(n) / 60.0).astype(np.float32),
        "terminated": terminated,
        "truncated": truncated,
        "done": terminated | truncated,
        "reward_cumulative": np.linspace(0, episode_return, n).astype(np.float32),
    }
    meta = EpisodeMetadata(
        episode_id=episode_id, task_name="T", num_frames=n, dt=1 / 60, fps=60,
        duration_seconds=n / 60, truncated=True, episode_return=episode_return,
        signals=[SignalSpec(name="reward_cumulative", kind=SignalKind.reward_total)],
    )
    store.save_episode(meta, columns)


def test_best_worst_median_odd(tmp_path):
    store = EpisodeStore(tmp_path)
    for i, r in enumerate([10.0, 30.0, 20.0, 50.0, 40.0]):
        _write_episode(store, f"ep_{i:03d}", r)
    ranker = EpisodeRanker(tmp_path)
    assert ranker.get_best().episode_return == 50.0
    assert ranker.get_worst().episode_return == 10.0
    assert ranker.get_median().episode_return == 30.0  # middle of 10,20,30,40,50


def test_median_even_picks_lower_middle(tmp_path):
    store = EpisodeStore(tmp_path)
    for i, r in enumerate([10.0, 20.0, 30.0, 40.0]):
        _write_episode(store, f"ep_{i:03d}", r)
    ranker = EpisodeRanker(tmp_path)
    # ascending [10,20,30,40], lower-middle index (4-1)//2 = 1 -> 20
    assert ranker.get_median().episode_return == 20.0


def test_sorted_descending(tmp_path):
    store = EpisodeStore(tmp_path)
    for i, r in enumerate([5.0, 1.0, 9.0]):
        _write_episode(store, f"ep_{i:03d}", r)
    returns = [s.episode_return for s in EpisodeRanker(tmp_path).list_episodes()]
    assert returns == [9.0, 5.0, 1.0]


def test_equal_returns(tmp_path):
    store = EpisodeStore(tmp_path)
    for i in range(3):
        _write_episode(store, f"ep_{i:03d}", 7.0)
    ranker = EpisodeRanker(tmp_path)
    assert ranker.get_best().episode_return == 7.0
    assert ranker.get_worst().episode_return == 7.0


def test_empty_directory(tmp_path):
    ranker = EpisodeRanker(tmp_path)
    assert ranker.list_episodes() == []
    assert ranker.get_best() is None
    assert ranker.get_median() is None


def test_single_episode(tmp_path):
    store = EpisodeStore(tmp_path)
    _write_episode(store, "only", 3.0)
    ranker = EpisodeRanker(tmp_path)
    assert ranker.get_best().episode_id == "only"
    assert ranker.get_median().episode_id == "only"
    assert ranker.get_worst().episode_id == "only"


def test_corrupted_episode_skipped(tmp_path):
    store = EpisodeStore(tmp_path)
    _write_episode(store, "good", 4.0)
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    (bad_dir / "metadata.json").write_text("{ not valid json")
    ranker = EpisodeRanker(tmp_path)
    ids = [s.episode_id for s in ranker.list_episodes()]
    assert ids == ["good"]
    assert "bad" in ranker.errors
