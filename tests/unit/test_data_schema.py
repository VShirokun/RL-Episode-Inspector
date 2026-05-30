from __future__ import annotations

import json

from rl_episode_inspector.storage import EpisodeMetadata, EpisodeStore


def test_metadata_roundtrip_json(minimal_episode):
    metadata, _ = minimal_episode
    raw = metadata.model_dump_json()
    restored = EpisodeMetadata.model_validate(json.loads(raw))
    assert restored == metadata
    assert restored.done is True  # terminated -> done


def test_save_and_reload_episode(tmp_path, minimal_episode):
    metadata, columns = minimal_episode
    store = EpisodeStore(tmp_path)
    ep_dir = store.save_episode(metadata, columns)
    assert (ep_dir / "metadata.json").exists()
    assert (ep_dir / "frames.parquet").exists()

    episode = store.load_episode("cartpole_000001")
    assert episode.metadata.episode_return == 4.0
    assert episode.frames["frame_index"].tolist() == [0, 1, 2, 3]
    assert episode.frames["cart_position"].dtype.name == "float32"


def test_done_property():
    base = {
        "episode_id": "e", "task_name": "T", "num_frames": 1, "dt": 0.1, "fps": 10,
        "duration_seconds": 0.1, "episode_return": 0.0, "signals": [],
    }
    assert EpisodeMetadata(**base, terminated=True).done is True
    assert EpisodeMetadata(**base, truncated=True).done is True
    assert EpisodeMetadata(**base).done is False
