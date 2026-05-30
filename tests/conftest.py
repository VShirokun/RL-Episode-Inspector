"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from rl_episode_inspector.fake_data import generate_fake_cartpole_episodes
from rl_episode_inspector.storage import (
    EpisodeMetadata,
    EpisodeStore,
    SignalKind,
    SignalSpec,
    ViewerSpec,
)


@pytest.fixture
def minimal_episode():
    """A tiny hand-built (metadata, columns) pair that passes validation."""
    n = 4
    terminated = np.array([False, False, False, True])
    truncated = np.zeros(n, dtype=bool)
    columns = {
        "frame_index": np.arange(n, dtype=np.int32),
        "timestamp": (np.arange(n) / 60.0).astype(np.float32),
        "terminated": terminated,
        "truncated": truncated,
        "done": terminated | truncated,
        "cart_position": np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32),
        "reward_step_total": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        "reward_cumulative": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
    }
    metadata = EpisodeMetadata(
        episode_id="cartpole_000001",
        task_name="Cartpole",
        task_source="test",
        num_frames=n,
        dt=1 / 60,
        fps=60,
        duration_seconds=n / 60,
        terminated=True,
        truncated=False,
        reset_reason="pole_fell",
        episode_return=4.0,
        signals=[
            SignalSpec(name="cart_position", kind=SignalKind.state, unit="m"),
            SignalSpec(name="reward_step_total", kind=SignalKind.reward_total),
            SignalSpec(name="reward_cumulative", kind=SignalKind.reward_total),
        ],
        viewer=ViewerSpec(type="cartpole", state_mapping={"cart_position": "cart_position"}),
    )
    return metadata, columns


@pytest.fixture
def populated_store(tmp_path):
    """An EpisodeStore backed by a handful of generated fake episodes."""
    episodes_dir = tmp_path / "episodes"
    generate_fake_cartpole_episodes(episodes_dir, num_episodes=8, seed=7, max_frames=120)
    return EpisodeStore(episodes_dir), episodes_dir
