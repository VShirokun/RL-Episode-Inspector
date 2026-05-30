from __future__ import annotations

import numpy as np

from rl_episode_inspector.fake_data import generate_fake_cartpole_episodes
from rl_episode_inspector.storage import EpisodeStore
from rl_episode_inspector.storage.validation import validate_episode

REQUIRED_SIGNALS = {
    "cart_position", "cart_velocity", "pole_angle", "pole_angular_velocity",
    "action_cart_force", "reward_step_total", "reward_cumulative",
    "reward_alive_raw", "reward_alive_weighted",
}


def test_generates_requested_count(tmp_path):
    ids = generate_fake_cartpole_episodes(tmp_path, num_episodes=10, seed=1, max_frames=100)
    assert len(ids) == 10
    assert EpisodeStore(tmp_path).list_episode_ids() == sorted(ids)


def test_generated_episodes_valid(tmp_path):
    generate_fake_cartpole_episodes(tmp_path, num_episodes=5, seed=2, max_frames=80)
    store = EpisodeStore(tmp_path)
    for episode_id in store.list_episode_ids():
        ep = store.load_episode(episode_id)  # load_episode validates
        validate_episode(ep.metadata, ep.frames)
        assert REQUIRED_SIGNALS.issubset(set(ep.frames.keys()))


def test_episodes_have_varied_returns(tmp_path):
    generate_fake_cartpole_episodes(tmp_path, num_episodes=12, seed=3, max_frames=200)
    store = EpisodeStore(tmp_path)
    returns = [store.load_metadata(e).episode_return for e in store.list_episode_ids()]
    assert len({round(r, 3) for r in returns}) > 1
    assert np.std(returns) > 0


def test_metadata_matches_frames(tmp_path):
    generate_fake_cartpole_episodes(tmp_path, num_episodes=3, seed=4, max_frames=150)
    store = EpisodeStore(tmp_path)
    for episode_id in store.list_episode_ids():
        meta = store.load_metadata(episode_id)
        frames = store.load_frames(episode_id)
        assert meta.num_frames == len(frames["frame_index"])
        assert meta.viewer.type == "cartpole"
        # terminated episodes should be flagged as pole_fell
        if meta.terminated:
            assert meta.reset_reason == "pole_fell"
        elif meta.truncated:
            assert meta.reset_reason == "max_episode_length"


def test_deterministic_given_seed(tmp_path):
    a = generate_fake_cartpole_episodes(tmp_path / "a", num_episodes=4, seed=99, max_frames=100)
    b = generate_fake_cartpole_episodes(tmp_path / "b", num_episodes=4, seed=99, max_frames=100)
    sa = EpisodeStore(tmp_path / "a")
    sb = EpisodeStore(tmp_path / "b")
    assert a == b
    ra = [sa.load_metadata(e).episode_return for e in a]
    rb = [sb.load_metadata(e).episode_return for e in b]
    assert ra == rb
