from __future__ import annotations

from rl_episode_inspector.fake_data import generate_fake_cartpole_episodes
from rl_episode_inspector.ranking import EpisodeRanker
from rl_episode_inspector.storage import EpisodeStore


def test_full_generate_rank_load_cycle(tmp_path):
    ids = generate_fake_cartpole_episodes(tmp_path, num_episodes=15, seed=123, max_frames=250)
    assert len(ids) == 15

    ranker = EpisodeRanker(tmp_path)
    best, worst, median = ranker.get_best(), ranker.get_worst(), ranker.get_median()
    assert best.episode_return >= median.episode_return >= worst.episode_return

    store = EpisodeStore(tmp_path)
    episode = store.load_episode(best.episode_id)
    assert episode.metadata.num_frames == len(episode.frames["frame_index"])
    # cumulative reward is monotonically consistent with episode_return
    assert abs(episode.frames["reward_cumulative"][-1] - best.episode_return) < 1e-3
