"""Multi-agent reward recording: per-agent decomposition + team totals."""

from __future__ import annotations

import pytest

from rl_episode_inspector.recorder import EpisodeRecorder, RecorderError
from rl_episode_inspector.storage import AgentSpec, EpisodeStore


def _recorder(tmp_path, **kw):
    return EpisodeRecorder(
        output_dir=tmp_path, task_name="MARL", dt=0.1, episode_id_prefix="ma", **kw
    )


def _two_agent_episode(tmp_path):
    rec = _recorder(
        tmp_path,
        agents=[AgentSpec(id="cart", label="Cart"), AgentSpec(id="pendulum")],
    )
    rec.start_episode(episode_index=0)
    for f in range(3):
        rec.record_frame(
            frame_index=f, timestamp=f * 0.1, state={}, action={},
            terminated=False, truncated=(f == 2),
            rewards_by_agent={
                "cart": {"alive": 1.0, "pole_pos": 2.0},
                "pendulum": {"alive": 1.0},
            },
            reward_weights_by_agent={
                "cart": {"alive": 1.0, "pole_pos": -0.5},
                "pendulum": {"alive": 1.0},
            },
        )
    rec.end_episode(reset_reason="time_limit")
    return EpisodeStore(tmp_path).load_episode("ma_000000")


def test_per_agent_columns_and_signals(tmp_path):
    ep = _two_agent_episode(tmp_path)
    cols = ep.frames
    # Per-agent decomposition columns exist, namespaced by agent.
    assert cols["reward_cart_alive_raw"].tolist() == [1.0, 1.0, 1.0]
    assert cols["reward_cart_pole_pos_weighted"].tolist() == pytest.approx([-1.0, -1.0, -1.0])
    # Per-agent step total: cart = 1*1 + 2*-0.5 = 0.0 ; pendulum = 1.0
    assert cols["reward_cart_step_total"].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert cols["reward_pendulum_step_total"].tolist() == pytest.approx([1.0, 1.0, 1.0])
    # Team total = sum across agents (0 + 1) each frame; cumulative is its run-sum.
    assert cols["reward_step_total"].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert cols["reward_cumulative"].tolist() == pytest.approx([1.0, 2.0, 3.0])

    # Signals carry the agent tag so the UI can group them.
    by_name = {s.name: s for s in ep.metadata.signals}
    assert by_name["reward_cart_pole_pos_weighted"].agent == "cart"
    assert by_name["reward_pendulum_alive_raw"].agent == "pendulum"
    assert by_name["reward_step_total"].agent is None  # team total is shared


def test_agents_and_returns_in_metadata(tmp_path):
    ep = _two_agent_episode(tmp_path)
    m = ep.metadata
    assert [a.id for a in m.agents] == ["cart", "pendulum"]
    assert m.agents[0].label == "Cart"
    assert m.agent_returns["cart"] == pytest.approx(0.0)
    assert m.agent_returns["pendulum"] == pytest.approx(3.0)
    # Episode return is the team sum; summary carries the per-agent returns.
    assert m.episode_return == pytest.approx(3.0)
    assert m.summary().agent_returns == m.agent_returns


def test_cannot_mix_single_and_multi_agent(tmp_path):
    rec = _recorder(tmp_path)
    rec.start_episode(episode_index=0)
    rec.record_frame(
        frame_index=0, timestamp=0.0, state={}, action={},
        rewards_raw={"r": 1.0}, reward_weights={"r": 1.0}, truncated=False,
    )
    with pytest.raises(RecorderError, match="mix single-agent and per-agent"):
        rec.record_frame(
            frame_index=1, timestamp=0.1, state={}, action={}, truncated=True,
            rewards_by_agent={"a": {"r": 1.0}}, reward_weights_by_agent={"a": {"r": 1.0}},
        )


def test_single_agent_unchanged(tmp_path):
    """Single-agent episodes keep canonical names and no agent metadata."""
    rec = _recorder(tmp_path)
    rec.start_episode(episode_index=0)
    for f in range(2):
        rec.record_frame(
            frame_index=f, timestamp=f * 0.1, state={}, action={},
            rewards_raw={"alive": 1.0}, reward_weights={"alive": 1.0},
            truncated=(f == 1),
        )
    rec.end_episode(reset_reason="time_limit")
    ep = EpisodeStore(tmp_path).load_episode("ma_000000")
    assert "reward_alive_raw" in ep.frames and "reward_step_total" in ep.frames
    assert ep.metadata.agents == [] and ep.metadata.agent_returns == {}
    assert all(s.agent is None for s in ep.metadata.signals)
