"""CI-safe tests for the Cartpole adapter's pure logic (no Isaac Lab)."""

from __future__ import annotations

import math

from rl_episode_inspector.examples.cartpole.controller import Gains, balance_action
from rl_episode_inspector.examples.cartpole.reward import (
    DEFAULT_REWARD_WEIGHTS,
    REWARD_TERM_NAMES,
    compute_reward_terms,
    load_reward_weights,
)


def test_reward_terms_at_upright_center():
    terms = compute_reward_terms(0.0, 0.0, 0.0, 0.0, 0.0, terminated=False)
    assert terms["alive"] == 1.0
    assert terms["pole_upright"] == 1.0  # cos(0)
    assert terms["cart_centering"] == 0.0
    assert terms["action_penalty"] == 0.0
    assert set(terms) == set(REWARD_TERM_NAMES)


def test_reward_penalizes_tilt_and_offset():
    terms = compute_reward_terms(0.5, 0.0, 0.3, 0.0, 0.8, terminated=False)
    assert terms["pole_upright"] == math.cos(0.3)
    assert terms["cart_centering"] == -0.25
    assert terms["action_penalty"] == -(0.8**2)


def test_alive_zero_when_terminated():
    assert compute_reward_terms(0, 0, 0, 0, 0, terminated=True)["alive"] == 0.0


def test_load_weights_defaults_and_override(tmp_path):
    assert load_reward_weights(None) == DEFAULT_REWARD_WEIGHTS
    cfg = tmp_path / "w.yaml"
    cfg.write_text("alive: 5.0\nunknown_term: 9.0\n")
    w = load_reward_weights(cfg)
    assert w["alive"] == 5.0
    assert "unknown_term" not in w  # unknown keys ignored
    assert w["pole_upright"] == DEFAULT_REWARD_WEIGHTS["pole_upright"]


def test_balance_action_is_clamped_and_signed():
    g = Gains()
    # Sign verified against Isaac Lab: a pole leaning +theta is corrected by a
    # negative normalized force (feedback is negated), clamped to [-1, 1].
    assert balance_action(0, 0, 1.0, 5.0, g) == -1.0
    assert balance_action(0, 0, -1.0, -5.0, g) == 1.0
    # upright/centered -> 0
    assert balance_action(0, 0, 0, 0, g) == 0.0
