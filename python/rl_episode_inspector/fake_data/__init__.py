"""Fake (no-Isaac-Lab) Cartpole-like episode generation for demos and tests."""

from __future__ import annotations

from .generate_fake_cartpole import (
    DEFAULT_REWARD_WEIGHTS,
    generate_fake_cartpole_episodes,
)

__all__ = ["DEFAULT_REWARD_WEIGHTS", "generate_fake_cartpole_episodes"]
