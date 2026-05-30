"""Cartpole task adapter (fake-free, importable reward logic + Isaac Lab demo)."""

from __future__ import annotations

from .reward import DEFAULT_REWARD_WEIGHTS, compute_reward_terms, load_reward_weights

__all__ = ["DEFAULT_REWARD_WEIGHTS", "compute_reward_terms", "load_reward_weights"]
