"""Franka Reach task adapter: visit a sequence of 3D targets (sparse bonus).

A second, richer demo task than Cartpole — longer episodes and a genuinely
*sparse* reward term (``target_reached``: zero on almost every frame, a spike
only when the end-effector enters a target zone). Like the Cartpole adapter, it
lives entirely outside the core; the core only sees generic signals.
"""

from __future__ import annotations

from .reward import DEFAULT_REWARD_WEIGHTS, REWARD_TERM_NAMES, compute_reward_terms

__all__ = ["DEFAULT_REWARD_WEIGHTS", "REWARD_TERM_NAMES", "compute_reward_terms"]
