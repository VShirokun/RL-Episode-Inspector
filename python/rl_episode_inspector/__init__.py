"""RL Episode Inspector — record, store, rank and replay RL episodes.

The core package is intentionally task-agnostic: it deals in generic
time-series *signals*. Task-specific code (e.g. Cartpole) lives under
``rl_episode_inspector.examples`` and must never be imported by the core.
"""

from __future__ import annotations

__version__ = "0.1.0"
SCHEMA_VERSION = "0.1.0"

__all__ = ["__version__", "SCHEMA_VERSION"]
