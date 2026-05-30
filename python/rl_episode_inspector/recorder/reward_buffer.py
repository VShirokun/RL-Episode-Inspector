"""Reward bookkeeping for a single episode.

Keeps raw and weighted values for every named reward term, plus the per-frame
total and the running cumulative sum. The naming convention is fixed here so
the rest of the system can rely on it:

    reward_<name>_raw        raw term before weighting
    reward_<name>_weighted   raw * weight
    reward_step_total        sum of all weighted terms this frame
    reward_cumulative        running sum of reward_step_total
"""

from __future__ import annotations

import math


class RewardBuffer:
    """Accumulates reward decomposition across the frames of one episode."""

    def __init__(self) -> None:
        self._cumulative: float = 0.0
        # term name -> list of raw / weighted values, one per frame
        self.raw: dict[str, list[float]] = {}
        self.weighted: dict[str, list[float]] = {}
        self.step_total: list[float] = []
        self.cumulative: list[float] = []
        self._term_names: list[str] | None = None

    @property
    def cumulative_return(self) -> float:
        return self._cumulative

    def add_frame(
        self,
        rewards_raw: dict[str, float],
        reward_weights: dict[str, float],
    ) -> float:
        """Record one frame's reward terms and return its ``reward_step_total``."""
        names = list(rewards_raw.keys())
        if self._term_names is None:
            self._term_names = names
            for n in names:
                self.raw[n] = []
                self.weighted[n] = []
        elif names != self._term_names:
            raise ValueError(
                f"Reward term set changed mid-episode: expected {self._term_names}, "
                f"got {names}"
            )

        step_total = 0.0
        for name, raw_value in rewards_raw.items():
            weight = float(reward_weights.get(name, 1.0))
            raw_value = float(raw_value)
            if not math.isfinite(raw_value):
                raise ValueError(f"Non-finite raw reward for term {name!r}: {raw_value}")
            weighted = raw_value * weight
            self.raw[name].append(raw_value)
            self.weighted[name].append(weighted)
            step_total += weighted

        self._cumulative += step_total
        self.step_total.append(step_total)
        self.cumulative.append(self._cumulative)
        return step_total

    def column_dict(self) -> dict[str, list[float]]:
        """Return all reward columns under their canonical names."""
        out: dict[str, list[float]] = {}
        for name in self.raw:
            out[f"reward_{name}_raw"] = self.raw[name]
            out[f"reward_{name}_weighted"] = self.weighted[name]
        out["reward_step_total"] = self.step_total
        out["reward_cumulative"] = self.cumulative
        return out

    @property
    def term_names(self) -> list[str]:
        return list(self._term_names or [])
