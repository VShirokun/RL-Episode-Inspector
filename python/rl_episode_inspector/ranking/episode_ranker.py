"""Rank episodes by ``episode_return``.

For the MVP ranking is over all completed episodes found in the directory
(which, per the vectorized-env policy, come from one selected ``env_id``).
The design intentionally leaves room for richer ranking later (by reward
component, length, failure reason) — those would be additional methods that
read the same summary table.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..storage import EpisodeStore, EpisodeSummary
from ..storage.validation import EpisodeValidationError

logger = logging.getLogger(__name__)


class EpisodeRanker:
    """Builds a summary table from an episodes directory and ranks it."""

    def __init__(self, episodes_dir: str | Path, *, strict: bool = False):
        self.store = EpisodeStore(episodes_dir)
        self.strict = strict
        self._errors: dict[str, str] = {}

    @property
    def errors(self) -> dict[str, str]:
        """``{episode_id: error message}`` for episodes that failed to load."""
        return dict(self._errors)

    def list_episodes(self) -> list[EpisodeSummary]:
        """All loadable episodes, sorted by ``episode_return`` descending."""
        summaries = self._summaries()
        summaries.sort(key=lambda s: s.episode_return, reverse=True)
        return summaries

    def _summaries(self) -> list[EpisodeSummary]:
        self._errors = {}
        summaries: list[EpisodeSummary] = []
        for episode_id in self.store.list_episode_ids():
            try:
                meta = self.store.load_metadata(episode_id)
                summaries.append(meta.summary())
            except (EpisodeValidationError, ValueError, OSError) as exc:
                self._errors[episode_id] = str(exc)
                if self.strict:
                    raise
                logger.warning("Skipping unreadable episode %s: %s", episode_id, exc)
        return summaries

    def get_best(self) -> EpisodeSummary | None:
        summaries = self.list_episodes()
        return summaries[0] if summaries else None

    def get_worst(self) -> EpisodeSummary | None:
        summaries = self.list_episodes()
        return summaries[-1] if summaries else None

    def get_median(self) -> EpisodeSummary | None:
        """Median episode by return.

        Sorted ascending; for an even count the lower-middle element is chosen
        (index ``(n-1)//2``) so the result is always a real, selectable episode
        rather than an interpolated value.
        """
        summaries = self.list_episodes()
        if not summaries:
            return None
        ascending = sorted(summaries, key=lambda s: s.episode_return)
        return ascending[(len(ascending) - 1) // 2]

    def get(self, mode: str) -> EpisodeSummary | None:
        modes = {"best": self.get_best, "worst": self.get_worst, "median": self.get_median}
        if mode not in modes:
            raise ValueError(f"Unknown ranking mode {mode!r}; expected one of {sorted(modes)}")
        return modes[mode]()
