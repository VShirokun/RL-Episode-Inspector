"""On-disk episode store: one directory per episode.

Layout::

    <root>/
      <episode_id>/
        metadata.json
        frames.parquet
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .parquet_reader import read_frames, read_num_rows
from .parquet_writer import write_frames
from .paths import safe_episode_dir
from .schemas import EpisodeMetadata
from .validation import EpisodeValidationError, validate_episode

METADATA_FILE = "metadata.json"
FRAMES_FILE = "frames.parquet"


@dataclass
class Episode:
    """An episode loaded into memory: metadata + frame columns."""

    metadata: EpisodeMetadata
    frames: dict[str, np.ndarray]


class EpisodeStore:
    """Reads and writes episodes under a single root directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    # -- discovery ---------------------------------------------------------
    def list_episode_ids(self) -> list[str]:
        if not self.root.exists():
            return []
        ids = [
            p.name
            for p in sorted(self.root.iterdir())
            if p.is_dir() and (p / METADATA_FILE).exists()
        ]
        return ids

    def exists(self, episode_id: str) -> bool:
        try:
            return (safe_episode_dir(self.root, episode_id) / METADATA_FILE).exists()
        except ValueError:
            return False

    def episode_dir(self, episode_id: str) -> Path:
        return safe_episode_dir(self.root, episode_id)

    # -- reading -----------------------------------------------------------
    def load_metadata(self, episode_id: str) -> EpisodeMetadata:
        path = self.episode_dir(episode_id) / METADATA_FILE
        if not path.exists():
            raise FileNotFoundError(f"No metadata for episode {episode_id!r} at {path}")
        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise EpisodeValidationError(
                f"Episode {episode_id!r}: metadata.json is not valid JSON: {exc}"
            ) from exc
        return EpisodeMetadata.model_validate(raw)

    def frames_path(self, episode_id: str) -> Path:
        return self.episode_dir(episode_id) / FRAMES_FILE

    def num_frames(self, episode_id: str) -> int:
        return read_num_rows(self.frames_path(episode_id))

    def load_frames(
        self,
        episode_id: str,
        start: int | None = None,
        end: int | None = None,
        names: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        return read_frames(self.frames_path(episode_id), start=start, end=end, names=names)

    def load_episode(self, episode_id: str, *, validate: bool = True) -> Episode:
        metadata = self.load_metadata(episode_id)
        frames = self.load_frames(episode_id)
        if validate:
            validate_episode(metadata, frames)
        return Episode(metadata=metadata, frames=frames)

    # -- writing -----------------------------------------------------------
    def save_episode(
        self,
        metadata: EpisodeMetadata,
        columns: dict[str, np.ndarray],
        *,
        validate: bool = True,
    ) -> Path:
        if validate:
            validate_episode(metadata, columns)
        ep_dir = self.episode_dir(metadata.episode_id)
        ep_dir.mkdir(parents=True, exist_ok=True)
        write_frames(ep_dir / FRAMES_FILE, columns)
        (ep_dir / METADATA_FILE).write_text(
            metadata.model_dump_json(indent=2, exclude_none=False)
        )
        return ep_dir
