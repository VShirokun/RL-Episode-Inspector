"""Generic episode storage layer (signal metadata + Parquet frames)."""

from __future__ import annotations

from .episode_store import Episode, EpisodeStore
from .schemas import (
    REQUIRED_FRAME_COLUMNS,
    SCHEMA_VERSION,
    EpisodeMetadata,
    EpisodeSummary,
)
from .signal_schema import BodySpec, LightSpec, MarkerSpec, SignalKind, SignalSpec, ViewerSpec
from .validation import EpisodeValidationError, validate_episode

__all__ = [
    "Episode",
    "EpisodeStore",
    "EpisodeMetadata",
    "EpisodeSummary",
    "SignalKind",
    "SignalSpec",
    "ViewerSpec",
    "BodySpec",
    "MarkerSpec",
    "LightSpec",
    "EpisodeValidationError",
    "validate_episode",
    "REQUIRED_FRAME_COLUMNS",
    "SCHEMA_VERSION",
]
