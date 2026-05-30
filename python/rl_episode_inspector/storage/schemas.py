"""Episode-level schemas: metadata, summary and required frame columns."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from .signal_schema import SignalSpec, ViewerSpec

SCHEMA_VERSION = "0.1.0"

# Columns every episode must contain regardless of task, mapped to their
# logical dtype. ``validate_episode`` enforces these.
REQUIRED_FRAME_COLUMNS: dict[str, str] = {
    "frame_index": "int32",
    "timestamp": "float32",
    "terminated": "bool",
    "truncated": "bool",
    "done": "bool",
}


def utcnow_iso() -> str:
    """ISO-8601 UTC timestamp with a trailing ``Z`` (used for ``created_at``)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class EpisodeMetadata(BaseModel):
    """Everything needed to interpret an episode without reading the frames."""

    schema_version: str = SCHEMA_VERSION
    episode_id: str
    run_id: str | None = None
    task_name: str
    task_source: str = "unknown"
    env_id: int = 0
    episode_index: int = 0
    created_at: str = Field(default_factory=utcnow_iso)

    num_frames: int
    dt: float
    fps: float
    duration_seconds: float

    global_step_start: int = 0
    global_step_end: int = 0

    terminated: bool = False
    truncated: bool = False
    reset_reason: str | None = None

    episode_return: float

    policy_checkpoint: str | None = None
    seed: int | None = None

    signals: list[SignalSpec] = Field(default_factory=list)
    viewer: ViewerSpec = Field(default_factory=ViewerSpec)

    @property
    def done(self) -> bool:
        return bool(self.terminated or self.truncated)

    def summary(self) -> EpisodeSummary:
        return EpisodeSummary(
            episode_id=self.episode_id,
            task_name=self.task_name,
            episode_return=self.episode_return,
            num_frames=self.num_frames,
            duration_seconds=self.duration_seconds,
            created_at=self.created_at,
            terminated=self.terminated,
            truncated=self.truncated,
            reset_reason=self.reset_reason,
        )


class EpisodeSummary(BaseModel):
    """Lightweight per-episode info used by the selector and ranking endpoints."""

    episode_id: str
    task_name: str
    episode_return: float
    num_frames: int
    duration_seconds: float
    created_at: str
    terminated: bool
    truncated: bool
    reset_reason: str | None = None
