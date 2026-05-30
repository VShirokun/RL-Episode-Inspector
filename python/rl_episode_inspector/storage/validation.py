"""Validation of episode metadata + frame data.

Errors are intended to be human-readable so a user staring at a corrupted
episode directory can tell what is wrong without reading source code.
"""

from __future__ import annotations

import numpy as np

from .schemas import REQUIRED_FRAME_COLUMNS, SCHEMA_VERSION, EpisodeMetadata


class EpisodeValidationError(ValueError):
    """Raised when an episode's metadata/frames are inconsistent or corrupted."""


def _major_minor(version: str) -> tuple[int, int]:
    parts = version.split(".")
    return int(parts[0]), int(parts[1] if len(parts) > 1 else 0)


def check_schema_version(version: str) -> None:
    """Compatible if the major version matches the library's schema version."""
    try:
        got_major, _ = _major_minor(version)
        want_major, _ = _major_minor(SCHEMA_VERSION)
    except (ValueError, IndexError) as exc:
        raise EpisodeValidationError(f"Unparseable schema_version {version!r}") from exc
    if got_major != want_major:
        raise EpisodeValidationError(
            f"Incompatible schema_version {version!r}; this build supports "
            f"{SCHEMA_VERSION!r} (major version must match)"
        )


def validate_episode(
    metadata: EpisodeMetadata,
    columns: dict[str, np.ndarray],
    *,
    check_finite: bool = True,
) -> None:
    """Validate that ``metadata`` and ``columns`` agree and are well-formed.

    Checks: schema version, required columns present, every declared signal has
    a backing column, frame count matches ``num_frames``, ``done == terminated
    or truncated``, and (optionally) no NaN/Inf in numeric columns.
    """
    check_schema_version(metadata.schema_version)

    missing_required = [c for c in REQUIRED_FRAME_COLUMNS if c not in columns]
    if missing_required:
        raise EpisodeValidationError(
            f"Episode {metadata.episode_id!r} is missing required frame columns: "
            f"{missing_required}"
        )

    signal_names = {s.name for s in metadata.signals}
    missing_signals = [n for n in signal_names if n not in columns]
    if missing_signals:
        raise EpisodeValidationError(
            f"Episode {metadata.episode_id!r} declares signals with no frame "
            f"column: {missing_signals}"
        )

    # Every non-required column should be declared as a signal so the UI knows
    # how to interpret it. (Required columns are implicit.)
    undeclared = [
        c for c in columns if c not in REQUIRED_FRAME_COLUMNS and c not in signal_names
    ]
    if undeclared:
        raise EpisodeValidationError(
            f"Episode {metadata.episode_id!r} has frame columns not declared as "
            f"signals in metadata: {undeclared}"
        )

    lengths = {len(arr) for arr in columns.values()}
    if len(lengths) != 1:
        raise EpisodeValidationError(
            f"Episode {metadata.episode_id!r} has ragged frame columns: lengths={lengths}"
        )
    (n_rows,) = lengths
    if n_rows != metadata.num_frames:
        raise EpisodeValidationError(
            f"Episode {metadata.episode_id!r}: metadata.num_frames="
            f"{metadata.num_frames} but frames have {n_rows} rows"
        )

    if "done" in columns and "terminated" in columns and "truncated" in columns:
        done = np.asarray(columns["done"], dtype=bool)
        expected = np.asarray(columns["terminated"], dtype=bool) | np.asarray(
            columns["truncated"], dtype=bool
        )
        if not np.array_equal(done, expected):
            raise EpisodeValidationError(
                f"Episode {metadata.episode_id!r}: 'done' column must equal "
                f"'terminated or truncated' at every frame"
            )

    if check_finite:
        _check_finite(metadata.episode_id, columns)


def _check_finite(episode_id: str, columns: dict[str, np.ndarray]) -> None:
    for name, arr in columns.items():
        a = np.asarray(arr)
        if not np.issubdtype(a.dtype, np.floating):
            continue
        if np.isnan(a).any():
            idx = int(np.argmax(np.isnan(a)))
            raise EpisodeValidationError(
                f"Episode {episode_id!r}: column {name!r} contains NaN at frame {idx}"
            )
        if np.isinf(a).any():
            idx = int(np.argmax(np.isinf(a)))
            raise EpisodeValidationError(
                f"Episode {episode_id!r}: column {name!r} contains Inf at frame {idx}"
            )
