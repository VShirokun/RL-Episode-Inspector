from __future__ import annotations

import numpy as np
import pytest

from rl_episode_inspector.storage import EpisodeStore
from rl_episode_inspector.storage.paths import UnsafeEpisodeIdError, safe_episode_dir
from rl_episode_inspector.storage.validation import (
    EpisodeValidationError,
    check_schema_version,
    validate_episode,
)


def test_valid_episode_passes(minimal_episode):
    metadata, columns = minimal_episode
    validate_episode(metadata, columns)  # should not raise


def test_missing_required_column(minimal_episode):
    metadata, columns = minimal_episode
    del columns["done"]
    with pytest.raises(EpisodeValidationError, match="required frame columns"):
        validate_episode(metadata, columns)


def test_signal_without_column(minimal_episode):
    metadata, columns = minimal_episode
    del columns["cart_position"]
    with pytest.raises(EpisodeValidationError, match="no frame"):
        validate_episode(metadata, columns)


def test_undeclared_column(minimal_episode):
    metadata, columns = minimal_episode
    columns["mystery"] = np.zeros(metadata.num_frames, dtype=np.float32)
    with pytest.raises(EpisodeValidationError, match="not declared as signals"):
        validate_episode(metadata, columns)


def test_num_frames_mismatch(minimal_episode):
    metadata, columns = minimal_episode
    metadata.num_frames = 99
    with pytest.raises(EpisodeValidationError, match="num_frames"):
        validate_episode(metadata, columns)


def test_done_inconsistent(minimal_episode):
    metadata, columns = minimal_episode
    columns["done"] = np.zeros(metadata.num_frames, dtype=bool)  # but terminated has a True
    with pytest.raises(EpisodeValidationError, match="terminated or truncated"):
        validate_episode(metadata, columns)


def test_nan_detection(minimal_episode):
    metadata, columns = minimal_episode
    columns["cart_position"][2] = np.nan
    with pytest.raises(EpisodeValidationError, match="NaN at frame 2"):
        validate_episode(metadata, columns)


def test_inf_detection(minimal_episode):
    metadata, columns = minimal_episode
    columns["cart_position"][1] = np.inf
    with pytest.raises(EpisodeValidationError, match="Inf at frame 1"):
        validate_episode(metadata, columns)


def test_schema_version_incompatible():
    with pytest.raises(EpisodeValidationError, match="Incompatible schema_version"):
        check_schema_version("9.0.0")
    check_schema_version("0.99.0")  # same major -> ok


def test_path_traversal_rejected(tmp_path):
    for bad in ["../secret", "..", "a/b", "/etc/passwd", "x/../../y", ""]:
        with pytest.raises(UnsafeEpisodeIdError):
            safe_episode_dir(tmp_path, bad)
    # valid id resolves inside root
    good = safe_episode_dir(tmp_path, "cartpole_000001")
    assert str(good).startswith(str(tmp_path.resolve()))


def test_store_exists_false_for_bad_id(tmp_path):
    store = EpisodeStore(tmp_path)
    assert store.exists("../escape") is False
