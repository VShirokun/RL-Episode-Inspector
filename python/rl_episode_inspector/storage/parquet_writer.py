"""Write frame columns to Parquet via PyArrow (no pandas dependency)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def write_frames(path: str | Path, columns: dict[str, np.ndarray]) -> None:
    """Write a dict of equal-length numpy arrays to ``path`` as Parquet.

    dtypes are preserved from the numpy arrays, so callers are responsible for
    casting to the intended dtype (e.g. ``float32``) before calling.
    """
    if not columns:
        raise ValueError("Cannot write frames: no columns provided")

    lengths = {name: len(arr) for name, arr in columns.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"All frame columns must have equal length, got {lengths}")

    arrays = {name: pa.array(np.asarray(arr)) for name, arr in columns.items()}
    table = pa.table(arrays)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path), compression="snappy")
