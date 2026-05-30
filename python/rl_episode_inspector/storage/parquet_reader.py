"""Read frame columns back from Parquet into numpy arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def read_frames(
    path: str | Path,
    start: int | None = None,
    end: int | None = None,
    names: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Read frames as ``{column_name: np.ndarray}``.

    ``start``/``end`` slice rows (half-open ``[start, end)``); ``names`` selects
    a subset of columns. Slicing happens after the read for the MVP — Parquet
    supports lazy/row-group reads, but full Cartpole episodes are tiny so a
    simple read-then-slice keeps the code obvious.
    """
    pf = pq.ParquetFile(str(path))
    available = pf.schema_arrow.names
    columns = names if names is not None else available
    missing = [c for c in columns if c not in available]
    if missing:
        raise KeyError(f"Requested columns not present in {Path(path).name}: {missing}")

    table = pf.read(columns=list(columns))
    n = table.num_rows
    lo = 0 if start is None else max(0, start)
    hi = n if end is None else min(n, end)
    if lo or hi != n:
        table = table.slice(lo, max(0, hi - lo))

    return {name: table.column(name).to_numpy(zero_copy_only=False) for name in table.column_names}


def read_num_rows(path: str | Path) -> int:
    return pq.ParquetFile(str(path)).metadata.num_rows
