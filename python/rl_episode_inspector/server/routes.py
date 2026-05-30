"""API routes. See docs/architecture.md §Backend for the contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from ..ranking import EpisodeRanker
from ..storage import EpisodeStore
from ..storage.paths import UnsafeEpisodeIdError
from ..storage.validation import EpisodeValidationError
from .security import require_valid_episode_id


def _jsonify_columns(columns: dict[str, np.ndarray]) -> dict[str, list]:
    """Convert numpy frame columns to JSON-serializable lists."""
    out: dict[str, list] = {}
    for name, arr in columns.items():
        a = np.asarray(arr)
        if np.issubdtype(a.dtype, np.bool_):
            out[name] = a.astype(bool).tolist()
        elif np.issubdtype(a.dtype, np.integer):
            out[name] = a.astype(int).tolist()
        else:
            out[name] = a.astype(float).tolist()
    return out


def create_router(episodes_dir: str | Path) -> APIRouter:
    store = EpisodeStore(episodes_dir)
    router = APIRouter(prefix="/api")

    def _load_metadata(episode_id: str):
        require_valid_episode_id(episode_id)
        if not store.exists(episode_id):
            raise HTTPException(status_code=404, detail=f"Unknown episode_id: {episode_id}")
        try:
            return store.load_metadata(episode_id)
        except EpisodeValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.get("/episodes")
    def list_episodes() -> dict:
        ranker = EpisodeRanker(episodes_dir)
        return {"episodes": [s.model_dump() for s in ranker.list_episodes()]}

    @router.get("/episodes/{episode_id}/metadata")
    def get_metadata(episode_id: str) -> dict:
        return _load_metadata(episode_id).model_dump()

    @router.get("/episodes/{episode_id}/frames")
    def get_frames(
        episode_id: str,
        start: int | None = Query(default=None, ge=0),
        end: int | None = Query(default=None, ge=0),
        names: str | None = Query(default=None),
    ) -> dict:
        require_valid_episode_id(episode_id)
        if not store.exists(episode_id):
            raise HTTPException(status_code=404, detail=f"Unknown episode_id: {episode_id}")
        wanted = _parse_names(names, store, episode_id) if names else None
        try:
            columns = store.load_frames(episode_id, start=start, end=end, names=wanted)
        except (UnsafeEpisodeIdError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        json_columns = _jsonify_columns(columns)
        first: list = next(iter(json_columns.values()), [])
        return {
            "episode_id": episode_id,
            "start": start or 0,
            "count": len(first),
            "columns": json_columns,
        }

    @router.get("/episodes/{episode_id}/signals")
    def get_signals(episode_id: str, names: str | None = Query(default=None)) -> dict:
        meta = _load_metadata(episode_id)
        if not names:
            return {"signals": [s.model_dump() for s in meta.signals]}
        wanted = [n.strip() for n in names.split(",") if n.strip()]
        specs = [s for s in meta.signals if s.name in wanted]
        found = [s.name for s in specs]
        columns = store.load_frames(episode_id, names=["frame_index", *found])
        series = {name: _jsonify_columns({name: columns[name]})[name] for name in found}
        return {
            "signals": [s.model_dump() for s in specs],
            "frame_index": _jsonify_columns({"frame_index": columns["frame_index"]})[
                "frame_index"
            ],
            "series": series,
        }

    @router.get("/ranking")
    def ranking(mode: str = Query(..., pattern="^(best|worst|median)$")) -> dict:
        ranker = EpisodeRanker(episodes_dir)
        summary = ranker.get(mode)
        if summary is None:
            raise HTTPException(status_code=404, detail="No episodes available to rank")
        return summary.model_dump()

    return router


def _parse_names(names: str, store: EpisodeStore, episode_id: str) -> list[str]:
    requested = [n.strip() for n in names.split(",") if n.strip()]
    # Always include required columns so the frontend can align series to frames.
    required = ["frame_index", "timestamp", "terminated", "truncated", "done"]
    return list(dict.fromkeys([*required, *requested]))
