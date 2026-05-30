"""FastAPI application factory."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from ..storage.paths import UnsafeEpisodeIdError, safe_subpath
from .routes import create_router

DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


def create_app(
    episodes_dir: str | Path | None = None,
    *,
    assets_dir: str | Path | None = None,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """Create the API app serving episodes from ``episodes_dir``.

    ``episodes_dir`` / ``assets_dir`` may also come from the ``RLEI_EPISODES_DIR``
    / ``RLEI_ASSETS_DIR`` env vars. ``assets_dir`` holds robot mesh GLBs served at
    ``/assets`` (defaults to ``<episodes_dir>/../assets``).
    """
    if episodes_dir is None:
        episodes_dir = os.environ.get("RLEI_EPISODES_DIR", "sample_data/cartpole/episodes")
    if assets_dir is None:
        assets_dir = os.environ.get("RLEI_ASSETS_DIR") or (Path(episodes_dir).parent / "assets")

    app = FastAPI(
        title="RL Episode Inspector API",
        version="0.1.0",
        summary="Serve recorded RL episodes (metadata, frames, signals, ranking, meshes).",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or DEFAULT_CORS_ORIGINS,
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    app.state.episodes_dir = str(episodes_dir)
    app.state.assets_dir = str(assets_dir)
    app.include_router(create_router(episodes_dir))

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "episodes_dir": str(episodes_dir), "assets_dir": str(assets_dir)}

    @app.get("/api/assets/{path:path}")
    def get_asset(path: str) -> FileResponse:
        """Serve a robot mesh (or other asset) file, with path-traversal protection.

        Under /api so the frontend dev proxy forwards it and it never clashes with
        the built frontend's own /assets bundle.
        """
        try:
            resolved = safe_subpath(assets_dir, path)
        except UnsafeEpisodeIdError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not resolved.is_file():
            raise HTTPException(status_code=404, detail=f"Asset not found: {path}")
        return FileResponse(resolved)

    return app


# Module-level app for `uvicorn ...server.app:app` (reads RLEI_EPISODES_DIR).
app = create_app()
