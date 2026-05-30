"""FastAPI application factory."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """Create the API app serving episodes from ``episodes_dir``.

    ``episodes_dir`` may also be supplied via the ``RLEI_EPISODES_DIR`` env var
    (used by ``uvicorn rl_episode_inspector.server.app:app`` style launches).
    """
    if episodes_dir is None:
        episodes_dir = os.environ.get("RLEI_EPISODES_DIR", "sample_data/cartpole/episodes")

    app = FastAPI(
        title="RL Episode Inspector API",
        version="0.1.0",
        summary="Serve recorded RL episodes (metadata, frames, signals, ranking).",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or DEFAULT_CORS_ORIGINS,
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    app.state.episodes_dir = str(episodes_dir)
    app.include_router(create_router(episodes_dir))

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "episodes_dir": str(episodes_dir)}

    return app


# Module-level app for `uvicorn ...server.app:app` (reads RLEI_EPISODES_DIR).
app = create_app()
