"""Local FastAPI backend serving episodes to the frontend viewer."""

from __future__ import annotations

from .app import create_app

__all__ = ["create_app"]
