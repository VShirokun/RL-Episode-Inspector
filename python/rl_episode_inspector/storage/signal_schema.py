"""Generic signal metadata model.

A *signal* is one named time-series column inside ``frames.parquet``. Signals
are described generically so the tool is not tied to any specific task: a
Cartpole adapter contributes ``cart_position``/``pole_angle`` signals, a
humanoid adapter would contribute different ones, but the storage, backend and
frontend only ever reason about ``SignalSpec`` objects.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class SignalKind(str, Enum):
    """How a signal should be interpreted by the UI and ranking layers."""

    state = "state"
    reward_raw = "reward_raw"
    reward_weighted = "reward_weighted"
    reward_total = "reward_total"
    action = "action"
    observation = "observation"
    debug = "debug"
    event = "event"


class SignalSpec(BaseModel):
    """Metadata describing a single recorded signal/column."""

    name: str
    kind: SignalKind
    dtype: str = "float32"
    shape: list[int] = Field(default_factory=list)
    unit: str | None = None
    description: str | None = None
    # Optional UI hints (color, group, default-visible, ...). Kept free-form on
    # purpose so the frontend can evolve without a schema migration.
    display: dict | None = None

    model_config = {"use_enum_values": True}


class ViewerSpec(BaseModel):
    """Tells the frontend which 3D viewer to use and how to feed it state.

    ``state_mapping`` maps a viewer-internal role (e.g. ``"cart_position"``) to
    the name of a recorded signal. The Cartpole viewer is the only one shipped
    in the MVP, but the indirection keeps the core viewer-agnostic.
    """

    type: str = "generic"
    state_mapping: dict[str, str] = Field(default_factory=dict)
