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
    pose = "pose"  # a component of a rigid body's 3D pose (position or quaternion)
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


class BodySpec(BaseModel):
    """One rigid body of an articulation, for the articulation viewer.

    ``pos``/``quat`` name the frame columns holding this body's pose (a position
    triple and a wxyz quaternion). ``parent`` indexes another body in the same
    list (-1 = root) so the viewer can draw a "bone" to the parent. This is
    generic over any robot: Franka now, a humanoid later.
    """

    name: str
    parent: int = -1
    pos: list[str]  # [px_col, py_col, pz_col]
    quat: list[str]  # [qw_col, qx_col, qy_col, qz_col]
    # Optional path to this body's 3D mesh (GLB), relative to the assets dir the
    # backend serves at /assets. When absent (or in "cubes" render mode) the
    # viewer draws a proxy box instead.
    mesh: str | None = None


class MarkerSpec(BaseModel):
    """A point marker to render (e.g. a reach target). ``pos`` names 3 columns."""

    name: str
    pos: list[str]
    color: str | None = None


class ViewerSpec(BaseModel):
    """Tells the frontend which 3D viewer to use and how to feed it state.

    ``state_mapping`` maps a viewer-internal role (e.g. ``"cart_position"``) to
    the name of a recorded signal (used by the simple cartpole/reach viewers).
    For full-robot replay, ``bodies`` describes the articulation (one entry per
    rigid body, each pointing at its pose columns) and ``markers`` adds point
    markers; the generic ``articulation3d`` viewer renders these. ``up_axis``
    tells the viewer which axis is "up" in the recorded poses ("z" for Isaac).
    """

    type: str = "generic"
    state_mapping: dict[str, str] = Field(default_factory=dict)
    bodies: list[BodySpec] = Field(default_factory=list)
    markers: list[MarkerSpec] = Field(default_factory=list)
    up_axis: str = "z"
    # How the articulation viewer orients each body's mesh:
    #   "quaternion" — use the recorded body quaternion (correct when poses come
    #                  straight from the sim, e.g. Franka/Cartpole).
    #   "bone"       — orient each mesh's long axis along the bone to its child
    #                  (robust when recorded orientations are in a different frame
    #                  than the geometry, e.g. retargeted humanoid mocap).
    orient_mode: str = "quaternion"
