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
    # Which agent this signal belongs to in a multi-agent episode (matches an
    # ``AgentSpec.id``). ``None`` = single-agent or a shared/team signal. The UI
    # groups reward signals by this so each agent gets its own reward panels.
    agent: str | None = None
    # Optional UI hints (color, group, default-visible, ...). Kept free-form on
    # purpose so the frontend can evolve without a schema migration.
    display: dict | None = None

    model_config = {"use_enum_values": True}


class AgentSpec(BaseModel):
    """One agent of a multi-agent episode (e.g. Isaac ``DirectMARLEnv``).

    ``id`` matches ``SignalSpec.agent`` on that agent's signals. ``team`` groups
    cooperating/competing agents (optional). Single-agent episodes carry no
    ``AgentSpec``s at all, so nothing about them changes.
    """

    id: str
    label: str | None = None
    team: str | None = None


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


class LightSpec(BaseModel):
    """A scene light captured from the source sim, for the 3D viewer.

    Renderer-agnostic so the frontend can map it onto Three.js lights. ``kind``
    is one of ``directional`` (a far light like a USD DistantLight/sun),
    ``point`` (a local SphereLight), ``ambient`` (flat fill) or ``hemisphere``
    (sky/ground fill, e.g. a DomeLight). ``color`` is linear RGB in 0..1.
    ``intensity`` is normalized for real-time rendering (NOT the source's
    physical photometric units, which don't map 1:1 to a rasterizer).
    ``direction`` (for ``directional``) is the unit vector the light travels
    along, and ``position`` (for ``point``) is its location — both in the
    recorded sim frame (z-up for Isaac), same as body poses.
    """

    name: str
    kind: str = "directional"  # directional | point | ambient | hemisphere
    color: list[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0])
    intensity: float = 1.0
    direction: list[float] | None = None  # [x, y, z], sim frame
    position: list[float] | None = None  # [x, y, z], sim frame


class CameraSpec(BaseModel):
    """A fixed viewpoint captured from the source sim (e.g. the env's play camera).

    ``eye``/``lookat`` are in the recorded sim frame (z-up for Isaac), same as
    body poses. When present, the viewer parks the camera here instead of
    auto-framing + following a body, so the replay matches the task's play view.
    """

    eye: list[float]
    lookat: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])


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
    # Lights captured from the source sim. Empty => the viewer uses its built-in
    # default light rig (which the user can also toggle).
    lights: list[LightSpec] = Field(default_factory=list)
    # Fixed camera captured from the task (e.g. its play viewpoint). None => the
    # viewer auto-frames and follows the root body.
    camera: CameraSpec | None = None
    up_axis: str = "z"
