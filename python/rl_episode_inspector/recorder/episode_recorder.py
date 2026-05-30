"""Task-agnostic episode recorder.

The recorder turns a stream of per-frame callbacks (state, action, raw rewards +
weights, termination flags) into a validated episode on disk. It hardcodes
nothing about Cartpole: the signal set is inferred from the keys passed to
``record_frame``. A task adapter (e.g. the Cartpole example) is responsible for
extracting those values from its environment.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from ..storage import (
    BodySpec,
    EpisodeMetadata,
    EpisodeStore,
    MarkerSpec,
    SignalKind,
    SignalSpec,
    ViewerSpec,
)
from ..storage.schemas import utcnow_iso
from .reward_buffer import RewardBuffer

# Suffixes for the 7 flattened columns of a body pose (position + wxyz quaternion).
_POSE_SUFFIXES = ("px", "py", "pz", "qw", "qx", "qy", "qz")


def pose_columns(body: str) -> list[str]:
    """The 7 frame-column names that store ``body``'s pose."""
    return [f"pose_{body}_{s}" for s in _POSE_SUFFIXES]


class RecorderError(RuntimeError):
    """Raised on misuse of the recorder (e.g. record before start)."""


def _finite(value: float, what: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite value for {what}: {value}")
    return value


class EpisodeRecorder:
    """Record one or more episodes in sequence to an episodes directory.

    For the MVP a single recorder instance follows one selected ``env_id``.
    """

    def __init__(
        self,
        output_dir: str | Path,
        task_name: str,
        dt: float,
        *,
        env_id: int = 0,
        run_id: str | None = None,
        task_source: str = "recorder",
        episode_id_prefix: str | None = None,
        viewer_type: str = "generic",
        state_mapping: dict[str, str] | None = None,
        signal_units: dict[str, str] | None = None,
        signal_descriptions: dict[str, str] | None = None,
        markers: list[MarkerSpec] | None = None,
        up_axis: str = "z",
        max_saved_episodes: int | None = None,
    ) -> None:
        self.store = EpisodeStore(output_dir)
        self.task_name = task_name
        self.dt = float(dt)
        self.fps = 1.0 / self.dt if self.dt > 0 else 0.0
        self.env_id = env_id
        self.run_id = run_id
        self.task_source = task_source
        self.episode_id_prefix = episode_id_prefix or _slug(task_name)
        self.viewer_type = viewer_type
        self.state_mapping = dict(state_mapping or {})
        self.signal_units = dict(signal_units or {})
        self.signal_descriptions = dict(signal_descriptions or {})
        self.markers = list(markers or [])
        self.up_axis = up_axis
        self.max_saved_episodes = max_saved_episodes

        # Articulation structure (persists across episodes once registered).
        self._body_names: list[str] | None = None
        self._body_specs: list[BodySpec] = []
        self._pose_cols: dict[str, list[str]] = {}

        self._saved_count = 0
        self._reset_active_episode()

    def register_bodies(
        self,
        names: list[str],
        parents: list[int],
        meshes: Sequence[str | None] | None = None,
    ) -> None:
        """Declare the articulation's rigid bodies (call before recording poses).

        ``parents[i]`` is the index into ``names`` of body ``i``'s kinematic
        parent (-1 for a root). Optional ``meshes[i]`` is a GLB path (relative to
        the backend's assets dir) used by the viewer's "models" mode; bodies
        without a mesh (or in "cubes" mode) render as proxy boxes. Each body's
        pose is recorded via the ``poses`` argument of :meth:`record_frame`.
        """
        if len(names) != len(parents):
            raise ValueError("names and parents must have the same length")
        if meshes is not None and len(meshes) != len(names):
            raise ValueError("meshes must have the same length as names")
        self._body_names = list(names)
        self._pose_cols = {b: pose_columns(b) for b in names}
        self._body_specs = [
            BodySpec(
                name=b,
                parent=int(parents[i]),
                pos=self._pose_cols[b][:3],
                quat=self._pose_cols[b][3:],
                mesh=(meshes[i] if meshes is not None else None),
            )
            for i, b in enumerate(names)
        ]

    # -- lifecycle ---------------------------------------------------------
    def _reset_active_episode(self) -> None:
        self._active = False
        self._episode_index = 0
        self._global_step_start = 0
        self._seed: int | None = None
        self._rewards = RewardBuffer()
        self._frame_index: list[int] = []
        self._timestamp: list[float] = []
        self._terminated: list[bool] = []
        self._truncated: list[bool] = []
        self._state: dict[str, list[float]] = {}
        self._action: dict[str, list[float]] = {}
        self._poses: dict[str, list[float]] = {}  # pose column -> per-frame values
        self._state_keys: list[str] | None = None
        self._action_keys: list[str] | None = None

    def start_episode(
        self, episode_index: int, global_step: int = 0, seed: int | None = None
    ) -> None:
        if self._active:
            raise RecorderError("start_episode called while an episode is already active")
        self._reset_active_episode()
        self._active = True
        self._episode_index = episode_index
        self._global_step_start = global_step
        self._seed = seed

    def record_frame(
        self,
        frame_index: int,
        timestamp: float,
        state: dict[str, float],
        action: dict[str, float],
        rewards_raw: dict[str, float],
        reward_weights: dict[str, float],
        terminated: bool,
        truncated: bool,
        poses: Mapping[str, Sequence[float]] | None = None,
    ) -> None:
        if not self._active:
            raise RecorderError("record_frame called before start_episode")

        self._check_keys("state", state, "_state_keys")
        self._check_keys("action", action, "_action_keys")

        self._frame_index.append(int(frame_index))
        self._timestamp.append(_finite(timestamp, "timestamp"))
        self._terminated.append(bool(terminated))
        self._truncated.append(bool(truncated))

        for key, value in state.items():
            self._state.setdefault(key, []).append(_finite(value, f"state[{key}]"))
        for key, value in action.items():
            self._action.setdefault(key, []).append(_finite(value, f"action[{key}]"))
        self._record_poses(poses)

        self._rewards.add_frame(rewards_raw, reward_weights)

    def _record_poses(self, poses: Mapping[str, Sequence[float]] | None) -> None:
        if not poses:
            if self._body_names:
                raise RecorderError("bodies registered but no poses passed to record_frame")
            return
        if self._body_names is None:
            raise RecorderError("call register_bodies(...) before recording poses")
        if list(poses.keys()) != self._body_names:
            raise RecorderError(
                f"pose body set changed: expected {self._body_names}, got {list(poses)}"
            )
        for body in self._body_names:
            values = poses[body]
            if len(values) != 7:
                raise RecorderError(
                    f"pose for body {body!r} must be 7 numbers (px,py,pz,qw,qx,qy,qz), "
                    f"got {len(values)}"
                )
            for col, value in zip(self._pose_cols[body], values, strict=True):
                self._poses.setdefault(col, []).append(_finite(value, f"pose[{col}]"))

    def end_episode(
        self, global_step: int | None = None, reset_reason: str | None = None
    ) -> str | None:
        if not self._active:
            raise RecorderError("end_episode called before start_episode")

        n = len(self._frame_index)
        if n == 0:
            # Nothing recorded; drop the episode rather than write an empty one.
            self._reset_active_episode()
            return None

        if self.max_saved_episodes is not None and self._saved_count >= self.max_saved_episodes:
            self._reset_active_episode()
            return None

        terminated = bool(self._terminated[-1])
        truncated = bool(self._truncated[-1])
        global_step_end = (
            global_step if global_step is not None else self._global_step_start + n - 1
        )

        columns = self._build_columns()
        signals = self._build_signals()
        episode_id = f"{self.episode_id_prefix}_{self._episode_index:06d}"

        metadata = EpisodeMetadata(
            episode_id=episode_id,
            run_id=self.run_id,
            task_name=self.task_name,
            task_source=self.task_source,
            env_id=self.env_id,
            episode_index=self._episode_index,
            created_at=utcnow_iso(),
            num_frames=n,
            dt=self.dt,
            fps=self.fps,
            duration_seconds=n * self.dt,
            global_step_start=self._global_step_start,
            global_step_end=int(global_step_end),
            terminated=terminated,
            truncated=truncated,
            reset_reason=reset_reason,
            episode_return=float(self._rewards.cumulative_return),
            seed=self._seed,
            signals=signals,
            viewer=ViewerSpec(
                type=self.viewer_type,
                state_mapping=self.state_mapping,
                bodies=self._body_specs,
                markers=self.markers,
                up_axis=self.up_axis,
            ),
        )

        self.store.save_episode(metadata, columns)
        self._saved_count += 1
        self._reset_active_episode()
        return episode_id

    # -- helpers -----------------------------------------------------------
    def _check_keys(self, what: str, values: dict[str, float], attr: str) -> None:
        keys = list(values.keys())
        existing = getattr(self, attr)
        if existing is None:
            setattr(self, attr, keys)
        elif keys != existing:
            raise RecorderError(
                f"{what} keys changed mid-episode: expected {existing}, got {keys}"
            )

    def _build_columns(self) -> dict[str, np.ndarray]:
        n = len(self._frame_index)
        terminated = np.asarray(self._terminated, dtype=bool)
        truncated = np.asarray(self._truncated, dtype=bool)
        columns: dict[str, np.ndarray] = {
            "frame_index": np.asarray(self._frame_index, dtype=np.int32),
            "timestamp": np.asarray(self._timestamp, dtype=np.float32),
            "terminated": terminated,
            "truncated": truncated,
            "done": terminated | truncated,
        }
        for key, vals in self._state.items():
            columns[key] = np.asarray(vals, dtype=np.float32)
        for key, vals in self._action.items():
            columns[key] = np.asarray(vals, dtype=np.float32)
        for name, vals in self._rewards.column_dict().items():
            columns[name] = np.asarray(vals, dtype=np.float32)
        for name, vals in self._poses.items():
            columns[name] = np.asarray(vals, dtype=np.float32)
        # Defensive: ensure all columns are the right length.
        for name, arr in columns.items():
            if len(arr) != n:
                raise RecorderError(f"Column {name!r} has length {len(arr)}, expected {n}")
        return columns

    def _build_signals(self) -> list[SignalSpec]:
        signals: list[SignalSpec] = []

        def spec(name: str, kind: SignalKind, unit: str | None = None) -> SignalSpec:
            return SignalSpec(
                name=name,
                kind=kind,
                dtype="float32",
                shape=[],
                unit=self.signal_units.get(name, unit),
                description=self.signal_descriptions.get(name),
            )

        for key in self._state:
            signals.append(spec(key, SignalKind.state))
        for key in self._action:
            signals.append(spec(key, SignalKind.action))
        for name in self._rewards.term_names:
            signals.append(spec(f"reward_{name}_raw", SignalKind.reward_raw))
            signals.append(spec(f"reward_{name}_weighted", SignalKind.reward_weighted))
        signals.append(spec("reward_step_total", SignalKind.reward_total))
        signals.append(spec("reward_cumulative", SignalKind.reward_total))
        for col in self._poses:
            signals.append(spec(col, SignalKind.pose))
        return signals

    @property
    def saved_count(self) -> int:
        return self._saved_count


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_") or "episode"
