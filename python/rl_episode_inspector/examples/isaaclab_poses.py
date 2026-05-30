"""Helpers to auto-capture full-body poses from an Isaac Lab articulation.

Generic over any robot (Franka, humanoid, ...): given an Isaac Lab Articulation,
enumerate all rigid bodies, their kinematic parents, and read every body's world
pose each frame (converted to the env-local frame). Task adapters feed these to
``EpisodeRecorder.register_bodies`` / ``record_frame(poses=...)``.

This module imports nothing at top level that requires Isaac Lab, but its
functions expect Isaac Lab objects at call time (so it's only used inside a
launched-app rollout). It is example/adapter code — the core never imports it.
"""

from __future__ import annotations

from typing import Any


def to_numpy(x: Any):
    """Convert a torch tensor or (Newton backend) warp array to numpy."""
    if hasattr(x, "numpy") and "warp" in type(x).__module__:
        return x.numpy()
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    import numpy as np

    return np.asarray(x)


def body_structure(robot: Any) -> tuple[list[str], list[int]]:
    """Return (body_names, parent_indices) for an articulation.

    ``parent_indices[i]`` indexes ``body_names`` (-1 for a root). Tries to read
    the true kinematic tree from the PhysX articulation; falls back to a serial
    chain (parent = previous body), which is a reasonable skeleton for arms.
    """
    names = list(robot.data.body_names)
    parents = _parents_or_chain(robot, names)
    return names, parents


def _parents_or_chain(robot: Any, names: list[str]) -> list[int]:
    n = len(names)
    # Best effort: several Isaac Sim versions expose link parents differently.
    for getter in (
        lambda: list(robot.root_physx_view.shared_metatype.link_parents),
        lambda: list(robot.root_physx_view.get_link_parents()),
    ):
        try:
            parents = getter()
            if len(parents) == n:
                # PhysX uses -1 (or self) for the root; normalize self->-1.
                return [(-1 if int(p) in (i, -1) else int(p)) for i, p in enumerate(parents)]
        except Exception:  # noqa: BLE001 - any API mismatch -> fall back
            pass
    # Fallback: serial chain (root + each body parented to the previous one).
    return [-1] + list(range(n - 1))


def read_body_poses(robot: Any, env_origin) -> dict[str, tuple[float, ...]]:
    """Per-body pose ``{name: (px,py,pz,qw,qx,qy,qz)}`` in the env-local frame."""
    pos = to_numpy(robot.data.body_pos_w)[0]  # (num_bodies, 3), world frame
    quat = to_numpy(robot.data.body_quat_w)[0]  # (num_bodies, 4), wxyz
    out: dict[str, tuple[float, ...]] = {}
    for i, name in enumerate(robot.data.body_names):
        p = pos[i] - env_origin
        q = quat[i]
        out[name] = (
            float(p[0]), float(p[1]), float(p[2]),
            float(q[0]), float(q[1]), float(q[2]), float(q[3]),
        )
    return out
