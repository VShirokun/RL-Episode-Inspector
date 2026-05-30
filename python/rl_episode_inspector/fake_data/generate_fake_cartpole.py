"""Generate fake Cartpole-like episodes without Isaac Lab.

The trajectories come from a toy closed-loop inverted-pendulum model, not a real
simulator — the point is to produce *plausible* signals with real reward
decomposition so the storage layer, backend, frontend, charts and E2E tests can
be exercised end-to-end without GPU/Omniverse. Each episode is given a random
"skill" that controls how well the controller stabilizes the pole, which yields
a spread of episode returns (so best/worst/median ranking is meaningful).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..recorder import EpisodeRecorder

# Reward weights for the demo task. Mirrors docs/§13 and the Cartpole example
# config; kept here so fake data and the real Cartpole example agree.
DEFAULT_REWARD_WEIGHTS: dict[str, float] = {
    "alive": 1.0,
    "pole_upright": 2.0,
    "cart_centering": 0.5,
    "cart_velocity_penalty": 0.05,
    "pole_angular_velocity_penalty": 0.05,
    "action_penalty": 0.01,
}

# Cartpole signal metadata (units/descriptions) advertised to the UI.
_STATE_MAPPING = {"cart_position": "cart_position", "pole_angle": "pole_angle"}
_SIGNAL_UNITS = {
    "cart_position": "m",
    "cart_velocity": "m/s",
    "pole_angle": "rad",
    "pole_angular_velocity": "rad/s",
    "action_cart_force": "N",
}
_SIGNAL_DESCRIPTIONS = {
    "cart_position": "Cart position along the track",
    "cart_velocity": "Cart linear velocity",
    "pole_angle": "Pole angle from vertical",
    "pole_angular_velocity": "Pole angular velocity",
    "action_cart_force": "Force applied to the cart",
}

_POLE_FALL_THRESHOLD = 0.5  # rad (~28 deg)


def _reward_terms(x: float, x_dot: float, theta: float, theta_dot: float, action: float):
    """Raw reward terms for one frame (before weighting)."""
    return {
        "alive": 1.0,
        "pole_upright": float(max(-1.0, min(1.0, math.cos(theta)))),
        "cart_centering": -(x * x),
        "cart_velocity_penalty": -(x_dot * x_dot),
        "pole_angular_velocity_penalty": -(theta_dot * theta_dot),
        "action_penalty": -(action * action),
    }


def _simulate_one(
    recorder: EpisodeRecorder,
    *,
    episode_index: int,
    skill: float,
    max_frames: int,
    dt: float,
    weights: dict[str, float],
    rng: np.random.Generator,
    seed: int,
) -> str | None:
    """Roll out one toy episode and record it. Returns the episode_id."""
    # Toy unstable pole + stabilizing controller whose gain scales with skill.
    # Closed-loop pole accel is (a - b*kp)*theta - b*kd*theta_dot, so the pole is
    # only stabilized when b*kp > a, i.e. skill > a/(b*4) = 0.5. Skills below that
    # diverge and the pole falls early -> a wide spread of episode returns.
    a, b, c = 16.0, 8.0, 0.5  # pole instability, control authority, cart accel
    kp, kd = skill * 4.0, skill * 0.8
    kx, kxd = 0.4, 0.5
    # Per-step angular disturbance that a weak controller cannot always reject.
    # Lower skill -> bigger kicks -> the pole random-walks across the fall
    # threshold at a skill-dependent (and therefore varied) time.
    disturbance = 0.16 * max(0.1, 1.3 - min(skill, 1.2))

    x, x_dot = 0.0, 0.0
    theta = float(rng.normal(0.0, 0.05))
    theta_dot = float(rng.normal(0.0, 0.05))

    recorder.start_episode(episode_index=episode_index, global_step=0, seed=seed)

    terminated = truncated = False
    reset_reason: str | None = None
    n = 0
    for frame in range(max_frames):
        action = kp * theta + kd * theta_dot + kx * x + kxd * x_dot
        action += float(rng.normal(0.0, 0.2))
        action = float(np.clip(action, -20.0, 20.0))

        raw = _reward_terms(x, x_dot, theta, theta_dot, action)
        terminated = abs(theta) > _POLE_FALL_THRESHOLD
        truncated = (not terminated) and (frame == max_frames - 1)

        recorder.record_frame(
            frame_index=frame,
            timestamp=frame * dt,
            state={
                "cart_position": x,
                "cart_velocity": x_dot,
                "pole_angle": theta,
                "pole_angular_velocity": theta_dot,
            },
            action={"action_cart_force": action},
            rewards_raw=raw,
            reward_weights=weights,
            terminated=terminated,
            truncated=truncated,
        )
        n += 1

        if terminated:
            reset_reason = "pole_fell"
            break
        if truncated:
            reset_reason = "max_episode_length"
            break

        # Integrate toy dynamics: closed-loop pole response where the stabilizing
        # controller (gain ~ skill) pushes theta back toward upright.
        theta_ddot = (a - b * kp) * theta - (b * kd) * theta_dot
        theta_dot += float(rng.normal(0.0, disturbance))
        x_ddot = c * action - kx * x - kxd * x_dot

        theta_dot += theta_ddot * dt
        theta += theta_dot * dt
        x_dot += x_ddot * dt
        x += x_dot * dt

    return recorder.end_episode(global_step=n, reset_reason=reset_reason)


def generate_fake_cartpole_episodes(
    output_dir: str | Path,
    *,
    num_episodes: int = 20,
    seed: int = 42,
    max_frames: int = 400,
    fps: int = 60,
    weights: dict[str, float] | None = None,
    run_id: str = "fake_demo_run",
) -> list[str]:
    """Generate ``num_episodes`` fake episodes into ``output_dir``.

    Returns the list of created episode_ids.
    """
    weights = {**DEFAULT_REWARD_WEIGHTS, **(weights or {})}
    dt = 1.0 / fps
    root_rng = np.random.default_rng(seed)

    recorder = EpisodeRecorder(
        output_dir=output_dir,
        task_name="Cartpole",
        dt=dt,
        env_id=0,
        run_id=run_id,
        task_source="fake_data",
        episode_id_prefix="cartpole",
        viewer_type="cartpole",
        state_mapping=_STATE_MAPPING,
        signal_units=_SIGNAL_UNITS,
        signal_descriptions=_SIGNAL_DESCRIPTIONS,
    )

    created: list[str] = []
    for i in range(num_episodes):
        # Concentrate skills around the stability boundary (~0.5) so many runs
        # are marginally stable and fall at *varying* times, filling the mid-range
        # of returns; a few clearly-skilled runs survive to max length.
        skill = 0.32 + 0.85 * (i + 0.5) / num_episodes
        skill += float(root_rng.normal(0.0, 0.12))
        ep_seed = seed + i
        episode_id = _simulate_one(
            recorder,
            episode_index=i + 1,
            skill=max(0.2, skill),
            max_frames=max_frames,
            dt=dt,
            weights=weights,
            rng=np.random.default_rng(ep_seed),
            seed=ep_seed,
        )
        if episode_id is not None:
            created.append(episode_id)
    return created
