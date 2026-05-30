"""Opt-in Isaac Lab integration test.

Marked ``isaaclab`` so it never runs in standard CI, and skips automatically if
Isaac Lab can't be imported. Run with a working Isaac Lab under its Python::

    /path/to/IsaacLab/isaaclab.sh -p -m pytest -m isaaclab

It records a couple of real Cartpole episodes and checks they load, contain the
expected reward decomposition + state columns, and have a sensible episode_return.

NOTE: Isaac Sim's Kit runtime does not always cooperate with pytest's process
model and can segfault at app launch/teardown *inside* pytest (a known Isaac Sim
limitation, independent of this code). The verified, reliable way to exercise the
exact same integration path is the standalone script
``examples/cartpole/generate_demo_episodes.py``. This marked test is provided for
environments where the Kit-under-pytest combination behaves.
"""

from __future__ import annotations

import importlib.util

import pytest

pytestmark = pytest.mark.isaaclab

isaaclab_available = importlib.util.find_spec("isaaclab") is not None


@pytest.mark.skipif(not isaaclab_available, reason="Isaac Lab is not installed")
def test_records_real_cartpole_episodes(tmp_path):
    # Imports are deferred so collection works even without Isaac Lab present.
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    simulation_app = app_launcher.app
    try:
        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401
        import numpy as np
        import torch
        from isaaclab_tasks.utils import parse_env_cfg

        from rl_episode_inspector.examples.cartpole.controller import Gains, balance_action
        from rl_episode_inspector.examples.cartpole.reward import compute_reward_terms
        from rl_episode_inspector.ranking import EpisodeRanker
        from rl_episode_inspector.recorder import EpisodeRecorder

        task = "Isaac-Cartpole-Direct-v0"
        env = gym.make(task, cfg=parse_env_cfg(task, num_envs=1))
        u = env.unwrapped
        robot = u.cartpole
        cart_i = robot.find_joints("slider_to_cart")[0][0]
        pole_i = robot.find_joints("cart_to_pole")[0][0]

        rec = EpisodeRecorder(
            output_dir=tmp_path, task_name="Cartpole", dt=float(u.step_dt),
            task_source="isaac_lab", episode_id_prefix="cartpole", viewer_type="cartpole",
            state_mapping={"cart_position": "cart_position", "pole_angle": "pole_angle"},
        )
        env.reset(seed=0)
        weights = {"alive": 1.0, "pole_upright": 2.0, "cart_centering": 0.5,
                   "cart_velocity_penalty": 0.05, "pole_angular_velocity_penalty": 0.05,
                   "action_penalty": 0.01}

        for ep in range(2):
            rec.start_episode(episode_index=ep + 1)
            terminated = truncated = False
            for frame in range(500):
                jp, jv = robot.data.joint_pos, robot.data.joint_vel
                cp, cv = float(jp[0, cart_i]), float(jv[0, cart_i])
                pa, pv = float(jp[0, pole_i]), float(jv[0, pole_i])
                act = balance_action(cp, cv, pa, pv, Gains())
                _, _, term_t, trunc_t, _ = env.step(
                    torch.tensor([[act]], dtype=torch.float32, device=u.device)
                )
                terminated, truncated = bool(term_t[0]), bool(trunc_t[0])
                rec.record_frame(
                    frame_index=frame, timestamp=frame * float(u.step_dt),
                    state={"cart_position": cp, "cart_velocity": cv,
                           "pole_angle": pa, "pole_angular_velocity": pv},
                    action={"action_cart_force": act * 100.0},
                    rewards_raw=compute_reward_terms(cp, cv, pa, pv, act, terminated),
                    reward_weights=weights, terminated=terminated, truncated=truncated,
                )
                if terminated or truncated:
                    break
            rec.end_episode(reset_reason="terminated" if terminated else "max_episode_length")

        env.close()

        ranker = EpisodeRanker(tmp_path)
        episodes = ranker.list_episodes()
        assert len(episodes) == 2
        ep = ranker.store.load_episode(episodes[0].episode_id)
        for col in ("cart_position", "pole_angle", "reward_step_total", "reward_cumulative",
                    "reward_alive_weighted", "reward_pole_upright_weighted"):
            assert col in ep.frames, col
        assert np.isfinite(ep.metadata.episode_return)
        assert ep.metadata.task_source == "isaac_lab"
    finally:
        simulation_app.close()
