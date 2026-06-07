"""Record real Isaac Lab Cart–Double-Pendulum episodes (a simple MARL task).

``Isaac-Cart-Double-Pendulum-Direct-v0`` is a ``DirectMARLEnv`` with two agents,
``cart`` and ``pendulum``, that share one articulation but get *different* reward
decompositions (the cart cares about the cart/pole, the pendulum about the
pendulum). This exercises the recorder's multi-agent reward path end to end.

We reproduce the env's own reward terms (so each agent's per-frame ``step_total``
equals the env's reward for that agent) and record them via ``rewards_by_agent``.
A small random action drives the under-actuated system until the pole/pendulum
fall out of bounds (terminated) or time runs out (truncated), giving a spread of
returns across episodes. Run under Isaac Lab's Python::

    PYTHONPATH=$PWD/python /path/to/isaaclab-env/bin/python \
        python/rl_episode_inspector/examples/cart_double_pendulum/generate_demo_episodes.py \
        --output-dir sample_data/cart_double_pendulum/episodes --num-episodes 6
"""

from __future__ import annotations

import argparse
import math


def _wrap(a: float) -> float:
    """Normalize an angle to [-pi, pi] (matches isaaclab normalize_angle)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def main() -> None:
    parser = argparse.ArgumentParser(description="Record real Isaac Lab MARL cart-double-pendulum.")
    parser.add_argument("--output-dir", default="sample_data/cart_double_pendulum/episodes")
    parser.add_argument("--num-episodes", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", default="Isaac-Cart-Double-Pendulum-Direct-v0")
    parser.add_argument("--action-scale", type=float, default=0.25,
                        help="magnitude of the random normalized action")

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    if args.enable_cameras is None:
        args.enable_cameras = False

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import numpy as np
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    from rl_episode_inspector.examples.isaaclab_poses import to_numpy
    from rl_episode_inspector.recorder import EpisodeRecorder
    from rl_episode_inspector.storage import AgentSpec

    env = gym.make(args.task, cfg=parse_env_cfg(args.task, num_envs=1))
    unwrapped = env.unwrapped
    device = unwrapped.device
    cfg = unwrapped.cfg
    robot = unwrapped.robot
    cart_i = robot.find_joints(cfg.cart_dof_name)[0][0]
    pole_i = robot.find_joints(cfg.pole_dof_name)[0][0]
    pend_i = robot.find_joints(cfg.pendulum_dof_name)[0][0]

    control_dt = float(cfg.sim.dt * cfg.decimation)
    max_steps = int(cfg.episode_length_s / control_dt)
    max_cart = float(cfg.max_cart_pos)

    # Per-agent reward weights = the env's reward scales (so weighted terms sum to
    # exactly the env's per-agent reward). cart_pos scale is 0 and unused.
    cart_w = {
        "alive": cfg.rew_scale_alive, "terminated": cfg.rew_scale_terminated,
        "pole_pos": cfg.rew_scale_pole_pos, "cart_vel": cfg.rew_scale_cart_vel,
        "pole_vel": cfg.rew_scale_pole_vel,
    }
    pend_w = {
        "alive": cfg.rew_scale_alive, "terminated": cfg.rew_scale_terminated,
        "pendulum_pos": cfg.rew_scale_pendulum_pos, "pendulum_vel": cfg.rew_scale_pendulum_vel,
    }

    recorder = EpisodeRecorder(
        output_dir=args.output_dir,
        task_name="CartDoublePendulum",
        dt=control_dt,
        run_id=f"marl_cartdp_seed{args.seed}",
        task_source="isaac_lab",
        episode_id_prefix="cartdp",
        viewer_type="cartpole",  # cart + first pole; the rewards are the focus
        state_mapping={"cart_position": "cart_pos", "pole_angle": "pole_angle"},
        agents=[AgentSpec(id="cart", label="Cart"), AgentSpec(id="pendulum", label="Pendulum")],
        signal_units={"cart_pos": "m", "pole_angle": "rad", "pendulum_angle": "rad"},
        signal_descriptions={
            "pole_angle": "First pole angle (normalized)",
            "pendulum_angle": "Second pendulum angle (normalized)",
        },
    )

    rng = np.random.default_rng(args.seed)
    created: list[str] = []
    for ep in range(args.num_episodes):
        env.reset(seed=args.seed + ep)
        recorder.start_episode(episode_index=ep + 1, global_step=ep * max_steps)

        terminated = truncated = False
        for frame in range(max_steps):
            jp = to_numpy(robot.data.joint_pos)[0]
            jv = to_numpy(robot.data.joint_vel)[0]
            cart_pos, cart_vel = float(jp[cart_i]), float(jv[cart_i])
            pole_raw, pole_vel = float(jp[pole_i]), float(jv[pole_i])
            pend_raw, pend_vel = float(jp[pend_i]), float(jv[pend_i])
            pole_pos, pend_pos = _wrap(pole_raw), _wrap(pend_raw)

            out_of_bounds = abs(cart_pos) > max_cart or abs(pole_raw) > math.pi / 2
            truncated = (frame == max_steps - 1) and not out_of_bounds
            terminated = bool(out_of_bounds)
            dead = 1.0 if terminated else 0.0

            # Per-agent RAW terms (the env multiplies these by the scales above).
            cart_raw = {
                "alive": 1.0 - dead, "terminated": dead,
                "pole_pos": pole_pos**2, "cart_vel": abs(cart_vel), "pole_vel": abs(pole_vel),
            }
            pend_raw = {
                "alive": 1.0 - dead, "terminated": dead,
                "pendulum_pos": (pole_pos + pend_pos) ** 2, "pendulum_vel": abs(pend_vel),
            }

            a_cart = float(rng.uniform(-1, 1)) * args.action_scale
            a_pend = float(rng.uniform(-1, 1)) * args.action_scale
            recorder.record_frame(
                frame_index=frame, timestamp=frame * control_dt,
                state={"cart_pos": cart_pos, "pole_angle": pole_pos, "pendulum_angle": pend_pos},
                action={"cart_action": a_cart, "pendulum_action": a_pend},
                terminated=terminated, truncated=truncated,
                rewards_by_agent={"cart": cart_raw, "pendulum": pend_raw},
                reward_weights_by_agent={"cart": cart_w, "pendulum": pend_w},
            )
            if terminated or truncated:
                break

            actions = {
                "cart": torch.tensor([[a_cart]], dtype=torch.float32, device=device),
                "pendulum": torch.tensor([[a_pend]], dtype=torch.float32, device=device),
            }
            env.step(actions)

        episode_id = recorder.end_episode(
            reset_reason="out_of_bounds" if terminated else "time_limit"
        )
        if episode_id:
            created.append(episode_id)

    # Summary to a file (Omniverse hijacks stdout).
    with open("/tmp/rlei_cartdp_summary.txt", "w") as fh:
        for eid in created:
            m = recorder.store.load_metadata(eid)
            fh.write(f"{eid}: frames={m.num_frames} team_return={m.episode_return:.2f} "
                     f"agent_returns={m.agent_returns}\n")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
