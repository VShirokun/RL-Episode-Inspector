"""Record real Isaac Lab Cartpole episodes for the viewer.

Drives the ``Isaac-Cartpole-Direct-v0`` task with a simple PD controller whose
skill is varied per episode (plus a per-episode disturbance), so the recorded
set has a spread of returns. Real physics + real joint state; the six-term
reward decomposition (see ``reward.py``) is computed explicitly so every term is
available at every frame.

Run under Isaac Lab's Python, e.g.::

    /path/to/IsaacLab/isaaclab.sh -p \
        python/rl_episode_inspector/examples/cartpole/generate_demo_episodes.py \
        --output-dir sample_data/cartpole/episodes --num-episodes 12 --seed 42

Verified with Isaac Lab 2.3.2 / Isaac Sim 6.0.0 / Python 3.12 (see
docs/isaac_lab_integration.md).
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Record real Isaac Lab Cartpole episodes.")
    parser.add_argument("--output-dir", default="sample_data/cartpole/episodes")
    parser.add_argument("--num-episodes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env-id", type=int, default=0, help="env index recorded into metadata")
    parser.add_argument("--task", default="Isaac-Cartpole-Direct-v0")
    parser.add_argument("--reward-config", default=None, help="YAML of reward weights")
    parser.add_argument("--max-steps", type=int, default=2000, help="safety cap per episode")
    parser.add_argument(
        "--control-sign", type=float, default=1.0,
        help="multiplier on the controller output; flip to -1 if signs are inverted",
    )
    parser.add_argument(
        "--probe-sign", action="store_true",
        help="diagnostic: report which control sign keeps the pole upright, then exit",
    )

    # Isaac Lab's AppLauncher contributes --headless and friends.
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True  # this is a batch recorder; never open a window
    if args.enable_cameras is None:
        args.enable_cameras = False

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Imports that require the app to be running must come after launch.
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401  (registers the gym ids)
    import numpy as np
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    from rl_episode_inspector.examples.cartpole.controller import Gains, balance_action
    from rl_episode_inspector.examples.cartpole.reward import (
        compute_reward_terms,
        load_reward_weights,
    )
    from rl_episode_inspector.recorder import EpisodeRecorder

    weights = load_reward_weights(args.reward_config)

    env_cfg = parse_env_cfg(args.task, num_envs=1)
    env = gym.make(args.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    device = unwrapped.device
    action_scale = float(getattr(unwrapped.cfg, "action_scale", 100.0))
    # control step = physics dt * decimation; prefer the env's own value.
    control_dt = float(
        getattr(unwrapped, "step_dt", None)
        or unwrapped.cfg.sim.dt * unwrapped.cfg.decimation
    )

    robot = getattr(unwrapped, "cartpole", None)
    if robot is None:
        robot = unwrapped.scene["cartpole"]
    cart_ids, _ = robot.find_joints("slider_to_cart")
    pole_ids, _ = robot.find_joints("cart_to_pole")
    cart_i, pole_i = cart_ids[0], pole_ids[0]

    recorder = EpisodeRecorder(
        output_dir=args.output_dir,
        task_name="Cartpole",
        dt=control_dt,
        env_id=args.env_id,
        run_id=f"isaaclab_cartpole_seed{args.seed}",
        task_source="isaac_lab",
        episode_id_prefix="cartpole",
        viewer_type="cartpole",
        state_mapping={"cart_position": "cart_position", "pole_angle": "pole_angle"},
        signal_units={
            "cart_position": "m", "cart_velocity": "m/s",
            "pole_angle": "rad", "pole_angular_velocity": "rad/s",
            "action_cart_force": "N",
        },
        signal_descriptions={
            "cart_position": "Cart position along the track",
            "pole_angle": "Pole angle from vertical",
        },
    )

    rng = np.random.default_rng(args.seed)
    env.reset(seed=args.seed)

    def _to_numpy(x):
        # Isaac Lab 2.3 with the Newton backend exposes joint state as warp
        # arrays; older/torch backends expose torch tensors. Handle both.
        if hasattr(x, "numpy") and "warp" in type(x).__module__:
            return x.numpy()
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def read_state() -> tuple[float, float, float, float]:
        jp = _to_numpy(robot.data.joint_pos)
        jv = _to_numpy(robot.data.joint_vel)
        return (
            float(jp[0, cart_i]),
            float(jv[0, cart_i]),
            float(jp[0, pole_i]),
            float(jv[0, pole_i]),
        )

    def step_u(u_norm: float):
        u_norm = float(np.clip(u_norm, -1.0, 1.0))
        action = torch.tensor([[u_norm]], dtype=torch.float32, device=device)
        return env.step(action)

    if args.probe_sign:
        # Determine the correct control sign empirically: apply +/- control for a
        # short horizon and see which keeps the pole closer to upright.
        lines = []
        for sign in (1.0, -1.0):
            env.reset(seed=args.seed)
            survived, max_abs = 0, 0.0
            for _ in range(80):
                cp, cv, pa, pv = read_state()
                _, _, term_t, trunc_t, _ = step_u(sign * balance_action(cp, cv, pa, pv, Gains()))
                max_abs = max(max_abs, abs(pa))
                survived += 1
                if bool(term_t[0].item()) or bool(trunc_t[0].item()):
                    break
            lines.append(f"sign={sign:+.0f}: survived {survived}/80 steps, max|pole|={max_abs:.3f}")
        # stdout is hijacked by Omniverse, so write the result to a file.
        with open("/tmp/rlei_probe.txt", "w") as fh:
            fh.write("\n".join(lines) + "\n")
        env.close()
        simulation_app.close()
        return

    created: list[str] = []
    for ep in range(args.num_episodes):
        # skill in ~[0.4, 1.25]; lower -> weaker control + bigger disturbance, so
        # weak runs fall early while skilled runs survive to max length (truncated).
        skill = 0.4 + 0.85 * (ep + 0.5) / args.num_episodes
        # Mild cart centering balances pole-holding against staying on the track.
        # With a hand-tuned PD controller, episodes still end (pole tilt or the
        # ±3 m cart bound) rather than surviving the full 5 s — realistic, and it
        # yields a clear spread of returns for best/median/worst.
        gains = Gains(kp=14.0 * skill, kd=2.4 * skill, kx=1.2, kxd=1.0)
        disturbance = 0.06 * max(0.04, 1.2 - skill)

        recorder.start_episode(
            episode_index=ep + 1, global_step=ep * args.max_steps, seed=args.seed + ep
        )
        frame = 0
        while frame < args.max_steps:
            cart_pos, cart_vel, pole_angle, pole_vel = read_state()
            u = args.control_sign * balance_action(cart_pos, cart_vel, pole_angle, pole_vel, gains)
            u += float(rng.normal(0.0, disturbance))
            u = float(np.clip(u, -1.0, 1.0))

            action = torch.tensor([[u]], dtype=torch.float32, device=device)
            _, _, terminated_t, truncated_t, _ = env.step(action)
            terminated = bool(terminated_t[0].item())
            truncated = bool(truncated_t[0].item())

            raw = compute_reward_terms(
                cart_pos, cart_vel, pole_angle, pole_vel, u, terminated
            )
            recorder.record_frame(
                frame_index=frame,
                timestamp=frame * control_dt,
                state={
                    "cart_position": cart_pos,
                    "cart_velocity": cart_vel,
                    "pole_angle": pole_angle,
                    "pole_angular_velocity": pole_vel,
                },
                action={"action_cart_force": u * action_scale},
                rewards_raw=raw,
                reward_weights=weights,
                terminated=terminated,
                truncated=truncated,
            )
            frame += 1
            if terminated or truncated:
                break

        reset_reason = (
            "pole_fell_or_out_of_bounds" if terminated else "max_episode_length"
        )
        episode_id = recorder.end_episode(reset_reason=reset_reason)
        if episode_id:
            created.append(episode_id)
            print(f"[cartpole] recorded {episode_id}: {frame} frames, "
                  f"return={recorder.store.load_metadata(episode_id).episode_return:.2f}, "
                  f"{'terminated' if terminated else 'truncated'}")

    print(f"[cartpole] done: {len(created)} episodes in {args.output_dir}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
