"""Record real Isaac Lab Franka Reach episodes (visit a sequence of 3D targets).

Uses ``Isaac-Reach-Franka-IK-Abs-v0``: the action is an absolute end-effector
pose and Isaac Lab's differential IK solves the joint motion, so "control" is
just feeding the next target pose (no hand-tuned joint controller needed). Per
episode the EE visits a fixed waypoint sequence; a per-episode "skill" sets how
fast the commanded pose low-passes toward each target, giving a spread of
returns (fast runs reach all targets = success/terminated; slow runs run out of
time = truncated).

The reward decomposition includes a **sparse** ``target_reached`` term (see
reward.py). Run under Isaac Lab's Python, e.g.::

    PYTHONPATH=$PWD/python /path/to/isaaclab-env/bin/python \
        python/rl_episode_inspector/examples/reach/generate_demo_episodes.py \
        --output-dir sample_data/reach/episodes --num-episodes 10 --seed 42

Verified with Isaac Lab 2.3.2 / Isaac Sim 6.0.0 / Python 3.12.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Record real Isaac Lab Franka Reach episodes.")
    parser.add_argument("--output-dir", default="sample_data/reach/episodes")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env-id", type=int, default=0)
    parser.add_argument("--task", default="Isaac-Reach-Franka-IK-Abs-v0")
    parser.add_argument("--reward-config", default=None)
    parser.add_argument("--episode-length", type=float, default=30.0, help="seconds")

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
    import omni.usd
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    from rl_episode_inspector.examples.isaaclab_poses import (
        body_structure,
        read_body_poses,
        to_numpy,
    )
    from rl_episode_inspector.examples.reach.reward import compute_reward_terms, load_reward_weights
    from rl_episode_inspector.examples.reach.targets import (
        DEFAULT_WAYPOINTS,
        REACH_THRESHOLD,
        WaypointTracker,
        approach,
        distance,
    )
    from rl_episode_inspector.examples.scene_geometry import export_articulation_meshes
    from rl_episode_inspector.recorder import EpisodeRecorder
    from rl_episode_inspector.storage import MarkerSpec

    weights = load_reward_weights(args.reward_config)

    env_cfg = parse_env_cfg(args.task, num_envs=1)
    env_cfg.episode_length_s = args.episode_length  # longer episodes
    env = gym.make(args.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    device = unwrapped.device
    control_dt = float(
        getattr(unwrapped, "step_dt", None)
        or unwrapped.cfg.sim.dt * unwrapped.cfg.decimation
    )
    max_steps = int(args.episode_length / control_dt)

    robot = unwrapped.scene["robot"]
    ee_ids, _ = robot.find_bodies("panda_hand")
    ee_i = ee_ids[0]

    # The IK action's body_offset: the controlled point is 0.107 m down the
    # panda_hand local z-axis (the fingertip), not panda_hand's own frame. We
    # must measure/record that same point or the EE never "reaches" the target.
    EE_OFFSET = np.array([0.0, 0.0, 0.107])
    env_origin = to_numpy(unwrapped.scene.env_origins)[0]

    # Auto-capture structure for ALL of the robot's rigid bodies.
    body_names, body_parents = body_structure(robot)

    def _rotate(quat_wxyz, v):
        # rotate vector v by quaternion (w, x, y, z)
        w, x, y, z = quat_wxyz
        q = np.array([x, y, z])
        return v + 2.0 * np.cross(q, np.cross(q, v) + w * v)

    def read_ee() -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        # Controlled fingertip pose in the env-local (base) frame.
        pos_w = to_numpy(robot.data.body_pos_w)[0, ee_i]
        quat_w = to_numpy(robot.data.body_quat_w)[0, ee_i]  # (w, x, y, z)
        tip_w = pos_w + _rotate(quat_w, EE_OFFSET)
        origin = env_origin
        pos = (
            float(tip_w[0] - origin[0]),
            float(tip_w[1] - origin[1]),
            float(tip_w[2] - origin[2]),
        )
        quat = tuple(float(v) for v in quat_w)
        return pos, quat  # type: ignore[return-value]

    recorder = EpisodeRecorder(
        output_dir=args.output_dir,
        task_name="FrankaReach",
        dt=control_dt,
        env_id=args.env_id,
        run_id=f"isaaclab_reach_seed{args.seed}",
        task_source="isaac_lab",
        episode_id_prefix="reach",
        viewer_type="articulation3d",  # full robot, not just the EE
        state_mapping={
            "ee_x": "ee_x", "ee_y": "ee_y", "ee_z": "ee_z",
            "target_x": "target_x", "target_y": "target_y", "target_z": "target_z",
        },
        # Render the current target as a point marker alongside the robot bodies.
        markers=[
            MarkerSpec(
                name="target", pos=["target_x", "target_y", "target_z"], color="#4ecb71"
            )
        ],
        up_axis="z",  # Isaac world frame is z-up
        signal_units={
            "ee_x": "m", "ee_y": "m", "ee_z": "m",
            "target_x": "m", "target_y": "m", "target_z": "m",
            "distance_to_target": "m", "ee_speed": "m/s",
        },
        signal_descriptions={
            "ee_x": "End-effector x (base frame)",
            "distance_to_target": "Distance from EE to the current target",
            "targets_reached": "Cumulative number of targets reached",
        },
    )
    # Automatically export the robot's EXACT visual geometry (meshes/primitives)
    # to the assets dir the backend serves, and wire each body's mesh path. No
    # manual step: recording the task produces the meshes too.
    from pathlib import Path

    assets_dir = Path(args.output_dir).parent / "assets"
    try:
        root_path = list(robot.root_physx_view.prim_paths)[0]
    except Exception:  # noqa: BLE001
        root_path = str(robot.cfg.prim_path).replace("env_.*", "env_0").replace(".*", "0")
    mesh_map = export_articulation_meshes(
        omni.usd.get_context().get_stage(), root_path, body_names, assets_dir, "franka"
    )
    body_meshes = [mesh_map.get(name) for name in body_names]
    recorder.register_bodies(body_names, body_parents, meshes=body_meshes)

    rng = np.random.default_rng(args.seed)
    created: list[str] = []

    # One fixed, known-reachable EE orientation (the default downward-ish pose),
    # reused for every episode. Holding a consistent orientation keeps all
    # workspace waypoints reachable; reading a fresh per-reset orientation made
    # some episodes get stuck.
    env.reset(seed=args.seed)
    _, hold_quat = read_ee()

    for ep in range(args.num_episodes):
        env.reset(seed=args.seed + ep)
        ee_pos, _ = read_ee()
        cmd = ee_pos

        skill = 0.4 + 0.6 * (ep + 0.5) / args.num_episodes
        move_gain = 0.06 + 0.24 * skill          # higher skill -> faster approach
        noise = 0.004 * max(0.2, 1.2 - skill)    # small command jitter
        tracker = WaypointTracker(DEFAULT_WAYPOINTS, threshold=REACH_THRESHOLD)

        recorder.start_episode(
            episode_index=ep + 1, global_step=ep * max_steps, seed=args.seed + ep
        )
        prev_pos = ee_pos
        terminated = truncated = False
        for frame in range(max_steps):
            target = tracker.current_target
            cmd = approach(cmd, target, move_gain)
            cmd = (
                cmd[0] + float(rng.normal(0.0, noise)),
                cmd[1] + float(rng.normal(0.0, noise)),
                cmd[2] + float(rng.normal(0.0, noise)),
            )

            action = torch.tensor(
                [[cmd[0], cmd[1], cmd[2], *hold_quat]], dtype=torch.float32, device=device
            )
            env.step(action)

            ee_pos, _ = read_ee()
            dist = distance(ee_pos, target)
            ee_speed = distance(ee_pos, prev_pos) / control_dt
            prev_pos = ee_pos

            reached_now = tracker.update(ee_pos)
            terminated = tracker.done                      # success: all targets reached
            truncated = (not terminated) and (frame == max_steps - 1)

            raw = compute_reward_terms(dist, ee_speed, reached_now)
            recorder.record_frame(
                frame_index=frame,
                timestamp=frame * control_dt,
                state={
                    "ee_x": ee_pos[0], "ee_y": ee_pos[1], "ee_z": ee_pos[2],
                    "target_x": target[0], "target_y": target[1], "target_z": target[2],
                    "distance_to_target": dist,
                    "targets_reached": float(tracker.reached_count),
                },
                action={"cmd_x": cmd[0], "cmd_y": cmd[1], "cmd_z": cmd[2], "ee_speed": ee_speed},
                rewards_raw=raw,
                reward_weights=weights,
                terminated=terminated,
                truncated=truncated,
                poses=read_body_poses(robot, env_origin),  # all robot bodies
            )
            if terminated or truncated:
                break

        reset_reason = "all_targets_reached" if terminated else "time_limit"
        episode_id = recorder.end_episode(reset_reason=reset_reason)
        if episode_id:
            created.append(episode_id)

    # Summary to a file (Omniverse hijacks stdout).
    with open("/tmp/rlei_reach_summary.txt", "w") as fh:
        for eid in created:
            m = recorder.store.load_metadata(eid)
            fh.write(f"{eid}: frames={m.num_frames} return={m.episode_return:.2f} "
                     f"{'success' if m.terminated else 'timeout'}\n")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
