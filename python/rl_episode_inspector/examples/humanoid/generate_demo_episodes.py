"""Record humanoid episodes by replaying AMP mocap THROUGH the Isaac AMP env.

Earlier this replayed the .npz mocap offline. The mocap's body_rotations use a
different per-body frame convention than the robot's USD geometry, so applying
them to the exported meshes rotated every limb 90° off. Instead we now drive the
real Isaac Humanoid through each mocap clip (set joint + root state per frame)
and record the genuine body_pos_w/body_quat_w — the same convention as the
exported meshes — so the real geometry orients correctly. Bonus: it also exports
the exact humanoid meshes in the same run.

Run under Isaac Lab's Python:
    PYTHONPATH=$PWD/python /path/to/isaaclab-env/bin/python \
        -m rl_episode_inspector.examples.humanoid.generate_demo_episodes \
        --output-dir sample_data/humanoid/episodes
"""

from __future__ import annotations

import argparse

FOOT_GROUND_Z = 0.08


def main() -> None:
    parser = argparse.ArgumentParser(description="Record humanoid mocap replay (through sim).")
    parser.add_argument("--output-dir", default="sample_data/humanoid/episodes")
    parser.add_argument("--task", default="Isaac-Humanoid-AMP-Walk-Direct-v0")
    parser.add_argument("--clips", default="walk,run,dance")

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    if args.enable_cameras is None:
        args.enable_cameras = False

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from pathlib import Path

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import isaaclab_tasks.direct.humanoid_amp.motions as motions_pkg
    import numpy as np
    import omni.usd
    import torch
    from isaaclab_tasks.direct.humanoid_amp.motions import MotionLoader
    from isaaclab_tasks.utils import parse_env_cfg

    from rl_episode_inspector.examples.humanoid.motion_replay import (
        DEFAULT_REWARD_WEIGHTS,
        frame_metrics,
        parents_for,
    )
    from rl_episode_inspector.examples.isaaclab_poses import body_world_poses
    from rl_episode_inspector.examples.scene_geometry import export_articulation_meshes
    from rl_episode_inspector.recorder import EpisodeRecorder

    env = gym.make(args.task, cfg=parse_env_cfg(args.task, num_envs=1))
    env.reset()
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    sim = unwrapped.sim
    device = unwrapped.device
    # num_envs=1 sits at the world origin, so we record world poses directly.

    body_names = list(robot.body_names)
    parents = parents_for(body_names)
    motions_dir = Path(motions_pkg.__file__).parent

    # Export the exact robot geometry once (meshes + primitive capsules/spheres).
    try:
        root_path = list(robot.root_physx_view.prim_paths)[0]
    except Exception:  # noqa: BLE001
        root_path = "/World/envs/env_0/Robot"
    assets_dir = Path(args.output_dir).parent / "assets"
    mesh_map = export_articulation_meshes(
        omni.usd.get_context().get_stage(), root_path, body_names, assets_dir, "humanoid28",
        body_poses=body_world_poses(robot),
    )

    def to_np(x):
        return x.numpy() if "warp" in type(x).__module__ else x.detach().cpu().numpy()

    def read_poses() -> dict[str, tuple[float, ...]]:
        pos = to_np(robot.data.body_pos_w)[0]
        quat_raw = robot.data.body_quat_w
        quat = to_np(quat_raw)[0]  # (num_bodies, 4)
        if "warp" in type(quat_raw).__module__:
            quat = quat[:, [3, 0, 1, 2]]  # warp xyzw -> wxyz (reorder components)
        out: dict[str, tuple[float, ...]] = {}
        for i, name in enumerate(body_names):
            p = pos[i]
            out[name] = (
                float(p[0]), float(p[1]), float(p[2]),
                float(quat[i][0]), float(quat[i][1]), float(quat[i][2]), float(quat[i][3]),
            )
        return out

    clips = [c.strip() for c in args.clips.split(",") if c.strip()]
    created: list[str] = []
    selfcheck: list[str] = []

    for ep_index, clip in enumerate(clips, start=1):
        npz = motions_dir / f"humanoid_{clip}.npz"
        if not npz.exists():
            continue
        ml = MotionLoader(motion_file=str(npz), device=device)
        dof_idx = ml.get_dof_index(robot.data.joint_names)
        torso_idx = ml.get_body_index(["torso"])[0]
        n_frames = ml.num_frames
        dt = ml.dt
        clip_pos = ml.body_positions  # (F, B, 3) tensor
        clip_rot = ml.body_rotations  # (F, B, 4) wxyz

        recorder = EpisodeRecorder(
            output_dir=args.output_dir, task_name="Humanoid", dt=float(dt),
            run_id=f"amp_{clip}", task_source="isaac_lab_amp",
            episode_id_prefix="humanoid", viewer_type="articulation3d", up_axis="z",
            signal_units={"root_height": "m", "forward_velocity": "m/s"},
            signal_descriptions={"footstep": "Sparse: a foot just touched the ground"},
        )
        recorder.register_bodies(
            body_names, parents, meshes=[mesh_map.get(n) for n in body_names]
        )
        recorder.start_episode(episode_index=ep_index, global_step=0)

        prev_root = None
        foot_idx = [body_names.index(b) for b in ("right_foot", "left_foot") if b in body_names]
        prev_foot_z = dict.fromkeys(foot_idx)

        for fr in range(n_frames):
            # set the robot to this mocap frame (root = torso, like the AMP env reset)
            root_state = torch.zeros((1, 13), device=device)
            root_state[:, 0:3] = clip_pos[fr, torso_idx]
            root_state[:, 3:7] = clip_rot[fr, torso_idx]  # torso orientation = root
            dof_pos = clip_pos.new_zeros((1, len(robot.data.joint_names)))
            dof_pos[0] = ml.dof_positions[fr, dof_idx]
            dof_vel = torch.zeros_like(dof_pos)
            eids = torch.tensor([0], device=device)
            robot.write_root_link_pose_to_sim(root_state[:, :7], eids)
            robot.write_root_com_velocity_to_sim(root_state[:, 7:], eids)
            robot.write_joint_state_to_sim(dof_pos, dof_vel, None, eids)
            sim.step(render=False)
            robot.update(dt)

            poses = read_poses()
            rb = poses.get("pelvis") or poses["torso"]
            root = (rb[0], rb[1], rb[2])
            metrics = frame_metrics(root, prev_root, dt)
            prev_root = root
            strike = False
            for i in foot_idx:
                z = poses[body_names[i]][2]
                pz = prev_foot_z[i]
                if pz is not None and pz > FOOT_GROUND_Z >= z:
                    strike = True
                prev_foot_z[i] = z

            if fr == 0 and ep_index == 1:
                # self-check: read torso pos should match the mocap torso target
                tgt = (clip_pos[0, torso_idx]).detach().cpu().numpy()
                got = np.array(poses["torso"][:3])
                selfcheck.append(f"torso mocap={np.round(tgt,3)} read={np.round(got,3)}")

            recorder.record_frame(
                frame_index=fr, timestamp=fr * dt,
                state={"root_height": root[2], "forward_velocity": metrics["forward_velocity"]},
                action={},
                rewards_raw={**metrics, "footstep": 1.0 if strike else 0.0},
                reward_weights=DEFAULT_REWARD_WEIGHTS,
                terminated=False, truncated=(fr == n_frames - 1),
                poses=poses,
            )
        eid = recorder.end_episode(reset_reason="motion_end")
        if eid:
            created.append(f"{eid}:{clip}:{n_frames}f")

    with open("/tmp/rlei_humanoid_gen.txt", "w") as fh:
        fh.write(f"meshes={len(mesh_map)} bodies; episodes={created}\n")
        fh.write("selfcheck (torso read vs mocap target should match):\n")
        for s in selfcheck:
            fh.write(f"  {s}\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
