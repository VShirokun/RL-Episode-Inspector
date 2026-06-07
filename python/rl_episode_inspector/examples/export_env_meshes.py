"""Auto-export exact robot geometry from any Isaac Lab task to GLB.

Boots the task, finds the articulation, and writes one GLB per body containing
its real visual geometry (meshes + primitive shapes), via
``scene_geometry.export_articulation_meshes``. No manual configuration — point it
at a task and it produces the meshes the viewer needs.

    PYTHONPATH=$PWD/python /path/to/isaaclab-env/bin/python \
        -m rl_episode_inspector.examples.export_env_meshes \
        --task Isaac-Reach-Franka-IK-Abs-v0 --out-dir sample_data/reach/assets \
        --robot-key franka
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export exact robot meshes from a task.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--asset-name", default="robot", help="scene articulation key")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--robot-key", required=True, help="subdir under out-dir, e.g. 'franka'")
    parser.add_argument(
        "--frame",
        choices=["physics", "usd"],
        default="physics",
        help=(
            "Which body frame to bake geometry into. 'physics' uses the live "
            "body_pos_w/quat_w (correct when the reset pose matches the authored "
            "USD pose, e.g. Franka). 'usd' uses each body link's authored USD "
            "transform (pose-independent — required when reset != authored pose, "
            "e.g. the AMP humanoid is reset to a mocap frame)."
        ),
    )

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
    import omni.usd
    from isaaclab_tasks.utils import parse_env_cfg

    from rl_episode_inspector.examples.isaaclab_poses import body_world_poses
    from rl_episode_inspector.examples.scene_geometry import export_articulation_meshes
    from rl_episode_inspector.examples.scene_lights import extract_stage_lights

    env = gym.make(args.task, cfg=parse_env_cfg(args.task, num_envs=1))
    env.reset()
    unwrapped = env.unwrapped
    robot = unwrapped.scene[args.asset_name]
    body_names = list(robot.body_names)
    body_poses = body_world_poses(robot) if args.frame == "physics" else None

    # robot prim path for env 0
    root_path = None
    try:
        root_path = list(robot.root_physx_view.prim_paths)[0]
    except Exception:  # noqa: BLE001
        root_path = str(robot.cfg.prim_path).replace("env_.*", "env_0").replace(".*", "0")

    stage = omni.usd.get_context().get_stage()
    diag: list[str] = []
    meshes = export_articulation_meshes(
        stage, root_path, body_names, args.out_dir, args.robot_key,
        body_poses=body_poses, diag=diag,
    )

    # Capture the task's scene lights so the viewer can light the robot the same
    # way the sim does. Written next to the meshes; episode generators load it and
    # pass the lights to the recorder. Best-effort: a failure here must not abort
    # the mesh export (lights have a viewer-side default).
    light_diag: list[str] = []
    try:
        import json

        lights = extract_stage_lights(stage, diag=light_diag)
        if lights:
            lights_path = Path(args.out_dir) / args.robot_key / "lights.json"
            lights_path.parent.mkdir(parents=True, exist_ok=True)
            lights_path.write_text(json.dumps(lights, indent=2))
    except Exception as exc:  # noqa: BLE001
        light_diag.append(f"light extraction failed: {exc!r}")

    with open("/tmp/rlei_mesh_export.txt", "w") as fh:
        fh.write(f"task={args.task} root={root_path}\n")
        fh.write(f"bodies={body_names}\n")
        fh.write(f"exported {len(meshes)}/{len(body_names)} bodies:\n")
        for b in body_names:
            fh.write(f"  {b}: {meshes.get(b, 'NO GEOMETRY')}\n")
        fh.write("DIAGNOSTICS:\n")
        for d in diag:
            fh.write(f"  {d}\n")
        fh.write("LIGHTS:\n")
        for d in light_diag:
            fh.write(f"  {d}\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
