"""Dump physics-body vs authored-USD frames for the AMP humanoid (diagnostic)."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Isaac-Humanoid-AMP-Walk-Direct-v0")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    if args.enable_cameras is None:
        args.enable_cameras = False
    app = AppLauncher(args).app

    import gymnasium as gym
    import numpy as np
    import omni.usd
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from pxr import Gf, UsdGeom

    from rl_episode_inspector.examples.isaaclab_poses import to_numpy

    env = gym.make(args.task, cfg=parse_env_cfg(args.task, num_envs=1))
    env.reset()
    u = env.unwrapped
    robot = u.scene["robot"]

    # Force DEFAULT pose so physics == authored USD pose.
    import torch

    dev = robot.device
    def T(x):
        return torch.as_tensor(to_numpy(x), device=dev)

    root = T(robot.data.default_root_state).clone()
    root[:, :3] += T(u.scene.env_origins)
    robot.write_root_pose_to_sim(root[:, :7])
    robot.write_root_velocity_to_sim(root[:, 7:])
    robot.write_joint_state_to_sim(T(robot.data.default_joint_pos),
                                   T(robot.data.default_joint_vel))
    u.sim.step(render=False)
    robot.update(u.sim.get_physics_dt())

    names = list(robot.data.body_names)
    pos = to_numpy(robot.data.body_pos_w)[0]
    qraw = robot.data.body_quat_w
    quat = to_numpy(qraw)[0]
    if "warp" in type(qraw).__module__:
        quat = quat[:, [3, 0, 1, 2]]

    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.XformCache()
    root_path = list(robot.root_physx_view.prim_paths)[0]

    def mat_to_quat(m):
        r = Gf.Matrix4d(m).GetOrthonormalized().ExtractRotationQuat()
        im = r.GetImaginary()
        return np.array([r.GetReal(), im[0], im[1], im[2]])

    def find_prim(body):
        from pxr import Usd
        rp = stage.GetPrimAtPath(root_path)
        for p in Usd.PrimRange(rp, Usd.TraverseInstanceProxies()):
            if p.GetName() == body:
                return p
        return None

    lines = []
    for tgt in ["pelvis", "right_upper_arm", "right_thigh", "right_shin", "right_foot"]:
        i = names.index(tgt)
        link = find_prim(tgt)
        link_w = cache.GetLocalToWorldTransform(link)
        link_q = mat_to_quat(link_w)
        link_t = np.array(link_w.ExtractTranslation())
        # capsule/geom under this link
        from pxr import Usd
        geomq = None
        gtype = None
        for p in Usd.PrimRange(link, Usd.TraverseInstanceProxies()):
            if p.GetTypeName() in ("Capsule", "Cube", "Sphere", "Mesh"):
                gw = cache.GetLocalToWorldTransform(p)
                geomq = mat_to_quat(gw)
                gtype = p.GetTypeName()
                axis = None
                if gtype in ("Capsule", "Cylinder", "Cone"):
                    axis = p.GetAttribute("axis").Get()
                break
        lines.append(f"{tgt}: phys_pos={pos[i].round(3)} phys_quat={quat[i].round(3)}")
        lines.append(f"   authored link_t={link_t.round(3)} link_quat={link_q.round(3)}")
        lines.append(f"   geom type={gtype} axis={axis} geom_quat={None if geomq is None else geomq.round(3)}")

    out = "\n".join(lines)
    print(out)
    with open("/tmp/rlei_frames.txt", "w") as f:
        f.write(out + "\n")

    env.close()
    app.close()


if __name__ == "__main__":
    main()
