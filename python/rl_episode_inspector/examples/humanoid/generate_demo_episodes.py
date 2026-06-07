"""Record humanoid episodes from Isaac Lab's AMP reference mocap (walk/run/dance).

Offline + fast (no simulator): loads the ``.npz`` mocap clips, which carry the
correct full-body world positions (a real standing/walking figure) plus per-body
world rotations in the PhysX body frame. The exact meshes are baked into that
same link frame (``export_env_meshes --frame usd``), so the viewer applies the
recorded quaternion directly — an exact replay of the reference motion with the
real geometry.

The exact meshes come from a one-time ``export_env_meshes --frame usd`` run on the
AMP env (committed under sample_data/humanoid/assets/humanoid28). Runs under the
normal project venv:

    python -m rl_episode_inspector.examples.humanoid.generate_demo_episodes \
        --output-dir sample_data/humanoid/episodes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rl_episode_inspector.examples.humanoid.motion_replay import (
    DEFAULT_REWARD_WEIGHTS,
    frame_metrics,
    parents_for,
)
from rl_episode_inspector.recorder import EpisodeRecorder
from rl_episode_inspector.storage import LightSpec

# Scene lights captured at mesh-export time (sample_data/humanoid/assets), so the
# replay is lit like the AMP task. Absent => the viewer uses its default rig.
DEFAULT_LIGHTS_PATH = "sample_data/humanoid/assets/humanoid28/lights.json"


def load_lights(path: str) -> list[LightSpec]:
    p = Path(path)
    if not p.exists():
        return []
    return [LightSpec(**d) for d in json.loads(p.read_text())]

DEFAULT_MOTIONS_DIR = (
    "/mnt/nvme2n1/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/"
    "direct/humanoid_amp/motions"
)
FOOT_GROUND_Z = 0.08


def main() -> None:
    parser = argparse.ArgumentParser(description="Record humanoid mocap replay episodes.")
    parser.add_argument("--motions-dir", default=DEFAULT_MOTIONS_DIR)
    parser.add_argument("--output-dir", default="sample_data/humanoid/episodes")
    parser.add_argument("--clips", default="walk,run,dance")
    parser.add_argument("--lights", default=DEFAULT_LIGHTS_PATH,
                        help="lights.json captured at export time (optional)")
    args = parser.parse_args()

    motions_dir = Path(args.motions_dir)
    clips = [c.strip() for c in args.clips.split(",") if c.strip()]
    lights = load_lights(args.lights)
    if lights:
        print(f"using {len(lights)} scene light(s) from {args.lights}")

    created: list[str] = []
    for ep_index, clip in enumerate(clips, start=1):
        npz_path = motions_dir / f"humanoid_{clip}.npz"
        if not npz_path.exists():
            print(f"SKIP {clip}: {npz_path} not found")
            continue
        data = np.load(npz_path)
        body_names = [str(n) for n in data["body_names"]]
        positions = data["body_positions"]  # (F, B, 3) world, z-up
        rotations = data["body_rotations"]  # (F, B, 4) wxyz (mocap frame)
        fps = int(data["fps"])
        dt = 1.0 / fps
        n_frames = positions.shape[0]

        parents = parents_for(body_names)
        pelvis = body_names.index("pelvis") if "pelvis" in body_names else 0
        foot_idx = [body_names.index(b) for b in ("right_foot", "left_foot") if b in body_names]

        recorder = EpisodeRecorder(
            output_dir=args.output_dir,
            task_name="Humanoid",
            dt=dt,
            run_id=f"amp_{clip}",
            task_source="amp_motion",
            episode_id_prefix="humanoid",
            viewer_type="articulation3d",
            up_axis="z",
            lights=lights,
            signal_units={"root_height": "m", "forward_velocity": "m/s"},
            signal_descriptions={
                "root_height": "Pelvis height above ground",
                "forward_velocity": "Pelvis forward (x) velocity",
                "footstep": "Sparse: a foot just touched the ground",
            },
        )
        recorder.register_bodies(
            body_names, parents, meshes=[f"humanoid28/{name}.glb" for name in body_names]
        )
        recorder.start_episode(episode_index=ep_index, global_step=0)

        prev_root: tuple[float, float, float] | None = None
        prev_foot_z: dict[int, float | None] = dict.fromkeys(foot_idx)
        for f in range(n_frames):
            root = (
                float(positions[f, pelvis, 0]),
                float(positions[f, pelvis, 1]),
                float(positions[f, pelvis, 2]),
            )
            metrics = frame_metrics(root, prev_root, dt)
            prev_root = root

            strike = False
            for i in foot_idx:
                z = float(positions[f, i, 2])
                pz = prev_foot_z[i]
                if pz is not None and pz > FOOT_GROUND_Z >= z:
                    strike = True
                prev_foot_z[i] = z

            poses = {}
            for b, name in enumerate(body_names):
                px, py, pz = (float(v) for v in positions[f, b])
                qw, qx, qy, qz = (float(v) for v in rotations[f, b])
                poses[name] = (px, py, pz, qw, qx, qy, qz)

            recorder.record_frame(
                frame_index=f,
                timestamp=f * dt,
                state={"root_height": root[2], "forward_velocity": metrics["forward_velocity"]},
                action={},
                rewards_raw={**metrics, "footstep": 1.0 if strike else 0.0},
                reward_weights=DEFAULT_REWARD_WEIGHTS,
                terminated=False,
                truncated=(f == n_frames - 1),
                poses=poses,
            )
        episode_id = recorder.end_episode(reset_reason="motion_end")
        if episode_id:
            created.append(episode_id)
            print(f"recorded {episode_id}: {clip}, {n_frames} frames")

    print(f"done: {len(created)} humanoid episodes in {args.output_dir}")


if __name__ == "__main__":
    main()
