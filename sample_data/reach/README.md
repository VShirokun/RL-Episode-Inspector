# Franka Reach demo episodes (committed)

Unlike the Cartpole sample (which is regenerated locally and git-ignored), these
**real Isaac Lab Franka Reach episodes are committed to the repo** so the viewer
works on any machine — **no Isaac Sim / GPU required to view them**.

```
episodes/
  reach_000001/ … reach_000008/
    metadata.json
    frames.parquet
```

- Task: `Isaac-Reach-Franka-IK-Abs-v0` (real physics), recorded with
  `examples/reach/generate_demo_episodes.py` (seed 42, 25 s episodes).
- 8 episodes, `task_source: isaac_lab`. A mix of successes (all 8 targets
  reached → `terminated`/`all_targets_reached`) and timeouts (`truncated`).
- Demonstrates **longer episodes** and a **sparse** `target_reached` reward
  (cumulative reward is a staircase).
- **Full-body capture**: every one of the 11 Franka rigid bodies has its pose
  recorded each frame (`pose_<body>_*` columns), so the `articulation3d` viewer
  replays the whole arm moving — not just the end-effector.

## View them (no Isaac needed)

```bash
make install            # once
make frontend-install   # once
make serve-reach        # backend on :8000 serving these episodes
make frontend-dev       # in another terminal -> http://localhost:3000
```

Then press **Best** to load the successful episode and play. To regenerate /
extend the set (needs Isaac Lab): `make generate-reach-demo`.

Total size is ~0.6 MB; these are small, clean fixtures with no private data.
