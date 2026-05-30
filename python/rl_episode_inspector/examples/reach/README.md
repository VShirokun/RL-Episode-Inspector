# Franka Reach example (sparse-reward, longer episodes)

A second, richer Isaac Lab demo task than Cartpole. The Franka end-effector
visits a sequence of 3D targets; reaching each one fires a **sparse** reward.

| File | Purpose |
|------|---------|
| `reward.py` | Reward decomposition incl. the sparse `target_reached` term (pure, CI-tested). |
| `targets.py` | Waypoint sequence + `WaypointTracker` + EE approach low-pass (pure, CI-tested). |
| `generate_demo_episodes.py` | **Runnable** — launches Isaac Sim, drives the EE via IK to each target, records. |
| `configs/reward.yaml` | Reward-term weights. |

## Why this task

- **Longer episodes** — 30 s (≈900 control steps at 1/30 s) vs Cartpole's ~5 s.
- **Sparse reward** — `target_reached` is 0 on almost every frame, +10 (weighted)
  only when the EE enters a target zone. `reward_cumulative` becomes a staircase —
  a good stress test for chart rendering and seeking to rare events.
- **Easy to drive** — `Isaac-Reach-Franka-IK-Abs-v0` takes an absolute EE pose as
  the action and Isaac's differential IK does the joint motion. No hand-tuned
  joint controller.
- **New 3D viewer** (`reach3d`) — EE sphere + current-target marker on a plane,
  exercising the tool's generic per-task viewer design.

## Quick run

```bash
PYTHONPATH=$PWD/python /path/to/isaaclab-env/bin/python \
    python/rl_episode_inspector/examples/reach/generate_demo_episodes.py \
    --output-dir sample_data/reach/episodes --num-episodes 10 --seed 42

rl-episode-inspector serve --episodes-dir sample_data/reach/episodes   # then open the viewer
```

Task ID `Isaac-Reach-Franka-IK-Abs-v0`; verified with Isaac Lab 2.3.2 / Isaac Sim
6.0.0 / Python 3.12. See [`docs/isaac_lab_integration.md`](../../../../docs/isaac_lab_integration.md).

## Reward terms

`reach_progress` (−distance, dense), `action_smoothness` (−speed², dense),
`time_penalty` (−1/step, dense), `target_reached` (**sparse**, +1 raw on reach),
plus `reward_step_total`, `reward_cumulative`, `episode_return`.
