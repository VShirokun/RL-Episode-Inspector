# Cartpole example (Isaac Lab task adapter)

This is the only task-specific code in the project. The core never imports it.

| File | Purpose |
|------|---------|
| `reward.py` | Pure six-term reward decomposition + weight loading (CI-tested, no Isaac Lab). |
| `controller.py` | Tiny PD balancing controller (pure, CI-tested). |
| `generate_demo_episodes.py` | **Runnable** — launches Isaac Sim, rolls out real Cartpole, records episodes. |
| `play_cartpole_with_recorder.py` | Template: record a trained *policy* (copy the rollout loop, swap the action). |
| `train_cartpole_with_recorder.py` | Template: record during training. |
| `configs/reward.yaml` | Reward-term weights. |
| `configs/recorder.yaml` | Demo rollout settings. |

## Quick run

```bash
# from the repo root, with our package on PYTHONPATH and Isaac Lab's Python
PYTHONPATH=$PWD/python /path/to/isaaclab-env/bin/python \
    python/rl_episode_inspector/examples/cartpole/generate_demo_episodes.py \
    --output-dir sample_data/cartpole/episodes --num-episodes 12 --seed 42
```

Verified with **Isaac Lab 2.3.2 / Isaac Sim 6.0.0 / Python 3.12** on an NVIDIA
A10G (headless). Task ID: `Isaac-Cartpole-Direct-v0`. See
[`docs/isaac_lab_integration.md`](../../../../docs/isaac_lab_integration.md) for
details, gotchas, and how to record a trained policy.

## Reward terms

`alive`, `pole_upright`, `cart_centering`, `cart_velocity_penalty`,
`pole_angular_velocity_penalty`, `action_penalty` — each recorded raw and
weighted, plus `reward_step_total`, `reward_cumulative`, and `episode_return`.
