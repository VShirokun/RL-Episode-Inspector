# Isaac Lab integration

## Supported versions

The Cartpole demo was implemented and **verified** against:

```
Isaac Lab    : 2.3.2
Isaac Sim    : 6.0.0
Python       : 3.12
Physics      : Newton backend (joint state exposed as warp arrays)
GPU          : NVIDIA A10G (CUDA 13.0), headless rendering
Task ID      : Isaac-Cartpole-Direct-v0
```

Other 2.x versions are likely compatible but untested. The recorder/storage core
has **no** Isaac Lab dependency — only the example under
`python/rl_episode_inspector/examples/cartpole/` does.

## Running the demo

You need our package importable under Isaac Lab's Python. Two options:

```bash
# A) Isaac Lab launcher (activate its env first so `python` resolves)
PYTHONPATH=$PWD/python /path/to/IsaacLab/isaaclab.sh -p \
    python/rl_episode_inspector/examples/cartpole/generate_demo_episodes.py \
    --output-dir sample_data/cartpole/episodes --num-episodes 12 --seed 42

# B) Call the Isaac Lab venv's Python directly (what we used)
PYTHONPATH=$PWD/python /path/to/isaaclab-env/bin/python \
    python/rl_episode_inspector/examples/cartpole/generate_demo_episodes.py \
    --output-dir sample_data/cartpole/episodes --num-episodes 12 --seed 42
```

The example imports `pyarrow`/`pydantic`; install them into the Isaac Lab env if
missing (`uv pip install --python /path/to/isaaclab-env pyarrow pydantic`).

Then serve + view exactly like the fake-data demo:

```bash
make backend-dev      # uses .venv; reads the episodes you just wrote
make frontend-dev
```

CLI options: `--output-dir`, `--num-episodes`, `--seed`, `--env-id`, `--task`,
`--reward-config`, `--max-steps`, `--control-sign`, `--probe-sign` (diagnostic).

## How the recorder hooks in

`generate_demo_episodes.py` is the reference. Per control step it:

1. reads real joint state from the articulation (`slider_to_cart`,
   `cart_to_pole`), converting warp/torch arrays to numpy;
2. computes a normalized cart force with a PD controller
   (`controller.balance_action`) — skill varied per episode for a return spread;
3. calls `env.step(action)` (action is scaled by `action_scale = 100 N`);
4. computes the six raw reward terms (`reward.compute_reward_terms`);
5. `recorder.record_frame(state=…, action=…, rewards_raw=…, reward_weights=…,
   terminated=…, truncated=…)`;
6. on `terminated or truncated`, `recorder.end_episode(reset_reason=…)`.

To record a **trained policy** instead of the PD controller, copy this loop and
replace step 2 with your policy's action (see `play_cartpole_with_recorder.py`).

## Reward decomposition

Six explicit terms (`alive`, `pole_upright`, `cart_centering`,
`cart_velocity_penalty`, `pole_angular_velocity_penalty`, `action_penalty`), each
recorded raw **and** weighted, plus `reward_step_total`, `reward_cumulative`, and
`episode_return`. Weights live in
`examples/cartpole/configs/reward.yaml` (override with `--reward-config`). The
decomposition is computed by us (pure functions in `reward.py`) rather than read
from Isaac Lab's `RewardManager`, so every term is available at every frame and
the formulas match the fake generator.

## Vectorized environments (MVP limitation)

> For the MVP the recorder records one selected environment (`num_envs=1`,
> `env_id=0`). Ranking is over the completed episodes recorded from that env.

The schema already carries `env_id`, `episode_index`, and `global_step_start/end`,
so multi-env recording and top-k selection are additive (see
`docs/roadmap.md`).

## Notes / gotchas (learned during implementation)

- **Newton backend → warp arrays.** `robot.data.joint_pos` is a `warp.array`;
  item indexing throws. Convert with `.numpy()` (handled by `_to_numpy`).
- **Omniverse hijacks stdout.** `print()` from your script is swallowed; write
  diagnostics to a file (the demo does this for `--probe-sign`).
- **Control sign is environment-specific.** We verified empirically that a pole
  leaning +θ needs a *negative* normalized force here (`--probe-sign`).
- **PD demo episodes terminate** (pole tilt or the ±3 m cart bound) rather than
  surviving the full 5 s — expected for a hand-tuned controller, and it produces
  a clear spread of returns. A trained policy will do better.

## Optional integration test

`tests/isaaclab/test_cartpole_integration.py` (marked `isaaclab`) records two real
episodes and asserts they load with the expected columns. It auto-skips without
Isaac Lab. Run it under Isaac Lab's Python:

```bash
PYTHONPATH=$PWD/python /path/to/isaaclab-env/bin/python -m pytest -m isaaclab
```
