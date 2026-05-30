# Data format

Schema version: **0.1.0**. Compatibility is by **major** version (a reader
accepts any `0.x` produced by the same major; see `storage/validation.py`).

## Episode directory

```
<episodes_root>/
  cartpole_000001/
    metadata.json     # episode metadata + signal specs
    frames.parquet    # one row per frame
```

Future-reserved (not written by the MVP): `events.parquet`, `assets.json`,
`thumbnail.png`.

## `metadata.json`

Produced by `EpisodeMetadata` (`storage/schemas.py`). Key fields:

| Field | Meaning |
|-------|---------|
| `schema_version` | Format version, e.g. `"0.1.0"`. |
| `episode_id` | Directory name; must match `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`. |
| `task_name` / `task_source` | e.g. `"Cartpole"` / `"fake_data"` or `"isaac_lab"`. |
| `env_id`, `episode_index`, `global_step_start/end` | Provenance (vectorized-env aware). |
| `num_frames`, `dt`, `fps`, `duration_seconds` | Timing. |
| `terminated`, `truncated`, `reset_reason` | Termination — see below. |
| `episode_return` | Final cumulative weighted reward. |
| `seed`, `policy_checkpoint` | Reproducibility. |
| `signals` | List of `SignalSpec` (one per non-required column). |
| `viewer` | `{ type, state_mapping }` for the 3D viewer. |

### `SignalSpec`

`{ name, kind, dtype, shape, unit, description, display? }` where `kind` ∈
`state | reward_raw | reward_weighted | reward_total | action | observation |
debug | event`.

## `frames.parquet`

**Required columns** (every episode, any task):

| Column | dtype |
|--------|-------|
| `frame_index` | int32 |
| `timestamp` | float32 (seconds, **simulation** time, not wall clock) |
| `terminated` | bool |
| `truncated` | bool |
| `done` | bool (`terminated or truncated`) |

**Task columns** — every column that isn't required must be declared as a signal
in metadata (validation enforces this). Cartpole writes:

```
cart_position, cart_velocity, pole_angle, pole_angular_velocity   # state
action_cart_force                                                 # action
reward_<term>_raw, reward_<term>_weighted   for each reward term  # reward_raw/weighted
reward_step_total, reward_cumulative                              # reward_total
```

## Reward naming convention

| Name | Definition |
|------|------------|
| `reward_<name>_raw` | Raw term **before** applying its weight. |
| `reward_<name>_weighted` | `raw * weight`. |
| `reward_step_total` | Sum of all weighted terms at this frame. |
| `reward_cumulative` | Running sum of `reward_step_total`. |
| `episode_return` | Final `reward_cumulative` (stored in metadata). |

Never use an ambiguous name like `total_reward` for both per-frame and final
return. Raw vs weighted is always explicit, in both data and UI.

## Termination convention

Store all three, not just `done`:

- `terminated` — task-specific terminal condition (e.g. pole fell).
- `truncated` — time limit / external truncation (e.g. max episode length).
- `done` — `terminated or truncated`.
- `reset_reason` — human-readable string (`"pole_fell"`, `"max_episode_length"`, …).

This lets you distinguish *failed* episodes from ones that simply ran out of time.

## Units & conventions

Timestamps in seconds (sim time); angles in radians; positions in meters;
velocities in m/s or rad/s; reward units may be `null`. NaN/Inf are rejected at
write time.

## Why Parquet

Easy to write from Python, columnar/typed, chunk-readable, test-friendly. The
**frontend never reads Parquet** — the backend serves JSON. If you need to swap
storage later, only `parquet_reader.py`/`parquet_writer.py` change.
