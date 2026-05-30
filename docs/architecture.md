# Architecture

RL Episode Inspector is built around one idea: **everything is a generic
time-series _signal_.** The recorder, storage, backend, and frontend never hard-
code Cartpole columns — they move `SignalSpec`s around. Cartpole is just the
first task adapter.

## Components & data flow

```
            ┌──────────────────────────────────────────────────────────────┐
            │  Task adapter (examples/cartpole/  OR  fake_data/)            │
            │  extracts state/action/reward terms per frame                 │
            └───────────────┬──────────────────────────────────────────────┘
                            │ record_frame(...)
                    ┌───────▼────────┐
                    │ EpisodeRecorder│  computes weighted + cumulative rewards,
                    │ (+ RewardBuffer)│ infers SignalSpecs, builds metadata
                    └───────┬────────┘
                            │ save_episode(metadata, columns)
                    ┌───────▼────────┐
                    │  EpisodeStore  │  <root>/<episode_id>/{metadata.json, frames.parquet}
                    └───────┬────────┘
              load_episode  │   list / load_metadata / load_frames(slice, names)
                ┌───────────┼────────────┐
                │           │            │
         ┌──────▼─────┐ ┌───▼────────┐  (validation.py guards every read/write)
         │EpisodeRanker│ │ FastAPI    │  GET /api/episodes, /metadata, /frames,
         │best/worst/  │ │ server     │      /signals, /ranking
         │median       │ └───┬────────┘
         └─────────────┘     │ JSON
                       ┌──────▼─────────────────────────────────────────────┐
                       │ Frontend (React + Vite)                            │
                       │  api/client → playbackStore (Zustand, 1 source of  │
                       │  truth: currentFrame) → Viewer3D (Three.js),       │
                       │  RewardCharts / CombinedRewardChart (custom SVG),  │
                       │  TimelineControls, EpisodeSelector                 │
                       └────────────────────────────────────────────────────┘
```

## Python packages (`python/rl_episode_inspector/`)

| Package | Responsibility |
|---------|----------------|
| `storage/` | `SignalSpec`/`ViewerSpec`/`EpisodeMetadata`/`EpisodeSummary` schemas, Parquet read/write, `EpisodeStore`, validation, safe path resolution (`paths.py`). |
| `recorder/` | `EpisodeRecorder` (task-agnostic) + `RewardBuffer` (raw→weighted→cumulative). |
| `ranking/` | `EpisodeRanker` — best/worst/median by `episode_return`. |
| `replay/` | `build_replay_data` — resolve `viewer.state_mapping` into per-frame series. |
| `server/` | FastAPI app factory, routes, HTTP security helpers. |
| `fake_data/` | Toy Cartpole generator (no Isaac Lab) for demos/tests. |
| `examples/cartpole/` | **Task adapter** — real Isaac Lab integration. Never imported by core. |
| `cli.py` | `generate-fake-cartpole`, `serve`, `rank`, `version`. |

## Frontend (`frontend/src/`)

- **`playback/playbackStore.ts`** — the single source of truth. `currentFrame`,
  `isPlaying`, `speed`, loaded episode, episode list, plus all actions. The
  viewer, charts, controls, and value panels are pure subscribers. No component
  owns frame state (Risk 5 mitigation).
- **`playback/frameSync.ts`** — pure pixel↔frame math + playback advancement,
  unit-tested in isolation.
- **`components/Viewer3D.tsx`** — Three.js scene, updated from frame columns each
  rAF via `metadata.viewer.state_mapping` (deterministic visual reconstruction,
  not physics replay).
- **`components/TimeSeriesChart.tsx`** — custom interactive SVG chart.

## Notable design decisions

- **Custom SVG charts instead of Plotly/ECharts.** The spec suggested a charting
  library, but the hard requirement is *precise* click-to-seek and drag-to-scrub
  with a cursor-following marker. Mapping pointer-x→frame directly (frameSync) is
  simpler and more reliable than bending a charting lib's interaction model, and
  it keeps the bundle small and the seek math testable. This is the one
  documented deviation from the recommended stack.
- **E2E specs live under `frontend/e2e/`** (not repo-root `e2e/`) so Node resolves
  `@playwright/test` from `frontend/node_modules`.
- **One env per recorder (MVP).** See `docs/isaac_lab_integration.md` §Vectorized
  environments; the schema already carries `env_id`/`episode_index`/`global_step_*`
  for future multi-env support.

## Extension points

- **New task** → add an adapter under `examples/<task>/` that calls
  `EpisodeRecorder.record_frame`; set `viewer.type` + `state_mapping`.
- **New viewer** → add a component keyed on `metadata.viewer.type`. For full
  robots use the generic `articulation3d` viewer: record every body's pose with
  `EpisodeRecorder.register_bodies` + `record_frame(poses=...)` (see
  `examples/isaaclab_poses.py`) — no new frontend code needed for a new robot.
- **New chart** → build `Series[]` from columns and render with `TimeSeriesChart`.
- **Richer ranking** → add methods to `EpisodeRanker` over the same summary table.
