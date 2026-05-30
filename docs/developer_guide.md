# Developer guide

## Prerequisites

- Python 3.10+ (3.12 used in CI), [`uv`](https://docs.astral.sh/uv/)
- Node 22+ / npm

## Set up

```bash
make install           # .venv + editable install with dev deps
make frontend-install  # npm install in frontend/
```

## Run the app (fake data)

```bash
make generate-fake-cartpole-demo
make backend-dev       # terminal 1 -> http://127.0.0.1:8000 (OpenAPI at /docs)
make frontend-dev      # terminal 2 -> http://localhost:3000
```

The Vite dev server proxies `/api` and `/health` to the backend (`vite.config.ts`),
so there's no CORS to worry about in development. Point it elsewhere with
`RLEI_BACKEND=http://host:port npm run dev`.

## Checks

```bash
make lint        # ruff
make typecheck   # mypy
make test        # pytest (CI-safe; excludes -m isaaclab)
make frontend-test
make e2e         # Playwright (boots both servers itself)
make ci          # everything CI runs
```

## How-tos

### Add a new reward signal (Cartpole)

1. Add the raw term in the reward function (fake: `fake_data/generate_fake_cartpole.py`
   `_reward_terms`; real: `examples/cartpole/reward.py`).
2. Add its weight to the reward config (`examples/cartpole/configs/reward.yaml`).
3. That's it — the recorder infers `reward_<name>_raw/_weighted` columns and
   `SignalSpec`s automatically, and the charts pick up any `reward_weighted`
   signal. No frontend change needed.

### Add a new task

1. Create `examples/<task>/` with an adapter that, per frame, calls
   `EpisodeRecorder.record_frame(state=..., action=..., rewards_raw=...,
   reward_weights=..., terminated=..., truncated=...)`.
2. Pass `viewer_type` + `state_mapping` to the `EpisodeRecorder` constructor.
3. Core code needs **no** changes (don't import task code from core).

### Add a new viewer type

Add a component that reads `loaded.metadata.viewer.type` and renders accordingly;
switch on it where `Viewer3D` is used in `App.tsx`. Reuse `state_mapping` to find
the columns to drive it.

### Add a new chart

Build `Series[]` (`{name, color, values}`) from `loaded.columns` (see
`components/rewardSeries.ts`) and render with `<TimeSeriesChart … onSeek={seek} />`.
You get click/drag-seek and the synced marker for free.

## Project layout

See `docs/architecture.md`. Core invariant: **`python/rl_episode_inspector/`
(except `examples/`) must not import task-specific modules.**
