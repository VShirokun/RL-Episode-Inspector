# Testing

The suite is split so **CI never needs a GPU, Isaac Sim, Omniverse, or a
display**. Real Isaac Lab tests are opt-in.

## Layers

| Layer | Where | Run |
|-------|-------|-----|
| Python unit | `tests/unit/` | `make test` |
| Python integration | `tests/integration/` | `make test` |
| Frontend unit (vitest) | `frontend/src/**/*.test.ts` | `make frontend-test` |
| E2E (Playwright) | `frontend/e2e/` | `make e2e` |
| Isaac Lab (opt-in) | `tests/isaaclab/` | `pytest -m isaaclab` |

## CI-safe Python tests

`make test` runs `pytest -m "not isaaclab"`. Covered:

- schemas & signal specs, storage round-trip, validation (missing columns,
  undeclared columns, NaN/Inf, schema version, done-consistency, path traversal)
- recorder (start/record/end, weighted + cumulative reward, episode_return,
  multiple episodes, error paths, NaN/Inf rejection, max-saved cap)
- fake-data generator (validity, varied returns, determinism)
- ranking (best/worst/median for odd/even/equal/empty/single, corrupted-skip)
- backend API (episodes, metadata, frames + slicing, signals + filtering,
  ranking, unknown id 404, empty dir, **path traversal rejected**)

## Frontend unit tests

`frameSync` (clamp, pixel↔frame round-trip, advance/speed/end) and the playback
store (stepping/clamp, seek-pauses-playback, play-restarts-at-end, tick).

## E2E tests

Playwright boots the real backend (auto-generating a small fake dataset) and the
Vite dev server, then drives Chromium **at deviceScaleFactor 2 (HiDPI)**: page
load, episode list, best-select, play/pause (button + Space), arrow stepping,
chart click-seek, chart drag-scrub, speed change, backend-error surfacing, and a
canvas-overflow regression guard.

## Isaac Lab tests (opt-in)

`tests/isaaclab/test_cartpole_integration.py` is marked `@pytest.mark.isaaclab`
and **skips automatically** when `isaaclab` can't be imported, so it's safe in
any environment. To run it you need a working Isaac Lab and to launch under its
Python — see `docs/isaac_lab_integration.md`:

```bash
/path/to/IsaacLab/isaaclab.sh -p -m pytest -m isaaclab
```

> **Caveat:** Isaac Sim's Kit runtime can segfault when driven *inside* pytest
> (a known Isaac Sim limitation, not a bug here). The verified, reliable way to
> exercise the same integration path is the standalone
> `examples/cartpole/generate_demo_episodes.py` — that's how the real-Cartpole
> integration was validated (records, loads, validates, ranks).

## CI

`.github/workflows/ci.yml` runs three jobs on every push/PR: **python** (ruff +
mypy + pytest), **frontend** (tsc + vitest + build), **e2e** (Playwright with
fake data). Isaac Lab tests are never run in CI.
