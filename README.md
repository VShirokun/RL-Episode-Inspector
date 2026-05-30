# RL Episode Inspector

Interactive, frame-by-frame analysis and 3D replay of individual reinforcement-learning
episodes from [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab/).

TensorBoard tells you the *aggregate* story — mean reward, losses, curves across thousands
of parallel environments. It does **not** help you answer *"why did this one episode fail?"*
RL Episode Inspector records individual episodes with full **reward decomposition**, lets you
pick the **best / worst / median** run by return, and replays it in a browser 3D viewer with
reward charts that stay synchronized to playback. Click or drag on a chart to jump to the
exact frame where something interesting happened.

> Status: MVP. The core is task-agnostic (generic time-series *signals*); **Cartpole** is the
> first demo task. A fake-data generator lets you run the whole pipeline with **no GPU and no
> Isaac Lab**.

## Key features

- 🎬 **3D replay** of recorded episodes in the browser (Three.js), no Isaac Sim required to view.
- 🤖 **Full-robot replay** — every rigid body's pose is recorded; the viewer shows the whole robot as **real 3D meshes** (default) or lightweight **cubes** (toggle, for when you don't want to ship/store meshes).
- 📊 **Reward decomposition** — every reward term recorded raw *and* weighted, per frame.
- 🏆 **Ranking** — select best / worst / median episode by `episode_return`.
- 🔗 **Synchronized timeline** — one source of truth; click/drag charts to scrub, keyboard shortcuts.
- 🧱 **Generic signal model** — the core stores `SignalSpec`s, not Cartpole columns; easy to extend.
- 🧪 **CI-safe** — fake data exercises storage → backend → frontend → E2E without GPU/Omniverse.

## Quick start (fake data, no Isaac Lab)

```bash
git clone <repo> && cd rl-episode-inspector

# Python backend
make install                       # create .venv and install the package
make generate-fake-cartpole-demo   # write sample episodes to sample_data/
make backend-dev                   # serve API at http://127.0.0.1:8000

# Frontend (in a second terminal)
make frontend-install
make frontend-dev                  # open http://localhost:3000
```

Then open <http://localhost:3000>, hit **Best**, and press **Space** to play.

## Quick start (real Isaac Lab Cartpole)

Requires a working Isaac Lab install (see [docs/isaac_lab_integration.md](docs/isaac_lab_integration.md)).

```bash
make generate-cartpole-demo        # real Isaac Lab Cartpole (balance)
# or a richer task with longer episodes + a SPARSE reward:
make generate-reach-demo           # real Franka Reach: visit a sequence of 3D targets
make backend-dev
make frontend-dev
```

Two real Isaac Lab demo tasks ship as examples:
- **Cartpole** (`examples/cartpole/`) — balance; dense multi-term reward; `cartpole` viewer.
- **Franka Reach** (`examples/reach/`) — visit 3D targets via IK; ~30 s episodes; a
  **sparse** `target_reached` reward (cumulative reward is a staircase); `reach3d` viewer.

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/architecture.md](docs/architecture.md) | Components, data flow, generic signal design, extension points |
| [docs/data_format.md](docs/data_format.md) | Episode directory, `metadata.json`, `frames.parquet`, conventions |
| [docs/developer_guide.md](docs/developer_guide.md) | Dev setup, running backend/frontend, adding signals/tasks/charts |
| [docs/isaac_lab_integration.md](docs/isaac_lab_integration.md) | Supported versions, recorder hooks, running the Cartpole demo |
| [docs/testing.md](docs/testing.md) | Unit / integration / E2E / Isaac Lab tests, CI |
| [docs/security.md](docs/security.md) | Local backend assumptions, path-traversal protection |
| [docs/roadmap.md](docs/roadmap.md) | Post-MVP plans (comparison, actions/observations, humanoid) |

## Development

```bash
make test        # run CI-safe Python tests
make lint        # ruff
make typecheck   # mypy
make ci          # lint + typecheck + tests + frontend build
```

## License

[MIT](LICENSE).
