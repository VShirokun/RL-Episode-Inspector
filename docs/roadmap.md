# Roadmap

The MVP deliberately stays small. The architecture (generic signals, slice/filter
API, single playback store) is designed so the following can be added without a
rewrite.

## Near-term

- **Side-by-side episode comparison** (best vs worst / median, checkpoints,
  reward configs). Two viewers + shared/normalized timeline. See spec §32.
- **Actions & observations panels** — a signal selector (Rewards / Actions /
  Observations / Debug) over the already-generic signal model. Recording stays
  configurable so users choose what to save (Risk 4). Spec §33.
- **Multi-environment recording** — record several `env_id`s from a vectorized
  rollout and keep top-k by return. The schema already carries `env_id`,
  `episode_index`, `global_step_*`.

## Medium-term

- **Humanoid / tennis integration** — needs a separately designed state format
  (root pose, joint positions, body poses, skeleton hierarchy, name maps) and a
  skeleton/mesh viewer. Must not block or complicate the Cartpole core. Spec §34.
- **AMP discriminator reward** decomposition and visualization.
- **Checkpoint comparison** across training iterations.

## Longer-term

- Train-mode live dashboard (online recording).
- Streaming very large episodes (the API already supports `start`/`end` slicing
  and `names` filtering for this).
- Isaac Sim / USD physics replay (the MVP is deterministic visual reconstruction
  only).
- Visual-regression screenshot tests once the UI stabilizes.

## Explicitly *not* in the MVP

Humanoid support, deep AMP analysis, full observation/action recording for large
spaces, two-episode comparison, dataset streaming, live dashboards, distributed
logging, in-sim physics replay, importing humanoid USD/glTF into the browser.
