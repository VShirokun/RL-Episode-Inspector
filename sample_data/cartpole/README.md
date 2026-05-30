# Sample Cartpole episodes

This directory holds generated episodes for the demo. **It is intentionally
empty in git** (episode data is `.gitignore`d) — regenerate it locally:

```bash
# Fake data (no Isaac Lab):
make generate-fake-cartpole-demo
# or:
rl-episode-inspector generate-fake-cartpole --output-dir sample_data/cartpole/episodes -n 20

# Real Isaac Lab Cartpole (needs Isaac Lab — see docs/isaac_lab_integration.md):
make generate-cartpole-demo
```

Each episode is a directory:

```
episodes/
  cartpole_000001/
    metadata.json     # episode metadata + signal specs
    frames.parquet    # one row per frame
```

See [docs/data_format.md](../../docs/data_format.md) for the full schema. No
private data, credentials, or large binaries belong here.
