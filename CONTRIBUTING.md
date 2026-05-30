# Contributing

Thanks for your interest in RL Episode Inspector!

## Dev setup

```bash
make install            # Python .venv + dev deps
make frontend-install   # frontend deps
```

## Before opening a PR

Run the full CI-safe suite locally — it must be green:

```bash
make ci    # ruff + mypy + pytest + frontend typecheck/build/unit
make e2e   # Playwright (optional locally; runs in CI)
```

Isaac Lab tests (`-m isaaclab`) are optional and only run on a machine with Isaac
Lab installed; see [docs/isaac_lab_integration.md](docs/isaac_lab_integration.md).

## Ground rules

- **Keep the core task-agnostic.** Code in `python/rl_episode_inspector/` (outside
  `examples/`) must not import task-specific modules. New tasks go under
  `examples/<task>/`.
- **One source of truth for playback.** Frontend frame state lives only in the
  Zustand store; don't duplicate `currentFrame` in component state.
- **Add tests** for new behavior. Pure logic (reward formulas, frame-sync math)
  should be unit-tested without Isaac Lab or a browser where possible.
- Keep new code consistent with the surrounding style (ruff + mypy for Python,
  `tsc --strict` for TypeScript).

## Reporting bugs / requesting features

Use the issue templates. For data/format questions see
[docs/data_format.md](docs/data_format.md); for architecture see
[docs/architecture.md](docs/architecture.md).
