# Security

## Threat model

The backend is a **local developer tool** that reads episode files from a
configured directory and serves them over HTTP to the local frontend. It is not
intended to be exposed to the public internet. Even so, the most realistic risk
— a crafted `episode_id` reading arbitrary files — is defended against.

## Path-traversal protection

`episode_id` arrives from the network and is untrusted. Two layers:

1. **Charset allow-list** (`storage/paths.py`): `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`
   and an explicit `".." not in id` check. This rejects `/`, `..`, encoded
   traversal, and absolute paths before any filesystem access. The HTTP layer
   (`server/security.py`) maps a rejected id to **400**.
2. **Containment re-check**: `safe_episode_dir` resolves `root/episode_id` and
   verifies the result is still inside the resolved episodes root (guards against
   symlink trickery).

Covered by `tests/unit/test_storage_validation.py::test_path_traversal_rejected`
and `tests/integration/test_api_server.py::test_path_traversal_rejected`.

## Other measures

- **Read-only API.** All routes are `GET`; CORS allows only `GET` from localhost
  dev origins (`server/app.py`).
- **No information leak on errors.** Unknown ids return a generic 404; validation
  errors describe the *data* problem, not host filesystem layout.
- **No secrets / no private paths in the repo.** Sample/episode data is
  git-ignored and regenerated from the deterministic fake generator. The Isaac
  Lab path is configurable (`ISAACLAB` make var), not baked into committed code.

## If you must expose it

Put it behind a reverse proxy with auth/TLS and bind the backend to `127.0.0.1`
(the default). Do not point `--episodes-dir` at a directory that contains
anything but episodes.
