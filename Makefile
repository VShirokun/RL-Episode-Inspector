# RL Episode Inspector — developer commands.
# Most targets use the local .venv created by `make install`.

VENV ?= .venv
PY := $(VENV)/bin/python
RLEI := $(VENV)/bin/rl-episode-inspector
EPISODES_DIR ?= sample_data/cartpole/episodes
ISAACLAB ?= /mnt/nvme2n1/IsaacLab

.PHONY: help
help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-28s\033[0m %s\n", $$1, $$2}'

# ---- Python ----
.PHONY: install
install: ## Create .venv and install the package (with dev deps)
	uv venv $(VENV) --python 3.12
	uv pip install --python $(VENV) -e ".[dev]"

.PHONY: test
test: ## Run CI-safe Python tests (excludes Isaac Lab)
	$(PY) -m pytest -m "not isaaclab"

.PHONY: lint
lint: ## Lint Python with ruff
	$(VENV)/bin/ruff check python tests

.PHONY: typecheck
typecheck: ## Type-check Python with mypy
	$(VENV)/bin/mypy python/rl_episode_inspector

# ---- data ----
.PHONY: generate-fake-cartpole-demo
generate-fake-cartpole-demo: ## Generate fake demo episodes (no Isaac Lab)
	$(RLEI) generate-fake-cartpole --output-dir $(EPISODES_DIR) --num-episodes 20 --seed 42

.PHONY: generate-cartpole-demo
generate-cartpole-demo: ## Generate REAL Isaac Lab Cartpole episodes (needs Isaac Lab)
	$(ISAACLAB)/isaaclab.sh -p python/rl_episode_inspector/examples/cartpole/generate_demo_episodes.py \
		--output-dir $(EPISODES_DIR) --num-episodes 12 --seed 42 --env-id 0

# ---- servers ----
.PHONY: backend-dev
backend-dev: ## Run the backend API on :8000
	$(RLEI) serve --episodes-dir $(EPISODES_DIR) --host 127.0.0.1 --port 8000

.PHONY: frontend-install
frontend-install: ## Install frontend deps
	cd frontend && npm install

.PHONY: frontend-dev
frontend-dev: ## Run the Vite dev server on :3000
	cd frontend && npm run dev

.PHONY: frontend-build
frontend-build: ## Type-check and build the frontend
	cd frontend && npm run build

.PHONY: frontend-test
frontend-test: ## Run frontend unit tests (vitest)
	cd frontend && npm run test

.PHONY: e2e
e2e: ## Run Playwright E2E tests (boots backend + frontend)
	cd frontend && npm run e2e

# ---- aggregate ----
.PHONY: ci
ci: lint typecheck test frontend-build frontend-test ## Run the full CI-safe suite
