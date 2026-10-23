.PHONY: sync-cpu sync-gpu sync-dev-cpu sync-dev-gpu smoke-test smoke-export train train-best hpo export super-train mlflow api format lint test security devsecops check deploy-gcp teardown-gcp

CONFIG ?= configs/config.yaml
EXTRA_ARGS ?=
RAY_ENV = RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
RUN_SRC = uv run python -m src -c $(CONFIG)

sync-cpu:
	uv sync --extra cpu

sync-gpu:
	uv sync --extra gpu

sync-dev-cpu:
	uv sync --extra cpu --all-groups

sync-dev-gpu:
	uv sync --extra gpu --all-groups

smoke-test:
	$(MAKE) sync-dev-cpu
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) test
	$(MAKE) train CONFIG=configs/smoke.yaml
	$(MAKE) hpo CONFIG=configs/smoke.yaml
	$(MAKE) smoke-export
	$(MAKE) sync-dev-gpu
	$(MAKE) train CONFIG=configs/smoke.yaml
	$(MAKE) hpo CONFIG=configs/smoke.yaml
	$(MAKE) smoke-export

smoke-export:
	$(MAKE) export CONFIG=configs/smoke.yaml $(EXTRA_ARGS)

train:
	$(RAY_ENV) RAY_LOGGER_LEVEL=error $(RUN_SRC) train $(EXTRA_ARGS)

train-best:
	$(RAY_ENV) RAY_LOGGER_LEVEL=error $(RUN_SRC) train --use-best-hpo $(EXTRA_ARGS)

hpo:
	$(RAY_ENV) $(RUN_SRC) hpo $(EXTRA_ARGS)

export:
	uv run python -m src -c $(CONFIG) export $(EXTRA_ARGS)

super-train:
	$(RAY_ENV) RAY_LOGGER_LEVEL=error $(RUN_SRC) super-train $(EXTRA_ARGS)

mlflow:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

api:
	uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload $(EXTRA_ARGS)


format:
	uv run ruff format .

lint:
	uv run ruff check . --fix
	uv run mypy .

test:
	uv run pytest tests/ --cov=src --cov-fail-under=80

security:
	uv run bandit -c pyproject.toml -r src

devsecops:
	uv run python -m src.devsecops.orchestrator

check:
	uv run pre-commit run --all-files

deploy-gcp:
	./scripts/deploy_gcp.sh

teardown-gcp:
	./scripts/teardown_gcp.sh
