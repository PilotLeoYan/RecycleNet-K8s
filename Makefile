.PHONY: sync-cpu sync-gpu sync-dev-cpu sync-dev-gpu smoke-test train train-best hpo export super-train mlflow format lint test check

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
	$(MAKE) sync-dev-gpu
	$(MAKE) train CONFIG=configs/smoke.yaml
	$(MAKE) hpo CONFIG=configs/smoke.yaml

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


format:
	uv run ruff format .

lint:
	uv run ruff check . --fix
	uv run mypy .

test:
	uv run pytest

check:
	uv run pre-commit run --all-files
