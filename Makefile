.PHONY: sync-cpu sync-gpu sync-dev-cpu sync-dev-gpu smoke-test train hpo mlflow format lint test check

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

hpo:
	$(RAY_ENV) $(RUN_SRC) hpo $(EXTRA_ARGS)

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
