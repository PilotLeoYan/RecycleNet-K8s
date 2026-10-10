.PHONY: sync-cpu sync-gpu train hpo mlflow format lint test check

sync-cpu:
	uv sync --extra cpu

sync-gpu:
	uv sync --extra gpu

train:
	RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 RAY_LOGGER_LEVEL=error uv run python -m src train

hpo:
	RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 uv run python -m src hpo

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
