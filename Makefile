.PHONY: train hpo mlflow format lint test check

train:
	uv run python -m src train

hpo:
	uv run python -m src hpo

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
