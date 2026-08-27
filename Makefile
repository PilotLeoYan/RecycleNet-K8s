.PHONY: run format lint test fix check

run:
	uv run python src

format:
	uv run ruff format .

lint:
	uv run ruff check .
	uv run mypy .

test:
	uv run pytest

fix:
	uv run ruff check . --fix

check:
	uv run pre-commit run --all-files
