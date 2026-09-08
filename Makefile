.PHONY: run format lint test check

run:
	uv run python -m src


format:
	uv run ruff format .

lint:
	uv run ruff check . --fix
	uv run mypy .

test:
	uv run pytest

check:
	uv run pre-commit run --all-files
