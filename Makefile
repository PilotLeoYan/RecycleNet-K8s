.PHONY: format lint test check

format:
	poetry run ruff format .

lint:
	poetry run ruff check .
	poetry run mypy .

test:
	poetry run pytest

check:
	poetry run pre-commit run --all-files
