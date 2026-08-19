.PHONY: format lint test fix check

format:
	poetry run ruff format .

lint:
	poetry run ruff check .
	poetry run mypy .

test:
	poetry run pytest

fix:
	poetry run ruff check . --fix

check:
	poetry run pre-commit run --all-files
