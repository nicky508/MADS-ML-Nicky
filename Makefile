
.PHONY: install test lint format

install:
	pdm install

test:
	pdm run pytest

lint:
	pdm run ruff MADS-ML-Nicky
	pdm run mypy MADS-ML-Nicky

format:
	pdm run isort -v MADS-ML-Nicky
	pdm run black MADS-ML-Nicky
