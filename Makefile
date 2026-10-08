.PHONY: setup lint fmt typecheck test check smoke release

setup:            ## create the venv and install runtime + dev dependencies
	uv sync --dev

lint:             ## ruff lint + format check
	uv run ruff check .
	uv run ruff format --check .

fmt:              ## auto-format and fix imports
	uv run ruff format .
	uv run ruff check --fix .

typecheck:        ## mypy
	uv run mypy

test:             ## pytest
	uv run pytest

check: lint typecheck test   ## everything CI runs

smoke:            ## one-world smoke run (plan §7 stage 0.4); fills in as code lands
	uv run memshift version

release:          ## open-source package into dist/ (allowlist + internal-identifier scan)
	uv run python scripts/make_release.py
