.PHONY: setup lint fmt typecheck test check smoke

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

smoke:            ## offline, no model: generate one world branch from a seed, re-check its SHA256
	@out=$$(mktemp -d) && dir=$$(uv run memshift gen --seed 0 --branch src --out "$$out") && \
	  echo "$$dir" && uv run memshift verify "$$dir" && rm -r "$$out"
