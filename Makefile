.PHONY: setup lint fmt typecheck test check smoke results

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

results:          ## tier 1a: every reported number, table and figure from the committed analyses
	mkdir -p results/tables results/figures
	uv run python -m memshift.analysis.paper_numbers > results/numbers.tex
	uv run python -m memshift.analysis.paper_tables --out results/tables
	uv run python -m memshift.analysis.figures --out results/figures --tables results/tables
	uv run python -m memshift.analysis.paper_extras --out results/tables
	uv run python -m memshift.analysis.paper_round3b --analysis-dir runs/stage9_round3b/analysis \
	  --out results/tables
	uv run python -m memshift.analysis.audit_table > results/tables/audit_table.tex
	uv run python -m memshift.analysis.neighbour_table audit/neighbours/properties.json \
	  results/tables
	uv run python scripts/readme_results.py --numbers results/numbers.tex
