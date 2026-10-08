# MemShift

MemShift measures per-item *carryover* of a personal data layer's derived knowledge across four
shifts: source revision (S_src), definition change (S_def), writer-model change (S_wr) and
implementation change (S_impl). The property it defines is *shift-safe*; HoldMem is the reference
implementation, instantiable in the four cells of repair determinism D and binding B.

## Contents

- `src/memshift/gen`: seeded world generator (six sources, twelve months, 48 knowledge items,
  versioned definitions, consent), shift branches and gold with dependency sets.
- `src/memshift/holdmem`: HoldMem (writer, checker, binding, maintenance, serving).
- `src/memshift/adapters`: zero system (BM25 over raw objects), RAG, oracle, Mem0 and Graphiti
  adapters, Portable Memory (.mem) export / import with a derivation extension.
- `src/memshift/calsync`: official incremental-sync semantics per source.
- `src/memshift/experiments`, `src/memshift/analysis`: E1-E7 drivers and analyses (no LLM judge).
- `prereg/`: pre-registration v0 and v1 (frozen by commit hash) and the decision log.
- `runs/`: the manifest, environment and analysis of every run the paper uses; raw outputs are
  regenerated from seeds and the pinned models.
- `paper/`: LaTeX sources; every number is a macro generated from `runs/` analyses.

## Reproduce

```bash
uv sync --dev          # Python >= 3.12
make check             # ruff, mypy, pytest
cp .env.example .env   # point MEMSHIFT_ENDPOINTS at OpenAI-compatible (vLLM) servers
uv run python -m memshift.experiments.e1 --seeds 1000-1019 --discriminative \
    --branches src,def,orphan \
    --cells D0B0,D1B0,D0B1_source,D1B1_source,D0B1_cited,D1B1_cited
```

Models and revisions are listed in `configs/models.yaml` and in each run's `env.json`.

## License

MIT (see `LICENSE`). Repository: <https://github.com/zichuanxu/memshift>.
