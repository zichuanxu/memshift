# Reproducing MemShift

This guide complements the [README](../README.md). It gives the repository layout, the commands that
rebuild every reported number, table and figure, the map from the paper to the code, the contents of
the raw outputs and the interface for evaluating a new memory system.

Reproduction has two tiers. Tier 1 recomputes every reported number on a CPU from the released
outputs. Tier 2 reruns the experiments from seeds on GPUs.

## Repository layout

```
src/memshift/
  gen/            seeded world generator: sources, items, versioned definitions, shift branches, gold
  probes/         point queries with an as-of time, gold value and gold status
  scoring/        typed matching and bootstrap statistics
  holdmem/        HoldMem: writer, checker, keys, the writer's programs, sandbox
  adapters/       no-store BM25, plain RAG, oracle context, native memory systems, Portable Memory export
  calsync/        incremental-sync semantics of the sources
  experiments/    experiment drivers
  analysis/       analyses, staging of runs, generators of numbers, tables and figures
configs/          model roles and fixed parameters
runs/             per run: manifest.json, env.json, analysis/ (raw outputs are on Hugging Face)
results/          sources.json: the staged analysis behind each group of reported numbers
audit/            the system audit and the neighbour audit
docker/           CPU run image and per-system images of the native memory systems
scripts/          vLLM serving, container runner, native image builder, results-table check
tests/            pytest suite
docs/             COMPUTE.md: hardware, GPU-hours, model revisions; REPRODUCE.md: this guide
```

## Worlds and tests

Worlds regenerate byte for byte from their seeds. This checks a test world against the hash that a
released run recorded:

```bash
uv run python - <<'EOF'
import json
from memshift.gen.shifts import generate_branch
from memshift.runinfo import world_record

man = json.load(open("runs/e1_test_s1000-1003_20260928T202242Z/manifest.json"))
for w in man["worlds"]:
    if w["seed"] == 1000:
        new = world_record(generate_branch(1000, w["branch"]))["events_sha256"]
        print(w["branch"], new == w["events_sha256"])
EOF
```

`make check` runs ruff, mypy and the pytest suite (about 45 minutes on a laptop). Tests that read the
paper sources carry the `paper` marker and skip without them.

## Tier 1a: from the analyses in this repository (about one minute)

Each `runs/stage*/` directory holds a merged analysis (`analysis/*.json`) and a manifest that lists
its source runs with the SHA256 of their manifests. `results/sources.json` maps each group of
numbers to its stage. `make results` runs these commands, which rebuild the numbers, tables and
figures, and then checks the results table of the [README](../README.md#main-results):

```bash
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
```

With the raw outputs of the Hugging Face dataset in `runs/`, every `.tex` output is byte-identical to
the file the paper was built from, and the figures render the same (their PDF bytes differ in the
creation date and in the names of the embedded font subsets). A few cells read raw files, and with
the files of this repository alone they print a visible placeholder (`??`):

- Table 7 (worked example): `probes.jsonl`, `answers.jsonl` and `items.jsonl` of
  `runs/e1_test_s1000-1003_20260928T202242Z`.
- Table 13 and the `ExAttr*` macros (changed item-times by cause, and `\ExAttrWorldSlots`, the
  number of items per world that the abstract and the benchmark section print):
  `runs/stage2_e1/analysis/e1_item_table.jsonl` and `probes.jsonl` of the five
  `runs/e1_test_s1*_20260928T*` runs.
- `\ClaudeTokensTotal` and `\ClaudeSpendUSD` (Claude tokens and cost): `calls.jsonl` of every run
  whose manifest names a Claude model.

`\GpuHoursNativeSystems` also counts the 18 Cognee runs, whose manifests are in the dataset
(`runs/e5_cognee_*/manifest.json`). Without them it counts only the other native-system runs.

## Tier 1b: from the raw outputs (minutes per stage)

Download the raw outputs from the Hugging Face dataset into `runs/` ([Raw outputs](#raw-outputs)).
The paths are the same as in this repository. `uv run memshift verify runs/RUN` re-hashes a run's
files against its manifest. Then rebuild a stage from the runs its manifest lists and compare it
with the committed analysis. For the main experiment:

```bash
uv run python -m memshift.analysis.stage e1 /tmp/stage2_e1 $(uv run python -c "import json; \
    print(' '.join('runs/' + s['run_id'] for s in json.load(open('runs/stage2_e1/manifest.json'))['sources']))")
cmp /tmp/stage2_e1/analysis/e1_metrics.json runs/stage2_e1/analysis/e1_metrics.json
```

The `experiment` field of a stage manifest (`merged-KIND`) names its kind.
`memshift.analysis.stage` builds the kinds e1, m2, e2, e3, e4, e4x, e5, e5_infer, e5_cost, e6, e7,
ctl and engine. The other stages are written by their own analysis modules (`e1_keys`, `ex_account`,
`ex_e1_cross`, `ex_e1_more`, `ex_misses`, `ex_round3`, `ex_round3_review`, `ex_round4_detection`,
`l_tokens`, `multiplicity`, `rank_corr`, `round3_paper`, and `round3b_typed`, `round3b_feedback`
and `round3b_extract` for `stage9_round3b`), whose docstrings give the command.

## Map from the paper to the code

Table and figure numbers refer to the preprint. Generators are modules of `memshift.analysis`,
drivers of `memshift`. Stage keys are the keys of `results/sources.json`, and `results/README.md`
gives the directory of each. The table names the main stages of each result, and
`results/sources.json` lists every stage a generator reads. The driver flags are those the run
manifests record (`params`), without seeds and run names. A cell list is the value of `--cells`.
Most `experiments.e1` runs also have `--discriminative`, which the manifest records as
`probes_after_fork`. `analysis.ex_round4_detection` adds no run: it reads the raw outputs of the
runs named in its stage manifest.

| Paper | Generator | Stage keys | Tier 2 driver |
|---|---|---|---|
| Fig. 1, Table 4 (main experiment, the identity, detection counted from the latest change) | `paper_numbers` | `ex_round4_detection` (stage10_ex_round4_detection), `ex_account` (stage5_ex_account), `ex_round3_review` (stage8_ex_round3_review), `e1` (stage2_e1) | `experiments.e1 --branches src,def,orphan` with D0B0,D1B0,D0B1_source,D1B1_source,D0B1_cited,D1B1_cited |
| Table 3 (overview) | `paper_extras` | `e1`, `c` (stage4_c_e1), `n` (stage4_n_e5), `m2_llama`, `m2_gemma`, `m2_gptoss`, and `ex_round4_detection` for the item counts | as the main experiment, the writer's programs, other model families and native systems |
| Figs. 3, 5, 6, Tables 12, 20, 31, 37, 39 (main experiment by item class, JSON arm, account by writer, misses of the citation key against a forced re-derivation, no-store BM25 after the shift) | `paper_tables`, `figures` | `ex` (stage5_ex_e1), `ex_account`, `ex_more` (stage5_ex_more), `ex_misses` (stage5_ex_misses), `ex_round3_review`, `e1`, `rep_e1` (stage4_rep_e1) | as the main experiment |
| Tables 13, 16 (changed item-times by cause, per definition) | `paper_tables` | `e1` | as the main experiment |
| Tables 14, 26, 27, 34, 36, 38 (cause of the change, evidence coverage, typed-input sizes, the key against the refresh, misses of the citation key, changes from the clock alone) | `paper_numbers` | `ex_round3_review`, and `ex_round4_detection` for Table 38 | none (`analysis.ex_round3_review` and `analysis.ex_round4_detection` on the raw outputs) |
| Table 33 (oracle key) | `paper_numbers` | `ex_round4_detection` | as the controls (Table 29), then `analysis.ex_round4_detection` on the raw outputs |
| Tables 21, 22, 48, 49 (the writer's programs, by item class, after a deletion) | `paper_tables` | `c`, `ex_round3` (stage6_ex_round3) | `experiments.e1 --no-baselines --branches src,def,orphan` with D0B0,D1B0,D0B1_source,D1B1_source,D0B1_cited,D1B1_cited,D1sB0,D1sB1_source,D1sB1_cited |
| Tables 18, 19, 30 (store and reader, query-time view) | `paper_tables` | `ex_round3`, `ctl` (stage5_controls) | as the main experiment, the writer's programs and the controls |
| Table 29 (controls) | `paper_tables` | `ctl` | `experiments.e1 --no-baselines` with D0B1_source,D0B1_gold,D1B1_gold,D1B1_scope,D1B0_refresh,D0tB1_source, with zero_Dm (query-time view) and with D0B1_source,D0fullB1_source (`--branches src,def`) |
| Table 32 (offline key audit) | `paper_tables` | `keyaudit` (stage5_key_audit) | none (`analysis.e1_keys`, no model) |
| Table 35 (refresh and upsert) | `paper_tables` | `round3` (stage7_round3) | `experiments.e1 --no-baselines` with D0B0,D0B0_refresh,D1B0_refresh,D1sB0_refresh and D0B0,D0B0_upsert,D1B0_upsert,D1sB0_upsert; for three other model families `--writer MODEL --reader MODEL --post-probes definition` with D0B0,D0B0_refresh,D1B0_refresh,D0B0_upsert,D1B0_upsert |
| Table 15 (unshifted branch) | `paper_tables` | `round3`, `ex_round3` | `experiments.e1 --no-baselines --branches natural` with D0B0,D0B1_source,D1B0,D1B1_source |
| Table 17 (held-out definitions) | `paper_tables` | `round3` | `experiments.e1 --no-baselines --kappa-set heldout:SEED --post-probes definition` with D0B0,D0B1_source,D1B0,D1B1_source,D1sB0,D1sB1_source |
| Table 18, Fig. 4 (Claude reader) | `paper_tables`, `figures` | `round3`, `ex_round3` | `experiments.e1 --no-baselines --reader claude-sonnet-5-5` with D0B0,D0B1_source,D1B0,D1B1_source,D1sB0,D1sB1_source |
| Table 25 (definitions in the user's own words) | `paper_tables` | `round3` | `experiments.e1 --no-baselines --program-user-words --program-model MODEL --program-temperature 0.7 --program-sample I` with D1sB0,D1sB1_source (temperature 0 and `--post-probes definition` for Claude) |
| Fig. 2 (program sets from six writers, Claude writers) | `figures` | `round3`, `ex_round3` | `experiments.e1 --no-baselines --program-model MODEL --program-temperature 0.7 --program-sample I` with D1sB0,D1sB1_source (`--post-probes definition` for the five writers other than Qwen3.5-9B, temperature 0 for Claude); `--defwriter MODEL --program-model MODEL` with D0B0,D0B1_source,D1B0,D1B1_source,D1sB0,D1sB1_source |
| Table 28 (every record as typed fields) | `paper_round3b` | `round3b` (stage9_round3b) | `experiments.e1 --no-baselines --branches src,def` with D0B1_source,D1B1_source,D0typedB1_source; Claude Sonnet 5.5 as writer of the definition items with `--defwriter claude-sonnet-5-5` |
| Table 23 (execution feedback) | `paper_round3b` | `round3b` | `experiments.e1 --no-baselines --branches src,def,orphan` with D0B0,D0B1_source,D1sB0,D1sB1_source,D1sfbB0,D1sfbB1_source |
| Table 24 (extracted fields) | `paper_round3b` | `round3b` | `experiments.e1 --no-baselines --branches src,def` with D0B0,D0B1_source,D1sB0,D1sB1_source,D1sxB0,D1sxB1_source |
| Fig. 8 (writer switch) | `figures` | `e2` (stage2_e2) | `experiments.e2 --tiers same,upgrade,downgrade` |
| Fig. 7 (horizon) | `figures` | `e3` (stage2_e3) | `experiments.e3` |
| Table 40 (diagnostic tests) | `figures` | `e1`, `e2`, `e3`, `rep_e2`, `rep_e3`, `m1`, `n`, `e6` (stage3_e6) | as those experiments |
| Tables 41, 42 (switches to other writers, other model families) | `paper_tables` | `m1` (stage4_m1_e2), `m2_llama`, `m2_gemma`, `m2_gptoss`, `m3` (stage4_m3) | `experiments.e2 --tiers cross:MODEL` (new writer); `experiments.e1 --writer MODEL --reader MODEL`; `experiments.e1 --reader MODEL --branches src,def` |
| Table 43 (replication) | `paper_tables` | `rep_e1`, `rep_e2`, `rep_e3` and `e1`, `e2`, `e3` | `experiments.e1`, `e2` and `e3` on worlds 1020-1039 |
| Table 44 (rewritten worlds) | `paper_tables` | `l` (stage4_l_rendered), `ltokens` | `experiments.e1 --render google/gemma-4-12B-it` |
| Fig. 9, Tables 45, 46 (native systems) | `figures`, `paper_tables` | `n`, `ncost`, `e5` (stage3_e5), `e5cost`, `e5wbshort` | `experiments.e5 --months 3 --systems SYSTEM`; the cells and baselines with `experiments.e1 --months 3` |
| Table 47 (Mem0 interventions) | `paper_tables` | `round3` | `experiments.x_mem0` |
| Table 7 (worked example) | `paper_extras` | raw outputs of `runs/e1_test_s1000-1003_20260928T202242Z` and the world generator | as the main experiment |
| Table 9 (models) | `paper_extras` | `manifest.json` and `env.json` of every run under `runs/` | none |
| Table 6 (neighbour audit), Table 8 (systems) | `neighbour_table`, `audit_table` | `audit/neighbours/properties.json`, `audit/system_audit.json` | none |

Sync (`e4`, `e4x`), export (`e6`), the failure-mode probes (`e7`) and the unnamed probes (`round3`)
enter the paper as numbers of `paper_numbers`. They are rerun with `experiments.e4`,
`experiments.e4x`, `experiments.e6`, `experiments.e7` and
`experiments.e1 --no-baselines --probe-unnamed-defs --branches def`. So do the key's effect with
Claude Sonnet 5.5 and Claude Opus 5.5 as writers of the definition items (`ex_round4_detection`, from
the runs of `experiments.e1 --defwriter MODEL` on worlds 1000-1009 with
D0B0,D0B1_source,D1B0,D1B1_source,D1sB0,D1sB1_source) and the rank correlations of Section 6 and
Appendix F.9 (`rank`, stage4_rank_corr). The rank correlations are computed over every system on
the three-month worlds and within three subsets, each with its own world bootstrap: the HoldMem
cells, the native systems, and the native systems with no-store BM25 and plain RAG.
`memshift.analysis.rank_corr` computes them from the raw runs of `experiments.e5` and of the HoldMem
cells and baselines on the same worlds. The numbers in Tables 2, 5 and 11 are macros of
`paper_numbers`, `paper_round3b` and `neighbour_table`, and the compute facts (`setup_facts.tex`)
come from `paper_extras`.

## Tier 2: rerun from seeds on GPUs

1. Download the model weights at the revisions in `docs/COMPUTE.md`. Each run's `env.json` records
   the revision and serving settings it used.
2. Serve each model with vLLM 0.30.0. `scripts/serve_model.sh MODEL REVISION GPU PORT [vLLM args]`
   starts one server in a Docker container and records its settings for the run manifests. For the
   main model and the embedding model:

   ```bash
   MAX_LEN=32768 scripts/serve_model.sh Qwen/Qwen3.5-9B \
       c202236235762e1c871ad0ccb60c8ee5ba337b9a 0 8001 --language-model-only --reasoning-parser qwen3
   MAX_LEN=8192 MEM_UTIL=0.12 scripts/serve_model.sh Qwen/Qwen3-Embedding-0.6B \
       97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 1 8011 --runner pooling
   ```

   `scripts/serve_stage4.sh MODEL GPU PORT` serves the other model families with their pinned
   revisions and engine flags.

3. Point the code at the servers. The code reads environment variables, not the file itself:

   ```bash
   cp .env.example .env    # set MEMSHIFT_ENDPOINTS, and CLAUDE_API_KEY for the Claude runs
   set -a; . ./.env; set +a
   ```

4. Run an experiment. The main experiment on the 20 test worlds:

   ```bash
   uv run python -m memshift.experiments.e1 --seeds 1000-1019 --discriminative \
       --branches src,def,orphan \
       --cells D0B0,D1B0,D0B1_source,D1B1_source,D0B1_cited,D1B1_cited --name e1_test
   ```

   The output goes to a new directory under `runs/`. Stage it and compare as in tier 1b.
   `scripts/run.sh` runs the same driver in a CPU container built from `docker/Dockerfile`
   (`RUN_MODULE=memshift.experiments.e1 MEMSHIFT_RUN_IMAGE=IMAGE scripts/run.sh ARGS`).

Every run's `manifest.json` records its parameters, worlds, code commit and models, so any released
run can be repeated with its own settings. A rerun sends every request again. Greedy decoding does not
always return the same reply twice, so a rerun can differ slightly from the released outputs. The
model servers held the GPUs for about 140 A100 GPU-hours in all ([COMPUTE.md](COMPUTE.md)).

## Raw outputs

The Hugging Face dataset [`zichuanxu/memshift`](https://huggingface.co/datasets/zichuanxu/memshift)
holds the answers, probes, items and call logs of every run at the same paths as here, the response
caches with the full text of every model reply in `cache/llm/`, and the logs of the native systems'
stores. Download it into the root of this checkout. A rerun with the recorded settings then reuses
the cached replies, since the cache key is the SHA256 of the request. The native systems' own
database files are not released, and the run manifests here record their SHA256. The dataset card
describes the files and the terms of each model that produced them.

In this repository, `runs/` holds every run's `manifest.json` (code commit, worlds and their SHA256,
parameters, output hashes) and `env.json` (serving image, model revisions), and the merged analyses
of the stages. [runs/README.md](../runs/README.md) describes the run directories and lists the runs
that no merged analysis reads.

## Evaluating your own memory system

A memory system reaches the benchmark through `src/memshift/adapters`. The drivers give it source
events in time order, ask it for evidence at each probe, and pass that evidence to the shared reader
within a 4,096-token budget. The reader's answer is scored by
`memshift.scoring.normalize.contract_valid`.

- **Serve only.** A system that keeps no derived state implements `serve(probe) -> ServeResult` and a
  `cost` dict (`adapters/baselines.py`: `ZeroSystem`, `PlainRAG`, `Oracle`). `ServeResult` holds the
  evidence blocks, the `(object_id, revision)` pairs they cite, per-item diagnostics, and whether the
  probe is blocked by consent.
- **Ingest and serve.** A system with its own store subclasses `NativeSystem`
  (`adapters/native.py`) and implements `_open`, `_close`, `_ingest_events(events)`,
  `_search(query) -> list[Hit]`, `export_raw()` and `retains_raw_text`. The base class handles the
  ledger, usage metering, snapshots for branching and packing the evidence. `event_text` and
  `event_metadata` give each event's text and source metadata. The adapters for Mem0, Graphiti,
  A-Mem, Cognee, HippoRAG 2, LightRAG and MemOS are worked examples.

Register the class in `NATIVE` in `src/memshift/experiments/e5.py` and run it on the short worlds:

```bash
uv run python -m memshift.experiments.e5 --seeds 1000-1014 --months 3 --systems mysystem \
    --on-error skip
```

`python -m memshift.analysis.stage e5 OUT e5:RUN e1:RUN noise:RUN base:RUN` then infers the system's
(D, B) label from its answers and compares its accuracy after each shift with the HoldMem cell of
the same label and with plain RAG. For the HoldMem cells, `analysis/e1_item_table.jsonl` of each
main-experiment stage records per changed item whether it was invalidated and whether its repair was
right. A memory that exposes its invalidations can be scored the same way.
