# MemShift: Noticing a Change Is Not Repairing It in LLM-Derived Personal Memory

MemShift measures whether what a memory derived about a person stays correct after the records, the
person's definitions, the writer model or the memory product change.

- Preprint: link to be added.
- Raw model outputs: Hugging Face dataset [`zichuanxu/memshift`](https://huggingface.co/datasets/zichuanxu/memshift).

## Overview

An assistant stores values that a language model, the writer, derived from a person's records.
Keeping such a value correct after a change takes two steps. Detection notices that a dependency
changed. Repair produces the correct new value. End-answer benchmarks score both steps as one.

MemShift applies five controlled shifts to synthetic worlds. Each world is one synthetic person's
twelve months across six sources (e-mail, calendar, tasks, contacts, notes, chat), with 48 knowledge
items whose gold values and dependencies are fixed before any text is rendered.

| Shift | What changes |
|---|---|
| source revision | records behind about a fifth of the items are revised |
| deletion | the record behind half of the series, task and contact items is deleted |
| redefinition | half of the person's definitions move to a second version, and every record stays the same byte for byte |
| writer switch | the writer model is replaced |
| export | the memory is exported to Portable Memory and imported by a second implementation |

Every answer is scored by typed matching, without an LLM judge. The reference memory, HoldMem, crosses
two properties:

- **Binding B.** Each item keeps a key over the versions of its sources, its definition and its
  writer, and is invalidated and re-derived when the key changes.
- **Executability D.** Each item has a program that recomputes it: the authored derivation, or a
  model-written derivation D_m that the writer programs itself.

An accounting identity splits accuracy after a shift into what the key misses, what re-derivation
repairs and what the memory serves without invalidating.

Main results, with Qwen3.5-9B as writer and reader on 20 test worlds (95% world-bootstrap intervals):

- After a source revision the writer re-derived only 16.2% [10.1, 21.8] of changed definition items
  correctly, against 99.1% [98.0, 100.0] for the authored derivation. On these items binding was
  worth +6.0 pp [+2.5, +9.6] with the writer's own re-derivation and +33.3 pp [+28.2, +38.0] with the
  authored derivation, a binding-by-executability interaction of +27.3 pp [+21.6, +32.6].
- Programs that the writer wrote from a one-sentence definition kept most of the interaction:
  +25.3 pp [+18.8, +31.7] on accuracy over all changed items after a redefinition.
- Keys built from the records a writer cites missed 14.9% [11.7, 17.9] of changes after a source
  revision, mostly records that joined an item's dependencies after it was written.
- Of the 73 registered predictions with a verdict, 55 held, 13 failed and 5 held for some shifts,
  switches or systems but not others.

## Repository layout

```
src/memshift/
  gen/            seeded world generator: sources, items, versioned definitions, shift branches, gold
  probes/         point queries with an as-of time, gold value and gold status
  scoring/        typed matching (ContractValid) and bootstrap statistics
  holdmem/        HoldMem: writer, checker, binding, model-written programs, sandbox
  adapters/       zero system, plain RAG, oracle, native memory systems, Portable Memory export
  calsync/        incremental-sync semantics of the sources
  experiments/    experiment drivers
  analysis/       analyses, staging of runs, generators of numbers, tables and figures
configs/          model roles and fixed parameters, frozen with the registrations
prereg/           registrations v0 to v4b and the decision log
registration-history/   git objects that order the registrations before the runs, and a verifier
runs/             per run: manifest.json, env.json, analysis/ (raw outputs are on Hugging Face)
results/          sources.json: the staged analysis behind each group of reported numbers
audit/            the system audit and the neighbour audit
docker/           CPU run image and per-system images of the native memory systems
scripts/          vLLM serving, container runner, native image builder
tests/            pytest suite
docs/             COMPUTE.md, PREREGISTRATION_EN.md, E5_PROTOCOL.md
```

## Installation

MemShift needs Python 3.12 or later and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --dev                    # the environment of uv.lock
uv sync --dev --extra native     # adds the Mem0, Graphiti and Portable Memory adapters
```

A-Mem, Cognee, HippoRAG 2, LightRAG and MemOS run in their own Docker images, built with
`scripts/build_native.sh SYSTEM` from `docker/native/SYSTEM.txt`. The figure generator renders text
with LaTeX, so it needs a TeX installation with the `newtx` fonts.

## Quick start without GPUs

```bash
make smoke                                                 # generate one world branch, check its SHA256
uv run memshift gen --seed 1000 --branch src --out worlds  # one test world, source-revision branch
uv run python registration-history/verify.py               # order of the registrations
make check                                                 # ruff, mypy and pytest (about 45 minutes)
```

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

## Reproducing the paper

Reproduction has two tiers. Tier 1 recomputes every reported number on a CPU from the released
outputs. Tier 2 reruns the experiments from seeds on GPUs.

### Tier 1a: from the analyses in this repository (about one minute)

Each `runs/stage*/` directory holds a merged analysis (`analysis/*.json`) and a manifest that lists
its source runs with the SHA256 of their manifests. `results/sources.json` maps each group of
numbers to its stage. These commands rebuild the numbers, tables and figures:

```bash
mkdir -p results/tables results/figures
uv run python -m memshift.analysis.paper_numbers > results/numbers.tex
uv run python -m memshift.analysis.paper_tables --out results/tables
uv run python -m memshift.analysis.figures --out results/figures \
    --counts results/tables/registered_counts.tex
uv run python -m memshift.analysis.audit_table > results/tables/audit_table.tex
uv run python -m memshift.analysis.neighbour_table audit/neighbours/properties.json \
    results/tables/tab_neighbours.tex
```

The `.tex` outputs are byte-identical to the files the paper was built from. Tables 13 and 38 and the
`ExAttr*` and `ExThink*` macros also read 12 raw files (about 60 MB) from the Hugging Face dataset:
`runs/stage2_e1/analysis/e1_item_table.jsonl`, `runs/stage5_controls/analysis/ctl_item_table.jsonl`,
`probes.jsonl` of the five `runs/e1_test_s1*_20260928T*` runs and `items.jsonl` of the five
`runs/e1_think_*` runs. Without them those cells print a visible placeholder.

### Tier 1b: from the raw outputs (minutes per stage)

Download the raw outputs from the Hugging Face dataset into `runs/`. The paths are the same as in
this repository. `uv run memshift verify runs/RUN` re-hashes a run's files against its manifest.
Then rebuild a stage from the runs its manifest lists and compare it with the committed analysis.
For the main experiment:

```bash
uv run python -m memshift.analysis.stage e1 /tmp/stage2_e1 $(uv run python -c "import json; \
    print(' '.join('runs/' + s['run_id'] for s in json.load(open('runs/stage2_e1/manifest.json'))['sources']))")
cmp /tmp/stage2_e1/analysis/e1_metrics.json runs/stage2_e1/analysis/e1_metrics.json
```

The `experiment` field of a stage manifest (`merged-KIND`) names its kind.
`memshift.analysis.stage` builds the kinds e1, m2, e2, e3, e4, e4x, e5, e5_infer, e5_cost, e6, e7,
ctl and engine. The other stages are written by their own analysis modules (`e1_keys`, `ex_account`,
`ex_e1_cross`, `ex_e1_more`, `ex_misses`, `ex_round3`, `l_tokens`, `multiplicity`, `round3_paper`),
whose docstrings give the command.

### Map from the paper to the code

Table and figure numbers refer to the preprint. Stage keys are the keys of `results/sources.json`.
The table names the main stages of each result. Some generators also read further stages for
comparison rows, and `results/sources.json` lists them all.

| Paper | Generator | Stage keys (directories under `runs/`) | Tier 2 driver |
|---|---|---|---|
| Fig. 1 results panels, Table 1, Fig. 2 | `paper_numbers`, `paper_tables`, `figures` | `e1` (stage2_e1), `ex` (stage5_ex_e1), `ex_account`, `ex_round3` (stage6_ex_round3), `rep_e1` (stage4_rep_e1) | `experiments.e1` |
| Figs. 3, 5, Tables 14, 16, 19, 23, 29, 36 | `paper_tables` (`round3_paper`), `figures` | `round3` (stage7_round3), `ex_round3` | `experiments.e1`, `experiments.x_mem0` |
| Fig. 4, Tables 10, 11 (registered verdicts) | `figures`, `paper_tables` | `gate` (stage2_gate) and the stages of every registered test | as the tests |
| Tables 12, 13, 15, 17, 18, 20, 26, 27, 30, 31, Figs. 6, 7 (main experiment by item class) | `paper_tables`, `figures` | `e1`, `ex`, `ex_account`, `ex_more`, `ex_misses`, `ex_round3`, `rep_e1` | `experiments.e1` |
| Table 28 (offline key audit) | `paper_tables` | `keyaudit` (stage5_key_audit) | `analysis.e1_keys` (no model) |
| Tables 21, 22 (model-written derivations) | `paper_tables` | `c` (stage4_c_e1) | `experiments.e1 --cells D1sB0,D1sB1_source,D1sB1_cited` |
| Tables 24, 25, 38 (controls) | `paper_tables` | `ctl` (stage5_controls), `ex_round3` | `experiments.e1` |
| Fig. 8 (writer switches) | `figures` | `e2` (stage2_e2) | `experiments.e2` |
| Fig. 9 (horizons) | `figures` | `e3` (stage2_e3) | `experiments.e3` |
| Table 32 (other model families) | `paper_tables` | `m1` (stage4_m1_e2), `m2_llama`, `m2_gemma`, `m2_gptoss`, `m3` | `experiments.e2` (new writer), `experiments.e1 --writer --reader` |
| Table 33 (replication) | `paper_tables` | `rep_e1`, `rep_e2`, `rep_e3` and `e1`, `e2`, `e3` | `experiments.e1`, `e2`, `e3` |
| Table 34 (rewritten worlds) | `paper_tables` | `l` (stage4_l_rendered), `ltokens` | `experiments.e1 --render` |
| Fig. 10, Table 35 (native systems) | `figures`, `paper_tables` | `e5`, `e5wbshort`, `e5cost`, `n` (stage4_n_e5), `ncost` | `experiments.e5` |
| Table 37 (deletion failures of programs) | `paper_tables` | `ex_round3` | none (analysis of committed outputs) |
| Table 3 (neighbour audit), Table 5 (system audit) | `neighbour_table`, `audit_table` | `audit/neighbours/properties.json`, `audit/system_audit.json` | none |

Sync (`e4`, `e4x`), export (`e6`) and the failure-mode probes (`e7`) enter the paper through
`paper_numbers` and are rerun with `experiments.e4`, `experiments.e4x`, `experiments.e6` and
`experiments.e7`. Generators are modules of `memshift.analysis`, drivers of `memshift`.

### Tier 2: rerun from seeds on GPUs

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
   <!-- UNVERIFIED: needs GPUs and the model weights -->

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
   <!-- UNVERIFIED: needs the model servers -->

   The output goes to a new directory under `runs/`. Stage it and compare as in tier 1b.
   `scripts/run.sh` runs the same driver in a CPU container built from `docker/Dockerfile`
   (`RUN_MODULE=memshift.experiments.e1 MEMSHIFT_RUN_IMAGE=IMAGE scripts/run.sh ARGS`).
   <!-- UNVERIFIED: needs Docker and the model servers -->

Every run's `manifest.json` records its parameters, worlds, code commit and models, so any released
run can be repeated with its own settings. A rerun sends every request again. Greedy decoding does not
always return the same reply twice, so a rerun can differ slightly from the released outputs. The
released runs used about 140 A100 GPU-hours ([docs/COMPUTE.md](docs/COMPUTE.md)).

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
<!-- UNVERIFIED: needs the model servers -->

`python -m memshift.analysis.stage e5 OUT e5:RUN e1:RUN noise:RUN base:RUN` then infers the system's
(D, B) label from its answers and compares its accuracy after each shift with the HoldMem cell of
the same label and with plain RAG. For the HoldMem cells, `analysis/e1_item_table.jsonl` of each E1
stage records per changed item whether it was invalidated and whether its repair was right. A memory
that exposes its invalidations can be scored the same way.

## Data

- **Worlds and gold labels** are not stored. They regenerate from their seeds: development worlds
  0-9, calibration worlds 900-909, test worlds 1000-1019 and replication worlds 1020-1039. A world is
  identified by its seed and the generator version. Seeds 1000-1039 are evaluation seeds, so a system
  tuned on them is not comparable.
- **Run metadata** is in `runs/`: every run's `manifest.json` (code commit, worlds and their SHA256,
  parameters, output hashes) and `env.json` (serving image, model revisions), and the merged
  analyses of the stages. `runs/README.md` lists the voided runs.
- **Raw outputs** are in the Hugging Face dataset
  [`zichuanxu/memshift`](https://huggingface.co/datasets/zichuanxu/memshift): answers, probes,
  items and call logs of every run at the same paths as here, the response caches with the full
  text of every model reply in `cache/llm/`, and the logs of the native systems' stores. Download it
  into the root of this checkout. A rerun with the recorded settings then reuses the cached replies,
  since the cache key is the SHA256 of the request. The native systems' own
  database files are not released, and the run manifests here record their SHA256. The dataset card
  describes the files and the terms of each model that produced them.
- **Audits.** `audit/` holds the system audit and the neighbour audit ([audit/README.md](audit/README.md)).

## Preregistration

The predictions, metrics, contrasts and decision rules of each experiment were registered before its
runs. The paper reports every registered result, whether it held or failed.

- `prereg/` holds the registrations v0 to v4b and the decision log of amendments. v0, v1, v2 and the
  decision log are in Chinese, v3, v4 and v4b in English. They stay byte-identical to their
  registration commits.
- [docs/PREREGISTRATION_EN.md](docs/PREREGISTRATION_EN.md) is a non-normative English summary of every
  registration and of the decision log.
- `registration-history/` publishes the git objects that place each registration before the runs it
  concerns. `uv run python registration-history/verify.py` recomputes every object hash, follows the
  parent chain, finds each registration file in its commit and checks which run manifests descend
  from each registration. The chain fixes the order of the commits, not their dates.

## License

- Code: MIT ([LICENSE](LICENSE)).
- Data (generated worlds, gold labels, model outputs, audit data): CC BY 4.0. Model outputs are also
  subject to the terms of the model that produced them. The dataset card lists them.
- The open-weight models and the third-party memory systems are not redistributed and keep their
  own licenses.

## Citation

```bibtex
@misc{xu2026memshift,
  title  = {{MemShift}: Noticing a Change Is Not Repairing It in {LLM}-Derived Personal Memory},
  author = {Xu, Zichuan and Zhang, Haochen and Dong, Yuyang and Onizuka, Makoto and Xiao, Chuan},
  year   = {2026},
  note   = {Preprint}
}
```

`CITATION.cff` holds the same reference for GitHub's citation button.
