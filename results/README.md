# Results

`sources.json` maps each group of reported numbers to the staged analysis it is computed from. A key
names a group, and its value is a directory under `runs/` whose `analysis/` folder holds the merged
analysis. The generators compute every measured number, generated table and figure from these
directories, and the two audit tables from `audit/`.

## Who reads `sources.json`

- `memshift.analysis.paper_numbers`: every number in the text, as LaTeX macros.
- `memshift.analysis.paper_tables`: the tables. It loads the stages through `paper_data`,
  `ex_e1_paper`, `ex_review_paper`, `ex_tables`, `controls_paper`, `key_audit`, `ex_round2`,
  `ex_round3` and `round3_paper`.
- `memshift.analysis.figures`: the figures, with `round3_figures` for the registered verdicts.
- The tests that check the generators against the committed analyses: `tests/conftest.py`,
  `test_controls_paper.py`, `test_ex_e1_paper.py`, `test_ex_paper.py`, `test_ex_review_paper.py`,
  `test_ex_round2.py`, `test_ex_round3.py`, `test_ex_round3_writers.py`, `test_key_audit.py`,
  `test_paper_stage4.py`, `test_round3_figures.py` and `test_round3_paper.py`.

The commands in the top-level README write their outputs to `results/numbers.tex`,
`results/tables/` and `results/figures/`.

## Keys

| Key | Directory | Stage kind | What it holds |
|---|---|---|---|
| `dev_e1`, `dev_orphan_e1` | `stage1_e1`, `stage1_orphan` | e1 | main experiment on the development worlds |
| `e4` | `stage1_e4` | e4 | sync on the development worlds |
| `testset` | `testset_20260928T183400Z` | testset | SHA256 and fidelity check of every test-world branch |
| `e1` | `stage2_e1` | e1 | main experiment on the test worlds |
| `e2` | `stage2_e2` | e2 | writer switches at the three registered tiers |
| `e3` | `stage2_e3` | e3 | horizons of 1, 3 and 12 months |
| `gate` | `stage2_gate` | gate | falsification checks and claim level (`memshift.analysis.gate`) |
| `e1nofb` | `stage3_e1_nofb` | e1 | check-only cell |
| `e4x` | `stage3_e4x` | e4x | official sync semantics on all sources |
| `e5`, `e5cost`, `e5wb`, `e5wbshort` | `stage3_e5`, `stage3_e5_cost`, `stage3_e5_whitebox`, `stage3_e5_whitebox_short` | e5, e5_cost, e5_infer | Mem0, native ingestion cost, label inference on the white-box cells |
| `e6` | `stage3_e6` | e6 | export and import (leaving test) |
| `e7` | `stage3_e7` | e7 | failure-mode probes |
| `rep_e1`, `rep_e2`, `rep_e3` | `stage4_rep_e1`, `stage4_rep_e2`, `stage4_rep_e3` | e1, e2, e3 | replication on worlds 1020-1039 |
| `pool_e1`, `pool_e2`, `pool_e3` | `stage4_pool_e1`, `stage4_pool_e2`, `stage4_pool_e3` | | pooled 40-world estimates (secondary). Not in this release, so their macros print a placeholder |
| `c` | `stage4_c_e1` | e1 | model-written derivations |
| `m1` | `stage4_m1_e2` | e2 | writer switches to other model families |
| `m2_llama`, `m2_gemma`, `m2_gptoss` | `stage4_m2_*` | m2 | other model families as writer and reader |
| `m3` | `stage4_m3` | e7 | other model families as reader |
| `n`, `ncost` | `stage4_n_e5`, `stage4_n_cost` | e5, e5_cost | six native memory systems (Mem0 and the five of block N) and their cost |
| `l`, `ltokens` | `stage4_l_rendered`, `stage4_l_tokens` | rendered, l_tokens | rewritten worlds and the share of rewritten tokens |
| `rank` | `stage4_rank_corr` | rank_corr | rank correlation across shifts (exploratory) |
| `mult` | `stage4_multiplicity` | multiplicity | exploratory multiplicity corrections |
| `short_e1` | `stage5_short_e1` | e1 | white-box cells on the 3-month worlds |
| `ex`, `ex_account`, `ex_more`, `ex_misses` | `stage5_ex_e1`, `stage5_ex_account`, `stage5_ex_more`, `stage5_ex_misses` | ex_e1, ex_account, ex_e1_more, ex_misses | exploratory analyses of the main experiment |
| `keyaudit` | `stage5_key_audit` | key_audit | offline audit of the source-epoch, oracle and scope keys (no model) |
| `ctl` | `stage5_controls` | ctl | the control cells of `prereg/v3_controls.md` |
| `ex_round3` | `stage6_ex_round3` | ex_round3 | store/reader split, natural-branch composition, deletion failures of programs, query-time view |
| `round3` | `stage7_round3` | round3 | the blocks of `prereg/v4_round3.md` and `prereg/v4b_round3_claude.md` |

A stage's `manifest.json` lists its source runs with the SHA256 of their manifests, the code that
built it and the SHA256 of each analysis file. Its `experiment` field (`merged-KIND`) names the
stage kind. `stage2_gate`, `stage4_l_rendered` and `stage4_rank_corr` have no stage manifest. They
are written by `memshift.analysis.gate`, `memshift.analysis.rendered` and
`memshift.analysis.rank_corr`, whose docstrings give their inputs.

The `analysis/*_item_table.jsonl` files (per-item tables of the E1 and control stages) are large and
live in the Hugging Face dataset at the same paths.
