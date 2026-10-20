# Results

`sources.json` maps each group of reported numbers to the staged analysis it is computed from. A key
names a group, and its value is a directory under `runs/` whose `analysis/` folder holds the merged
analysis. The generators compute every measured number, generated table and figure from these
directories, and the two audit tables from `audit/`.

## Who reads `sources.json`

- `memshift.analysis.paper_numbers`: the numbers in the text, as LaTeX macros (`numbers.tex`).
- `memshift.analysis.paper_tables`: the tables. It loads the stages through `paper_data`,
  `ex_e1_paper`, `ex_review_paper`, `ex_tables`, `controls_paper`, `key_audit`, `ex_round2`,
  `ex_round3` and `round3_paper`.
- `memshift.analysis.figures` and `memshift.analysis.round3_figures`: the figures and the table of
  the four diagnostic tests.
- `memshift.analysis.paper_extras`: the overview table, the worked example, the models table
  and the setup facts (`setup_facts.tex`: Claude model identifiers, call dates, tokens and
  cost, GPU model and GPU-hours). It also reads the `manifest.json` and `env.json` of every run.
- The tests that check the generators against the committed analyses: `tests/conftest.py`,
  `test_controls_paper.py`, `test_ex_e1_paper.py`, `test_ex_paper.py`, `test_ex_review_paper.py`,
  `test_ex_round2.py`, `test_ex_round3.py`, `test_ex_round3_review.py`,
  `test_ex_round3_writers.py`, `test_key_audit.py`, `test_paper_extras.py`,
  `test_paper_stage4.py`, `test_round3_figures.py` and `test_round3_paper.py`.

`memshift.analysis.paper_round3b` writes the macros (`round3b_numbers.tex`) and tables of every
record as typed fields, execution feedback and extracted fields. It reads the analyses of the
`round3b` stage from the directory given by `--analysis-dir`.

`make results` and the commands in the README of the public package write their outputs to
`results/numbers.tex`, `results/tables/` and `results/figures/`.

## Keys

| Key | Directory | Stage kind | What it holds |
|---|---|---|---|
| `dev_e1`, `dev_orphan_e1` | `stage1_e1`, `stage1_orphan` | e1 | main experiment on the development worlds |
| `e4` | `stage1_e4` | e4 | sync on the development worlds |
| `testset` | `testset_20260928T183400Z` | testset | SHA256 and fidelity check of every test-world branch |
| `e1` | `stage2_e1` | e1 | main experiment on the test worlds |
| `e2` | `stage2_e2` | e2 | writer switch at the three tiers |
| `e3` | `stage2_e3` | e3 | horizons of 1, 3 and 12 months |
| `gate` | `stage2_gate` | gate | binding's effect on Update pooled over D, D's effect on missed invalidation, writer agreement as a predictor of survival after a switch (`memshift.analysis.gate`) |
| `e1nofb` | `stage3_e1_nofb` | e1 | check-only cell |
| `e4x` | `stage3_e4x` | e4x | official sync semantics on all sources |
| `e5`, `e5cost`, `e5wb`, `e5wbshort` | `stage3_e5`, `stage3_e5_cost`, `stage3_e5_whitebox`, `stage3_e5_whitebox_short` | e5, e5_cost, e5_infer | Mem0, native ingestion cost, label inference on the HoldMem cells |
| `e6` | `stage3_e6` | e6 | export and import (leaving test) |
| `e7` | `stage3_e7` | e7 | failure-mode probes |
| `rep_e1`, `rep_e2`, `rep_e3` | `stage4_rep_e1`, `stage4_rep_e2`, `stage4_rep_e3` | e1, e2, e3 | replication on worlds 1020-1039 |
| `pool_e1`, `pool_e2`, `pool_e3` | `stage4_pool_e1`, `stage4_pool_e2`, `stage4_pool_e3` | | pooled 40-world estimates. Not in this release, so their macros print a placeholder (the paper uses none of them) |
| `c` | `stage4_c_e1` | e1 | model-written derivations |
| `m1` | `stage4_m1_e2` | e2 | writer switches to other model families |
| `m2_llama`, `m2_gemma`, `m2_gptoss` | `stage4_m2_*` | m2 | other model families as writer and reader |
| `m3` | `stage4_m3` | e7 | other model families as reader |
| `n`, `ncost` | `stage4_n_e5`, `stage4_n_cost` | e5, e5_cost | six native memory systems (Mem0, A-Mem, MemOS, LightRAG, HippoRAG 2, Cognee) and their cost |
| `l`, `ltokens` | `stage4_l_rendered`, `stage4_l_tokens` | rendered, l_tokens | rewritten worlds and the share of rewritten tokens |
| `rank` | `stage4_rank_corr` | rank_corr | rank correlation across shifts |
| `mult` | `stage4_multiplicity` | multiplicity | multiplicity corrections |
| `short_e1` | `stage5_short_e1` | e1 | HoldMem cells on the three-month worlds |
| `ex`, `ex_account`, `ex_more`, `ex_misses` | `stage5_ex_e1`, `stage5_ex_account`, `stage5_ex_more`, `stage5_ex_misses` | ex_e1, ex_account, ex_e1_more, ex_misses | the main experiment by item class, the accounting identity, further splits, the misses of the cited-cluster key |
| `keyaudit` | `stage5_key_audit` | key_audit | offline audit of the source-epoch, oracle and scope keys (no model) |
| `ctl` | `stage5_controls` | ctl | controls: the arms that vary one confound at a time (other keys, B=0 variants and writer inputs), test worlds 1000-1009 |
| `ex_round3` | `stage6_ex_round3` | ex_round3 | store and reader split, natural-branch composition, deletion failures of programs, query-time view |
| `round3` | `stage7_round3` | round3 | refresh and upsert, Claude writers, held-out definitions, program sets from six writers, Mem0 interventions, unnamed probes, natural branch, Claude reader, definitions in the user's own words |
| `ex_round3_review` | `stage8_ex_round3_review` | ex_round3_review | refresh against binding, shift and drift splits, clock-only changes, the writer's evidence coverage, typed-input sizes, forced re-derivation on the cited-cluster key's misses, and the estimates the design fixes (`fixed.json`) |
| `round3b` | `stage9_round3b` | round3b_typed, round3b_feedback, round3b_extract | every record as typed fields, execution feedback, extracted fields (read by `memshift.analysis.paper_round3b`) |

A stage's `manifest.json` lists its source runs with the SHA256 of their manifests, the code that
built it and the SHA256 of each analysis file. Its `experiment` field (`merged-KIND`) names the
stage kind. `stage2_gate`, `stage4_l_rendered` and `stage4_rank_corr` have no stage manifest. They
are written by `memshift.analysis.gate`, `memshift.analysis.rendered` and
`memshift.analysis.rank_corr`, whose docstrings give their inputs.

The `analysis/*_item_table.jsonl` files (per-item tables of the main-experiment and control stages)
are large and live in the Hugging Face dataset at the same paths.
