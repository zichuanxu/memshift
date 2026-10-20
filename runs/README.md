# runs

One directory per run, named after the experiment, its worlds or system, and the UTC start time. In
this repository a run directory holds `manifest.json` (code commit, worlds with their seeds and event
SHA256, parameters, models, SHA256 of every output file), `env.json` (serving image, model ids and
revisions) and `hash_check_*.json` (the hash assertions taken before each shift). The raw outputs
(JSONL) are in the Hugging Face dataset at the same paths. A directory named `stage*` holds a merged
analysis in `analysis/`; its manifest lists the source runs and the SHA256 of each of their
manifests.

The `manifest.json` of each of the 18 Cognee runs (`e5_cognee_*`, `e5_cognee_noise_*`) is 12-13 MB,
because `files_sha256` lists the 56,000 to 63,000 files of that run's Cognee store. These manifests
are in the Hugging Face dataset, and this repository keeps the runs' `env.json`. The manifest of
`stage4_n_e5` records their SHA256.

## Typed input, execution feedback and extracted fields

`stage9_round3b` merges the runs of three experiments on the definition items: `e1_t_qwen_*` and
`e1_t_sonnet_*` (typed input: Qwen3.5-9B or Claude Sonnet 5.5 re-derives a definition item from
every live record of its sources, given as typed fields), `e1_f2_qwen_*` (execution feedback: the
writer's programs get the interpreter's error back and may be rewritten) and `e1_w2_qwen_*`
(extracted fields: the programs read fields the writer extracted from prose). `e1_f_qwen_*` and
`e1_w_qwen_*` are runs of the same two experiments whose results the paper does not use; they are
kept because their GPU time counts in the compute total. They differ from `e1_f2_qwen_*` and
`e1_w2_qwen_*` in two points of the code: in them the sandbox's 2 s program limit counts wall
time, not the program's CPU time, and the writer extracts calendar records with one flat schema,
not one schema per record kind.

## Runs that no merged analysis reads

No staged analysis that `results/sources.json` names reads these 142 runs (`stage4_l_e1` reads the
`l_e1_*` runs, which are not among them). They are kept with their metadata, by kind:

| Runs | What they are |
|---|---|
| `e1_dev_a_*`, `e1_dev_b_*`, `e1_dev_cited_a_*`, `e1_dev_final_*`, `e1_dev_orphan_s0-4_*`, `e2_dev_*`, `e3_dev_*`, `e4_dev_s0-1_*` (14) | development runs of E1 to E4 |
| `smoke_*`, `l_render_warm_*` and three of the four `m1_*_warm_*` runs (5) | a smoke run and warm-up runs of the model servers |
| `e5_measure2_mem0_*` (1) | a cost measurement of Mem0 on 100 events of one world |
| `e7_theta_r_s1000_*`, `e7_theta_r_s1001-1002_*`, `e7_theta_r_s1003-1004_*` (3) | the swapped reader of E7 served by vLLM 0.20.0, another engine version than the other servers; the engine comparison `stage3_theta_r_engine` reads them |
| `e1_f_qwen_*`, `e1_w_qwen_*` (10) | runs of the execution-feedback and extracted-fields experiments with the code described above |
| 102 of the `e1_ga_heldout*`, `e1_gb_*` and `e1_v_*` runs | held-out-definition, program-set and user-words runs whose setting (writer, sample, definitions, worlds) has a run with a later timestamp, which the stages read |
| `e1_synth_s1016-1019_20260929T150913Z` and the `m2_meta-llama_llama-3_1-8b-instruct_*` and `m2_openai_gpt-oss-20b_*` runs of worlds 1000-1001, 1002-1003 and 1006-1007 timestamped 20260930 (7) | E1 runs whose writer and worlds have a run with a later timestamp, which the stages read |
