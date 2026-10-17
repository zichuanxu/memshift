# Preregistration: English summary

**The registrations are the files in `prereg/`.** `v0.md`, `v1.md`, `v2.md` and the decision log
`decisions.md` are written in Chinese. `v3_controls.md`, `v4_round3.md` and `v4b_round3_claude.md`
are written in English. This document is a non-normative summary for English readers. Where it
differs from a registration, the registration holds. The files in `prereg/` stay byte-identical to
the commits that registered them, and `registration-history/verify.py` checks this.

## Registrations and their commits

Each registration is identified by the commit that adds it, as listed in
`registration-history/README.md`.

| Registration | File | Language | Commit | What it registers |
|---|---|---|---|---|
| v0 | `prereg/v0.md` | Chinese | `f469617` | stage 1: parameters, metrics, predictions, falsification paths, rules of gate A0 |
| v1 | `prereg/v1.md` | Chinese | `2051086` | test set, final thresholds, contrast list, equivalence margins, gate A; `docs/E5_PROTOCOL.md` frozen with it |
| v2 | `prereg/v2.md` | Chinese | `525f7ab` | stage 4: blocks R, M, N, C, L |
| v3 | `prereg/v3_controls.md` | English | `a4b5474` | control cells added after review |
| v4 | `prereg/v4_round3.md` | English | `72261c0` | blocks U, H, G, X, Q, N |
| v4b | `prereg/v4b_round3_claude.md` | English | `02567d6` | blocks Y and V |

`prereg/v3_controls.md` gained its section 6 (readings of pending results and the control D0full)
in a later commit, `b7c0bc0`, before the results it concerns. The verifier reports this file as
"edited later". The amendments of all registrations are rows of `prereg/decisions.md`, summarised at
the end of this document.

## Terms used in the registrations

The registrations use the internal names of the research plan. The paper uses other names for some
of them.

| Registrations | Paper |
|---|---|
| D (determinism) | executability D |
| E1, E2, E3, E4, E5, E6, E7 | main experiment, writer switch, horizon, sync, native systems, export, failure-mode probes |
| S_src, S_def, orphan branch, S_wr, S_impl | source revision, redefinition, deletion, writer switch, export |
| S_cons | consent revocation |
| T1, T2, T3, T4 | retention test, leaving test, utility floor, cost ceiling |
| falsification paths 1, 2, 3 | gain check, leak check, agreement check |
| s_k | the item's one-sentence statement |
| κ (kappa) | a person's definition |
| θ_w, θ_r, θ_e | writer, reader, embedding model |
| source-epoch granularity, cited (object-cluster) granularity | source-epoch key, cited-cluster key |
| D1s | model-written derivation D_m |
| gate A0, gate A, gate B | decision points after stages 1, 2 and 3 |
| card B, card C | alternative framings of the paper, used only if a gate selects them |

Block N means native systems in v2 and the natural branch in v4.

## v0: end of stage 0

v0 freezes the stage-1 parameters, metric definitions, predictions, falsification paths and the
rules of gate A0. Stage 1 reports estimates and interval widths only, without equivalence tests. The
test-set size, equivalence margins and final thresholds are left to v1. Any deviation is first
written to `decisions.md`.

**Parameters.**
- ε = 3 pp, c = 0.5, q = 20 queries per day.
- Worlds of 12 months with about 2M tokens of source text.
- B=0 is append-only. Two scope granularities: source epoch and object-cluster epoch.
- Fork at the end of month 9 (2026-10-01T00:00). Probe times k = 1, 2, 4 and 8 weeks.
- Reader evidence budget of 4,096 tokens for every system.
- Development set of 10 worlds (seeds 0-9). Test-set starting point N = 40.
- Two ingestions per world for a noise floor.

**Stage-0 measurements.** One 3-month smoke world with HoldMem (1,1), Qwen3.5-9B as writer and
reader, gives the cost per ingested event and per probe and a checker pass rate. Extrapolated scale:
about 1,480-1,620 events and 2.2M tokens per 12-month world, 48 knowledge items (6 user attributes,
24 contact attributes, 4 recurring schedules, 2 appointments, 6 tasks, 6 definitions). Compute
budgets: stage 1 at most 12 GPU-hours, stages 2 and 3 at most 24 GPU-hours each. Over budget, the
number of worlds is cut, not the cells.

**Metrics** (paired by knowledge item, clustered by world). For world w, branch b, item k and
checkpoint t_k, compare with t0 = t_f − 1 minute.
- Changed item: its gold value or status at t_k differs from t0. Unchanged item: it does not.
- Detected: the cell invalidated k's original item in (t0, t_k].
- MissedInvalidation = P(not detected | changed). FalseInvalidation = P(detected | unchanged).
- RepairCorrectness (primary) = P(probe at t_k is ContractValid | k repaired at t_k). Mechanical
  variant: the repaired value typed-matches the gold at t_k.
- Update and Stability: ContractValid on changed and on unchanged items of the discriminative
  probes. Discriminative probes are all changed items plus an equal number of unchanged items
  sampled by seed, per world, branch and k.
- Retention (S_wr) = P(ContractValid after | ContractValid before), same probes before and after,
  sources frozen at t0.
- ContractValid: the status matches and, for status `answer`, the value matches by type. No LLM
  judge.

**Predictions.**
- S_src and S_def: missed invalidation varies only with B and granularity. Repair correctness
  varies only with D.
- S_wr: B=1 items are all invalidated by construction. B=0 items are not invalidated, so survival
  is about 1. For B=1, per-item survival of D=0 items is predicted by the per-item agreement rate,
  and survival of D=1 items is about the checker pass rate (with fallback).
- S_impl (stage 3): with the derivation extension B and D are kept. Without it the import falls
  back to B=0.

**Falsification paths.**
1. Binding effect too small. B effect = Update(B=1, source epoch) − Update(B=0), pooled over D,
   S_src primary and S_def secondary. Point estimate ≤ +0.02: detection is not worth a theory, and
   the paper falls back to a repair-only account.
2. D leaks into detection. D effect on missed invalidation = MissedInvalidation(D=0, B=1) −
   MissedInvalidation(D=1, B=1), for each granularity. |point estimate| ≥ 2 pp with a 95% interval
   that excludes 0: check the sidecar for leaks, fix, and rerun E1. If it cannot be fixed, the paper
   becomes a measurement-method paper. The 2 pp threshold is provisional until v1.
3. Quantitative prediction fails. Logistic regression of per-item survival of D=0, B=1 items on the
   per-item agreement rate. The slope's 90% world-bootstrap lower bound ≤ 0, or AUROC < 0.60: the
   quantitative prediction becomes qualitative.

If all three fail, the paper becomes "MemShift measurement method plus the zero-system result".

**Contrast list.** The list is fixed. Anything else is exploratory and labelled so.
- E1, per branch: B=1 against B=0 (D fixed, both granularities) on detection and Update; D=1
  against D=0 (B=1 fixed, both granularities) on missed and false invalidation and on
  RepairCorrectness (behavioural and mechanical); the two granularities on false and missed
  invalidation; the JSON arm against the s_k arm on ContractValid before the intervention.
- E2: three tiers (same, up, down) × D on Retention; calibration slope and AUROC at D=0; D=1
  survival against the checker pass rate.
- E3: zero system, HoldMem(1,1), RAG and oracle at horizons of 1, 3 and 12 months: static
  ContractValid, Update and Stability after S_src and S_def (from E1), five costs.
- E4: {ideal, official} × {B=0, B=1} × {HoldMem(1,·), zero system}, lag L ∈ {1, 5, 30} minutes:
  ContractValid and state divergence.

**Rules of gate A0.**
- E1: report estimates and 95% world-bootstrap interval widths for missed and false invalidation,
  repair correctness, and Update and Stability on the discriminative probes. Judge "missed
  invalidation varies only with B" (path 2 not triggered), "repair varies only with D" (the interval
  of D's effect on repair excludes 0), and "the JSON arm shows a presentation effect" (the interval of
  JSON − s_k before the intervention excludes 0, and the six cells' answers before the intervention
  are identical). The stage-2 N is the number of worlds that brings the 95% interval width of the main
  contrasts to at most 0.10 (extrapolated by 1/√n), capped by the budget.
- E2: the direction of the D=0 calibration slope (point estimate and 90% lower bound).
- E3: the zero system passes if and only if (a) its 12-month static ContractValid ≥ that of
  HoldMem(1,1) − 3 pp, (b) its Update after S_src and S_def ≥ that of HoldMem(1,1) − 3 pp, and
  (c) its serving cost ≤ 0.5 × a full re-read per query. A pass switches the paper to card B (change
  rate as the intervention) and requires a new registration.
- E4: interaction = [ContractValid(official) − ContractValid(ideal)] at B=1 minus the same at B=0.
  An interval that excludes 0 with |point estimate| ≥ 3 pp makes sync semantics a main axis.
  Otherwise sync is an external-validity section only.
- If D's effect shows up in missed invalidation, path 2 is handled before the gate is passed.

**Statistics.** Item-paired differences clustered by world, world bootstrap with 2,000 draws (seed
996), 95% intervals. N and the equivalence margins (TOST, α = 0.05) follow in v1.

## v1: after gate A0, before the test worlds were generated

v1 inherits every parameter, metric, prediction and falsification path of v0. It fixes what v0 left
open and the design revisions of stage 1.

**Test set.** Seeds 1000-1019, N = 20, disjoint from the development (0-9) and calibration (900-909)
worlds. By the v0 rule, the largest number of worlds needed is 18 (B effect on the orphan branch,
width 0.132); the other main contrasts need at most 11. Worlds are generated deterministically after
this commit, and their SHA256, the code commit and the model revisions go into each run's manifest.
Each test world gets a second ingestion for the noise floor. Stage-3 worlds come from the test set:
E5 from 1000 (the number set by measured native cost), E6 all 20, E7 and the sync extension
1000-1004.

**Design revisions from stage 1** (each recorded in `decisions.md`).
1. The second granularity is the cluster of the writer's citations ("cited"). Read-set clusters do
   not enter the test set. Both D arms take clusters from the writer's raw citations.
2. E1 probes: all 48 items before the fork, discriminative probes after it. Detection metrics use
   the gold and item history of every item.
3. E1 cells: D ∈ {0,1} × {B=0, B=1 source epoch, B=1 cited}. JSON arm on D=1, B=1 source epoch.
   Baselines: zero system, RAG, oracle. Branches: S_src (primary), S_def, orphan (a dependency is
   deleted with no new evidence).
4. B=0 append-only makes no writer call when the delta holds no new in-scope record.
5. The D=1 fallback statement of a definition item names the definition version and its statement.
6. E4 samples series, appointment and settings changes first, then one-off meetings.

**Final falsification thresholds.**
1. Path 1: B effect = Update(B=1 source epoch) − Update(B=0), pooled over D, judged on S_src. A
   point estimate ≤ +0.02 triggers it. S_def and orphan are reported and not judged.
2. Path 2: D's effect on missed invalidation for both granularities and the three branches. Any
   |point estimate| ≥ 2 pp with a 95% interval that excludes 0 triggers it (2 pp final). Stage 1 gave
   exactly 0 in all six.
3. Path 3: logistic regression of per-item survival (D=0, B=1 source epoch) on the old/new writer
   agreement rate of the item's type on the calibration worlds. Primary judgement pools the three
   tiers with a world bootstrap. A 90% lower bound of the slope ≤ 0 or AUROC < 0.60 triggers it.
   Each tier is reported. If the pooled test passes but a tier's slope estimate is ≤ 0, the
   quantitative claim is marked "does not apply to this tier".

**Contrast list** (changes to v0).
- E1 (S_src, S_def, orphan): as in v0, plus two additions. First, the B×D interaction on Update,
  [B effect | D=1] − [B effect | D=0], reported on S_src and S_def, predicted > 0 (secondary, with
  direction). Stage-1 values (source epoch): S_def +30.3 pp [26.0, 34.5] (B effect +31.0 pp at D=1,
  +0.7 pp at D=0); S_src +10.8 pp [7.2, 14.6]. Second, the B effect on the orphan branch, pooled
  over D.
- E2 (test worlds 1000-1019, calibration worlds 900-909; same, up, down): T1; calibration slope and
  AUROC at D=0 (path 3); D=1 survival against the checker pass rate. B=0 survival is reported
  separately and not in T1, since B=0 does not re-derive after a switch.
- E3 (test worlds): zero system, HoldMem(1,1), RAG, oracle at 1, 3 and 12 months: static
  ContractValid, Update and Stability after S_src and S_def, five costs. Judges T3 (ε = 3 pp against
  the oracle) and T4 (c = 0.5).
- E6 (stage 3): T2, plus an export arm without the derivation extension, predicted to fall back to
  B=0 detection.
- E4 extension (stage 3): official sync semantics for all six sources. Cells {ideal, official L = 5
  minutes} × B ∈ {0,1} (D=1) × {HoldMem, zero system}, discriminative probes. Replication on the
  calendar is the main result, the other five sources are descriptive.

**Equivalence margins** (TOST, α = 0.05: the 90% world-bootstrap interval lies within ± margin).
Both margins are ε = 3 pp, fixed by the plan and not derived from stage 1.
- T1 (writer-switch equivalence): per tier, Δ = Retention(tier, HoldMem(1,1) source epoch) − 1,
  with the 90% interval inside (−3, +3) pp. Retention ≤ 1, so this is a non-inferiority test. The
  noise floor is reported beside it. T1 is judged per tier. Stage 1, same tier, 10 worlds: Δ =
  −2.2 pp, 90% [−3.2, −1.2], which fails T1 on the development set.
- T2 (leaving equivalence): paired on world, branch and probe, the 90% intervals of three
  differences lie inside (−3, +3) pp: ContractValid(import with extension) − ContractValid(original)
  at t0; the same after S_src and S_def (k pooled); MissedInvalidation(import) −
  MissedInvalidation(original). ExportFidelity = ContractValid(import) / ContractValid(original) is
  reported as a description.

**Gate A: the level of the paper's claim.**

| Path 1 | Path 2 | Path 3 | Claim level |
|---|---|---|---|
| not triggered | not triggered | not triggered | two-factor account with quantitative prediction |
| not triggered | not triggered | triggered | two-factor account, qualitative prediction |
| triggered | not triggered | any | repair only (D) |
| any | triggered and not fixable | any | measurement-method paper |
| triggered | triggered | triggered | measurement method plus zero-system result |

T1-T4 decide whether HoldMem meets the definition of shift-safe (membership), reported separately
from the claim level.

**Statistics.** As v0: item-paired differences, clustered by world, world bootstrap with 2,000 draws
(seed 996), 95% intervals for estimates and 90% intervals for TOST and path 3. The contrast list is
fixed and each path is an independent decision rule, without a multiplicity correction. Every listed
contrast is reported whatever its direction.

**Stage-2 order.** Commit v1. Run one 27B window (tensor parallel over two GPUs) that pre-computes
the Qwen3.8-27B writes on the development, test and calibration worlds. Run E1, E2 and E3 on the 20
test worlds. Merge the analyses and judge gate A.

**E5 protocol** (`docs/E5_PROTOCOL.md`, in Chinese, frozen with v1). Black-box inference of (D, B)
for native systems.
- B probe (orphan test): the gold dependency objects of structured items are deleted as source
  events (tombstone text). Orphan persistence P = P(the system still gives the old value | dependency
  deleted). P ≤ 0.5 gives B=1, otherwise B=0.
- D probe (structured repair precision): after a value-changing source revision, R = P(updated
  value is ContractValid | the system gave a new value). R ≥ 0.98 and an export that keeps typed
  values with locatable source spans gives D=1, otherwise D=0.
- Labels are {0, 1, unidentifiable}. Fewer than 20 valid samples, or a configuration that keeps no
  raw text or source metadata, makes a label unidentifiable. Inference and measurement use disjoint
  halves of the items.
- White-box validation: the probes are applied to HoldMem's four source-epoch cells on the
  development worlds. A dimension with Cohen's κ < 0.7 is only descriptive for native systems.
- Sufficient-statistic test: a native system's carryover is compared with the 95% interval of the
  white-box cell with the same inferred (D, B) on the same worlds and shifts. Inside means
  consistent, outside is reported as a boundary of the account.
- Cost control: one prefix ingestion per system and world, then a snapshot from which every branch
  starts. Native cost is measured on one world first. The noise floor uses two ingestions on 3 worlds.

## v2: stage 4

v2 inherits v0 and v1, including the gate-A judgement, which stage 4 does not revisit. It answers
four expected objections. The main result comes from the authors' own reference memory, with D
computed like the gold. Only Mem0 was tested as a native system. Only one model family was used.
The world text is templated. No stage-4 run starts before this commit.

**Shared rules.** Statistics as v1 (item-paired, world-clustered, 2,000 draws, seed 996, 95%
intervals, 90% for equivalence and path 3). The contrast list of each block is fixed. Model
families and systems are judged one by one and all reported, without a multiplicity correction, with
the number of families or systems that hold. No LLM judge. Reader template and 4,096-token budget
unchanged. Each block first runs one world and checks its manifest.

**Block R: independent replication.** Seeds 1020-1039 (20 new worlds). The frozen stage-2 design:
E1, E2 (calibration worlds 900-909, agreement rates not recomputed) and E3 (1, 3, 12 months).
Every contrast of the v1 list is estimated on R. A predicted non-zero effect replicates if it has
the same sign with a 95% interval that excludes 0. The same sign with an interval that includes 0 is
"same direction, not significant", and the opposite sign does not replicate. Equivalence tests (T1, T3)
replicate if the verdict equals stage 2's. The three paths are re-judged on R. The 40-world pooled
estimate is secondary.

**Block M: model families.** Llama-3.1-8B-Instruct, gemma-4-12B-it, gpt-oss-20b (reasoning
effort low), Ministral-3-14B-Instruct-2512-BF16, Qwen3-8B, all at temperature 0 with JSON-schema
decoding.
- M1, writer switch across families: old writer Qwen3.5-9B, new writer one of the five, agreement
  per type on the calibration worlds, four cells on the test worlds.
  - M1-a: pooling the three stage-2 tiers with the five switches (eight switches), path 3 is still
    not triggered (slope 90% lower bound > 0, AUROC ≥ 0.60).
  - M1-b: in every switch, D=1 survival ≥ D=0 survival (B=1 source epoch).
  - T1 per switch, without a predicted direction.
- M2, another family as writer and reader in E1: Llama-3.1-8B, gemma-4-12B, gpt-oss-20b on worlds
  1000-1009. Per family: (i) path 2, D has no effect on missed invalidation; (ii) B effect on Update
  after S_src > +2 pp; (iii) D effect on repair > 0 (S_src and S_def); (iv) B×D after S_def > 0;
  (v) after S_def the zero system's ContractValid is below HoldMem(1,1)'s. The two-factor pattern
  holds for a family if (ii)-(iv) have 95% intervals that exclude 0 and the predicted sign.
- M3, reader swap: Llama-3.1-8B, gemma-4-12B, gpt-oss-20b as readers on worlds 1000-1004, S_src and
  S_def. Prediction: the four registered contrasts (B effect, D on repair, B×D, D on missed
  invalidation) have the main reader's sign under every reader.

**Block N: native systems.** A-Mem (official code, fixed commit), Letta (`letta==0.33.6`), MemOS
(`MemoryOS==2.0.33`), LightRAG (`lightrag-hku==1.5.7`), HippoRAG 2 (`hipporag==2.0.0a4`);
substitutes in order Cognee (`cognee==1.6.1`) and SimpleMem (`simplemem==0.1.0`). A system that
cannot run one short world under the ingestion contract within one engineering day, or that costs
more than 3 times Mem0 per event, is replaced by the next substitute. Graphiti stays cost-only.
Worlds as E5: short worlds 1000-1014 (3 months), one prefix ingestion and snapshot, three branches,
and a second ingestion on 1000-1002. White-box references: the four source-epoch cells, plus the cited
D0B1 and D1B1 cells on the same short worlds.
- Labels predicted from each system's documentation: A-Mem (0,0), Letta (0,0), MemOS (0,0),
  LightRAG (D=0, B=1 at document level), HippoRAG 2 (D=0, B=1 at document level), Cognee (0,1),
  SimpleMem (0,0).
- N-1: the inferred label matches the documented label for at least 4 of 5 systems.
- N-2 (per system): of 6 comparisons (S_src, S_def, orphan × Update, Stability), at least 4 fall in
  the 95% interval of the white-box cell with the inferred label.
- N-3 (all systems, with Mem0): the share of comparisons inside the interval is at least 0.5.
  Below 0.5, the paper withdraws "(D, B) is a sufficient statistic of native carryover".
- N-4: if both B=0 and B=1 are inferred, B=1 systems have higher orphan Stability on average.
- Membership: T3 against the oracle on the same short worlds and T4 from metered tokens. T1 does
  not apply. T2 for exports without the derivation extension is predicted as B=0 (not run).

**Block C: the writer's own executable derivation (D1s).** The checker is D=1's, but a failed check
is replaced by the output of a program the current writer wrote, `derive(records, as_of)`, run in a
restricted interpreter (AST allowlist without imports, dunder attributes, open, eval, exec,
compile or globals; allowed built-ins plus `datetime`, `timedelta`, `parse_ts`; 2-second timeout).
Errors, timeouts or wrong types make the item unverifiable. Definition items are checked against the
program instead of the gold evaluator. Cells D1s × {B=0, B=1 source epoch, B=1 cited} on worlds
1000-1019, same write cache as E1.
- C1: repair D1s − D0 (B=1 source epoch) > 0 with a 95% interval that excludes 0, after S_src and
  S_def.
- C2: D1 − D1s ≥ 0, with its share of the original D effect reported.
- C3: B×D on Update after S_def > 0 with D1s as D=1.
- C4: path 2 holds on the D1s cells.

**Block L: LLM-rendered worlds.** gemma-4-12B-it (temperature 0) rewrites e-mail, chat and note
bodies. Headers and structured records stay. Every rewritten body must contain the surface form of
every gold fact it carries, or it is retried once and otherwise kept as template text. Worlds
1000-1009, with M2's cells, branches and probes and the main writer and reader. Predictions (i)-(v)
of M2 are judged on these worlds, and paired differences from the template worlds of the same seeds
are reported.

**Neighbour comparison** (no runs): a property table of the nearest benchmarks, each property quoted
from its paper. Exploratory: Spearman correlation between the ranking by Update after S_src and the
rankings by Update after S_def and by orphan Stability, over all systems tested on the short worlds.

**Budget and cuts.** At most 80 GPU-hours (estimate about 70). Over budget, cut in this order: M2 to
6 worlds, L to 5 worlds, the last substitute of N, E3 of R to 10 worlds. Cuts reduce worlds, not
cells.

## v3: control cells added after review

The original is in English (`prereg/v3_controls.md`). It registers cells that separate binding from
refresh and execution from presentation, access and reasoning, before any of them ran on a model.
Before the first run the scope was reduced to worlds 1000-1009 and the cells listed in
`decisions.md` (2026-09-30), and the predictions on cells that were not run were withdrawn there.

**Facts computed offline** (section 1). The detection of the oracle, scope and source-epoch keys is a
function of the world, audited with `python -m memshift.analysis.e1_keys --seeds 1000-1019
--branches src,def,orphan`. A run whose rates differ from the audit is void. Missed invalidation is
0.0% for all three keys. False invalidation (all / definition / rule / free text):
- S_src: source epoch 99.7 / 98.6 / 99.7 / 100.0%; oracle 16.8 / 50.3 / 16.4 / 1.5%; scope
  (definition items) 53.1%.
- S_def: source epoch 99.3 / 95.7 / 99.3 / 100.0%; oracle 4.3 / 61.3 / 2.9 / 1.1%; scope 64.5%.
- Orphan: source epoch 100.0% in every class; oracle 7.0 / 49.4 / 3.7 / 1.1%; scope 52.3%.

**Predictions** (section 3), all on worlds 1000-1019 unless stated, discriminative probes,
checkpoints pooled.
- Oracle key (`B1_gold`). P-G1: Update(B1_gold) − Update(B1_source) ≥ −0.02 in all six (D,
  branch) contrasts. P-G2: |difference| ≤ 0.03 on definition items in each of the six. P-G3:
  Stability(B1_gold) − Stability(B1_source) ≥ 0 in at least five of six.
- Refresh without invalidation (`B0_refresh`). P-R1: Update(B0) − 0.01 ≤ Update(B0_refresh) ≤
  Update(B1_source) + 0.01 in at least five of six. P-R2: on definition items at D=1, after S_src and
  S_def, (Update(B0_refresh) − Update(B0)) / (Update(B1_source) − Update(B0)) ≥ 0.5 (not evaluated if
  the denominator is below 0.05).
- Template presentation (`D0t`). P-T1: |Update(D0t) − Update(D0)| ≤ 0.03 on definition items in six
  contrasts. P-T2: at B=1 after S_src and S_def on definition items, (Update(D0t) − Update(D0)) /
  (Update(D1) − Update(D0)) ≤ 0.15. P-T3: |Update(D0t) − Update(D0)| ≤ 0.03 over all items.
- Retraction-aware programs (`_ra`). P-M1 (orphan, B=1 source epoch): (a) RepairCorrectness(ra) −
  RepairCorrectness(D0B1_source) ≥ 0, and (b) RepairCorrectness(ra) −
  RepairCorrectness(D1sB1_source) has a 95% interval above 0. P-M2: Update(ra) − Update(D1sB1_source)
  ≥ −0.03 after S_src and S_def. P-M3 (validity): the share of `_ra` programs that run at t0 is at
  least 0.8 times that of the D1s programs, otherwise P-M1 and P-M2 are not judged.
- Scope-aware key (`B1_scope`). P-K1: Update(B1_scope) − Update(B1_source) ≥ −0.02 on definition
  items in six contrasts. P-K2: |Update(B1_scope) − Update(B1_cited)| ≤ 0.01 on the other items, and
  the scope cell's overall missed rate is above 0.05.
- Query-time view (`zero_Dm`). P-Z1: Update(zero_Dm) − Update(zero) ≥ 0.30 on definition items
  after S_src and S_def (pooled lower bound above 0.20). P-Z2: on rule items ≥ 0 (S_src, S_def); on
  free-text items ≤ 0.03; on rule items in the orphan branch ≤ 0; Stability difference ≥ −0.03
  (S_src, S_def). P-Z3: |Update(zero_Dm) − Update(D1sB1_source)| ≤ 0.10 on definition items after
  S_src and S_def. At k = 0 the definition-item gain is also predicted to be at least +0.30.
- Thinking writer (`D0think`, worlds 1000-1009, 8,192 output tokens). P-D1:
  RepairCorrectness(D1B1_source) − RepairCorrectness(D0thinkB1_source) ≥ 0.30 on definition items
  after S_src and S_def, also for the mechanical variant. P-D2 (validity): at most 20% of thinking
  writes are unreadable, otherwise P-D1 is not judged.

**Section 6** (added in `b7c0bc0`, 2026-10-01, before the results it concerns).
- Block N: a comparison supports a system's B label only if the native value lies in the predicted
  cell's interval and outside the intervals of the zero system and plain RAG.
- Block L: a pass shows independence from template regularity of message bodies, not from
  paraphrased facts. A fail restricts the headline to template text.
- M1: agreement predicts beyond type difficulty only if AUROC(agreement) − AUROC(new writer's
  per-type calibration accuracy) has a type-level interval above zero.
- Rank correlation (exploratory): ρ ≥ 0.8 with a 95% lower bound ≥ 0.5 means a knowledge-update
  ranking predicts the other rankings; ρ < 0.5 or an interval that includes 0 means it does not,
  anything else is inconclusive.
- Consequences of P-Z1, P-Z3 and P-G1 for the paper's design rules.
- New control D0full: the D=0 writer re-derives definition items from all structured records of
  their declared sources, without the BM25 selection (truncated above 28,000 tokens), worlds
  1000-1009, S_src and S_def. P-F1: RepairCorrectness(D1B1_source) − RepairCorrectness(D0full) ≥
  30 pp on definition items in each branch. A gap of at most 15 pp means access explains most of the
  D effect.

## v4: blocks U, H, G, X, Q, N

The original is in English (`prereg/v4_round3.md`), registered before any of these blocks ran on a
model. It changes no earlier contrast, margin, threshold or verdict. Shared settings:
- Reader Qwen3.5-9B at revision `c202236`, vLLM 0.30.0, temperature 0, 4,096-token budget.
- Discriminative probes at fork + 1, 2, 4, 8 weeks (one week for S_del).
- An anchor cell in every run, whose requests are all cache hits. A run whose anchor answers differ
  from the committed answers is void.
- World-clustered paired bootstrap with 2,000 draws, without a multiplicity correction.

- **U, refresh and upsert.** R_x = (Update(x) − Update(B0)) / (Update(B1_source) − Update(B0)), not
  evaluated if the denominator is below 5 pp. U1: R_refresh ≥ 0.8 on definition items at D=1 and
  D_m after S_src and S_def, in at least 3 of 4. U2: R_refresh at D=0, reported. U3: pooled
  R_upsert ≥ 0.8 means binding's benefit is a fresh derivation an upsert memory also gets. R_upsert
  ≤ 0.5 means the key's schedule adds beyond overwriting. In between, both. U4: after S_del at D=1,
  Update(B0_upsert) − Update(B0) > 0 with a 95% interval above 0. U5: U3's reading is the same in
  at least 2 of 3 M2 families.
- **H, a writer that can count over windows.** H-a: Claude Sonnet 5.5 writes the definition items
  after the shift (cells D0B0, D0B1_source, D1B0, D1B1_source, D1sB0, D1sB1_source; worlds
  1000-1009). H-c: Claude Opus 5.5 with the same design, run if and only if Sonnet's r_0 on changed
  definition items after S_src is below 70%. H-b: the thinking Qwen3.5-9B arm of v3 with 32,768
  output tokens. H1: B×D on definition items after S_def with the Claude writer. At most 10 pp, or an
  interval that includes 0, means the interaction belongs to writers that cannot re-derive windowed
  rules. At least 20 pp with the interval above 0 means execution still pays. In between, both are
  reported with the numbers. H2: Sonnet's r_0 exceeds Qwen3.5-9B's (16.2%) by at least 20 pp. H3: v3's P-D1 with the 32k limit, at least 30 pp
  in each of S_src and S_def.
- **G, held-out definitions and program sets.** G-a: three draws of six definitions from a grammar
  disjoint from the six fixed definitions (sampler seeds 4001, 4002, 4003), worlds 1000-1009. G-b:
  programs from six writers (Qwen3.5-9B, Claude Sonnet 5.5, Claude Opus 5.5, Llama-3.1-8B,
  gemma-4-12B, gpt-oss-20b), three samples each at temperature 0.7, 18 program sets on worlds
  1000-1019. G1: B×D on definition items after S_def positive with the two-way interval (worlds ×
  definitions) above 0, at D=1 and D_m. G2: the same after S_src. G3: B×D at D_m after S_def
  positive with the two-way interval (worlds × program sets) above 0.
- **X, diagnosis and intervention on Mem0.** Short worlds 1000-1009. Detection of an item-time is
  defined in advance, and m, r and a are computed per class and branch. The dominant failure is
  detection if m(1−a) > (1−m)(1−r), judged where the larger loss is at least 5 pp. Arms `mem0_det` (source-keyed invalidation) and `mem0_rep`
  (the item's cached program value leads the evidence). X1: in at least 4 judged cells the arm that
  targets the dominant failure gains more. At least three cells must be judged.
- **Q, unnamed probes.** S_def, worlds 1000-1019. Definition-item probes omit "Definition in
  effect: <statement>". Q1: the B effect on definition items at D=1 is larger by at least 5 pp than
  with named probes. Q2: B×D on definition items stays positive with a 95% interval above 0.
- **N, natural branch.** Cells D0B0, D0B1_source, D1B0, D1B1_source on the natural branch, worlds
  1000-1019, every slot at every checkpoint. On item-times changed by the shift, the paired
  difference is B×D on the shifted branch minus B×D on the natural branch. N1: positive with a 95%
  interval above 0 on definition items after S_src and after S_def.
- **Budget and cuts.** One A100 for about 25-45 GPU-hours. Claude API about USD 95 (H-a), USD 3 (G-b
  programs) and USD 190 (H-c if triggered). Cut order: U5, then G-b beyond Qwen3.5-9B and the two
  Claude writers, then Q, then X. U, H, G-a and N are not cut.

## v4b: blocks Y and V

The original is in English (`prereg/v4b_round3_claude.md`), registered before any request of these
blocks. It records what was known when it was written: the exploratory store/reader decomposition of
the main runs, from which Y's predictions come.

- **Y, a strong reader on the same stores.** Claude Sonnet 5.5 reads the committed stores of six
  cells on worlds 1000-1019, with the reader request unchanged. Validity: at most 2% of answers
  unreadable or cut off. Y1: B×D on the 42 other items at D=1 after S_def positive with a 95%
  interval above 0. Y2: the same after S_del. Y3: B×D on definition items at D=1 after S_src and
  S_def within 5 pp of B×D with Qwen3.5-9B as reader, in both.
- **V, definitions in the user's own words.** The D_m program request states each definition as the
  user would, with the same thresholds and windows (twelve statements fixed in the registration).
  Nine program sets (Qwen3.5-9B, Claude Sonnet 5.5, Claude Opus 5.5, three samples each), paired
  with G-b's. V1: B×D at D_m on definition items after S_def positive with the two-way interval
  (worlds × program sets) above 0. V2: the same after S_src.
- **Budget.** Y about USD 330 at list prices, no GPU. V under USD 10 of programs and about 3-5
  GPU-hours of reader requests.

## Amendments: the decision log

`prereg/decisions.md` has one dated row per decision, with its reason and the clause it touches.
Rows written before a registration shaped it. Rows written after a registration amend it. Most rows
say whether they were written before the runs or analyses they concern, and the summary below keeps
that where a row gives it. The rows are grouped by date. Rows that only rescheduled serving (batch
limits, GPU assignment, pausing and resuming runs) changed no model, revision, request, probe or
reader. They are summarised in one line each.

**2026-09-28 (before and at v0 and v1).**
- Names: MemShift (benchmark), carryover (measure), shift-safe (property), HoldMem (reference
  implementation).
- The leaving test T2 and the implementation shift S_impl stay, in stage 3, carried by Portable
  Memory (`.mem`) with a derivation extension.
- The world generator is built from scratch for six sources, with deterministic templates and no
  LLM. Two existing generators were evaluated and not reused.
- Models: Qwen3.5-9B as the main model, MiMo-V2.6-Distill-Qwen-9B for the same tier (revision
  `2367e865…`, MIT), Qwen3.8-27B for the up tier, gpt-oss-20b and Ministral-3-14B across families,
  Qwen3-Embedding-0.6B for embeddings. All are served by `vllm/vllm-openai:v0.30.0`.
- Before the fork the D=1 checker writes its result to the sidecar without replacing s_k, so the
  cells are byte-identical before the intervention.
- Maintenance runs at the probe checkpoints. E3's 12-month maintenance cost is a cost model
  (simulated invalidations × measured tokens per write).
- Shifts that change no gold (S_wr, S_impl, S_cons) are measured on the same probes before and after,
  with sources frozen.
- The zero system receives the probe text, including the definition in effect.
- E4 calendar semantics: 15-minute polling, sync-token paging, 7-day token expiry, a 410 rate of
  0.001 per incremental poll, lags {1, 5, 30} minutes, the S_src branch. Sampled events per world were
  cut from 30 to 15, then stage 1 used 5 development worlds with 10 events each, and sampling was
  stratified (older E4 runs void).
- The second granularity becomes the writer's cited clusters. The D=1 fallback names the
  definition version. B=0 skips the writer on empty deltas. Stage-1 runs were rerun after each fix
  from cached writes.
- Gate A0: N = 20. The E4 interaction triggered, so sync semantics became a main axis, reduced in
  stage 3 to {ideal, official L = 5} × B × {HoldMem, zero system} on test worlds 1000-1004.
- v1 margins and thresholds as registered. The B×D interaction is added as a registered secondary
  contrast before the test worlds exist, with its development estimates known.
- E2 bootstrap draws corrected from 1,000 to 2,000 (slope lower bound 1.53 → 1.51, same verdict).
- E5 white-box κ is computed per cell, as the protocol says. A first analysis per cell × world gave
  κ_D = 0.75. Per cell gave κ_D = 0.5 and κ_B = 1.0, so D is only descriptive for native systems.
- E7 embedding swap: BAAI/bge-m3 (revision `5617a9f6…`, MIT).
- E4 extension sampling: only changes that touch at least one item, at most 10 calendar changes
  and 2 changes of each other source per world, probes δ ∈ {10, 60} minutes after each change, and
  as many untouched items as touched ones.
- Disclosure: the D=1 fallback computes 42 of 48 items the way gold does (corrected the next day to
  43). An exploratory split of D's repair effect by item class follows.
- Exploratory: E2 retention by item class.
- Rules for the window in which Qwen3.8-27B runs over both GPUs.

**2026-09-29.**
- Correction to 43 items with a rule (`u.timezone` has one) and 5 free-text items. The class splits
  are recomputed.
- Exploratory cell D=1 without fallback (`holdmem_D1B1_source_nofb`), to split D's repair effect
  into checking and executable extraction.
- E5 reduced before running: Mem0 only, 15 short worlds (3 months, seeds 1000-1014), white-box
  cells rerun on them, second ingestion on 1000-1002, ingestion failures counted as system behaviour.
  Graphiti reports cost only. T3 and T4 references for E5 on the same short worlds (oracle, zero system,
  RAG).
- E7 reader swap: first served on vLLM 0.20.0, then rerun on vLLM 0.30.0 with an import shim that
  changes no text computation. The first run is kept as the engine comparison.
- Gate B: Mem0's inferred label is (0,0), and 5 of 6 comparisons fall inside the HoldMem(0,0)
  interval. Its orphan persistence 0.549 [0.490, 0.609] includes the threshold, so the paper claims
  no generalisation. Card B is not adopted, and card C is not the main line.
- Stage 4 frozen in v2. Model roles added to `configs/models.yaml` (Llama-3.1-8B, gemma-4-12B-it and
  Qwen3-8B, with gpt-oss-20b and Ministral-3-14B as stage-4 writers and readers and gemma-4-12B-it as
  the renderer).
- Ingestion contract for stage-4 native systems: every system gets one time-ordered event stream.
  Systems with a documented upsert or delete by external id get one document per source object.
  Others get revisions through their add path and deletions as tombstone text. The LLM is
  Qwen3.5-9B with thinking off. Only the system's retrieval context reaches the shared reader.
- Block N: Letta replaced by Cognee under v2's substitution rule (the registered Letta version is a
  command-line tool, not the MemGPT server). MemOS runs `general_text` because `tree_text` needs an
  external graph database, and its predicted label is unchanged. Cognee registers the served model as
  JSON-schema capable, without which it silently built empty graphs.
- Block L: newsletter e-mails, which carry no gold fact, are not rewritten (budget).
- Block C: the sandbox treats `from datetime import datetime, timedelta` as a no-op. All other
  imports stay disallowed.
- Judgement rules fixed before the stage-4 analyses: M1-b per switch by point estimate, M3 by
  sign, N-1 over the five stage-4 systems only, the predicted cell by the inferred D and B. C1 and C3
  need an interval that excludes 0, C2 a point estimate ≥ 0, C4 the path-2 rule. A native system's
  D label follows its export structure.
- Native runs: longer LightRAG timeouts, documented batch ingestion for HippoRAG 2, LightRAG and
  MemOS, and serving and scheduling changes. Runs disrupted by deployment load are void and rerun
  (`runs/README.md` lists every void run).

**2026-09-30.**
- Exploratory Holm-Bonferroni (α = 0.05) and Benjamini-Hochberg (q = 0.05) corrections of the
  directional v1 contrasts, on stage 2 and on R.
- Exploratory analyses after review: class splits of C1 and C2, T1 per cell, E3 without definition
  items, native comparisons against the other label's cell, calibration curves, replication against
  prediction.
- The control cells of v3 with their offline audit and predictions, then the executed scope: worlds
  1000-1009 and the cells D0B1_gold, D1B1_gold, D1B1_scope, D1B0_refresh, D0tB1_source,
  D1sB1_source_ra, zero_Dm and D0thinkB1_source. Predictions on cells not run were withdrawn:
  P-R1 is judged on the three D=1 contrasts (holds if at least two hold), P-T1 on the three B=1
  contrasts, P-K1 on the three D=1 contrasts.
- Further exploratory analyses of committed outputs (accounting identity, class splits,
  out-of-sample checks, writer-switch predictors, native comparisons with label-free baselines).
- Serving and scheduling changes, with no change to any request.

**2026-10-01.**
- The control runs use their own commit. The registered batch jobs keep the commit they were
  registered with.
- Readings of pending results and the new control D0full, registered before those results (section
  6 of v3).
- D0full implementation before its runs: all live records of the item's declared sources,
  truncation at 25,000 approximate tokens instead of 28,000, per source and counted, with a
  sensitivity analysis without close_friend and vip_sender. P-F1's bounds apply to four gaps (repair
  and mechanical repair × S_src and S_def). All four in one range give "execution", "access" or
  "both", otherwise "mixed".
- Thinking-writer client timeout from 600 to 1,800 seconds (transport only).
- Exploratory analyses after a second internal review (accounting across writers, why cited keys
  miss, the zero system after a shift).
- Void runs from deployment faults, rerun with the same arguments.

**2026-10-08.**
- Exploratory analyses of committed outputs after the second paper review, with no model calls:
  the main-text level table, attribution of changes to the shift or to drift, thinking-writer
  truncation per definition, path 3 under type resampling, the class split of D_m, B×D per
  definition. No registered verdict changes.

**2026-10-16.**
- v4 registered before any model call. One A100 is available, and the Anthropic API is used only for Claude
  Sonnet 5.5 and Opus 5.5 as writers and program writers.
- v4 amendments before any Claude request or X run: the Claude 5.5 models accept neither a thinking
  budget nor a temperature, so they write with adaptive thinking at effort high, 17,000 output tokens
  and default sampling, and validity becomes at most 10% of writes cut off by the output limit and at
  most 2% unreadable. Claude programs: Sonnet without thinking before answering, Opus at effort low,
  samples told apart in the request body. X: Mem0 2.2.1 only adds memories, so m = 1 in every cell
  and the dominant failure is detection. X1 needs at least 3 judged cells and the targeted arm winning
  in at least 4, or in all if fewer than 4.
- G-a's held-out definitions generated and listed before any model call
  (`python -m memshift.gen.kappa --list-heldout 4001 4002 4003`), with the rejection rule that
  removed 7, 3 and 1 draws. Before the grammar was fixed, the implementer had seen the
  specifications and gold statistics of one draw, but no model output.
- v4b registered before any request of Y and V.
- Execution amendment for G-a, G-b, V and U5: `--post-probes definition` answers only the definition
  items of the same discriminative draw after the fork.
- H-b serves a 43,008-token context so that every prompt fits with the registered 32,768 output
  tokens.

**2026-10-17.**
- X runs are void by the anchor rule (3 of 480 t0 answers differ from E5, because Mem0's retrieval
  is not numerically reproducible across serving instances). X1 is not judged. X's cells are
  reported as exploratory.
- A response-cache race (fixed: the first stored reply wins) affected some runs. Six stage-4 M2 runs
  (Llama-3.1-8B and gpt-oss-20b on worlds 1000-1001, 1002-1003, 1006-1007) and the main-study run
  `e1_synth_s1016-1019` were rerun from the cache and replace the committed runs. U5 is judged on the
  reruns. G-a, G-b and V were rerun from the cache. This decision came after V1, V2, G1 and G2 had
  been judged. It rests on the execution defect, the reruns replace the earlier results whatever
  their outcome, and the earlier verdicts are reported as superseded.
