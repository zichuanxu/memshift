# Registration of the round-3 experiments (2026-10-16)

Status: registered before any of these blocks ran on a model. It answers two external reviews
(soundness and significance) of the submitted paper. The analyses of committed outputs that the same
round adds (shift-versus-natural composition, stored-value versus reader repair, the deletion failures
of model-written programs) are exploratory and logged in `prereg/decisions.md`. They are not
registered here. Every main-study registration (v0 to v3) is unchanged. Nothing below changes a
contrast, margin, threshold or verdict already reported.

Six blocks, in the order of the budget cut (section 7):

| Block | Question (reviewer) | Worlds | Model calls |
|---|---|---|---|
| U | Is binding's benefit invalidation, or refresh? Matched refresh and slot-keyed upsert (R1 Q3, R2 Q1) | 1000-1019, 12 months; M2 families on 1000-1009 | reader only (writes are cached) |
| H | Does a writer that can count over windows remove the interaction? (R1 Q1, W5) | 1000-1009 | definition-item writer: Claude Sonnet 5.5 (Opus 5.5 if triggered); thinking Qwen3.5-9B at 32k |
| G | Does the interaction hold on held-out definitions and other programs? (R1 Q2, R2 Q2) | 1000-1009 (held-out), 1000-1019 (programs) | writer and program requests for new definitions; program samples |
| X | Does the per-item diagnosis predict which intervention helps a native system? (R2 Q4) | 1000-1009, 3 months | Mem0 ingestion as in E5 |
| Q | Does detection of a redefinition depend on probes naming the version? (R1 W4, Q4) | 1000-1019, S_def only | reader only |
| N | Is the interaction caused by the shift, or would it appear on the same item-times without it? (R2 W3, Q3) | 1000-1019, natural branch | reader only |

Shared settings. Reader: Qwen3.5-9B at revision `c202236`, vLLM 0.30.0 image `sha256:d076d77d`,
temperature 0, 4,096-token evidence budget, one A100 (the main study used two replicas of the same
server; this changes only throughput). Embeddings: Qwen3-Embedding-0.6B at `97b0c61`. Writers: as
stated per block. Probes: discriminative, checkpoints fork + 1, 2, 4, 8 weeks (one week for S_del), as
in E1. Every run includes an anchor cell whose requests are all cache hits, and the pre-intervention
context-hash assertion covers every cell. A run whose anchor answers differ from the committed answers
is void. Intervals: world-clustered paired bootstrap, 2,000 draws. Where definitions or programs are
also resampled (block G), the bootstrap draws worlds and definitions (or program sets) independently
and takes the product sample. Verdicts are per prediction, without a multiplicity correction.

## U. Refresh, upsert and invalidation

Cells (all B=0, built from the same writer requests as existing cells, so no new writes):

- `D{0,1}B0_refresh`, `D1sB0_refresh`: at each checkpoint, re-derive every item whose source-epoch key
  moved, with the request of the B=1 source-epoch cell, and append it. Nothing is invalidated (v3 §3.2).
  `D0B0_refresh` and `D1sB0_refresh` were not run before. `D1B0_refresh` ran on 1000-1009 only.
- `D{0,1}B0_upsert`, `D1sB0_upsert`: the B=0 delta derivation (the requests of `D{0,1}B0`, `D1sB0`).
  When it yields a new item for a slot, every older valid or unverifiable item of that slot is retired.
  A slot is the knowledge item, so this is a slot-keyed upsert, as in add/update memories. No key is kept.

Qwen3.5-9B on 1000-1019, branches src, def, orphan. M2 families (Llama-3.1-8B, gemma-4-12B,
gpt-oss-20b as writer and reader, their cached M2 writes) on 1000-1009 with `D{0,1}B0_refresh` and
`D{0,1}B0_upsert`.

Quantities, on definition items and on the other items, per branch and D arm:
`R_x = (Update(x) - Update(B0)) / (Update(B1_source) - Update(B0))` (ratio of pooled means; not
evaluated if the denominator is below 5 pp), for x in {refresh, upsert}.

- **U1** (definition items, D=1 and D_m, S_src and S_def): `R_refresh >= 0.8`. This extends the v3
  result (`R_refresh` 1.00 at D=1 on 1000-1009) to D_m and to 20 worlds. Holds if true in at least 3 of
  the 4.
- **U2** (definition items, D=0, S_src and S_def): `R_refresh` reported, no threshold. At D=0 the
  denominator can be below 5 pp (the binding effect at D=0 after S_def was +0.2 pp).
- **U3** (upsert, the reading that decides the framing; definition items, D=1 and D_m, S_src and
  S_def, pooled over the four): `R_upsert >= 0.8` means binding's benefit on these items is a fresh
  derivation that an upsert memory also gets, and the paper leads with executability.
  `R_upsert <= 0.5` means the key's schedule adds beyond overwriting. Between the two: both.
- **U4** (S_del, all items, D=1): `Update(B0_upsert) - Update(B0) > 0` with a 95% interval above 0,
  because an upsert replaces the stale item only when the delta yields a new one. A deletion leaves no
  delta for its slot, so the expected gain is small. The reading: if U4 fails, the deletion benefit of
  binding (+51.7 pp against append-only) is not available to an upsert memory.
- **U5** (M2 families): the reading of U3 is the same in at least 2 of the 3 families.

## H. A writer that can count over windows

- **H-a, Claude Sonnet 5.5 as the definition-item writer** (`claude-sonnet-5-5`, Anthropic Messages
  API, extended thinking with a 16,000-token budget, JSON output; the API requires temperature 1 with
  thinking, so these writes are not deterministic and every response is cached and released). The
  design is the v3 thinking arm's: the pre-shift snapshot and every other item are written by the main
  writer (cache hits, so the pre-intervention hashes are those of every cell), and definition items
  written after the shift come from Sonnet, with the same request text. Cells `D0B0`, `D0B1_source`,
  `D1B0`, `D1B1_source` (the D=1 cells check Sonnet's value against the authored derivation),
  `D1sB0`, `D1sB1_source` (programs written by Sonnet without thinking, temperature 0). The reader is
  the main reader. Worlds 1000-1009, branches src, def, orphan. Validity: at most 10% of Sonnet's writes
  reach the thinking budget or the output limit, and at most 2% are unreadable. If either fails, H1 and
  H2 are not judged.
- **H-c, Claude Opus 5.5, conditional** (`claude-opus-5-5`, same design and settings). Runs if and only
  if Sonnet's r_0 on changed definition items after S_src is below 70%, decided from H-a's analysis
  before any Opus request. H1 and H2 are then judged for Opus as well, and the paper reports both.
- **H-b, thinking Qwen3.5-9B at 32k**: the v3 arm `D0thinkB1_source` (anchor `D0B1_source`) re-run with
  `max_tokens` 32,768 (server context 40,960) on 1000-1009. Validity as v3 P-D2 (at most 10% of
  thinking writes cut off). Its prediction is v3's P-D1, unchanged.

Predictions:

- **H1** (the reading that decides the framing): B×D on Update on definition items after S_def with the
  Sonnet writer. Point estimate at most 10 pp, or a 95% interval including 0: the interaction is a
  property of writers that cannot re-derive windowed rules. Rule 2 becomes "make definition items
  executable unless the writer re-derives them", and the abstract says so. At least 20 pp with the
  interval above 0: execution still pays with a frontier writer. Between: both, with the numbers.
- **H2**: Sonnet's repair correctness r_0 on changed definition items after S_src exceeds Qwen3.5-9B's
  (16.2%) by at least 20 pp. If it does not, the stronger writer did not re-derive these rules better,
  and H1's reading is restricted accordingly.
- **H3** (P-D1 of v3, H-b): repair of D=1 minus thinking D=0 on definition items at least 30 pp in each
  of S_src and S_def.

## G. Held-out definitions and programs from other writers

- **G-a, held-out definitions.** A grammar in `gen/kappa.py` samples definitions as (source, predicate
  over records, aggregate, window, threshold). Its predicate set is disjoint from the six fixed
  definitions' (source, predicate) pairs. Version 2 changes the threshold or the window, as the fixed
  definitions do. Three draws of six definitions each (sampler seeds 4001, 4002, 4003, fixed here) replace
  the six fixed definitions in worlds 1000-1009. Events, other items and the other 42 items' requests
  are unchanged. Gold comes from the same generic evaluator that the authored derivation runs. S_def
  moves half of the draw's definitions, as in the main study. The drawn definitions are a function of
  the seeds. They are listed in the decision log when generated, before any model call. Cells as
  in H-a, with Qwen3.5-9B as writer.
- **G-b, program sets.** For each of the twelve fixed definition versions, programs from six writers:
  Qwen3.5-9B, Claude Sonnet 5.5, Claude Opus 5.5, Llama-3.1-8B, gemma-4-12B and gpt-oss-20b. Three
  samples each at temperature 0.7 (sample index in the request, so each has its own cache key; Claude
  models without thinking), eighteen program sets in all. Each set is replayed in `D1sB0` and `D1sB1_source` on
  1000-1019, branches src, def, orphan. Only definition items change, and every other request is a
  cache hit.

Predictions:

- **G1**: B×D on definition items after S_def is positive with the two-way interval (worlds ×
  definitions, 18 definitions) above 0, at D=1 and at D_m.
- **G2**: the same after S_src.
- **G3**: B×D at D_m on definition items after S_def is positive with the two-way interval (worlds ×
  program sets, 18 sets) above 0.
- Reported without a threshold: the share of the variance between program sets that lies between
  writers; the interaction after S_del per writer (design rule 3).
- Reading: if G1 or G3 fails, the paper restricts the definition-item result to the six fixed
  definitions or to the main writer's programs, respectively.

## X. Diagnosis and intervention on Mem0

Mem0 as in E5 (same version, configuration, LLM Qwen3.5-9B, embeddings), short worlds 1000-1009,
branches src, def, orphan. The prefix stores of E5 are reused where their configuration matches.
Mem0's add results are logged per event (ADD, UPDATE, DELETE, NONE, with memory ids).

- **Detection of an item-time** (defined here, before any run): at least one UPDATE or DELETE after
  the fork and at or before the checkpoint, on a memory whose ledger links it to a record in the item's
  gold dependency set at the checkpoint or at the fork. Then m, r and a follow eq. (1) per class.
- **Dominant failure** of a (class, branch) cell: detection if m(1-a) > (1-m)(1-r), else repair.
  Judged only where the larger loss is at least 5 pp.
- **Arms.** `mem0_det`: before a record's revision or deletion is ingested, the memories its earlier
  revision created or updated (from the ledger) are deleted. This is a source-keyed invalidation and
  needs no gold. `mem0_rep`: the evidence for a definition-item probe starts with the value of the
  item's model-written program (the cached `D1s` program), run on the live records, rendered as in
  `zero_Dm`. Other items are served as by Mem0.
- **X1**: in at least 4 of the judged (class, branch) cells, the arm that targets the dominant failure
  raises Update more than the other arm (point estimates). At least three cells must be judged, or X1
  is not judged.
- Reported: m, r and a per class and branch, and each arm's effect with its interval.

## Q. Probes that do not name the definition version

S_def only, 1000-1019. Cells `D0B0`, `D0B1_source`, `D1B0`, `D1B1_source`, `D1sB0`, `D1sB1_source`. The
probes for definition items omit the appended sentence "Definition in effect: <statement>"
(`probes.make_probes`). They keep the question's "under the user's current definition". The items,
keys and writes are unchanged, so only reader requests are new.

- **Q1**: the B effect on Update on definition items at D=1 is larger with unnamed probes than with
  named ones by at least 5 pp. A B=0 memory serves items under both versions, and an unnamed probe
  cannot choose between them.
- **Q2**: B×D on definition items stays positive with a 95% interval above 0.

## N. The natural-branch counterfactual

The generator's natural branch has the same event stream as the shifted branches up to the fork and no
shift after it. No committed run answers it after the fork (E2 probed it one minute before the fork).
Cells `D0B0`, `D0B1_source`, `D1B0`, `D1B1_source` on the natural branch, worlds 1000-1019, every slot
probed at every checkpoint (fork + 1, 2, 4, 8 weeks; not `--discriminative`, which would cover about a
third of the item-times that change in a shifted branch). Writes are cache hits except for the
natural-branch derivations not already in the cache (about 135 of 8,000).

For each shifted branch (src, def, orphan), on the item-times changed by the shift (gold or status differs from t0, and
the gold differs from the natural branch's at the same world, item and time, as in `tab:redef_only`), the paired difference is B×D on the shifted branch minus B×D on the same
item-times on the natural branch, per class. Interval: world-clustered bootstrap, 2,000 draws, seed 996
(`analysis/ex_round3_natural.py --natural-run`).

- **N1**: on definition items after S_src and after S_def, the paired difference is positive with a 95%
  interval above 0. Reading: the interaction on these items is caused by the shift, not by the time
  that has passed.
- Reported without a threshold: the same on the other items and after S_del.

## 7. Budget and the order of cuts

GPU: one A100 for the reader, the Qwen writer and the M2 families, about 25-45 GPU hours. API: Claude
Sonnet 5.5 for H-a, estimated at about US$95, Claude Sonnet and Opus 5.5 for G-b programs, about US$3,
and Claude Opus 5.5 for H-c if triggered, about US$190. If time or budget runs short, blocks are cut in
this order, each cut logged before the affected runs: U5 (M2 families), then G-b beyond Qwen3.5-9B and
the two Claude writers, then Q, then X.
U, H, G-a and N are not cut (N is reader only, about 1 to 2.5 GPU hours).
