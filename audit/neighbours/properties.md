# MemShift and its nearest neighbours: verified property table

Status: audit record for prereg/v2.md section 6 (no run added). This file is a snapshot written when the
audit was made; the paper table is generated from `properties.json` (`python -m
memshift.analysis.neighbour_table`), which is the source of truth for every cell, and the MemShift row
must be updated there once stage 4 has run. Built from the full text of each
neighbour's arXiv PDF (versions in the Status column of Table 1); every quote in `properties.json` was machine-checked as a
verbatim substring of the paper text (alphanumeric-normalised) with its PDF page. MemShift row: from
`paper/sections/benchmark.tex`, `paper/sections/appendix.tex`, `prereg/v1.md`, `prereg/v2.md`. Items of
prereg v2 (stage 4) are marked PLANNED unless they have run: block R (20 replication worlds) is observed
and counted; blocks M, N and L are not.

Legend. yes / part. (partial) / no / n/s (not stated in the paper: no mention found; grep terms are in the JSON
notes). `no` means the described design contains no such element (checked by reading the design sections and
grepping the full text), not that the authors say it is excluded. Sizes are as printed; arithmetic by us is
marked in the JSON.

## Table 1. Data origin, gold timing, scale

| Benchmark / study | Status | Data origin | Gold before or after text | Scale: histories / questions / context |
|---|---|---|---|---|
| LoCoMo | ACL 2024 (arXiv 2402.17753v1) | LLM-generated (two gpt-3.5 agents), annotators edit turns; QA authoring not described | mixed (event graph before text; QA after, author not stated) | 50 conversations / 7,512 Q / ~9.2K tokens (~300 turns) |
| LongMemEval | ICLR 2025 (2410.10813v2) | LLM self-chat sessions + expert-written/rewritten questions; ~70% of simulated sessions human-edited | before (questions/evidence first, sessions written to embed them) | 500 histories / 500 Q / ~115K (S), ~1.5M (M) tokens |
| MemoryAgentBench | ICLR 2026 (2507.05257v4) | mostly existing datasets re-chunked; two new sets by scripted/LLM pipelines; no human check stated | mixed (inherited gold; not discussed) | 130 sequences / 2,071 Q / 103K-1.44M tokens |
| HaluMem | arXiv (2511.03506v3) | LLM-generated personas and dialogues (six-stage pipeline); 8 annotators rated 700 sessions | mostly before (memory points first, refined after) | 20 users / 3,467 Q per set (+14,948 memory points) / ~160K and ~1M tokens per user |
| MemoryArena | ICML 2026 (2602.16313v2) | human-crafted tasks on WebShop, TravelPlanner, BrowseComp-Plus, math/physics; rule templates + human verification | mixed, mostly gold-first | 701 tasks (4,850 subtasks) / users not stated / 14K-122K tokens per task |
| AMA-Bench | ICML 2026 (2602.22769v4) | real agent trajectories + human-written QA (real); TextWorld/BabyAI + programmatic QA (synthetic) | mixed (real: after; synthetic: from state) | 208 real + synthetic trajectories / 2,496 real + 1,200 synthetic QA / ~57.5K tokens avg (max 997K) |
| EvoArena | arXiv (2606.13681v2) | mixed: real Terminal-Bench and commit histories; PersonaMem-Evo LLM-generated | mixed (persona preferences first; questions drafted after; executable tests) | PersonaMem-Evo: 10 persona histories / 505 Q / median 174.7K tokens (+ 89 and 50 task chains) |
| STALE | arXiv (2605.06527v1) | LLM-generated (different LLM per stage); every instance reviewed by a team member | before (conflict pair fixed before dialogue rendering) | 400 instances / 1,200 Q / ~151.8K tokens |
| RECON | arXiv (2607.16716v1) | synthetic: deterministic generator + LLM narrator; 200 questions human-validated | before (gold in code, then narration) | 24 case files / 1,604 Q / 50-100K tokens |
| DynamicMem | arXiv (2606.22877v1) | fully LLM-generated synthetic (one model); small human samples | before (state first; gold validated against logs) | 10 users / 3,634 problems / ~2.2M tokens per user |
| Ground Truth First | arXiv (2607.21962v1) | life-script sampler + LLM renderer; no human check of text or gold | before (script first) | 20 users / 383 Q / 9-week histories (~30-60 records per short user) |
| Goyal & Ray | arXiv (2609.05339v1) | scripted histories with randomized codes; rendering method not stated | before, implied (not stated) | 48 histories / 160 Q each (7,680) / length not stated |
| Jiang et al. | arXiv (2607.16019v1) | real GitHub/Wikipedia streams + LLM-produced gold; 150 items expert-checked | after, inferred (gold attached to real text) | streams not stated / 2,907 QA (+1,198 auxiliary) / length not stated |
| MemSecBench | arXiv (2607.27080v1) | LLM-generated cases (GPT-5.5) with mandatory human approval | before for case specs; no per-item answers | 310 cases (48 contexts) / 930 stage tasks / context not stated |
| GateMem | arXiv (2606.18829v1) | LLM-assisted episodes from human scenario specs; automated checks + manual review | mixed (plan first, checkpoints inserted after) | 91 episodes / 2,218 checkpoints / ~223 turns per episode |
| **MemShift** | this work | template-rendered synthetic (seeded generator, no LLM in world text in observed runs); LLM-rendered worlds PLANNED (v2 block L) | before (script first) | 60 worlds built (10 dev + 10 calib. + 20 test + 20 replication) / 48 items per world (2 versions each) / 1,484-1,600 events, ~2.2M tokens per world |

## Table 2. Change types covered

| Benchmark / study | new or updated facts | retraction or deletion of a source | implicit conflict | user redefines a concept (no new facts) | writer-model switch | implementation / export-import | consent revocation |
|---|---|---|---|---|---|---|---|
| LoCoMo | no | no | no | no | no | no | no |
| LongMemEval | yes | no | part. | no | part. | no | no |
| MemoryAgentBench | yes | no | part. | no | part. | no | no |
| HaluMem | yes | part. | part. | no | no | no | no |
| MemoryArena | part. | no | no | no | no | no | no |
| AMA-Bench | yes | no | part. | no | part. | no | no |
| EvoArena | yes | part. | yes | no | part. | no | part. |
| STALE | yes | no | yes | no | no | no | no |
| RECON | part. | yes | yes | no | no | no | no |
| DynamicMem | yes | no | yes | no | no | no | no |
| Ground Truth First | yes | no | part. | no | no | no | no |
| Goyal & Ray | yes | no | part. | no | yes | part. | no |
| Jiang et al. | yes | part. | yes | no | no | no | no |
| MemSecBench | part. | part. | part. | no | part. | no | no |
| GateMem | yes | yes | no | no | part. | no | yes |
| **MemShift** | yes | yes | part. | yes | yes | part. | yes |

Reading rules used for the cells: writer-model switch = the memory-writing model is switched over stored state
and the effect is measured; comparing backbones in independent runs is at most `part.`. Retraction = a source,
record or memory is explicitly withdrawn (supersession by a newer value alone does not count). Consent
revocation = a user asks to forget or a permission is withdrawn (poisoning removal does not count).
MemShift: implicit conflict is only an E7 stress test on 5 test worlds; consent revocation and conflicts are
E7-scale; the other five are main-design shifts (S_src, orphan, S_def, S_wr, S_impl). The implementation
cell is partial: the importer is a second implementation of the same data model written for this work, not a
third-party system.

## Table 3. Items, gold, scoring, systems, counterfactuals, pre-registration

| Benchmark / study | derived / aggregated items | gold dependency or provenance | scoring | memory systems evaluated | controlled counterfactual (same history with vs without the change) | pre-registration or frozen plan |
|---|---|---|---|---|---|---|
| LoCoMo | part. (multi-hop 14.6%, temporal 20.6%) | part. (evidence turn ids) | token F1 (QA); FactScore/ROUGE; no LLM judge for QA | 0 memory systems; LLMs, long-context, RAG over 3 stores | no | n/s |
| LongMemEval | part. (multi-session, temporal, update items) | part. (evidence sessions/rounds) | LLM judge (gpt-4o-2024-08-06), expert-written prompts; recall@k | 2 commercial assistants (ChatGPT, Coze; 97-Q subset) + own RAG design space | no | n/s |
| MemoryAgentBench | part. (multi-hop in 2 tasks only) | no (none annotated) | SubEM / accuracy; GPT-4o judge for one subset only | 21 rows: 6 long-context, 5 RAG, 10 structure-augmented/agentic (Mem0, Zep, MemGPT, ...) | no | n/s |
| HaluMem | part. (198 of 3,467 multi-hop) | yes (memory-point links) | LLM judge (GPT-4o) | 6: Mem0, Mem0-Graph, Memobase, MemOS, Supermemory, Zep | part. (Medium vs Long: same users, added noise dialogue) | n/s |
| MemoryArena | yes (task execution over earlier subtasks) | part. (dependency structure in task design) | task success rate; checking procedure not stated | 9 memory/RAG (Letta, Mem0, Mirix, Mem0-g, ReasoningBank + 4 RAG) + 4 long-context | no | n/s |
| AMA-Bench | part. (62-72% multi-hop; counts/summaries) | part. (turn-id sets, synthetic subset only) | LLM judge (Qwen3-32B); 92.67% agreement with humans | 15 baselines (4 RAG + 11 memory agents) + own AMA-Agent; 8 long-context | no | n/s |
| EvoArena | part. (1-10 source preferences per question) | part. (PersonaMem-Evo only) | deterministic (tests, multiple-choice accuracy); LLM judge only on an add-on | 4 agents, each +/- own patch; only A-Mem on the memory subset | no | n/s |
| STALE | part. (one commonsense hop in Type II) | yes (old + new session pair) | LLM judge (Gemini-3.1-flash-lite); 95.8% human agreement | 5 frameworks (LightMem, Zep, LiCoMemory, A-mem, mem0) + own CUPMem; 9 long-context | no (same memory, 3 query framings; no history without the change) | n/s |
| RECON | part. (1,368 of 1,604 compositional) | yes (minimal DAG proof trace) | ~85% deterministic rules; 15% free-form by two LLM judges | 4 (Mem0, Mem0-Graph, Supermemory, Hindsight) + 8 long-context, 4 RAG, oracle | no (counterfactual = question type, one narrated file) | n/s |
| DynamicMem | part. (habits aggregated from scattered logs) | part. (first-occurrence logs only) | LLM judge (gpt-5.4) | 5 (RAG, HippoRAG2, MemoryOS, A-Mem, SimpleMem) + oracle | no (nested prefixes; change confounded with history) | n/s |
| Ground Truth First | part. (multi-record types undefined) | part. (template-to-fact keys) | LLM judge (claude-sonnet-5) + cross-family re-judge | 5 own architectures + no-memory control; 2 history baselines | part. (same users at weeks 3/6/9; no with/without pair) | n/s |
| Goyal & Ray | part. (multi-step relations, aliases) | part. (required evidence spans) | exact match on randomized codes; no LLM judge | 4 idealised formats (LC-RAW, RAG, NOTES, KG-fixed); no third-party system | yes (same history; only the writer, embedder or repair source differs: a component swap) | yes (signed Git tag fixed 4 tests before data; paper: 'not a full preregistration') |
| Jiang et al. | part. (supersession chains, single values) | yes (supporting + deprecated event ids) | LLM judge (DeepSeek-V3) primary; exact match on the TempLAMA replication | 3 in-house classes; no off-the-shelf system | part. (render-matched control on identical items; mechanism off, history unchanged) | no (explicitly not pre-registered; some rules fixed before judging) |
| MemSecBench | no (lifecycle checkpoints, not QA) | part. (case-level anchors) | LLM judge (DeepSeek-V4-Pro) + programmatic gates | 4 backends (Native, Mem0, Mem0-Graph, A-MEM) x 2 harnesses x 3 LLMs | part. (matched backend contrasts; no clean-vs-poisoned pair) | n/s |
| GateMem | part. (multi-slot utility items) | part. (supporting spans, leak targets) | LLM judge (GPT-4o) | 3 prior systems (A-Mem, Mem0, ReMem; 4 configs) + 3 own baselines; 6 backbones | no | n/s |
| **MemShift** | part. (all 48 derived from in-scope records; 6 definition predicates, 4 series) | yes (gold dependency objects per item) | deterministic evaluators only; no LLM judge | OBSERVED: 1 native (Mem0) + white-box layer (6 cells) + zero/RAG/oracle. PLANNED: +5 native (A-Mem, Letta, MemOS, LightRAG, HippoRAG 2) | yes (byte-identical branches of one world; equal reader-context hashes asserted) | yes (v0/v1 before first run and test set; v2 frozen before stage 4) |

## Where MemShift differs from the neighbours

Counted per property over the 15 neighbours: a neighbour differs when its cell is not identical to MemShift's
(yes vs part./no/n/s). `strict` counts only verified absences (`no`).

| property | MemShift | neighbours that differ (of 15) | strict `no` | neighbour distribution |
|---|---|---|---|---|
| user redefines a concept (definition change) | yes | 15 | 15 | {'no': 15} |
| memory implementation / export-import change | partial | 14 | 14 | {'no': 14, 'partial': 1} |
| consent revocation | yes | 14 | 13 | {'no': 13, 'partial': 1, 'yes': 1} |
| controlled counterfactual branches | yes | 14 | 10 | {'no': 10, 'partial': 4, 'yes': 1} |
| writer-model switch | yes | 14 | 8 | {'no': 8, 'partial': 6, 'yes': 1} |
| pre-registration or frozen plan | yes | 14 | 1 | {'no': 1, 'not stated': 13, 'yes': 1} |
| retraction / deletion of a source | yes | 13 | 9 | {'no': 9, 'partial': 4, 'yes': 2} |
| gold dependency / provenance | yes | 11 | 1 | {'no': 1, 'partial': 10, 'yes': 4} |
| implicit conflict | partial | 8 | 3 | {'no': 3, 'partial': 7, 'yes': 5} |
| new or updated facts | yes | 4 | 1 | {'no': 1, 'partial': 3, 'yes': 11} |
| derived / aggregated items | partial | 2 | 1 | {'no': 1, 'partial': 13, 'yes': 1} |

Reading the ranking: definition change is the only property on which all 15 neighbours differ from MemShift.
Implementation change, consent revocation, controlled counterfactual branches, writer-model switch and
pre-registration tie at 14 of 15; in each the single matching neighbour is GateMem (consent) or Goyal & Ray
(implementation change through an embedder-version migration, no export/import, and MemShift's own cell is `part.`
too, because its importer is a second implementation of the same data model and not a third-party system;
counterfactual at component level, writer switch, and pre-registration as an internal analysis lock). The
pre-registration count is mostly absence of mention (13 `n/s`); the strict column shows how firm each difference is.

Scoring (not a categorical cell above): MemShift uses no LLM judge at all. An LLM judge is the primary scorer in
9 of 15 neighbours (LongMemEval, HaluMem, AMA-Bench, STALE, DynamicMem, Ground Truth First, Jiang et al.,
MemSecBench, GateMem), scores a minority of items in 2 more (MemoryAgentBench, RECON), and is absent from the
scoring of LoCoMo (QA), Goyal & Ray and EvoArena's own subsets; MemoryArena does not say how it checks answers.

Where MemShift is weaker than several neighbours (from the same sources): native memory systems evaluated
(1 observed, 5 planned, against 4 to 21 rows in MemoryAgentBench, AMA-Bench, MemoryArena, HaluMem, STALE,
RECON, DynamicMem, MemSecBench), text realism (template-rendered; LLM rendering only planned), and number of
questions per benchmark.

## Key evidence (verbatim, checked against the paper text)

First quote per field; PDF page of the arXiv version listed in the status column. Full evidence, locators
and caveats: `properties.json`. Fields without quotes rest on a scope statement or a grep of the full text
(terms in the JSON note).

**LoCoMo** (`maharana2024locomo`)

- counterfactual (no): no quote; note: absence: grepped counterfactual, matched, ablat; no hit. 'with and without' occurs only in the Figure 4 caption (MiniGPT-5 with vs without retrieved context, a method ablation on the same dialogue). scope = one fixed conversation
- gold dependency (partial): "Each QA sample is also annotated with the turn IDs in the conversation logs that contain the answer." (Sec. 4.1, p.5)
- scoring (see Table 3): "employ the F1 partial match metric for evaluating the predictions" (Sec. 4.1, p.5)
- pre-registration (not stated): no quote; note: grepped preregist, pre-regist, frozen (only a BLIP-2 title), hypothes (only FactScore hypothesis text), registered report: none apply. Single inference run per model at temperature 0 (App. C.2).

**LongMemEval** (`wu2025longmemeval`)

- counterfactual (no): no quote; note: absence: grepped counterfactual, matched, ablat; no hit. 'with and without' occurs only in the Figure 13 prompt caption (with vs without Chain-of-Note). scope = each question has its own history; oracle-evidence vs LongMemEval_S v
- gold dependency (partial): "LongMemEval contains human-annotated answer location labels" (Sec. 3.3 (Memory Recall), p.6)
- scoring (see Table 3): "we prompt-engineer the gpt-4o-2024-08-06 model via the OpenAI API" (Sec. 3.3 (Question Answering), p.6)
- pre-registration (not stated): no quote; note: grepped preregist, pre-regist, frozen, hypothes (only explanatory 'we hypothesize' sentences), registered report: none apply.

**MemoryAgentBench** (`hu2025memoryagentbench`)

- counterfactual (no): no quote; note: absence: grepped counterfactual (only the MQuAKE edit-pair source), matched, paired, with vs without; scope = each FactConsolidation context contains both the old and the new fact, with no matched version without the update. 'Coun
- gold dependency (no): "making sure the new context containing the gold passages" (App. B.1.2 (Document QA), p.18)
- scoring (see Table 3): "we use substring exact match (SubEM) to calculate the accuracy of QA" (App. B.1.2, p.18)
- pre-registration (not stated): no quote; note: grepped preregist, pre-regist, frozen, registered report: no hits; 'hypothes' has one hit (a 'core hypothesis' about zero-shot TTL in App. H.2) that is not a pre-registered plan.

**HaluMem** (`chen2025halumem`)

- counterfactual (partial): "HaluMem-Long is built upon HaluMem-Medium to test memory systems under ultra-long context scenarios" (App. A.4, p.18)
- gold dependency (yes): "Each question–answer pair is annotated with a difficulty level and accompanied by traceable evidence, explicitly linking the answer to the supporting memory points." (Sec. 4 (Stage 6), p.8)
- scoring (see Table 3): "we use GPT-4o for consistency determination and scoring" (Sec. 6.1, p.10)
- pre-registration (not stated): no quote; note: grepped preregist, pre-regist, frozen, hypothes, registered report: no hits.

**MemoryArena** (`he2026memoryarena`)

- counterfactual (no): no quote; note: absence: grepped counterfactual, no hit; scope = one pass per task with no with/without-change pair of the same history. Nearest controlled contrast is the oracle-memory injection in App. C.3 / Table 8 (same tasks with gold prior
- gold dependency (partial): "Need to recall the answer from Question 2 and 3." (Fig. 7 (App. A.1, math example), p.19)
- scoring (see Table 3): "a task is successful if the final bundle or plan satisfies all group members" (Sec. 4.2, p.11)
- pre-registration (not stated): no quote; note: grepped preregist, pre-regist, frozen, registered report, hypothes (only 'hypothesis class' in the math examples): no hit.

**AMA-Bench** (`zhao2026amabench`)

- counterfactual (no): no quote; note: absence: grepped counterfactual, no hit; scope = one offline pass per trajectory, no same-history with/without-change pairs. Nearest controlled contrasts: needle-protocol ablation on BabyAI (raw needle turns vs method-constructed
- gold dependency (partial): "The needle here is the minimal set of trajectory turn IDs that contains all the evidence necessary to answer a query." (Sec. 3.3.2, p.5)
- scoring (see Table 3): "Accuracy measures the fraction of predictions judged as correct by an LLM-as-judge based on Qwen3-32B" (Sec. 6.1 (Metrics), p.7)
- pre-registration (not stated): no quote; note: grepped preregist, pre-regist, frozen, registered report, hypothes: no hit.

**EvoArena** (`xu2026evoarena`)

- counterfactual (no): "Each version is a standalone terminal episode with a fixed container/workspace; the environment does not change during that episode." (Sec. 3.1, p.5)
- gold dependency (partial): "we further annotate each question with its source preferences, i.e., the gold observed preference evidence required to answer correctly" (Sec. 3.3, p.9)
- scoring (see Table 3): "Step Accuracy, which measures accuracy on each individual multiple-choice question" (Sec. 3.3, p.9)
- pre-registration (not stated): no quote; note: grepped preregist, pre-regist, frozen, registered report: no hits. 'hypothes' appears only in Sec. 6.1 for a post hoc operationalization hypothesis explicitly called an 'observational mechanism analysis'.

**STALE** (`arxiv2605_06527`)

- counterfactual (no): "separately against the same constructed memory, keeping the memory fixed during probing" (Sec. 4.1, p.7)
- gold dependency (yes): "built around a single implicit conflict triggered by a new observation mn that invalidates a belief supported by an earlier observation mo" (Sec. 3.4, p.5)
- scoring (see Table 3): "We use Gemini-3.1-flash-lite as the LLM judge to assess whether each response demonstrates awareness of the conflict and the updated user state." (Sec. 4.1, p.7)
- pre-registration (not stated): no quote; note: grepped preregist, pre-regist, frozen, hypothes, registered report: no hits. The only 'fixed before' statement concerns the CUPMem state schema ('fixed before CUPMem evaluation', App. F.1), not a pre-registered analysis plan.

**RECON** (`arya2026recon`)

- counterfactual (no): "Systems see only the narrated case file." (Sec 3.1, p.3)
- gold dependency (yes): "Every question carries a proof trace: a minimal spanning subgraph of the DAG whose nodes are necessary and sufficient to derive the answer" (Sec 3.2 Provenance DAG, p.4)
- scoring (see Table 3): "free-form responses are graded by two independent LLM judges from different model families (gpt-4o and gemini-2.5-flash)" (Sec 4.1 Scoring, p.6)
- pre-registration (not stated): no quote; note: absence: grepped preregist, pre-regist, frozen, hypothes, registered report, pre-regis; no hit. The paper mentions a fixed question matrix and snapshot-pinned models but no pre-registered hypotheses or thresholds.

**DynamicMem** (`arxiv2606_22877`)

- counterfactual (no): "At each checkpoint, the system only has access to app logs up to the end of that quarter" (Sec 4.1, p.6)
- gold dependency (partial): "compute evidence recall as the fraction of gold supporting logs cited" (Sec 5.1 Finding 1, p.9)
- scoring (see Table 3): "Both tasks are evaluated using LLM-as-judge method." (Sec 4.3, p.8)
- pre-registration (not stated): no quote; note: absence: grepped preregist, pre-regist, frozen, hypothes, registered report, pre-regis; no hit.

**Ground Truth First** (`arxiv2607_21962`)

- counterfactual (partial): "a measured ranking inversion in a single variable (history length)" (Sec. 2.3, p.5)
- gold dependency (partial): "the question templates and the audit checklist reference the same script fields" (Sec. 5.5, p.12)
- scoring (see Table 3): "The judge (claude-sonnet-5, versioned rubric) is treated as part of the benchmark definition" (Sec. 3.4, p.8)
- pre-registration (not stated): no quote; note: grepped preregist, pre-regist, frozen, hypothes, registered report, freeze: no hit. The paper instead says hybrid v2 was designed after seeing hybrid v1 fail on the same short-horizon corpus (benchmark-informed), and that the tenu

**Goyal & Ray** (`arxiv2609_05339`)

- counterfactual (yes): "We evaluate this handoff by isolating one variable at a time" (Sec. 1, p.2)
- gold dependency (partial): "Qwen’s notes retained 85.6% of the required evidence spans, compared with 64.5% for Llama" (Sec. 3.4, p.6)
- scoring (see Table 3): "exact scoring avoids the need for a model judge" (Sec. 1, p.2)
- pre-registration (yes): "we saved four planned tests and their five-point decision threshold in a signed Git tag" (Sec. 1, p.2)

**Jiang et al.** (`arxiv2607_16019`)

- counterfactual (partial): "the fine-d ledger’s exact structured layout, but with the deprecation mechanism switched off" (Section 4.1 (render-matched control), p.5)
- gold dependency (yes): "Each item stores a gold answer, supporting event ids, deprecated event ids, and a phenomenon label." (Section 3, p.4)
- scoring (see Table 3): "Headline semantic-correctness numbers are judged by DeepSeek-V3" (Section 3, p.5)
- pre-registration (no): "Status: retrodictive but falsifiable, not preregistered." (Appendix A (Proposition 5), p.21)

**MemSecBench** (`chen2026memsecbench`)

- counterfactual (partial): "Matched memorybackend comparisons fix the task package, harness, LLM backend, runtime, clean initialization, external environment, and evidence protocol." (Experiments, Evaluated Systems, p.5)
- gold dependency (partial): "The full design brief records label rationales, a failure-mechanism contract, current safe authority, task-subject bindings, benign-memory bindings, and the target-repair basis." (App. A.2, p.10)
- scoring (see Table 3): "Except for deterministic W1 , a judge-model (DeepSeek-V4-Pro) invocation outside the agent run applies the corresponding rubric." (Methodology, Lifecycle Evaluation Workflow step 5, p.4)
- pre-registration (not stated): no quote; note: grepped preregist, pre-regist, frozen, hypothes, registered report: no pre-registration or frozen analysis plan; 'frozen' occurs only for Hermes' frozen system-prompt snapshot and 'frozen implementations' (pinned software). Rates

**GateMem** (`ren2026gatemem`)

- counterfactual (no): "the agent is reset per episode, processes turns in temporal order, answers checkpoint queries at annotated points, and then resumes the episode." (Section 4.1, p.7)
- gold dependency (partial): "Validation confirms that all utility gold answers are explicitly supported by the preceding interaction history" (Section 3.4, p.6)
- scoring (see Table 3): "Primary results are computed from GPT-4o judge labels." (Section 4.1, p.7)
- pre-registration (not stated): no quote; note: grepped: preregist, pre-regist, registered, frozen, hypothes; no hit.

## Cells that could not be fully verified from the paper text

- Inferred, not stated: Goyal & Ray gold timing (before) rests on the word 'scripted'; Jiang et al. gold timing (after) and origin of gold answers (LLM vs script) are inferred from real sources; the Jiang et al. dataset section is one paragraph.
- Not described: LoCoMo (who wrote the QA and when), MemoryAgentBench (gold timing; human checks of the two new sets), MemoryArena (how answers are scored; number of users), Jiang et al. (number of streams, context length), Goyal & Ray (history length), MemSecBench (context length), GateMem (whether gold precedes or follows text; per-item support sets released), HaluMem/EvoArena/GateMem (generator LLM unnamed), AMA-Bench (whether an LLM is inside synthetic QA generation; whether real QA carry per-item evidence sets), STALE (per-instance release of dependency labels).
- Internal inconsistencies in the papers: EvoArena SWE-Chain-Evo 50 chains/493 steps (main text) vs 48/135 (App. E.4.4) and rendered at 32k/128k vs a 174.7K median; AMA-Bench 208 vs 209 real trajectories (Tables 9 vs 8); LongMemEval text says four long-context models, Figure 3b shows five; MemSecBench says the authoring model differs from the evaluated backends, yet GPT-5.5 is both, and the DeepSeek-V4-Pro judge is also an evaluated backend.
- Judgement calls (marked in the JSON notes): LoCoMo/LongMemEval gold dependency (`part.`: evidence turn or session pointers only; could be read as `yes`); RECON counterfactual (`no`: a question type answered from one file; could be read as `part.`); STALE and EvoArena counterfactual (`no`); DynamicMem retraction (`no`: remove/drop operations exist but nothing is retracted explicitly); HaluMem gold timing; Goyal migration (`part.`: embedder-version migration, no export/import); EvoArena retraction and consent (`part.`: forgetting requests are generated, no metric is reported).
- Pre-registration `n/s` (13 papers) means no mention was found with the terms preregist, pre-regist, frozen, hypothes, registered report; it is absence of mention, not a verified `no`. Goyal & Ray (`yes`, signed tag, internal analysis lock) and Jiang et al. (`no`, explicit) are the only papers that speak to it.
- Parts of the PDFs that were not machine-readable: plots and radar charts (MemSecBench App. F, AMA-Bench Figs. 2/3/6/7/9, MemoryArena Fig. 3) and some flattened result tables (LoCoMo Tables 2-4, LongMemEval Tables 4 and 8-10, RECON Tables 1-3/8, GateMem Table 3 via layout re-extraction); no field depends on them. MemoryArena `loc` pages follow the .txt page count (35 form feeds vs 31 printed pages).
- Versions: the text read is the latest arXiv version at fetch time (Table 1, Status column); venue strings for LoCoMo, MemoryArena and AMA-Bench come from the bib, not from the fetched PDF (LoCoMo PDF prints no venue).

## Discussed, not audited

Eight later neighbours are compared in words in `paper/sections/related.tex` and in the extended related work
(`paper/sections/appendix.tex`) and are not rows of the table, because a row needs a verbatim, machine-checked
quote for every field. The sentence on each rests on its abstract and on the design and evaluation sections of
the arXiv PDF.

| Neighbour | arXiv (version read) | Kind |
|---|---|---|
| StateMemBench, Fan et al. | 2608.19652v1 | benchmark (scripted, LLM-rendered dialogues; LLM judge; eight existing systems) |
| MemTrace, Long et al. | 2606.17328v1 | benchmark (HaluMem-derived knowledge points; LLM judge; 13 configurations) |
| Selective QA over conflicting multi-source personal memory, Yang et al. | 2605.30087v1 | benchmark (scripted latent events; deterministic labels; resolvers, no memory system) |
| PLANFENCE, "Fresh Memory, Stale Plans", Chen et al. | 2609.03340v2 | method with live and replayed multi-agent workflows |
| Dependency-guided rollback repair, Yu et al. | 2608.10502v1 | method with a 150-case injected-fault benchmark |
| Supersede, Patel | 2606.27472v1 | RL environment on LongMemEval knowledge-update questions |
| RD-Forget, Li and Li | 2609.10263v1 | method |
| Reliable post-retrieval assembly, Reddy and Challaram | 2606.01435v2 | method evaluated on FactConsolidation |
