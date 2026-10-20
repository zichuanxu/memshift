# Extraction rubric (read fully before starting)

Context: we build a verified property table comparing a new benchmark (MemShift: synthetic personal-data
worlds with byte-identical counterfactual branches; shifts of source revision, of the person's own
definitions with source bytes unchanged, source deletion/orphan, memory-writing model, memory
implementation/export-import; 48 derived knowledge items per world with gold dependency sets; deterministic
scoring, no LLM judge) with its nearest neighbours. You extract properties of ONE neighbour
paper at a time from its full text. Work only from the paper text in
`src/<name>.txt` next to this file (not part of the repository)
(pdftotext of the arXiv PDF; form-feed characters separate PDF pages; `<name>.pdf` is next to it; use `pdftotext -layout -f P -l P`
or the Read tool with `pages` on the PDF if a table or figure is unreadable in the .txt). Do not use memory or other sources for
facts about the paper. Do not touch anything outside the output file you are told to write.

Read the WHOLE paper text, including appendices (dataset construction, evaluation protocol, and baselines details are often in appendices).
Use grep on the .txt to cross-check absence claims. If part of the paper is unreadable (garbled table, missing figure), say so in `readability`.

## Fields (one JSON object per paper; every field has value + evidence)

Each field is `{"value": <string>, "evidence": [{"quote": "<verbatim, <=25 words>", "loc": "<section/table/figure, p.N>"}], "note": "<optional short caveat>"}`.

- Quotes must be VERBATIM substrings of the .txt (hyphenation/whitespace/ligatures are tolerated by the checker; wording is not). Never paraphrase inside `quote`.
  Max 25 words per quote. 1-2 quotes per field; pick the ones that best justify the value. No quote may be reused across unrelated fields unless it truly supports both.
- `loc` = section number/name or table/figure, plus the PDF page number of the .txt (count form feeds; page 1 is the first page).
- Values use these vocabularies where given. `not stated` means the paper does not say (evidence = [] and `note` lists the grep terms you tried).
  `no` means the benchmark's described design contains no such element; for a `no` cell give either a quote that bounds the scope (what the benchmark does cover)
  or, if no quote can prove it, `evidence: []` and in `note` "absence: grepped <terms>, no relevant hit; scope = <what the paper covers>". Never guess.

Fields:
1. `data_origin` — free text, short: human-written / LLM-generated (with or without human editing/verification) / template-rendered / real logs / mixture; name the LLM(s) if stated; state the human-check step if any.
2. `gold_timing` — was the gold (answer/labels) derived BEFORE text generation (script/state first) or AFTER (annotators or LLM write/label from text), or mixed; or `not stated`.
3. `scale` — free text with three numbers where stated: users/histories/trajectories/worlds; questions/queries (total and, if useful, per user); context length (tokens or turns/sessions). Say `not stated` per number if missing.
4. Change types, each with value in {`yes`, `no`, `partial`, `not stated`}; put in `note` a one-line explanation for `partial`:
   - `chg_new_updated_facts` — new or updated facts arrive in the input stream (a fact changes value over time and the benchmark tests the current value / knowledge update).
   - `chg_retraction_deletion` — a source/record/memory is retracted, deleted, deprecated or invalidated explicitly (not just superseded by a newer value).
   - `chg_implicit_conflict` — conflicting information without an explicit "this replaces that" statement, which the system must detect (implicit invalidation, contradiction, staleness that is not announced).
   - `chg_definition_change` — the user redefines a concept/rule/preference/criterion so that derived answers change WITHOUT any new underlying facts (definition or policy change with source records unchanged). Preference updates that arrive as new facts/statements count only as `partial` if the paper says the underlying facts stay the same; otherwise `no`.
   - `chg_writer_model_change` — the model that writes/extracts/consolidates memory is switched (upgraded, downgraded, swapped) and the effect on stored memory is measured (NOT merely comparing different backbones as separate runs; a controlled switch of the writer over the same stored state or history is `yes`; comparing backbones across independent runs is `partial` at most, say so).
   - `chg_implementation_migration` — memory implementation changes: vendor migration, export/import, re-platforming, format conversion of the stored memory.
   - `chg_consent_revocation` — the user withdraws consent / asks to forget / a permission is revoked, and the memory must stop using or must delete the data (unlearning/forget requests count; memory poisoning removal does not).
5. `derived_knowledge` — {`yes`,`no`,`partial`}: are the graded items derived/aggregated knowledge over several records (counts, sets, sums, multi-hop over records, temporal aggregation), versus single stored facts? `partial` when both exist; say in `note` which.
6. `gold_dependency` — {`yes`,`no`,`partial`}: does each answer come with gold evidence / dependency / provenance annotation (which records/turns/sessions it depends on)? `partial` if only for some items or only evidence turns (not full dependency sets); explain.
7. `scoring` — free text: exact/deterministic match, token F1, LLM judge (name the judge model and prompt origin if given), human evaluation, rule-based; list all used and which is primary.
8. `memory_systems` — free text: number and names of memory systems/architectures evaluated (count them from tables; say whether long-context/RAG baselines are counted separately).
9. `counterfactual` — {`yes`,`no`,`partial`}: controlled counterfactual comparison, i.e. the same history/input with vs without the change (matched pair, byte-identical or otherwise controlled), so the change's effect is isolated. `partial` if the contrast is controlled in some other way (e.g. same question with a swapped rendering); explain.
10. `extras` — optional free text (1-3 sentences) of anything important for comparing with MemShift that the fields above miss (e.g. explicit detection-vs-repair split, temporal validity handling). Keep factual, with an evidence quote if possible.

Top-level keys: `key` (bib key given in your task), `src` (the <name> of the .txt), `title`, `arxiv` (id and version read), `venue_status` (as printed in the PDF: e.g. "ICLR 2025", "arXiv preprint"), `readability` ("full" or which parts were unreadable), `fields`.

## Output

Write ONE JSON file per paper to `.../neighbours/parts/<src>.json` (create the dir). Then run
`python3 .../neighbours/verify.py parts/<src>.json` (from the neighbours dir) and FIX every [NOT FOUND]/[TOO LONG]/[NO LOC] until it reports 0 problems.
The checker also adds `pdf_page`; make sure your `loc` page number agrees with it (edit loc if not).

Return to the caller only a compact summary: for each paper, the value of every field in one line each (no quotes), the verifier result, and a list of cells where you are unsure and why.
Do not write any other files. No emojis.

## Derived table fields and PLANNED flags (added for the paper table)

`properties.json` is the single source of the paper's neighbour table (`python -m
memshift.analysis.neighbour_table audit/neighbours/properties.json paper/generated` writes the
table `tab_neighbours_full.tex` and the count macros `neighbour_counts.tex`;
`tests/test_neighbour_table.py` fails if a generated file is out of date). Every object also carries `short_name` (the row label).
Besides the fields above, each object has derived fields of the form `{"value", "derived_from",
"note"}`, condensed by hand from the free-text field named in `derived_from` (the note says how):

- `text_origin`: who writes the world text. T template-rendered, L LLM, L+H LLM with human editing or
  review, H human-written, R real records, M mixture (also of existing datasets), n/s not stated.
- `gold_first`: yes / partial / no, from `gold_timing`. yes = 'before' without qualification;
  partial = 'mixed', 'mostly before', or a scope qualifier; no = 'after'.
- `llm_judge`: primary / minor / none / not stated, from `scoring`. primary = an LLM judge is the
  primary scorer; minor = it scores a minority of items or an add-on; none = no LLM judge for the
  benchmark's own items.
- `native_systems`: number of off-the-shelf memory systems evaluated, counted as the paper groups
  them; long-context and plain RAG baselines, the paper's own systems and oracles are not counted.
- `n_histories`, `n_questions`: worlds, users or trajectories and graded questions, as printed. On the
  MemShift row the question unit is the graded probe (one knowledge item asked at one as-of time on
  one branch): `n_questions` carries `unit` (shown in the caption), `items` and repository evidence
  (`runs/stage2_e1/analysis/e1_metrics.json`), and a test recomputes the count from that file.

Two optional flags are read on the MemShift row only (a neighbour row that carries one is rejected):
`planned_extra` (a string shown as `value (+extra dagger)`, for something not yet run on top of the
observed value) and `planned` (the whole cell is a plan; dagger). `inferred` (double dagger) marks a
value the audit inferred but the paper does not state. Once the result exists, set the observed
value, delete `planned_extra` / `planned`, and rerun the generator; the footnote disappears with the
last mark. The MemShift quotes name a repository file and are checked by `verify.py`; the neighbour
quotes are checked against the paper texts, which are not in the repository (see `verify.py`).
