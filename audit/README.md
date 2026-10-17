# Audits

This directory holds two audits that the paper reports as tables. Each file is research data. The
paper's tables are generated from the JSON files, so no table cell is typed by hand.

## System audit

| File | Contents | Read by |
|---|---|---|
| `system_audit.json` | 30 systems placed on the (D, B) grid and on the four diagnostic tests (T1 retention test, T2 leaving test, T3 utility floor, T4 cost ceiling). One record per system: owner, the value of each property, its basis (documented, inferred, measured or by construction), the reason, and the evidence that supports it (id, kind, pointer, access date, note). 21 systems form the paper's table; 9 come from an earlier starting audit (`group` "extra (starting audit)"). | `memshift.analysis.audit_table` (paper Table 5), `memshift.analysis.paper_tables` (`tab_membership`) |
| `evidence.jsonl` | 105 evidence entries, one per line: id, kind (doc, code, paper, record), URL or path with the commit or version read, access date, the judgement it supports, and the systems that use it. | the audit itself (not read by code) |
| `system_audit.md` | A snapshot of the same audit written in Chinese when the audit was made, with the reasoning per system. Its measured cells predate the final runs. `system_audit.json` is the source for every reported cell. | not read by code |

Upstream code and documentation are cited by repository path at a fixed commit, for example
`docs/platform/platform-vs-oss.mdx @94c3fe9…` in Mem0's repository.

Some pointers name files of the authors' working repository that are not released:
`reference/research_1/evidence/sources.json`, `reference/research_2/evidence_registry.jsonl` and
`reference/research_2/system_audit.json` (earlier evidence registries and a documentary audit),
`docs/MEMSHIFT_RESEARCH_PLAN_ZH.md` (the research plan),
`docs/STATUS.md` (the internal status log) and `docs/background/IDEA_PAIN_POINTS_AND_CARDS_ZH.md`
(background notes). In `evidence.jsonl` these are the entries PLAN-S3, STATUS-S1, BG-CARDS,
RA2-AUDIT, E01, E04, E05, E12, E20, E21, E23, E25, E27, E28, E29 and E30. They are internal notes.
The `supports` field of `evidence.jsonl` and the `note` field of each evidence item in
`system_audit.json` state what was taken from them. Pointers to the research plan mark its
expectations as hypotheses, not as evidence.

## Neighbour audit

| File | Contents | Read by |
|---|---|---|
| `neighbours/properties.json` | 16 objects: the 15 nearest benchmarks and the MemShift row (`is_reference`). Each field (data origin, gold timing, scale, the seven change types, derived knowledge, gold dependency, scoring, memory systems, counterfactual control, preregistration, and derived table fields) has a value and verbatim evidence quotes of at most 25 words with their location and PDF page. | `memshift.analysis.neighbour_table` (paper Table 3 and the counts quoted in the related-work section), `tests/test_neighbour_table.py` |
| `neighbours/properties.md` | A snapshot of the same table with the key evidence, written when the audit was made. `properties.json` is the source for every reported cell. | not read by code |
| `neighbours/RUBRIC.md` | The extraction rubric: field definitions, value vocabularies, evidence rules. Its example command writes into `paper/generated/`, the paper's build directory; the top-level README writes to `results/tables/` instead. | not read by code |
| `neighbours/verify.py` | Checks that every quote is a verbatim substring of its source text and reports its PDF page. | run by hand |

`verify.py` needs the text of each neighbour's arXiv PDF (`pdftotext` output) in
`audit/neighbours/src/<name>.txt`. These texts are not redistributed. Without them the script
reports the objects as skipped (`--allow-missing` keeps the exit code at 0).

The MemShift row quotes the run analyses in `runs/`, the registrations in `prereg/` and the text of
an earlier draft of the paper (`paper/sections/*.tex`, `paper/generated/numbers.tex`). The paper
sources are not part of this repository, so those quotes record what the draft said and cannot be
checked here. `verify.py` reports them as problems and exits with code 1 on `properties.json`.

## Regenerating the tables

```bash
uv run python -m memshift.analysis.audit_table > results/tables/audit_table.tex
uv run python -m memshift.analysis.neighbour_table audit/neighbours/properties.json \
    results/tables/tab_neighbours.tex
```

The second command also writes `tab_neighbours_full.tex` and `neighbour_counts.tex` next to its
output.
