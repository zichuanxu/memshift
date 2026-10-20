# Audits

This directory holds two audits that the paper reports as tables. Each file is research data. The
paper's tables are generated from the JSON files, so no table cell is typed by hand.

## System audit

| File | Contents | Read by |
|---|---|---|
| `system_audit.json` | 30 systems placed on the (D, B) grid. One record per system: name, short name, group, owner and a note on the owner, the value of D and of B with its basis (documented, code, paper or inferred, alone or combined) and reason, the components of the system's key (`B_key_components`), and the evidence that supports the placement (id, kind, pointer, access date, note). 21 rows (16 systems and our five cells) form the paper's audit table; 9 come from an earlier starting audit (`group` "extra (starting audit)"). | `memshift.analysis.audit_table` (paper Table 10) |

Upstream code and documentation are cited by repository path at a fixed commit, for example
`docs/platform/platform-vs-oss.mdx @94c3fe9…` in Mem0's repository. Files of this repository are
cited by path and symbol or section name, and run analyses by their path under `runs/`. Evidence
items with a `reference/` or `docs/background/` pointer cite earlier evidence registries and
background notes of the authors, which are not released; the `note` of each such item states what
was taken from it. Some pointers and notes name the four diagnostic tests of the paper by short
names: T1 the retention test, T2 the leaving test, T3 the utility floor and T4 the cost ceiling.
Notes also name experiments by their driver: E1 the main experiment (`memshift.experiments.e1`), E5
the native systems and E6 the export.

## Neighbour audit

| File | Contents | Read by |
|---|---|---|
| `neighbours/properties.json` | 16 objects: the 15 nearest benchmarks and the MemShift row (`is_reference`). Each object has its key, short name, title, arXiv version, venue status, the name of its source text (`src`) and a note on how its text was read, and these fields: data origin, gold timing, scale, the seven change types, derived knowledge, gold dependency, scoring, memory systems, counterfactual control and extras. Each field has a value and verbatim evidence quotes of at most 25 words with their location and PDF page (file and line for the MemShift row). Six table fields (text origin, gold first, LLM judge, native systems, histories, questions) are condensed by hand from the field named in their `derived_from`. | `memshift.analysis.neighbour_table` (paper Table 7 and the counts in Table 2 and the text), `tests/test_neighbour_table.py`, `neighbours/verify.py` |
| `neighbours/RUBRIC.md` | The extraction rubric: field definitions, value vocabularies, evidence rules. Its example command writes into `paper/generated/`, the paper's build directory; the top-level README writes to `results/tables/` instead. | not read by code |
| `neighbours/verify.py` | Checks that every quote is a verbatim substring of its source text and reports its PDF page. | run by hand |

`verify.py` needs the text of each neighbour's arXiv PDF (`pdftotext` output) in
`audit/neighbours/src/<name>.txt`. These texts are not redistributed. Without them the script
reports the objects as skipped (`--allow-missing` keeps the exit code at 0).

The MemShift row quotes files of this repository (code, configs, run analyses and
`results/README.md`), so its quotes are checked without external text:

```bash
uv run python audit/neighbours/verify.py --allow-missing audit/neighbours/properties.json
```

## Regenerating the tables

```bash
mkdir -p results/tables
uv run python -m memshift.analysis.audit_table > results/tables/audit_table.tex
uv run python -m memshift.analysis.neighbour_table audit/neighbours/properties.json \
    results/tables
```

The second command writes `tab_neighbours_full.tex` and `neighbour_counts.tex` into
`results/tables`.
