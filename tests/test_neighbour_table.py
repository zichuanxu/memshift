import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from memshift.analysis import neighbour_table as nt

ROOT = Path(__file__).resolve().parents[1]
PROPS = ROOT / "audit" / "neighbours" / "properties.json"
GENERATED = ROOT / "paper" / "generated"
GOYAL = "arxiv2609_05339"


@pytest.fixture(scope="module")
def raw() -> list[dict[str, Any]]:
    objs: list[dict[str, Any]] = json.loads(PROPS.read_text())
    return objs


def _write(tmp_path: Path, objs: list[dict[str, Any]]) -> Path:
    path = tmp_path / "properties.json"
    path.write_text(json.dumps(objs))
    return path


def _data_rows(tex: str) -> list[str]:
    return [
        ln
        for ln in tex.splitlines()
        if re.search(r"\\cite[pt]\{", ln) or ln.startswith(r"\textbf{\memshift")
    ]


def test_sixteen_rows_and_the_memshift_row_last() -> None:
    rows = nt.load(PROPS)
    assert len(rows) == 16
    assert [r["key"] for r in rows].count("memshift") == 1
    assert rows[-1]["key"] == "memshift" and rows[-1]["is_reference"]
    tex = nt.tab_neighbours_full(rows)
    body = _data_rows(tex)
    assert len(body) == 16 and body[-1].startswith(r"\textbf{\memshift{}} &")
    assert sum(r"\citep{" in ln or r"\citet{" in ln for ln in body) == 15
    # a row named after its authors prints them once, through the citation
    assert [ln.split(" & ")[0] for ln in body if r"\citet{" in ln] == [
        r"\citet{arxiv2609_05339}",
        r"\citet{arxiv2607_16019}",
    ]


def test_every_row_has_one_cell_per_column() -> None:
    tex = nt.tab_neighbours_full(nt.load(PROPS))
    spec = re.search(r"\\begin\{mttabular\}\{@\{\}l ([cr ]+)@\{\}\}", tex)
    assert spec is not None
    n_columns = len(spec.group(1).replace(" ", ""))
    assert spec.group(1).endswith(" rrr")  # the counts of the scale group right-aligned
    for ln in _data_rows(tex):
        assert ln.count(" & ") == n_columns  # name + n_columns cells
        assert ln.endswith(r" \\")


@pytest.mark.paper
def test_neighbour_keys_are_in_the_bibliography_and_cited_in_related_work() -> None:
    bib = (ROOT / "paper" / "refs" / "references.bib").read_text()
    entries = set(re.findall(r"^@\w+\{([^,\s]+),", bib, re.M))
    related = (ROOT / "paper" / "sections" / "related.tex").read_text()
    cited = {k.strip() for m in re.findall(r"\\cite\w*\{([^}]*)\}", related) for k in m.split(",")}
    for r in nt.load(PROPS)[:-1]:
        assert r["key"] in entries, r["key"]
        assert r["key"] in cited, f"{r['key']} is not cited in related.tex"


def test_observed_cells_render_without_a_dagger_and_no_process_note() -> None:
    tex = nt.tab_neighbours_full(nt.load(PROPS))
    ms = _data_rows(tex)[-1]
    # native systems: six observed, none planned
    assert " & 6 & " in ms and r"6\,(+" not in ms
    assert r"T\,(+L)" in ms and nt.DAGGER not in ms  # world text: block L has run
    assert "not yet run" not in tex and "frozen plan" not in tex  # no process note or column
    assert all(nt.DAGGER not in ln for ln in _data_rows(tex)[:-1])
    assert " 40 & " in ms and r"40\,(+" not in ms  # histories: block R has run, no mark


def test_flipping_the_json_removes_the_marks(raw: list[dict[str, Any]], tmp_path: Path) -> None:
    objs = copy.deepcopy(raw)
    ms = next(o for o in objs if o.get("is_reference"))["fields"]
    ms["native_systems"] = {"value": "6", "derived_from": "memory_systems", "note": "run"}
    assert "planned_extra" not in ms["n_histories"]  # block R has run
    assert "planned_extra" not in ms["text_origin"]  # block L has run
    tex = nt.tab_neighbours_full(nt.load(_write(tmp_path, objs)))
    assert nt.DAGGER not in tex and "planned" not in tex
    assert " 6 & " in _data_rows(tex)[-1]


def test_a_planned_categorical_cell_and_an_inferred_cell(
    raw: list[dict[str, Any]], tmp_path: Path
) -> None:
    objs = copy.deepcopy(raw)
    ms = next(o for o in objs if o.get("is_reference"))["fields"]
    ms["chg_writer_model_change"]["planned"] = True
    rows = nt.load(_write(tmp_path, objs))
    writer = nt.cell(rows[-1], nt.COLUMNS["writer"])
    assert writer == r"\checkmark$^\dagger$"
    goyal = next(r for r in rows if r["key"] == GOYAL)
    assert nt.cell(goyal, nt.COLUMNS["gold_first"]) == r"\checkmark$^\ddagger$"
    assert "inferred by the audit" in nt.tab_neighbours_full(rows)


def test_only_the_memshift_row_can_be_planned(raw: list[dict[str, Any]], tmp_path: Path) -> None:
    objs = copy.deepcopy(raw)
    objs[0]["fields"]["chg_definition_change"]["planned"] = True
    with pytest.raises(ValueError, match="only the MemShift row"):
        nt.load(_write(tmp_path, objs))


def test_invalid_input_is_rejected(raw: list[dict[str, Any]], tmp_path: Path) -> None:
    bad = copy.deepcopy(raw)
    bad[0]["fields"]["chg_definition_change"]["value"] = "maybe"
    with pytest.raises(ValueError, match="bad value"):
        nt.load(_write(tmp_path, bad))
    missing = copy.deepcopy(raw)
    del missing[1]["fields"]["native_systems"]
    with pytest.raises(ValueError, match="missing field"):
        nt.load(_write(tmp_path, missing))
    no_ref = [o for o in raw if not o.get("is_reference")]
    with pytest.raises(ValueError, match="exactly one MemShift row"):
        nt.load(_write(tmp_path, no_ref))


@pytest.mark.paper
def test_output_is_deterministic_and_the_generated_files_are_current(tmp_path: Path) -> None:
    first, second = nt.load(PROPS), nt.load(PROPS)
    assert nt.tab_neighbours_full(first) == nt.tab_neighbours_full(second)
    assert nt.macros(first) == nt.macros(second)
    committed = {
        "tab_neighbours_full.tex": nt.tab_neighbours_full(first),
        "neighbour_counts.tex": nt.macros(first),
    }
    for name, text in committed.items():
        assert (GENERATED / name).read_text() == nt.HEAD + text, f"regenerate {name}"


def test_cli_writes_the_table_and_the_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "gen"
    monkeypatch.setattr(sys, "argv", ["neighbour_table", str(PROPS), str(out)])
    nt.main()
    table = (out / "tab_neighbours_full.tex").read_text()
    assert table.startswith(nt.HEAD + r"\begin{table}")
    assert r"\newcommand{\NbrTotal}{15}" in (out / "neighbour_counts.tex").read_text()
    assert sorted(p.name for p in out.iterdir()) == [
        "neighbour_counts.tex",
        "tab_neighbours_full.tex",
    ]


def test_counts_partition_the_neighbours_and_back_the_related_work_text() -> None:
    rows = nt.load(PROPS)
    c = nt.counts(rows)
    assert c["NbrTotal"] == "15"
    for col in nt.COLUMNS.values():
        if col.kind == "text":
            continue
        total = sum(int(c[f"Nbr{col.stem}{nt.VALUE_KEY[v]}"]) for v in nt._vocabulary(col))
        assert total == 15, col.field
    # related.tex says that no neighbour redefines a concept or migrates a memory between
    # implementations and that only Goyal and Ray switch the writer and hold the history fixed;
    # revise the paragraph if the audit changes these
    assert c["NbrRedefYes"] == "0" and c["NbrImplYes"] == "0"
    yes = {
        col: [r["key"] for r in rows[:-1] if r["fields"][col]["value"] == "yes"]
        for col in ("chg_writer_model_change", "counterfactual")
    }
    assert yes == {"chg_writer_model_change": [GOYAL], "counterfactual": [GOYAL]}
    impl = [
        r["key"] for r in rows[:-1] if r["fields"]["chg_implementation_migration"]["value"] != "no"
    ]
    assert impl == [GOYAL]


def _graded_probes(run: str) -> tuple[int, int, int]:
    """Graded probes, knowledge items and worlds of one E1 run: the pre-fork probes plus the asked
    post-fork ones, recomputed from the committed E1 metrics."""
    metrics = json.loads((ROOT / "runs" / run / "analysis" / "e1_metrics.json").read_text())
    pre = metrics["pre_intervention"]
    branches = metrics["cells"]
    for system, entry in pre.items():
        if isinstance(entry, dict) and "n_items" in entry and system != "json_minus_sk":
            assert entry["n_items"] == pre["zero"]["n_items"], system
    post = sum(b["zero"]["ContractValid_post"]["n_items"] for b in branches.values())
    # 48 items per world, asked before the fork on each of the three branches
    items = pre["zero"]["n_items"] // len(branches)
    assert items == 48 * metrics["n_worlds"]
    return pre["zero"]["n_items"] + post, items, int(metrics["n_worlds"])


def test_memshift_questions_are_the_graded_probes_of_the_e1_test_runs() -> None:
    """The unit is the graded probe every system answers. Stage 2 (20 worlds) gives 9,782 and the
    replication worlds of block R (20 further worlds) give 9,780."""
    stage2, stage2_items, stage2_worlds = _graded_probes("stage2_e1")
    rep, rep_items, rep_worlds = _graded_probes("stage4_rep_e1")
    assert (stage2, rep) == (9782, 9780)
    field = nt.load(PROPS)[-1]["fields"]["n_questions"]
    assert stage2 + rep == 19562 == nt._number(field["value"])
    assert field["items"] == stage2_items + rep_items
    assert stage2_worlds + rep_worlds == int(nt.load(PROPS)[-1]["fields"]["n_histories"]["value"])
    assert "graded probes" in field["unit"] and f"{field['items']:,}" in field["unit"]
    unit = f"graded probes over {field['items']:,} knowledge items"
    assert unit in nt.tab_neighbours_full(nt.load(PROPS))
    assert "planned_extra" not in field


def test_the_weaknesses_named_in_the_text_hold_in_the_counts() -> None:
    c = nt.counts(nt.load(PROPS))
    # related.tex: MemShift has fewer test worlds than the median neighbour history count and one
    # six native systems (block N); several neighbours have three or more; probes are no weakness
    assert int(c["MsHistObs"]) < float(c["NbrHistMedian"].replace(",", ""))
    assert int(c["MsNativeObs"]) == 6 and int(c["NbrNativeThreePlus"]) == 9  # block N
    assert int(c["MsQ"].replace(",", "")) > int(c["NbrQMedian"].replace(",", ""))
    assert (c["NbrHistCount"], c["NbrHistMedian"], c["NbrHistMore"]) == ("12", "49", "7")


@pytest.mark.paper
def test_every_count_macro_used_in_the_paper_is_defined() -> None:
    defined = set(nt.counts(nt.load(PROPS)))
    sections = ROOT / "paper" / "sections"
    for path in [sections / "related.tex", *sorted(sections.glob("app*.tex"))]:
        text = path.read_text()
        used = set(re.findall(r"\\((?:Nbr|Ms)[A-Za-z]+)\b", text))
        assert used <= defined, used - defined
    assert r"\input{generated/neighbour_counts}" in (ROOT / "paper" / "main.tex").read_text()


def test_derived_fields_agree_with_the_audit_free_text(raw: list[dict[str, Any]]) -> None:
    for o in raw:
        f = o["fields"]
        timing = f["gold_timing"]["value"].lower()
        expected = {"before": {"yes", "partial"}, "after": {"no"}}
        lead = (
            "before"
            if timing.startswith("before")
            else "after"
            if timing.startswith("after")
            else ""
        )
        if lead:
            assert f["gold_first"]["value"] in expected[lead], o["key"]
        else:  # mixed, mostly before, ...
            assert f["gold_first"]["value"] == "partial", o["key"]
        for name, field in f.items():
            if "derived_from" in field:
                assert field["derived_from"] in f and field["note"], (o["key"], name)
    assert next(o for o in raw if o["src"] == "goyal")["fields"]["gold_first"]["inferred"]


def test_latex_special_characters_are_escaped() -> None:
    tex = nt.tab_neighbours_full(nt.load(PROPS))
    assert r"\citet{arxiv2609_05339} &" in tex and "Goyal" not in tex  # no name twice
    assert r"LoCoMo~\citep{maharana2024locomo}" in tex  # a benchmark keeps its name
    assert nt._tex("a_b & 50%") == r"a\_b \& 50\%"
    assert nt._author_named("Goyal & Ray") and nt._author_named("Jiang et al.")
    assert not nt._author_named("Ground Truth First")


def test_the_notes_key_every_mark_and_code() -> None:
    rows = nt.load(PROPS)
    full = nt.tab_neighbours_full(rows)
    caption, rest = full.split(r"\label{tab:neighbours-full}")
    assert caption.endswith(r"\caption{" + nt.CAPTION + "}\n")  # one sentence; the key below
    notes = rest.split(r"\end{mttabular}")[1]
    assert notes.startswith("\n" + r"\mtnotes{")
    assert r"\cref{" not in full  # no table or section to refer to
    for mark in (r"\checkmark{}~yes", r"$\sim$~partial", r"$\times$~no such element",
                 "n/s~not stated", r"$^\ddagger$~inferred by the audit"):  # fmt: skip
        assert mark in notes, mark
    codes = {str(v) for r in rows for v in nt._text_values(r)}
    for code in codes - {"n/s"}:
        assert f"{code}~{nt.TEXT_CODES[code]}" in notes, code
    assert "(+L)~plus an experiment in which an LLM rewrote the message bodies" in notes
    # a short key: the column definitions are in the text of the appendix
    assert "Derived items:" not in notes and "Consent revoked:" not in notes
    header = next(ln for ln in full.splitlines() if ln.startswith("Benchmark &"))
    assert r"\rotatebox{90}{Deletion}" in header and "retraction" not in full
    # the counts' headers centred over their right-aligned columns
    assert header.endswith(r"\multicolumn{1}{c@{}}{\rotatebox{90}{Questions}} \\")


@pytest.mark.paper
def test_the_appendix_text_defines_the_columns_the_key_leaves_out() -> None:
    text = " ".join((ROOT / "paper" / "sections" / "app_related.tex").read_text().split())
    for term in (
        "has derived items when its graded items aggregate several records",
        "of a world's 48 items only the six definition items and the four recurring series",
        "An implicit conflict is conflicting information without an explicit replacement",
        "Consent is revoked when the user withdraws consent or asks to forget",
        "counterfactual when it runs the same history with and without the change",
        "without long-context and plain RAG baselines",
        "Histories (worlds, users or trajectories) and questions are counted as printed",
    ):
        assert term in text, term


def _header_words(header: str) -> str:
    """A column header as words (a header on two rotated lines as one line)."""
    return (
        header.replace(r"\begin{tabular}{@{}l@{}}", "")
        .replace(r"\end{tabular}", "")
        .replace("\\\\", " ")
    )


@pytest.mark.paper
def test_the_overview_table_names_the_properties_as_the_full_table_does() -> None:
    """Table 2 in the main text (paper/sections/benchmark.tex) counts the neighbours on seven of
    these properties, each under the name of its column here."""
    text = (ROOT / "paper" / "sections" / "benchmark.tex").read_text()
    stats = text[text.index(r"\label{tab:stats}") :]
    for key in ("gold_first", "gold_dep", "redef", "writer", "impl", "branch", "judge"):
        words = _header_words(nt.COLUMNS[key].header)
        name = words[0].lower() + words[1:]
        assert rf"\mtin {name} & \Nbr{nt.COLUMNS[key].stem}" in stats, name


def test_neighbour_rows_have_a_small_gap_every_five_rows() -> None:
    full = nt.tab_neighbours_full(nt.load(PROPS))
    body = full.split(r"\midrule", 1)[1].split(r"\textbf{\memshift{}}")[0].splitlines()
    gaps = [i for i, ln in enumerate(body) if ln == r"\mtgap"]
    rows = [ln for ln in body if r"\cite" in ln]
    assert len(gaps) == (len(rows) - 1) // nt.GAP_EVERY
    for g in gaps:  # each gap follows a full group of rows
        assert sum(r"\cite" in ln for ln in body[:g]) % nt.GAP_EVERY == 0
