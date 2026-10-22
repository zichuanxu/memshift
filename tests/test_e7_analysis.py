import json
import re
from pathlib import Path

import pytest

from memshift.analysis.e7 import conflict_cons, readset


def _a(system, world, e7, status, gold_status, cv=False, conflict=None):
    tags = {"e7": e7}
    if conflict is not None:
        tags["conflict"] = conflict
    return {
        "system": system,
        "world_id": world,
        "status": status,
        "gold_status": gold_status,
        "contract_valid": cv,
        "tags": tags,
    }


def test_conflict_and_cons_rates():
    rows = [
        _a("s", "w0", "conflict", "abstain", "abstain", True, conflict=True),
        _a("s", "w0", "conflict", "answer", "abstain", False, conflict=True),
        _a("s", "w0", "conflict", "answer", "answer", True, conflict=False),
        _a("s", "w0", "cons", "answer", "blocked"),
        _a("s", "w0", "cons", "blocked", "blocked"),
        _a("s", "w0", "cons", "blocked", "answer"),
        _a("s", "w0", "cons", "answer", "answer"),
    ]
    res = conflict_cons(rows)
    assert res["conflict"]["s"]["AbstainAccuracy"]["est"] == 0.5
    assert res["conflict"]["s"]["ControlContractValid"]["est"] == 1.0
    assert res["cons"]["s"]["ScopeLeak"]["est"] == 0.5
    assert res["cons"]["s"]["BlockAccuracy"]["est"] == 0.5
    assert res["cons"]["s"]["OverBlock"]["est"] == 0.5


def test_readset_splits_predicate_items():
    def row(fact, detected):
        return {
            "system": "holdmem_D1B1_cited",
            "branch": "src",
            "world_id": "w0",
            "fact_id": fact,
            "changed": True,
            "detected": detected,
        }

    res = readset([row("k.important_contact", False), row("c.p01.city", True)])
    assert res["src:holdmem_D1B1_cited:predicate"]["est"] == 1.0
    assert res["src:holdmem_D1B1_cited:other"]["est"] == 0.0


def test_e7_numbers_macros_and_missing():
    from memshift.analysis.paper_numbers import MISSING, _macro, e7_numbers

    est = {"est": 0.5, "lo": 0.4, "hi": 0.6, "n_worlds": 5, "n_items": 20}
    m = {
        "conflict": {"zero": {"AbstainAccuracy": est, "ControlContractValid": est}},
        "cons": {"zero": {"ScopeLeak": est}},
        "implicit": {"items": 15, "ceiling": 0.0667},
    }
    out = e7_numbers(m)
    assert out["ESevenAbstainZero"].startswith("50.0")
    assert out["ESevenAbstainDoneBone"] == MISSING
    assert out["ESevenImplicitCeiling"] == "6.7\\%"
    assert all(_macro(k, v) for k, v in out.items())


def test_swap_compares_same_worlds(tmp_path):
    from memshift.analysis.e7 import swap
    from memshift.experiments.e1 import run_world
    from memshift.holdmem.system import CellConfig
    from tests.fakes import FakeLLM

    cells = [CellConfig(d, b) for d in (0, 1) for b in (0, 1)]
    for name, seeds in (("main", (5, 6)), ("swap", (5,))):
        (tmp_path / name).mkdir()
        for seed in seeds:
            run_world(seed, tmp_path / name, branches=("src",), baselines=False,
                      writer=FakeLLM(), cells_cfg=cells)  # fmt: skip
    res = swap([tmp_path / "main"], [tmp_path / "swap"])
    assert res["worlds"] == ["w0005"]
    c = res["src"]["contrasts"]["D_effect_repair_source"]
    assert c["main"]["est"] == c["swap"]["est"]  # same fake writer and reader: identical
    assert c["same_sign"] is (None if c["main"]["est"] == 0 else True)
    assert set(res["src"]["systems"]) == {cfg.name for cfg in cells} | {"holdmem_json_arm"}


def test_same_sign():
    from memshift.analysis.e7 import same_sign

    assert same_sign({"est": 0.1}, {"est": 0.02}) is True
    assert same_sign({"est": 0.1}, {"est": -0.02}) is False
    assert same_sign({"est": -0.1}, {"est": -0.3}) is True
    assert same_sign({"est": 0.1}, {"est": 0.0}) is False
    assert same_sign({"est": 0.0}, {"est": 0.1}) is None  # a zero estimate: no sign to keep
    assert same_sign(None, {"est": 0.1}) is None


def _audit_row(name, group, d, b, d_basis="code", b_basis="code"):
    return {"short_name": name, "group": group, "D": d, "B": b, "D_basis": d_basis,
            "B_basis": b_basis}  # fmt: skip


def test_audit_table_places_each_system_and_excludes_extra_rows():
    from memshift.analysis.audit_table import table

    rows = [
        _audit_row("A_sys", "vendor", "partial", "0", "inferred", "documented + inferred"),
        _audit_row("Extra", "extra (starting audit)", "0", "0"),
        _audit_row("Index", "personal data store", "N/A", "N/A", "documented", "documented"),
        _audit_row("HoldMem (1,1)", "ours", "1", "1"),
    ]
    tex = table(rows)
    assert r"A\_sys & partial & 0 &  & documentation, indirect evidence \\" in tex
    assert "Extra" not in tex
    assert r"Index & --- & --- &  & documentation \\" in tex  # stores no derived item
    # our cells in one row, at the D and B they name; no space before the cell
    assert r"\mtin \holdmem{}(1,1) & \multicolumn{2}{l}{as named} &  & code \\" in tex
    # the native systems that the audit does not cover take their documented label
    assert r"A-Mem \citep{xu2025amem} & 0 & 0 & \checkmark &" in tex
    assert r"HippoRAG~2 \citep{gutierrez2025hipporag2} & 0 & 1 & \checkmark &" in tex
    # the table places systems on the (D,B) grid: no test marks, versions or commits
    assert "?" not in tex.split(r"\label{tab:audit}")[1]
    assert not re.search(r"\d+\.\d+\.\d+|commit|mem0ai|lightrag-hku", tex)


def test_audit_table_groups_rows_and_notes_a_label_that_differs():
    from memshift.analysis.audit_table import GROUPS, table

    rows = [
        _audit_row("Mem0 (self-hosted)", "memory layer", "0", "0"),
        _audit_row("Cognee", "memory layer", "partial", "0"),
        _audit_row("zero system (ours)", "ours", "N/A", "N/A"),
    ]
    tex = table(rows)
    titles = [rf"\mtgroup{{5}}{{{title}}}" for title, _ in GROUPS if title != "Products"]
    positions = [tex.index(t) for t in titles if t in tex]
    assert positions == sorted(positions) and len(positions) == 2
    # Mem0's audit placement equals its documented label; Cognee's does not
    mem0 = next(ln for ln in tex.splitlines() if ln.startswith(r"\mtin Mem0"))
    assert r"& 0 & 0 & \checkmark &" in mem0 and "$^{a}$" not in mem0
    cognee = next(ln for ln in tex.splitlines() if ln.startswith(r"\mtin Cognee"))
    assert r"& partial & 0$^{a}$ & \checkmark &" in cognee
    assert r"\NatCogneeLabel{}" in tex and tex.count("$^{a}$") == 2  # the mark and its note
    assert r"\mtin no-store BM25 & --- & --- &  & code \\" in tex
    # the note follows the key under the table
    notes = tex.split(r"\end{mttabular}")[1]
    assert notes.startswith("\n" + r"\mtnotes{\checkmark: run natively") and "$^{a}$" in notes


def test_audit_table_lists_our_cells_in_one_row_in_the_overview_order():
    from memshift.analysis.audit_table import table

    rows = [
        _audit_row("Plain RAG", "baseline", "N/A", "N/A"),
        _audit_row("zero system (ours)", "ours", "N/A", "N/A"),
        *(_audit_row(f"HoldMem ({d},{b})", "ours", d, b) for d, b in ("11", "01", "10", "00")),
    ]
    lines = table(rows).splitlines()
    group = lines[lines.index(r"\mtgroup{5}{Our baselines and cells}") + 1 :]
    assert group[:3] == [
        r"\mtin plain RAG \citep{lewis2020rag} & --- & --- &  & code \\",
        r"\mtin no-store BM25 & --- & --- &  & code \\",
        r"\mtin \holdmem{}(0,0), (0,1), (1,0), (1,1) & \multicolumn{2}{l}{as named} &  & code \\",
    ]
    assert group[3] == r"\bottomrule"
    with pytest.raises(ValueError, match="is placed at"):
        table([_audit_row("HoldMem (1,0)", "ours", "0", "1")])


def test_audit_caption_keys_every_mark_and_states_no_method():
    from memshift.analysis.audit_table import table

    caption, rest = table([]).split(r"\label{tab:audit}")
    assert "on the $(D,B)$ grid" in caption and caption.count(". ") == 0  # one sentence
    notes = rest.split(r"\end{mttabular}")[1]  # the key under the table
    for term in (
        r"\checkmark: run natively",
        "---: the system stores no derived item",
        "Basis: what the placement rests on",
    ):
        assert term in notes, term
    assert notes.count(". ") <= 3  # a short key
    for gone in ("Membership", "verdict", "pass", "rubric", "version", "Owner"):
        assert gone not in caption + notes, gone


def test_audit_basis_names_the_evidence_once_in_a_fixed_order():
    from memshift.analysis.audit_table import basis

    assert basis(_audit_row("x", "g", "0", "0", "inferred (low confidence)", "documented")) == (
        "documentation, indirect evidence (low confidence)"
    )
    assert basis(_audit_row("x", "g", "0", "0", "code + paper", "paper + documented")) == (
        "code, paper, documentation"
    )


@pytest.mark.paper
def test_committed_audit_table_is_current():
    from memshift.analysis.audit_table import table

    root = Path(__file__).resolve().parents[1]
    rows = json.loads((root / "audit" / "system_audit.json").read_text())
    committed = (root / "paper" / "generated" / "audit_table.tex").read_text()
    assert committed == "% generated by memshift.analysis.audit_table; do not edit\n" + table(rows)
    labels = (r"\label{tab:audit}", r"\label{tab:native-labels}")
    assert all(label in committed for label in labels)
    assert committed.count(r"\checkmark &") == 6  # the six native systems
