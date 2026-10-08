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


def test_same_sign_verdict():
    from memshift.analysis.e7 import same_sign

    assert same_sign({"est": 0.1}, {"est": 0.02}) is True
    assert same_sign({"est": 0.1}, {"est": -0.02}) is False
    assert same_sign({"est": -0.1}, {"est": -0.3}) is True
    assert same_sign({"est": 0.1}, {"est": 0.0}) is False
    assert same_sign({"est": 0.0}, {"est": 0.1}) is None  # predicted zero: no sign to keep
    assert same_sign(None, {"est": 0.1}) is None


def test_audit_table_marks_and_excludes_extra_rows():
    from memshift.analysis.audit_table import table

    rows = [
        {"short_name": "A_sys", "group": "vendor", "owner": "vendor", "D": "partial", "B": "0",
         "T1": "pass", "T1_basis": "measured", "T2": "fail", "T2_basis": "inferred",
         "T3": "pending E5", "T4": "undeterminable", "T4_basis": "by construction"},
        {"short_name": "Extra", "group": "extra (starting audit)", "owner": "agent", "D": "0",
         "B": "0", "T1": "pass", "T2": "pass", "T3": "pass", "T4": "pass"},
    ]  # fmt: skip
    tex = table(rows)
    assert r"A\_sys & vendor & part. & 0 & \checkmark & $\times$$^\dagger$ & pend.~E5" in tex
    assert "?$^\\ddagger$" in tex and "Extra" not in tex


def test_audit_caption_explains_every_mark_and_how_its_labels_differ_from_block_n():
    from memshift.analysis.audit_table import table

    caption = table([]).split(r"\label{tab:audit}")[0]
    for term in (
        "Diagnostic tests",
        "retention test (T1), leaving test (T2), utility floor (T3), cost ceiling (T4)",
        r"(so ?$^\dagger$ and ?$^\ddagger$ are undeterminable for that reason)",
        "-- = not applicable (the system stores no derived item)",
        r"counts an incomplete key as $B{=}0$",
        "infers its label from measurement",
        "(Cognee: (part., 0) here, (0,1) in block N)",
    ):
        assert term in caption, term
    assert "Membership" not in caption
