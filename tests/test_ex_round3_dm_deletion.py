"""Why D_m repairs worse after a deletion (`analysis.ex_round3_dm_deletion`): the classes of an
item-time are checked on hand-made item histories and answers, the origin of a stored value and the
program's own score follow the write path of `holdmem.system`, and the accounting of D_m - D0 adds
up on a table small enough to count by hand."""

import pytest

from memshift.analysis import ex_round3_dm_deletion as dm

AS_OF = "2026-10-08T00:00"


def probe(status: str, value=None, vtype: str = "city") -> dict:
    return {"gold_status": status, "gold_value": value, "vtype": vtype, "as_of": AS_OF}


def answer(valid: bool, status: str = "abstain", value=None) -> dict:
    return {"contract_valid": valid, "status": status, "value": value}


def state(status="valid", value=None, rederived=True, prog_ok=None, prog_value=None) -> dm.State:
    return dm.State(rederived, status, value, prog_ok, prog_value)


# --------------------------------------------------------------------------------------- classes
def test_contract_valid_rows_are_not_failures():
    assert dm.classify(probe("abstain"), state(), None, answer(True), "x", 1) == ("ok", "")


def test_a_row_that_was_not_rederived_is_class_e():
    st = state(rederived=False, value="Oslo")
    assert dm.classify(probe("abstain"), st, None, answer(False, "answer", "Oslo"), "x", 1) == (
        "e",
        "not_rederived",
    )
    assert dm.classify(probe("abstain"), st, None, answer(False, "answer", "Oslo"), "x", 0)[1] == (
        "no_appended_item"
    )


def test_an_unverifiable_item_is_class_c_with_the_cause():
    st = state(status="unverifiable", value="Oslo", prog_ok=False)
    got = dm.classify(probe("answer", "Oslo"), st, state(value="Oslo"), answer(False), "why", 1)
    assert got == ("c", "why")


@pytest.mark.parametrize(
    ("gold", "ans", "sub"),
    [
        (probe("abstain"), answer(False, "answer", None), "answer_without_value"),
        (probe("abstain"), answer(False, "answer", "Oslo"), "gave_value"),
        (probe("answer", "Oslo"), answer(False, "abstain"), "abstained"),
        (probe("answer", "Oslo"), answer(False, "answer", None), "answer_without_value"),
    ],
)
def test_a_right_stored_value_with_a_wrong_reader_is_class_d(gold, ans, sub):
    value = None if gold["gold_status"] == "abstain" else "Oslo"
    assert dm.classify(gold, state(value=value), state(value=value), ans, "x", 1) == ("d", sub)


def test_a_value_for_a_deleted_record_is_class_a_with_its_origin():
    gold, ans = probe("abstain"), answer(False, "answer", "Oslo")
    writer = state(value=None)
    assert dm.classify(gold, state(value="Oslo"), writer, ans, "x", 1) == ("a", "replacement")
    assert dm.classify(gold, state(value="Oslo"), state(value="Oslo"), ans, "x", 1) == (
        "a",
        "writer",
    )
    assert dm.classify(gold, state(value="Oslo"), None, ans, "x", 1) == ("a", "writer")


def test_a_wrong_value_for_an_answer_is_class_b_with_origin_and_kind():
    gold = probe("answer", "Oslo")
    bad = answer(False, "answer", "Rome")
    assert dm.classify(gold, state(value="Rome"), state(value="Oslo"), bad, "x", 1) == (
        "b",
        "replacement_value",
    )
    assert dm.classify(gold, state(value=None), state(value=None), bad, "x", 1) == (
        "b",
        "writer_none",
    )


def test_the_classes_are_tried_in_order():
    """Not re-derived beats unverifiable beats a right stored value."""
    gold = probe("abstain")
    st = state(status="unverifiable", rederived=False)
    assert dm.classify(gold, st, None, answer(False), "x", 1)[0] == "e"
    assert dm.classify(gold, state(status="unverifiable"), None, answer(False), "x", 1)[0] == "c"


# ------------------------------------------------------------------------------ stored state
def event(ts: str, kind: str, new: str, value=None, program=None) -> dict:
    return {
        "ts": ts,
        "event": kind,
        "new": new,
        "value": value,
        **({"program": program} if program else {}),
    }


def test_stored_state_at_b1_reads_the_last_event_and_whether_it_was_a_repair_now():
    item = {
        "det": {"program_key": "k"},
        "history": [
            event("2026-09-30T23:59", "derive", "valid", "Oslo"),
            event(AS_OF, "invalidate", "invalid"),
            event(AS_OF, "repair", "valid", None, {"ok": True, "value": "Rome"}),
            event("2026-10-15T00:00", "invalidate", "invalid"),
        ],
    }
    st = dm.stored_state([item], AS_OF, 1)
    assert (st.rederived, st.status, st.value, st.prog_ok, st.prog_value) == (
        True,
        "valid",
        None,
        True,
        "Rome",
    )
    assert st.prog_key == "k"
    later = dm.stored_state([item], "2026-10-09T00:00", 1)
    assert not later.rederived and later.value is None  # nothing happened that day: carried over


def test_stored_state_at_b0_takes_the_newest_appended_item_beside_the_pre_shift_one():
    pre = {"history": [event("2026-09-30T23:59", "derive", "valid", "Oslo")]}
    first = {"history": [event("2026-10-01T00:00", "append", "valid", "Rome")]}
    second = {"history": [event("2026-10-15T00:00", "append", "unverifiable", "Bern")]}
    st = dm.stored_state([pre, first, second], AS_OF, 0)
    assert (st.rederived, st.status, st.value, st.pre_value) == (True, "valid", "Rome", "Oslo")
    none = dm.stored_state([pre, second], AS_OF, 0)
    assert (none.rederived, none.pre_value) == (False, "Oslo")


# -------------------------------------------------------------------------- program scores
def test_the_program_is_scored_alone_and_its_use_is_named():
    gold = probe("abstain")
    # the writer's value stood although the program, run alone, said none
    st = state(value="Oslo", prog_ok=True, prog_value=None)
    assert dm.program_class(gold, st) == "correct"
    assert dm.program_use(gold, st, state(value="Oslo")) == "overridden"
    # the program put a value where the record is gone
    st = state(value="Rome", prog_ok=True, prog_value="Rome")
    assert dm.program_class(gold, st) == "value_for_deleted"
    assert dm.program_use(gold, st, state(value=None)) == "used"
    # a failure the passed check masked, and one that left the item unverifiable
    masked = state(value="Oslo", prog_ok=False)
    assert dm.program_use(gold, masked, state(value="Oslo")) == "failed_masked"
    lost = state(status="unverifiable", prog_ok=False)
    assert dm.program_use(gold, lost, state(value="Oslo")) == "failed_unverifiable"
    assert dm.program_class(probe("answer", "Oslo"), state(prog_ok=True, prog_value="Rome")) == (
        "wrong_value"
    )
    assert dm.program_class(probe("answer", "Oslo"), state(prog_ok=True, prog_value=None)) == (
        "none_for_answer"
    )
    assert dm.program_class(gold, state(rederived=False, prog_ok=True)) is None
    assert dm.program_class(gold, state()) is None  # a cell without programs


@pytest.mark.parametrize(
    ("reason", "kind"),
    [
        ("null value; program failed: no code in the response", "no_code"),
        ("program failed: syntax error: unmatched '}' (<unknown>, line 57)", "syntax_error"),
        ("x; program failed: disallowed syntax: Import", "rejected_by_sandbox"),
        ("program failed: timeout", "timeout"),
        ("program failed: type mismatch: derive returned int", "wrong_type"),
        ("program failed: AttributeError: 'str' object has no attribute 'get'", "runtime_error"),
        ("spans verified", "unknown"),
        (None, "unknown"),
    ],
)
def test_error_type_names_the_failure_a_write_recorded(reason, kind):
    assert dm.error_type(reason) == kind


# ------------------------------------------------------------------------------ accounting
def rec(klass: str, sub: str = "", changed: bool = True, gold: str = "abstain") -> dm.Rec:
    return dm.Rec(
        changed,
        gold,
        klass,
        sub,
        True,
        "valid",
        None,
        None,
        None,
        None,
        None,
        None,
        False,
        "",
        None,
    )


def test_the_accounting_of_d_m_minus_d0_adds_up_by_item_time_pairs():
    pairs = [  # (D0, D1, D_m) per item-time, two worlds
        ("ok", "ok", "a"),
        ("ok", "ok", "a"),
        ("ok", "d", "d"),
        ("ok", "ok", "ok"),
        ("a", "ok", "ok"),
        ("b", "ok", "ok"),
        ("b", "ok", "b"),
        ("d", "d", "ok"),
    ]
    joint: dm.Joint = {}
    for i, (d0, d1, dmk) in enumerate(pairs):
        sub = "replacement" if dmk == "a" else ""
        joint[f"w{i % 2}", f"f{i}", 1] = {"D0": rec(d0), "D1": rec(d1), "Dm": rec(dmk, sub)}
    keys = sorted(joint)
    out = dm.block(joint, keys, draws=20)
    acc = out["accounting_vs_D0"]
    n = len(pairs)
    assert acc["loss_total"]["est"] == pytest.approx(3 / n)  # D0 right, D_m not: a, a, d
    assert acc["loss_by_Dm_class"]["a"]["est"] == pytest.approx(2 / n)
    assert acc["loss_by_Dm_class"]["d"]["est"] == pytest.approx(1 / n)
    assert acc["gain_total"]["est"] == pytest.approx(3 / n)
    assert acc["net"]["est"] == pytest.approx(acc["gain_total"]["est"] - acc["loss_total"]["est"])
    assert out["Dm_minus_D0"]["ok"]["est"] == pytest.approx(acc["net"]["est"])
    # the class shares of a cell sum to one, and the differences in the failure classes cancel ok
    for cell in dm.CELLS:
        assert sum(out["share"][cell][c]["est"] for c in dm.CLASSES) == pytest.approx(1.0)
    assert sum(out["Dm_minus_D0"][c]["est"] for c in dm.CLASSES) == pytest.approx(0.0)
    shares = acc["loss_share_of_total"]
    assert shares["a"]["est"] == pytest.approx(2 / 3) and shares["a_replacement"]["est"] == (
        pytest.approx(2 / 3)
    )
