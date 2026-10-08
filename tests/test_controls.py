"""The control-cell analysis (`analysis.controls`) on synthetic item tables: every registered
contrast and verdict path (holds, fails, not judged) is checked on tables built so that the answer
is known, the contrasts agree with the published E1 code, and the reported quantities count what
they say. The runs themselves (anchors, probe sets, audit) are tested in test_controls_stage.py."""

import math
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import pytest

from memshift.analysis import controls as ctl
from memshift.analysis import controls_data as cd
from memshift.analysis import e1
from memshift.analysis.e1_synth import _share
from memshift.gen.kappa import KAPPA_DEFS
from memshift.scoring.stats import world_bootstrap

WORLDS = ["w0001", "w0002", "w0003"]
BRANCHES = ctl.BRANCHES
DEFS = [  # 100, changed; the two that D0full truncates are at indices 2 and 3
    "k.active_project",
    "k.overdue_task",
    "k.close_friend",
    "k.vip_sender",
    *(f"k.d{i:02d}" for i in range(96)),
]
RULES = [f"c.p{i:02d}.city" for i in range(100)]  # 100, changed
FREE = ["u.employer", "u.manager", "u.home_city", "u.diet", "u.commute"]  # 5, changed
STABLE = [f"t.t{i:02d}" for i in range(100)]  # 100 rule items whose gold does not change
N_FACTS = len(DEFS) + len(RULES) + len(FREE) + len(STABLE)
CLASS_FACTS = {"definition": DEFS, "rule": RULES, "free": FREE}
SYSTEMS = {short: system for short, system in ctl.CELLS.items()}
ANCHORS = ("D0B1", "D1B1", "D1B1cited", "D1sB1", "D0B0", "D1B0", "zero")

# share of the changed items each cell answers right, by class; the table is built so that every
# registered prediction holds (see `holds`), and each test moves one cell
BASE = {"definition": 0.96, "rule": 0.90, "free": 0.80}
SHARES: dict[str, dict[str, float]] = {
    "D0B0": {"definition": 0.10, "rule": 0.20, "free": 0.20},
    "D1B0": {"definition": 0.50, "rule": 0.20, "free": 0.20},
    "D1B0refresh": {"definition": 0.80, "rule": 0.30, "free": 0.30},  # (0.80-0.50)/(0.96-0.50) > .5
    "D0B1": {"definition": 0.14, "rule": 0.90, "free": 0.80},
    "D0B1gold": {"definition": 0.14, "rule": 0.90, "free": 0.80},
    "D0tB1": {"definition": 0.14, "rule": 0.90, "free": 0.80},
    "D0thinkB1": {"definition": 0.40, "rule": 0.90, "free": 0.80},
    "D0fullB1": {"definition": 0.30, "rule": 0.90, "free": 0.80},  # gap to D1 of 66 pp
    "D1sB1": {"definition": 0.90, "rule": 0.90, "free": 0.80},
    "D1sB1ra": {"definition": 0.96, "rule": 0.90, "free": 0.80},
    "zero": {"definition": 0.06, "rule": 0.60, "free": 0.20},
    "zeroDm": {"definition": 0.86, "rule": 0.70, "free": 0.20},
}
# the view helps rule items, except on the orphan branch (the program of a deleted record still
# answers there: the mechanism of the -7.8 pp)
DEFAULTS = {("zeroDm", "rule", "orphan"): 0.50}
# cells that were run on two branches only (the D0full runs: S_src and S_def)
RAN_ON = {"D0fullB1": ("src", "def")}
# what the D0full writer was given for the definitions that do not fit the cap: the source that was
# cut, the records left out and the rule; the other definitions are given every record
TRUNCATED = {"k.close_friend": ("messages", 12, "source"), "k.vip_sender": ("email", 40, "source")}


def share(cell: str, cls: str, branch: str, overrides: dict[tuple[str, str, str], float]) -> float:
    for key in ((cell, cls, branch), (cell, cls, "*")):
        if key in overrides:
            return overrides[key]
    return SHARES.get(cell, BASE)[cls]


def right(index: int, n: int, frac: float) -> bool:
    """Exactly `round(frac * n)` of the `n` items of a class are answered right."""
    return index < round(frac * n)


class Builder:
    """A table of every control cell and comparison cell, with the answers decided by `overrides`
    (cell, class, branch -> share right; branch `*` for all). B = 1 cells detect every changed
    item, except that the scope and cited cells miss one non-definition item in five."""

    def __init__(self, overrides: dict[tuple[str, str, str], float] | None = None):
        self.overrides = {**DEFAULTS, **(overrides or {})}
        self.drop: set[str] = {"D0B1scope"}  # not run (decisions.md 2026-09-30, WS2 scope)
        self.stab: dict[str, float] = {}
        self.k0: dict[tuple[str, str], float] = {}
        self.prog_ok = {"D1sB1": 1.0, "D1sB1ra": 1.0}
        self.unreadable = 0.0
        self.none_share: dict[str, float] = {}
        self.withdrawn_none: set[str] = set()
        self.truncated = dict(TRUNCATED)

    @staticmethod
    def branches(short: str) -> tuple[str, ...]:
        return RAN_ON.get(short, BRANCHES)

    def rows(self) -> list[e1.Row]:
        rows: list[e1.Row] = []
        for short, system in SYSTEMS.items():
            if short in self.drop:
                continue
            detects = "B1" in short
            for w in WORLDS:
                for b in self.branches(short):
                    for k in (1, 2):
                        for cls, facts in (*CLASS_FACTS.items(), ("stable", STABLE)):
                            for i, fact in enumerate(facts):
                                rows.append(
                                    self._row(short, system, w, b, k, cls, i, fact, detects)
                                )
        return rows

    def _row(
        self,
        short: str,
        system: str,
        w: str,
        b: str,
        k: int,
        cls: str,
        i: int,
        fact: str,
        detects: bool,
    ) -> e1.Row:
        changed = cls != "stable"
        frac = share(short, cls, b, self.overrides) if changed else self.stab.get(short, 1.0)
        n = len(CLASS_FACTS.get(cls, STABLE))
        cv = right(i, n, frac)
        row: e1.Row = {
            "world_id": w,
            "branch": b,
            "system": system,
            "fact_id": fact,
            "vtype": "city",
            "k": k,
            "changed": changed,
            "cv": cv,
            "cv_pre": True,
            "gold_status": "answer",
        }
        if short not in ("zero", "zeroDm"):
            miss = (
                short in ("D1B1scope", "D1B1cited", "D0B1scope")
                and cls in ("rule", "free")
                and i % 5 == 0
            )
            row["detected"] = bool(detects and changed and not miss)
            row["repaired_now"] = row["detected"]
            if row["repaired_now"]:
                row["repair_mech"] = cv
                row["repair_none"] = fact in self.withdrawn_none and short == "D1sB1ra"
        return row

    def table(self) -> cd.Table:
        rows = self.rows()
        k0: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
        for short, system in SYSTEMS.items():
            if short in self.drop:
                continue
            for w in WORLDS:
                for b in self.branches(short):
                    for cls, facts in CLASS_FACTS.items():
                        frac = self.k0.get((short, cls), share(short, cls, b, self.overrides))
                        for i, fact in enumerate(facts):
                            v = float(right(i, len(facts), frac))
                            k0.setdefault(system, {}).setdefault(b, {}).setdefault(w, {})[fact] = v
        return cd.Table(
            rows=rows,
            items=self.items(),
            k0=k0,
            worlds=list(WORLDS),
            origin={},
            months=3,
            checks={},
        )

    def items(self) -> list[e1.Row]:
        items: list[e1.Row] = []
        for w in WORLDS:
            for b in BRANCHES:
                slots = [*DEFS[:5], *RULES[:5]]
                for short in ("D1sB1", "D1sB1ra"):
                    if short in self.drop:
                        continue
                    for i, slot in enumerate(slots):
                        ok = right(i, len(slots), self.prog_ok[short])
                        note = {"ok": ok, "value": None}
                        history: list[dict[str, Any]] = [
                            {"ts": "t0", "event": "derive", "value": None, "program": note}
                        ]
                        if short == "D1sB1ra":
                            history.append(
                                {
                                    "ts": "t1",
                                    "event": "repair",
                                    "value": None,
                                    "program": {**note, "n_reads": 2, "lineage_ok": i != 0},
                                }
                            )
                        items.append(
                            {
                                "world_id": w,
                                "branch": b,
                                "system": SYSTEMS[short],
                                "slot_id": slot,
                                "det": {"program_ok": ok, "reason": "x"},
                                "history": history,
                            }
                        )
                if "D0thinkB1" not in self.drop:
                    for i, slot in enumerate(DEFS[:10]):
                        ok = not right(i, 10, self.unreadable)
                        tokens = 8192 if not ok else 3000 + 100 * i
                        items.append(
                            {
                                "world_id": w,
                                "branch": b,
                                "system": SYSTEMS["D0thinkB1"],
                                "slot_id": slot,
                                "det": None,
                                "history": [
                                    {
                                        "ts": "t1",
                                        "event": "repair",
                                        "think": {"ok": ok, "completion_tokens": tokens},
                                    }
                                ],
                            }
                        )
                if "D0fullB1" not in self.drop and b in self.branches("D0fullB1"):
                    items += self.full_items(w, b)
        return items

    def full_items(self, w: str, b: str) -> list[e1.Row]:
        """The D0full cell's items: one per definition, re-derived at the two checkpoints, each
        repair event carrying the note of `FullAccess.note` (the write at t0 carries none)."""
        out: list[e1.Row] = []
        for key, definition in KAPPA_DEFS.items():
            fact = f"k.{key}"
            source, dropped, rule = self.truncated.get(fact, ("", 0, "none"))
            by_source = {s: {"records": 10, "dropped": 0} for s in definition["sources"]}
            for s in by_source:
                if rule == "newest" or s == source:
                    by_source[s] = {"records": 10, "dropped": dropped}
            note = {
                "records": sum(n["records"] for n in by_source.values()),
                "dropped": sum(n["dropped"] for n in by_source.values()),
                "capped": 3,
                "tokens": 24_900 if rule != "none" else 9_000,
                "truncation": rule,
                "by_source": by_source,
            }
            history: list[dict[str, Any]] = [{"ts": "t0", "event": "derive", "value": None}]
            history += [
                {"ts": f"t{k}", "event": "repair", "value": None, "full": note} for k in (1, 2)
            ]
            out.append(
                {
                    "world_id": w,
                    "branch": b,
                    "system": SYSTEMS["D0fullB1"],
                    "slot_id": fact,
                    "det": None,
                    "history": history,
                }
            )
        return out


def no_audit(seeds, branches, months):
    return {}


FAST = 30  # bootstrap draws for the tests of the verdict logic (point estimates decide)


def ev(table: cd.Table, audit: Callable = no_audit) -> dict:
    return ctl.evaluate(table, audit, draws=FAST)


def run(builder: Builder) -> dict:
    return ev(builder.table())


@pytest.fixture(scope="module")
def holds() -> dict:
    return run(Builder())


def status(res: dict, pred: str) -> str:
    return str(res["verdicts"][pred]["status"])


# ------------------------------------------------------------------------------------ registry
def test_contrast_keys_are_unique_and_macro_safe():
    keys = ctl.contrast_keys()
    assert len(keys) == len(set(keys)) > 200
    assert all(k.replace("_", "").isalnum() for k in keys)
    fams = ctl.families()
    assert {s.tag for s in ctl.specs()} <= set(fams) and "R_ratio" in fams
    assert set(ctl.PREDICTIONS) == set(holds_verdict_names())
    assert ctl.TASK_DEFS == ("k.active_project", "k.overdue_task")


def holds_verdict_names() -> list[str]:
    return list(ctl.Judge({}, {}, {}).all())


def test_the_builder_table_is_well_formed():
    table = Builder().table()
    cells = len(SYSTEMS) - 1  # no D0B1scope, which was not run
    assert (
        len(table.rows) == (3 * (cells - 1) + 2) * len(WORLDS) * 2 * N_FACTS
    )  # D0full: 2 branches
    assert {r["fact_id"] for r in table.rows if r["changed"]} >= set(DEFS)


# ------------------------------------------------------------------------------ verdict paths
def test_every_prediction_holds_when_the_table_satisfies_it(holds):
    assert {p: status(holds, p) for p in ctl.PREDICTIONS} == dict.fromkeys(
        ctl.PREDICTIONS, ctl.HOLDS
    )
    assert holds["summary"] == {"holds": 20, "fails": 0, "not judged": 0}
    v = holds["verdicts"]
    assert (v["G1"]["hits"], v["G1"]["total"], v["G1"]["strong"]) == (6, 6, True)
    assert (v["G3"]["need"], v["R1"]["need"], v["R1"]["total"]) == (5, 2, 3)
    assert v["T1"]["presentation_dominated"] is False
    assert (v["Z2"]["total"], v["Z1"]["total"], v["K2"]["total"]) == (8, 3, 6)
    assert (v["D1"]["total"], v["M1"]["total"]) == (4, 2)
    assert (v["F1"]["total"], v["F1"]["reading"]) == (4, "execution")
    assert holds["verdicts"]["M3"]["share_ra"]["est"] == 1.0


def est(res: dict, key: str) -> float:
    return float(res["contrasts"][key]["est"])


def test_contrast_estimates_are_the_paired_differences_asked_for(holds):
    assert est(holds, "G_upd_D1_src_definition") == pytest.approx(0.0)
    assert est(holds, "Z_upd_src_definition") == pytest.approx(0.86 - 0.06)
    assert est(holds, "D_rep_def_definition") == pytest.approx(0.96 - 0.40)
    assert est(holds, "R_ref_orphan_definition") == pytest.approx(0.80 - 0.50)
    assert est(holds, "R_ratio_src_definition") == pytest.approx((0.80 - 0.50) / (0.96 - 0.50))
    assert est(holds, "T_ratio_src_definition") == pytest.approx(0.0)
    assert est(holds, "K_cited_src_other") == pytest.approx(0.0)
    assert est(holds, "Z0_upd_src_definition") == pytest.approx(0.86 - 0.06)
    e = holds["contrasts"]["Z_upd_src_definition"]
    assert e["n_worlds"] == 3 and e["n_items"] == 3 * 2 * 100  # worlds x checkpoints x items


def with_shares(**cells: float) -> Builder:
    """A builder whose definition-item shares are moved: `cell_branch=share` or `cell=share`."""
    b = Builder()
    for name, value in cells.items():
        cell, _, branch = name.partition("__")
        b.overrides[cell, "definition", branch or "*"] = value
    return b


def test_g1_is_a_non_inferiority_margin_on_the_point_estimate_and_keeps_the_strong_flag():
    # 2 pp below the source cell on the definition items is under 1 pp on all items: inside
    edge = run(with_shares(D0B1gold=0.12, D1B1gold=0.94))
    assert status(edge, "G1") == ctl.HOLDS
    # 12 pp lower on the definition items (a half of the changed items) moves Update by -5.9 pp
    low = run(with_shares(D1B1gold__orphan=0.84))
    assert (
        status(low, "G1") == ctl.FAILS and "G_upd_D1_orphan_all" in low["verdicts"]["G1"]["reason"]
    )
    assert status(low, "G2") == ctl.FAILS  # 12 pp on definition items, beyond 3 pp
    # the strong flag is the pooled lower bound above -4 pp, read when the point estimates hold
    mild = run(with_shares(D1B1gold__orphan=0.94))
    assert status(mild, "G1") == ctl.HOLDS and mild["verdicts"]["G1"]["strong"] is True
    # the registered margin is on the point estimate of all items (1,230 changed item-checkpoints
    # of a branch here): 24 more wrong answers is -1.95 pp and holds, 25 is -2.03 pp and does not
    tab = Builder().table()
    victims = [
        r for r in tab.rows
        if r["system"] == "holdmem_D1B1_gold" and r["branch"] == "src" and r["changed"] and r["cv"]
    ]  # fmt: skip
    for r in victims[:24]:
        r["cv"] = False
    res = ev(tab)
    assert est(res, "G_upd_D1_src_all") == pytest.approx(-24 / 1230)
    assert res["verdicts"]["G1"]["tests"][3]["ok"] is True  # D = 1, S_src
    victims[24]["cv"] = False
    res = ev(tab)
    assert est(res, "G_upd_D1_src_all") == pytest.approx(-25 / 1230)
    assert res["verdicts"]["G1"]["tests"][3]["ok"] is False and status(res, "G1") == ctl.FAILS


def test_g2_is_two_sided_and_inclusive():
    assert status(run(with_shares(D1B1gold__orphan=0.99)), "G2") == ctl.HOLDS  # +3 pp, at the bound
    assert status(run(with_shares(D1B1gold__orphan=1.00)), "G2") == ctl.FAILS  # +4 pp
    at_bound = run(with_shares(D1B1gold__orphan=0.93))
    assert status(at_bound, "G2") == ctl.HOLDS
    assert est(at_bound, "G_upd_D1_orphan_definition") == pytest.approx(-0.03)
    assert status(run(with_shares(D1B1gold__orphan=0.92)), "G2") == ctl.FAILS  # -4 pp


def test_g3_needs_five_of_six_stability_contrasts_at_least_zero():
    one = Builder()
    one.stab = {"D0B1gold": 0.9}  # one (D, branch) pair... all three branches of D = 0 below
    res = run(one)
    assert res["verdicts"]["G3"]["hits"] == 3 and status(res, "G3") == ctl.FAILS  # 3 of 6
    # a single contrast below zero still leaves five
    b = Builder()
    rows = b.table()
    for r in rows.rows:
        if r["system"] == "holdmem_D0B1_gold" and r["branch"] == "src" and not r["changed"]:
            r["cv"] = r["fact_id"] != "t.t00"
    res = ev(rows)
    assert res["verdicts"]["G3"]["hits"] == 5 and status(res, "G3") == ctl.HOLDS


def test_r1_holds_with_two_of_three_and_ordering_is_checked_against_one_percentage_point():
    # refresh answers every changed item right, above the B = 1 cell, in S_src and S_def: the
    # upper bound (refresh <= B1 + 1 pp) breaks twice
    def refresh_above(b: Builder, *branches: str) -> Builder:
        for br in branches:
            for cls in ("definition", "rule", "free"):
                b.overrides["D1B0refresh", cls, br] = 1.0
        return b

    res = run(refresh_above(Builder(), "src", "def"))
    assert res["verdicts"]["R1"]["hits"] == 1 and status(res, "R1") == ctl.FAILS
    res2 = run(refresh_above(Builder(), "src"))
    assert res2["verdicts"]["R1"]["hits"] == 2 and status(res2, "R1") == ctl.HOLDS
    # the lower bound: refresh below B = 0 by more than 1 pp breaks it as well
    worse = Builder({("D1B0refresh", "definition", "orphan"): 0.10})
    assert run(worse)["verdicts"]["R1"]["tests"][2]["ok"] is False


def test_r2_share_of_the_b_effect_and_its_denominator_guard():
    lo = run(with_shares(D1B0refresh__src=0.60, D1B0=0.50))  # (0.60-0.50)/(0.96-0.50) = 0.22
    assert status(lo, "R2") == ctl.FAILS and est(lo, "R_ratio_src_definition") < 0.5
    flat = Builder()
    flat.overrides["D1B1", "definition", "src"] = 0.52  # B effect 2 pp: below the 5 pp guard
    res = run(flat)
    assert status(res, "R2") == ctl.NOT_JUDGED and "below 5 pp" in res["verdicts"]["R2"]["reason"]
    assert res["verdicts"]["R2"]["tests"] == []


def test_t1_and_t3_are_absolute_bounds_and_flag_a_presentation_effect():
    b = with_shares(D0tB1__def=0.30)  # +16 pp on definition items in S_def
    res = run(b)
    assert (
        status(res, "T1") == ctl.FAILS and res["verdicts"]["T1"]["presentation_dominated"] is True
    )
    assert status(res, "T3") == ctl.FAILS  # 16 pp on 50 of 125 changed items is +6.4 pp on all
    assert status(res, "T2") == ctl.FAILS  # share of the D effect 0.16 / 0.82 > 0.15
    small = run(with_shares(D0tB1__def=0.16))  # +2 pp
    assert (
        status(small, "T1") == ctl.HOLDS
        and small["verdicts"]["T1"]["presentation_dominated"] is False
    )
    assert status(small, "T2") == ctl.HOLDS


def test_t2_is_not_judged_when_there_is_no_d_effect_to_share():
    res = run(with_shares(D1B1__src=0.10, D0B1__src=0.14))
    assert status(res, "T2") == ctl.NOT_JUDGED and "not positive" in res["verdicts"]["T2"]["reason"]


def test_p_m3_is_a_validity_condition_for_p_m1_and_p_m2():
    b = Builder()
    b.prog_ok = {"D1sB1": 1.0, "D1sB1ra": 0.7}  # 0.7 of the D1s share, below 0.8
    res = run(b)
    assert status(res, "M3") == ctl.FAILS
    for pred in ("M1", "M2"):
        v = res["verdicts"][pred]
        assert v["status"] == ctl.NOT_JUDGED and "P-M3" in v["reason"] and v["tests"] == []
    assert res["verdicts"]["M3"]["tests"][0]["value"] == pytest.approx(0.7)
    ok = Builder()
    ok.prog_ok = {"D1sB1": 1.0, "D1sB1ra": 0.8}
    assert status(run(ok), "M3") == ctl.HOLDS  # at least 0.8 times, inclusive


def test_p_m1_needs_both_parts():
    # (a) repair of the retraction-aware cell below D = 0 on the orphan branch
    a = run(with_shares(D1sB1ra__orphan=0.05, D0B1__orphan=0.30))
    assert status(a, "M1") == ctl.FAILS
    # (b) the interval must exclude zero: equal repair in every world has a degenerate interval at 0
    b = run(with_shares(D1sB1ra__orphan=0.90, D1sB1=0.90))
    e = b["contrasts"]["M_rep_D1s_orphan_all"]
    assert e["est"] == pytest.approx(0.0) and status(b, "M1") == ctl.FAILS
    assert status(run(with_shares()), "M1") == ctl.HOLDS


def test_p_m2_is_a_three_point_margin_on_all_items():
    res = run(with_shares(D1sB1ra__src=0.80))  # -16 pp on 50 of 125 changed items: -6.4 pp
    assert status(res, "M2") == ctl.FAILS and res["verdicts"]["M2"]["hits"] == 1


def test_p_d2_gates_p_d1_and_counts_unreadable_writes():
    b = Builder()
    b.unreadable = 0.5
    res = run(b)
    assert status(res, "D2") == ctl.FAILS and status(res, "D1") == ctl.NOT_JUDGED
    t = res["reported"]["think"]
    assert t["all"]["writes"] == 3 * 3 * 10
    assert t["all"]["unreadable"]["est"] == pytest.approx(0.5)
    assert t["all"]["at_limit"]["est"] == pytest.approx(0.5)  # the unreadable ones used the budget
    assert t["src"]["completion_tokens_median"] >= 3000 and t["limit_tokens"] == 8192
    edge = Builder()
    edge.unreadable = 0.2
    assert status(run(edge), "D2") == ctl.HOLDS  # at most 20%
    weak = run(with_shares(D0thinkB1__src=0.80))
    assert status(weak, "D1") == ctl.FAILS  # 0.96 - 0.80 < 0.30 in S_src


def test_p_d1_judges_the_mechanical_variant_too():
    res = run(Builder())
    keys = [t["key"] for t in res["verdicts"]["D1"]["tests"]]
    assert keys == [
        "D_rep_src_definition",
        "D_rep_def_definition",
        "D_repmech_src_definition",
        "D_repmech_def_definition",
    ]
    b = Builder()
    tab = b.table()
    for r in tab.rows:
        if r["system"] == "holdmem_D0thinkB1_source":
            r["repair_mech"] = r["cv"] or r["fact_id"].startswith("k.")  # mechanically all right
    res = ev(tab)
    assert res["verdicts"]["D1"]["hits"] == 2  # behavioural still passes, mechanical does not


def test_p_k1_is_judged_at_d1_only_and_p_k2_needs_both_halves():
    res = run(with_shares(D1B1scope__def=0.90))
    assert status(res, "K1") == ctl.FAILS and res["verdicts"]["K1"]["hits"] == 2
    assert [t["key"] for t in res["verdicts"]["K1"]["tests"]] == [
        f"K_upd_{b}_definition" for b in BRANCHES
    ]
    # the scope cell's overall missed rate must exceed 5%: here it misses nothing
    b = Builder()
    tab = b.table()
    for r in tab.rows:
        if r["system"] == "holdmem_D1B1_scope" and r["changed"]:
            r["detected"] = True
    res = ev(tab)
    assert status(res, "K2") == ctl.FAILS
    assert all(
        not t["ok"] for t in res["verdicts"]["K2"]["tests"] if t["key"].startswith("scope_missed")
    )
    # the scope cell must be the cited cell on the other items: 3 pp apart breaks the 1 pp bound
    c = Builder({("D1B1scope", "rule", "src"): 0.87})
    res = ev(c.table())
    assert status(res, "K2") == ctl.FAILS
    assert est(res, "K_cited_src_other") == pytest.approx(-3 / 105)  # 3 of 100 rules, over 105


def test_p_z1_needs_point_estimates_and_a_pooled_lower_bound():
    res = run(with_shares(zeroDm__def=0.30))  # 24 pp in S_def, below 30 pp
    assert status(res, "Z1") == ctl.FAILS and status(res, "Z1k") == ctl.FAILS
    ok = run(Builder())
    pooled = ok["contrasts"]["Z_pool_definition"]
    assert pooled["lo"] > 0.20 and pooled["n_items"] == 2 * 3 * 2 * 100
    # 30 pp in both branches is at the bound of the point estimates; the pooled bound is 30 pp too
    edge = run(with_shares(zeroDm__src=0.36, zeroDm__def=0.36))
    assert status(edge, "Z1") == ctl.HOLDS and est(edge, "Z_upd_src_definition") == pytest.approx(
        0.30
    )


def test_p_z2_is_a_conjunction_of_clauses():
    res = run(Builder())
    assert [t["key"] for t in res["verdicts"]["Z2"]["tests"]] == [
        "Z_upd_src_rule", "Z_upd_def_rule",
        "Z_upd_src_free", "Z_upd_def_free", "Z_upd_orphan_free",
        "Z_upd_orphan_rule",
        "Z_stab_src_all", "Z_stab_def_all",
    ]  # fmt: skip
    # the view helps rule items on the orphan branch (the program of a deleted record still answers)
    bad = Builder({("zeroDm", "rule", "orphan"): 0.9})
    assert status(ev(bad.table()), "Z2") == ctl.FAILS
    # free text must not gain more than 3 pp
    free = Builder({("zeroDm", "free", "def"): 0.8})
    assert status(ev(free.table()), "Z2") == ctl.FAILS
    # stability may drop by 3 pp, not 4
    a = Builder()
    a.stab = {"zeroDm": 0.97}
    assert status(ev(a.table()), "Z2") == ctl.HOLDS
    c = Builder()
    c.stab = {"zeroDm": 0.90}
    assert status(ev(c.table()), "Z2") == ctl.FAILS


def test_p_z3_is_two_sided_against_the_stored_derivation():
    assert status(run(with_shares(zeroDm__src=0.70)), "Z3") == ctl.FAILS  # 20 pp below D1s
    assert status(run(with_shares(zeroDm__src=0.99, D1sB1=0.90)), "Z3") == ctl.HOLDS  # 9 pp above
    assert status(run(with_shares(zeroDm__src=1.0, D1sB1=0.88)), "Z3") == ctl.FAILS  # 12 pp above


def test_a_cell_that_was_not_run_leaves_its_predictions_not_judged_and_the_rest_intact():
    b = Builder()
    b.drop |= {"zeroDm"}
    res = run(b)
    for pred in ("Z1", "Z1k", "Z2", "Z3"):
        v = res["verdicts"][pred]
        assert v["status"] == ctl.NOT_JUDGED and v["reason"].startswith("missing:")
    for pred in (
        "G1",
        "G2",
        "G3",
        "R1",
        "R2",
        "T1",
        "T2",
        "T3",
        "M1",
        "M2",
        "M3",
        "K1",
        "K2",
        "D1",
        "D2",
        "F1",
    ):
        assert status(res, pred) == ctl.HOLDS, pred
    assert res["summary"] == {"holds": 16, "fails": 0, "not judged": 4}
    assert math.isnan(res["contrasts"]["Z_upd_src_definition"]["est"])
    assert res["contrasts"]["Z_upd_src_definition"]["n_worlds"] == 0


def test_a_missing_anchor_makes_every_contrast_against_it_not_judged():
    b = Builder()
    b.drop |= {"D1B0"}
    res = run(b)
    assert status(res, "R1") == ctl.NOT_JUDGED and status(res, "R2") == ctl.NOT_JUDGED
    assert status(res, "G1") == ctl.HOLDS


def test_a_verdict_carries_its_tests_and_reason():
    res = run(with_shares(D1B1gold__orphan=0.84))
    v = res["verdicts"]["G2"]
    assert v["status"] == ctl.FAILS and v["total"] == 6 and v["hits"] == 5
    assert "not met: G_upd_D1_orphan_definition -12.0 pp" in v["reason"]
    bad = [t for t in v["tests"] if not t["ok"]]
    assert [t["key"] for t in bad] == ["G_upd_D1_orphan_definition"] and bad[0][
        "threshold"
    ] == "|.| <= 3 pp"


# --------------------------------------------------------------------- P-F1 (the D0full control)
def set_full_answers(tab: cd.Table, right_facts: set[str]) -> None:
    """The D0full cell answers the definition items right exactly on `right_facts` (behaviourally
    and in the stored value)."""
    for r in tab.rows:
        if r["system"] == SYSTEMS["D0fullB1"] and r["fact_id"].startswith("k."):
            r["cv"] = r["fact_id"] in right_facts
            if "repair_mech" in r:
                r["repair_mech"] = r["cv"]


def f1_reading(**cells: float) -> tuple[str, str | None]:
    """The P-F1 verdict and reading when the D0full definition share is moved (`D0fullB1=share`,
    or `D0fullB1__src=share` for one branch); D1 answers 96 of 100 definition items right."""
    res = run(with_shares(**cells))
    return status(res, "F1"), res["verdicts"]["F1"]["reading"]


def test_p_f1_needs_thirty_points_in_each_branch_for_repair_and_for_mechanical_repair(holds):
    v = holds["verdicts"]["F1"]
    assert [t["key"] for t in v["tests"]] == [
        "F_rep_src_definition",
        "F_rep_def_definition",
        "F_repmech_src_definition",
        "F_repmech_def_definition",
    ]  # the orphan branch was not run, and the sensitivity class is not judged
    assert (v["status"], v["hits"], v["total"], v["need"]) == (ctl.HOLDS, 4, 4, 4)
    assert {t["threshold"] for t in v["tests"]} == {">= 30 pp"}
    assert est(holds, "F_rep_src_definition") == pytest.approx(0.96 - 0.30)
    assert est(holds, "F_repmech_def_definition") == pytest.approx(0.96 - 0.30)
    # 66 of 100 right is a gap of exactly 30 pp, inclusive; 67 right is 29 pp in S_src only
    edge = run(with_shares(D0fullB1=0.66))
    assert status(edge, "F1") == ctl.HOLDS
    assert est(edge, "F_rep_def_definition") == pytest.approx(0.30)
    one = run(with_shares(D0fullB1__src=0.67))
    v = one["verdicts"]["F1"]
    assert v["status"] == ctl.FAILS and v["hits"] == 2  # S_src, for both variants
    assert "F_rep_src_definition +29.0 pp" in v["reason"]
    assert [t["ok"] for t in v["tests"]] == [False, True, False, True]


def test_p_f1_mechanical_variant_is_judged_the_same_way():
    tab = Builder().table()
    for r in tab.rows:
        if (
            r["system"] == SYSTEMS["D0fullB1"]
            and "repair_mech" in r
            and r["fact_id"].startswith("k.")
        ):
            r["repair_mech"] = True  # the stored values are all right, the answers are not
    res = ev(tab)
    v = res["verdicts"]["F1"]
    assert v["hits"] == 2 and v["status"] == ctl.FAILS  # behavioural repair passes, mechanical not
    assert [t["ok"] for t in v["tests"]] == [True, True, False, False]
    assert est(res, "F_repmech_src_definition") == pytest.approx(0.96 - 1.0)


def test_p_f1_readings_are_execution_access_or_both_at_the_registered_bounds():
    assert f1_reading(D0fullB1=0.30) == (ctl.HOLDS, "execution")  # 66 pp
    assert f1_reading(D0fullB1=0.66) == (ctl.HOLDS, "execution")  # 30 pp, inclusive
    assert f1_reading(D0fullB1=0.67) == (ctl.FAILS, "both")  # 29 pp
    assert f1_reading(D0fullB1=0.80) == (ctl.FAILS, "both")  # 16 pp
    assert f1_reading(D0fullB1=0.81) == (ctl.FAILS, "access")  # 15 pp, inclusive
    assert f1_reading(D0fullB1=0.96) == (ctl.FAILS, "access")  # no gap
    assert f1_reading(D0fullB1=1.00) == (ctl.FAILS, "access")  # D0full ahead of D1: -4 pp


def test_p_f1_readings_that_disagree_are_mixed_and_each_gap_keeps_its_region():
    res = run(with_shares(D0fullB1__src=0.30, D0fullB1__def=0.85))  # 66 pp in S_src, 11 pp in S_def
    v = res["verdicts"]["F1"]
    assert (v["status"], v["reading"]) == (ctl.FAILS, "mixed") and v["hits"] == 2
    assert v["regions"] == {
        "F_rep_src_definition": "execution",
        "F_rep_def_definition": "access",
        "F_repmech_src_definition": "execution",
        "F_repmech_def_definition": "access",
    }
    assert ctl.gap_region(None) is None
    assert [ctl.gap_region(g) for g in (0.30, 0.2999, 0.15, 0.1501, -0.04)] == [
        "execution", "both", "access", "both", "access",
    ]  # fmt: skip


def test_p_f1_is_not_judged_when_the_full_access_cell_was_not_run():
    b = Builder()
    b.drop |= {"D0fullB1"}
    res = run(b)
    v = res["verdicts"]["F1"]
    assert (
        v["status"] == ctl.NOT_JUDGED
        and v["reading"] is None
        and v["reason"].startswith("missing:")
    )
    assert status(res, "D1") == ctl.HOLDS and res["summary"] == {
        "holds": 19,
        "fails": 0,
        "not judged": 1,
    }
    e = res["contrasts"]["F_rep_src_definition"]
    assert math.isnan(e["est"]) and e["n_worlds"] == 0
    full = res["reported"]["full"]
    assert full["definitions"] == {} and full["truncated_definitions"] == []
    assert full["as_registered"] is False and "all" not in full


def test_the_control_was_run_on_src_and_def_only():
    assert ctl.family_branches()["F_rep"] == ("src", "def")
    assert ctl.family_branches()["D_rep"] == ("src", "def", "orphan")
    assert ctl.family_branches()["R_ratio"] == ("src", "def", "orphan")
    keys = ctl.contrast_keys()
    assert not [k for k in keys if k.startswith("F_") and "orphan" in k]
    assert [k for k in keys if k.startswith("F_rep_")] == [
        f"F_rep_{br}_{cls}" for br in ("src", "def") for cls in ("all", "definition", "untruncated")
    ]
    res = run(Builder())
    assert set(res["cells"]["holdmem_D0fullB1_source"]) == {"src", "def"}  # no orphan cell rates
    assert set(res["cells"]["holdmem_D0B1_source"]) == set(BRANCHES)
    assert res["branches"] == list(BRANCHES) and "F1" not in res["withdrawn"]


def test_the_access_gain_is_d0full_minus_d0_on_repair_and_update_without_a_threshold(holds):
    # D0 answers 14 of 100 definition items right, D0full 30 of 100: 16 pp on Update and on repair
    assert est(holds, "F_gain_upd_src_definition") == pytest.approx(0.30 - 0.14)
    assert est(holds, "F_gain_rep_def_definition") == pytest.approx(0.30 - 0.14)
    # the other items are the same requests in both cells: no gain on all items but the definitions
    assert est(holds, "F_gain_upd_src_all") == pytest.approx(0.16 * 100 / 205)
    res = run(with_shares(D0fullB1=0.80))  # only the definition items differ between the cells
    assert est(res, "F_gain_rep_src_definition") == pytest.approx(0.80 - 0.14)
    assert est(res, "F_gain_upd_def_all") == pytest.approx(0.66 * 100 / 205)
    assert status(res, "F1") == ctl.FAILS  # the gain has no verdict of its own
    assert not any(t["key"].startswith("F_gain") for t in holds["verdicts"]["F1"]["tests"])


def test_the_sensitivity_analysis_leaves_the_truncated_definitions_out_and_has_no_verdict():
    assert ctl.TRUNCATED_DEFS == ("k.close_friend", "k.vip_sender")
    untruncated = [f for f in DEFS if f not in ctl.TRUNCATED_DEFS]
    assert len(untruncated) == 98 and set(ctl.TRUNCATED_DEFS) <= set(DEFS)
    # D0full answers 66 of the 98 other definition items right and neither truncated one; D1 answers
    # 96 of 100 right, the truncated ones among them
    tab = Builder().table()
    set_full_answers(tab, set(untruncated[:66]))
    res = ev(tab)
    assert est(res, "F_rep_src_definition") == pytest.approx(30 / 100)  # 96 - 66 over 100 items
    assert est(res, "F_rep_src_untruncated") == pytest.approx(28 / 98)  # 94 - 66 over 98 items
    assert est(res, "F_repmech_def_untruncated") == pytest.approx(28 / 98)
    e = res["contrasts"]["F_rep_src_untruncated"]
    assert e["n_items"] == 3 * 2 * 98  # worlds x checkpoints x the untruncated definitions
    assert res["contrasts"]["F_rep_src_definition"]["n_items"] == 3 * 2 * 100
    # the verdict is the registered one, on all definition items: 30 pp holds although the
    # sensitivity analysis is at 28.6 pp, which is not judged
    assert status(res, "F1") == ctl.HOLDS and res["verdicts"]["F1"]["reading"] == "execution"
    assert not any("untruncated" in t["key"] for t in res["verdicts"]["F1"]["tests"])
    # the access gain, too, is reported on both sets of definitions
    assert est(res, "F_gain_rep_src_untruncated") == pytest.approx((66 - 12) / 98)
    assert est(res, "F_gain_upd_src_definition") == pytest.approx(66 / 100 - 0.14)
    # the class is the definition items without the two, and nothing else
    assert ctl.CLASSES["untruncated"]("k.active_project") and ctl.CLASSES["untruncated"]("k.d00")
    assert not ctl.CLASSES["untruncated"]("k.close_friend")
    assert not ctl.CLASSES["untruncated"]("k.vip_sender")
    assert not ctl.CLASSES["untruncated"]("c.p00.city") and not ctl.CLASSES["untruncated"]("u.diet")


def test_truncation_counts_per_definition_and_source_come_from_the_notes(holds):
    full = holds["reported"]["full"]
    assert (
        full["limit_tokens"] == 25_000
        and set(full) >= {"all", "src", "def"}
        and "orphan" not in full
    )
    # 3 worlds x 6 definitions x 2 checkpoints per branch; the write at t0 carries no note
    assert [full[b]["rederivations"] for b in ("src", "def", "all")] == [36, 36, 72]
    a = full["all"]
    assert (a["truncated"], a["by_source_rule"], a["newest_fallback"]) == (24, 24, 0)
    assert a["truncated_share"]["est"] == pytest.approx(24 / 72) and a["capped"] == 72 * 3
    assert full["src"]["dropped"] == 3 * 2 * (12 + 40) and a["dropped"] == 2 * 3 * 2 * (12 + 40)
    assert (a["tokens_median"], a["tokens_max"]) == (9_000, 24_900)
    vip = full["definitions"]["k.vip_sender"]["src"]
    assert (vip["rederivations"], vip["truncated"], vip["newest"], vip["dropped"]) == (6, 6, 0, 240)
    assert vip["by_source"] == {
        "email": {"rederivations": 6, "truncated": 6, "records": 60, "dropped": 240},
        "contacts": {"rederivations": 6, "truncated": 0, "records": 60, "dropped": 0},
    }  # the small source stays whole
    friend = full["definitions"]["k.close_friend"]["all"]
    assert friend["by_source"]["messages"] == {
        "rederivations": 12, "truncated": 12, "records": 120, "dropped": 12 * 12,
    }  # fmt: skip
    assert full["definitions"]["k.active_project"]["all"]["truncated"] == 0
    assert set(full["definitions"]) == {f"k.{k}" for k in KAPPA_DEFS}
    assert full["truncated_definitions"] == ["k.close_friend", "k.vip_sender"]
    assert full["registered_truncated"] == list(ctl.TRUNCATED_DEFS) and full["as_registered"]


def test_the_truncation_count_says_when_other_definitions_were_cut_than_registered():
    b = Builder()
    b.truncated["k.active_project"] = ("tasks", 5, "newest")  # the fallback, on a third definition
    full = run(b)["reported"]["full"]
    assert full["truncated_definitions"] == ["k.active_project", "k.close_friend", "k.vip_sender"]
    assert full["as_registered"] is False
    assert full["all"]["newest_fallback"] == 12 and full["all"]["by_source_rule"] == 24
    assert full["definitions"]["k.active_project"]["all"]["newest"] == 12
    none = Builder()
    none.truncated = {}
    full = run(none)["reported"]["full"]
    assert full["truncated_definitions"] == [] and full["as_registered"] is False
    assert full["all"]["truncated"] == 0 and full["all"]["truncated_share"]["est"] == 0.0
    one = Builder()
    one.truncated = {"k.close_friend": ("messages", 12, "source")}
    assert run(one)["reported"]["full"]["as_registered"] is False  # vip_sender was not cut


# ------------------------------------------------------------------------ agreement with E1
def test_engine_contrasts_are_the_published_e1_contrasts():
    """The same paired difference through `e1.contrasts` and through the engine, to the digit."""
    table = Builder().table()
    eng = ctl.Engine(table)
    four = ("holdmem_D0B0", "holdmem_D1B0", "holdmem_D0B1_source", "holdmem_D1B1_source")
    for br in BRANCHES:
        pub = e1.contrasts([r for r in table.rows if r["branch"] == br and r["system"] in four], br)
        got = eng.diff(ctl.Spec("x", "D1B1", "D1B0", br, "update", "all"))
        assert got.to_json() == pub["B_effect_update_D1_source"]
        got = eng.diff(ctl.Spec("y", "D1B1", "D0B1", br, "repair", "all"))
        assert got.to_json() == pub["D_effect_repair_source"]
        got = eng.diff(ctl.Spec("z", "D1B1", "D0B1", br, "repair_mech", "all"))
        assert got.to_json() == pub["D_effect_repair_mech_source"]


def test_ratio_and_pooled_contrasts_use_the_published_estimators():
    eng = ctl.Engine(Builder().table(), draws=FAST)
    eng.compute_all()
    num, den = eng.diffs["R_ref_src_definition"], eng.diffs["R_b1_src_definition"]
    assert ctl.ratio_estimate(num, den) == _share(
        num, den
    )  # the registered draws: `e1_synth._share`
    parts = ctl.POOLED["G_pool_all"]
    by_world = {w: [v for p in parts for v in eng.diffs[p].get(w, {}).values()] for w in WORLDS}
    assert eng.contrasts["G_pool_all"] == world_bootstrap(by_world, draws=FAST, seed=996)
    assert (
        eng.contrasts["G_pool_all"].n_items == 6 * 3 * 2 * 205
    )  # six contrasts, all changed items


def test_bootstrap_is_the_registered_one():
    """2,000 draws, seed 996, worlds as clusters (`scoring.stats`): a constant difference has a
    degenerate interval; with one world that gains nothing it is the registered interval."""
    tab = Builder().table()
    spec = ctl.Spec("z", "D1B1", "D0B1", "src", "update", "definition")
    e = ctl.Engine(tab).diff(spec)
    assert (e.lo, e.est, e.hi) == pytest.approx((0.82, 0.82, 0.82))
    for r in tab.rows:
        if r["system"] == "holdmem_D1B1_source" and r["world_id"] == "w0003":
            r["cv"] = False
    e = ctl.Engine(tab).diff(spec)
    d0 = [right(i, 100, 0.14) for i in range(100)]
    per_world = {
        w: [
            float(a) - float(b)
            for _ in (1, 2)  # two checkpoints
            for a, b in zip(
                [False] * 100 if w == "w0003" else [right(i, 100, 0.96) for i in range(100)],
                d0,
                strict=True,
            )
        ]
        for w in WORLDS
    }
    want = world_bootstrap(per_world, draws=2000, seed=996)
    assert e.to_json() == pytest.approx(want.to_json())
    assert (e.est, e.n_worlds, e.n_items) == pytest.approx((0.5, 3, 600))
    assert e.lo < e.est < e.hi


# ------------------------------------------------------------------------------ k = 0
def test_contrasts_before_the_shift_use_the_accuracy_at_t0_of_the_items_asked_then():
    b = Builder()
    b.k0 = {("zeroDm", "definition"): 0.50, ("zero", "definition"): 0.10}
    res = run(b)
    assert est(res, "Z0_upd_src_definition") == pytest.approx(0.40)
    assert est(res, "Z0_upd_src_rule") == pytest.approx(0.70 - 0.60)  # the ordinary shares
    e = res["contrasts"]["Z0_upd_src_definition"]
    assert e["n_items"] == 3 * 100  # worlds x definition items, asked once each
    low = Builder()
    low.k0 = {("zeroDm", "definition"): 0.20, ("zero", "definition"): 0.10}
    assert status(run(low), "Z1k") == ctl.FAILS


# ---------------------------------------------------------------------------- reported
def test_program_t0_share_ratio_and_lineage_counts():
    b = Builder()
    b.prog_ok = {"D1sB1": 0.8, "D1sB1ra": 0.6}
    res = run(b)
    p = res["reported"]["program_t0"]
    assert p["all"]["ra"]["est"] == pytest.approx(0.6) and p["all"]["d1s"]["est"] == pytest.approx(
        0.8
    )
    assert p["all"]["ratio"]["est"] == pytest.approx(0.75)
    assert set(p) == {"all", "src", "def", "orphan"}
    ra = res["reported"]["ra"]["orphan"]["all"]
    # one of the ten slots per (world, branch) returns a value although a prior read is gone
    assert ra["runs"] == 3 * 10 and ra["lineage_false"]["est"] == pytest.approx(0.1)
    assert ra["final_items"] == 3 * 10
    assert ra["final_undeclared"]["est"] == pytest.approx(0.0)


def test_undeclared_reads_are_counted_on_the_final_state_of_failed_items():
    b = Builder()
    items = b.items()
    for it in items:
        if it["system"] == "holdmem_D1sB1_source_ra" and it["slot_id"] == "k.active_project":
            it["det"] = {"program_ok": False, "reason": "program failed: undeclared read: email:e1"}
        if it["system"] == "holdmem_D1sB1_source_ra" and it["slot_id"] == "k.overdue_task":
            it["det"] = {"program_ok": False, "reason": "program failed: timeout"}
    tab = b.table()
    tab.items = items
    ra = ev(tab)["reported"]["ra"]["def"]["definition"]
    assert ra["final_items"] == 3 * 5
    assert ra["final_failed"]["est"] == pytest.approx(2 / 5)
    assert ra["final_undeclared"]["est"] == pytest.approx(1 / 5)
    assert ra["final_undeclared_of_failed"]["est"] == pytest.approx(1 / 2)


def test_over_retraction_counts_withdrawn_values_of_task_definitions():
    b = Builder()
    b.withdrawn_none = {"k.active_project", "k.overdue_task"}
    over = run(b)["reported"]["over_retraction"]["orphan"]
    ra, dm = over["D1sB1ra"], over["D1sB1"]
    assert ra["withdrawn_share"]["taskdef"]["est"] == pytest.approx(1.0)
    assert ra["withdrawn_share"]["otherdef"]["est"] == pytest.approx(0.0)
    assert dm["withdrawn_share"]["taskdef"]["est"] == pytest.approx(0.0)
    f = ra["task_definitions"]["k.active_project"]
    assert (f["repaired"], f["withdrawn"]) == (3 * 2, 3 * 2)  # worlds x checkpoints
    # rows whose gold has no answer are not over-retraction: a withdrawn value is right there
    tab = b.table()
    for r in tab.rows:
        if r["system"] == "holdmem_D1sB1_source_ra" and r["fact_id"] == "k.active_project":
            r["gold_status"] = "abstain"
    f2 = ev(tab)["reported"]["over_retraction"]["orphan"]["D1sB1ra"]
    assert f2["task_definitions"]["k.active_project"]["repaired"] == 0


def test_over_retraction_contrasts_split_task_and_other_definitions():
    tab = Builder().table()
    for r in tab.rows:
        if (
            r["system"] == "holdmem_D1sB1_source_ra"
            and r["branch"] == "orphan"
            and r["fact_id"] in ctl.TASK_DEFS
        ):
            r["cv"] = False  # rule 3 withdrew the task definitions: the answer is wrong
    res = ev(tab)
    assert est(res, "O_upd_orphan_taskdef") == pytest.approx(0.0 - 1.0)  # D1s answers both right
    assert est(res, "O_upd_orphan_otherdef") == pytest.approx(6 / 98)  # ra 94 of 98, D1s 88 of 98
    assert est(res, "O_upd_src_taskdef") == pytest.approx(0.0)  # other branches untouched


# ----------------------------------------------------------------------------------- checks
CELL_OF_KEY = {"gold": "holdmem_D0B1_gold", "scope": "holdmem_D1B1_scope"}  # one cell per key


def expected_audit(table: cd.Table, key: str) -> Callable:
    """The offline audit that agrees with the table: per branch and class, the counts of the
    cell of `key` computed by plain loops over the rows."""

    def audit(seeds, branches, months):
        out: dict = {}
        for r in table.rows:
            if "detected" not in r or r["system"] != CELL_OF_KEY[key]:
                continue
            fact = r["fact_id"]
            cls = (
                "definition"
                if fact.startswith("k.")
                else "norule"
                if fact.startswith("u.")
                else "rule"
            )
            if key == "scope" and cls != "definition":
                continue
            c = (
                out.setdefault(r["branch"], {})
                .setdefault(key, {})
                .setdefault(cls, defaultdict(int))
            )
            c["changed" if r["changed"] else "unchanged"] += 1
            c["changed_detected"] += r["changed"] and r["detected"]
            c["unchanged_detected"] += (not r["changed"]) and r["detected"]
        for b in out.values():
            for c in b[key].values():
                c["rederivations"] = 0  # the synthetic items carry no invalidation events
        return out

    return audit


def test_audit_check_marks_each_cell_pass_or_fail():
    tab = Builder().table()

    def both(seeds, branches, months):
        out = expected_audit(tab, "gold")(seeds, branches, months)
        for br, node in expected_audit(tab, "scope")(seeds, branches, months).items():
            out.setdefault(br, {}).update(node)
        return out

    res = ev(tab, both)
    audit = res["checks"]["audit"]
    assert audit["pass"] is True
    assert set(audit["cells"]) == {"holdmem_D0B1_gold", "holdmem_D1B1_gold", "holdmem_D1B1_scope"}
    scope = audit["cells"]["holdmem_D1B1_scope"]["branches"]["src"]
    assert set(scope) == {"definition"}  # the other items of a scope cell are keyed by citations
    gold = audit["cells"]["holdmem_D0B1_gold"]["branches"]["orphan"]
    assert set(gold) == {"definition", "rule", "norule"}
    assert gold["definition"]["run_rates"] == {"missed": 0.0, "false_invalidation": None}

    def off_by_one(seeds, branches, months):
        out = both(seeds, branches, months)
        out["def"]["gold"]["definition"]["changed_detected"] -= 1
        return out

    bad = ev(tab, off_by_one)["checks"]
    assert bad["audit"]["pass"] is False and bad["failed"] == 1
    cell = bad["audit"]["cells"]["holdmem_D0B1_gold"]
    assert cell["pass"] is False
    grp = cell["branches"]["def"]["definition"]
    assert (
        grp["pass"] is False
        and grp["run"]["changed_detected"] == grp["audit"]["changed_detected"] + 1
    )
    assert bad["audit"]["cells"]["holdmem_D1B1_scope"]["pass"] is True


def test_scope_cell_on_non_definition_items_must_be_the_cited_cell():
    tab = Builder().table()
    ok = ctl.scope_cited_check(tab)
    assert ok["pass"] is True and ok["row_mismatches"] == 0 and ok["rows"] > 0
    assert ok["scope"] == ok["cited"] and ok["scope"]["missed"] == pytest.approx(0.2)
    flipped = False
    for r in tab.rows:
        if r["system"] == "holdmem_D1B1_scope" and r["fact_id"] == "u.employer" and r["changed"]:
            r["detected"] = not r["detected"]
            flipped = True
    assert flipped
    bad = ctl.scope_cited_check(tab)
    assert bad["pass"] is False and bad["row_mismatches"] > 0
    missing = Builder()
    missing.drop |= {"D1B1cited"}
    assert ctl.scope_cited_check(missing.table())["pass"] is None


def test_cell_rates_carry_missed_and_false_invalidation_of_each_cell():
    res = run(Builder())
    scope = res["cells"]["holdmem_D1B1_scope"]["src"]
    assert scope["definition"]["MissedInvalidation"]["est"] == 0.0
    assert scope["other"]["MissedInvalidation"]["est"] == pytest.approx(0.2)
    assert scope["all"]["MissedInvalidation"]["est"] == pytest.approx(21 / 205)
    assert "MissedInvalidation" not in res["cells"]["zero"]["src"]["all"]
    assert res["cells"]["holdmem_D0B0"]["src"]["all"]["Update"]["n_items"] > 0


def test_verdict_helper_boundaries():
    t = [ctl.Test("a", -0.02, True, ""), ctl.Test("b", None, None, "")]
    assert ctl.verdict(t)["status"] == ctl.NOT_JUDGED
    assert ctl.verdict([ctl.Test("a", 0.0, True, "")], need=1)["status"] == ctl.HOLDS
    assert ctl.verdict([ctl.Test("a", 0.0, False, "")], need=1)["status"] == ctl.FAILS
    assert (
        ctl._at_least("x", -0.02, -0.02).ok is True
        and ctl._at_least("x", -0.0201, -0.02).ok is False
    )
    assert ctl._within("x", 0.03, 0.03).ok is True and ctl._within("x", -0.0301, 0.03).ok is False
    assert ctl.pp_text(None) == "missing" and ctl.pp_text(-0.078) == "-7.8 pp"
    assert ctl.not_judged("why")["tests"] == []
