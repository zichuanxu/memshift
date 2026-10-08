"""Exploratory writer-switch predictors, baselines and type-level intervals
(`analysis.ex_e2_predictors`, wired into `e2.analyse` as `explore_writer_*`)."""

import json
import math
import random
from pathlib import Path

import numpy as np
import pytest

from memshift.analysis import e2
from memshift.analysis import ex_e2_predictors as ex
from memshift.analysis.ex_e2_predictors import Point

# ------------------------------------------------------------------ count statistics = registered
XS = [0.0, 0.3, 0.6, 0.83, 0.97, 0.995, 1.0]  # includes values the logit clip merges


def _random_points(seed: int, n: int = 400) -> tuple[list[float], list[float]]:
    rng = random.Random(seed)
    x = [rng.choice(XS) for _ in range(n)]
    y = [float(rng.random() < 0.2 + 0.75 * v) for v in x]
    return x, y


def _one_cluster(x: list[float], y: list[float]) -> ex._Tables:
    return ex._Tables(np.zeros(len(x), dtype=int), np.array(y), np.array(x), 1)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_statistics_from_count_tables_equal_the_registered_functions(seed):
    x, y = _random_points(seed)
    auc, a, b = _one_cluster(x, y).stats(np.ones((1, 1)))
    assert auc[0] == pytest.approx(e2.auroc(x, y), abs=1e-12)
    assert b[0] == pytest.approx(e2.slope_on_logit(x, y), abs=1e-6)
    want_a, want_b = e2.logistic_fit([e2._logit(v) for v in x], y)
    assert (a[0], b[0]) == pytest.approx((want_a, want_b), abs=1e-6)


def test_statistics_at_the_edges_match_too():
    # one class, one predictor value: undefined, as in the registered functions
    for x, y in (([0.5, 0.9], [1.0, 1.0]), ([0.9, 0.9], [1.0, 0.0])):
        auc, a, b = _one_cluster(x, y).stats(np.ones((1, 1)))
        assert math.isnan(b[0]) and math.isnan(e2.slope_on_logit(x, y))
    auc, _, _ = _one_cluster([0.5, 0.9], [1.0, 1.0]).stats(np.ones((1, 1)))
    assert math.isnan(auc[0]) and math.isnan(e2.auroc([0.5, 0.9], [1.0, 1.0]))
    # ties count half; a perfectly separating predictor reaches AUROC 1 and the same iteration limit
    x, y = [0.2, 0.2, 0.9, 0.9], [0.0, 1.0, 1.0, 1.0]
    auc, _, b = _one_cluster(x, y).stats(np.ones((1, 1)))
    assert auc[0] == e2.auroc(x, y) == pytest.approx(5 / 6)  # (0.5 + 1 + 1) / 3
    assert b[0] == pytest.approx(e2.slope_on_logit(x, y), rel=1e-6)
    x, y = [0.2, 0.2, 0.9, 0.9], [0.0, 0.0, 1.0, 1.0]
    _, _, b = _one_cluster(x, y).stats(np.ones((1, 1)))
    assert b[0] == pytest.approx(e2.slope_on_logit(x, y), rel=1e-6) and b[0] > 10


def test_multiplicities_replay_the_registered_draw_order():
    m = ex.multiplicities(5, 7, seed=996)
    assert m.shape == (7, 5) and (m.sum(axis=1) == 5).all()
    rng = random.Random(996)
    first = [rng.randrange(5) for _ in range(5)]
    assert list(m[0]) == [first.count(i) for i in range(5)]
    assert (ex.multiplicities(5, 7, seed=996) == m).all()


def _by_world(seed: int = 4) -> dict[str, list[tuple[float, float]]]:
    rng = random.Random(seed)
    out = {}
    for w in range(9):
        pts = []
        for _ in range(60):
            p = rng.choice([0.3, 0.6, 0.83, 0.97, 1.0])
            pts.append((p, float(rng.random() < 0.15 + 0.85 * p)))
        out[f"w{w}"] = pts
    return out


def test_world_bootstrap_of_the_agreement_rate_reproduces_the_registered_intervals():
    by_world = _by_world()
    points = [
        Point(
            "t",
            w,
            "ty",
            False,
            y,
            {"agree": x, "new_acc": x, "old_acc": x, "cond": x, "nondef": 1.0},
        )
        for w, pts in by_world.items()
        for x, y in pts
    ]
    got = ex._scope(points, ("agree", "new_acc"), draws=200, seed=996)["predictors"]["agree"]
    slope, auroc = (
        e2.boot(by_world, e2.slope_on_logit, 200, 996),
        e2.boot(by_world, e2.auroc, 200, 996),
    )
    assert got["slope"]["est"] == pytest.approx(slope["est"], abs=1e-6)
    assert got["auroc"]["est"] == pytest.approx(auroc["est"], abs=1e-12)
    for key in ("lo90", "hi90", "lo95", "hi95"):
        assert got["slope"]["world"][key] == pytest.approx(slope[key], abs=1e-6)
        assert got["auroc"]["world"][key] == pytest.approx(auroc[key], abs=1e-9)
    # a predictor equal to the agreement rate does not differ from it
    assert (
        got["auroc"]["est"]
        == ex._scope(points, ("agree", "new_acc"), 50, 1)["predictors"]["new_acc"]["auroc"]["est"]
    )


# ------------------------------------------------------------------------- calibration and rates
def _cal(t: str, n: int, old: int, new: int, both: int) -> list[dict]:
    """n calibration rows of type `t`; `old` / `new` rows right for the old / new writer, `both`
    right for both (the first `both` rows)."""
    rows = []
    for i in range(n):
        o = i < old
        nw = i < both or (i >= old and i < old + (new - both))
        rows.append(
            {
                "world_id": "w0900",
                "slot_id": "k.x" if t.startswith("kappa") else "c.x",
                "type": t,
                "old_correct": o,
                "new_correct": nw,
                "agree": o == nw,
                "new_check_pass": nw,
            }
        )
    return rows


def test_type_rates_follow_the_calibration_rows():
    rows = _cal("contact.city", 10, 8, 6, 5)  # old 8, new 6, both 5
    r = ex.type_rates(rows)["contact.city"]
    assert r["old_acc"] == 0.8 and r["new_acc"] == 0.6 and r["cond"] == pytest.approx(5 / 8)
    assert r["agree"] == pytest.approx((5 + 1) / 10)  # both right on 5, both wrong on 1
    # a type the old writer never gets right: P(new | old) is undefined, independence fallback
    never = ex.type_rates(_cal("contact.city", 10, 0, 4, 0))["contact.city"]
    assert never["cond"] == never["new_acc"] == 0.4


TYPES = (  # type, slot id, calibration rows, right for the old writer, the new one, both
    ("contact.city", "c.p01.city", 10, 10, 10, 10),
    ("contact.employer", "c.p01.employer", 10, 9, 9, 9),
    ("kappa.active_project", "k.active_project", 10, 10, 6, 6),
    ("kappa.overdue_task", "k.overdue_task", 10, 10, 2, 2),
    ("task_due", "t.t01.due", 10, 10, 7, 7),
)


def _synthetic(tmp_path: Path, tiers=("same", "downgrade"), with_cal=True) -> list[dict]:
    """Survival rows for 6 worlds x the types of `TYPES` (D = 0, B = 1, valid before); an item of a
    type survives with the type's new-writer accuracy, deterministically by world index."""
    rows = []
    for tier in tiers:
        cal = []
        for t, _slot, n, old, new, both in TYPES:
            cal += _cal(t, n, old, new, both)
        if with_cal:
            (tmp_path / f"calibration_{tier}.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in cal)
            )
        rates = ex.type_rates(cal)
        for w in range(6):
            for t, slot, _n, *_ in TYPES:
                for i in range(10):
                    rows.append(
                        {
                            "tier": tier,
                            "system": "holdmem_D0B1_source",
                            "world_id": f"w{1000 + w}",
                            "slot_id": slot,
                            "type": t,
                            "D": 0,
                            "B": 1,
                            "valid_before": True,
                            "valid_after": (i + w) % 10 < round(10 * rates[t]["new_acc"]),
                            "predicted": rates[t]["agree"],
                        }
                    )
    return rows


def test_analyse_scores_every_predictor_pooled_and_per_tier(tmp_path):
    rows = _synthetic(tmp_path)
    out = ex.analyse(tmp_path, rows, draws=100)
    w = out["explore_writer_predictors"]
    assert w["tiers"] == ["downgrade", "same"] and w["tiers_without_calibration"] == []
    pooled = w["sets"]["all"]["pooled"]
    assert pooled["n_items"] == 2 * 6 * 50 == len(rows)
    failures = sum(not r["valid_after"] for r in rows)
    assert pooled["n_failures"] == failures and pooled["n_worlds"] == 6 and pooled["n_types"] == 5
    assert set(pooled["predictors"]) == {"agree", "new_acc", "old_acc", "cond", "nondef"}
    # the registered predictor reproduces path 3's estimates on the same rows
    pts: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        pts.setdefault(r["world_id"], []).append((r["predicted"], float(r["valid_after"])))
    agree = pooled["predictors"]["agree"]
    assert agree["slope"]["est"] == pytest.approx(
        e2.boot(pts, e2.slope_on_logit, 10)["est"], abs=1e-6
    )
    assert agree["auroc"]["est"] == pytest.approx(e2.boot(pts, e2.auroc, 10)["est"], abs=1e-12)
    # new-writer accuracy is exactly the survival rate here, so it ranks items at least as well
    new = pooled["predictors"]["new_acc"]
    assert new["auroc"]["est"] >= agree["auroc"]["est"] - 1e-12
    assert new["auroc_minus_agree"]["est"] == pytest.approx(
        new["auroc"]["est"] - agree["auroc"]["est"]
    )
    assert "auroc_minus_agree" not in agree
    # the definition indicator: definition items are the failing ones (accuracy 0.6 and 0.2)
    nd = pooled["predictors"]["nondef"]
    assert nd["auroc"]["est"] > 0.5
    for p in pooled["predictors"].values():
        for kind in ("world", "type"):
            assert p["slope"][kind]["draws"] > 0 or math.isnan(p["slope"][kind]["lo90"])
    # items outside the definition class only
    nodef = w["sets"]["nodef"]["pooled"]
    assert nodef["n_items"] == 2 * 6 * 30 and "nondef" not in nodef["predictors"]
    assert w["n_items_nodef"] == nodef["n_items"]


def test_type_bootstrap_resamples_item_types_and_world_bootstrap_does_not(tmp_path):
    rows = _synthetic(tmp_path, tiers=("same",))
    out = ex.analyse(tmp_path, rows, draws=300)
    s = out["explore_writer_predictors"]["sets"]["all"]["same"]["predictors"]["agree"]["slope"]
    # five types, each one agreement value: resampling types changes the slope a lot; resampling
    # worlds (every world holds every type) changes it little
    type_width = s["type"]["hi90"] - s["type"]["lo90"]
    world_width = s["world"]["hi90"] - s["world"]["lo90"]
    assert type_width > 3 * world_width
    # both are deterministic
    again = ex.analyse(tmp_path, rows, draws=300)
    assert json.dumps(again, sort_keys=True) == json.dumps(out, sort_keys=True)


def test_leave_one_type_out_drops_the_type_and_recomputes(tmp_path):
    rows = _synthetic(tmp_path, tiers=("same",))
    loto = ex.analyse(tmp_path, rows, draws=20)["explore_writer_loto"]["pooled"]
    assert set(loto["types"]) == {t for t, *_ in TYPES}
    for t, *_ in TYPES:
        kept = [r for r in rows if r["type"] != t]
        xs, ys = [r["predicted"] for r in kept], [float(r["valid_after"]) for r in kept]
        got = loto["types"][t]["predictors"]["agree"]
        assert got["slope"] == pytest.approx(e2.slope_on_logit(xs, ys), abs=1e-6)
        assert got["auroc"] == pytest.approx(e2.auroc(xs, ys), abs=1e-12)
        dropped = [r for r in rows if r["type"] == t]
        assert loto["types"][t]["n_dropped"] == len(dropped)
        assert loto["types"][t]["failures_dropped"] == sum(not r["valid_after"] for r in dropped)
    rng = loto["range"]["agree"]
    slopes = [e["predictors"]["agree"]["slope"] for e in loto["types"].values()]
    assert rng["slope_min"] == min(slopes) and rng["slope_max"] == max(slopes)
    assert loto["types"][rng["slope_min_type"]]["predictors"]["agree"]["slope"] == min(slopes)


def test_failures_are_counted_per_type_and_tier(tmp_path):
    rows = _synthetic(tmp_path)
    f = ex.analyse(tmp_path, rows, draws=10)["explore_failures_by_type"]
    by = {(r["tier"], r["type"]): 0 for r in rows}
    for r in rows:
        by[r["tier"], r["type"]] += not r["valid_after"]
    for (tier, t), n in by.items():
        assert f[tier][t]["failures"] == n and f[tier][t]["n"] == 60
    assert f["pooled"]["kappa.overdue_task"] == {"n": 120, "failures": 96, "definition": True}
    s = f["summary"]
    assert s["n_failures"] == sum(by.values()) and s["n_items"] == len(rows)
    assert s["definition_failures"] == sum(n for (_, t), n in by.items() if t.startswith("kappa"))
    assert s["top_two_types"] == ["kappa.overdue_task", "kappa.active_project"]
    assert s["top_two_failures"] == 96 + f["pooled"]["kappa.active_project"]["failures"]
    assert s["type_tier_cells"] == 10 and s["types"] == 5
    assert s["type_tier_cells_with_failures"] == sum(v > 0 for v in by.values())


def test_a_tier_without_calibration_is_left_out_not_guessed(tmp_path):
    rows = _synthetic(tmp_path, tiers=("same",)) + [
        {**r, "tier": "cross_model"} for r in _synthetic(tmp_path, tiers=("same",), with_cal=False)
    ]
    out = ex.analyse(tmp_path, rows, draws=10)["explore_writer_predictors"]
    assert out["tiers"] == ["same"] and out["tiers_without_calibration"] == ["cross_model"]
    assert out["sets"]["all"]["pooled"]["n_items"] == 6 * 50


def test_no_calibration_at_all_leaves_empty_scopes_and_no_crash(tmp_path):
    out = ex.analyse(tmp_path, _synthetic(tmp_path, with_cal=False), draws=10)
    assert out["explore_writer_predictors"]["sets"]["all"]["pooled"]["n_items"] == 0
    assert out["explore_writer_loto"]["pooled"] == {"types": {}, "range": {}}
    assert out["explore_failures_by_type"]["summary"]["n_failures"] == 0


def test_e2_analyse_stores_the_explore_fields_beside_the_registered_ones(tmp_path):
    rows = _synthetic(tmp_path)
    for r in rows:  # e2.analyse needs the other cells and keys of a survival row
        r.setdefault("check", None)
        r.setdefault("invalidated", 0)
    (tmp_path / "survival.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    res = e2.analyse(tmp_path)
    for key in (
        "explore_writer_predictors",
        "explore_writer_loto",
        "explore_failures_by_type",
        "explore_survival_recovery",
        "explore_t1_eps",
        "explore_t1_by_class",
    ):
        assert key in res
    reg = res["path3"]["pooled_slope"]
    mine = res["explore_writer_predictors"]["sets"]["all"]["pooled"]["predictors"]["agree"]["slope"]
    assert mine["est"] == pytest.approx(reg["est"], abs=1e-6)
    assert mine["world"]["lo90"] == pytest.approx(reg["lo90"], abs=1e-6)
    assert mine["world"]["hi90"] == pytest.approx(reg["hi90"], abs=1e-6)
    saved = json.loads((tmp_path / "analysis" / "e2_metrics.json").read_text())
    assert "explore_writer_predictors" in saved
