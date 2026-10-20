"""Multiplicity check (`analysis.multiplicity`): Holm-Bonferroni and Benjamini-Hochberg
on hand-computed p-value vectors, the direction of the one-sided bootstrap p-value, and the whole
pipeline on a tiny synthetic E1 item table (recomputed estimates equal the ones in the metrics).
The committed stages are checked too when their item tables are there (they are not committed)."""

import json
import math
import random
from pathlib import Path

import pytest

from memshift.analysis import e1, multiplicity
from memshift.analysis.replication import PREDICTED_NONZERO
from memshift.scoring.stats import world_bootstrap, world_bootstrap_draws

ROOT = Path(__file__).resolve().parents[1]
DRAWS = multiplicity.DRAWS


# ----------------------------------------------------------------------------- Holm and BH
def test_holm_hand_computed():
    # sorted: .005 .01 .03 .04 (m = 4): multipliers 4 3 2 1 -> .02 .03 .06 .04, running maximum
    # .02 .03 .06 .06; back in input order
    adj, reject = multiplicity.holm([0.01, 0.04, 0.03, 0.005])
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert reject == [True, False, False, True]
    # the thresholds .05 / (m - i + 1) = .0125, .0167, .025, .05: .03 fails at step 3, so .04
    # (which is below .05) is not tested either
    assert sum(reject) == 2


def test_holm_stops_at_the_first_failure():
    # every p is below alpha, yet the smallest already fails its .05 / 3 threshold: step-down
    # stops, and the later, larger p-values are not rejected (the running maximum)
    adj, reject = multiplicity.holm([0.02, 0.021, 0.022])
    assert adj == pytest.approx([0.06, 0.06, 0.06])
    assert reject == [False, False, False]
    # but a first success lets the next ones through as long as each meets its own threshold
    adj, reject = multiplicity.holm([0.01, 0.02, 0.5])
    assert adj == pytest.approx([0.03, 0.04, 0.5])
    assert reject == [True, True, False]


def test_holm_ties_share_the_adjusted_value_and_the_decision():
    adj, reject = multiplicity.holm([0.01, 0.2, 0.01])
    assert adj == pytest.approx([0.03, 0.2, 0.03])  # the tied pair takes the larger multiplier
    assert reject == [True, False, True]
    adj, reject = multiplicity.holm([0.02, 0.02, 0.02])  # .06 > alpha for all three
    assert adj == pytest.approx([0.06] * 3) and reject == [False] * 3


def test_holm_caps_at_one_and_a_single_test_is_unadjusted():
    assert multiplicity.holm([0.6, 0.9])[0] == [1.0, 1.0]
    assert multiplicity.holm([0.03]) == ([0.03], [True])
    assert multiplicity.holm([0.05]) == ([0.05], [True])  # alpha itself is rejected
    assert multiplicity.holm([]) == ([], [])
    assert multiplicity.holm([0.001, 0.2], alpha=0.001)[1] == [False, False]  # 2 * .001 > .001


def test_bh_hand_computed():
    # sorted .005 .01 .03 .04 (m = 4): m p / j = .02 .02 .04 .04; running minimum from the
    # largest p down: .04 .04 .02 .02; back in input order
    adj, reject = multiplicity.benjamini_hochberg([0.01, 0.04, 0.03, 0.005])
    assert adj == pytest.approx([0.02, 0.04, 0.04, 0.02])
    assert reject == [True, True, True, True]  # p_(4) = .04 <= 4 * .05 / 4


def test_bh_steps_up_where_holm_stops():
    ps = [0.001, 0.04, 0.041]
    adj, reject = multiplicity.benjamini_hochberg(ps)
    # p_(3) = .041 <= 3 q / 3 = .05, so all three are rejected, including .04, which fails its own
    # k q / m = .0333 threshold
    assert adj == pytest.approx([0.003, 0.041, 0.041])
    assert reject == [True, True, True]
    assert multiplicity.holm(ps)[1] == [True, False, False]


def test_bh_ties_and_cap_and_edges():
    adj, reject = multiplicity.benjamini_hochberg([0.02, 0.02, 0.02])
    assert adj == pytest.approx([0.02] * 3) and reject == [True] * 3
    adj, reject = multiplicity.benjamini_hochberg([0.9, 0.95])  # capped, then the running minimum
    assert adj == pytest.approx([0.95, 0.95]) and reject == [False, False]
    assert multiplicity.benjamini_hochberg([0.04]) == ([0.04], [True])
    assert multiplicity.benjamini_hochberg([]) == ([], [])
    # nothing passes k q / m for any k: no rejection, and the smallest adjusted value is > q
    adj, reject = multiplicity.benjamini_hochberg([0.03, 0.2, 0.4])
    assert reject == [False] * 3 and min(adj) > multiplicity.Q


def _holm_reference(ps: list[float], alpha: float) -> list[bool]:
    """The textbook loop: reject in increasing p while p_(i) <= alpha / (m - i + 1)."""
    m, order = len(ps), sorted(range(len(ps)), key=lambda i: ps[i])
    out = [False] * m
    for rank, i in enumerate(order):
        if ps[i] > alpha / (m - rank):
            break
        out[i] = True
    return out


def _bh_reference(ps: list[float], q: float) -> list[bool]:
    """The textbook rule: reject every p up to the largest p_(k) <= k q / m."""
    m, order = len(ps), sorted(range(len(ps)), key=lambda i: ps[i])
    k = max((r + 1 for r, i in enumerate(order) if ps[i] <= (r + 1) * q / m), default=0)
    return [k > 0 and p <= ps[order[k - 1]] for p in ps]


def test_procedures_agree_with_the_textbook_rules_on_random_vectors():
    rng = random.Random(7)
    for _ in range(300):
        m = rng.randint(1, 12)
        # multiples of 1 / 2001 (the bootstrap's own grid, many ties): no p sits exactly on a
        # threshold, so the adjusted-p and the threshold formulations must agree
        ps = [rng.randint(1, rng.choice([8, 60, 400, 2001])) / 2001 for _ in range(m)]
        holm_adj, holm_rej = multiplicity.holm(ps)
        bh_adj, bh_rej = multiplicity.benjamini_hochberg(ps)
        assert holm_rej == _holm_reference(ps, multiplicity.ALPHA), ps
        assert bh_rej == _bh_reference(ps, multiplicity.Q), ps
        for adj in (
            holm_adj,
            bh_adj,
        ):  # never below p (up to the rounding of m p / m), never above 1
            assert all(p <= a + 1e-12 and a <= 1 for p, a in zip(ps, adj, strict=True))
        assert all(b <= h + 1e-12 for b, h in zip(bh_adj, holm_adj, strict=True))  # BH is looser
        assert not any(h and not b for h, b in zip(holm_rej, bh_rej, strict=True))
        perm = rng.sample(range(m), m)  # order of the input changes nothing
        assert multiplicity.holm([ps[i] for i in perm])[0] == pytest.approx(
            [holm_adj[i] for i in perm]
        )
        assert multiplicity.benjamini_hochberg([ps[i] for i in perm])[0] == pytest.approx(
            [bh_adj[i] for i in perm]
        )
        # monotone: a larger p never gets a smaller adjusted value
        for adj in (holm_adj, bh_adj):
            by_p = sorted(zip(ps, adj, strict=True))
            assert all(a[1] <= b[1] + 1e-12 for a, b in zip(by_p, by_p[1:], strict=False))


# ----------------------------------------------------------------------- the one-sided p-value
def test_one_sided_p_is_taken_in_the_predicted_direction():
    draws = [-0.1, 0.0, 0.2, 0.3]
    assert multiplicity.one_sided_p(draws, 1) == pytest.approx((1 + 2) / 5)  # -0.1 and 0.0
    assert multiplicity.one_sided_p(draws, -1) == pytest.approx((1 + 3) / 5)  # 0.0, 0.2, 0.3
    assert multiplicity.one_sided_p([0.2, 0.3], 1) == pytest.approx(
        1 / 3
    )  # never below 1 / (1 + n)
    assert multiplicity.one_sided_p([-0.2, -0.3], -1) == pytest.approx(1 / 3)
    assert multiplicity.one_sided_p([-0.2, -0.3], 1) == 1.0
    assert multiplicity.one_sided_p([], 1) == 1.0
    with pytest.raises(ValueError, match="sign"):
        multiplicity.one_sided_p(draws, 0)
    # a mirrored sample mirrors the p-value
    mirrored = [-d for d in draws]
    assert multiplicity.one_sided_p(mirrored, -1) == multiplicity.one_sided_p(draws, 1)


def test_every_predicted_non_zero_contrast_is_predicted_positive():
    for prefix in PREDICTED_NONZERO:
        assert multiplicity.predicted_sign(prefix + "source") == 1
    for key in ("D_effect_missed_source", "D_effect_false_inval_cited", "explore_D_effect_repair"):
        with pytest.raises(ValueError, match="predicted non-zero"):
            multiplicity.predicted_sign(key)


def test_world_bootstrap_draws_are_the_draws_behind_world_bootstrap():
    rng = random.Random(3)
    by_world = {f"w{i}": [rng.random() - 0.3 for _ in range(rng.randint(1, 6))] for i in range(9)}
    est, draws = world_bootstrap_draws(by_world)
    assert est == world_bootstrap(by_world)
    assert len(draws) == DRAWS and draws == sorted(draws)
    # the interval is read off these draws
    assert est.lo == draws[int(0.025 * (len(draws) - 1))]
    assert est.hi == draws[int(0.975 * (len(draws) - 1))]
    empty, none = world_bootstrap_draws({})
    assert none == [] and math.isnan(empty.est) and empty.n_worlds == 0
    assert world_bootstrap_draws({"w": []})[1] == []


# ------------------------------------------------------------ a tiny synthetic E1 item table
ITEMS = [f"c.p{i:02d}.city" for i in range(6)]
# HoldMem(1,1) answers this many of the six items right after the shift, per world; the other
# cells answer none right (so B, D and B x D all have the same effect: the share of right items)
N_RIGHT = (6, 5, 4, 6, 5, 6, 3, 5)


def _table(n_right: tuple[int, ...] = N_RIGHT, branch: str = "src") -> list[dict]:
    """Every item changed at k = 1. B=1 cells detect and repair every item, B=0 cells neither; the
    D=1, B=1 cell's repair is right (ContractValid, and the repaired value matches gold) for the
    first `n_right[world]` items, every other cell answers wrong."""
    rows = []
    for w, right in enumerate(n_right):
        for i, fact in enumerate(ITEMS):
            for d in (0, 1):
                for b in (0, 1):
                    system = f"holdmem_D{d}B1_source" if b else f"holdmem_D{d}B0"
                    good = int(d == 1 and b == 1 and i < right)
                    rows.append(
                        {
                            "world_id": f"w{1000 + w}",
                            "branch": branch,
                            "system": system,
                            "fact_id": fact,
                            "vtype": "org",
                            "k": 1,
                            "changed": True,
                            "cv": good,
                            "cv_pre": 1,
                            "detected": bool(b),
                            "repaired_now": bool(b),
                            "repair_mech": bool(good),
                        }
                    )
    return rows


def _metrics(rows: list[dict]) -> dict:
    """What `e1.analyse` stores of the contrasts (through JSON, as the file holds them)."""
    branches = sorted({r["branch"] for r in rows})
    stored = {
        "contrasts": {b: e1.contrasts(rows, b) for b in branches},
        "n_worlds": len({r["world_id"] for r in rows}),
    }
    return dict(json.loads(json.dumps(stored)))


def _stage(path: Path, rows: list[dict]) -> Path:
    (path / "analysis").mkdir(parents=True)
    (path / "analysis" / "e1_metrics.json").write_text(json.dumps(_metrics(rows)))
    with (path / "analysis" / "e1_item_table.jsonl").open("w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    return path


def _by_key(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(r["branch"], r["key"]): r for r in rows}


def _interval_matches_p(row: dict) -> bool:
    """With 2,000 draws the 95% interval's lower bound is the 50th smallest draw, so it excludes 0
    exactly when at most 49 draws are <= 0, i.e. when p <= 50 / 2,001 (mirrored for a negative
    prediction)."""
    return bool(row["interval_excludes_zero"] == (row["p"] <= 50 / (DRAWS + 1)))


def test_contrast_rows_recompute_the_stored_estimates_and_take_the_p_value_from_the_draws():
    rows = _table()
    metrics = _metrics(rows)
    out, worst = multiplicity.contrast_rows(rows, metrics)
    got = _by_key(out)
    assert worst == 0.0
    # the seven predicted-non-zero contrasts of source granularity on the one branch; the
    # D effect on repair among changed items, and the rest of the E1 output, are not in the family
    assert sorted(k for _, k in got) == sorted(
        [
            "B_effect_detected_D0_source",
            "B_effect_detected_D1_source",
            "B_effect_update_D1_source",
            "B_effect_update_pooled_source",
            "BxD_interaction_update_source",
            "D_effect_repair_source",
            "D_effect_repair_mech_source",
        ]
    )
    for (branch, key), r in got.items():
        stored = metrics["contrasts"][branch][key]
        assert (r["est"], r["lo"], r["hi"]) == (stored["est"], stored["lo"], stored["hi"])
        assert r["n_worlds"] == 8 and r["n_items"] == stored["n_items"]
        assert r["predicted_sign"] == 1 and _interval_matches_p(r)
    share = sum(N_RIGHT) / (6 * len(N_RIGHT))  # 34 of 48 items right
    assert got["src", "B_effect_detected_D0_source"]["est"] == 1.0
    assert got["src", "B_effect_update_D1_source"]["est"] == pytest.approx(share)
    assert got["src", "B_effect_update_pooled_source"]["est"] == pytest.approx(share / 2)
    assert got["src", "BxD_interaction_update_source"]["est"] == pytest.approx(share)
    assert got["src", "D_effect_repair_source"]["est"] == pytest.approx(share)
    assert got["src", "D_effect_repair_mech_source"]["est"] == pytest.approx(share)
    # a positive effect in every world: not one of the 2,000 resampled means is <= 0, so the
    # p-value is at its floor 1 / 2,001 (the detection contrasts are constant 1: point intervals)
    assert all(r["p"] == 1 / (DRAWS + 1) and r["interval_excludes_zero"] for r in out)
    assert got["src", "B_effect_detected_D1_source"]["lo"] == 1.0
    assert (
        got["src", "B_effect_update_D1_source"]["lo"]
        < share
        < got["src", "B_effect_update_D1_source"]["hi"]
    )


def test_p_value_and_interval_agree_for_an_effect_present_in_half_of_the_worlds():
    rows = _table((6, 6, 6, 6, 0, 0, 0, 0))  # HoldMem(1,1) right everywhere in four worlds only
    out, _ = multiplicity.contrast_rows(rows, _metrics(rows))
    got = _by_key(out)
    update = got["src", "B_effect_update_D1_source"]
    assert update["est"] == pytest.approx(0.5)
    # the resampled mean is <= 0 only when all eight drawn worlds come from the null half:
    # about (1/2)^8 of the draws
    assert 0.0005 < update["p"] < 0.02
    assert update["interval_excludes_zero"] == (update["lo"] > 0) and _interval_matches_p(update)
    # detection is 1 in every world whatever the worlds do
    assert got["src", "B_effect_detected_D0_source"]["p"] == 1 / (DRAWS + 1)


def test_no_effect_and_a_reversed_effect_are_not_rejected():
    flat = _table((0,) * 8)  # nothing is ever right: every Update contrast is exactly 0
    out, _ = multiplicity.contrast_rows(flat, _metrics(flat))
    got = _by_key(out)
    for key in (
        "B_effect_update_D1_source",
        "D_effect_repair_source",
        "BxD_interaction_update_source",
    ):
        r = got["src", key]
        assert r["est"] == 0.0 and r["p"] == 1.0 and not r["interval_excludes_zero"]
    # reverse it: HoldMem(1,1) is the only cell that is wrong, everyone else right
    rows = _table((6,) * 8)
    for r in rows:
        r["cv"] = 1 - r["cv"]
        r["repair_mech"] = not r["repair_mech"]
    out, _ = multiplicity.contrast_rows(rows, _metrics(rows))
    rev = _by_key(out)["src", "D_effect_repair_source"]
    assert rev["est"] == -1.0 and rev["p"] == 1.0 and not rev["interval_excludes_zero"]
    assert (
        multiplicity.summarise(multiplicity.correct(out))["n_interval_excludes_zero_opposite"] >= 1
    )


def test_the_correction_counts_survivors_and_names_the_ones_it_removes():
    rows = _table((6, 6, 6, 6, 0, 0, 0, 0))
    out = multiplicity.correct(multiplicity.contrast_rows(rows, _metrics(rows))[0])
    s = multiplicity.summarise(out)
    assert s["n_contrasts"] == 7 and s["n_interval_excludes_zero"] == 7
    # the detection contrasts (p = 1/2001), and the rest with p in (.0005, .02): m = 7, so the
    # Holm thresholds run from .05 / 7 = .0071 up
    assert s["n_survive_holm"] == sum(r["survives_holm"] for r in out)
    assert s["n_survive_bh"] == sum(r["survives_bh"] for r in out)
    assert s["n_survive_bh"] >= s["n_survive_holm"] >= 2
    survivors = [r["holm_p"] for r in out if r["survives_holm"]]
    rest = [r["holm_p"] for r in out if not r["survives_holm"]]
    assert s["largest_holm_p_among_survivors"] == max(survivors)
    assert s["smallest_holm_p_among_nonsurvivors"] == (min(rest) if rest else None)
    assert all(a <= multiplicity.ALPHA for a in survivors) and all(
        a > multiplicity.ALPHA for a in rest
    )
    assert set(s["interval_excludes_zero_but_not_holm"]) == {
        f"{r['branch']}:{r['key']}" for r in out if not r["survives_holm"]
    }
    assert s["holm_but_interval_includes_zero"] == []


def test_a_stored_estimate_that_does_not_reproduce_stops_the_analysis():
    rows = _table()
    metrics = _metrics(rows)
    metrics["contrasts"]["src"]["D_effect_repair_source"]["hi"] += 1e-6
    with pytest.raises(ValueError, match="D_effect_repair_source"):
        multiplicity.contrast_rows(rows, metrics, "stage_x")
    metrics = _metrics(rows)
    metrics["contrasts"]["src"]["B_effect_update_D1_source"]["est"] = math.nan
    with pytest.raises(ValueError, match="B_effect_update_D1_source"):
        multiplicity.contrast_rows(rows, metrics)
    # a stored contrast of the family that the item table cannot build is an error too
    metrics = _metrics(rows)
    metrics["contrasts"]["src"]["B_effect_update_D1_cited"] = metrics["contrasts"]["src"][
        "B_effect_update_D1_source"
    ]
    with pytest.raises(ValueError, match="does not build"):
        multiplicity.contrast_rows(rows, metrics)


def test_contrasts_outside_the_family_are_left_out():
    rows = _table()
    metrics = _metrics(rows)
    assert any(k.startswith("D_effect_missed_") for k in metrics["contrasts"]["src"])
    assert any(k.startswith("D_effect_repair_changed_") for k in metrics["contrasts"]["src"])
    keys = {r["key"] for r in multiplicity.contrast_rows(rows, metrics)[0]}
    assert keys and all(k.startswith(PREDICTED_NONZERO) for k in keys)


def test_analyse_writes_the_json_and_a_stage_manifest_and_keeps_stages_apart(tmp_path):
    a = _stage(tmp_path / "stage_a", _table())
    b = _stage(tmp_path / "stage_b", _table((0,) * 8))
    res = multiplicity.analyse([a, b], tmp_path / "out")
    written = json.loads((tmp_path / "out" / "analysis" / "multiplicity.json").read_text())
    assert written == json.loads(json.dumps(res))
    assert list(written["stages"]) == ["stage_a", "stage_b"]
    # one family per stage: stage_b's nulls do not enter stage_a's correction
    assert written["stages"]["stage_a"]["summary"]["n_contrasts"] == 7
    assert written["stages"]["stage_b"]["summary"]["n_survive_holm"] == 2  # detection only
    assert written["stages"]["stage_a"]["summary"]["n_survive_holm"] == 7
    assert written["stages"]["stage_a"]["n_worlds"] == 8
    assert written["stages"]["stage_a"]["recomputed_vs_e1_metrics"]["max_abs_deviation"] == 0.0
    m = written["method"]
    assert m["bootstrap"] == {"draws": 2000, "seed": 996}
    assert m["holm"] == {"familywise_alpha": 0.05} and m["benjamini_hochberg"] == {"q": 0.05}
    assert written["status"] == multiplicity.STATUS
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["experiment"] == "merged-multiplicity"
    assert [s["run_id"] for s in manifest["sources"]] == ["stage_a", "stage_b"]
    assert "analysis/multiplicity.json" in manifest["files_sha256"]
    with pytest.raises(ValueError, match="distinct"):
        multiplicity.analyse([a, tmp_path / "x" / "stage_a"], tmp_path / "out2")


# ------------------------------------------------------------------ the committed stages
STAGES = (ROOT / "runs" / "stage2_e1", ROOT / "runs" / "stage4_rep_e1")
needs_tables = pytest.mark.skipif(
    not all((s / "analysis" / "e1_item_table.jsonl").exists() for s in STAGES),
    reason="the E1 item tables are not committed (gitignored derivatives of the raw runs)",
)


@needs_tables
def test_recomputed_estimates_equal_e1_metrics_on_the_committed_stages():
    for stage in STAGES:
        stored = json.loads((stage / "analysis" / "e1_metrics.json").read_text())["contrasts"]
        res = multiplicity.analyse_stage(stage)  # stops on any estimate or interval that differs
        want = {
            (b, k) for b, cons in stored.items() for k in cons if k.startswith(PREDICTED_NONZERO)
        }
        assert {(r["branch"], r["key"]) for r in res["contrasts"]} == want and len(want) == 42
        for r in res["contrasts"]:
            s = stored[r["branch"]][r["key"]]
            assert (r["est"], r["lo"], r["hi"]) == pytest.approx(
                (s["est"], s["lo"], s["hi"]), abs=multiplicity.MATCH_TOL
            )
            assert _interval_matches_p(r), (stage.name, r["branch"], r["key"])
        assert res["recomputed_vs_e1_metrics"]["max_abs_deviation"] <= multiplicity.MATCH_TOL


COMMITTED = ROOT / "runs" / "stage4_multiplicity" / "analysis" / "multiplicity.json"
# stage -> (contrasts, interval excludes 0 in the predicted direction, Holm, BH), as observed
COMMITTED_COUNTS = {"stage2_e1": (42, 36, 35, 36), "stage4_rep_e1": (42, 36, 36, 36)}


def test_committed_multiplicity_is_consistent_with_its_own_p_values():
    res = json.loads(COMMITTED.read_text())
    assert res["status"] == multiplicity.STATUS
    assert list(res["stages"]) == list(COMMITTED_COUNTS)
    for name, stage in res["stages"].items():
        rows = stage["contrasts"]
        assert stage["n_worlds"] == 20
        s = stage["summary"]
        assert (
            s["n_contrasts"],
            s["n_interval_excludes_zero"],
            s["n_survive_holm"],
            s["n_survive_bh"],
        ) == COMMITTED_COUNTS[name]
        assert s == multiplicity.summarise(rows)
        # three branches, each with the same fourteen predicted-non-zero contrasts (source and
        # cited granularity), and no contrast outside the family
        assert len(rows) == len({(r["branch"], r["key"]) for r in rows}) == 3 * 14
        assert all(r["key"].startswith(PREDICTED_NONZERO) for r in rows)
        # the corrections are what holm / benjamini_hochberg give for the stored p-values
        holm_p, holm_reject = multiplicity.holm([r["p"] for r in rows])
        bh_q, bh_reject = multiplicity.benjamini_hochberg([r["p"] for r in rows])
        assert [r["holm_p"] for r in rows] == holm_p
        assert [r["survives_holm"] for r in rows] == holm_reject
        assert [r["bh_q"] for r in rows] == bh_q
        assert [r["survives_bh"] for r in rows] == bh_reject
        assert all(_interval_matches_p(r) for r in rows)
        assert all(r["p"] >= 1 / (DRAWS + 1) for r in rows)
        # Holm-Bonferroni rejects nothing that Benjamini-Hochberg keeps
        assert not any(r["survives_holm"] and not r["survives_bh"] for r in rows)
        assert stage["recomputed_vs_e1_metrics"]["max_abs_deviation"] <= multiplicity.MATCH_TOL


def test_the_stage_2_contrast_that_holm_removes_is_named():
    stages = json.loads(COMMITTED.read_text())["stages"]
    assert stages["stage2_e1"]["summary"]["interval_excludes_zero_but_not_holm"] == [
        "def:BxD_interaction_update_cited"
    ]
    assert stages["stage4_rep_e1"]["summary"]["interval_excludes_zero_but_not_holm"] == []
    # the one contrast whose interval lies on the wrong side of 0 (the orphan B x D interaction at
    # cited granularity, negative in both stages) is not counted, and is not a Holm survivor
    for stage in stages.values():
        assert stage["summary"]["n_interval_excludes_zero_opposite"] == 1
        wrong = [r for r in stage["contrasts"] if r["hi"] < 0]
        assert [(r["branch"], r["key"]) for r in wrong] == [
            ("orphan", "BxD_interaction_update_cited")
        ]
        assert not wrong[0]["survives_holm"] and wrong[0]["p"] > 0.98


@needs_tables
def test_committed_multiplicity_is_what_the_analysis_writes_from_the_committed_stages(tmp_path):
    multiplicity.analyse(list(STAGES), tmp_path / "out")
    fresh = json.loads((tmp_path / "out" / "analysis" / "multiplicity.json").read_text())
    assert fresh == json.loads(COMMITTED.read_text())
