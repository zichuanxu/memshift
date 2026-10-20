import json
import math
from pathlib import Path

import pytest

from memshift.analysis import e5, rank_corr
from memshift.gen.shifts import inference_half
from memshift.scoring.stats import world_bootstrap

_IDS = [f"c.p{i:03d}.city" for i in range(80)]
_MEAS = [f for f in _IDS if not inference_half(f)]
_INFER = [f for f in _IDS if inference_half(f)]
CHANGED, UNCHANGED = _MEAS[:10], _MEAS[10:20]  # the measurement half: 10 changed, 10 unchanged
INFER_CHANGED, INFER_UNCHANGED = _INFER[:2]  # inference half: never counted, always invalid
FACTS = CHANGED + UNCHANGED + [INFER_CHANGED, INFER_UNCHANGED]
IS_CHANGED = {*CHANGED, INFER_CHANGED}

# system -> (valid changed items out of 10 after S_src, after S_def, valid unchanged after orphan)
LEVELS = {
    "holdmem_D0B0": (8, 4, 2),
    "holdmem_D1B0": (6, 6, 4),
    "holdmem_D0B1_source": (4, 8, 6),
    "holdmem_D1B1_source": (2, 2, 8),
    "mem0": (7, 5, 3),
}


def _rows(system: str, world: str, levels: tuple[int, int | None, int | None]) -> list[dict]:
    """Item-table rows of one system in one world; a None level: the branch was not measured.
    The branch's metric items (changed after S_src and S_def, unchanged after the orphan branch)
    are valid in the first `level` of 10; the other class is constant, so counting it by mistake
    would move the metric."""
    rows = []
    for branch, level, on_changed in zip(
        ("src", "def", "orphan"), levels, (True, True, False), strict=True
    ):
        if level is None:
            continue
        for fact in FACTS:
            changed = fact in IS_CHANGED
            if fact in (INFER_CHANGED, INFER_UNCHANGED):
                valid = False
            elif changed == on_changed:
                valid = (CHANGED if changed else UNCHANGED).index(fact) < level
            else:
                valid = not on_changed
            rows.append(
                {
                    "world_id": world,
                    "branch": branch,
                    "system": system,
                    "fact_id": fact,
                    "k": 1,
                    "changed": changed,
                    "cv": valid,
                }
            )
    return rows


def _table(worlds: dict[str, dict[str, tuple]]) -> list[dict]:
    return [r for w, spec in worlds.items() for s, lv in spec.items() for r in _rows(s, w, lv)]


WORLDS = {"w1000", "w1001", "w1002"}


def test_average_ranks_and_spearman_by_hand():
    assert rank_corr.average_ranks([3, 1, 2, 2]) == [4, 1, 2.5, 2.5]
    assert rank_corr.spearman([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(0.8)  # 1 - 6 * 2 / 60
    assert rank_corr.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # ties: ranks (1, 2.5, 2.5, 4) and (1, 3, 2, 4): 4.5 / sqrt(4.5 * 5)
    assert rank_corr.spearman([1, 2, 2, 4], [1, 3, 2, 4]) == pytest.approx(4.5 / math.sqrt(22.5))
    assert math.isnan(rank_corr.spearman([1, 2], [2, 1]))  # fewer than 3 systems
    assert math.isnan(rank_corr.spearman([1, 1, 1], [1, 2, 3]))  # a constant ranking


def test_metrics_are_the_e5_definitions():
    rows = _table({w: {"mem0": LEVELS["mem0"]} for w in sorted(WORLDS)})
    samples = rank_corr.measure(rows, WORLDS)["mem0"]
    for name, (branch, metric) in rank_corr.METRICS.items():
        assert samples[name] == e5.carry_samples(rows, "mem0", branch, WORLDS)[metric]
        assert (
            world_bootstrap(samples[name]).to_json()
            == e5._carry(rows, "mem0", branch, WORLDS)[metric]
        )
    # 7 of 10 changed items valid after S_src, 5 after S_def; 3 of 10 unchanged valid after orphan;
    # the inference-half item and the branch's other items do not count
    assert {
        n: sum(map(sum, s.values())) / sum(map(len, s.values())) for n, s in samples.items()
    } == {
        "update_src": 0.7,
        "update_def": 0.5,
        "stability_orphan": 0.3,
    }
    assert all(len(v) == 10 for by in samples.values() for v in by.values())


def test_known_rank_correlations_and_a_collapsed_bootstrap():
    rows = _table({w: LEVELS for w in sorted(WORLDS)})
    samples = rank_corr.measure(rows, WORLDS)
    # S_src ranks (D, C, B, mem0, A) = 1..5; S_def ranks: D 1, A 2, mem0 3, B 4, C 5; sum d^2 = 20
    # -> rho = 1 - 6 * 20 / 120 = 0; orphan Stability ranks: A 1, mem0 2, B 3, C 4, D 5;
    # sum d^2 = 40 -> rho = -1
    corr = rank_corr.correlations(samples)
    a, b = corr["update_src_vs_update_def"], corr["update_src_vs_stability_orphan"]
    assert a["rho"] == pytest.approx(0.0, abs=1e-12) and b["rho"] == pytest.approx(-1.0)
    # identical worlds: every world resample has the same metrics, so the interval is the point
    for c in (a, b):
        assert c["lo"] == pytest.approx(c["rho"]) and c["hi"] == pytest.approx(c["rho"])
        assert c["n_systems"] == 5 and c["n_worlds"] == 3 and c["left_out"] == []


def test_ties_in_pooled_metrics_and_a_real_interval():
    # two worlds like LEVELS, two worlds where S_def orders the systems like S_src does
    same_as_src = {s: (lv[0], lv[0], lv[2]) for s, lv in LEVELS.items()}
    rows = _table({"w1000": LEVELS, "w1001": LEVELS, "w1002": same_as_src, "w1003": same_as_src})
    worlds = WORLDS | {"w1003"}
    samples = rank_corr.measure(rows, worlds)
    # pooled S_def: A (4 + 8) / 2 = 6, B 6, C (8 + 4) / 2 = 6, mem0 (5 + 7) / 2 = 6, D 2: four ties.
    # ranks vs S_src (A 5, B 3, C 2, D 1, mem0 4): x = (5, 3, 2, 1, 4), y = (3.5, 3.5, 3.5, 1, 3.5);
    # cov 5, var x 10, var y 5 -> 5 / sqrt(50)
    c = rank_corr.correlations(samples)["update_src_vs_update_def"]
    assert c["rho"] == pytest.approx(5 / math.sqrt(50))
    assert -1.0 <= c["lo"] < c["rho"] < c["hi"] <= 1.0  # worlds disagree: a real interval
    # the bootstrap is deterministic
    assert rank_corr.correlations(samples) == rank_corr.correlations(samples)


def test_system_missing_a_branch_is_tabled_but_not_correlated():
    spec = {**LEVELS, "oracle": (10, None, None)}  # baselines were run on the src branch only
    rows = _table({w: spec for w in sorted(WORLDS)})
    samples = rank_corr.measure(rows, WORLDS)
    corr = rank_corr.correlations(samples)
    assert corr["update_src_vs_update_def"]["left_out"] == ["oracle"]
    assert corr["update_src_vs_update_def"]["rho"] == pytest.approx(0.0, abs=1e-12)
    rows_ = {r["system"]: r for r in rank_corr.system_rows(samples, {"mem0"})}
    assert rows_["oracle"]["kind"] == "baseline" and rows_["mem0"]["kind"] == "native"
    assert rows_["holdmem_D0B0"]["kind"] == "white-box"
    assert rows_["oracle"]["update_def"] is None and rows_["oracle"]["rank_update_def"] is None
    # oracle 10, A 8, mem0 7, B 6, C 4, D 2 on S_src (rank 1 = highest)
    assert [
        rows_[s]["rank_update_src"] for s in ("oracle", "holdmem_D0B0", "mem0", "holdmem_D1B0")
    ] == [1, 2, 3, 4]
    assert rows_["oracle"]["update_src"]["est"] == 1.0


def test_json_arm_is_excluded():
    spec = {**LEVELS, "holdmem_json_arm": (9, 9, 9)}
    samples = rank_corr.measure(_table({w: spec for w in sorted(WORLDS)}), WORLDS)
    assert "holdmem_json_arm" not in samples and set(samples) == set(LEVELS)


# ------------------------------------------------------------------------------- raw run files
def _write(run: Path, probes: list[dict], answers: list[dict]) -> Path:
    run.mkdir(parents=True)
    for name, rows in (("probes", probes), ("answers", answers)):
        (run / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return run


def _probes(worlds: list[str]) -> list[dict]:
    return [
        {
            "world_id": w,
            "branch": branch,
            "fact_id": fact,
            "vtype": "text",
            "gold_status": "answer",
            "gold_value": "new" if k and fact in IS_CHANGED else "old",
            "as_of": "t",
            "tags": {"k": k},
        }
        for w in worlds
        for branch in ("src", "def", "orphan")
        for fact in FACTS
        for k in (0, 1)
    ]


def _answers(rows: list[dict]) -> list[dict]:
    return [
        {
            "system": r["system"],
            "world_id": r["world_id"],
            "branch": r["branch"],
            "fact_id": r["fact_id"],
            "tags": {"k": r["k"]},
            "contract_valid": r["cv"],
        }
        for r in rows
    ]


def test_analyse_merges_native_white_box_and_baseline_runs(tmp_path):
    worlds = ["w1000", "w1001", "w1002", "w1020"]  # w1020 lies outside the short worlds
    white = {s: lv for s, lv in LEVELS.items() if s != "mem0"} | {"holdmem_json_arm": (9, 9, 9)}
    wb = _write(
        tmp_path / "wb",
        _probes(worlds),
        _answers(_table({w: white for w in worlds})),
    )
    nat = _write(
        tmp_path / "nat", [], _answers(_table({w: {"mem0": LEVELS["mem0"]} for w in worlds}))
    )
    base = _write(
        tmp_path / "base",
        _probes(worlds),
        _answers(_table({w: {"oracle": (10, None, None)} for w in worlds})),
    )
    res = rank_corr.analyse([nat], [wb], tmp_path / "out", [base], seeds=[1000, 1001, 1002])
    assert res["status"] == rank_corr.STATUS
    assert res["worlds"] == ["w1000", "w1001", "w1002"]
    by = {r["system"]: r for r in res["systems"]}
    assert set(by) == {*LEVELS, "oracle"}
    assert {by[s]["kind"] for s in ("mem0", "oracle", "holdmem_D0B0")} == {
        "native",
        "baseline",
        "white-box",
    }
    assert by["mem0"]["update_src"]["est"] == pytest.approx(0.7)
    assert by["mem0"]["update_src"]["n_worlds"] == 3
    a = res["correlations"]["update_src_vs_update_def"]
    assert a["rho"] == pytest.approx(0.0, abs=1e-12) and a["left_out"] == ["oracle"]
    assert res["correlations"]["update_src_vs_stability_orphan"]["rho"] == pytest.approx(-1.0)
    on_disk = json.loads((tmp_path / "out" / "analysis" / "rank_corr.json").read_text())
    assert on_disk["correlations"] == res["correlations"]
    text = (tmp_path / "out" / "analysis" / "rank_corr.md").read_text()
    assert rank_corr.STATUS in text and "| mem0 | native | 70.0" in text and "Spearman rho" in text
    # a narrower world set uses only those worlds
    two = rank_corr.analyse([nat], [wb], tmp_path / "out2", [base], seeds=[1000, 1001])
    assert two["worlds"] == ["w1000", "w1001"]
    assert two["correlations"]["update_src_vs_update_def"]["n_worlds"] == 2
