"""The verdicts of the round-3 registration (`analysis.round3_registered`, prereg/v4_round3.md):
the ratio and threshold logic, the readings, the not-judged rules, the two-way bootstrap, and the
pairing checks that refuse a judgement, on small hand-built runs (no model, no committed run)."""

import json
import math
from collections.abc import Callable, Collection
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from memshift.analysis import controls
from memshift.analysis import ex_e1_classes as ex
from memshift.analysis import round3_registered as rr
from memshift.experiments.common import MAIN_MODEL
from memshift.llm import ResponseCache
from memshift.scoring.stats import item_diffs

WORLDS = ("w1000", "w1001", "w1002", "w1003")
BRANCHES = ("src", "def", "orphan")
FACTS = ("k.alpha", "k.beta", "c.p1.city", "t.t001")  # t.t001 never changes
KS = (0, 1, 2)
CV = Callable[[str, str, int], bool]  # (world, fact, k) -> ContractValid


# -------------------------------------------------------------------------------- fixtures
def _probe(w: str, b: str, f: str, k: int, unnamed: bool = False) -> dict[str, Any]:
    moved = k > 0 and f != "t.t001"
    named = f.startswith("k.") and not unnamed
    return {
        "probe_id": f"{w}.{b}.{f}.{k}",
        "world_id": w,
        "branch": b,
        "fact_id": f,
        "vtype": "str",
        "as_of": f"2026-10-{k + 1:02d}T00:00",
        "question_text": f"As of k={k}: {f}?" + (" Definition in effect: v1." if named else ""),
        "gold_value": f"v{k}" if moved else "v0",
        "gold_status": "answer",
        "tags": {"k": k},
        **({"asked": True} if k else {}),
    }


def _answer(system: str, p: dict[str, Any], cv: bool) -> dict[str, Any]:
    keys = ("probe_id", "world_id", "branch", "fact_id", "tags", "as_of")
    return {
        "system": system,
        **{k: p[k] for k in keys},
        "status": "answer",
        "value": p["gold_value"] if cv else "wrong",
        "contract_valid": cv,
        "context_sha256": f"{system}|{p['probe_id']}",  # the store the reader read
    }


def write_run(
    d: Path,
    cells: dict[str, CV],
    created: str,
    worlds: tuple[str, ...] = WORLDS,
    params: dict[str, Any] | None = None,
    skip: Collection[str] = (),
    manifest: bool = True,
    reader: str = MAIN_MODEL,
    definition_only: bool = False,
) -> Path:
    """A run directory: probes of every (world, branch, fact, k), the cells' answers, passing
    hash checks and (unless `manifest` is False) a manifest. `skip` leaves probe ids unanswered;
    `definition_only` answers only the definition items after the fork (`--post-probes
    definition`) and records the other items after it as not asked."""
    d.mkdir(parents=True)
    unnamed = bool((params or {}).get("probe_unnamed_defs"))
    probes = [
        _probe(w, b, f, k, unnamed) for w in worlds for b in BRANCHES for f in FACTS for k in KS
    ]
    if definition_only:
        for p in probes:
            if p["tags"]["k"] and not p["fact_id"].startswith("k."):
                p["asked"] = False
    with (d / "probes.jsonl").open("w") as fh:
        for p in probes:
            fh.write(json.dumps(p) + "\n")
    with (d / "answers.jsonl").open("w") as fh:
        for system, cv in cells.items():
            for p in probes:
                if p["probe_id"] not in skip and p.get("asked", True):
                    a = _answer(system, p, cv(p["world_id"], p["fact_id"], p["tags"]["k"]))
                    fh.write(json.dumps(a) + "\n")
    for w in worlds:
        for b in BRANCHES:
            (d / f"hash_check_{w}_{b}.json").write_text(json.dumps({"ok": True}))
    if manifest:
        man = {
            "experiment": "E1",
            "created_utc": created,
            "models": {"writer": {"model": MAIN_MODEL}, "reader": {"model": reader}},
            "params": {
                "cells": list(cells),
                "branches": list(BRANCHES),
                "world_params": {},
                "probes_after_fork": "discriminative",
                **({"post_probes": "definition"} if definition_only else {}),
                **(params or {}),
            },
        }
        (d / "manifest.json").write_text(json.dumps(man))
    return d


def const(x: bool) -> CV:
    return lambda w, f, k: x


def on(*facts: str, k: int | None = None) -> CV:
    """True on the listed facts (at checkpoint k only, when given)."""
    return lambda w, f, kk: f in facts and (k is None or kk == k)


# the committed cells: at D=1 B0 knows one of the four changed definition item-times per world
COMMITTED: dict[str, CV] = {
    "holdmem_D0B0": const(False),
    "holdmem_D0B1_source": const(False),
    "holdmem_D1B0": on("k.alpha", k=1),
    "holdmem_D1B1_source": const(True),
    "holdmem_D1sB0": const(False),
    "holdmem_D1sB1_source": const(True),
}
U_NEW: dict[str, CV] = {
    "holdmem_D0B0": COMMITTED["holdmem_D0B0"],  # the anchor
    "holdmem_D0B0_refresh": const(False),
    "holdmem_D1B0_refresh": const(True),  # R = 1
    "holdmem_D1sB0_refresh": on("k.alpha"),  # R = 0.5 on definition items
    "holdmem_D1B0_upsert": const(True),
    "holdmem_D1sB0_upsert": const(False),
}


REAL_WORLDS = dict(rr.REGISTERED_WORLDS)


@pytest.fixture(autouse=True)
def fixture_worlds(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fixture runs cover four worlds: register those for every prediction."""
    monkeypatch.setattr(rr, "REGISTERED_WORLDS", {k: WORLDS for k in REAL_WORLDS})


@pytest.fixture
def committed(tmp_path: Path) -> Path:
    return write_run(tmp_path / "e1_test_s1000-1003_20260928T202242Z", COMMITTED, "2026-09-29")


# ---------------------------------------------------------------------------- ratio and R
def _vec(values: dict[str, list[float]]) -> ex.Vec:
    return {w: {f"i{i}": x for i, x in enumerate(v)} for w, v in values.items()}


def test_ratio_is_the_ratio_of_pooled_means_and_matches_controls():
    x = _vec({"w1": [1, 1, 0, 1], "w2": [1, 0, 0, 0], "w3": [1, 1, 1, 0]})
    b0 = _vec({"w1": [0, 1, 0, 0], "w2": [0, 0, 0, 0], "w3": [0, 1, 0, 0]})
    b1 = _vec({"w1": [1, 1, 1, 1], "w2": [1, 1, 0, 0], "w3": [1, 1, 1, 1]})
    q = rr.ratio_quantity(x, b0, b1, draws=400)
    # (mean x - mean b0) / (mean b1 - mean b0) over the same 12 item-times
    assert q["R"]["est"] == pytest.approx((7 / 12 - 2 / 12) / (10 / 12 - 2 / 12))
    assert q["evaluated"] and q["n_items"] == 12 and q["n_worlds"] == 3
    ref = controls.ratio_estimate(item_diffs(x, b0), item_diffs(b1, b0), draws=400)
    for key in ("est", "lo", "hi"):
        assert q["R"][key] == pytest.approx(getattr(ref, key), abs=1e-12)
    assert q["update"]["x"]["est"] == pytest.approx(7 / 12)


def test_ratio_uses_only_item_times_all_three_cells_answered():
    x = _vec({"w1": [1, 1], "w2": [1, 1]})
    b0 = _vec({"w1": [0, 0], "w2": [0, 0]})
    b1 = {"w1": {"i0": 1.0}}  # one item-time of one world
    q = rr.ratio_quantity(x, b0, b1, draws=50)
    assert q["n_items"] == 1 and q["n_worlds"] == 1 and q["R"]["est"] == 1.0


def test_r_is_not_evaluated_below_five_pp():
    x = _vec({"w1": [1] + [0] * 99})
    b0 = _vec({"w1": [0] * 100})
    for hits, evaluated in ((4, False), (5, True)):
        b1 = _vec({"w1": [1] * hits + [0] * (100 - hits)})
        q = rr.ratio_quantity(x, b0, b1, draws=50)
        assert q["evaluated"] is evaluated
        assert (q["R"] is None) is (not evaluated)
    assert "< 5 pp" in rr.ratio_quantity(x, b0, _vec({"w1": [0] * 100}), draws=50)["reason"]


# ------------------------------------------------------------------------------- U verdicts
def _r(value: float | None) -> dict[str, Any]:
    if value is None:
        return {"evaluated": False, "R": None, "reason": "denominator +1.0 pp < 5 pp"}
    return {"evaluated": True, "R": {"est": value, "lo": value, "hi": value, "n_worlds": 4}}


def _main(
    r: dict[tuple[str, str], float | None],
    upsert: float | None = 0.9,
    combinations: tuple[str, ...] = rr.U3_COMBINATIONS,
) -> dict[str, Any]:
    node: dict[str, Any] = {}
    for (arm, b), v in r.items():
        node.setdefault(b, {}).setdefault(arm, {})["definition"] = _r(v)
    pooled = {"upsert": {**_r(upsert), "combinations": list(combinations)}}
    return {
        "R": {"refresh": node},
        "pooled_definition_D1_Dm_src_def": pooled if upsert != -1 else {},
        "U4": None,
    }


def four(*values: float | None) -> dict[tuple[str, str], float | None]:
    keys = [(a, b) for a in ("D1", "Dm") for b in ("src", "def")]
    return dict(zip(keys, values, strict=True))


@pytest.mark.parametrize(
    "values,status",
    [
        ((1.0, 0.99, 0.93, 0.80), rr.HOLDS),  # 0.80 is inclusive
        ((1.0, 0.99, 0.93, 0.79), rr.HOLDS),  # 3 of 4
        ((1.0, 0.99, 0.79, 0.79), rr.FAILS),
        ((1.0, 0.99, 0.93, None), rr.HOLDS),  # 3 true, one not evaluated
        ((1.0, 0.99, None, 0.5), rr.NOT_JUDGED),  # 2 true, the unevaluated one decides
        ((1.0, None, 0.5, 0.5), rr.FAILS),  # at most 2 can be true
    ],
)
def test_u1_three_of_four(values, status):
    assert rr.u_predictions(_main(four(*values)), {})["U1"]["verdict"] == status


@pytest.mark.parametrize(
    "r,reading",
    [
        (0.80, "executability"),
        (0.95, "executability"),
        (0.50, "key_schedule"),
        (0.2, "key_schedule"),
        (0.65, "both"),
    ],
)
def test_u3_readings(r, reading):
    u3 = rr.u_predictions(_main(four(1, 1, 1, 1), upsert=r), {})["U3"]
    assert u3["verdict"] == rr.JUDGED and u3["reading"] == reading


def test_u3_and_u5_not_judged_rules():
    preds = rr.u_predictions(_main(four(1, 1, 1, 1), upsert=None), {})
    assert preds["U3"]["verdict"] == rr.NOT_JUDGED and preds["U3"]["reading"] is None
    assert preds["U5"]["verdict"] == rr.NOT_JUDGED  # no M2 family, and no U3 reading
    no_cells = rr.u_predictions(_main(four(1, 1, 1, 1), upsert=-1), {})
    assert "no upsert cell" in no_cells["U3"]["reason"]
    assert rr.u_predictions(None, {})["U1"]["verdict"] == rr.NOT_JUDGED
    # the main study pools all four (arm, shift) combinations; an M2 family the two D=1 ones
    d1_only = _main(four(1, 1, 1, 1), upsert=0.9, combinations=rr.U5_COMBINATIONS)
    u3 = rr.u_predictions(d1_only, {})["U3"]
    assert u3["verdict"] == rr.NOT_JUDGED and "Dm/src, Dm/def" in u3["reason"]
    u5 = rr.u_predictions(_main(four(1, 1, 1, 1)), {"fam": d1_only})["U5"]
    assert u5["families"]["fam"]["reading"] == "executability"


@pytest.mark.parametrize(
    "fam_r,status",
    [
        ((0.9, 0.85, 0.3), rr.HOLDS),  # two of three read executability, as the main study
        ((0.9, 0.3, 0.6), rr.FAILS),
        ((0.9, None, 0.3), rr.NOT_JUDGED),  # the unevaluated family decides
        ((0.9, 0.85), rr.HOLDS),
        ((0.9,), rr.NOT_JUDGED),  # two families missing: undecidable
    ],
)
def test_u5_same_reading_in_two_of_three_families(fam_r, status):
    m2 = {f"fam{i}": _main(four(1, 1, 1, 1), upsert=r) for i, r in enumerate(fam_r)}
    u5 = rr.u_predictions(_main(four(1, 1, 1, 1), upsert=0.9), m2)["U5"]
    assert u5["verdict"] == status and u5["main_reading"] == "executability"


# ------------------------------------------------------------------- U end to end, pairing
def _u_runs(tmp_path: Path, new: dict[str, CV] | None = None, **kw: Any) -> list[Path]:
    halves = (WORLDS[:2], WORLDS[2:])
    return [
        write_run(tmp_path / f"e1_u_s{i}_20261016T060737Z", new or U_NEW, "2026-10-16", ws, **kw)
        for i, ws in enumerate(halves)
    ]


def test_block_u_end_to_end(tmp_path, committed):
    res = rr.block_u(_u_runs(tmp_path), [committed], draws=200)
    assert res["pairing"]["ok"], res["pairing"]["counts"]
    assert res["pairing"]["anchor_answers_compared"] == len(WORLDS) * 3 * len(FACTS) * len(KS)
    main = res["quantities"]["families"]["main"]
    d1 = main["R"]["refresh"]["src"]["D1"]["definition"]
    assert d1["update"]["B0"]["est"] == pytest.approx(0.25)  # 1 of 4 changed item-times
    assert d1["R"]["est"] == pytest.approx(1.0)
    assert main["R"]["refresh"]["def"]["Dm"]["definition"]["R"]["est"] == pytest.approx(0.5)
    d0 = main["R"]["refresh"]["src"]["D0"]["definition"]
    assert not d0["evaluated"] and d0["R"] is None  # B1 - B0 = 0
    preds = res["predictions"]
    assert preds["U1"]["verdict"] == rr.FAILS  # D1 1.0 twice, D_m 0.5 twice
    # U3 pooled: D=1 (0.75 / 0.75 per item) and D_m (0 / 1) over 16 item-times each
    pooled = main["pooled_definition_D1_Dm_src_def"]["upsert"]
    assert pooled["R"]["est"] == pytest.approx(24 / 56)
    assert preds["U3"]["reading"] == "key_schedule"
    assert preds["U4"]["verdict"] == rr.HOLDS
    assert res["reported"]["U2_R_refresh_D0_definition"]["src"]["R"] is None
    assert json.loads(json.dumps(res, default=str))  # JSON-serialisable
    assert set(res["notes"]) == {"U1", "U3", "U4", "U5"}


def test_missing_registered_worlds_are_not_judged(tmp_path, committed, monkeypatch):
    monkeypatch.setattr(rr, "REGISTERED_WORLDS", REAL_WORLDS)
    assert REAL_WORLDS["U"] == tuple(f"w{s}" for s in range(1000, 1020))
    assert REAL_WORLDS["H"] == REAL_WORLDS["G-a"] == REAL_WORLDS["X"] == REAL_WORLDS["U5"]
    assert REAL_WORLDS["Q"] == REAL_WORLDS["G-b"] == REAL_WORLDS["U"]
    res = rr.block_u(_u_runs(tmp_path), [committed], draws=50)
    u1 = res["predictions"]["U1"]
    assert u1["verdict"] == rr.NOT_JUDGED and u1["reason"].startswith("4 of 20 registered worlds")
    assert u1["interim_on_worlds_present"]["verdict"] == rr.FAILS
    u3 = res["predictions"]["U3"]
    assert u3["reading"] is None and u3["interim_on_worlds_present"]["reading"] == "key_schedule"
    assert res["predictions"]["U5"]["verdict"] == rr.NOT_JUDGED
    assert res["quantities"]["families"]["main"]["worlds"]["refresh"] == list(WORLDS)


def test_anchor_mismatch_refuses_every_verdict(tmp_path, committed):
    flipped = {**U_NEW, "holdmem_D0B0": on("c.p1.city", k=2)}  # differs from the committed D0B0
    res = rr.block_u(_u_runs(tmp_path, flipped), [committed], draws=50)
    counts = res["pairing"]["counts"]
    assert not res["pairing"]["ok"] and counts["anchor_answers_differ"] == 4 * 3
    for p in res["predictions"].values():
        assert p["verdict"] == rr.NOT_JUDGED and "pairing check failed" in p["reason"]
    assert res["predictions"]["U3"]["reading"] is None
    assert res["quantities"]["families"]["main"]["R"]["refresh"]  # still reported


def test_probe_id_mismatch_refuses(tmp_path, committed):
    res = rr.block_u(_u_runs(tmp_path, skip={"w1000.src.k.alpha.1"}), [committed], draws=50)
    counts = res["pairing"]["counts"]
    assert counts["probe_keys_missing"] == len(U_NEW)  # every cell of the run lacks it
    assert counts["anchor_answers_missing"] == 1
    assert res["predictions"]["U1"]["verdict"] == rr.NOT_JUDGED


def test_gold_mismatch_and_missing_reference_refuse(tmp_path, committed):
    runs = _u_runs(tmp_path)
    probes = [json.loads(x) for x in (runs[0] / "probes.jsonl").read_text().splitlines()]
    probes[5]["gold_value"] = "other"
    (runs[0] / "probes.jsonl").write_text("".join(json.dumps(p) + "\n" for p in probes))
    res = rr.block_u(runs, [committed], draws=50)
    assert res["pairing"]["counts"]["gold_differs"] == 1
    lone = write_run(tmp_path / "e1_u_s9_20261016T060737Z", U_NEW, "2026-10-16", ("w1009",))
    res = rr.block_u([lone], [committed], draws=50)
    assert res["pairing"]["counts"]["no_committed_reference"] == 3  # three branches
    assert res["predictions"]["U1"]["verdict"] == rr.NOT_JUDGED


def test_hash_failure_refuses(tmp_path, committed):
    runs = _u_runs(tmp_path)
    (runs[1] / "hash_check_w1002_def.json").write_text(json.dumps({"ok": False}))
    (runs[1] / "hash_check_w1003_src.json").unlink()
    counts = rr.block_u(runs, [committed], draws=50)["pairing"]["counts"]
    assert counts["hash_failed"] == 1 and counts["hash_missing"] == 1


def test_incomplete_runs_are_skipped_and_the_newest_run_wins(tmp_path, committed):
    runs = _u_runs(tmp_path)
    # an older complete run of the same worlds whose refresh cell differs, and a run in progress
    older = write_run(
        tmp_path / "e1_u_old_20261015T000000Z",
        {**U_NEW, "holdmem_D1B0_refresh": const(False)},
        "2026-10-15",
    )
    partial = write_run(tmp_path / "e1_u_partial", U_NEW, "", manifest=False)
    res = rr.block_u([older, *runs, partial], [committed], draws=50)
    assert res["pairing"]["detail"]["incomplete_runs_skipped"] == ["e1_u_partial"]
    sup = [s for s in res["pairing"]["detail"]["superseded"] if s["run"] == older.name]
    assert sup and all(s["kept"] in {r.name for r in runs} for s in sup)
    # the older run's answers differ in one cell: the merge is not safe
    assert res["pairing"]["counts"]["superseded_answers_differ"] > 0
    assert res["predictions"]["U1"]["verdict"] == rr.NOT_JUDGED
    same = write_run(tmp_path / "e1_u_same_20261015T000000Z", U_NEW, "2026-10-15")
    res = rr.block_u([same, *runs], [committed], draws=50)
    assert res["pairing"]["ok"]
    d1 = res["quantities"]["families"]["main"]["R"]["refresh"]["src"]["D1"]["definition"]
    assert d1["R"]["est"] == pytest.approx(1.0)


def test_cli_writes_json_and_a_summary(tmp_path, committed, capsys):
    out = tmp_path / "u.json"
    runs = [str(p) for p in _u_runs(tmp_path)]
    rr.main(["U", "--runs", *runs, "--compare", str(committed), "--draws", "50", "--out", str(out)])
    res = json.loads(out.read_text())
    assert res["block"] == "U" and res["predictions"]["U1"]["verdict"] == rr.FAILS
    text = capsys.readouterr().out
    assert "block U: pairing ok" in text and "U1: fails" in text and "R 1.00" in text


# --------------------------------------------------------------------------------- block Q
def test_block_q_unnamed_probes(tmp_path: Path, committed: Path) -> None:
    q_cells: dict[str, CV] = {
        **COMMITTED,
        # answers to the definition items change with the question; they are not anchors
        "holdmem_D0B0": lambda w, f, k: f == "k.beta",
        "holdmem_D1B0": const(False),  # B effect at D=1: 1.0 unnamed against 0.75 named
    }
    run = write_run(
        tmp_path / "e1_q_20261016T000000Z",
        q_cells,
        "2026-10-16",
        params={"probe_unnamed_defs": True},
    )
    res = rr.block_q([run], [committed], draws=200)
    assert res["pairing"]["ok"], res["pairing"]["counts"]
    q1 = res["quantities"]["Q1"]
    assert q1["B_effect_D1_unnamed"]["est"] == pytest.approx(1.0)
    assert q1["B_effect_D1_named"]["est"] == pytest.approx(0.75)
    assert q1["difference"]["est"] == pytest.approx(0.25)
    assert res["predictions"]["Q1"]["verdict"] == rr.HOLDS
    # B x D unnamed per item: (1 - 0) - (0 - D0B0), D0B0 right on k.beta only: mean 1.5
    assert res["quantities"]["Q2"]["D1"]["unnamed"]["est"] == pytest.approx(1.5)
    assert res["predictions"]["Q2"]["verdict"] == rr.HOLDS


# --------------------------------------------------------------------------- two-way bootstrap
def _sums(values: dict[tuple[int, int], list[float]], nw: int, nc: int):
    s, n = np.zeros((nw, nc)), np.zeros((nw, nc))
    for (w, c), v in values.items():
        s[w, c], n[w, c] = sum(v), len(v)
    return s, n


def test_two_way_with_one_cluster_is_the_world_bootstrap():
    rng = np.random.default_rng(0)
    groups = {f"w{i}": list(rng.integers(0, 2, size=5 + i).astype(float)) for i in range(7)}
    s = np.array([[sum(v)] for v in groups.values()])
    n = np.array([[len(v)] for v in groups.values()])
    two = rr.two_way(s, n, draws=500)
    ref = ex.mean_estimate(groups, draws=500)
    for key in ("est", "lo", "hi", "n_worlds", "n_items"):
        assert two[key] == pytest.approx(ref[key], abs=1e-12)
    assert two["n_clusters"] == 1
    one = rr.one_way(s, n, draws=500)
    assert (one["lo"], one["hi"]) == (two["lo"], two["hi"])


def test_two_way_shape_drops_empty_and_widens_with_cluster_spread():
    # 6 worlds x 4 clusters; clusters differ a lot, worlds do not; one empty cluster, one world
    vals = {(w, c): [float(c % 2)] * 3 for w in range(6) for c in range(4)}
    s, n = _sums(vals, 7, 5)
    two = rr.two_way(s, n, draws=500)
    assert set(two) == {"est", "lo", "hi", "width", "n_worlds", "n_items", "n_clusters"}
    assert (two["n_worlds"], two["n_clusters"], two["n_items"]) == (6, 4, 72)
    assert two["est"] == pytest.approx(0.5)
    one = rr.one_way(s, n, draws=500)
    assert one["width"] == pytest.approx(0.0)  # every world has the same mean
    assert two["width"] > 0.3  # resampling the clusters moves the mean
    again = rr.two_way(s, n, draws=500)
    assert again == two  # seeded
    assert math.isnan(rr.two_way(np.zeros((2, 2)), np.zeros((2, 2)))["est"])


def test_cluster_arrays_from_vectors():
    parts = {
        "a": {"w1": {"x@1": 1.0, "y@1": 0.0}, "w2": {"x@1": 1.0}},
        "b": {"w2": {"x@2": -1.0}},
        "empty": {"w3": {}},
    }
    worlds, clusters, s, n = rr.cluster_arrays(parts)
    assert worlds == ["w1", "w2"] and clusters == ["a", "b"]
    assert s.tolist() == [[1.0, 0.0], [1.0, -1.0]] and n.tolist() == [[2, 0], [1, 1]]


def test_writer_variance_share():
    writer = {f"{w}#{i}": w for w in ("A", "B") for i in range(3)}
    split = {s: (1.0 if w == "A" else 0.0) for s, w in writer.items()}
    out = rr._writer_share(split, writer)
    assert out["eta_squared"] == pytest.approx(1.0)
    assert out["variance_component_share"] == pytest.approx(1.0)
    mixed = {"A#0": 1.0, "A#1": 0.0, "A#2": 0.5, "B#0": 1.0, "B#1": 0.0, "B#2": 0.5}
    assert rr._writer_share(mixed, writer)["eta_squared"] == pytest.approx(0.0)
    assert rr._writer_share({"A#0": 1.0}, {"A#0": "A"})["eta_squared"] is None


def _table(cells: dict[str, CV], worlds: tuple[str, ...] = WORLDS) -> ex.Table:
    rows = [
        {
            "world_id": w,
            "branch": b,
            "system": s,
            "fact_id": f,
            "k": k,
            "changed": f != "t.t001",
            "cv": float(cv(w, f, k)),
        }
        for s, cv in cells.items()
        for w in worlds
        for b in BRANCHES
        for f in FACTS
        for k in (1, 2)
    ]
    return ex.Table(rows)


def test_g_programs_two_way_over_sets_and_per_writer():
    tc = _table({"holdmem_D0B0": const(False), "holdmem_D0B1_source": const(False)})
    sets = {}
    for writer, good in (("A", True), ("B", False)):
        for i in range(3):
            cells = {"holdmem_D1sB0": const(False), "holdmem_D1sB1_source": const(good)}
            sets[f"{writer}#s{i}"] = (writer, _table(cells))
    out = rr.g_programs(sets, tc, draws=300)
    node = out["branches"]["def"]["definition"]
    assert node["two_way"]["n_clusters"] == 6 and node["two_way"]["est"] == pytest.approx(0.5)
    assert node["per_writer"]["A"]["est"] == pytest.approx(1.0)
    assert node["per_writer"]["B"]["est"] == pytest.approx(0.0)
    assert out["writer_variance_share"]["eta_squared"] == pytest.approx(1.0)
    assert node["two_way"]["lo"] < node["two_way"]["est"] < node["two_way"]["hi"]


def test_g_heldout_clusters_are_draw_definitions():
    cells = {
        **{c: const(False) for c in rr.ARM_CELLS},
        "holdmem_D1B1_source": const(True),
        "holdmem_D1sB1_source": on("k.alpha"),
    }
    tables = {f"heldout:{s}": _table(cells) for s in (4001, 4002, 4003)}
    out = rr.g_heldout(tables, draws=200)
    d1 = out["def"]["D1"]
    assert d1["two_way"]["n_clusters"] == 6  # 3 draws x the 2 definition items of the fixture
    assert d1["definitions_with_items"][0] == "heldout:4001|k.alpha"
    assert d1["two_way"]["est"] == pytest.approx(1.0)
    assert out["def"]["Dm"]["two_way"]["est"] == pytest.approx(0.5)


# --------------------------------------------------------------------------------- block H
def _h(bxd: dict[str, float] | None, r0: float | None, valid: bool | None = True) -> dict:
    def est(e: dict[str, float] | None) -> dict[str, float] | None:
        return None if e is None else {"n_worlds": 10, **e}

    return {
        "validity": {
            "valid": valid,
            "tests": [{"key": "at_limit_share", "value": 0.2, "ok": False}],
        },
        "bxd": {"def": {"definition": {"D1": est(bxd)}}},
        "r0": {"writer": est(None if r0 is None else {"est": r0, "lo": r0, "hi": r0})},
    }


@pytest.mark.parametrize(
    "e,reading",
    [
        ({"est": 0.05, "lo": 0.01, "hi": 0.09}, "writer_property"),
        ({"est": 0.10, "lo": 0.02, "hi": 0.20}, "writer_property"),  # 10 pp inclusive
        ({"est": 0.25, "lo": -0.01, "hi": 0.50}, "writer_property"),  # interval includes 0
        ({"est": 0.25, "lo": 0.10, "hi": 0.40}, "execution_pays"),
        ({"est": 0.20, "lo": 0.01, "hi": 0.40}, "execution_pays"),  # 20 pp inclusive
        ({"est": 0.15, "lo": 0.02, "hi": 0.30}, "both"),
    ],
)
def test_h1_readings(e, reading):
    h1 = rr.h_predictions("claude-sonnet-5-5", _h(e, 0.5))["H1"]
    assert h1["verdict"] == rr.JUDGED and h1["reading"] == reading


def test_h2_margin_trigger_and_validity():
    preds = rr.h_predictions("claude-sonnet-5-5", _h({"est": 0.3, "lo": 0.1, "hi": 0.5}, 0.40))
    assert preds["H2"]["verdict"] == rr.HOLDS  # 0.40 - 0.162 >= 0.20
    assert preds["H_c_trigger"]["runs_H_c"]  # 0.40 < 0.70
    low = rr.h_predictions("claude-sonnet-5-5", _h({"est": 0.3, "lo": 0.1, "hi": 0.5}, 0.36))
    assert low["H2"]["verdict"] == rr.FAILS and "reading_if_fails" in low["H2"]
    assert "H_c_trigger" not in rr.h_predictions("claude-opus-5-5", _h(None, 0.40))
    void = rr.h_predictions("claude-sonnet-5-5", _h({"est": 0.3, "lo": 0.1, "hi": 0.5}, 0.9, False))
    assert void["H1"]["verdict"] == void["H2"]["verdict"] == rr.NOT_JUDGED
    assert "validity failed: at_limit_share 20.0%" in void["H1"]["reason"]


def _calls_run(d: Path, n: int, cut: int, cell_writes: dict[str, int]) -> Path:
    d.mkdir(exist_ok=True)
    with (d / "calls.jsonl").open("w") as fh:
        for i in range(n):
            row = {
                "key": f"key{i}",
                "model": "claude-sonnet-5-5",
                "purpose": "write",
                "finish_reason": "length" if i < cut else "end_turn",
                "completion_tokens": 17000 if i < cut else 900,
            }
            fh.write(json.dumps(row) + "\n" + json.dumps({**row, "cached": True}) + "\n")
        other = {"key": "p", "model": "claude-sonnet-5-5", "purpose": "program"}
        fh.write(json.dumps(other) + "\n")
    summary = {"branches": {"def": {"defwriter": {"holdmem_D0B0": cell_writes}}}}
    (d / "world_summaries.jsonl").write_text(json.dumps(summary) + "\n")
    return d


def test_h_validity_from_calls_summaries_and_cache(tmp_path):
    run = _calls_run(tmp_path / "ha", 50, 1, {"writes": 100, "unreadable": 3, "at_limit": 2})
    v = rr.h_validity([run], "claude-sonnet-5-5")
    assert v["calls"]["distinct_write_requests"] == 50 and v["calls"]["at_limit"] == 1
    assert v["unreadable_basis"].startswith("cell writes")
    assert v["valid"] is False  # 3% of the cell writes unreadable
    cache = ResponseCache(tmp_path / "anthropic.sqlite")
    for i in range(50):
        text = "" if i < 1 else json.dumps({"value": 1, "statement": "s", "citations": []})
        fr = "length" if i < 1 else "end_turn"
        cache.put(
            f"key{i}",
            "claude-sonnet-5-5",
            {"text": text, "finish_reason": fr, "api": {"stop_reason": fr}},
        )
    v = rr.h_validity([run], "claude-sonnet-5-5", tmp_path / "anthropic.sqlite")
    assert v["cache"]["unreadable"] == 1 and v["unreadable_basis"].startswith("distinct")
    assert v["valid"] is True  # 2% at the limit and 2% unreadable, both inclusive
    over = _calls_run(tmp_path / "hb", 50, 6, {"writes": 50, "unreadable": 0, "at_limit": 6})
    assert rr.h_validity([over], "claude-sonnet-5-5")["valid"] is False  # 12% at the limit
    assert rr.h_validity([run], "claude-opus-5-5")["valid"] is None  # no request of that model


def test_block_h_sonnet_run_end_to_end(tmp_path, committed):
    # the Sonnet run: every cell equals the committed one at k = 0 (the anchors) and its D=0
    # writer re-derives every definition item right after the shift
    def after(base: CV, then: CV) -> CV:
        return lambda w, f, k: base(w, f, k) if k == 0 else then(w, f, k)

    cells = {**COMMITTED, "holdmem_D0B1_source": after(const(False), const(True))}
    run = write_run(
        tmp_path / "e1_ha_20261016T000000Z",
        cells,
        "2026-10-16",
        params={"defwriter": {"model": "claude-sonnet-5-5", "max_tokens": 17000}},
    )
    _calls_run(run, 20, 0, {"writes": 40, "unreadable": 0, "at_limit": 0})
    with (run / "items.jsonl").open("w") as fh:  # the source-epoch key invalidates at k = 1
        for w in WORLDS:
            for b in BRANCHES:
                for f in FACTS:
                    hist = [{"ts": "2026-10-02T00:00", "event": "invalidate"}]
                    row = {
                        "world_id": w,
                        "branch": b,
                        "system": "holdmem_D0B1_source",
                        "slot_id": f,
                        "history": hist,
                    }
                    fh.write(json.dumps(row) + "\n")
    res = rr.block_h([run], [committed], draws=200)
    assert res["pairing"]["ok"], res["pairing"]["counts"]
    # anchors at k = 0 only: 6 cells x 4 worlds x 3 branches x 4 facts
    assert res["pairing"]["anchor_answers_compared"] == 6 * 4 * 3 * 4
    preds = res["predictions"]
    assert preds["H1"]["reading"] == "writer_property"  # B x D = (1 - 0.25) - (1 - 0) < 0
    assert preds["H2"]["verdict"] == rr.HOLDS  # r_0 = 1.0
    q = res["quantities"]["writers"]["claude-sonnet-5-5"]
    assert q["r0"]["writer"]["est"] == pytest.approx(1.0)
    assert q["validity"]["valid"] is True
    assert res["reported"]["H_c_trigger"]["runs_H_c"] is False
    assert preds["H3"]["verdict"] == rr.NOT_JUDGED  # no thinking run given


# ------------------------------------------------------------------------------ blocks N, X
def _cf(lo: float, covered: int = 10) -> dict[str, Any]:
    node = {
        "difference": {"est": lo + 0.1, "lo": lo, "hi": lo + 0.2, "n_worlds": 20},
        "item_times_in_branch": 10,
        "item_times_with_natural": covered,
    }
    return {b: {"shift": {"definition": node}} for b in ("src", "def")}


def test_n1_verdicts():
    assert rr.n_predictions(_cf(0.05))["N1"]["verdict"] == rr.HOLDS
    assert rr.n_predictions(_cf(-0.01))["N1"]["verdict"] == rr.FAILS
    short = rr.n_predictions(_cf(0.05, covered=9))["N1"]
    assert short["verdict"] == rr.NOT_JUDGED and "cover" in short["reason"]


def test_n_anchor_compares_natural_t0_with_the_committed_def_branch(tmp_path):
    natural = tmp_path / "e1_natural_20261016T000000Z"
    natural.mkdir()
    stage_run = tmp_path / "e1_test_x"
    stage_run.mkdir()
    rows_n, rows_d = [], []
    for system in ("holdmem_D0B0", "holdmem_D1B1_source"):
        for f in FACTS:
            for b, rows in (("natural", rows_n), ("def", rows_d)):
                p = _probe("w1000", b, f, 0)
                rows.append(_answer(system, p, f != "k.beta" or b == "def"))
    (natural / "answers.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows_n))
    (stage_run / "answers.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows_d))
    (natural / "manifest.json").write_text(json.dumps({"created_utc": "x", "params": {}}))
    runs, _ = rr.read_runs([natural])
    out = rr.n_anchor(rr.Merged(runs, rr.by_family), [stage_run])
    assert out == {"compared": 8, "differ": 2, "missing": 0}


def _xcell(dom: str, judged: bool, det: float, rep: float) -> dict[str, Any]:
    return {
        "diagnosis": {"dominant": dom, "judged": judged},
        "effects": {"mem0_det": {"est": det}, "mem0_rep": {"est": rep}},
    }


def test_x1_as_amended():
    win, lose = _xcell("detection", True, 0.2, 0.0), _xcell("detection", True, 0.0, 0.1)
    off = _xcell("detection", False, 0.5, 0.0)
    three = {"a": win, "b": win, "c": win, "d": off}
    assert rr.x1_amended(three)["verdict"] == rr.HOLDS  # all of 3 (the registered rule: not)
    assert rr.x1_amended({**three, "c": lose})["verdict"] == rr.FAILS
    five = {"a": win, "b": win, "c": win, "d": win, "e": lose}
    assert rr.x1_amended(five)["verdict"] == rr.HOLDS  # 4 of 5
    assert rr.x1_amended({**five, "d": lose})["verdict"] == rr.FAILS
    assert rr.x1_amended({"a": win, "b": win})["verdict"] == rr.NOT_JUDGED


def test_x1_of_a_void_run_is_not_computed(monkeypatch):
    """A failed anchor voids X: X1 is not judged and carries no tests, so the result does not
    hold the outcome X1 would have had (decisions.md 2026-10-17)."""
    monkeypatch.setattr(rr, "REGISTERED_WORLDS", REAL_WORLDS)
    win = _xcell("detection", True, 0.2, 0.0)
    cells = {"a": win, "b": win, "c": win}
    worlds = list(REAL_WORLDS["X"])
    ok = rr.x_predictions(cells, worlds, {"ok": True, "counts": {"anchor_answers_differ": 0}})
    assert ok["X1"]["verdict"] == rr.HOLDS and ok["X1"]["n_favour_target"] == 3
    void = rr.x_predictions(cells, worlds, {"ok": False, "counts": {"anchor_answers_differ": 3}})
    assert void == {
        "X1": {
            "verdict": rr.NOT_JUDGED,
            "reason": "pairing check failed (anchor_answers_differ 3); the run is void until "
            "repaired",
            "tests": [],
        }
    }
    # judged runs keep the registered world rule
    part = rr.x_predictions(cells, worlds[:9], {"ok": True, "counts": {}})
    assert part["X1"]["verdict"] == rr.NOT_JUDGED and "w1009" in part["X1"]["reason"]


# ----------------------------------------------------------- definition-only runs (G, V, U5)
def _program(model: str, sample: int, user_words: bool = False) -> dict[str, Any]:
    temp = 0.7 if model == MAIN_MODEL else 0.0
    return {
        "program_writer": {
            "model": model,
            "temperature": temp,
            "sample": sample,
            "user_words": user_words,
        }
    }


# a program set's D1s cells: the anchors at k = 0 equal the committed cells
def _d1s(after: CV) -> dict[str, CV]:
    return {
        "holdmem_D1sB0": const(False),
        "holdmem_D1sB1_source": lambda w, f, k: k == 0 or after(w, f, k),
    }


def test_a_definition_only_run_pairs_on_what_it_answers(tmp_path, committed):
    run = write_run(
        tmp_path / "e1_gb_q_p0_20261016T000000Z",
        _d1s(on("k.alpha", "k.beta")),
        "2026-10-16",
        params=_program(MAIN_MODEL, 0),
        definition_only=True,
    )
    res = rr.block_g([run], [committed], draws=100)
    assert res["pairing"]["ok"], res["pairing"]["counts"]
    # the anchors: the two D1s cells at k = 0 (4 worlds x 3 branches x 4 facts each)
    assert res["pairing"]["anchor_answers_compared"] == 2 * 4 * 3 * 4
    node = res["quantities"]["programs"]["branches"]["def"]
    assert set(node) == {"definition"}  # no other item is answered after the fork
    assert node["definition"]["two_way"]["est"] == pytest.approx(1.0)
    # read as a run that answers every item, it misses the other items after the fork
    man = json.loads((run / "manifest.json").read_text())
    man["params"]["post_probes"] = "all"
    (run / "manifest.json").write_text(json.dumps(man))
    counts = rr.block_g([run], [committed], draws=50)["pairing"]["counts"]
    assert counts["probe_keys_missing"] == 2 * 4 * 3 * 2 * 2  # 2 cells, 2 other facts, k = 1, 2
    assert counts["gold_differs"] > 0  # recorded as not asked


def _set_runs(tmp_path: Path, tag: str, after: CV, user_words: bool) -> list[Path]:
    out = []
    for model in (MAIN_MODEL, "claude-sonnet-5-5", "claude-opus-5-5"):
        for sample in range(3):
            name = f"e1_{tag}_{model.split('/')[-1]}_p{sample}_20261016T000000Z"
            run = write_run(
                tmp_path / name,
                _d1s(after),
                "2026-10-16",
                params=_program(model, sample, user_words),
                definition_only=True,
            )
            agree = "v0" if user_words else "other"  # the program's value at t0
            with (run / "items.jsonl").open("w") as fh:
                for w in WORLDS:
                    for b in BRANCHES:
                        for f in ("k.alpha", "k.beta"):
                            hist = [
                                {
                                    "ts": "x",
                                    "event": "derive",
                                    "program": {"ok": True, "value": agree},
                                }
                            ]
                            row = {
                                "world_id": w,
                                "branch": b,
                                "system": "holdmem_D1sB1_source",
                                "slot_id": f,
                                "history": hist,
                            }
                            fh.write(json.dumps(row) + "\n")
            out.append(run)
    return out


def test_block_v_judges_the_nine_sets_and_pairs_them_with_g_b(tmp_path, committed):
    v = _set_runs(tmp_path, "v", on("k.alpha", "k.beta"), user_words=True)
    gb = _set_runs(tmp_path, "gb", on("k.alpha"), user_words=False)
    res = rr.block_v([*v, *gb], [committed], draws=100)
    assert res["pairing"]["ok"], res["pairing"]["counts"]
    assert len(res["quantities"]["V_sets"]) == len(res["quantities"]["G_b_sets"]) == 9
    preds = res["predictions"]
    assert preds["V1"]["verdict"] == preds["V2"]["verdict"] == rr.HOLDS
    assert preds["V1"]["sets"] == 9
    two = res["quantities"]["programs"]["branches"]["def"]["definition"]["two_way"]
    assert two["n_clusters"] == 9 and two["est"] == pytest.approx(1.0)
    per = res["reported"]["V_minus_G_b_per_writer"]
    assert set(per) == {MAIN_MODEL, "claude-sonnet-5-5", "claude-opus-5-5"}
    for node in per.values():
        assert len(node["sets"]) == 3
        # k.beta is right with V's programs only: half of the changed definition item-times
        assert node["update_D1sB1_source"]["pooled"]["difference"]["est"] == pytest.approx(0.5)
        assert node["agreement_t0"]["difference"]["est"] == pytest.approx(1.0)
    assert res["reported"]["S_del_interaction"]["two_way"]["est"] == pytest.approx(1.0)
    # G reads the same runs and leaves V's sets to block V
    g = rr.block_g([*v, *gb], [committed], draws=50)
    assert len(g["quantities"]["program_sets"]) == 9 and len(g["quantities"]["ignored_runs"]) == 9


def test_block_v_needs_all_nine_sets(tmp_path, committed):
    v = _set_runs(tmp_path, "v", on("k.alpha", "k.beta"), user_words=True)
    res = rr.block_v(v[:6], [committed], draws=50)
    v1 = res["predictions"]["V1"]
    assert v1["verdict"] == rr.NOT_JUDGED and v1["reason"] == "6 of 9 registered program sets given"
    assert v1["interim_on_sets_present"]["verdict"] == rr.HOLDS
    assert res["reported"]["V_minus_G_b_per_writer"] == {}  # no G-b set given
    assert rr.block_v([committed], [committed], draws=50)["predictions"]["V1"]["verdict"] == (
        rr.NOT_JUDGED
    )


# --------------------------------------------------------------------------------- block Y
SONNET = "claude-sonnet-5-5"
# the strong reader on the committed stores: the other item right at D=1 with binding only (B x D
# +1 per item there), the definition items as the committed reader except D0B1 after the shift
Y_CELLS: dict[str, CV] = {
    **COMMITTED,
    "holdmem_D0B1_source": on("k.beta", k=2),
    "holdmem_D1B0": lambda w, f, k: f == "k.alpha" and k == 1,
    "holdmem_D1B1_source": lambda w, f, k: f != "c.p1.city" or k > 0,
}


def _y_run(tmp_path: Path, cells: dict[str, CV] = Y_CELLS, status: str = "answer") -> Path:
    run = write_run(tmp_path / "e1_y_reader_20261016T000000Z", cells, "2026-10-16", reader=SONNET)
    rows = [json.loads(x) for x in (run / "answers.jsonl").read_text().splitlines()]
    with (run / "calls.jsonl").open("w") as fh:
        for i, a in enumerate(rows):
            if status == "error" and i % 20 == 0:
                a["status"] = "error"
            call = {"key": f"k{i}", "model": SONNET, "purpose": "serve", "cached": True}
            fh.write(json.dumps({**call, "finish_reason": "end_turn"}) + "\n")
        fh.write(json.dumps({"key": "w", "model": MAIN_MODEL, "purpose": "write", "cached": True}))
    (run / "answers.jsonl").write_text("".join(json.dumps(a) + "\n" for a in rows))
    # the stores: derived at t0; a B=1 cell re-derives every changed item at each checkpoint
    with (run / "items.jsonl").open("w") as fh:
        for w in WORLDS:
            for b in BRANCHES:
                for system in cells:
                    for f in FACTS:
                        hist = [{"ts": "2026-10-01T00:00", "event": "derive", "new": "valid"}]
                        hist[0]["value"] = "v0"
                        if "B1" in system and f != "t.t001":
                            for k in (1, 2):
                                ts = f"2026-10-{k + 1:02d}T00:00"
                                hist.append(
                                    {
                                        "ts": ts,
                                        "event": "invalidate",
                                        "new": "valid",
                                        "value": f"v{k}",
                                    }
                                )
                        row = {"world_id": w, "branch": b, "system": system, "slot_id": f}
                        fh.write(json.dumps({**row, "history": hist}) + "\n")
    return run


def test_block_y_end_to_end(tmp_path, committed):
    res = rr.block_y([_y_run(tmp_path)], [committed], draws=200)
    pair = res["pairing"]
    assert pair["ok"], pair["counts"]
    # the context of every answer is compared: 6 cells x 4 worlds x 3 branches x 4 facts x 3 k
    assert pair["anchor_answers_compared"] == 6 * 4 * 3 * 4 * 3
    q = res["quantities"]
    assert q["validity"]["valid"] is True and q["validity"]["answers"] == 6 * 144
    other = q["contrasts"]["def"]["other"]["D1"]
    assert other["reader"]["est"] == pytest.approx(1.0)  # D1B1 right at k > 0, the rest wrong
    assert other["difference"]["est"] == pytest.approx(0.0)  # as the committed reader
    assert other["n_items"] == other["n_items_paired"] == 4 * 2
    preds = res["predictions"]
    assert preds["Y1"]["verdict"] == preds["Y2"]["verdict"] == rr.HOLDS
    assert preds["Y1"]["reading"].startswith("the interaction is not confined")
    # definition items: B x D 0.75 committed; 0.5 with the reader (D0B1 right on k.beta at k=2)
    d = q["contrasts"]["src"]["definition"]["D1"]
    assert d["committed_reader"]["est"] == pytest.approx(0.75)
    assert d["difference"]["est"] == pytest.approx(-0.25)
    assert preds["Y3"]["verdict"] == rr.FAILS
    # the store/reader split runs on the reader's answers; the parts add up to the interaction
    node = q["store_reader"]["results"]["D1"]["def"]["other"]["interaction"]
    parts = node["store"]["est"] + node["reader"]["est"]
    assert parts == pytest.approx(node["interaction"]["est"])
    assert "right_no_value_stores_misread" in res["reported"]
    assert {"Y", "Y1-Y3"} == set(res["notes"])


def test_y_contexts_that_differ_void_only_the_predictions_that_use_them(tmp_path, committed):
    run = _y_run(tmp_path)
    rows = [json.loads(x) for x in (run / "answers.jsonl").read_text().splitlines()]
    for a in rows:  # a D_m program that ran here and failed in the committed run
        if a["system"] == "holdmem_D1sB1_source" and a["world_id"] == "w1001" and a["tags"]["k"]:
            a["context_sha256"] = "other store"
    (run / "answers.jsonl").write_text("".join(json.dumps(a) + "\n" for a in rows))
    res = rr.block_y([run], [committed], draws=50)
    assert not res["pairing"]["ok"]
    differs = res["pairing"]["counts"]["context_hash_differs"]
    assert differs == 3 * 4 * 2
    assert res["quantities"]["answers_left_out_context_differs"] == differs
    for p in res["predictions"].values():
        assert p["verdict"] != rr.NOT_JUDGED and "holdmem_D1sB1_source" in p["pairing_note"]
    dm = res["reported"]["Dm"]["def"]["definition"]
    assert dm["n_items_paired"] == 3 * 2 * 2  # w1001's item-times left out
    # a context that differs in a cell the predictions use voids them all
    for a in rows:
        if a["system"] == "holdmem_D1B0" and a["world_id"] == "w1002" and a["tags"]["k"] == 1:
            a["context_sha256"] = "other store"
    (run / "answers.jsonl").write_text("".join(json.dumps(a) + "\n" for a in rows))
    res = rr.block_y([run], [committed], draws=50)
    for p in res["predictions"].values():
        assert p["verdict"] == rr.NOT_JUDGED and "pairing check failed" in p["reason"]


def test_y_validity_counts_unreadable_and_cut_off_answers(tmp_path, committed):
    res = rr.block_y([_y_run(tmp_path, status="error")], [committed], draws=50)
    v = res["quantities"]["validity"]
    assert v["unreadable"] == 6 * 144 // 20 + 1 and v["valid"] is False
    for p in res["predictions"].values():
        assert p["verdict"] == rr.NOT_JUDGED and p["reason"].startswith("validity failed")
    assert res["predictions"]["Y1"]["reading"] is None


def test_y_requires_the_reader_runs_writes_to_be_cache_hits(tmp_path, committed):
    run = _y_run(tmp_path)
    with (run / "calls.jsonl").open("a") as fh:
        fh.write("\n" + json.dumps({"model": MAIN_MODEL, "purpose": "program", "cached": False}))
    res = rr.block_y([run], [committed], draws=50)
    assert res["pairing"]["counts"]["store_requests_sent"] == 1
    assert res["predictions"]["Y1"]["verdict"] == rr.NOT_JUDGED


# ------------------------------------------------------------------------- block N, natural
def test_natural_moved_share_of_the_shift_item_times(tmp_path):
    natural = write_run(tmp_path / "e1_n_20261016T000000Z", {}, "2026-10-16")
    probes = [json.loads(x) for x in (natural / "probes.jsonl").read_text().splitlines()]
    for p in probes:  # the natural branch: k.alpha drifts, k.beta does not
        p["branch"] = "natural"
        if p["fact_id"] == "k.beta":
            p["gold_value"] = "v0"
    (natural / "probes.jsonl").write_text("".join(json.dumps(p) + "\n" for p in probes))
    runs, _ = rr.read_runs([natural])
    tab = _table(COMMITTED)
    cause = {(w, b, f, k): "shift" for w in WORLDS for b in BRANCHES for f in FACTS for k in (1, 2)}
    out = rr.natural_moved(tab, rr.Merged(runs, rr.by_family), cause)
    assert out["def"]["definition"]["est"] == pytest.approx(0.5)
    assert out["def"]["definition"]["n_items"] == 4 * 2 * 2


# ----------------------------------------------------------------------------------- notes
def test_every_reading_note_resolves_from_the_docstring():
    for block, keys in rr.NOTES.items():
        notes = rr._notes(keys)
        assert set(notes) == set(keys), block
        assert all(len(v) > 40 for v in notes.values())
