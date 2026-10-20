"""The control-cell stage end to end on offline runs (fake writer, three-month world): a published
run, and four new runs with the control cells (an anchor in each, the oracle, scope, refresh and
template cells, the retraction-aware programs, the query-time view and the thinking writer), and a
fifth new run of the full-access control on two branches only (src and def, as the D0full runs
have). The integrity checks (hash assertion, probe sets, anchors) stop
the stage, the key audit and the scope = cited check are marked, a run of two branches is merged
beside the runs of three, and the stage command writes the manifest."""

import json
import shutil
import sys
from pathlib import Path

import pytest

import memshift.experiments.e1 as e1_driver
from memshift.analysis import controls as ctl
from memshift.analysis import controls_data as cd
from memshift.analysis import controls_paper as cp
from memshift.analysis import stage
from memshift.experiments.common import read_jsonl, world_params, write_jsonl
from memshift.experiments.e1 import parse_cells
from memshift.experiments.e1 import run as run_e1
from memshift.gen.kappa import KAPPA_DEFS
from memshift.runinfo import file_sha256
from tests.fakes import FakeLLM

WORLD = "w0005"
BRANCHES = ("src", "def", "orphan")
ALL_CELLS = (
    "D0B0,D1B0,D0B1_source,D1B1_source,D1B1_cited,D1sB1_source,D0B1_gold,D1B1_gold,D1B1_scope,"
    "D1B0_refresh,D0tB1_source,D1sB1_source_ra,D0thinkB1_source,zero_Dm,D0fullB1_source"
)
# run -> the systems it keeps; a driver run writes every cell it was given, so the five control
# runs are cut from one offline run (the anchors are then the same rows, which is what the
# corruption tests below start from; real replays are checked on the published runs)
RUNS: dict[str, tuple[set[str], str, bool]] = {
    "base": (
        {"zero", "oracle", "holdmem_json_arm"}
        | {f"holdmem_{c}" for c in (
            "D0B0", "D1B0", "D0B1_source", "D1B1_source", "D1B1_cited", "D1sB1_source"
        )},
        "D0B0,D1B0,D0B1_source,D1B1_source,D1B1_cited,D1sB1_source",
        True,
    ),
    "ctl1": (
        {f"holdmem_{c}" for c in (
            "D0B1_source", "D0B1_gold", "D1B1_gold", "D1B1_scope", "D1B0_refresh", "D0tB1_source"
        )},
        "D0B1_source,D0B1_gold,D1B1_gold,D1B1_scope,D1B0_refresh,D0tB1_source",
        False,
    ),
    "ctl2": (
        {"holdmem_D1sB1_source", "holdmem_D1sB1_source_ra"},
        "D1sB1_source,D1sB1_source_ra",
        False,
    ),
    "ctl3": ({"zero_Dm"}, "zero_Dm", False),
    "ctl4": (
        {"holdmem_D0B1_source", "holdmem_D0thinkB1_source"},
        "D0B1_source,D0thinkB1_source",
        False,
    ),
    "ctl5": (
        {"holdmem_D0B1_source", "holdmem_D0fullB1_source"},
        "D0B1_source,D0fullB1_source",
        False,
    ),
}  # fmt: skip
ONLY_SD = {"ctl5": ("src", "def")}  # the runs that cover fewer than the three branches


def cut(
    src: Path,
    dst: Path,
    systems: set[str],
    cells: str,
    baselines: bool,
    branches: tuple[str, ...] = BRANCHES,
) -> Path:
    """One run directory with the systems (and branches) of `src` that belong to it, as the driver
    writes it: a run of fewer branches has no probe, answer, item or hash assertion of the others,
    and its manifest declares the branches it covers."""
    dst.mkdir()
    for name in ("world_summaries.jsonl", "env.json"):
        shutil.copy(src / name, dst / name)
    for f in src.glob("hash_check_*"):
        if f.stem.rsplit("_", 1)[1] in branches:
            shutil.copy(f, dst / f.name)
    write_jsonl(
        dst / "probes.jsonl",
        [r for r in read_jsonl(src / "probes.jsonl") if r["branch"] in branches],
    )
    for name in ("answers", "items", "maintenance"):
        rows = [
            r
            for r in read_jsonl(src / f"{name}.jsonl")
            if r["system"] in systems and r.get("branch", branches[0]) in branches
        ]
        write_jsonl(dst / f"{name}.jsonl", rows)
    man = json.loads((src / "manifest.json").read_text())
    tokens = [t for t in cells.split(",") if t]
    man["run_id"] = dst.name
    man["params"].update(
        {
            "cells": [c.name for c in parse_cells(cells)[0]],
            "views": [t for t in tokens if t == "zero_Dm"],
            "baselines": baselines,
            "branches": list(branches),
        }
    )
    (dst / "manifest.json").write_text(json.dumps(man))
    return dst


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("controls_stage")
    mp = pytest.MonkeyPatch()
    mp.setattr(e1_driver, "client", lambda model, run_dir, workers=48, **kw: FakeLLM(model))
    cells, views = parse_cells(ALL_CELLS)
    (root / "all").mkdir()
    try:
        whole = run_e1(
            [5],
            "all",
            out=root / "all",
            cells_cfg=cells,
            baselines=True,
            branches=BRANCHES,
            world_params=world_params(3),
            discriminative=True,
            views=views,
        )
    finally:
        mp.undo()
    return {
        name: cut(whole, root / name, *spec, branches=ONLY_SD.get(name, BRANCHES))
        for name, spec in RUNS.items()
    }


CTL = ("ctl1", "ctl2", "ctl3", "ctl4")
FULL = (*CTL, "ctl5")  # with the full-access control, which ran on two branches


def dirs(runs: dict[str, Path], *names: str) -> list[Path]:
    return [runs[n] for n in names]


def copy_ctl(runs: dict[str, Path], tmp: Path, name: str) -> Path:
    dest = tmp / f"{name}_copy"
    shutil.copytree(runs[name], dest)
    return dest


# ------------------------------------------------------------------------------ the happy path
@pytest.fixture(scope="module")
def staged(runs, tmp_path_factory):
    out = tmp_path_factory.mktemp("staged")
    return out, ctl.analyse(dirs(runs, *CTL), [runs["base"]], out, worlds=[WORLD])


def test_stage_writes_the_analysis_and_every_check_passes(staged):
    out, res = staged
    checks = res["checks"]
    assert {k: checks[k]["pass"] for k in ("hash", "discriminative", "probe_sets", "anchors")} == {
        "hash": True, "discriminative": True, "probe_sets": True, "anchors": True,
    }  # fmt: skip
    assert checks["audit"]["pass"] is True and checks["scope_cited"]["pass"] is True
    assert (checks["passed"], checks["failed"], checks["not_evaluated"]) == (5, 0, 0)
    adir = out / "analysis"
    assert json.loads((adir / "ctl_metrics.json").read_text())["n_worlds"] == 1
    assert (adir / "ctl_item_table.jsonl").stat().st_size > 0
    assert not (adir / "ctl_checks_failed.json").exists()
    assert res["worlds"] == [WORLD] and checks["worlds_missing"] == []


def test_every_cell_comes_from_one_run_and_anchors_are_the_published_ones(staged, runs):
    _, res = staged
    origin = res["cell_origin"]
    assert origin["holdmem_D0B1_gold"] == [runs["ctl1"].name]
    assert origin["holdmem_D1sB1_source_ra"] == [runs["ctl2"].name]
    assert origin["zero_Dm"] == [runs["ctl3"].name]
    assert origin["holdmem_D0thinkB1_source"] == [runs["ctl4"].name]
    for anchor in ("holdmem_D0B1_source", "holdmem_D1sB1_source", "zero", "holdmem_D1B0"):
        assert origin[anchor] == [runs["base"].name], anchor  # the published rows are used


def test_the_anchor_is_compared_in_every_run_that_carries_it(staged, runs):
    anchors = staged[1]["checks"]["anchors"]["runs"]
    assert set(anchors[runs["ctl1"].name]) == {"holdmem_D0B1_source"}
    assert set(anchors[runs["ctl4"].name]) == {"holdmem_D0B1_source"}
    assert set(anchors[runs["ctl2"].name]) == {"holdmem_D1sB1_source"}
    assert runs["ctl3"].name not in anchors or not anchors[runs["ctl3"].name]  # the view has none
    cell = anchors[runs["ctl1"].name]["holdmem_D0B1_source"]
    assert cell["answers"] > 100 and cell["differ"] == 0 and cell["share"] == 0.0
    assert cell["context_differ"] == 0 and cell["by_field"]["contract_valid"] == 0


def test_the_view_is_compared_with_the_published_zero_system(staged, runs):
    probes = staged[1]["checks"]["probe_sets"]["runs"][runs["ctl3"].name]
    assert probes["reference_system"] == "zero"
    assert probes["systems"]["zero_Dm"]["pass"] and probes["systems"]["zero_Dm"]["asked"] > 50
    assert probes["gold"]["pass"] and probes["gold"]["differ"] == 0


def test_the_key_audit_is_the_real_one_and_agrees_with_the_cells(staged):
    audit = staged[1]["checks"]["audit"]
    assert set(audit["cells"]) == {
        "holdmem_D0B1_gold", "holdmem_D1B1_gold", "holdmem_D1B1_scope",
    }  # fmt: skip
    for cell in audit["cells"].values():
        assert cell["pass"] and cell["worlds"] == 1
        groups = [g for b in cell["branches"].values() for g in b.values()]
        assert groups and all(g["run"] == g["audit"] for g in groups)
    scope = audit["cells"]["holdmem_D1B1_scope"]["branches"]["orphan"]
    assert set(scope) == {"definition"}
    assert scope["definition"]["run_rates"]["missed"] == 0.0  # the scope key misses nothing


def test_contrasts_thresholds_and_reported_quantities_are_there(staged):
    _, res = staged
    assert set(res["contrasts"]) == set(ctl.contrast_keys())
    assert set(res["threshold_tests"]) == set(ctl.CONTROLS)
    assert sum(res["summary"].values()) == len(ctl.CONTROLS)
    # the fake writer and reader make every contrast exactly zero, so each comparison is computed
    assert res["contrasts"]["G_upd_D1_orphan_all"]["est"] == 0.0
    assert ctl.outcome(res["threshold_tests"]["G1"]) == "met_required"
    assert res["reported"]["program_t0"]["all"]["ratio"]["est"] == 1.0  # both program kinds run
    assert ctl.outcome(res["threshold_tests"]["M3"]) == "met_required"
    think = res["reported"]["think"]["all"]
    assert think["writes"] > 0 and think["unreadable"]["est"] == 0.0
    assert ctl.outcome(res["threshold_tests"]["D2"]) == "met_required"
    assert "orphan" in res["reported"]["over_retraction"]
    assert res["reported"]["ra"]["orphan"]["all"]["runs"] > 0


def test_the_paper_renders_the_staged_result(staged):
    _, res = staged
    macros = cp.numbers(res)
    assert macros["CtlChecksPassed"] == "5 of 5"
    assert macros["CtlCheckAuditPass"] == "pass" and macros["CtlWorlds"] == "1"
    assert not any(k.endswith("Verdict") for k in macros)
    tex = cp.tab_controls(res)
    assert r"\pass" not in tex and "oracle & Update, oracle" in tex


def test_worlds_the_runs_do_not_have_are_listed(runs, tmp_path):
    res = ctl.analyse(dirs(runs, *CTL), [runs["base"]], tmp_path, worlds=[WORLD, "w0006"])
    assert res["checks"]["worlds_missing"] == ["w0006"] and res["worlds"] == [WORLD]


# ------------------------------------------------ a run of two branches beside runs of three
@pytest.fixture(scope="module")
def staged_full(runs, tmp_path_factory):
    out = tmp_path_factory.mktemp("staged_full")
    return out, ctl.analyse(dirs(runs, *FULL), [runs["base"]], out, worlds=[WORLD])


def test_a_run_of_src_and_def_only_is_what_the_fixture_says_it_is(runs):
    ids = {r["branch"] for r in read_jsonl(runs["ctl5"] / "answers.jsonl")}
    assert ids == {"src", "def"}
    assert {r["branch"] for r in read_jsonl(runs["ctl5"] / "probes.jsonl")} == {"src", "def"}
    assert {r["branch"] for r in read_jsonl(runs["ctl1"] / "answers.jsonl")} == set(BRANCHES)
    assert sorted(f.name for f in runs["ctl5"].glob("hash_check_*")) == [
        f"hash_check_{WORLD}_def.json",
        f"hash_check_{WORLD}_src.json",
    ]
    man = json.loads((runs["ctl5"] / "manifest.json").read_text())
    assert man["params"]["branches"] == ["src", "def"]
    assert cd.run_branches(cd.load_run(runs["ctl5"], {WORLD})) == {"src", "def"}
    assert cd.run_branches(cd.load_run(runs["ctl1"], {WORLD})) == set(BRANCHES)
    assert {a["system"] for a in read_jsonl(runs["ctl5"] / "answers.jsonl")} == {
        "holdmem_D0B1_source",
        "holdmem_D0fullB1_source",
    }


def test_a_run_of_src_and_def_only_merges_beside_three_branch_runs_and_every_check_passes(
    staged_full, runs
):
    out, res = staged_full
    checks = res["checks"]
    assert {k: checks[k]["pass"] for k in ("hash", "discriminative", "probe_sets", "anchors")} == {
        "hash": True, "discriminative": True, "probe_sets": True, "anchors": True,
    }  # fmt: skip
    assert (checks["passed"], checks["failed"], checks["not_evaluated"]) == (5, 0, 0)
    assert res["cell_origin"]["holdmem_D0fullB1_source"] == [runs["ctl5"].name]
    assert res["cell_origin"]["holdmem_D0B1_source"] == [runs["base"].name]  # the anchor: published
    assert res["branches"] == list(BRANCHES)  # the other runs still cover the orphan branch
    assert json.loads((out / "analysis" / "ctl_metrics.json").read_text())["n_worlds"] == 1
    # the probe set of the run is the published one on its two branches, and its gold is the gold
    node = checks["probe_sets"]["runs"][runs["ctl5"].name]
    assert node["branches"] == ["def", "src"] and node["reference_system"] == "holdmem_D0B1_source"
    assert (
        node["gold"]["pass"]
        and node["gold"]["probes"]
        < checks["probe_sets"]["runs"][runs["ctl1"].name]["gold"]["probes"]
    )
    for system in ("holdmem_D0B1_source", "holdmem_D0fullB1_source"):
        s5 = node["systems"][system]
        assert s5["pass"] and s5["missing"] == 0 and s5["extra"] == 0 and s5["asked"] > 0
    three = checks["probe_sets"]["runs"][runs["ctl1"].name]["systems"]["holdmem_D0B1_source"]
    assert node["systems"]["holdmem_D0B1_source"]["asked"] < three["asked"]  # no orphan probes


def test_the_anchor_of_a_run_of_two_branches_is_compared_on_its_branches(staged_full, runs):
    anchors = staged_full[1]["checks"]["anchors"]["runs"]
    c5, c1 = (anchors[runs[n].name]["holdmem_D0B1_source"] for n in ("ctl5", "ctl1"))
    assert c5["answers"] > 100 and c5["differ"] == 0 and c5["missing_on_one_side"] == 0
    assert c5["share"] == 0.0 and c5["pass"] is True
    assert (
        0 < c5["answers"] < c1["answers"]
    )  # the published anchor's orphan answers are not missing
    assert set(anchors[runs["ctl5"].name]) == {"holdmem_D0B1_source"}  # the new cell has no anchor


def test_the_full_access_cell_is_contrasted_with_the_published_d1_cell_on_src_and_def(
    staged_full, staged
):
    _, res = staged_full
    assert set(res["contrasts"]) == set(ctl.contrast_keys()) and "F1" in res["threshold_tests"]
    for br in ("src", "def"):
        for key in (f"F_rep_{br}_definition", f"F_repmech_{br}_definition", f"F_gain_upd_{br}_all"):
            e = res["contrasts"][key]
            assert e["n_worlds"] == 1 and e["n_items"] > 0 and e["est"] == e["est"], key
    # the fake reader answers alike from both cells (no behavioural gap) while the stored value of
    # D = 1 is the checker's fallback and D0full's the fake writer's (a mechanical gap of 100 pp):
    # two of the four tests are met, and the regions of the two variants disagree
    for br in ("src", "def"):
        assert res["contrasts"][f"F_rep_{br}_definition"]["est"] == 0.0
        assert res["contrasts"][f"F_repmech_{br}_definition"]["est"] == 1.0
    v = res["threshold_tests"]["F1"]
    assert (ctl.outcome(v), v["n_met"], v["n_tests"]) == ("below_required", 2, 4)
    assert v["regions"]["F_rep_src_definition"] == "access"
    assert v["regions"]["F_repmech_def_definition"] == "execution"
    assert set(res["cells"]["holdmem_D0fullB1_source"]) == {"src", "def"}
    assert set(res["cells"]["holdmem_D1B1_source"]) == set(BRANCHES)
    # the run adds a cell and no change to the others: every other control is what it was without
    before = staged[1]["threshold_tests"]
    assert ctl.outcome(before["F1"]) == "not_computed"  # no D0full cell in the four runs
    for name in ctl.CONTROLS:
        if name != "F1":
            assert res["threshold_tests"][name] == before[name], name


def test_the_notes_of_the_full_access_run_are_counted_per_definition_and_source(staged_full):
    full = staged_full[1]["reported"]["full"]
    assert full["limit_tokens"] == 25_000 and set(full) >= {"all", "src", "def"}
    assert "orphan" not in full and full["all"]["rederivations"] > 0
    assert (
        full["all"]["rederivations"] == full["src"]["rederivations"] + full["def"]["rederivations"]
    )
    assert set(full["definitions"]) <= {f"k.{k}" for k in KAPPA_DEFS}
    for node in full["definitions"].values():
        for label in ("src", "def"):
            for source, cell in node[label]["by_source"].items():
                assert cell["rederivations"] == node[label]["rederivations"], source
                assert cell["truncated"] <= cell["rederivations"]
    cut = {f for f, n in full["definitions"].items() if n["all"]["truncated"]}
    assert full["truncated_definitions"] == sorted(cut)
    m = cp.numbers(staged_full[1])
    assert "CtlFoneVerdict" not in m and "CtlFoneReading" not in m
    assert m["CtlFullWritesAll"] == f"{full['all']['rederivations']:,}"
    assert m["CtlFRepSrcDefinition"].startswith("0.0\\,pp")


def test_a_declared_branch_without_answers_fails_the_anchor_and_probe_set_checks(runs, tmp_path):
    ctl5 = copy_ctl(runs, tmp_path, "ctl5")
    man = json.loads((ctl5 / "manifest.json").read_text())
    man["params"]["branches"] = list(BRANCHES)  # the run says it covers the orphan branch too
    (ctl5 / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(cd.ControlsCheckError, match="probe_sets, anchors") as err:
        ctl.analyse([ctl5], [runs["base"]], tmp_path / "s", worlds=[WORLD])
    node = err.value.checks["probe_sets"]["runs"][ctl5.name]
    assert node["branches"] == sorted(BRANCHES)
    assert all(not v["pass"] and v["missing"] > 0 for v in node["systems"].values())
    cell = err.value.checks["anchors"]["runs"][ctl5.name]["holdmem_D0B1_source"]
    assert cell["pass"] is False and cell["missing_on_one_side"] == cell["differ"] > 0


def test_a_run_of_two_branches_that_lost_one_of_them_fails_the_same_way(runs, tmp_path):
    ctl5 = copy_ctl(runs, tmp_path, "ctl5")
    for name in ("answers", "items"):
        rows = [r for r in read_jsonl(ctl5 / f"{name}.jsonl") if r["branch"] != "def"]
        write_jsonl(ctl5 / f"{name}.jsonl", rows)
    with pytest.raises(cd.ControlsCheckError) as err:  # the manifest still says src and def
        ctl.analyse([ctl5], [runs["base"]], tmp_path / "s", worlds=[WORLD])
    checks = err.value.checks
    assert checks["anchors"]["pass"] is False and checks["probe_sets"]["pass"] is False
    assert checks["hash"]["pass"] is True
    assert checks["probe_sets"]["runs"][ctl5.name]["branches"] == ["def", "src"]


def test_a_run_without_a_declared_branch_list_is_read_by_the_branches_it_has_answers_for(
    runs, tmp_path
):
    ctl5 = copy_ctl(runs, tmp_path, "ctl5")
    man = json.loads((ctl5 / "manifest.json").read_text())
    del man["params"]["branches"]
    (ctl5 / "manifest.json").write_text(json.dumps(man))
    res = ctl.analyse([ctl5], [runs["base"]], tmp_path / "s", worlds=[WORLD])
    assert res["checks"]["anchors"]["pass"] and res["checks"]["probe_sets"]["pass"]
    assert cd.run_branches(cd.load_run(ctl5, {WORLD})) == {"src", "def"}


def test_stage_ctl_command_line_takes_the_full_access_runs_as_more_ctl_directories(
    runs, tmp_path, monkeypatch, capsys
):
    stage_dir = make_stage(runs, tmp_path)
    out = tmp_path / "stage5_controls_full"
    monkeypatch.setattr(cd, "CTL_SEEDS", (5,))
    argv = ["stage", "ctl", str(out), *(f"ctl:{runs[n]}" for n in FULL), f"base:{stage_dir}"]
    monkeypatch.setattr(sys, "argv", argv)
    stage.main()
    assert capsys.readouterr().out.strip().splitlines()[-1] == str(out)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["roles"]["ctl"] == [runs[n].name for n in FULL]
    res = json.loads((out / "analysis" / "ctl_metrics.json").read_text())
    assert res["checks"]["passed"] == 5 and "F1" in res["threshold_tests"]
    assert res["reported"]["full"]["all"]["rederivations"] > 0


# ---------------------------------------------------------------------------- anchor equality
def flip(path: Path, system: str, every: int, field: str = "contract_valid") -> int:
    """Flip `field` of every `every`-th answer of `system` in a run's answers.jsonl."""
    rows = read_jsonl(path / "answers.jsonl")
    n = 0
    for i, r in enumerate([r for r in rows if r["system"] == system]):
        if i % every == 0:
            r[field] = not r[field]
            n += 1
    write_jsonl(path / "answers.jsonl", rows)
    return n


def test_a_trivial_share_of_differing_anchor_answers_is_reported_not_fatal(runs, tmp_path):
    ctl1 = copy_ctl(runs, tmp_path, "ctl1")
    n_answers = sum(
        a["system"] == "holdmem_D0B1_source" for a in read_jsonl(ctl1 / "answers.jsonl")
    )
    assert flip(ctl1, "holdmem_D0B1_source", every=n_answers + 1) == 1  # the first answer only
    res = ctl.analyse(
        [ctl1, *dirs(runs, "ctl2", "ctl3", "ctl4")], [runs["base"]], tmp_path / "s", worlds=[WORLD]
    )
    cell = res["checks"]["anchors"]["runs"][ctl1.name]["holdmem_D0B1_source"]
    assert cell["differ"] == 1 and cell["by_field"]["contract_valid"] == 1
    assert 0 < cell["share"] < cd.ANCHOR_TOLERANCE and cell["pass"] is True
    assert res["checks"]["anchors"]["pass"] is True


def test_anchor_answers_that_differ_beyond_the_tolerance_fail_loudly(runs, tmp_path):
    ctl1 = copy_ctl(runs, tmp_path, "ctl1")
    flip(ctl1, "holdmem_D0B1_source", every=10)  # a tenth of the anchor's answers
    out = tmp_path / "s"
    with pytest.raises(cd.ControlsCheckError, match="anchors"):
        ctl.analyse(
            [ctl1, *dirs(runs, "ctl2", "ctl3", "ctl4")], [runs["base"]], out, worlds=[WORLD]
        )
    failed = json.loads((out / "analysis" / "ctl_checks_failed.json").read_text())
    cell = failed["anchors"]["runs"][ctl1.name]["holdmem_D0B1_source"]
    assert cell["pass"] is False and cell["share"] > 0.05 and cell["examples"]
    assert failed["probe_sets"]["pass"] is True  # the other checks are reported beside it
    assert not (out / "analysis" / "ctl_metrics.json").exists()  # no number from a void merge


def test_a_changed_value_or_status_is_a_difference_too(runs, tmp_path):
    ctl1 = copy_ctl(runs, tmp_path, "ctl1")
    rows = read_jsonl(ctl1 / "answers.jsonl")
    for r in rows:
        if r["system"] == "holdmem_D0B1_source":
            r["value"] = "changed"
            r["status"] = "answer"
    write_jsonl(ctl1 / "answers.jsonl", rows)
    with pytest.raises(cd.ControlsCheckError) as err:
        ctl.analyse([ctl1], [runs["base"]], tmp_path / "s", worlds=[WORLD])
    cell = err.value.checks["anchors"]["runs"][ctl1.name]["holdmem_D0B1_source"]
    assert cell["by_field"]["value"] > 0 and cell["differ"] >= cell["by_field"]["value"]


def test_the_rendered_context_is_reported_without_a_threshold(runs, tmp_path):
    ctl1 = copy_ctl(runs, tmp_path, "ctl1")
    rows = read_jsonl(ctl1 / "answers.jsonl")
    for r in rows:
        if r["system"] == "holdmem_D0B1_source":
            r["context_sha256"] = "0" * 64
    write_jsonl(ctl1 / "answers.jsonl", rows)
    res = ctl.analyse([ctl1], [runs["base"]], tmp_path / "s", worlds=[WORLD])
    cell = res["checks"]["anchors"]["runs"][ctl1.name]["holdmem_D0B1_source"]
    assert cell["pass"] is True and cell["differ"] == 0
    assert cell["context_differ"] == cell["answers"] and cell["context_share"] == 1.0


# ------------------------------------------------------------------------------ probe sets
def test_a_probe_asked_by_one_system_only_fails_the_probe_set_check(runs, tmp_path):
    ctl1 = copy_ctl(runs, tmp_path, "ctl1")
    rows = read_jsonl(ctl1 / "answers.jsonl")
    dropped = next(
        r for r in rows if r["system"] == "holdmem_D0B1_gold" and r["tags"].get("k", 0) == 2
    )
    rows = [r for r in rows if r is not dropped]
    write_jsonl(ctl1 / "answers.jsonl", rows)
    with pytest.raises(cd.ControlsCheckError, match="probe_sets") as err:
        ctl.analyse([ctl1], [runs["base"]], tmp_path / "s", worlds=[WORLD])
    node = err.value.checks["probe_sets"]["runs"][ctl1.name]["systems"]
    assert node["holdmem_D0B1_gold"]["pass"] is False and node["holdmem_D0B1_gold"]["missing"] == 1
    assert node["holdmem_D0B1_source"]["pass"] is True
    assert node["holdmem_D0B1_gold"]["examples"] == [
        [dropped["world_id"], dropped["branch"], dropped["fact_id"], 2]
    ]


def test_an_extra_probe_fails_the_probe_set_check_as_well(runs, tmp_path):
    ctl1 = copy_ctl(runs, tmp_path, "ctl1")
    rows = read_jsonl(ctl1 / "answers.jsonl")
    extra = dict(next(r for r in rows if r["system"] == "holdmem_D0B1_gold"))
    extra["fact_id"] = "u.invented"
    extra["probe_id"] = "invented"
    write_jsonl(ctl1 / "answers.jsonl", [*rows, extra])
    with pytest.raises(cd.ControlsCheckError) as err:
        ctl.analyse([ctl1], [runs["base"]], tmp_path / "s", worlds=[WORLD])
    assert (
        err.value.checks["probe_sets"]["runs"][ctl1.name]["systems"]["holdmem_D0B1_gold"]["extra"]
        == 1
    )


def test_the_gold_of_a_probe_must_be_the_published_gold(runs, tmp_path):
    ctl1 = copy_ctl(runs, tmp_path, "ctl1")
    rows = read_jsonl(ctl1 / "probes.jsonl")
    rows[3]["gold_value"] = "something else"
    write_jsonl(ctl1 / "probes.jsonl", rows)
    with pytest.raises(cd.ControlsCheckError) as err:
        ctl.analyse([ctl1], [runs["base"]], tmp_path / "s", worlds=[WORLD])
    gold = err.value.checks["probe_sets"]["runs"][ctl1.name]["gold"]
    assert (
        gold["pass"] is False and gold["differ"] == 1 and gold["examples"] == [rows[3]["probe_id"]]
    )


def test_the_asked_flags_are_part_of_the_gold_comparison(runs, tmp_path):
    ctl1 = copy_ctl(runs, tmp_path, "ctl1")
    rows = read_jsonl(ctl1 / "probes.jsonl")
    post = next(r for r in rows if "asked" in r)
    post["asked"] = not post["asked"]
    write_jsonl(ctl1 / "probes.jsonl", rows)
    with pytest.raises(cd.ControlsCheckError, match="probe_sets"):
        ctl.analyse([ctl1], [runs["base"]], tmp_path / "s", worlds=[WORLD])


def test_a_new_run_without_any_published_cell_to_compare_with_fails(runs, tmp_path):
    bare = tmp_path / "base_without_zero"
    shutil.copytree(runs["base"], bare)
    rows = [r for r in read_jsonl(bare / "answers.jsonl") if r["system"] != "zero"]
    write_jsonl(bare / "answers.jsonl", rows)
    with pytest.raises(cd.ControlsCheckError, match="probe_sets") as err:
        ctl.analyse(dirs(runs, "ctl3"), [bare], tmp_path / "s", worlds=[WORLD])
    node = err.value.checks["probe_sets"]["runs"][runs["ctl3"].name]
    assert node["reference_system"] is None
    assert node["systems"]["zero_Dm"]["reason"] == "no published cell to compare with"


# -------------------------------------------------------------------------- hash and protocol
def test_a_failed_pre_shift_hash_assertion_stops_the_stage(runs, tmp_path):
    ctl1 = copy_ctl(runs, tmp_path, "ctl1")
    f = next(ctl1.glob("hash_check_*_def.json"))
    check = json.loads(f.read_text())
    check["ok"] = False
    f.write_text(json.dumps(check))
    with pytest.raises(cd.ControlsCheckError, match="hash") as err:
        ctl.analyse([ctl1], [runs["base"]], tmp_path / "s", worlds=[WORLD])
    assert err.value.checks["hash"]["runs"][ctl1.name]["failed"] == [f.name]


def test_a_run_without_hash_assertions_is_not_accepted(runs, tmp_path):
    ctl1 = copy_ctl(runs, tmp_path, "ctl1")
    for f in ctl1.glob("hash_check_*.json"):
        f.unlink()
    with pytest.raises(cd.ControlsCheckError, match="hash"):
        ctl.analyse([ctl1], [runs["base"]], tmp_path / "s", worlds=[WORLD])


def test_discriminative_probes_are_required(runs, tmp_path):
    ctl1 = copy_ctl(runs, tmp_path, "ctl1")
    man = json.loads((ctl1 / "manifest.json").read_text())
    man["params"]["probes_after_fork"] = "all"
    (ctl1 / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(cd.ControlsCheckError, match="discriminative") as err:
        ctl.analyse([ctl1], [runs["base"]], tmp_path / "s", worlds=[WORLD])
    assert err.value.checks["discriminative"]["runs"][ctl1.name] == "all"


# ----------------------------------------------------------------------------- audit marks
def test_a_key_audit_mismatch_is_marked_and_does_not_stop_the_stage(runs, tmp_path, capsys):
    real = ctl.e1_keys.audit

    def off_by_one(seeds, branches, months):
        out = real(seeds, branches, months)
        out["orphan"]["gold"]["rule"]["unchanged"] += 1
        return out

    res = ctl.analyse(
        dirs(runs, *CTL), [runs["base"]], tmp_path, worlds=[WORLD], audit_fn=off_by_one
    )
    audit = res["checks"]["audit"]
    assert audit["pass"] is False and (res["checks"]["passed"], res["checks"]["failed"]) == (4, 1)
    assert audit["cells"]["holdmem_D1B1_gold"]["pass"] is False
    assert audit["cells"]["holdmem_D1B1_scope"]["pass"] is True  # only the gold key was changed
    assert "WARNING: implementation checks failed" in capsys.readouterr().out
    assert cp.numbers(res)["CtlCheckAuditPass"] == "fail"


# ----------------------------------------------------------------------------------- merging
def test_a_cell_in_two_new_runs_without_a_published_copy_is_ambiguous(runs, tmp_path):
    ctl1 = copy_ctl(runs, tmp_path, "ctl1")
    with pytest.raises(ValueError, match=r"holdmem_D[01]B1_\w+ in w0005 is in both"):
        ctl.analyse([runs["ctl1"], ctl1], [runs["base"]], tmp_path / "s", worlds=[WORLD])


def test_published_runs_of_other_worlds_are_skipped_not_loaded(runs, tmp_path):
    other = tmp_path / "other"
    shutil.copytree(runs["base"], other)
    man = json.loads((other / "manifest.json").read_text())
    man["params"]["seeds"] = [1012]
    (other / "manifest.json").write_text(json.dumps(man))
    # an empty directory would fail to load: the manifest says it is another world
    (other / "answers.jsonl").write_text("not json\n")
    res = ctl.analyse(dirs(runs, *CTL), [other, runs["base"]], tmp_path / "s", worlds=[WORLD])
    assert res["checks"]["anchors"]["pass"] is True


def test_the_new_runs_must_have_answers_in_the_requested_worlds(runs, tmp_path):
    with pytest.raises(ValueError, match="no answers"):
        ctl.analyse(dirs(runs, "ctl1"), [runs["base"]], tmp_path, worlds=["w9999"])


# ------------------------------------------------------------------------------- stage command
def make_stage(runs: dict[str, Path], tmp: Path, sha: str | None = None) -> Path:
    """A published stage directory next to the run directory it merged."""
    stage_dir = runs["base"].parent / "stage_published"
    stage_dir.mkdir(exist_ok=True)
    man = {
        "experiment": "merged-e1",
        "sources": [
            {
                "run_id": runs["base"].name,
                "manifest_sha256": sha or file_sha256(runs["base"] / "manifest.json"),
                "worlds": [WORLD, WORLD, WORLD],
            },
            # a source of other test worlds: not on this machine, and not needed
            {"run_id": "other_worlds", "manifest_sha256": "0" * 64, "worlds": ["w1012"]},
        ],
    }
    (stage_dir / "manifest.json").write_text(json.dumps(man))
    return stage_dir


def test_a_published_stage_expands_to_its_source_runs_checked_against_the_manifest_hash(
    runs, tmp_path
):
    stage_dir = make_stage(runs, tmp_path)
    assert cd.expand_base([stage_dir], {WORLD}) == [runs["base"]]
    assert cd.expand_base([stage_dir, runs["base"]], {WORLD}) == [runs["base"]]  # no duplicates
    with pytest.raises(FileNotFoundError, match="other_worlds"):  # without the worlds: all sources
        cd.expand_base([stage_dir])
    make_stage(runs, tmp_path, sha="0" * 64)
    with pytest.raises(ValueError, match="manifest differs"):
        cd.expand_base([stage_dir], {WORLD})
    man = json.loads((stage_dir / "manifest.json").read_text())
    man["sources"][0]["run_id"] = "gone"
    (stage_dir / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(FileNotFoundError, match="no raw JSONL"):
        cd.expand_base([stage_dir], {WORLD})
    with pytest.raises(FileNotFoundError, match="neither a run"):
        cd.expand_base([tmp_path])


def test_stage_ctl_command_line(runs, tmp_path, monkeypatch, capsys):
    stage_dir = make_stage(runs, tmp_path)
    out = tmp_path / "stage5_controls"
    monkeypatch.setattr(cd, "CTL_SEEDS", (5,))
    argv = ["stage", "ctl", str(out), *(f"ctl:{runs[n]}" for n in CTL), f"base:{stage_dir}"]
    monkeypatch.setattr(sys, "argv", argv)
    stage.main()
    assert capsys.readouterr().out.strip().splitlines()[-1] == str(out)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["experiment"] == "merged-ctl"
    assert manifest["roles"] == {
        "ctl": [runs[n].name for n in CTL],
        "base": [runs["base"].name],
    }
    assert [s["run_id"] for s in manifest["sources"]] == [
        *(runs[n].name for n in CTL),
        runs["base"].name,
    ]
    assert all(s["manifest_sha256"] for s in manifest["sources"])
    assert set(manifest["files_sha256"]) == {
        "analysis/ctl_metrics.json",
        "analysis/ctl_item_table.jsonl",
    }
    res = json.loads((out / "analysis" / "ctl_metrics.json").read_text())
    assert res["checks"]["passed"] == 5


def test_stage_ctl_rejects_unknown_roles_and_missing_sources(tmp_path, monkeypatch):
    for argv in (
        ["stage", "ctl", str(tmp_path / "o"), "ctl:a", "other:b"],
        ["stage", "ctl", str(tmp_path / "o"), "ctl:a"],
        ["stage", "ctl", str(tmp_path / "o"), "base:b"],
    ):
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit):
            stage.main()
