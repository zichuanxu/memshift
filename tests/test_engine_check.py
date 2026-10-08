import json

from memshift.analysis.engine_check import analyse


def _run(path, answers):
    path.mkdir()
    (path / "answers.jsonl").write_text("".join(json.dumps(a) + "\n" for a in answers))
    return path


def _a(probe, value, cv, system="zero", ctx="c"):
    return {
        "system": system,
        "world_id": "w1000",
        "branch": "src",
        "probe_id": probe,
        "status": "answered",
        "value": value,
        "contract_valid": cv,
        "context_sha256": ctx,
    }


def test_engine_check_matches_on_probe_and_counts_changes(tmp_path):
    old = _run(tmp_path / "old", [_a("p1", 1, True), _a("p2", 2, True), _a("p3", 3, False)])
    new = _run(
        tmp_path / "new",
        [_a("p1", 1, True), _a("p2", 5, False), _a("p3", 3, False), _a("p4", 4, True)],
    )
    res = analyse([old], [new], tmp_path / "out")
    assert res["n_matched"] == 3
    assert res["same_context_share"] == 1.0
    assert abs(res["same_answer_share"] - 2 / 3) < 1e-12
    cell = res["per_system_branch"]["zero:src"]
    assert cell["n"] == 3
    assert abs(cell["diff_pp"] + 100 / 3) < 1e-9
    assert abs(res["max_abs_cell_diff_pp"] - 100 / 3) < 1e-9
    written = json.loads((tmp_path / "out" / "analysis" / "engine_check.json").read_text())
    assert written["old_runs"] == ["old"] and written["new_runs"] == ["new"]
