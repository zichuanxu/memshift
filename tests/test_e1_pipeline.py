from memshift.analysis.e1 import analyse
from memshift.experiments.common import read_jsonl
from memshift.experiments.e1 import run_world
from memshift.runinfo import verify_manifest, write_manifest
from tests.fakes import FakeLLM


def test_e1_world_runs_and_analyses(tmp_path):
    summary = run_world(5, tmp_path, branches=("src",), baselines=True, writer=FakeLLM())
    assert summary["branches"]["src"]["hash_ok"]
    write_manifest(tmp_path, "E1-test", summary["worlds"], {"writer": "fake"}, {})
    assert verify_manifest(tmp_path) == []
    res = analyse(tmp_path)
    cells = res["cells"]["src"]
    # B=0 never invalidates; B=1 source-epoch detects every changed item
    assert cells["holdmem_D0B0"]["MissedInvalidation"]["est"] == 1.0
    assert cells["holdmem_D0B1_source"]["MissedInvalidation"]["est"] == 0.0
    # D does not move detection in the white-box implementation
    assert res["contrasts"]["src"]["D_effect_missed_source"]["est"] == 0.0
    assert "BxD_interaction_update_source" in res["contrasts"]["src"]
    assert res["pre_intervention"]["cells_identical_answers"]


def test_e1_baselines_only(tmp_path):
    summary = run_world(5, tmp_path, branches=("src",), baselines=True, writer=FakeLLM(),
                        cells_cfg=[])  # fmt: skip
    assert summary["branches"]["src"]["hash_ok"]
    systems = {r["system"] for r in read_jsonl(tmp_path / "answers.jsonl")}
    assert systems and not any(s.startswith("holdmem") for s in systems)


def test_noise_floor_request_body_unchanged_for_main_writer():
    """The noise-floor client for the main writer keeps the stage-2 request extras (cache keys)."""
    from memshift.experiments.common import MAIN_MODEL
    from memshift.llm import default_extra_body

    assert {**default_extra_body(MAIN_MODEL), "seed": 1} == {
        "chat_template_kwargs": {"enable_thinking": False},
        "seed": 1,
    }
