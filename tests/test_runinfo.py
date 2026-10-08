import json

from memshift.runinfo import verify_manifest, write_manifest


def test_manifest_roundtrip(tmp_path):
    run = tmp_path / "run_x"
    run.mkdir()
    (run / "answers.jsonl").write_text('{"a": 1}\n')
    (run / "analysis").mkdir()
    (run / "analysis" / "s.json").write_text("{}")
    m = write_manifest(run, "test", [{"world_id": "w0000", "seed": 0}], {"writer": "m"}, {"k": 1})
    assert set(m["files_sha256"]) == {"answers.jsonl", "analysis/s.json"}
    assert json.loads((run / "manifest.json").read_text())["run_id"] == "run_x"
    assert verify_manifest(run) == []
    (run / "answers.jsonl").write_text('{"a": 2}\n')
    assert verify_manifest(run) == ["answers.jsonl"]


def test_cache_tag_separates_engines(tmp_path, monkeypatch):
    from memshift.experiments.common import cache_path

    monkeypatch.delenv("MEMSHIFT_CACHE_TAGS", raising=False)
    shared = cache_path("org/Model-1")
    monkeypatch.setenv("MEMSHIFT_CACHE_TAGS", json.dumps({"org/Model-1": "vllm0.30.0"}))
    tagged = cache_path("org/Model-1")
    assert shared.name == "org_Model-1.sqlite"
    assert tagged.name == "org_Model-1@vllm0.30.0.sqlite"
    assert cache_path("org/Other").name == "org_Other.sqlite"
    run = tmp_path / "run_y"
    run.mkdir()
    m = write_manifest(run, "test", [], {"reader": "org/Model-1", "writer": "org/Other"}, {})
    assert m["models"]["reader"]["cache_tag"] == "vllm0.30.0"
    assert "cache_tag" not in m["models"]["writer"]
