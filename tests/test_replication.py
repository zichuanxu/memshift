import copy
import json
from pathlib import Path

from memshift.analysis.replication import (
    contrast_rows,
    in_predicted_direction,
    path2_within_threshold,
    replicate,
    replication_class,
)

STAGE2 = Path(__file__).resolve().parents[1] / "runs"


def _stage2() -> tuple[dict, dict, dict]:
    return tuple(
        json.loads((STAGE2 / f"stage2_e{n}" / "analysis" / f"e{n}_metrics.json").read_text())
        for n in (1, 2, 3)
    )


def test_replication_classes_and_the_path2_threshold():
    orig = {"est": 0.09, "lo": 0.06, "hi": 0.12}
    assert replication_class(orig, {"est": 0.05, "lo": 0.01, "hi": 0.09}) == "replicated"
    assert replication_class(orig, {"est": 0.02, "lo": -0.01, "hi": 0.05}) == "same direction"
    assert replication_class(orig, {"est": -0.02, "lo": -0.05, "hi": 0.01}) == "not replicated"
    neg = {"est": -0.02, "lo": -0.03, "hi": -0.01}
    assert replication_class(neg, {"est": -0.02, "lo": -0.04, "hi": -0.005}) == "replicated"
    assert path2_within_threshold({"est": 0.0, "lo": 0.0, "hi": 0.0}) is True
    assert path2_within_threshold({"est": 0.05, "lo": 0.03, "hi": 0.07}) is False


def test_self_replication_and_a_flipped_effect():
    o = _stage2()
    same = replicate(o, o)
    effects = [c for c in same["contrasts"] if c["kind"] == "effect"]
    assert effects and not any(c["replication"] == "not replicated" for c in effects)
    assert same["summary"]["membership_same"] == same["summary"]["membership"]
    assert same["paths_within_threshold"]["path1"]
    r = copy.deepcopy(o)
    key = r[0]["contrasts"]["src"]["B_effect_update_pooled_source"]
    key.update(est=-0.01, lo=-0.03, hi=0.01)
    flipped = replicate(o, r)
    row = next(
        c
        for c in flipped["contrasts"]
        if c["branch"] == "src" and c["contrast"] == "B_effect_update_pooled_source"
    )
    assert row["replication"] == "not replicated"
    assert not flipped["paths_within_threshold"]["path1"]


def test_in_predicted_direction_is_the_sign_against_the_stated_direction():
    pos = {"est": 0.01, "lo": -0.02, "hi": 0.04}  # the interval does not matter, only the sign
    neg = {"est": -0.01, "lo": -0.04, "hi": 0.02}
    assert in_predicted_direction("effect", pos) is True
    assert in_predicted_direction("effect", neg) is False
    assert in_predicted_direction("effect", {"est": 0.0, "lo": 0.0, "hi": 0.0}) is False
    # the D effect on missed invalidation is predicted zero: within the threshold of path 2
    assert in_predicted_direction("path2", {"est": 0.0, "lo": 0.0, "hi": 0.0}) is True
    assert in_predicted_direction("path2", {"est": 0.05, "lo": 0.03, "hi": 0.07}) is False
    assert in_predicted_direction("path2", {"est": -0.05, "lo": -0.07, "hi": -0.03}) is False
    assert in_predicted_direction("descriptive", pos) is None
    assert in_predicted_direction("effect", None) is None
    assert in_predicted_direction("effect", {"est": float("nan"), "lo": 0.0, "hi": 0.0}) is None


def test_replicated_is_not_the_same_as_in_predicted_direction():
    """A contrast negative in both stages with an interval that excludes 0 replicates, and is
    against the prediction; the committed stages have one such contrast (the orphan cited B x D)."""
    o = _stage2()
    rows = [r for r in contrast_rows(o[0], o[0]) if r["kind"] == "effect"]
    replicated = [r for r in rows if r["replication"] == "replicated"]
    against = [r for r in replicated if not in_predicted_direction("effect", r["orig"])]
    assert [(r["branch"], r["contrast"]) for r in against] == [
        ("orphan", "BxD_interaction_update_cited")
    ]
