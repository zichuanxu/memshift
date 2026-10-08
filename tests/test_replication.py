import copy
import json
from pathlib import Path

from memshift.analysis.replication import (
    as_predicted,
    contrast_rows,
    effect_verdict,
    path2_verdict,
    replicate,
)

STAGE2 = Path(__file__).resolve().parents[1] / "runs"


def _stage2() -> tuple[dict, dict, dict]:
    return tuple(
        json.loads((STAGE2 / f"stage2_e{n}" / "analysis" / f"e{n}_metrics.json").read_text())
        for n in (1, 2, 3)
    )


def test_verdict_rules():
    orig = {"est": 0.09, "lo": 0.06, "hi": 0.12}
    assert effect_verdict(orig, {"est": 0.05, "lo": 0.01, "hi": 0.09}) == "replicated"
    assert effect_verdict(orig, {"est": 0.02, "lo": -0.01, "hi": 0.05}) == "same direction"
    assert effect_verdict(orig, {"est": -0.02, "lo": -0.05, "hi": 0.01}) == "not replicated"
    neg = {"est": -0.02, "lo": -0.03, "hi": -0.01}
    assert effect_verdict(neg, {"est": -0.02, "lo": -0.04, "hi": -0.005}) == "replicated"
    assert path2_verdict({"est": 0.0, "lo": 0.0, "hi": 0.0}) == "not triggered"
    assert path2_verdict({"est": 0.05, "lo": 0.03, "hi": 0.07}) == "triggered"


def test_self_replication_and_a_flipped_effect():
    o = _stage2()
    same = replicate(o, o)
    effects = [c for c in same["contrasts"] if c["kind"] == "effect"]
    assert effects and not any(c["verdict"] == "not replicated" for c in effects)
    assert same["summary"]["membership_same"] == same["summary"]["membership"]
    assert not same["gate_on_replication"]["path1"]
    r = copy.deepcopy(o)
    key = r[0]["contrasts"]["src"]["B_effect_update_pooled_source"]
    key.update(est=-0.01, lo=-0.03, hi=0.01)
    flipped = replicate(o, r)
    row = next(
        c
        for c in flipped["contrasts"]
        if c["branch"] == "src" and c["contrast"] == "B_effect_update_pooled_source"
    )
    assert row["verdict"] == "not replicated"
    assert flipped["gate_on_replication"]["path1"]


def test_as_predicted_is_the_sign_against_the_preregistered_direction():
    pos = {"est": 0.01, "lo": -0.02, "hi": 0.04}  # the interval does not matter, only the sign
    neg = {"est": -0.01, "lo": -0.04, "hi": 0.02}
    assert as_predicted("effect", pos) is True
    assert as_predicted("effect", neg) is False
    assert as_predicted("effect", {"est": 0.0, "lo": 0.0, "hi": 0.0}) is False
    # the D effect on missed invalidation is predicted zero and judged by path 2
    assert as_predicted("path2", {"est": 0.0, "lo": 0.0, "hi": 0.0}) is True
    assert as_predicted("path2", {"est": 0.05, "lo": 0.03, "hi": 0.07}) is False
    assert as_predicted("path2", {"est": -0.05, "lo": -0.07, "hi": -0.03}) is False
    assert as_predicted("descriptive", pos) is None
    assert as_predicted("effect", None) is None
    assert as_predicted("effect", {"est": float("nan"), "lo": 0.0, "hi": 0.0}) is None


def test_replicated_is_not_the_same_as_as_predicted():
    """A contrast negative in both stages with an interval that excludes 0 replicates, and is
    against the prediction; the committed stages have one such contrast (the orphan cited B x D)."""
    o = _stage2()
    rows = [r for r in contrast_rows(o[0], o[0]) if r["kind"] == "effect"]
    replicated = [r for r in rows if r["verdict"] == "replicated"]
    against = [r for r in replicated if not as_predicted("effect", r["orig"])]
    assert [(r["branch"], r["contrast"]) for r in against] == [
        ("orphan", "BxD_interaction_update_cited")
    ]
