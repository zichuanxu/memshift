import pytest

from memshift.analysis.e1 import analyse
from memshift.experiments.e6 import run_world
from tests.fakes import FakeLLM

pytest.importorskip("portable_memory")


def test_e6_leave_test_degrades_without_extension(tmp_path):
    out = run_world(5, tmp_path, FakeLLM())
    carriers = out["carriers"]
    assert carriers is not None
    res = analyse(tmp_path)
    cells = res["cells"]["src"]
    # with the derivation record the imported store keeps its bindings (B=1 detection)
    assert cells["imported_with_ext"]["MissedInvalidation"]["est"] == 0.0
    # without it the import is append-only: every changed item is missed
    assert cells["imported_no_ext"]["MissedInvalidation"]["est"] == 1.0


def test_e6_t2_equivalence(tmp_path):
    from memshift.analysis.e6 import analyse as analyse_t2

    run_world(5, tmp_path, FakeLLM())
    res = analyse_t2(tmp_path)["imports"]
    # the derivation record carries the bindings: no detection loss relative to the native store
    assert res["imported_with_ext"]["diff_90"]["missed"]["est"] == 0.0
    # without it every changed item is missed, far outside the margin
    assert res["imported_no_ext"]["diff_90"]["missed"]["est"] == 1.0
    assert not res["imported_no_ext"]["t2_equivalent"]
