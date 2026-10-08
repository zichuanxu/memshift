import pytest
from jsonschema import ValidationError

from memshift.schemas import validate_envelope


def _envelope() -> dict:
    return {
        "task_id": "probe-0001",
        "status": "answer",
        "data": "Acme Corp",
        "source_refs": [{"object_id": "mail:42", "revision": "7"}],
        "contract_id": "point-lookup-v1",
        "snapshot_id": "world-0-seed-0-t9",
        "diagnostics": {
            "items": [
                {
                    "item_id": "k-employer",
                    "D": 1,
                    "B": 1,
                    "status": "valid",
                    "key": {
                        "scope_epoch": 3,
                        "kappa_version": "v1",
                        "theta_w_id": "Qwen/Qwen3.5-9B",
                    },
                }
            ]
        },
    }


def test_valid_envelope_passes() -> None:
    validate_envelope(_envelope())


@pytest.mark.parametrize("status", ["answer", "abstain", "blocked", "not_ready", "error"])
def test_all_statuses_allowed(status: str) -> None:
    env = _envelope()
    env["status"] = status
    validate_envelope(env)


def test_missing_snapshot_id_fails() -> None:
    env = _envelope()
    del env["snapshot_id"]
    with pytest.raises(ValidationError):
        validate_envelope(env)


def test_unknown_top_level_field_fails() -> None:
    env = _envelope()
    env["confidence"] = 0.9
    with pytest.raises(ValidationError):
        validate_envelope(env)
