"""Envelope schema (copied from reference/research_2/schemas/response.schema.json on 2026-09-28).

`diagnostics.items[{item_id, D, B, status, key}]` is the MemShift extension; the schema leaves
`diagnostics` open, so it validates unchanged.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

from jsonschema import Draft202012Validator


def load_envelope_schema() -> dict[str, Any]:
    text = resources.files(__package__).joinpath("envelope.schema.json").read_text(encoding="utf-8")
    schema: dict[str, Any] = json.loads(text)
    return schema


def validate_envelope(envelope: dict[str, Any]) -> None:
    """Raise jsonschema.ValidationError if the envelope does not match the schema."""
    Draft202012Validator(load_envelope_schema()).validate(envelope)
