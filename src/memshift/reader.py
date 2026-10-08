"""The shared reader (plan §4.3): one model, one fixed template, one evidence budget for every
system under test. Systems differ only in the evidence blocks they hand over.
"""

from __future__ import annotations

from typing import Any

from memshift.llm import LLMResult
from memshift.textutil import sha256_text

READER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"enum": ["answer", "abstain"]},
        "value": {
            "anyOf": [
                {"type": "string"},
                {"type": "array", "items": {"type": "string"}},
                {"type": "null"},
            ]
        },
    },
    "required": ["status", "value"],
    "additionalProperties": False,
}

SYSTEM = (
    "You answer questions about a person (called 'the user') using only the evidence provided. "
    "Evidence items may disagree or be outdated; prefer the most recent information that applies "
    "at the question's date. If the evidence does not determine the answer, abstain."
)

INSTRUCTIONS = (
    'Reply with JSON {"status": "answer" or "abstain", "value": ...}. When answering, give only '
    "the value in the requested format: a string, or a JSON array of strings for lists (use [] "
    "for an empty list). When abstaining, set value to null."
)


def render_context(blocks: list[str]) -> str:
    if not blocks:
        return "Evidence: (none)"
    return "Evidence:\n" + "\n\n".join(f"[{i + 1}] {b}" for i, b in enumerate(blocks))


def context_hash(blocks: list[str]) -> str:
    return sha256_text(render_context(blocks))


def read_request(question: str, blocks: list[str], purpose: str = "serve") -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": f"{render_context(blocks)}\n\nQuestion: {question}\n\n{INSTRUCTIONS}",
            },
        ],
        "max_tokens": 256,
        "schema": READER_SCHEMA,
        "purpose": purpose,
    }


def parse_answer(res: LLMResult) -> tuple[str, Any]:
    data = res.json()
    if not isinstance(data, dict) or data.get("status") not in ("answer", "abstain"):
        return "error", None
    return str(data["status"]), data.get("value")
