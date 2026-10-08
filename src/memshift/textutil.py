"""Text helpers shared by every system: tokenisation for BM25, token budgeting, hashing.

The token count used for context budgets is a deterministic approximation (word pieces and
punctuation). All systems use the same function, so budgets are comparable; billed token costs
come from the serving engine's `usage` field, not from this function.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]")
_WORD_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "for",
        "from",
        "has",
        "have",
        "i",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "so",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "you",
        "your",
        "do",
        "does",
        "did",
        "not",
        "no",
        "yes",
    ]
)


def _cost(tok: str) -> int:
    return max(1, (len(tok) + 3) // 4) if tok[0].isalnum() else 1


def approx_tokens(text: str) -> int:
    """Deterministic token estimate: alphanumeric runs cost 1 per 4 chars (min 1), punctuation 1."""
    return sum(_cost(tok) for tok in _TOKEN_RE.findall(text))


def truncate_tokens(text: str, budget: int) -> str:
    """Cut `text` so that approx_tokens(result) <= budget, on a token boundary."""
    used = 0
    end = 0
    for m in _TOKEN_RE.finditer(text):
        cost = _cost(m.group(0))
        if used + cost > budget:
            return text[:end]
        used += cost
        end = m.end()
    return text


def bm25_tokens(text: str) -> list[str]:
    """Lower-cased word tokens without stopwords, for lexical indexes."""
    return [w for w in _WORD_RE.findall(text.lower()) if w not in STOPWORDS]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str))
