"""Lexical retrieval with a token budget, shared by HoldMem (write and read paths), the zero system
and the oracle. Candidates are ranked by BM25, packed into the budget in rank order (each document
truncated to a per-document cap), and then presented in chronological order so that every system
sees evidence in the same arrangement.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime

from rank_bm25 import BM25Okapi

from memshift.textutil import approx_tokens, bm25_tokens, truncate_tokens


@dataclass(frozen=True)
class Doc:
    doc_id: str
    text: str
    ts: datetime  # used for chronological presentation
    revision: int = 0


_LOCK = threading.Lock()
_TOKENS: OrderedDict[tuple[str, int, int, int], list[str]] = OrderedDict()
_INDEXES: OrderedDict[Hashable, Future[LexIndex]] = OrderedDict()


def _tokens(d: Doc) -> list[str]:
    key = (d.doc_id, d.revision, len(d.text), hash(d.text))
    with _LOCK:
        hit = _TOKENS.get(key)
    if hit is None:
        hit = bm25_tokens(d.text) or ["_"]
        with _LOCK:
            _TOKENS[key] = hit
            if len(_TOKENS) > 200_000:
                _TOKENS.popitem(last=False)
    return hit


class LexIndex:
    def __init__(self, docs: Sequence[Doc]):
        self.docs = list(docs)
        self._bm25 = BM25Okapi([_tokens(d) for d in self.docs]) if docs else None

    def rank(self, query: str) -> list[tuple[Doc, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(bm25_tokens(query) or ["_"])
        order = sorted(range(len(self.docs)), key=lambda i: (-scores[i], self.docs[i].doc_id))
        return [(self.docs[i], float(scores[i])) for i in order]


def pack(
    ranked: Sequence[Doc], budget: int, per_doc_cap: int, chronological: bool = True
) -> list[tuple[Doc, str]]:
    """Take documents in rank order until the budget is used; return (doc, possibly cut text)."""
    chosen: list[tuple[Doc, str]] = []
    used = 0
    for d in ranked:
        text = truncate_tokens(d.text, per_doc_cap)
        cost = approx_tokens(text) + 8
        if used + cost > budget:
            remaining = budget - used - 8
            if remaining > 60:
                chosen.append((d, truncate_tokens(text, remaining)))
            break
        chosen.append((d, text))
        used += cost
    if chronological:
        chosen.sort(key=lambda x: (x[0].ts, x[0].doc_id))
    return chosen


def select(
    docs: Sequence[Doc], query: str, budget: int, per_doc_cap: int, min_score: float = 0.0
) -> list[tuple[Doc, str]]:
    ranked = [d for d, s in LexIndex(docs).rank(query) if s > min_score]
    return pack(ranked, budget, per_doc_cap)


def cached_index(key: Hashable, make_docs: Callable[[], Sequence[Doc]]) -> LexIndex:
    """One index per key, built once even when many threads ask for it at the same time."""
    with _LOCK:
        fut = _INDEXES.get(key)
        owner = fut is None
        if fut is None:
            fut = Future()
            _INDEXES[key] = fut
            if len(_INDEXES) > 512:
                _INDEXES.popitem(last=False)
    if owner:
        try:
            fut.set_result(LexIndex(make_docs()))
        except Exception as exc:
            fut.set_exception(exc)
            with _LOCK:
                _INDEXES.pop(key, None)
            raise
    return fut.result()
