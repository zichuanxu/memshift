#!/usr/bin/env python3
"""Check the evidence quotes of the neighbour audit.

Neighbour quotes: every quote must be a verbatim substring (alphanumeric-normalised, so hyphenation,
ligatures and whitespace do not matter) of the paper text src/<src>.txt, at most 25 words, and the
PDF page on which it is found is reported. The texts are pdftotext output of each neighbour's arXiv
PDF (form-feed characters separate pages) and are NOT part of the repository: put them in
audit/neighbours/src/ (or pass --src-dir) to run this check. Objects whose text is missing
are reported as SKIPPED and make the exit code 2 unless --allow-missing is given.

MemShift quotes (objects with "src": "repo"): each evidence names a repository file and the quote
must occur in it (whitespace-normalised); this needs no external text.

Fields without "evidence" quotes (the derived table fields) are not checked here; they are
consistency-checked by tests/test_neighbour_table.py.

usage: verify.py [--src-dir DIR] [--repo-root DIR] [--allow-missing] properties.json [more.json ...]
Each file holds one object or a list of objects with "key", "src" and "fields":
{field: {"value": ..., "evidence": [{"quote": ..., "loc": ..., "file": ... (repo only)}]}}.
Exit code 1 if any quote fails.
"""

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LIG = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}


def norm(s: str) -> str:
    for a, b in LIG.items():
        s = s.replace(a, b)
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"[^0-9a-z]", "", s)


_cache: dict[tuple[str, str], tuple[str, list[int]] | None] = {}


def load(src_dir: Path, name: str):
    key = (str(src_dir), name)
    if key in _cache:
        return _cache[key]
    path = src_dir / f"{name}.txt"
    if not path.exists():
        _cache[key] = None
        return None
    pages = path.read_text(errors="replace").split("\f")
    buf, starts, pos = [], [], 0
    for p in pages:
        n = norm(p)
        starts.append(pos)
        buf.append(n)
        pos += len(n)
    _cache[key] = ("".join(buf), starts)
    return _cache[key]


def page_of(src_dir: Path, name: str, quote: str):
    loaded = load(src_dir, name)
    q = norm(quote)
    if loaded is None or not q:
        return None
    full, starts = loaded
    i = full.find(q)
    if i < 0:
        return None
    pg = 1
    for k, s in enumerate(starts):
        if s <= i:
            pg = k + 1
    return pg


def ws(s: str) -> str:
    return re.sub(r"\s+", "", s)


def check_repo(o: dict, repo_root: Path) -> tuple[int, int]:
    bad = total = 0
    for fname, f in o["fields"].items():
        for ev in f.get("evidence", []):
            total += 1
            path = repo_root / ev["file"]
            if not path.exists():
                print(f"[NO FILE] {o['key']}.{fname}: {ev['file']}")
                bad += 1
            elif ws(ev["quote"]) not in ws(path.read_text()):
                print(f"[NOT FOUND] {o['key']}.{fname} in {ev['file']}: {ev['quote'][:70]}")
                bad += 1
    return total, bad


def check_paper(o: dict, src_dir: Path) -> tuple[int, int]:
    bad = total = 0
    for fname, f in o["fields"].items():
        for ev in f.get("evidence", []):
            total += 1
            q = ev.get("quote", "")
            nw = len(q.split())
            pg = page_of(src_dir, o["src"], q)
            if nw > 25:
                print(f"[TOO LONG {nw}w] {o['key']}.{fname}: {q[:80]}")
                bad += 1
            if pg is None:
                print(f"[NOT FOUND] {o['key']}.{fname}: {q}")
                bad += 1
            elif ev.get("pdf_page") not in (None, pg):
                print(f"[PAGE {ev['pdf_page']} != {pg}] {o['key']}.{fname}")
                bad += 1
            if not ev.get("loc"):
                print(f"[NO LOC] {o['key']}.{fname}")
                bad += 1
    return total, bad


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--src-dir", type=Path, default=HERE / "src")
    ap.add_argument("--repo-root", type=Path, default=REPO)
    ap.add_argument("--allow-missing", action="store_true")
    a = ap.parse_args(argv)
    bad = total = 0
    skipped: list[tuple[str, str]] = []
    for path in a.paths:
        data = json.loads(path.read_text())
        for o in data if isinstance(data, list) else [data]:
            if o["src"] == "repo":
                t, b = check_repo(o, a.repo_root)
            elif load(a.src_dir, o["src"]) is None:
                skipped.append((o["key"], o["src"]))
                continue
            else:
                t, b = check_paper(o, a.src_dir)
            total += t
            bad += b
    if skipped:
        names = ", ".join(f"{src}.txt" for _, src in skipped)
        print(f"[SKIPPED] no paper text for {len(skipped)} objects (not in the repo): {names}")
    print(f"checked {total} quotes, {bad} problems, {len(skipped)} objects skipped")
    if bad:
        return 1
    return 2 if skipped and not a.allow_missing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
