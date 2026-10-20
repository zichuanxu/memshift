"""Check the results table of the README against the numbers the paper is built from.

The table under "Main results" in the README is the paper's main table (Table 4, `tab:main` in
paper/sections/results.tex), cell for cell: per shift and item class, the class's share of the
changed items, repair correctness with a key (r_0, r_1) and Update without a key (a_0, a_1) at D=0
and D=1, and the interaction B x D on the changed items caused by the shift and on all changed
items, each with its 95% world-bootstrap interval. Every cell is a LaTeX macro of
`memshift.analysis.paper_numbers` (`rows()` names the macro of each cell, as Table 4 does), except
the dash that Table 4 prints where a redefinition changes no other item. This script rebuilds the
table from those macros and compares it with the README, so no number in the table is typed by
hand. It also checks the number in the README's text below the table (TEXT): the largest share of a
class's changed items that the source key missed.

The macros are read from `paper/generated/numbers.tex` when it exists, else from
`results/numbers.tex` (written by `make results`). The README is `docs/release/README.md` when it
exists (the source of the released README), else `README.md`.

Usage: python scripts/readme_results.py [--numbers FILE] [--readme FILE] [--write]
Exit code 0 when the README holds the table and the text the macros give, 1 otherwise (the expected
table and the README's, or the missing sentence, are printed). `--write` replaces the README's table
with the expected one; it does not rewrite the text.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHIFTS = (("Src", "source revision"), ("Def", "redefinition"), ("Orphan", "deletion"))
CLASSES = (("Definition", "definition"), ("Other", "other"))
# (column header, kind, macro pattern with {s} the shift and {c} the item class). Kind "pct" is a
# percentage printed without its unit (\mspct in Table 4), "ci" an estimate with its interval (\ci).
COLUMNS = (
    ("Share (%)", "pct", "ExMixMain{s}{c}Share"),
    ("r_0 (%)", "pct", "ExDetMain{s}{c}DzeroBoneRepairEst"),
    ("r_1 (%)", "pct", "ExDetMain{s}{c}DoneBoneRepairEst"),
    ("a_0 (%)", "pct", "ExDetMain{s}{c}DzeroBzeroUpdateEst"),
    ("a_1 (%)", "pct", "ExDetMain{s}{c}DoneBzeroUpdateEst"),
    ("B×D, caused by the shift (pp)", "ci", "ExRevCauseMain{s}Shift{c}BxD"),
    ("B×D, all (pp)", "ci", "ExAcctMain{s}{c}BxD"),
)
# Cells that Table 4 prints as a dash (a redefinition itself changes no other item): (shift, class,
# column header).
DASHED = {("Def", "Other", "B×D, caused by the shift (pp)")}
DASH = "—"
HEADER = "| Shift | Items | " + " | ".join(h for h, _, _ in COLUMNS) + " |"
MACRO = re.compile(r"^\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$")
CELL = re.compile(r"[+-]?\d+\.\d \[[+-]?\d+\.\d, [+-]?\d+\.\d\]")
PCT = re.compile(r"\d+\.\d")
# Sentences of the README text with a number: (macro of a percentage, sentence with {} for it).
TEXT = (("ExDetMainSourceMissedMax", "source key missed at most {}% of a class's changed items"),)

Cell = tuple[str, str | None]  # (kind, macro name); kind "dash" has no macro


def rows() -> list[tuple[str, str, list[Cell]]]:
    """(shift, item class, cells) for each row of Table 4, in its order."""
    out = []
    for skey, sname in SHIFTS:
        for ckey, cname in CLASSES:
            cells: list[Cell] = []
            for header, kind, pattern in COLUMNS:
                if (skey, ckey, header) in DASHED:
                    cells.append(("dash", None))
                else:
                    cells.append((kind, pattern.format(s=skey, c=ckey)))
            out.append((sname, cname, cells))
    return out


def macros(path: Path) -> dict[str, str]:
    """Macro name -> LaTeX value, for every one-line `\\newcommand` in `path`."""
    out = {}
    for line in path.read_text().splitlines():
        m = MACRO.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def cell(tex: str) -> str:
    """'$+$27.3\\,pp [$+$21.6, $+$32.6]' -> '+27.3 [+21.6, +32.6]'; '16.2\\% [10.1, 21.8]' ->
    '16.2 [10.1, 21.8]'. Anything else (a placeholder for a missing stage) raises ValueError."""
    s = tex.replace("$+$", "+").replace("$-$", "-").replace("\\%", "").replace("\\,pp", "")
    s = " ".join(s.split())
    if not CELL.fullmatch(s):
        raise ValueError(f"not an estimate with its interval: {tex!r}")
    return s


def pct(tex: str) -> str:
    """'16.3\\%' -> '16.3'. Anything else (a placeholder for a missing stage) raises ValueError."""
    s = tex.replace("\\%", "").strip()
    if not PCT.fullmatch(s):
        raise ValueError(f"not a percentage: {tex!r}")
    return s


def table(values: dict[str, str]) -> list[str]:
    """The README table, one line per row, from the macros."""
    lines = [HEADER, "|" + "---|" * (2 + len(COLUMNS))]
    missing = []
    for sname, cname, cells in rows():
        texts = []
        for kind, name in cells:
            if name is None:
                texts.append(DASH)
            elif name not in values:
                missing.append(name)
            else:
                texts.append(pct(values[name]) if kind == "pct" else cell(values[name]))
        lines.append(f"| {sname} | {cname} | " + " | ".join(texts) + " |")
    if missing:
        raise ValueError(f"macros not defined: {', '.join(missing)}")
    return lines


def sentences(values: dict[str, str]) -> list[str]:
    """The sentences of TEXT with their numbers from the macros."""
    missing = [name for name, _ in TEXT if name not in values]
    if missing:
        raise ValueError(f"macros not defined: {', '.join(missing)}")
    return [template.format(pct(values[name])) for name, template in TEXT]


def readme_table(lines: list[str]) -> tuple[int, int]:
    """Start and end (exclusive) of the table that begins with HEADER in the README lines."""
    if HEADER not in lines:
        raise ValueError(f"no table with the header {HEADER!r}")
    start = lines.index(HEADER)
    end = start
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    return start, end


def _default(*candidates: str) -> Path:
    return next((ROOT / c for c in candidates if (ROOT / c).exists()), ROOT / candidates[-1])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--numbers", type=Path, default=None)
    ap.add_argument("--readme", type=Path, default=None)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    numbers = a.numbers or _default("paper/generated/numbers.tex", "results/numbers.tex")
    readme = a.readme or _default("docs/release/README.md", "README.md")
    values = macros(numbers)
    expected = table(values)
    lines = readme.read_text().split("\n")
    text = " ".join(" ".join(lines).split())
    absent = [s for s in sentences(values) if s not in text]
    for s in absent:
        print(f"{readme}: no sentence {s!r}", file=sys.stderr)
    start, end = readme_table(lines)
    if lines[start:end] == expected:
        print(f"{readme}: results table matches {numbers}")
        return 1 if absent else 0
    if a.write:
        lines[start:end] = expected
        readme.write_text("\n".join(lines))
        print(f"{readme}: results table written from {numbers}")
        return 1 if absent else 0
    print(f"{readme}: results table differs from {numbers}", file=sys.stderr)
    print("expected:\n" + "\n".join(expected), file=sys.stderr)
    print("README:\n" + "\n".join(lines[start:end]), file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
