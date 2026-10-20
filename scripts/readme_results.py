"""Check the results table of the README against the numbers the paper is built from.

The table under "Main results" in the README is the main table of the paper: per shift and item
class, repair correctness with binding (r_0, r_1) and Update without binding (a_0, a_1) at D=0 and
D=1, and the binding-by-executability interaction B x D, each with its 95% world-bootstrap
interval. Every cell is a LaTeX macro of `memshift.analysis.paper_numbers`. This script rebuilds the
table from those macros and compares it with the README, so no number in the table is typed by
hand.

The macros are read from `paper/generated/numbers.tex` when it exists, else from
`results/numbers.tex` (written by `make results`). The README is `docs/release/README.md` when it
exists (the source of the released README), else `README.md`.

Usage: python scripts/readme_results.py [--numbers FILE] [--readme FILE] [--write]
Exit code 0 when the README holds the table the macros give, 1 otherwise (the expected table and the
README's are printed). `--write` replaces the README's table with the expected one.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHIFTS = (("Src", "source revision"), ("Def", "redefinition"), ("Orphan", "deletion"))
CLASSES = (("Definition", "definition"), ("Other", "other"))
# (column header, macro suffix); the macro is ExAcctMain<shift><class><suffix>
COLUMNS = (
    ("r_0 (%)", "RZero"),
    ("r_1 (%)", "ROne"),
    ("a_0 (%)", "AZero"),
    ("a_1 (%)", "AOne"),
    ("B×D (pp)", "BxD"),
)
HEADER = "| Shift | Items | " + " | ".join(h for h, _ in COLUMNS) + " |"
MACRO = re.compile(r"^\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$")
CELL = re.compile(r"[+-]?\d+\.\d \[[+-]?\d+\.\d, [+-]?\d+\.\d\]")


def macros(path: Path) -> dict[str, str]:
    """Macro name -> LaTeX value, for every one-line `\\newcommand` in `path`."""
    out = {}
    for line in path.read_text().splitlines():
        m = MACRO.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def cell(tex: str) -> str:
    """'16.2\\% [10.1, 21.8]' -> '16.2 [10.1, 21.8]'; '$+$27.3\\,pp [$+$21.6, $+$32.6]' ->
    '+27.3 [+21.6, +32.6]'. Anything else (a placeholder for a missing stage) raises ValueError."""
    s = tex.replace("$+$", "+").replace("$-$", "-").replace("\\%", "").replace("\\,pp", "")
    s = " ".join(s.split())
    if not CELL.fullmatch(s):
        raise ValueError(f"not an estimate with its interval: {tex!r}")
    return s


def table(values: dict[str, str]) -> list[str]:
    """The README table, one line per row, from the macros."""
    lines = [HEADER, "|" + "---|" * (2 + len(COLUMNS))]
    missing = []
    for skey, sname in SHIFTS:
        for ckey, cname in CLASSES:
            cells = []
            for _, suffix in COLUMNS:
                name = f"ExAcctMain{skey}{ckey}{suffix}"
                if name not in values:
                    missing.append(name)
                    continue
                cells.append(cell(values[name]))
            lines.append(f"| {sname} | {cname} | " + " | ".join(cells) + " |")
    if missing:
        raise ValueError(f"macros not defined: {', '.join(missing)}")
    return lines


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
    expected = table(macros(numbers))
    lines = readme.read_text().split("\n")
    start, end = readme_table(lines)
    if lines[start:end] == expected:
        print(f"{readme}: results table matches {numbers}")
        return 0
    if a.write:
        lines[start:end] = expected
        readme.write_text("\n".join(lines))
        print(f"{readme}: results table written from {numbers}")
        return 0
    print(f"{readme}: results table differs from {numbers}", file=sys.stderr)
    print("expected:\n" + "\n".join(expected), file=sys.stderr)
    print("README:\n" + "\n".join(lines[start:end]), file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
