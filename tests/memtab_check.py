"""Checks of generated tables in the paper's table style (paper/memtab-style.tex).

`check_memtab` runs `check_tex` (environments balance, every row has as many cells as its tabular
has columns) on a table written with the style's environments and column types, mapped to the
plain ones it parses: `mttabular` and `mttabular*` to `tabular` and `tabular*`, an `N{template}`
column to `r`, an `H{width}{indent}` column to `p{width}`. `MEMTAB_PREAMBLE` loads the style in a
test document, after xcolor and booktabs.
"""

import re
from pathlib import Path

from tests.test_paper_stage4 import _group, check_tex

STYLE = Path(__file__).resolve().parents[1] / "paper" / "memtab-style.tex"
# the style needs array, and the paper's grey of the figures for its muted text
MEMTAB_PREAMBLE = rf"\usepackage{{array}}\definecolor{{cgrey}}{{HTML}}{{6E6E6E}}\input{{{STYLE}}}"


def plain_spec(spec: str) -> str:
    """A column spec with the style's column types replaced by the plain ones they stand for."""
    out, i = [], 0
    while i < len(spec):
        c = spec[i]
        if c in "@>p" and spec[i + 1 : i + 2] == "{":
            _, j = _group(spec, i + 1)
            out.append(spec[i:j])
            i = j
        elif c == "N" and spec[i + 1 : i + 2] == "{":
            _, i = _group(spec, i + 1)  # the template
            out.append("r")
        elif c == "H" and spec[i + 1 : i + 2] == "{":
            width, i = _group(spec, i + 1)
            _, i = _group(spec, i)  # the hanging indent
            out.append(f"p{{{width}}}")
        else:
            out.append(c)
            i += 1
    return "".join(out)


def plain_tabulars(tex: str) -> str:
    """`tex` with every `mttabular` and `mttabular*` as a `tabular` or `tabular*` of plain
    column types."""
    tex = re.sub(r"\\(begin|end)\{mttabular(\*?)\}", r"\\\1{tabular\2}", tex)
    out, last = [], 0
    for m in re.finditer(r"\\begin\{tabular(\*?)\}", tex):
        i = m.end()
        if m.group(1):
            _, i = _group(tex, i)  # the width of a tabular*
        spec, j = _group(tex, i)
        out += [tex[last:i], "{", plain_spec(spec), "}"]
        last = j
    return "".join(out) + tex[last:]


def check_memtab(tex: str) -> None:
    """`check_tex` on a table in the paper's table style."""
    check_tex(plain_tabulars(tex))
