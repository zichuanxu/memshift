"""The LaTeX files of the paper's text, for the tests that read it.

`paper_sources` follows the `\\input` commands from paper/main.tex into paper/sections and
paper/appendix, so a file the paper does not input (for example an appendix section left out of
the build) is not part of the text the tests read. Commented-out inputs are skipped.
"""

import re
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1] / "paper"
INPUT = re.compile(r"\\input\{((?:sections|appendix)/[\w-]+)\}")
COMMENT = re.compile(r"(?<!\\)%.*")


def paper_sources() -> list[Path]:
    """The section and appendix files the paper inputs, in the order it inputs them; none in a
    checkout without paper/ (the public release), where some tests still collect over them."""
    out: list[Path] = []
    if not (PAPER / "main.tex").exists():
        return out

    def follow(path: Path) -> None:
        for line in path.read_text().splitlines():
            for name in INPUT.findall(COMMENT.sub("", line)):
                child = PAPER / f"{name}.tex"
                if child not in out:
                    out.append(child)
                    follow(child)

    follow(PAPER / "main.tex")
    return out


def paper_text() -> str:
    """The text of `paper_sources`, one file after another."""
    return "\n".join(p.read_text() for p in paper_sources())
