"""The check of the README results tables (scripts/readme_results.py): their cells are the macros
of the paper's Table 4 (a and b), and it fails on a table or a text number that the macros do not
give."""

import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "readme_results.py"


def _readme_results() -> ModuleType:
    spec = importlib.util.spec_from_file_location("readme_results", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _panel_b_cells() -> list[list[str]]:
    """Per body row of panel (b) of tab:main: the macro of each model's cell, in column order."""
    text = (ROOT / "paper" / "sections" / "results.tex").read_text()
    label = text.index(r"\label{tab:main}")
    table = text[label : text.index(r"\end{table*}", label)]
    panel = table[table.index("(b) Contrast") :]
    body = panel[panel.index(r"\midrule") + len(r"\midrule") : panel.index(r"\bottomrule")]
    rows = [re.findall(r"\\mtcipair\\([A-Za-z]+)", row) for row in body.split(r"\\")]
    return [row for row in rows if row]


def _tab_main_cells() -> list[list[tuple[str, str | None]]]:
    """Per body row of tab:main in paper/sections/results.tex: (kind, macro) of each cell after the
    shift and class, with kind pct (\\mspct), ci (\\mtcipair, an estimate and its interval) or dash
    (\\mtna, a --- for a class the shift leaves unchanged)."""
    text = (ROOT / "paper" / "sections" / "results.tex").read_text()
    label = text.index(r"\label{tab:main}")
    table = text[text.rindex(r"\begin{table*}", 0, label) : text.index(r"\end{table*}", label)]
    body = table[table.index(r"\midrule") + len(r"\midrule") : table.index(r"\bottomrule")]
    cell = re.compile(r"\\mspct\\([A-Za-z]+)|\\mtcipair\\([A-Za-z]+)|\\mtna\b")
    rows = []
    for row in body.split(r"\\"):
        cells: list[tuple[str, str | None]] = []
        for m in cell.finditer(row):
            if m.group(1):
                cells.append(("pct", m.group(1)))
            elif m.group(2):
                cells.append(("ci", m.group(2)))
            else:
                cells.append(("dash", None))
        if cells:
            rows.append(cells)
    return rows


@pytest.mark.paper
def test_cells_are_the_macros_of_table_4() -> None:
    rr = _readme_results()
    assert [cells for _, _, cells in rr.rows()] == _tab_main_cells()


@pytest.mark.paper
def test_model_cells_are_the_macros_of_table_4b() -> None:
    rr = _readme_results()
    assert [names for _, names in rr.model_rows()] == _panel_b_cells()


@pytest.mark.paper
def test_release_readme_matches_the_paper_numbers() -> None:
    rr = _readme_results()
    assert rr.main(["--numbers", str(ROOT / "paper" / "generated" / "numbers.tex")]) == 0


def test_mismatched_table_or_text_number_fails(tmp_path: Path) -> None:
    rr = _readme_results()
    values = {}
    for _, _, cells in rr.rows():
        for kind, name in cells:
            if name is not None:
                values[name] = "12.5\\%" if kind == "pct" else "$+$1.0\\,pp [$-$0.5, $+$2.5]"
    for _, names in rr.model_rows():
        values |= dict.fromkeys(names, "$+$3.0\\,pp [$+$1.5, $+$4.5]")
    values["ExDetMainSourceMissedMax"] = "0.6\\%"
    numbers = tmp_path / "numbers.tex"
    numbers.write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in values.items()))
    assert rr.model_table(values)[2].startswith(
        "| B effect on Update, source revision | +3.0 [+1.5"
    )
    table = rr.table(values)
    assert rr.DASH in table[5]  # redefinition, other items: no change caused by the shift
    assert "| source revision | definition | 12.5 |" in table[2]
    readme = tmp_path / "README.md"
    text = "The source key missed at most 0.6% of a class's\nchanged items, so the identity holds."
    models = rr.model_table(values)
    readme.write_text("\n".join(["## Main results", "", *models, "", *table, "", text, ""]))
    assert rr.main(["--numbers", str(numbers), "--readme", str(readme)]) == 0
    readme.write_text(
        "\n".join(["## Main results", "", *models, "", *table, "", text.replace("0.6", "0.7")])
    )
    assert rr.main(["--numbers", str(numbers), "--readme", str(readme)]) == 1
    wrong = [table[0], table[1], table[2].replace("12.5", "12.4"), *table[3:]]
    readme.write_text("\n".join(["## Main results", "", *models, "", *wrong, "", text]))
    assert rr.main(["--numbers", str(numbers), "--readme", str(readme)]) == 1
    assert rr.main(["--numbers", str(numbers), "--readme", str(readme), "--write"]) == 0
    assert rr.main(["--numbers", str(numbers), "--readme", str(readme)]) == 0
