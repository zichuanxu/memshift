"""Shared test configuration.

The `paper` marker tags a test that reads the paper sources: paper/main.tex, paper/sections, the
generated tables and numbers in paper/generated, the figures or the bibliography. The public release
ships no paper/ directory, so these tests skip there and run wherever paper/ is present. The data
they compare against (results/sources.json, audit/, runs/*/analysis) is shipped with the release.
"""

from pathlib import Path

import pytest

PAPER = Path(__file__).resolve().parents[1] / "paper"
NO_PAPER = "paper/ is not in this checkout (the public release ships no paper sources)"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "paper: reads paper/ (LaTeX sources, generated tables, figures)"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if PAPER.is_dir():
        return
    skip = pytest.mark.skip(reason=NO_PAPER)
    for item in items:
        if item.get_closest_marker("paper") is not None:
            item.add_marker(skip, append=False)  # its reason first, before an empty parameter set
