"""The release build's CJK check (scripts/make_release.py, which the public package does not ship:
these tests skip there). CJK characters are built with chr() so this file itself stays CJK-free."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_release.py"
pytestmark = pytest.mark.skipif(not SCRIPT.exists(), reason="scripts/make_release.py not shipped")
IDEOGRAPHS = chr(0x4E16) + chr(0x754C)
FULL_WIDTH = "x" + chr(0xFF08) + "y" + chr(0xFF09)  # full-width parentheses
CJK_COMMA = chr(0x3001)


def _make_release() -> ModuleType:
    spec = importlib.util.spec_from_file_location("make_release", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cjk_scan_lists_every_cjk_line_and_ignores_other_non_ascii(tmp_path: Path) -> None:
    mr = _make_release()
    (tmp_path / "en.md").write_text("κ, θ_w, ≥ 3 pp, B×D, 12–13 MB — a dash, “quoted”, naïve\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "zh.md").write_text(f"English\n{IDEOGRAPHS}\n{FULL_WIDTH}\n{CJK_COMMA}\n")
    (tmp_path / "blob.bin").write_bytes(b"\0" + IDEOGRAPHS.encode())  # binary: skipped
    assert mr.scan_cjk(tmp_path) == [
        f"sub/zh.md:2: {IDEOGRAPHS}",
        f"sub/zh.md:3: {FULL_WIDTH}",
        f"sub/zh.md:4: {CJK_COMMA}",
    ]
    assert mr.scan(tmp_path) == []  # the internal-identifier scan is unaffected
