"""The paper figures as drawn from the staged analyses: what each one plots is the data the paper's
numbers come from (asserted against paper/generated/numbers.tex), its marks follow the data, its
text is at least 6.5 pt and on the canvas, and a figure is the same file whatever was drawn before
it. Also the shared marks of `analysis.figstyle`."""

import copy
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from memshift.analysis import concept_figure as cf
from memshift.analysis import ex_e1_paper as xp
from memshift.analysis import ex_review_paper as rp
from memshift.analysis import figstyle as fs
from memshift.analysis import figures as figs
from memshift.analysis import round3_figures as rf
from memshift.analysis.e3 import COST_CEILING
from memshift.analysis.ex_e1_cross import cost_model
from memshift.analysis.paper_data import get, load_stage4, native_results
from memshift.analysis.paper_numbers import ci, pct, pp

ROOT = Path(__file__).resolve().parents[1]
SOURCES = json.loads((ROOT / "results/sources.json").read_text())
E2 = figs._analysis(ROOT, SOURCES, "e2", "e2_metrics.json")
E3 = figs._analysis(ROOT, SOURCES, "e3", "e3_metrics.json")
NAT = native_results(load_stage4(ROOT, SOURCES))
R3 = rf.load(ROOT, SOURCES)
ACCT = rp.load(ROOT, SOURCES)
E1 = xp.load(ROOT, SOURCES)
CONCEPT = cf.load(ROOT, SOURCES)
DRAW: dict[str, Callable[[Path], Path]] = {  # figure -> draws it from the staged data
    "fig_concept": lambda out: cf.fig_concept(CONCEPT, out),
    "fig_calibration": lambda out: figs.fig_calibration(E2 or {}, out),
    "fig_zero_cost": lambda out: figs.fig_zero_cost(E3 or {}, out),
    "fig_native_systems": lambda out: figs.fig_native_systems(NAT, out),
    "fig_classsplit": lambda out: xp.fig_classsplit(E1, out),
    "fig_account": lambda out: rp.fig_account(ACCT, out),
    "fig_writers": lambda out: rf.fig_writers(R3, out),
    "fig_reader": lambda out: rf.fig_reader(R3, out),
    "fig_misses": lambda out: rf.fig_misses(R3, out),
}
TIMES = fs.serif_fonts()[0] == "Times New Roman"  # the layout was set in it
SMALL_CAPS = {"OLD", "EM"}  # HoldMem's small capitals: as tall as the lower case of its name
BRANCH = {"src": "Src", "def": "Def", "orphan": "Orphan"}
CLS = {"definition": "Definition", "other": "Other"}


def macros() -> dict[str, str]:
    text = (ROOT / "paper/generated/numbers.tex").read_text()
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", text, flags=re.M))


def fmt_pp(e: dict[str, Any]) -> str:
    return ci(e, pp)


def values(macro: str) -> list[float]:
    """'$+$16.4\\,pp [$+$9.8, $+$23.0]' -> [16.4, 9.8, 23.0]"""
    s = macro.replace("$+$", "+").replace("$-$", "-")
    return [float(v) for v in re.findall(r"[+-]?\d+\.\d+", s)]


class Drawn:
    """What a generator drew, read where it saves the figure (inside its style): the figure, its
    texts, the texts under 6.5 pt and those that leave the canvas, and the saved file."""

    def __init__(self, fig: Any, path: Path) -> None:
        from matplotlib.text import Text

        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        w, h = fig.bbox.width, fig.bbox.height
        shown = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
        self.fig, self.path = fig, path
        self.texts = [t.get_text() for t in shown]
        self.small = [(t.get_text(), t.get_fontsize()) for t in shown
                      if t.get_fontsize() < 6.5 and t.get_text() not in SMALL_CAPS]  # fmt: skip
        self.outside: list[str] = []
        self.boxes: dict[str, list[Any]] = {}
        for t in shown:
            bb = t.get_window_extent(r)
            self.boxes.setdefault(t.get_text(), []).append(bb)
            if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > w + 0.5 or bb.y1 > h + 0.5:
                self.outside.append(t.get_text())


def drawn(monkeypatch: pytest.MonkeyPatch, name: str, out: Path) -> Drawn:
    seen: list[Drawn] = []
    save = fs.save_pdf

    def capture(fig: Any, path: Path) -> Path:
        seen.append(Drawn(fig, path))
        return save(fig, path)

    monkeypatch.setattr(fs, "save_pdf", capture)
    path = DRAW[name](out)
    monkeypatch.setattr(fs, "save_pdf", save)
    assert len(seen) == 1 and seen[0].path == path and path.name == f"{name}.pdf"
    return seen[0]


# ------------------------------------------------------------------------- every figure
@pytest.mark.parametrize("name", sorted(DRAW))
def test_every_figure_is_legible_vector_truetype_and_without_a_date(name, monkeypatch, tmp_path):
    d = drawn(monkeypatch, name, tmp_path)
    assert not d.small, d.small
    if TIMES:  # with another face the widths differ a little
        assert not d.outside, d.outside
    data = d.path.read_bytes()
    assert data.startswith(b"%PDF") and b"/CreationDate" not in data
    assert b"/FontFile2" in data and b"/FontFile3" not in data  # TrueType embedded, no Type 1
    assert re.search(rb"/FontFile\s", data) is None
    if name != "fig_concept":  # its icons are raster images
        assert b"/Subtype /Image" not in data


def test_a_figure_is_the_same_file_whatever_was_drawn_before(tmp_path):
    import matplotlib

    first = DRAW["fig_reader"](tmp_path).read_bytes()
    (tmp_path / "again").mkdir()
    with matplotlib.rc_context({"text.usetex": True, "font.size": 20, "axes.grid": True,
                                "figure.constrained_layout.use": True}):  # fmt: skip
        again = DRAW["fig_reader"](tmp_path / "again").read_bytes()
        assert matplotlib.rcParams["font.size"] == 20  # the caller's style is left alone
    assert first == again


def test_the_serif_face_falls_back_to_the_bundled_stix(monkeypatch):
    assert fs.serif_fonts()[-1] == "STIXGeneral"
    monkeypatch.setattr(fs, "SERIF", ("No Such Face",))
    assert fs.serif_fonts() == ["STIXGeneral"]


# ------------------------------------------------------------------------- calibration
@pytest.mark.paper
def test_calibration_plots_the_pooled_items_slope_and_auroc_of_the_text(monkeypatch, tmp_path):
    from matplotlib.patches import Circle

    assert E2 is not None
    mac = macros()
    bins = figs.calibration_bins(E2)
    assert f"{sum(b['n'] for b in bins):,}" == mac["ETwoPathThreeN"]
    assert E2["explore_pooled_curve"]["n_items"] == sum(b["n"] for b in bins)
    sl, au = E2["path3"]["pooled_slope"], E2["path3"]["pooled_auroc"]
    assert (
        f"{sl['est']:.2f} [{sl['lo90']:.2f}, {sl['hi90']:.2f}]"
        == mac["ETwoExWriterAgreePooledSlope"]
    )
    assert f"{au['est']:.3f}" == mac["ETwoExWriterAgreePooledAurocEst"]
    d = drawn(monkeypatch, "fig_calibration", tmp_path)
    # one point per bin at (mean agreement, survival) in (a), and again in (b) from 80%
    pa = figs._CalPanel(*figs.CAL_A, 0.0, 0.0)
    pb = figs._CalPanel(*figs.CAL_B, *figs.CAL_ZOOM)
    want = sorted((p.px(b["agreement"]), p.py(b["survival"]))
                  for p in (pa, pb) for b in bins if b["lo"] >= p.lo - 1e-9)  # fmt: skip
    got = sorted(c.center for c in d.fig.axes[0].patches
                 if isinstance(c, Circle) and c.get_radius() == figs.CAL_DOT_R)  # fmt: skip
    assert len(got) == len(want) == len(bins) + sum(b["lo"] >= figs.CAL_ZOOM[0] for b in bins)
    for (x, y), (wx, wy) in zip(got, want, strict=True):
        assert (x, y) == pytest.approx((wx, wy))
    for b in bins:  # the items per bin, in the strips
        assert f"{b['n']:,}" in d.texts
    for e in (sl, au):
        assert f"{e['est']:.2f}" in d.texts and f"[{e['lo90']:.2f}, {e['hi90']:.2f}]" in d.texts
    total = sum(b["n"] for b in bins)
    top = sum(b["n"] for b in bins if b["lo"] >= figs.CAL_ZOOM[0])
    assert f"{100 * top / total:.1f}% of the items" in d.texts
    assert f"{top:,} of {total:,}" in d.texts


# --------------------------------------------------------------------------- zero cost
@pytest.mark.paper
def test_zero_cost_prints_the_gap_and_the_cost_ratio_of_the_text(monkeypatch, tmp_path):
    assert E3 is not None
    mac = macros()
    gap = E3["horizons"]["12"]["zero_minus_holdmem"]
    assert (
        f"$-${-100 * gap['est']:.1f}\\,pp [$-${-100 * gap['lo']:.1f}, $-${-100 * gap['hi']:.1f}]"
        == mac["EThreeZeroMinusHoldTwelve"]
    )
    ratio = cost_model(E3)["horizons"]["12"]["holdmem_D1B1_source"]["total_over_zero"]
    assert f"{ratio:.2f}" == mac["ExCostModelMainTwelveHoldRatioZero"]
    d = drawn(monkeypatch, "fig_zero_cost", tmp_path)
    assert f"{-100 * gap['est']:.1f} pp" in d.texts
    assert f"[{-100 * gap['hi']:.1f}, {-100 * gap['lo']:.1f}]" in d.texts
    assert f"{ratio:.2f}\u00d7" in d.texts  # the generator's ratio is the cost model's
    cost = figs.zero_costs(E3)
    bars = cost["systems"]
    assert list(bars) == list(figs.ZERO_COST_ROWS)  # oracle context has no cost
    for c in bars.values():  # each bar: serving, then ingestion and maintenance, its total
        assert c["serve"] + c["ingest"] + c["maintain"] == pytest.approx(c["total"])
        assert f"{c['total']:,.0f}" in d.texts
    assert bars[figs.HOLD]["total"] / bars["zero"]["total"] == pytest.approx(ratio)
    assert cost["ceiling"] == pytest.approx(COST_CEILING * cost["reread"])
    for v in (cost["ceiling"], cost["reread"]):
        assert f"{v / 1e6:.2f}M" in d.texts
    data = d.path.read_bytes()
    box = re.search(rb"/MediaBox \[\s*0 0 ([\d.]+) ([\d.]+)\s*\]", data)
    assert box and float(box[1]) == pytest.approx(72 * fs.TEXT)


# ------------------------------------------------------------------------ native systems
NATIVE_MACROS = {"mem0": "Mem", "amem": "Amem", "memos": "Memos", "lightrag": "LightRag",
                 "hipporag": "HippoRag", "cognee": "Cognee"}  # fmt: skip


@pytest.mark.paper
def test_native_systems_count_inside_as_the_text_and_carry_their_labels(monkeypatch, tmp_path):
    mac = macros()
    panels = figs.native_panels(NAT)
    assert [p["system"] for p in panels] == list(NATIVE_MACROS) and all(p["staged"] for p in panels)
    figs.native_groups(panels)  # every cell's systems share its estimate
    keys = [(b, m) for b, _ in figs.NATIVE_BRANCHES for m in figs.NATIVE_METRICS]
    for p in panels:  # inside: the native estimate within the cell's 95% interval
        for k in keys:
            v = p["values"][k]
            assert v["inside"] == (v["pred"]["lo"] <= v["native"]["est"] <= v["pred"]["hi"])
    inside = {k: sum(bool(p["values"][k]["inside"]) for p in panels) for k in keys}
    total, pairs = sum(inside.values()), len(keys) * len(panels)
    assert f"{total} of {pairs}" == mac["NatNThreeInside"] == mac["NatExNullInsideOf"]
    assert f"{100 * total / pairs:.1f}\\%" == mac["NatNThreeShare"]
    for p in panels:
        stem = NATIVE_MACROS[p["system"]]
        n = sum(bool(p["values"][k]["inside"]) for k in keys)
        assert f"{n} of {len(keys)}" == mac[f"Nat{stem}ExNullInsideOf"], p["system"]
        assert p["inferred"] == mac[f"Nat{stem}Documented"] == f"({p['cell'][9]},{p['cell'][11]})"
    d = drawn(monkeypatch, "fig_native_systems", tmp_path)
    counts = [t for t in d.texts if t.endswith(" inside")]
    assert sorted(counts) == sorted(f"{inside[k]} of {len(panels)} inside" for k in keys)
    for p in panels:
        assert d.texts.count(p["label"]) == len(figs.NATIVE_METRICS)


def test_native_bars_start_at_zero_at_the_estimate_with_their_interval(monkeypatch, tmp_path):
    from matplotlib.colors import to_hex
    from matplotlib.patches import Rectangle

    panels = figs.native_panels(NAT)
    d = drawn(monkeypatch, "fig_native_systems", tmp_path)
    axes = [ax for ax in d.fig.axes if ax.get_xlim() == (0, figs.NATIVE_XMAX)]
    assert len(axes) == len(figs.NATIVE_BRANCHES) * len(figs.NATIVE_METRICS)
    for ax, (m, (b, _)) in zip(axes, [(m, b) for m in figs.NATIVE_METRICS
                                      for b in figs.NATIVE_BRANCHES], strict=True):  # fmt: skip
        bars = [r for r in ax.patches if isinstance(r, Rectangle)
                and r.get_height() == pytest.approx(figs.NATIVE_BAR_T)]  # fmt: skip
        ests = [100 * p["values"][b, m]["native"]["est"] for p in panels]
        assert [r.get_x() for r in bars] == [0] * len(panels)
        assert [r.get_width() for r in bars] == pytest.approx(ests)
        fills = [to_hex(r.get_facecolor()) for r in bars]  # dark inside the interval
        assert fills == [to_hex((figs.NATIVE_INSIDE if p["values"][b, m]["inside"]
                                 else figs.NATIVE_OUTSIDE)["fill"]) for p in panels]  # fmt: skip
        ends = {round(x, 9) for line in ax.lines for x in line.get_xdata()}
        for p in panels:
            e = p["values"][b, m]["native"]
            assert {round(100 * e["lo"], 9), round(100 * e["hi"], 9)} <= ends


def test_native_groups_join_a_cells_systems_and_refuse_inconsistent_ones():
    panels = figs.native_panels(NAT)
    groups = figs.native_groups(panels)
    assert [(c, len(ks)) for c, ks in groups] == [("holdmem_D0B0", 3), ("holdmem_D0B1_cited", 3)]
    other = copy.deepcopy(panels)
    other[1]["values"]["src", "Update"]["pred"] = {"est": 0.1, "lo": 0.0, "hi": 0.2}
    with pytest.raises(ValueError, match="differ"):
        figs.native_groups(other)
    split = copy.deepcopy(panels)
    split[1]["cell"] = "holdmem_D0B1_cited"
    with pytest.raises(ValueError, match="contiguous"):
        figs.native_groups(split)
    none = figs.native_panels(None)
    assert figs.native_groups(none) == [(None, list(range(len(none))))]
    assert (figs.native_cell_style("holdmem_D1B1_source") or {}).get("short") == "(1,1)"


# ---------------------------------------------------------------------------- writers
@pytest.mark.paper
def test_writers_plot_the_block_h_estimates_and_name_each_writer(monkeypatch, tmp_path):
    from matplotlib.patches import Ellipse, Rectangle

    mac = macros()
    data = rf.writer_points(R3)
    defw = {(p["writer"], p["shift"]): p["y"] for p in data["defwriters"]}
    for who, tag in ((rf.OPUS, "Opus"), (rf.SONNET, "Sonnet")):
        for b in ("def", "src"):
            got = [round(100 * defw[who, b][k], 1) for k in ("est", "lo", "hi")]
            assert got == values(mac[f"RThreeH{tag}{BRANCH[b]}DefinitionDoneBxD"]), (who, b)
    programs = data["programs"]
    assert len(programs) == int(mac["RThreeGBDefSets"]) and all(p["x"] and p["y"] for p in programs)
    assert len({p["writer"] for p in programs}) == int(mac["RThreeGBWriters"])
    ref = data["reference"]
    assert ref is not None and fmt_pp(ref) == mac["ExAcctMainDefDefinitionBxD"]
    assert set(defw) == {(w, b) for w in (rf.QWEN, rf.OPUS, rf.SONNET) for b in ("def", "src")}
    d = drawn(monkeypatch, "fig_writers", tmp_path)
    sets = {p["writer"] for p in programs}
    for who in sets | {w for w, _ in defw}:  # once in each panel that draws it
        label = fs.model_style(who)[0]
        assert d.texts.count(label) == (who in sets) + any(w == who for w, _ in defw), who
    assert d.texts.count("exact recomputation, 95% interval") == 1
    assert f"{100 * ref['est']:+.1f}" in d.texts
    axa, axb = d.fig.axes[:2]
    # (a): one square per program set, at its B x D
    squares = [r for r in axa.patches if isinstance(r, Rectangle) and r.get_zorder() == 5.0]
    assert sorted(r.get_y() + r.get_height() / 2 for r in squares) == pytest.approx(
        sorted(100 * p["y"]["est"] for p in programs)
    )
    # (b): every mark within 3.0 pp of its values (a nudge stays small), every interval end
    # exact
    marks = {"def": [(r.get_x() + r.get_width() / 2, r.get_y() + r.get_height() / 2)
                     for r in axb.patches if isinstance(r, Rectangle) and r.get_zorder() == 5.2],
             "src": [e.center for e in axb.patches
                     if isinstance(e, Ellipse) and e.get_zorder() > 5.25]}  # fmt: skip
    ends = {round(v, 9) for line in axb.lines for v in (*line.get_xdata(), *line.get_ydata())}
    for (_, shift), q in {(p["writer"], p["shift"]): p for p in data["defwriters"]}.items():
        x, y = 100 * q["x"]["est"], 100 * q["y"]["est"]
        near = min(marks[shift], key=lambda m: abs(m[0] - x) + abs(m[1] - y))
        marks[shift].remove(near)
        assert abs(near[0] - x) <= 3.05 and abs(near[1] - y) <= 3.05, (q, near)
        assert {round(100 * q[a][k], 9) for a in ("x", "y") for k in ("lo", "hi")} <= ends
    assert not marks["def"] and not marks["src"]
    if TIMES:  # the names keep clear of one another
        names = [bb for who in sets for bb in d.boxes[fs.model_style(who)[0]]]
        for i, a in enumerate(names):
            assert not any(a.overlaps(b) for b in names[i + 1 :])


# ---------------------------------------------------------------------------- concept
@pytest.mark.paper
def test_concept_prints_the_cells_and_key_effects_of_the_text(monkeypatch, tmp_path):
    from matplotlib.colors import to_hex
    from matplotlib.patches import Rectangle

    mac = macros()
    data = cf.concept_cells(CONCEPT)
    cells = data["cells"]
    tag = {0: "zero", 1: "one"}
    stem = "ExDetMainDefDefinition"
    assert set(cells) == {(d, b) for d in (0, 1) for b in (0, 1)}
    for (d, b), c in cells.items():
        assert ci(c["U"]) == mac[f"{stem}D{tag[d]}B{tag[b]}Update"], (d, b)
        if b:  # m and r of the key's cells, which give Update by Eq. (1)
            assert pct(c["m"]) == mac[f"{stem}D{tag[d]}BoneMissed"].split(" [")[0]
            assert pct(c["r"]) == mac[f"{stem}D{tag[d]}BoneRepair"].split(" [")[0]
            assert (1 - c["m"]) * c["r"] + c["m"] * c["a"] == pytest.approx(c["U"]["est"])
    for d in (0, 1):  # the key's effect: the difference of the two cells of its column
        assert ci(data["key"][d], pp) == mac[f"ExClsMainDefDefinitionBD{tag[d]}"]
        diff = cells[d, 1]["U"]["est"] - cells[d, 0]["U"]["est"]
        assert data["key"][d]["est"] == pytest.approx(diff, abs=1e-12)
    assert fmt_pp(data["bxd"]) == mac["ExAcctMainDefDefinitionBxD"]
    assert data["bxd"]["est"] == pytest.approx(data["key"][1]["est"] - data["key"][0]["est"])
    assert f"{data['items']:,}" == mac[f"{stem}Items"] and str(data["worlds"]) == mac["TestWorlds"]
    d = drawn(monkeypatch, "fig_concept", tmp_path)
    for c in cells.values():
        assert cf._printed(pct(c["U"]["est"])) in d.texts
    for d_ in (0, 1):
        assert cf._printed(pp(data["key"][d_]["est"])) in d.texts
    assert rf"$B{{\times}}D$ = {cf._printed(pp(data['bxd']['est']))}" in d.texts
    assert any(t.endswith(f"({data['items']:,} item-times)") for t in d.texts)
    # each bar holds every changed item, and its right share is the cell's Update
    ax = d.fig.axes[0]
    rects = [r for r in ax.patches if isinstance(r, Rectangle)]
    right = {to_hex(cf.SLA["mark"]), to_hex(cf.EXA["mark"])}
    shares = []
    for frame in (r for r in rects if r.get_zorder() == 7):
        x0, x1 = frame.get_x(), frame.get_x() + frame.get_width()
        segs = [r for r in rects if r.get_zorder() == 6 and r.get_y() == frame.get_y()
                and x0 - 1e-9 <= r.get_x() <= x1]  # fmt: skip
        assert sum(r.get_width() for r in segs) == pytest.approx(frame.get_width())
        shares.append(sum(r.get_width() for r in segs if to_hex(r.get_facecolor()) in right)
                      / frame.get_width())  # fmt: skip
    assert sorted(shares) == pytest.approx(sorted(c["U"]["est"] for c in cells.values()))


@pytest.mark.paper
def test_figure_one_is_the_generated_pdf():
    src = (ROOT / "src/memshift/analysis/figures.py").read_text()
    assert "concept_figure.fig_concept(concept_figure.load(a.root, sources), a.out)" in src
    intro = (ROOT / "paper/sections/intro.tex").read_text()
    assert r"\includegraphics{figures/fig_concept.pdf}" in intro
    assert not (ROOT / "paper/figures/fig_concept.tex").exists()  # the TikZ drawing is gone
    assert (ROOT / "paper/figures/fig_concept.pdf").read_bytes().startswith(b"%PDF")


def test_concept_figure_draws_without_its_stages_or_icons(tmp_path):
    outs = {}
    for tag, inp in (("full", CONCEPT), ("plain", {**CONCEPT, "icons": None}),
                     ("empty", {**cf.load(ROOT, {}), "icons": None})):  # fmt: skip
        (tmp_path / tag).mkdir()
        outs[tag] = cf.fig_concept(inp, tmp_path / tag).read_bytes()
        assert outs[tag].startswith(b"%PDF")
    assert (b"/Subtype /Image" in outs["full"]) == (ROOT / cf.ICONS).is_dir()
    assert b"/Subtype /Image" not in outs["plain"] + outs["empty"]
    assert outs["plain"] != outs["empty"]
    assert cf.concept_cells({}) == {"cells": {}, "key": {0: None, 1: None}, "bxd": None,
                                    "items": None, "worlds": None}  # fmt: skip
    bad = copy.deepcopy(CONCEPT["event"])
    cell = bad["main"]["terms"]["def"]["definition"]["source"]["cells"]["D0B1"]["latest_change"]
    cell["U"]["est"] += 0.01  # Update no longer (1 - m) r + m a
    with pytest.raises(ValueError, match="Update"):
        cf.concept_cells({**CONCEPT, "event": bad})


# ----------------------------------------------------------------------------- reader
@pytest.mark.paper
def test_reader_plots_the_paired_contrasts_and_prints_the_reader_parts(monkeypatch, tmp_path):
    mac = macros()
    rows = rf.reader_rows(R3)
    for r in rows:
        tag = f"{BRANCH[r['branch']]}{CLS[r['cls']]}"
        got = [round(100 * r["sonnet"][k], 1) for k in ("est", "lo", "hi")]
        assert got == values(mac[f"RThreeY{tag}DoneBxD"]), tag
        for key, name in (("store", "StorePart"), ("qwen", "BxD")):
            got = [round(100 * r[key][k], 1) for k in ("est", "lo", "hi")]
            assert got == values(mac[f"ExRThreeSRMain{tag}{name}"]), (tag, key)
    cell = next(r for r in rows if (r["cls"], r["branch"]) == rf.READER_GAP_CELL)
    parts = {k: round(100 * (cell[k]["est"] - cell["store"]["est"]), 1) for k in ("qwen", "sonnet")}
    assert parts["qwen"] == values(mac["ExRThreeSRMainDefOtherReaderPart"])[0]
    assert parts["sonnet"] == values(mac["RThreeYSRMainDefOtherReaderPart"])[0]
    d = drawn(monkeypatch, "fig_reader", tmp_path)
    for v in parts.values():
        assert f"{v:+.1f}".replace("-", "\u2212") in d.texts


# ----------------------------------------------------------------------------- misses
@pytest.mark.paper
def test_misses_stack_the_counts_of_the_text(monkeypatch, tmp_path):
    mac = macros()
    rows = rf.miss_rows(R3)
    suffix = {"a": "ACount", "b_old": "BExistingCount", "b_new": "BNewCount",
              "c_left": "CLeftCount"}  # fmt: skip
    stage = {"main": "Main", "rep": "Rep"}
    uncoverable = [k for k, _, _, covered in rf.MISS_CAUSES if not covered]
    for r in rows:
        stem = f"ExMissClass{stage[r['stage']]}{BRANCH[r['branch']]}{CLS[r['cls']]}"
        assert r["missed"] == int(mac[stem + "Items"]), stem
        for k, s in suffix.items():
            assert round(r["shares"][k] * r["missed"]) == int(mac[stem + s]), (stem, k)
        assert r["uncoverable"] == pytest.approx(sum(r["shares"][k] for k in uncoverable))
    d = drawn(monkeypatch, "fig_misses", tmp_path)
    for r in rows:  # n and the uncoverable share beside each bar
        assert str(r["missed"]) in d.texts and f"{100 * r['uncoverable']:.0f}" in d.texts
    widths = sorted(round(p.get_width(), 9) for ax in d.fig.axes[:2] for p in ax.patches
                    if p.get_width() > 0)  # fmt: skip
    want = sorted(round(100 * s, 9) for r in rows for s in r["shares"].values() if s > 0)
    assert widths == want


# ---------------------------------------------------------------------------- account
@pytest.mark.paper
def test_account_plots_every_cell_and_the_fit_of_the_text(monkeypatch, tmp_path):
    mac = macros()
    fit = ACCT["acct"]["explore_account_fit"]["two_class"]
    cells = rp._fit_cells(fit)
    fitted = sum(not held_out for _, held_out in cells)
    assert len(cells) == 30 and fitted == int(mac["ExAcctFitTwoCells"])
    assert mac["ExAcctFitTwoSlopeEst"] == f"{fit['slope']['est']:.2f}"
    assert mac["ExAcctFitTwoSpearmanEst"] == f"{fit['spearman']['est']:.2f}"
    for c, tag in (("definition", "Definition"), ("other", "Other")):
        assert mac[f"ExAcctFitTwo{tag}SpearmanEst"] == rp.snum(fit["spearman_by_class"][c]["est"])
    wtag = {w: t for w, t, _ in rp.WRITER_TAGS}
    for c, _ in cells:  # every marker against the per-cell macros of tab_writers
        stem = f"ExAcct{wtag[c['writer']]}{BRANCH[c['branch']]}{CLS[c['class']]}"
        assert mac[f"{stem}BxDEst"] == pp(c["BxD"]) and mac[f"{stem}GapEst"] == pp(c["gap"]), stem
    d = drawn(monkeypatch, "fig_account", tmp_path)
    ax = d.fig.axes[0]
    marks = [(x, y) for line in ax.lines if line.get_marker() not in (None, "", "None")
             for x, y in zip(line.get_xdata(), line.get_ydata(), strict=True)]  # fmt: skip
    assert sorted(marks) == pytest.approx(sorted((100 * c["gap"], 100 * c["BxD"])
                                                 for c, _ in cells))  # fmt: skip
    assert any(t.startswith(f"least-squares fit over the {fitted} cells") for t in d.texts)


# ------------------------------------------------------------------------- class split
@pytest.mark.paper
def test_classsplit_plots_the_estimates_and_shares_of_the_table(monkeypatch, tmp_path):
    mac = macros()
    ktag = {"B_effect_update_D0": "BDzero", "B_effect_update_D1": "BDone",
            "BxD_interaction_update": "BxD"}  # fmt: skip
    for stage, stag in (("main", "Main"), ("rep", "Rep")):
        for b, _ in xp.CLASSSPLIT_PANELS:
            for cls, _ in xp.CLASSSPLIT_CLASSES:
                for key, *_ in xp.CLASSSPLIT_STYLES:
                    e = xp._classsplit_node(E1, stage, b, cls, key)
                    name = f"ExCls{stag}{BRANCH[b]}{CLS[cls]}{ktag[key]}"
                    assert e is not None and mac[name] == xp.fmt(e, pp), name
                share = get(E1.get("main"), "explore_class_mix", b, "classes", cls, "share")
                assert mac[f"ExMixMain{BRANCH[b]}{CLS[cls]}Share"] == pct(share)
    assert str(get(E1.get("main"), "n_worlds")) == mac["TestWorlds"]
    assert str(get(E1.get("rep"), "n_worlds")) == mac["RepWorlds"]
    d = drawn(monkeypatch, "fig_classsplit", tmp_path)
    for b, _ in xp.CLASSSPLIT_PANELS:  # B x D of the definition items over its bar
        e = xp._classsplit_node(E1, "main", b, "definition", "BxD_interaction_update")
        assert e and f"{100 * e['est']:+.1f}".replace("-", "\u2212") in d.texts


# ------------------------------------------------------------------------ shared marks
def test_interval_line_has_exact_ends_and_caps_and_bars_start_at_zero():
    with fs.printed() as plt:
        fig, ax = plt.subplots()
        lines = fs.interval_line(ax, 1.0, 2.0, 8.0, fs.INK, 0.2)
        bar = fs.bar_from_zero(ax, 1.0, 0.4, -3.0, fs.BLUE)
        plt.close(fig)
    stem, *caps = lines
    assert list(stem.get_xdata()) == [1.0, 1.0] and list(stem.get_ydata()) == [2.0, 8.0]
    assert [list(c.get_ydata()) for c in caps] == [[2.0, 2.0], [8.0, 8.0]]
    assert all(list(c.get_xdata()) == pytest.approx([0.9, 1.1]) for c in caps)
    assert all(line.get_solid_capstyle() == "butt" for line in lines)
    assert (bar.get_x(), bar.get_y(), bar.get_width(), bar.get_height()) == pytest.approx(
        (0.8, 0.0, 0.4, -3.0)
    )


def test_colour_helpers_and_the_rounded_path():
    assert fs.tint("#E69F00", 1.0) == "#E69F00" and fs.tint("#E69F00", 0.0) == "#FFFFFF"
    assert fs.mix("#000000", "#FFFFFF", 0.5) == "#808080"
    straight = fs.rounded_path([(0, 0), (1, 0)], 1.0)
    assert straight.vertices.tolist() == [[0, 0], [1, 0]]
    corner = fs.rounded_path([(0, 0), (2, 0), (2, 2)], 0.5)
    assert corner.vertices.tolist() == [[0, 0], [1.5, 0], [2, 0], [2, 0.5], [2, 2]]
