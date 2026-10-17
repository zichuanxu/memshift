"""Figures of the round-3 results and of the registration record (regenerated from the staged
analyses named in results/sources.json; nothing here calls a model or reads a raw run).

fig_writers     Section 4.2: (a) per program set of block G-b, B x D at D_m on definition items
                after S_def (`round3_registered G`, staged as round3 g_b.json) against how often
                the set's programs computed the right value at the fork, before any shift
                (exploratory, ex_round3 writers.json), with the authored derivation's B x D on the
                same items as reference (main study); (b) per definition-item writer (Qwen3.5-9B,
                Claude Sonnet 5.5, Claude Opus 5.5) B x D with the authored derivation against
                r_0, both on the same worlds, after S_src and S_def (writers.json)
fig_reader      appendix: the store part of B x D on Update as a bar, and an arrow per reader of
                the same stores from it to B x D with that reader (y.json and the committed
                reader's split, ex_round3 store_reader.json)
fig_misses      appendix: why the cited-cluster key misses changed item-times, by shift and item
                class, main study and replication (ex_misses.json)
fig_registered  appendix, two pages (fig_registered.pdf, fig_registered_2.pdf): every registered
                prediction with an estimate as interval bars, one sub-row per shift or arm, with
                its registered bar, one panel and scale per question (v1 to v4b; the stages of
                the main study, blocks R, C, M, L, the controls and round 3)
tab_registered_tests (paper/generated, written with the counts when fig_registered is drawn):
                every other registered prediction with a verdict (judged per switch, system,
                family or reader, or without an estimate on the figure's scales) and the four
                diagnostic tests per system; the counts the paper quotes cover the figure's rows
                (Reg*), the table's predictions (RegTab*) and both (RegAll*)

Each figure has a data function (`writer_points`, `reader_rows`, `miss_rows`,
`registered_rows`, `registered_tests`) that reads the inputs of `load` and returns plain records;
an input that is not staged leaves its records out, and the figure draws what is there.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from memshift.analysis import figstyle as fs
from memshift.analysis import replication
from memshift.analysis.paper_data import (
    FAMILIES,
    NATIVES,
    TIERS,
    load_families,
    load_readers,
    m3_signs,
    replication_view,
    stage_json,
    tier_key,
)
from memshift.analysis.paper_data import get as _get

QWEN = "Qwen/Qwen3.5-9B"
H_MODELS = ("claude-sonnet-5-5", "claude-opus-5-5")
DH_SHIFTS = ("src", "def")  # the shifts of panel (b) of the writer figure
CLASS_NAMES = (("definition", "definition items"), ("other", "other items"))
SHIFT_ROWS = (
    ("src", r"$S_{\mathrm{src}}$"),
    ("def", r"$S_{\mathrm{def}}$"),
    ("orphan", r"$S_{\mathrm{del}}$"),
)


# ------------------------------------------------------------------------------------ loading
def load(root: Path, sources: Mapping[str, str]) -> dict[str, Any]:
    """The staged inputs of the four figures and of the table of the other registered verdicts
    (None where a stage or file is not there)."""
    src = dict(sources)

    def r3(name: str) -> dict[str, Any] | None:
        return stage_json(root, src, "round3", name)

    return {
        "g_b": r3("g_b.json"),
        "writers": stage_json(root, src, "ex_round3", "writers.json"),
        "sr": stage_json(root, src, "ex_round3", "store_reader.json"),
        "miss": stage_json(root, src, "ex_misses", "ex_misses.json"),
        "e1": stage_json(root, src, "e1", "e1_metrics.json"),
        "e2": stage_json(root, src, "e2", "e2_metrics.json"),
        "e3": stage_json(root, src, "e3", "e3_metrics.json"),
        "rep_e1": stage_json(root, src, "rep_e1", "e1_metrics.json"),
        "rep_e2": stage_json(root, src, "rep_e2", "e2_metrics.json"),
        "rep_e3": stage_json(root, src, "rep_e3", "e3_metrics.json"),
        "m3": load_readers(stage_json(root, src, "m3", "e7_metrics.json")),
        "nat": stage_json(root, src, "n", "e5_natives.json"),
        "e6": stage_json(root, src, "e6", "e6_t2.json"),
        "e4x": stage_json(root, src, "e4x", "e4x_metrics.json"),
        "gate": stage_json(root, src, "gate", "gate_a.json"),
        "m1": stage_json(root, src, "m1", "e2_metrics.json"),
        "c": stage_json(root, src, "c", "e1_metrics.json"),
        "ctl": stage_json(root, src, "ctl", "ctl_metrics.json"),
        "rendered": stage_json(root, src, "l", "rendered.json"),
        "m2": {
            f.key: load_families(root, src[f"m2_{f.key}"]) if f"m2_{f.key}" in src else None
            for f in FAMILIES
        },
        **{k: r3(f"{k}.json") for k in ("u", "n", "y", "q", "g_a", "v", "h_sonnet", "h_opus")},
    }


def _ok(e: Any) -> bool:
    """An estimate with a finite point value."""
    return (
        isinstance(e, Mapping)
        and isinstance(e.get("est"), int | float)
        and math.isfinite(float(e["est"]))
        and int(e.get("n_worlds", 1) or 0) > 0
    )


def _est(e: Any) -> dict[str, float] | None:
    """`{est, lo, hi}` of an estimate (lo, hi default to est), or None."""
    if not _ok(e):
        return None
    est = float(e["est"])
    lo, hi = e.get("lo", e.get("lo95")), e.get("hi", e.get("hi95"))
    lo = est if not isinstance(lo, int | float) or not math.isfinite(lo) else float(lo)
    hi = est if not isinstance(hi, int | float) or not math.isfinite(hi) else float(hi)
    return {"est": est, "lo": lo, "hi": hi}


# =================================================================================== writers
def coincide(points: Iterable[Mapping[str, Any]], tol: float = 5e-4) -> list[dict[str, Any]]:
    """Points of one writer at the same (x, y) (within `tol`, i.e. 0.05 pp) merged into one
    group with their count: identical program sets draw as one marker."""
    groups: list[dict[str, Any]] = []
    for p in points:
        if p["x"] is None or p["y"] is None:
            continue
        for g in groups:
            if (
                g["writer"] == p["writer"]
                and abs(g["x"]["est"] - p["x"]["est"]) <= tol
                and abs(g["y"]["est"] - p["y"]["est"]) <= tol
            ):
                g["n"] += 1
                g["sets"].append(p["set"])
                break
        else:
            groups.append({**p, "n": 1, "sets": [p["set"]]})
    return groups


def writer_points(inp: Mapping[str, Any]) -> dict[str, Any]:
    """The data of `fig_writers`.

    `programs`: per G-b program set (S_def, definition items): writer, x = share of definition
    slots whose program value at the fork equals gold (ex_round3 writers.json, S_def branch), y =
    B x D at D_m (g_b.json, world-bootstrap interval); a set without x or y keeps None there.
    `groups`: the sets merged where one writer's sets coincide. `reference`: the authored
    derivation's B x D on the same item-times (main study, e1 class split). `defwriters`: per
    writer and shift (S_src, S_def), x = r_0 and y = B x D with the authored D=1 cells, Qwen3.5-9B
    on the same worlds and item-times as the other writers."""
    gb, wr = inp.get("g_b"), inp.get("writers")
    prog = _get(gb, "quantities", "programs") or {}
    per_set = _get(prog, "branches", "def", "definition", "per_set") or {}
    sets = _get(wr, "program_sets", "sets") or {}
    points = []
    for sid in sorted(per_set):
        writer = _get(prog, "sets", sid, "writer") or _get(sets, sid, "writer") or sid.split("#")[0]
        points.append(
            {
                "set": sid,
                "writer": str(writer),
                "x": _est(_get(sets, sid, "agreement_t0", "def")),
                "y": _est(per_set[sid]),
            }
        )
    ref = _get(
        inp.get("e1"), "explore_class_split", "D1", "def", "source", "definition",
        "BxD_interaction_update",
    )  # fmt: skip
    defw: list[dict[str, Any]] = []
    writers = _get(wr, "definition_writers", "writers") or {}
    models = [m for m in H_MODELS if m in writers] + sorted(set(writers) - set(H_MODELS))
    for b in DH_SHIFTS:
        qwen_done = False
        for model in models:
            node = _get(writers, model, "branches", b) or {}
            pairs = [(model, "writer", "writer")]
            if not qwen_done:
                pairs.insert(0, (QWEN, "qwen_same_worlds", "qwen_same_item_times"))
            for who, rkey, bkey in pairs:
                x, y = _est(_get(node, "r0", rkey)), _est(_get(node, "bxd_D1", bkey))
                if x is None or y is None:
                    continue
                defw.append({"writer": who, "shift": b, "x": x, "y": y})
                qwen_done = qwen_done or who == QWEN
    return {
        "programs": points,
        "groups": coincide(points),
        "reference": _est(ref),
        "defwriters": defw,
    }


def _pt(fig: Any) -> float:
    """Display pixels per typographic point."""
    return float(fig.dpi) / 72.0


@dataclass
class _Labeller:
    """Direct labels placed where they cover no marker, interval or other label: each label tries
    candidate offsets (points) around its anchor in order and takes the first that fits inside the
    axes."""

    ax: Any
    renderer: Any
    obstacles: list[Any] = field(default_factory=list)

    def block(self, bbox: Any) -> None:
        self.obstacles.append(bbox)

    def block_marker(self, x: float, y: float, size_pt: float, dx_pt: float = 0.0) -> None:
        from matplotlib.transforms import Bbox

        px, py = self.ax.transData.transform((x, y))
        px += dx_pt * _pt(self.ax.figure)
        r = size_pt * _pt(self.ax.figure) / 2 + 1
        self.obstacles.append(Bbox.from_extents(px - r, py - r, px + r, py + r))

    def block_segment(
        self, x0: float, y0: float, x1: float, y1: float, width_pt: float, dx_pt: float = 0.0
    ) -> None:
        from matplotlib.transforms import Bbox

        (ax0, ay0), (ax1, ay1) = self.ax.transData.transform([(x0, y0), (x1, y1)])
        ax0, ax1 = ax0 + dx_pt * _pt(self.ax.figure), ax1 + dx_pt * _pt(self.ax.figure)
        r = width_pt * _pt(self.ax.figure) / 2 + 0.5
        self.obstacles.append(
            Bbox.from_extents(min(ax0, ax1) - r, min(ay0, ay1) - r, max(ax0, ax1) + r,
                              max(ay0, ay1) + r)
        )  # fmt: skip

    def place(
        self,
        text: str,
        xy: tuple[float, float],
        candidates: Sequence[tuple[float, float, str, str]],
        fallback: bool = True,
        **kw: Any,
    ) -> Any:
        """The first candidate that fits; else the first candidate, or nothing (None) when
        `fallback` is False."""
        frame = self.ax.get_window_extent(self.renderer)
        first = None
        for dx, dy, ha, va in candidates:
            t = self.ax.annotate(text, xy, xytext=(dx, dy), textcoords="offset points", ha=ha,
                                 va=va, annotation_clip=False, **kw)  # fmt: skip
            t.set_in_layout(False)
            bb = t.get_window_extent(self.renderer).expanded(1.04, 1.12)
            inside = (
                bb.x0 >= frame.x0 and bb.x1 <= frame.x1 and bb.y0 >= frame.y0
                and bb.y1 <= frame.y1
            )  # fmt: skip
            if inside and not any(bb.overlaps(o) for o in self.obstacles):
                self.obstacles.append(bb)
                if first is not None:
                    first.remove()
                return t
            if first is None:
                first = t
            else:
                t.remove()
        assert first is not None
        if not fallback:
            first.remove()
            return None
        self.obstacles.append(first.get_window_extent(self.renderer))
        return first


def _around(half_w: float, half_h: float, gap: float = 3.0) -> list[tuple[float, float, str, str]]:
    """Candidate label offsets (points) around a box of half-size (half_w, half_h): right, left,
    above, below, then the corners and further out."""
    out = []
    for k in (1.0, 2.2, 3.6):
        w, h, g = half_w, half_h, gap * k
        out += [
            (w + g, 0, "left", "center"),
            (-w - g, 0, "right", "center"),
            (0, h + g, "center", "bottom"),
            (0, -h - g, "center", "top"),
            (w + g, h + g, "left", "bottom"),
            (-w - g, h + g, "right", "bottom"),
            (w + g, -h - g, "left", "top"),
            (-w - g, -h - g, "right", "top"),
        ]
    return out


def _dodge(ax: Any, pts: Sequence[Mapping[str, Any]], size_pt: float) -> list[float]:
    """Horizontal offsets (points) that set apart markers drawn on top of each other: markers
    closer than one marker width form a cluster, whose members are drawn side by side in the
    order of their values, centred on the cluster."""
    pos = [ax.transData.transform((100 * p["x"]["est"], 100 * p["y"]["est"])) for p in pts]
    scale = _pt(ax.figure)
    parent = list(range(len(pts)))

    def root(i: int) -> int:
        while parent[i] != i:
            i = parent[i]
        return i

    for i in range(len(pts)):
        for j in range(i):
            if math.dist(pos[i], pos[j]) / scale < size_pt * 0.95:
                parent[root(i)] = root(j)
    dx = [0.0] * len(pts)
    for r in set(map(root, range(len(pts)))):
        members = sorted((i for i in range(len(pts)) if root(i) == r),
                         key=lambda i: (pos[i][0], pts[i]["writer"], pos[i][1]))  # fmt: skip
        for k, i in enumerate(members):
            dx[i] = (k - (len(members) - 1) / 2) * size_pt * 1.05
    return dx


WRITER_MS = 4.6  # marker size (points) of the writer figure


def _writer_marks(ax: Any, pts: Sequence[Mapping[str, Any]], lab: _Labeller, xerr: bool) -> None:
    """Markers (colour = writer, shape = shift) with their y (and x) intervals, markers of
    different writers on one spot drawn side by side; each mark blocks the direct labels."""
    from matplotlib.transforms import ScaledTranslation

    fig = ax.figure
    for p, dx in zip(pts, _dodge(ax, pts, WRITER_MS), strict=True):
        _, colour = fs.model_style(p["writer"])
        tr = ax.transData + ScaledTranslation(dx / 72, 0, fig.dpi_scale_trans)
        x, y = 100 * p["x"]["est"], 100 * p["y"]["est"]
        bars = [([x, x], [100 * p["y"]["lo"], 100 * p["y"]["hi"]])]
        if xerr:
            bars.append(([100 * p["x"]["lo"], 100 * p["x"]["hi"]], [y, y]))
        for xs, ys in bars:
            ax.plot(xs, ys, color=colour, lw=0.85, alpha=0.8, zorder=3, transform=tr,
                    solid_capstyle="butt")  # fmt: skip
            lab.block_segment(xs[0], ys[0], xs[1], ys[1], 0.85, dx)
        ax.plot([x], [y], marker=fs.SHIFTS[p["shift"]][1], ms=WRITER_MS, mfc=colour,
                mec="white", mew=0.5, ls="none", zorder=4, transform=tr)  # fmt: skip
        lab.block_marker(x, y, WRITER_MS + 1, dx)


def _side(writer: str, pts: Sequence[Mapping[str, Any]], near: float = 5.0) -> str | None:
    """Where a writer's label goes when another writer's markers share its x range: above if its
    markers lie higher on average, else below (so two neighbours' labels never meet); None when
    no other writer is near."""
    mine = [p for p in pts if p["writer"] == writer]
    lo = min(100 * p["x"]["est"] for p in mine) - near
    hi = max(100 * p["x"]["est"] for p in mine) + near
    others = [p for p in pts if p["writer"] != writer and lo <= 100 * p["x"]["est"] <= hi]
    if not others:
        return None
    mean_mine = sum(p["y"]["est"] for p in mine) / len(mine)
    return "above" if mean_mine > sum(p["y"]["est"] for p in others) / len(others) else "below"


def _writer_names(ax: Any, pts: Sequence[Mapping[str, Any]], lab: _Labeller) -> None:
    """One direct label per writer next to its own markers: beyond the end of its intervals on
    the side `_side` gives when another writer is near, else next to the box around its markers."""
    for writer in dict.fromkeys(p["writer"] for p in pts):
        mine = [p for p in pts if p["writer"] == writer]
        xs = [100 * p["x"]["est"] for p in mine]
        ys = [100 * p["y"]["est"] for p in mine]
        (x0, y0), (x1, y1) = ax.transData.transform([(min(xs), min(ys)), (max(xs), max(ys))])
        hw = (x1 - x0) / 2 / _pt(ax.figure) + WRITER_MS * 0.8
        hh = (y1 - y0) / 2 / _pt(ax.figure) + WRITER_MS / 2
        label, colour = fs.model_style(writer)
        cx = (min(xs) + max(xs)) / 2
        side = _side(writer, pts)
        placed = None
        # first beside the markers on that side, then beyond the intervals; then the other side
        # (three writers can share an x range, so the side `_side` gives may be taken)
        sides = [side == "above", side != "above"] if side else []
        for up in sides:
            if placed is not None:
                break
            va, sign = ("bottom", 1.0) if up else ("top", -1.0)
            near = [
                (dx, sign * (hh + 3.0), ha, va)
                for dx, ha in ((-hw - 3.0, "right"), (hw + 3.0, "left"), (0.0, "center"))
            ]
            placed = lab.place(label, (cx, (min(ys) + max(ys)) / 2), near, fallback=False,
                               fontsize=7.5, color=colour)  # fmt: skip
            edge = (max if up else min)(100 * p["y"]["hi" if up else "lo"] for p in mine)
            beyond = [
                (dx, sign * g, ha, va)
                for g in (2.0, 5.0, 9.0)
                for dx, ha in ((0.0, "center"), (-hw, "left"), (hw, "right"))
            ]
            if placed is None:
                placed = lab.place(label, (cx, edge), beyond, fallback=False, fontsize=7.5,
                                   color=colour)  # fmt: skip
        if placed is None:
            lab.place(label, (cx, (min(ys) + max(ys)) / 2), _around(hw, hh), fontsize=7.5,
                      color=colour)  # fmt: skip


def fig_writers(inp: Mapping[str, Any], out: Path) -> Path:
    """(a) G-b program sets, (b) definition-item writers: see the module docstring."""
    plt = fs.plt()
    data = writer_points(inp)
    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(fs.TEXT, 1.65), sharey=True, gridspec_kw={"width_ratios": [1.22, 1]}
    )
    for ax in (axa, axb):
        ax.axhline(0, color=fs.GREY, lw=0.6, zorder=1)
        ax.set_ylim(-8, 50)
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
    axa.set_xlim(-4, 104)
    axa.set_xticks([0, 25, 50, 75, 100])
    axb.set_xlim(-4, 72)
    axb.set_xticks([0, 20, 40, 60])
    axa.set_ylabel(r"$B{\times}D$, definition items (pp)")
    axa.set_xlabel(r"programs right before the shift (\%)")
    axb.set_xlabel(r"$r_0$: changed values re-derived at $D{=}0$ (\%)")
    fs.panel_label(axa, "a")
    fs.panel_label(axb, "b")
    ref = data["reference"]
    if ref:
        axa.axhspan(100 * ref["lo"], 100 * ref["hi"], color=fs.LIGHT, alpha=0.55, lw=0, zorder=0.5)
        axa.axhline(100 * ref["est"], color=fs.GREY, lw=0.9, ls=(0, (4, 2)), zorder=1)
    fig.canvas.draw()  # settle the layout: label placement works in display space
    renderer = fig.canvas.get_renderer()
    lab_a, lab_b = _Labeller(axa, renderer), _Labeller(axb, renderer)
    for ax, lab in ((axa, lab_a), (axb, lab_b)):  # no label across the zero line
        lab.block_segment(ax.get_xlim()[0], 0, ax.get_xlim()[1], 0, 1.0)

    # (a) program sets, shape of S_def; the authored derivation's B x D as reference
    if ref:
        lab_a.block_segment(-4, 100 * ref["est"], 104, 100 * ref["est"], 1.2)
        t = axa.text(1, 100 * ref["hi"] + 1.0, "authored derivation", fontsize=7.5,
                     color=fs.GREY, ha="left", va="bottom")  # fmt: skip
        t.set_in_layout(False)
        lab_a.block(t.get_window_extent(renderer).expanded(1.05, 1.2))
    groups = [{**g, "shift": "def"} for g in data["groups"]]
    _writer_marks(axa, groups, lab_a, xerr=False)
    for g in groups:  # the count of identical program sets, next to their marker
        if g["n"] > 1:
            _, colour = fs.model_style(g["writer"])
            lab_a.place(rf"${{\times}}{g['n']}$", (100 * g["x"]["est"], 100 * g["y"]["est"]),
                        _around(WRITER_MS / 2 + 1.5, WRITER_MS / 2 + 1, 1.0)[:8], fontsize=7,
                        color=colour)  # fmt: skip
    _writer_names(axa, groups, lab_a)
    if not groups:
        axa.text(0.5, 0.5, "no program set staged", transform=axa.transAxes, ha="center",
                 fontsize=7.5, color=fs.GREY)  # fmt: skip

    # (b) definition-item writers: shape = shift, intervals on both axes
    pts = data["defwriters"]
    _writer_marks(axb, pts, lab_b, xerr=True)
    if pts:  # the marker of each shift, in the empty upper right corner
        from matplotlib.lines import Line2D

        handles = [
            Line2D([], [], marker=fs.SHIFTS[b][1], ls="none", ms=WRITER_MS, mfc=fs.GREY,
                   mec="white", mew=0.5, label=fs.SHIFTS[b][0])
            for b in DH_SHIFTS
        ]  # fmt: skip
        leg = axb.legend(handles=handles, loc="upper right", fontsize=7.5, handletextpad=0.2,
                         borderaxespad=0.2, labelspacing=0.3)  # fmt: skip
        lab_b.block(leg.get_window_extent(renderer))  # no writer's name over the key
    _writer_names(axb, pts, lab_b)
    path = out / "fig_writers.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


# ==================================================================================== reader
def reader_rows(inp: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The data of `fig_reader`: per item class and shift, B x D on Update (authored D=1) with the
    Qwen3.5-9B reader and with the Claude Sonnet 5.5 reader on the same stores, paired per
    item-time (y.json; the Qwen value falls back to the committed split), and the store part,
    the same for both readers (y.json's split, else the committed reader's)."""
    y, sr = inp.get("y"), _get(inp.get("sr"), "results")
    con = _get(y, "quantities", "contrasts")
    ysr = _get(y, "quantities", "store_reader", "results")
    rows = []
    for cls, _ in CLASS_NAMES:
        for b, _ in SHIFT_ROWS:
            node = _get(con, b, cls, "D1")
            q = _get(sr, "D1", b, cls, "interaction")
            s = _get(ysr, "D1", b, cls, "interaction")
            rows.append(
                {
                    "cls": cls,
                    "branch": b,
                    "store": _est(_get(s, "store")) or _est(_get(q, "store")),
                    "qwen": _est(_get(node, "committed_reader")) or _est(_get(q, "observed")),
                    "sonnet": _est(_get(node, "reader")),
                }
            )
    return rows


READERS = (("qwen", QWEN, -0.13), ("sonnet", "claude-sonnet-5-5", 0.13))  # key, model, y offset
STORE_GREY, STORE_LINE = "#E3E3E3", "#9E9E9E"
SAME_PP = 1.5  # a reader that changes B x D by less than this draws a stroke, not an arrow


class _ArrowHandler:
    """A legend handle that draws an arrow in the handle's colour."""

    def legend_artist(self, legend: Any, orig: Any, fontsize: float, box: Any) -> Any:
        from matplotlib.patches import FancyArrowPatch

        x0, y0, w, h = box.xdescent, box.ydescent, box.width, box.height
        arrow = FancyArrowPatch((-x0, -y0 + h / 2), (-x0 + w, -y0 + h / 2),
                                arrowstyle="-|>,head_length=3.2,head_width=1.7",
                                color=orig.get_color(), lw=1.2, shrinkA=0, shrinkB=0,
                                transform=box.get_transform())  # fmt: skip
        box.add_artist(arrow)
        return arrow


def fig_reader(inp: Mapping[str, Any], out: Path) -> Path:
    """Per item class (panel) and shift (row): the store part as a light grey bar from 0, its
    interval as a thin grey line; for each reader an arrow from the end of the bar to B x D with
    that reader (what the reader adds or loses), its interval a soft band behind the arrowhead,
    and a short stroke instead of an arrow where the reader changes almost nothing. The panels
    share one pp scale."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt = fs.plt()
    rows = reader_rows(inp)
    spans = {}
    for cls, _ in CLASS_NAMES:
        vals = [100 * r[k][e] for r in rows if r["cls"] == cls
                for k in ("store", "qwen", "sonnet") if r[k] for e in ("lo", "hi")]  # fmt: skip
        lo, hi = min([0.0, *vals]), max([10.0, *vals])
        spans[cls] = (math.floor(lo / 5) * 5 - 2, math.ceil(hi / 5) * 5 + 2)
    widths = [spans[c][1] - spans[c][0] for c, _ in CLASS_NAMES]
    fig, axes = plt.subplots(1, 2, figsize=(fs.TEXT, 1.95), sharey=True,
                             gridspec_kw={"width_ratios": widths})  # fmt: skip
    for ax, (cls, title) in zip(axes, CLASS_NAMES, strict=True):
        ax.axvline(0, color=fs.BLACK, lw=0.6, zorder=1)
        for i, (b, _) in enumerate(SHIFT_ROWS):
            r = next((r for r in rows if r["cls"] == cls and r["branch"] == b), None)
            if r is None:
                continue
            s = r["store"]
            if s:
                ax.barh([i], [100 * s["est"]], height=0.56, color=STORE_GREY, lw=0, zorder=1.5)
                ax.plot([100 * s["lo"], 100 * s["hi"]], [i - 0.37] * 2, color=STORE_LINE,
                        lw=0.9, solid_capstyle="butt", zorder=2)  # fmt: skip
            start = 100 * s["est"] if s else 0.0
            for key, model, dy in READERS:
                e = r[key]
                if not e:
                    continue
                _, colour = fs.model_style(model)
                end = 100 * e["est"]
                fs.interval_bar(ax, i + dy, None, 100 * e["lo"], 100 * e["hi"], colour,
                                alpha=0.2, zorder=2.2)  # fmt: skip
                if abs(end - start) < SAME_PP:
                    ax.plot([end], [i + dy], marker="|", ms=6.5, mew=1.7, color=colour,
                            ls="none", zorder=4)  # fmt: skip
                else:
                    ax.annotate("", xy=(end, i + dy), xytext=(start, i + dy), zorder=4,
                                arrowprops={"arrowstyle": "-|>,head_length=0.32,head_width=0.17",
                                            "color": colour, "lw": 1.2, "shrinkA": 0,
                                            "shrinkB": 0})  # fmt: skip
        ax.set_title(title, pad=3)
        ax.set_xlim(*spans[cls])
        ax.xaxis.set_major_locator(plt.MultipleLocator(10))
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(range(len(SHIFT_ROWS)), [t for _, t in SHIFT_ROWS])
    axes[0].set_ylim(len(SHIFT_ROWS) - 0.45, -0.6)
    fig.supxlabel(r"$B{\times}D$ on Update (pp)", fontsize=9)
    handles: list[Any] = [Patch(color=STORE_GREY, label="store part (both readers)")]
    handler: dict[Any, Any] = {}
    for _, model, _ in READERS:
        label, colour = fs.model_style(model)
        h = Line2D([], [], color=colour, label=f"{label} reader")
        handler[h] = _ArrowHandler()
        handles.append(h)
    fig.legend(handles=handles, handler_map=handler, loc="outside upper center", ncol=3,
               columnspacing=1.8, handlelength=2.0, fontsize=7.5)  # fmt: skip
    path = out / "fig_reader.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


# ==================================================================================== misses
MISS_CAUSES = (  # key -> (label, colour, a write-time key could have covered it)
    ("a", r"$a$: an uncited record changed", "#8C8C8C", True),
    ("b_old", r"$b$ old: existing record entered scope", "#C8C8C8", True),
    ("b_new", r"$b$ new: a record created after the derivation", fs.VERMILION, False),
    ("c_left", r"$c$: records left window or scope", "#F0A877", False),
    ("c_rest", r"$c$: other", "#E8E8E8", True),
)
MISS_STAGES = (("main", "main study"), ("rep", "replication"))


def miss_rows(inp: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The data of `fig_misses`: per stage (main study, replication), shift and item class, the
    changed item-times the cited-cluster key missed and the share of each cause (exclusive
    classes a, b old, b new, c left and the rest of c, from the counts of ex_misses.json)."""
    miss = _get(inp.get("miss"), "explore_miss_classes")
    rows = []
    for stage, _ in MISS_STAGES:
        for b, _ in SHIFT_ROWS:
            for cls, _ in CLASS_NAMES:
                n = _get(miss, stage, b, cls)
                total = _get(n, "missed_item_times")
                if not n or not total:
                    continue
                counts = {
                    "a": n.get("a_count", 0),
                    "b_old": n.get("b_existing_count", 0),
                    "b_new": n.get("b_new_count", 0),
                    "c_left": n.get("c_left_scope_count", 0),
                    "c_rest": n.get("c_count", 0) - n.get("c_left_scope_count", 0),
                }
                rows.append(
                    {
                        "stage": stage,
                        "branch": b,
                        "cls": cls,
                        "missed": int(total),
                        "shares": {k: v / total for k, v in counts.items()},
                        "uncoverable": (counts["b_new"] + counts["c_left"]) / total,
                    }
                )
    return rows


def fig_misses(inp: Mapping[str, Any], out: Path) -> Path:
    """100% stacked bars of the missed changed item-times by cause, per shift and class (rows),
    main study and replication (panels), with the number missed at the bar end."""
    from matplotlib.patches import Patch

    plt = fs.plt()
    rows = miss_rows(inp)
    causes = [
        c for c in MISS_CAUSES if c[0] != "c_rest" or any(r["shares"]["c_rest"] for r in rows)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(fs.TEXT, 1.95), sharey=True)
    ypos: dict[tuple[str, str], float] = {}
    y = 0.0
    for b, _ in SHIFT_ROWS:
        for cls, _ in CLASS_NAMES:
            ypos[b, cls] = y
            y -= 1.0
        y -= 0.45
    for ax, (stage, title) in zip(axes, MISS_STAGES, strict=True):
        for r in (r for r in rows if r["stage"] == stage):
            yy, left = ypos[r["branch"], r["cls"]], 0.0
            for key, _, colour, _ in causes:
                w = 100 * r["shares"][key]
                if w > 0:
                    ax.barh([yy], [w], left=[left], height=0.72, color=colour,
                            edgecolor="white", lw=0.4, zorder=2)  # fmt: skip
                left += w
            ax.text(101.5, yy, f"{r['missed']}", fontsize=7, va="center", ha="left",
                    color=fs.BLACK)  # fmt: skip
        ax.set_title(title, pad=3)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.text(1.0, 1.0, "missed", transform=ax.transAxes, fontsize=7, ha="left",
                va="bottom", color=fs.GREY)  # fmt: skip
    ticks = [(ypos[b, cls], cname.split()[0]) for b, _ in SHIFT_ROWS for cls, cname in CLASS_NAMES]
    axes[0].set_yticks([t for t, _ in ticks], [lab for _, lab in ticks])
    for b, bt in SHIFT_ROWS:  # the shift once per pair of rows, left of the class names
        mid = (ypos[b, "definition"] + ypos[b, "other"]) / 2
        axes[0].text(-0.3, mid, bt, transform=axes[0].get_yaxis_transform(), ha="right",
                     va="center", fontsize=9)  # fmt: skip
    axes[0].set_ylim(min(ypos.values()) - 0.6, 0.6)
    fig.supxlabel(r"share of the missed changed item-times (\%)", fontsize=9)
    handles = [Patch(facecolor=c, edgecolor="none", label=lab) for _, lab, c, _ in causes]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, columnspacing=1.4,
               handlelength=1.2, fontsize=7.5)  # fmt: skip
    path = out / "fig_misses.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


# ================================================================================ registered
# The registered bar of a row: (kind, values). kinds: gt / ge (holds above the value), lt / le
# (below), abs_lt / abs_le (inside +-value), reading (no pass or fail: the two cut points that
# separate the readings)
Bar = tuple[str, tuple[float, ...]]
HOLDS, FAILS, READING = "holds", "fails", "reading"
SHIFT_TEXT = {b: t for b, t in SHIFT_ROWS}


@dataclass
class Row:
    """One registered prediction (or one arm of it) in the forest plot: its points are drawn in
    sub-rows, one per `sub` label (the shift unless a point names another), in their order."""

    label: str
    panel: str  # the key of its panel in PANELS
    bar: Bar
    points: list[dict[str, Any]]  # {est, lo, hi, shift (or None), sub, verdict}
    scale: str = "pp"  # pp (x 100) or ratio (as is)

    def subrows(self) -> list[tuple[str, list[dict[str, Any]]]]:
        """(label, points) per sub-row; points of one sub-row with the same values are one mark."""
        out: dict[str, list[dict[str, Any]]] = {}
        for p in self.points:
            same = out.setdefault(p["sub"], [])
            if not any(all(q[k] == p[k] for k in ("est", "lo", "hi", "verdict")) for q in same):
                same.append(p)
        return list(out.items())


def _verdict_word(v: Any) -> str | None:
    """holds / fails / reading from the words of the analyses (None: not judged)."""
    if v is True:
        return HOLDS
    if v is False:
        return FAILS
    w = str(v or "").replace("_", " ").lower()
    if w in ("holds", "fails"):
        return w
    if w == "judged":
        return READING
    return None


def _point(
    e: Any, shift: str | None, verdict: Any, sub: str | None = None
) -> dict[str, Any] | None:
    est, word = _est(e), _verdict_word(verdict)
    if est is None or word is None:
        return None
    label = sub if sub is not None else SHIFT_TEXT.get(shift or "", "pooled")
    return {**est, "shift": shift, "sub": label, "verdict": word}


def _row(label: str, panel: str, bar: Bar, pts: Iterable[dict[str, Any] | None], **kw: Any) -> Row:
    return Row(label, panel, bar, [p for p in pts if p], **kw)


def _pred(res: Any, name: str) -> Any:
    return _get(res, "predictions", name, "verdict")


def _arm(d: int, b: str) -> str:
    return rf"$D{{=}}{d}$, {SHIFT_TEXT[b]}"


def _round1(inp: Mapping[str, Any]) -> list[Row]:
    """v0/v1: the falsification paths and the registered E1 contrasts (predicted positive: the
    point estimate > 0, `replication.as_predicted`), main study."""
    e1, gate = inp.get("e1"), inp.get("gate")
    con = _get(e1, "contrasts") or {}

    def pos(key: str, shifts: Sequence[str]) -> list[dict[str, Any] | None]:
        out = []
        for b in shifts:
            e = _get(con, b, key)
            out.append(_point(e, b, _ok(e) and e["est"] > 0))
        return out

    p1 = _get(gate, "path1")
    p2 = _get(gate, "path2", "checks") or {}
    p3 = _get(gate, "path3") or {}
    path3 = p3 and not p3.get("triggered")
    return [
        _row(r"Gain check: binding's effect, pooled over $D$", "binding", ("gt", (2.0,)),
             [_point(_get(p1, "estimate"), "src", p1 and not p1.get("triggered"))]),
        _row(r"Binding's effect on Update at $D{=}1$", "binding", ("gt", (0.0,)),
             pos("B_effect_update_D1_source", ("src", "def", "orphan"))),
        _row(r"Interaction $B{\times}D$ on Update", "binding", ("gt", (0.0,)),
             pos("BxD_interaction_update_source", ("src", "def"))),
        _row(r"Executability's effect on repair", "repair", ("gt", (0.0,)),
             pos("D_effect_repair_source", ("src", "def", "orphan"))),
        _row(r"Leak check: executability's effect on missed invalidation, both keys", "missed",
             ("abs_lt", (2.0,)),
             [_point(c.get("estimate"), k.split(":")[0], not c.get("triggered"))
              for k, c in sorted(p2.items(), key=lambda kc: BRANCH_ORDER[kc[0].split(":")[0]])]),
        _row(r"Cited-cluster key: binding's effect on Update at $D{=}1$", "cited", ("gt", (0.0,)),
             pos("B_effect_update_D1_cited", ("src", "def", "orphan"))),
        _row(r"Cited-cluster key: binding's effect, pooled over $D$", "cited", ("gt", (0.0,)),
             pos("B_effect_update_pooled_cited", ("src", "def", "orphan"))),
        _row(r"Cited-cluster key: interaction $B{\times}D$", "cited", ("gt", (0.0,)),
             pos("BxD_interaction_update_cited", ("src", "def"))),
        _row("Agreement check: calibration slope (90\\% interval)", "ratio", ("gt", (0.0,)),
             [_point(_ci90(p3.get("pooled_slope")), None, path3)], scale="ratio"),
        _row("Agreement check: AUROC (90\\% interval)", "ratio", ("ge", (0.60,)),
             [_point(_ci90(p3.get("pooled_auroc")), None, path3)], scale="ratio"),
    ]  # fmt: skip


BRANCH_ORDER = {b: i for i, (b, _) in enumerate(SHIFT_ROWS)}


def _ci90(e: Any) -> dict[str, Any] | None:
    """An estimate with its 90% bounds as lo, hi (path 3 is judged on them)."""
    if not _ok(e):
        return None
    return {"est": e["est"], "lo": e.get("lo90", e["est"]), "hi": e.get("hi90", e["est"])}


def _round2(inp: Mapping[str, Any]) -> list[Row]:
    """v2 (blocks R, C, M1, M2, L) and v3 (the controls)."""
    from memshift.analysis.paper_data import synth_holds

    rep = _get(inp.get("rep_e1"), "contrasts") or {}
    c = _get(inp.get("c"), "contrasts") or {}
    ctl = inp.get("ctl") or {}
    cc, cv = ctl.get("contrasts") or {}, ctl.get("verdicts") or {}
    m1 = _get(inp.get("m1"), "path3") or {}
    m1_ok = m1 and not m1.get("triggered")

    def ctl_row(
        label: str,
        pred: str,
        panel: str,
        bar: Bar,
        keys: Sequence[tuple[str, str, str | None]],
        **kw: Any,
    ) -> Row:
        v = _get(cv, pred, "status")
        return _row(label, panel, bar, [_point(cc.get(k), b, v, sub) for k, b, sub in keys], **kw)

    def c_row(label: str, key: str, rule: str, panel: str, bar: Bar, shifts: Sequence[str]) -> Row:
        return _row(label, panel, bar,
                    [_point(_get(c, b, key), b, synth_holds(rule, _get(c, b, key)))
                     for b in shifts])  # fmt: skip

    fams = [(f.label, inp.get("m2", {}).get(f.key)) for f in FAMILIES]
    rendered = _get(inp.get("rendered"), "rendered")

    def fam_row(label: str, check: str, key: str, shift: str, floor: float = 0.0) -> Row:
        return _row(label, "generality", ("gt", (floor,)),
                    [_point(_get(f, "estimates", key), shift, _get(f, "checks", check), name)
                     for name, f in fams if f])  # fmt: skip

    def rep_bxd(b: str) -> dict[str, Any] | None:
        e = _get(rep, b, "BxD_interaction_update_source")
        return _point(e, b, _ok(e) and e["est"] > 0)

    rep_p1 = _get(rep, "src", "B_effect_update_pooled_source")
    sd, alls = ("src", "def"), ("src", "def", "orphan")
    return [
        c_row(r"Repair, model-written minus $D{=}0$", "C1_repair_D1s_minus_D0_source",
              "above", "repair", ("gt", (0.0,)), sd),
        c_row(r"Repair, authored minus model-written", "C2_repair_D1_minus_D1s_source",
              "nonneg", "repair", ("ge", (0.0,)), sd),
        c_row(r"Interaction, model-written derivations",
              "C3_BxD_interaction_update_D1s_source", "above", "repair", ("gt", (0.0,)), ("def",)),
        ctl_row(r"Repair, $D{=}1$ minus a $D{=}0$ writer with every record", "F1", "repair",
                ("ge", (30.0,)), [(f"F_rep_{b}_definition", b, None) for b in sd]),
        ctl_row(r"Query-time view minus the zero system", "Z1", "repair", ("ge", (30.0,)),
                [(f"Z_upd_{b}_definition", b, None) for b in sd]),
        ctl_row(r"$D{=}0$ value in the $D{=}1$ template, definition items", "T1", "format",
                ("abs_le", (3.0,)), [(f"T_upd_{b}_definition", b, None) for b in alls]),
        ctl_row(r"The same, all items", "T3", "format", ("abs_le", (3.0,)),
                [(f"T_upd_{b}_all", b, None) for b in alls]),
        ctl_row(r"Query-time view minus stored model-written value", "Z3", "format",
                ("abs_le", (10.0,)), [(f"Z_vs_D1s_{b}_definition", b, None) for b in sd]),
        c_row(r"Model-written derivations' effect on missed invalidation",
              "C4_D1s_effect_missed_source", "path2", "missed", ("abs_lt", (2.0,)), sd),
        ctl_row(r"Oracle minus source-epoch key, Update", "G1", "keys", ("ge", (-2.0,)),
                [(f"G_upd_D{d}_{b}_all", b, _arm(d, b)) for d in (0, 1) for b in alls]),
        ctl_row(r"Oracle minus source-epoch key, Stability", "G3", "keys", ("ge", (0.0,)),
                [(f"G_stab_D{d}_{b}_all", b, _arm(d, b)) for d in (0, 1) for b in alls]),
        ctl_row(r"Scope minus source-epoch key, definition items", "K1", "keys",
                ("ge", (-2.0,)), [(f"K_upd_{b}_definition", b, None) for b in alls]),
        ctl_row(r"Scope minus cited-cluster key, other items", "K2", "keys",
                ("abs_le", (1.0,)), [(f"K_cited_{b}_other", b, None) for b in alls]),
        _row(r"Gain check on the replication worlds", "generality", ("gt", (2.0,)),
             [_point(rep_p1, "src", _ok(rep_p1) and rep_p1["est"] > 0.02)]),
        _row(r"Interaction $B{\times}D$ on the replication worlds", "generality",
             ("gt", (0.0,)), [rep_bxd(b) for b in sd]),
        fam_row(r"Other families: binding's effect, pooled over $D$", "ii_B_update_src",
                "B_update_src", "src", floor=2.0),  # check (ii): > +2 pp, as path 1
        fam_row(r"Other families: interaction $B{\times}D$", "iv_BxD_def", "BxD_def", "def"),
        _row(r"Interaction, rewritten message bodies", "generality",
             ("gt", (0.0,)),
             [_point(_get(rendered, "estimates", "BxD_def"), "def",
                     _get(rendered, "checks", "iv_BxD_def"))]),
        _row("Agreement check, eight switches: slope (90\\%)", "ratio", ("gt", (0.0,)),
             [_point(_ci90(m1.get("pooled_slope")), None, m1_ok)], scale="ratio"),
        _row("Agreement check, eight switches: AUROC (90\\%)", "ratio", ("ge", (0.60,)),
             [_point(_ci90(m1.get("pooled_auroc")), None, m1_ok)], scale="ratio"),
        ctl_row(r"Add-only control, share of binding's effect", "R2", "ratio",
                ("ge", (0.5,)), [(f"R_ratio_{b}_definition", b, None) for b in sd],
                scale="ratio"),
        ctl_row(r"Template, share of the effect of $D$", "T2", "ratio", ("le", (0.15,)),
                [(f"T_ratio_{b}_definition", b, None) for b in sd], scale="ratio"),
    ]  # fmt: skip


def _round3(inp: Mapping[str, Any]) -> list[Row]:
    """v4 and v4b: the round-3 verdicts of `round3_registered` (staged in round3)."""
    u, n, y, q, ga, v, gb = (inp.get(k) for k in ("u", "n", "y", "q", "g_a", "v", "g_b"))
    fam = _get(u, "quantities", "families") or {}
    main = fam.get("main")
    cf = _get(n, "quantities", "counterfactual")
    con = _get(y, "quantities", "contrasts")
    held = _get(ga, "quantities", "heldout")
    vprog = _get(v, "quantities", "programs", "branches")
    gprog = _get(gb, "quantities", "programs", "branches")
    sd = ("src", "def")
    rows = [
        _row(r"Unnamed minus named probes, effect of $B$", "binding",
             ("ge", (5.0,)),
             [_point(_get(q, "quantities", "Q1", "difference"), "def", _pred(q, "Q1"))]),
        _row(r"Upsert minus append-only, all items", "binding", ("gt", (0.0,)),
             [_point(_get(main, "U4", "difference"), "orphan", _pred(u, "U4"))]),
        _row(r"Interaction, shifted minus natural branch", "binding", ("gt", (0.0,)),
             [_point(_get(cf, b, "shift", "definition", "difference"), b, _pred(n, "N1"))
              for b in sd]),
        *(
            _row(rf"Held-out definitions, {arm_t}", "generality", ("gt", (0.0,)),
                 [_point(_get(held, b, arm, "two_way"), b, _pred(ga, p))
                  for b, p in (("src", "G2"), ("def", "G1"))])
            for arm, arm_t in (("D1", r"$D{=}1$"), ("Dm", "model-written"))
        ),
        _row(r"Definitions in the user's own words", "generality", ("gt", (0.0,)),
             [_point(_get(vprog, b, "definition", "two_way"), b, _pred(v, p))
              for b, p in (("src", "V2"), ("def", "V1"))]),
        _row(r"Interaction, unnamed probes", "generality", ("gt", (0.0,)),
             [_point(_get(q, "quantities", "Q2", "D1", "unnamed"), "def", _pred(q, "Q2"))]),
        _row(r"Claude Sonnet 5.5 reader, other items", "generality", ("gt", (0.0,)),
             [_point(_get(con, b, "other", "D1", "reader"), b, _pred(y, p))
              for b, p in (("def", "Y1"), ("orphan", "Y2"))]),
        _row(r"Sonnet minus Qwen3.5-9B reader, definition items", "generality",
             ("abs_le", (5.0,)),
             [_point(_get(con, b, "definition", "D1", "difference"), b, _pred(y, "Y3"))
              for b in sd]),
    ]  # fmt: skip
    r0s = []
    for key, who in (("h_sonnet", "Sonnet"), ("h_opus", "Opus")):
        res = inp.get(key)
        model = "claude-sonnet-5-5" if who == "Sonnet" else "claude-opus-5-5"
        hq = _get(res, "quantities", "writers", model)
        suffix = "" if who == "Sonnet" else f"_{model}"  # round3_registered's prediction names
        rows.append(
            _row(rf"Claude {who} 5.5 writes the definition items", "generality",
                 ("reading", (10.0, 20.0)),
                 [_point(_get(hq, "bxd", "def", "definition", "D1"), "def",
                         _pred(res, f"H1{suffix}"))])
        )  # fmt: skip
        r0s.append(_point(_get(hq, "r0", "writer"), "src", _pred(res, f"H2{suffix}"), who))
    rows.append(
        _row(r"Program sets from six writers", "generality", ("gt", (0.0,)),
             [_point(_get(gprog, "def", "definition", "two_way"), "def", _pred(gb, "G3"))])
    )  # fmt: skip
    rows += [
        _row(r"Refresh, 20 worlds, share of binding's effect", "ratio", ("ge", (0.8,)),
             [_point(_get(main, "R", "refresh", b, arm, "definition", "R"), b, _pred(u, "U1"),
                     f"{arm_t}, {SHIFT_TEXT[b]}")
              for arm, arm_t in (("D1", r"$D{=}1$"), ("Dm", r"$D_{\mathrm{m}}$")) for b in sd],
             scale="ratio"),
        _row(r"Upsert, share of binding's effect", "ratio",
             ("reading", (0.5, 0.8)),
             [_point(_get(main, "pooled_definition_D1_Dm_src_def", "upsert", "R"), None,
                     _pred(u, "U3"))], scale="ratio"),
    ]  # fmt: skip
    m2u = [f for f in FAMILIES if f.model in fam]  # in the order and with the labels of the
    if m2u:  # other family rows; U5 appears once the M2 families' upsert runs are staged
        rows.append(
            _row(r"Upsert, other families", "ratio", ("reading", (0.5, 0.8)),
                 [_point(_get(fam, f.model, "pooled_definition_D1_Dm_src_def", "upsert", "R"),
                         None, _pred(u, "U5"), f.label)
                  for f in m2u], scale="ratio")
        )  # fmt: skip
    # H2's bar: Qwen3.5-9B's registered r_0 (16.2%) plus 20 pp
    rows.append(_row(r"Claude writers: changed definition values re-derived at $D{=}0$", "ratio",
                     ("ge", (0.362,)), r0s, scale="ratio"))  # fmt: skip
    return rows


PANELS = (  # key, title, unit, page of the figure
    ("binding", "Does binding pay?", "pp", 1),
    ("repair", "Does repair need an executable derivation?", "pp", 1),
    ("format", "Is it the value, not the format or where it is computed?", "pp", 1),
    ("missed", "Does executability leave detection alone?", "pp", 1),
    ("cited", "Does the cited-cluster key detect the changes?", "pp", 1),
    ("keys", "Do the oracle and scope keys keep binding's effect?", "pp", 2),
    ("generality", "Does the pattern generalise?", "pp", 2),
    ("ratio", "Ratios and shares", "ratio or share", 2),
)


def registered_rows(inp: Mapping[str, Any]) -> dict[str, list[Row]]:
    """The data of `fig_registered`: per panel (`PANELS`) the registered predictions with an
    estimate, in the order of the rounds, each with its points (estimate, interval, shift,
    sub-row label, verdict) and registered bar. A prediction that is not staged, or not judged,
    has no point and is left out."""
    rows = [*_round1(inp), *_round2(inp), *_round3(inp)]
    return {key: [r for r in rows if r.panel == key and r.points] for key, *_ in PANELS}


COUNT_MACROS = ("RegJudged", "RegHeld", "RegFailed", "RegMixed")


def registered_counts(rows: Mapping[str, Sequence[Row]]) -> dict[str, int]:
    """The counts the paper quotes over the rows of `fig_registered` (`registered_rows`): every
    row is one registered prediction (or arm) with an estimate. Its pass/fail points give its
    verdict: held if every point holds, failed if every point fails, mixed if it holds for some
    sub-rows (shifts, arms, writers) and fails for others. Points that are registered readings
    without pass or fail are left out, and a row with nothing else is not counted."""
    held = failed = mixed = 0
    for panel_rows in rows.values():
        for row in panel_rows:
            words = {p["verdict"] for p in row.points} - {READING}
            if words == {HOLDS}:
                held += 1
            elif words == {FAILS}:
                failed += 1
            elif words:
                mixed += 1
    return dict(zip(COUNT_MACROS, (held + failed + mixed, held, failed, mixed), strict=True))


def counts_tex(counts: Mapping[str, int]) -> str:
    """The counts (`registered_counts`, or `paper_counts` with the table's and the totals) as the
    LaTeX macros the paper inputs (paper/generated)."""
    lines = [
        "% generated by memshift.analysis.figures --only fig_registered; do not edit",
        "% Reg*: rows of fig:registered (registered predictions or arms with an estimate) with a",
        "% pass or fail verdict; RegTab*: the registered predictions of tab:registered-tests (the",
        "% diagnostic tests listed there are not predictions and are not counted); RegAll*: both.",
        "% *Mixed: held for some sub-rows or units and failed for others; registered readings",
        "% without pass or fail are not counted",
    ]
    lines += [rf"\newcommand{{\{name}}}{{{value}}}" for name, value in counts.items()]
    return "\n".join(lines) + "\n"


# ------------------------------------------------- registered verdicts outside the figure
MIXED = "mixed"
EPS = 0.03  # the plan's practical tolerance (3 pp), the bar of "D=1 survival ~ pass rate"
TABLE_COUNT_MACROS = ("RegTabJudged", "RegTabHeld", "RegTabFailed", "RegTabMixed")
ALL_COUNT_MACROS = ("RegAllJudged", "RegAllHeld", "RegAllFailed", "RegAllMixed")
TEST_PART = "Diagnostic tests (properties of a system, not predictions)"


@dataclass
class Registered:
    """One registered prediction or diagnostic test with a verdict that `fig_registered` does not
    plot: judged per switch, system, family or reader, or without an estimate on its pp or ratio
    scales. `units` is the verdict of each unit (shift, switch, system, ...) by its registered
    rule. `joint` is the staged verdict of a prediction whose registered rule spans its units (a
    count it must reach); None where each unit is judged on its own, so that the row holds,
    fails or is mixed by its units, as a row of the figure is by its sub-rows. A diagnostic test
    (`test`) is listed with its verdict per system (`note`) and is not counted."""

    group: str
    label: str
    rule: str
    estimate: str
    units: list[bool]
    joint: bool | None = None
    test: bool = False
    note: str = ""

    def verdict(self) -> str | None:
        if self.joint is not None:
            return HOLDS if self.joint else FAILS
        if not self.units:
            return None
        if all(self.units):
            return HOLDS
        return FAILS if not any(self.units) else MIXED

    def verdict_text(self) -> str:
        if self.note:
            return self.note
        n, k = len(self.units), sum(self.units)
        word = {HOLDS: "holds", FAILS: "fails", MIXED: "mixed"}.get(self.verdict() or "", "")
        return f"{word} ({k} of {n})" if n > 1 else word


def _pp(x: float) -> str:
    v = round(100 * x, 1)
    s = f"{abs(v):.1f}"
    return ("$+$" if v > 0 else "$-$" if v < 0 else "") + s


def _pp_range(xs: Sequence[float]) -> str:
    """The point estimates of a row's units in pp: one value, or the smallest and the largest."""
    if not xs:
        return "---"
    lo, hi = min(xs), max(xs)
    return _pp(lo) if round(100 * lo, 1) == round(100 * hi, 1) else f"{_pp(lo)} to {_pp(hi)}"


def _ests(pairs: Iterable[tuple[Any, bool | None]]) -> tuple[list[float], list[bool]]:
    """Point estimates and unit verdicts of the staged units (a unit without either is left
    out)."""
    xs, oks = [], []
    for e, ok in pairs:
        if _ok(e) and ok is not None:
            xs.append(float(e["est"]))
            oks.append(bool(ok))
    return xs, oks


def _reg(group: str, label: str, rule: str, xs: Sequence[float], oks: list[bool], **kw: Any
         ) -> list[Registered]:  # fmt: skip
    """One row in pp, or none when no unit is staged."""
    return [Registered(group, label, rule, _pp_range(xs), oks, **kw)] if oks else []


SD, ALLS = ("src", "def"), ("src", "def", "orphan")


def _main_tests(inp: Mapping[str, Any]) -> list[Registered]:
    """Main experiment: the contrasts of the v1 §4 list with a registered direction
    (`replication.as_predicted` on the main-study estimate) that the figure does not plot. The
    detection effects of the source-epoch key are 100 pp by construction and left out, as there."""
    con = _get(inp.get("e1"), "contrasts") or {}
    g = "Main experiment"

    def pos(keys: Iterable[tuple[str, str]]) -> tuple[list[float], list[bool]]:
        return _ests((e := _get(con, b, k), replication.as_predicted("effect", e)) for b, k in keys)

    return [
        *_reg(g, r"Cited-cluster key: $B$ on detection, $D{=}0$ and $D{=}1$", "$>0$",
              *pos((b, f"B_effect_detected_D{d}_cited") for d in (0, 1) for b in ALLS)),
        *_reg(g, r"Source-epoch key: $B$ on Update pooled over $D$, $\Sdef$ and $\Sdel$", "$>0$",
              *pos((b, "B_effect_update_pooled_source") for b in ("def", "orphan"))),
        *_reg(g, "Cited-cluster key: $D$ on repair", "$>0$",
              *pos((b, "D_effect_repair_cited") for b in ALLS)),
        *_reg(g, "Both keys: $D$ on repair (mechanistic)", "$>0$",
              *pos((b, f"D_effect_repair_mech_{k}") for k in ("source", "cited") for b in ALLS)),
    ]  # fmt: skip


def _switch_tests(inp: Mapping[str, Any]) -> list[Registered]:
    """Writer switches: D=1 survival against the checker pass rate (v0 §3, the three registered
    tiers; "~" read as the plan's 3 pp tolerance) and the survival of D=1 minus D=0 items in each
    of the eight switches (v2 M1-b, decisions.md 2026-09-29: a point estimate >= 0 holds)."""
    g = "Writer switch"
    out: list[Registered] = []
    diffs, oks = [], []
    for tier in ("same", "upgrade", "downgrade"):
        dc = _get(inp.get("e2"), "d1_vs_checker", tier) or {}
        surv, rate = dc.get("d1_b1_survival"), dc.get("checker_pass_rate_calibration")
        if isinstance(surv, int | float) and isinstance(rate, int | float):
            diffs.append(float(surv) - float(rate))
            oks.append(abs(float(surv) - float(rate)) <= EPS)
    out += _reg(g, r"$D{=}1$ survival minus checker pass rate, three tiers",
                r"within $\pm$3\,pp", diffs, oks)  # fmt: skip
    m1 = inp.get("m1")
    mb = [_get(m1, "m1b_d_protects_switch", k) or {} for t in TIERS if (k := tier_key(m1, t))]
    out += _reg(g, r"Survival of $D{=}1$ minus $D{=}0$ items, eight switches", r"$\ge0$ in each",
                *_ests((m.get("d1_minus_d0_survival"), m.get("holds")) for m in mb))  # fmt: skip
    return out


ABOVE, BELOW = r"$>0$, interval above 0", r"$<0$, interval below 0"


def _family_tests(inp: Mapping[str, Any]) -> list[Registered]:
    """Other model families (v2 §2) as writer and reader: checks (i), (iii) and (v), which the
    figure does not plot ((ii) and (iv) are its rows); as reader only: every sign of the main
    reader kept and the leak check not triggered, per reader (M3)."""
    g = "Other model families as writer and reader"
    fams = [f for f in (inp.get("m2") or {}).values() if f]
    out: list[Registered] = []
    for label, rule, checks in (
        ("Leak check", "not triggered", (("i_path2_not_triggered", None),)),
        (r"$D$ on repair, $\Ssrc$ and $\Sdef$", ABOVE,
         (("iii_D_repair_src", "D_repair_src"), ("iii_D_repair_def", "D_repair_def"))),
        (r"Zero system minus \holdmem{}(1,1), ContractValid, $\Sdef$", BELOW,
         (("v_zero_below_hm11_def", "zero_minus_hm11_def"),)),
    ):  # fmt: skip
        oks = [
            bool(c) for f in fams for key, _ in checks if (c := _get(f, "checks", key)) is not None
        ]
        xs = [float(e["est"]) for f in fams for _, name in checks
              if name and _ok(e := _get(f, "estimates", name))]  # fmt: skip
        out += _reg(g, label, rule, xs, oks)
    readers = inp.get("m3") or {}
    oks, kept, signed = [], 0, 0
    for f in FAMILIES:
        sw = readers.get(f.key)
        if not sw:
            continue
        signs = m3_signs(sw)
        leak = [replication.path2_verdict(s) == "not triggered" for b in SD
                if (s := _get(sw, b, "contrasts", "D_effect_missed_source", "swap"))]  # fmt: skip
        oks.append(bool(signs) and all(signs) and bool(leak) and all(leak))
        kept, signed = kept + sum(signs), signed + len(signs)
    if oks:
        out.append(Registered("Other model families as reader", "Main reader's signs kept, leak "
                              "check", "same sign, not triggered", f"{kept} of {signed} signs",
                              oks))  # fmt: skip
    return out


def _rendered_tests(inp: Mapping[str, Any]) -> list[Registered]:
    """Rewritten worlds (v2 §5): checks (i), (ii), (iii) and (v) of the other families; (iv) is a
    row of the figure."""
    m = inp.get("rendered") or {}
    rend = m.get("rendered") or {}
    g = "Rewritten worlds"
    leak = [abs(float(p["rendered"]["est"])) for p in m.get("paired") or []
            if p.get("check") == "i" and _ok(p.get("rendered"))]  # fmt: skip
    out: list[Registered] = []
    ok = _get(rend, "checks", "i_path2_not_triggered")
    if ok is not None:
        out += _reg(g, "Leak check (largest effect)", "not triggered", [max(leak)] if leak else [],
                    [bool(ok)])  # fmt: skip
    for label, rule, checks in (
        (r"$B$ on Update pooled over $D$, $\Ssrc$", r"$>$2\,pp, interval above 0",
         (("ii_B_update_src", "B_update_src"),)),
        (r"$D$ on repair, $\Ssrc$ and $\Sdef$", ABOVE,
         (("iii_D_repair_src", "D_repair_src"), ("iii_D_repair_def", "D_repair_def"))),
        (r"Zero system minus \holdmem{}(1,1), ContractValid, $\Sdef$", BELOW,
         (("v_zero_below_hm11_def", "zero_minus_hm11_def"),)),
    ):  # fmt: skip
        out += _reg(
            g,
            label,
            rule,
            *_ests(
                (_get(rend, "estimates", name), _get(rend, "checks", key)) for key, name in checks
            ),
        )
    return out


def _fresh_view(inp: Mapping[str, Any]) -> dict[str, Any]:
    """The main study against the fresh worlds of the replication (`replication_view`)."""
    return replication_view(
        {"e1": inp.get("e1"), "e2": inp.get("e2"), "e3": inp.get("e3")},
        {"e1": inp.get("rep_e1"), "e2": inp.get("rep_e2"), "e3": inp.get("rep_e3")},
    )


def _replication_tests(inp: Mapping[str, Any]) -> list[Registered]:
    """The leak and agreement checks re-judged on the fresh worlds (v2 §1; the gain check there
    is a row of the figure)."""
    paths = _fresh_view(inp).get("paths") or {}
    g = "Replication on fresh worlds"
    out: list[Registered] = []
    p2 = paths.get("path2")
    if p2:
        largest = [max((abs(c["estimate"]["est"]) for c in p2["checks"].values()), default=0.0)]
        out += _reg(g, "Leak check (largest effect)", "not triggered", largest,
                    [not p2["triggered"]])  # fmt: skip
    p3 = paths.get("path3")
    if p3:
        sl, au = p3["pooled_slope"], p3["pooled_auroc"]
        est = f"slope {sl['est']:.2f}, AUROC {au['est']:.2f}"
        out.append(Registered(g, "Agreement check", r"slope $>0$ (90\%), AUROC $\ge0.60$", est,
                              [not p3["triggered"]]))  # fmt: skip
    return out


CTL_TESTS = (  # registered control predictions that the figure does not plot: key, label, rule
    ("G2", "Oracle minus source-epoch key, Update, definition items",
     r"within $\pm$3\,pp, all six"),
    ("R1", r"Add-only refresh minus append-only, all items, $D{=}1$",
     r"between $B{=}0$ and $B{=}1$"),
    ("Z1k", "Query-time view minus zero system, definition items, before the shift",
     r"$\ge30$\,pp"),
    ("Z2", "Query-time view minus zero system, other items",
     r"single-field $\ge0$, free text $\le3$\,pp, Stability $\ge-3$\,pp"),
    ("M3", "Deletion-aware programs that run, relative to the others (validity)",
     r"$\ge0.8$ times"),
    ("D2", "Thinking writes cut off (validity)", r"$\le20$\%"),
)  # fmt: skip


def _control_tests(inp: Mapping[str, Any]) -> list[Registered]:
    """The controls (v3): each prediction's staged status decides (its rule needs a count of its
    contrasts); the contrasts give the estimate and the count."""
    verdicts = (inp.get("ctl") or {}).get("verdicts") or {}
    out = []
    for key, label, rule in CTL_TESTS:
        v = verdicts.get(key) or {}
        if v.get("status") not in ("holds", "fails"):
            continue
        tests = v.get("tests") or []
        vals = [float(t["value"]) for t in tests if isinstance(t.get("value"), int | float)]
        unit = tests[0].get("unit") if tests else "pp"
        if unit == "ratio":
            est = ", ".join(f"{x:.2f}" for x in vals) or "---"
        elif unit == "share":
            est = ", ".join(f"{100 * x:.1f}\\%" for x in vals) or "---"
        else:
            est = _pp_range(vals)
        oks = [bool(t.get("ok")) for t in tests]
        out.append(Registered("Controls", label, rule, est, oks, joint=v["status"] == "holds"))
    return out


def _other_tests(inp: Mapping[str, Any]) -> list[Registered]:
    """Sync (v0 §6 E4 and decisions.md 2026-09-28: the calendar interaction counts only if it
    reproduces on held-out worlds, interval excluding 0 and |estimate| >= 3 pp), export without
    the derivation extension (v1 §4 E6: detection degrades to that of B=0, every change missed)
    and the native-system predictions N-1 to N-4 (v2 §3)."""
    out: list[Registered] = []
    cal = _get(inp.get("e4x"), "primary", "L5")
    if _ok(cal):
        ok = bool(cal.get("excludes_zero")) and abs(float(cal["est"])) >= EPS
        est = rf"{_pp(cal['est'])} [{_pp(cal['lo'])}, {_pp(cal['hi'])}]"
        out.append(Registered("Sync and export", "Calendar interaction of official sync and "
                              r"binding", r"interval excludes 0, $|\cdot|\ge3$\,pp", est,
                              [ok]))  # fmt: skip
    missed = _get(inp.get("e6"), "imports", "imported_no_ext", "diff_90", "missed")
    if _ok(missed):
        out += _reg("Sync and export", "Import without the keys, missed invalidation minus the "
                    "original", r"$+$100\,pp (as $B{=}0$)", [float(missed["est"])],
                    [float(missed["est"]) >= 1.0 - 1e-9])  # fmt: skip
    s4 = _get(inp.get("nat"), "stage4") or {}
    per = s4.get("n2_per_system") or {}
    g = "Native systems"
    agree = s4.get("n1_label_agreement") or {}
    if agree and s4.get("n1_holds") is not None:
        k = sum(v is True for v in agree.values())
        out.append(Registered(g, "Inferred label equals the documented one", r"$\ge4$ of 5",
                              f"{k} of {len(agree)}", [v is True for v in agree.values()],
                              joint=s4["n1_holds"]))  # fmt: skip
    if per:
        inside = [v["inside"] for v in per.values()]
        out.append(Registered(g, "Comparisons inside the inferred cell, per system",
                              r"$\ge4$ of 6", f"{min(inside)} to {max(inside)} of 6",
                              [bool(v["n2_holds"]) for v in per.values()]))  # fmt: skip
    if per and s4.get("n3_holds") is not None:
        k = sum(v["inside"] for v in per.values())
        n = sum(v["comparisons"] for v in per.values())
        out.append(
            Registered(
                g,
                "Comparisons inside, all systems",
                "at least half",
                f"{k} of {n}",
                [bool(s4["n3_holds"])],
                joint=s4["n3_holds"],
            )
        )
    by_b = s4.get("n4_orphan_stability_by_B") or {}
    if s4.get("n4_holds") is not None and by_b.get("0") and by_b.get("1"):
        m0, m1 = (sum(by_b[b]) / len(by_b[b]) for b in ("0", "1"))
        out.append(Registered(g, r"Stability, $\Sdel$, $B{=}1$ minus $B{=}0$ systems", "$>0$",
                              _pp(m1 - m0), [bool(s4["n4_holds"])]))  # fmt: skip
    return out


def _passes(oks: Sequence[bool], names: Sequence[str] = ()) -> str:
    """The verdict cell of a diagnostic test: how many units pass, and which where named."""
    return f"passes {sum(oks)} of {len(oks)}" + (f" ({', '.join(names)})" if names else "")


def _diagnostic_tests(inp: Mapping[str, Any]) -> list[Registered]:
    """The four diagnostic tests with a registered threshold, per tier, switch, horizon and
    system: retention (v1 §5, per switch v2 §2), leaving (v1 §5), utility floor and cost ceiling
    (v1 §4 E3, natives v2 §3), and whether the fresh worlds reach the main-study verdicts."""
    g, out = TEST_PART, []
    e2, m1 = inp.get("e2"), inp.get("m1")
    t1 = [_get(e2, "t1", t) or {} for t in ("same", "upgrade", "downgrade")]
    xs, oks = _ests((t.get("delta_90"), t.get("equivalent")) for t in t1)
    out += _reg(g, r"Retention test, three registered tiers (change)", r"interval above $-$3\,pp",
                xs, oks, test=True, note=_passes(oks))  # fmt: skip
    cross = [
        (t.label, _get(m1, "t1", k) or {}) for t in TIERS if t.marker and (k := tier_key(m1, t))
    ]
    xs, oks = _ests((t.get("delta_90"), t.get("equivalent")) for _, t in cross)
    names = [lab for (lab, t) in cross if t.get("equivalent")]
    out += _reg(g, r"Retention test, five other writers (change)", r"interval above $-$3\,pp",
                xs, oks, test=True, note=_passes(oks, names))  # fmt: skip
    t2 = _get(inp.get("e6"), "imports", "imported_with_ext")
    if t2 and _ok(_get(t2, "diff_90", "post")):
        out += _reg(g, r"Leaving test, \holdmem{}(1,1) with its keys", r"within $\pm$3\,pp",
                    [float(t2["diff_90"]["post"]["est"])], [bool(t2.get("t2_equivalent"))],
                    test=True, note="passes" if t2.get("t2_equivalent") else "fails")  # fmt: skip
    horizons = (_get(inp.get("e3"), "horizons") or {}).values()
    names3 = (("holdmem_D1B1_source", r"\holdmem{}(1,1)"), ("zero", "zero system"),
              ("rag", "plain RAG"))  # fmt: skip
    per = {s: [h[s] for h in horizons if isinstance(h.get(s), dict) and "T3_pass" in h[s]]
           for s, _ in names3}  # fmt: skip
    if any(per.values()):
        note = ", ".join(f"{lab} {sum(bool(x['T3_pass']) for x in per[s])} of {len(per[s])}"
                         for s, lab in names3 if per[s])  # fmt: skip
        xs = [float(x["minus_oracle"]["est"]) for s, _ in names3 for x in per[s]
              if _ok(x.get("minus_oracle"))]  # fmt: skip
        out += _reg(g, "Utility floor, three horizons", r"$\ge-3$\,pp", xs,
                    [bool(x["T3_pass"]) for s, _ in names3 for x in per[s]], test=True,
                    note="passes: " + note)  # fmt: skip
        cost = [x for s, _ in names3 for x in per[s] if "T4_pass" in x]
        if cost:
            ratio = max(float(x["serve_ratio_to_full_reread"]) for x in cost)
            out.append(
                Registered(
                    g,
                    "Cost ceiling, three horizons",
                    "at most half",
                    f"at most {100 * ratio:.1f}\\%",
                    [bool(x["T4_pass"]) for x in cost],
                    test=True,
                    note=_passes([bool(x["T4_pass"]) for x in cost]),
                )
            )
    mem = _get(inp.get("nat"), "membership") or {}
    named = [(n.label, n.key) for n in NATIVES] + [("zero system", "zero"), ("plain RAG", "rag")]
    systems: list[tuple[str, dict[str, Any]]] = [
        (lab, m) for lab, key in named if isinstance(m := mem.get(key), dict) and "T3_pass" in m
    ]
    if systems:
        passing = [lab for lab, m in systems if m["T3_pass"]]
        out += _reg(g, "Utility floor, native systems and the two baselines", r"$\ge-3$\,pp",
                    [float(m["minus_oracle"]["est"]) for _, m in systems
                     if _ok(m.get("minus_oracle"))],
                    [bool(m["T3_pass"]) for _, m in systems], test=True,
                    note=_passes([bool(m["T3_pass"]) for _, m in systems], passing))  # fmt: skip
        ratio = max(float(m["serve_ratio_to_full_reread"]) for _, m in systems)
        out.append(
            Registered(
                g,
                "Cost ceiling, the same systems",
                "at most half",
                f"at most {100 * ratio:.1f}\\%",
                [bool(m["T4_pass"]) for _, m in systems],
                test=True,
                note=_passes([bool(m["T4_pass"]) for _, m in systems]),
            )
        )
    view = _fresh_view(inp)
    same = [v["verdict"] == "replicated" for k in ("t1", "t3", "t4") for v in view[k].values()]
    if same:
        out.append(Registered(g, "Retention, floor and ceiling on fresh worlds",
                              "main-study verdict", "---", same, test=True,
                              note=f"same in {sum(same)} of {len(same)}"))  # fmt: skip
    return out


def registered_tests(inp: Mapping[str, Any]) -> list[Registered]:
    """The data of `tab_registered_tests`: every registered prediction with a verdict that the
    figure does not plot, by experiment, then the diagnostic tests. A prediction that is not
    staged, or not judged, has no row."""
    return [
        *_main_tests(inp),
        *_switch_tests(inp),
        *_family_tests(inp),
        *_rendered_tests(inp),
        *_replication_tests(inp),
        *_control_tests(inp),
        *_other_tests(inp),
        *_diagnostic_tests(inp),
    ]


def table_counts(tests: Sequence[Registered]) -> dict[str, int]:
    """Held, failed and mixed registered predictions of the table (its tests are not counted)."""
    words = [t.verdict() for t in tests if not t.test]
    held, failed, mixed = (sum(w == x for w in words) for x in (HOLDS, FAILS, MIXED))
    return dict(zip(TABLE_COUNT_MACROS, (held + failed + mixed, held, failed, mixed), strict=True))


def paper_counts(inp: Mapping[str, Any]) -> dict[str, int]:
    """The counts of registered_counts.tex: the figure's rows, the table's predictions and both."""
    fig = registered_counts(registered_rows(inp))
    tab = table_counts(registered_tests(inp))
    both = [fig[a] + tab[b] for a, b in zip(COUNT_MACROS, TABLE_COUNT_MACROS, strict=True)]
    return {**fig, **tab, **dict(zip(ALL_COUNT_MACROS, both, strict=True))}


def tab_registered_tests(tests: Sequence[Registered]) -> str:
    """The table (paper/generated/tab_registered_tests.tex): one row per registered prediction
    or diagnostic test, grouped by experiment, with its rule, its estimate(s) and its verdict."""
    counts = table_counts(tests)
    lines = [
        "% generated by memshift.analysis.figures --only fig_registered; do not edit",
        r"\begin{table}[t]",
        r"\caption{The registered predictions that \cref{fig:registered} does not plot, and the "
        rf"four diagnostic tests. Of the {counts['RegTabJudged']} predictions, "
        rf"{counts['RegTabHeld']} hold, {counts['RegTabFailed']} fail and "
        rf"{counts['RegTabMixed']} hold for some units but not others. Estimate: the point "
        r"estimate in pp (a range over the units: shifts, switches, families, readers or "
        r"systems) unless stated. Verdict: by the registered criterion, with the number of units "
        r"that meet it where a row has several. A diagnostic test is a property of a system, "
        r"not a prediction, and is not counted. Retention test: the change in retention of "
        r"\holdmem{}(1,1). Utility floor: ContractValid before any shift minus that of the "
        r"gold-dependency context. Cost ceiling: serving tokens per query as a share of a full "
        r"re-read.}",
        r"\label{tab:registered-tests}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\newcolumntype{Y}[1]{>{\raggedright\arraybackslash}p{#1}}",
        r"\begin{tabular}{@{}Y{6.6cm} Y{3.4cm} Y{2.6cm} Y{3.1cm}@{}}",
        r"\toprule",
        r"prediction or test & registered criterion & estimate & verdict \\",
    ]
    group = None
    for t in tests:
        if t.group != group:
            group = t.group
            lines += [r"\midrule", rf"\multicolumn{{4}}{{@{{}}l}}{{\textit{{{group}}}}} \\"]
        lines.append(rf"{t.label} & {t.rule} & {t.estimate} & {t.verdict_text()} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


VERDICT_STYLE = {  # verdict -> (colour, filled): holds and fails differ in greyscale by the fill
    HOLDS: (fs.BLUE, True),
    FAILS: (fs.VERMILION, False),
    READING: (fs.GREY, True),
}
SUB_H = 0.101  # inches per sub-row of the forest plot (the second page fits the text height)
ROW_GAP = 0.03  # inches between two predictions
HEAD_H, AXIS_H, PANEL_GAP = 0.17, 0.23, 0.05  # panel title, x axis, space between panels
PAD = 0.02  # inches above the first and below the last sub-row of a panel
LABEL_CHARS = 44  # a longer row label takes two lines
LABEL_W, SUB_W = 2.2, 0.56  # widths (in) of the row labels and of the sub-row labels
BAR_COLOUR, BAND_COLOUR = "#7A8A9A", "#E4E9EF"


def _span(rows: Sequence[Row], k: float) -> tuple[float, float]:
    """The x range of a panel: its intervals and cut points, with 0, padded by 6%."""
    vals = [k * v for r in rows for p in r.points for v in (p["lo"], p["hi"])]
    vals += [c for r in rows for c in r.bar[1] for c in ((-c, c) if "abs" in r.bar[0] else (c,))]
    lo, hi = min([0.0, *vals]), max([0.0, *vals])
    pad = 0.06 * (hi - lo or 1.0)
    return lo - pad, hi + pad


def _rule(ax: Any, top: float, bottom: float, bar: Bar) -> None:
    """The registered bar inside its row: a band for a two-sided margin, a tick at every cut
    point other than 0 (the zero line marks 0)."""
    kind, vals = bar
    if kind.startswith("abs"):
        ax.fill_between([-vals[0], vals[0]], bottom, top, color=BAND_COLOUR, lw=0, zorder=0.6)
        cuts = [-vals[0], vals[0]]
    else:
        cuts = [c for c in vals if c != 0]
    for c in cuts:
        ax.plot([c, c], [bottom, top], color=BAR_COLOUR, lw=0.9, zorder=0.8,
                solid_capstyle="butt")  # fmt: skip


def _lines(label: str) -> int:
    """Lines a row label takes in the label column (its printed characters, roughly)."""
    import re

    text = re.sub(r"\\[a-zA-Z]+|[${}~]", "", label)
    return 1 if len(text) <= LABEL_CHARS else 2


def _row_height(row: Row) -> float:
    return max(len(row.subrows()), _lines(row.label)) * SUB_H


def _panel_height(rows: Sequence[Row]) -> float:
    return sum(map(_row_height, rows)) + max(len(rows) - 1, 0) * ROW_GAP + 2 * PAD


def _draw_panel(fig: Any, rect: tuple[float, float, float, float], rows: Sequence[Row],
                title: str, unit: str, height: float) -> Any:  # fmt: skip
    """One question: its rows top to bottom in inches (the y data unit is the inch)."""
    ax = fig.add_axes(rect)
    k = 100.0 if unit == "pp" else 1.0
    xlo, xhi = _span(rows, k)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(-height, 0)
    if xlo < 0 < xhi:
        ax.axvline(0, color=fs.GREY, lw=0.6, zorder=1)
    left = -(LABEL_W + SUB_W) / (rect[2] * fs.TEXT)  # axes fraction of the row labels' left edge
    y = -PAD
    for row in rows:
        subs = row.subrows()
        top, bottom = y, y - _row_height(row)
        y0 = (top + bottom) / 2 + len(subs) * SUB_H / 2  # the sub-rows, centred in the row
        _rule(ax, y0, y0 - len(subs) * SUB_H, row.bar)
        lab = rf"\parbox[c]{{{LABEL_W - 0.08:.2f}in}}{{\raggedright {row.label}}}"
        ax.text(left, (top + bottom) / 2, lab, transform=ax.get_yaxis_transform(), ha="left",
                va="center", fontsize=7.5, linespacing=0.95)  # fmt: skip
        for j, (sub, pts) in enumerate(subs):
            yy = y0 - (j + 0.5) * SUB_H
            ax.text(-0.012, yy, sub, transform=ax.get_yaxis_transform(), ha="right",
                    va="center", fontsize=7, color=fs.GREY)  # fmt: skip
            for p in pts:
                colour, filled = VERDICT_STYLE[p["verdict"]]
                fs.interval_bar(ax, yy, k * p["est"], k * p["lo"], k * p["hi"], colour,
                                filled=filled, size=4.2)  # fmt: skip
        y = bottom - ROW_GAP
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.tick_params(axis="x", labelsize=7, pad=1.5)
    ax.text(left, 1.0, rf"\textbf{{{title}}} ({unit})", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=7.5)  # fmt: skip
    return ax


def _legend(fig: Any, top: float) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        fs.interval_handle(VERDICT_STYLE[HOLDS][0], True, "holds"),
        fs.interval_handle(VERDICT_STYLE[FAILS][0], False, "fails"),
        fs.interval_handle(VERDICT_STYLE[READING][0], True, "registered reading"),
        Line2D([], [], color=BAR_COLOUR, lw=0.9, marker="|", ms=7, ls="none",
               label="registered cut point"),
        Patch(facecolor=BAND_COLOUR, edgecolor="none", label="registered margin"),
    ]  # fmt: skip
    fig.legend(handles=handles, handler_map=fs.INTERVAL_HANDLERS, loc="upper center",
               bbox_to_anchor=(0.5, top), ncol=5, fontsize=7.5, columnspacing=1.1,
               handletextpad=0.4, handlelength=1.8, borderaxespad=0)  # fmt: skip


LEGEND_H = 0.26


def _draw_page(rows: Mapping[str, list[Row]], page: int, path: Path) -> Path:
    plt = fs.plt()
    panels = [(key, title, unit) for key, title, unit, pg in PANELS if pg == page and rows[key]]
    heights = [_panel_height(rows[key]) for key, _, _ in panels]
    legend = LEGEND_H if page == 1 else 0.0  # the second page continues the first
    total = legend + sum(HEAD_H + h + AXIS_H + PANEL_GAP for h in heights) + 0.02
    fig = plt.figure(figsize=(fs.TEXT, max(total, 1.0)), layout="none")
    height = max(total, 1.0)
    x0 = (LABEL_W + SUB_W) / fs.TEXT
    width = 1 - x0 - 0.12 / fs.TEXT
    if legend:
        _legend(fig, 1 - 0.02 / height)
    y = height - legend
    for (key, title, unit), h in zip(panels, heights, strict=True):
        y -= HEAD_H + h
        _draw_panel(fig, (x0, y / height, width, h / height), rows[key], title, unit, h)
        y -= AXIS_H + PANEL_GAP
    if not panels:
        fig.text(0.5, 0.5, "no registered prediction staged", ha="center", fontsize=7.5,
                 color=fs.GREY)  # fmt: skip
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_registered(inp: Mapping[str, Any], out: Path) -> Path:
    """Every registered prediction with an estimate, by the question it answers, on two pages
    (fig_registered.pdf, fig_registered_2.pdf): one panel and x axis per question; per
    prediction one sub-row per shift (or arm), the estimate with its 95% interval as an interval
    bar (path 3: 90%), blue and filled where the prediction holds, vermilion and outlined where it
    fails, grey for a registered reading without pass or fail; the registered bar inside the row
    (a band for a two-sided margin, a tick at a cut point other than 0)."""
    rows = registered_rows(inp)
    _draw_page(rows, 2, out / "fig_registered_2.pdf")
    return _draw_page(rows, 1, out / "fig_registered.pdf")


FIGURES: dict[str, Callable[[Mapping[str, Any], Path], Path]] = {
    "fig_writers": fig_writers,
    "fig_reader": fig_reader,
    "fig_misses": fig_misses,
    "fig_registered": fig_registered,
}
