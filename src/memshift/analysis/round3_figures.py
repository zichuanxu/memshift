"""Figures of the round-3 results and the diagnostic-tests table (regenerated from the staged
analyses named in results/sources.json; nothing here calls a model or reads a raw run).

fig_writers     Section 4.2: (a) per program set (round3 g_b.json), B x D at D_m on definition
                items after S_def against how often the set's programs computed the right value at
                the fork, before any shift (ex_round3 writers.json), with the authored
                derivation's B x D on the same items as reference (main study); (b, c) per
                definition-item writer (Qwen3.5-9B, Claude Sonnet 5.5, Claude Opus 5.5) B x D with
                the authored derivation against r_0, both on the same worlds, after S_def (b) and
                S_src (c) (writers.json)
fig_reader      appendix: grouped bars of the store part of B x D on Update and of B x D with each
                reader of the same stores (y.json and the committed reader's split, ex_round3
                store_reader.json)
fig_misses      appendix: why the cited-cluster key misses changed item-times, by shift and item
                class, main study and replication (ex_misses.json)
tab_diagnostic_tests (paper/generated): the four diagnostic tests per set of systems, with their
                thresholds, estimates and how many units pass, in the main study and on the fresh
                worlds of the replication

Each figure has a data function (`writer_points`, `reader_rows`, `miss_rows`, and
`diagnostic_tests` for the table) that reads the inputs of `load` and returns plain records; an
input that is not staged leaves its records out, and the figure draws what is there.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from memshift.analysis import figstyle as fs
from memshift.analysis.paper_data import NATIVES, TIERS, stage_json, tier_key
from memshift.analysis.paper_data import get as _get

QWEN = "Qwen/Qwen3.5-9B"
H_MODELS = ("claude-sonnet-5-5", "claude-opus-5-5")
DH_SHIFTS = ("src", "def")  # the shifts of panel (b) of the writer figure
CLASS_NAMES = (("definition", "definition items"), ("other", "other items"))
SHIFT_ROWS = (("src", "source revision"), ("def", "redefinition"), ("orphan", "deletion"))


# ------------------------------------------------------------------------------------ loading
def load(root: Path, sources: Mapping[str, str]) -> dict[str, Any]:
    """The staged inputs of the three figures and of the diagnostic-tests table (None where a
    stage or file is not there)."""
    src = dict(sources)
    return {
        "g_b": stage_json(root, src, "round3", "g_b.json"),
        "y": stage_json(root, src, "round3", "y.json"),
        "writers": stage_json(root, src, "ex_round3", "writers.json"),
        "sr": stage_json(root, src, "ex_round3", "store_reader.json"),
        "miss": stage_json(root, src, "ex_misses", "ex_misses.json"),
        "e1": stage_json(root, src, "e1", "e1_metrics.json"),
        "e2": stage_json(root, src, "e2", "e2_metrics.json"),
        "e3": stage_json(root, src, "e3", "e3_metrics.json"),
        "rep_e2": stage_json(root, src, "rep_e2", "e2_metrics.json"),
        "rep_e3": stage_json(root, src, "rep_e3", "e3_metrics.json"),
        "m1": stage_json(root, src, "m1", "e2_metrics.json"),
        "nat": stage_json(root, src, "n", "e5_natives.json"),
        "e6": stage_json(root, src, "e6", "e6_t2.json"),
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


def _bbox(x0: float, y0: float, x1: float, y1: float) -> Any:
    from matplotlib.transforms import Bbox

    return Bbox.from_extents(x0, y0, x1, y1)


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

    def block_marker(
        self, x: float, y: float, size_pt: float, dx_pt: float = 0.0, dy_pt: float = 0.0
    ) -> None:
        from matplotlib.transforms import Bbox

        px, py = self.ax.transData.transform((x, y))
        px, py = px + dx_pt * _pt(self.ax.figure), py + dy_pt * _pt(self.ax.figure)
        r = size_pt * _pt(self.ax.figure) / 2 + 1
        self.obstacles.append(Bbox.from_extents(px - r, py - r, px + r, py + r))

    def block_segment(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        width_pt: float,
        dx_pt: float = 0.0,
        dy_pt: float = 0.0,
    ) -> None:
        from matplotlib.transforms import Bbox

        (ax0, ay0), (ax1, ay1) = self.ax.transData.transform([(x0, y0), (x1, y1)])
        ax0, ax1 = ax0 + dx_pt * _pt(self.ax.figure), ax1 + dx_pt * _pt(self.ax.figure)
        ay0, ay1 = ay0 + dy_pt * _pt(self.ax.figure), ay1 + dy_pt * _pt(self.ax.figure)
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


def _dodge(
    ax: Any, pts: Sequence[Mapping[str, Any]], size_pt: float, diagonal: bool = False
) -> list[tuple[float, float]]:
    """Offsets (dx, dy in points) that set apart markers drawn on top of each other: markers
    closer than one marker width form a cluster, whose members are drawn side by side in the
    order of their values, centred on the cluster. With `diagonal` (marks with intervals on both
    axes) they step as far down as across, so that both their horizontal and their vertical
    intervals stay apart: the first member up and left, the last down and right."""
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
    out = [(0.0, 0.0)] * len(pts)
    for r in set(map(root, range(len(pts)))):
        members = sorted((i for i in range(len(pts)) if root(i) == r),
                         key=lambda i: (pos[i][0], pts[i]["writer"], pos[i][1]))  # fmt: skip
        for k, i in enumerate(members):
            step = (k - (len(members) - 1) / 2) * size_pt * DODGE
            out[i] = (step, -step) if diagonal else (step, 0.0)
    return out


WRITER_MS = 4.6  # marker size (points) of the writer figure
DODGE = 1.3  # spacing of side-by-side markers, in marker widths: a thin white gap between them


def _writer_marks(ax: Any, pts: Sequence[Mapping[str, Any]], lab: _Labeller, xerr: bool) -> None:
    """Markers (colour = writer, shape = shift) with their y (and x) intervals, markers of
    different writers on one spot set apart by writer (side by side, or along the diagonal when
    the x intervals are drawn too); each mark blocks the direct labels."""
    from matplotlib.transforms import ScaledTranslation

    fig = ax.figure
    for p, (dx, dy) in zip(pts, _dodge(ax, pts, WRITER_MS, diagonal=xerr), strict=True):
        _, colour = fs.model_style(p["writer"])
        tr = ax.transData + ScaledTranslation(dx / 72, dy / 72, fig.dpi_scale_trans)
        x, y = 100 * p["x"]["est"], 100 * p["y"]["est"]
        bars = [([x, x], [100 * p["y"]["lo"], 100 * p["y"]["hi"]])]
        if xerr:
            bars.append(([100 * p["x"]["lo"], 100 * p["x"]["hi"]], [y, y]))
        for xs, ys in bars:
            ax.plot(xs, ys, color=colour, lw=0.85, alpha=0.8, zorder=3, transform=tr,
                    solid_capstyle="butt")  # fmt: skip
            lab.block_segment(xs[0], ys[0], xs[1], ys[1], 0.85, dx, dy)
        ax.plot([x], [y], marker=fs.SHIFTS[p["shift"]][1], ms=WRITER_MS, mfc=colour,
                mec="white", mew=0.5, ls="none", zorder=4, transform=tr)  # fmt: skip
        lab.block_marker(x, y, WRITER_MS + 1, dx, dy)


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
                               fontsize=8, color=colour)  # fmt: skip
            edge = (max if up else min)(100 * p["y"]["hi" if up else "lo"] for p in mine)
            beyond = [
                (dx, sign * g, ha, va)
                for g in (2.0, 5.0, 9.0)
                for dx, ha in ((0.0, "center"), (-hw, "left"), (hw, "right"))
            ]
            if placed is None:
                placed = lab.place(label, (cx, edge), beyond, fallback=False, fontsize=8,
                                   color=colour)  # fmt: skip
        if placed is None:
            lab.place(label, (cx, (min(ys) + max(ys)) / 2), _around(hw, hh), fontsize=8,
                      color=colour)  # fmt: skip


WRITERS_H = 1.65  # height of the writer figure (in), full width
R0_LABEL = r"$r_0$: repair correctness at $D{=}0$ (\%)"


def _panel_title(ax: Any, letter: str, text: str) -> None:
    """The panel letter in bold, then what the panel shows, at the top left."""
    ax.set_title(rf"\textbf{{({letter})}} {text}", loc="left", fontsize=9, pad=4)


def fig_writers(inp: Mapping[str, Any], out: Path) -> Path:
    """(a) G-b program sets after S_def, (b, c) definition-item writers after S_def and S_src:
    see the module docstring. Each panel's title names its shift in words (the main text uses no
    S_* symbols), and the marker shape repeats it (`figstyle.SHIFTS`); identical sets of one writer
    draw as one marker, and marks of different writers on one spot are set apart by writer (the
    caption says both). (b) and (c) share their x axis, whose title is centred under the two."""
    plt = fs.plt()
    data = writer_points(inp)
    fig, (axa, axb, axc) = plt.subplots(
        1, 3, figsize=(fs.TEXT, WRITERS_H), sharey=True,
        gridspec_kw={"width_ratios": [1.75, 1, 1]},
    )  # fmt: skip
    for ax in (axa, axb, axc):
        ax.axhline(0, color=fs.GREY, lw=0.6, zorder=1)
        ax.set_ylim(-8, 55)  # room above the authored derivation's band for its name and a writer's
        ax.set_yticks([0, 20, 40])
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
    axa.set_xlim(-4, 104)
    axa.set_xticks([0, 25, 50, 75, 100])
    for ax in (axb, axc):
        ax.set_xlim(-3, 63)
        ax.set_xticks([0, 20, 40, 60])
        ax.set_xlabel(r"$r_0$ (\%)")  # holds the row's height; replaced below
    axa.set_ylabel(r"$B{\times}D$, definition items (pp)")
    axa.set_xlabel(r"programs right before the shift (\%)")
    _panel_title(axa, "a", "program sets: redefinition")
    _panel_title(axb, "b", "writers: redefinition")
    _panel_title(axc, "c", "writers: source revision")
    ref = data["reference"]
    if ref:
        axa.axhspan(100 * ref["lo"], 100 * ref["hi"], color=fs.LIGHT, alpha=0.55, lw=0, zorder=0.5)
        axa.axhline(100 * ref["est"], color=fs.GREY, lw=0.9, ls=(0, (4, 2)), zorder=1)
    fig.canvas.draw()  # settle the layout: label placement works in display space
    fig.set_layout_engine("none")  # and keep it, so the shared x title can span (b) and (c)
    renderer = fig.canvas.get_renderer()
    # one x title for (b) and (c), centred under both, at the height of (a)'s
    top = axb.transAxes.inverted().transform((0, axb.xaxis.label.get_window_extent(renderer).y1))
    bb, bc = axb.get_position(), axc.get_position()
    axb.set_xlabel(R0_LABEL)
    axb.xaxis.set_label_coords(((bb.x0 + bc.x1) / 2 - bb.x0) / bb.width, top[1])
    axc.set_xlabel("")
    labs = {ax: _Labeller(ax, renderer) for ax in (axa, axb, axc)}
    for ax, lab in labs.items():  # no label across the zero line
        lab.block_segment(ax.get_xlim()[0], 0, ax.get_xlim()[1], 0, 1.0)

    # (a) program sets, shape of S_def; the authored derivation's B x D as reference
    lab_a = labs[axa]
    if ref:
        lab_a.block_segment(-4, 100 * ref["est"], 104, 100 * ref["est"], 1.2)
        # no writer's name inside the band: names go above or below it
        lab_a.block(axa.transData.transform_bbox(_bbox(-4, 100 * ref["lo"], 104, 100 * ref["hi"])))
        t = axa.text(1, 100 * ref["hi"] + 1.0, "exact recomputation", fontsize=8,
                     color=fs.GREY, ha="left", va="bottom")  # fmt: skip
        lab_a.block(t.get_window_extent(renderer).expanded(1.05, 1.2))
    groups = [{**g, "shift": "def"} for g in data["groups"]]
    _writer_marks(axa, groups, lab_a, xerr=False)
    _writer_names(axa, groups, lab_a)
    if not groups:
        axa.text(0.5, 0.5, "no program set staged", transform=axa.transAxes, ha="center",
                 fontsize=8, color=fs.GREY)  # fmt: skip

    # (b, c) definition-item writers, one shift per panel: intervals on both axes
    for ax, b in ((axb, "def"), (axc, "src")):
        pts = [p for p in data["defwriters"] if p["shift"] == b]
        _writer_marks(ax, pts, labs[ax], xerr=True)
        _writer_names(ax, pts, labs[ax])
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


# the bars of each shift of the reader figure, left to right: key, label, fill (the store part
# light grey, each reader in its model colour)
READER_BARS = (
    ("store", "store part (both readers)", "#C8C8C8"),
    ("qwen", f"{fs.model_style(QWEN)[0]} reader", fs.model_style(QWEN)[1]),
    ("sonnet", f"{fs.model_style('claude-sonnet-5-5')[0]} reader",
     fs.model_style("claude-sonnet-5-5")[1]),
)  # fmt: skip
READER_BAR_W = 0.26  # width of one bar, in shifts


def fig_reader(inp: Mapping[str, Any], out: Path) -> Path:
    """Grouped bars per item class (panel) and shift (group): the store part of B x D on Update
    (light grey, the same for both readers) and B x D with each reader of the same stores (in the
    reader's model colour), each with its 95% interval as an error bar. What a reader adds or loses
    is the gap between the grey bar and its own. The panels share one pp axis."""
    from matplotlib.patches import Patch

    plt = fs.plt()
    rows = reader_rows(inp)
    vals = [100 * r[k][e] for r in rows for k, *_ in READER_BARS if r[k] for e in ("lo", "hi")]
    lo, hi = min([0.0, *vals]), max([10.0, *vals])
    fig, axes = plt.subplots(1, 2, figsize=(fs.TEXT, 2.2), sharey=True)
    for ax, (cls, title) in zip(axes, CLASS_NAMES, strict=True):
        ax.axhline(0, color=fs.BLACK, lw=0.6, zorder=2.5)
        for i, (b, _) in enumerate(SHIFT_ROWS):
            r = next((r for r in rows if r["cls"] == cls and r["branch"] == b), None)
            if r is None:
                continue
            for k, (key, _, fill) in enumerate(READER_BARS):
                if e := r[key]:
                    fs.ci_bar(ax, i + (k - 1) * READER_BAR_W, 100 * e["est"], 100 * e["lo"],
                              100 * e["hi"], fill, READER_BAR_W)  # fmt: skip
        ax.set_title(title, pad=3)
        ax.set_xticks(range(len(SHIFT_ROWS)), [t for _, t in SHIFT_ROWS])
        ax.set_xlim(-0.6, len(SHIFT_ROWS) - 0.4)
        ax.tick_params(axis="x", length=0)
        ax.spines["bottom"].set_visible(False)
    axes[0].set_ylim(lo - 1.5, hi + 1.5)
    axes[0].yaxis.set_major_locator(plt.MultipleLocator(10))
    axes[0].set_ylabel(r"$B{\times}D$ on Update (pp)")
    handles = [Patch(facecolor=fill, edgecolor="none", label=label)
               for _, label, fill in READER_BARS]  # fmt: skip
    fig.legend(handles=handles, loc="outside upper center", ncol=3, columnspacing=2.2,
               handlelength=1.2, fontsize=8)  # fmt: skip
    path = out / "fig_reader.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


# ==================================================================================== misses
# key -> (label, fill, a write-time key could have covered it): greys for the causes a write-time
# set could have contained, warm colours for the two it cannot, the strong one on the dominant
# cause (b new)
MISS_CAUSES = (
    ("a", r"$a$: an uncited record changed", "#8C8C8C", True),
    ("b_old", r"$b$ old: existing record entered scope", "#C8C8C8", True),
    ("b_new", r"$b$ new: a record created after the derivation", fs.VERMILION, False),
    ("c_left", r"$c$: records left window or scope", "#F0A877", False),
    ("c_rest", r"$c$: other", "#E8E8E8", True),
)
MISS_STAGES = (("main", "main study"), ("rep", "replication"))
MISS_COL_PT = 14.0  # right edge of the column of counts, in points right of the bars' end


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
    main study and replication (panels), with the number missed in a column right of the bars
    under the header $n$."""
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
        count = {"xycoords": ("axes fraction", "data"), "textcoords": "offset points",
                 "ha": "right", "fontsize": 8, "color": fs.BLACK}  # fmt: skip
        for r in (r for r in rows if r["stage"] == stage):
            yy, left = ypos[r["branch"], r["cls"]], 0.0
            for key, _, colour, _ in causes:
                w = 100 * r["shares"][key]
                if w > 0:
                    ax.barh([yy], [w], left=[left], height=0.72, color=colour,
                            edgecolor="white", lw=0.4, zorder=2)  # fmt: skip
                left += w
            # the number missed, right-aligned in a column right of the bars
            ax.annotate(f"{r['missed']}", (1.0, yy), xytext=(MISS_COL_PT, 0), va="center",
                        **count)  # fmt: skip
        ax.set_title(title, pad=3)
        if any(r["stage"] == stage for r in rows):  # the column's header, in the title row
            ax.annotate("$n$", (1.0, 1.0), xytext=(MISS_COL_PT, 3), va="baseline",
                        **{**count, "xycoords": "axes fraction"})  # fmt: skip
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
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
               handlelength=1.2, fontsize=8)  # fmt: skip
    path = out / "fig_misses.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


# ========================================================================== diagnostic tests
@dataclass
class DiagnosticTest:
    """One diagnostic test on one set of systems or units: its threshold, the point estimate (or
    the range over the units) and how many units pass; `fresh` on the fresh worlds of the
    replication. The label is the test's name, a comma, then the systems or units."""

    label: str
    threshold: str
    estimate: str
    outcome: str
    fresh: bool = False


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
    """Point estimates and pass flags of the staged units (a unit without either is left out)."""
    xs, oks = [], []
    for e, ok in pairs:
        if _ok(e) and ok is not None:
            xs.append(float(e["est"]))
            oks.append(bool(ok))
    return xs, oks


def _passes(oks: Sequence[bool], names: Sequence[str] = ()) -> str:
    """The outcome cell of a diagnostic test: how many units pass, and which where named."""
    return f"passes {sum(oks)} of {len(oks)}" + (f" ({', '.join(names)})" if names else "")


def _test(label: str, threshold: str, xs: Sequence[float], oks: Sequence[bool], outcome: str
          ) -> list[DiagnosticTest]:  # fmt: skip
    """One row in pp, or none when no unit is staged."""
    return [DiagnosticTest(label, threshold, _pp_range(xs), outcome)] if oks else []


TIER_KEYS = ("same", "upgrade", "downgrade")
HORIZON_SYSTEMS = (("holdmem_D1B1_source", r"\holdmem{}(1,1)"), ("zero", "no-store BM25"),
                   ("rag", "plain RAG"))  # fmt: skip


def _retention(e2: Any) -> tuple[list[float], list[bool]]:
    """The retention test of HoldMem(1,1) at the three tiers: change and pass per tier."""
    return _ests((t.get("delta_90"), t.get("equivalent"))
                 for t in (_get(e2, "t1", k) or {} for k in TIER_KEYS))  # fmt: skip


def _per_horizon(e3: Any) -> dict[str, list[dict[str, Any]]]:
    """Per system of HORIZON_SYSTEMS its results at the horizons that report the utility floor."""
    horizons = (_get(e3, "horizons") or {}).values()
    return {s: [h[s] for h in horizons if isinstance(h.get(s), dict) and "T3_pass" in h[s]]
            for s, _ in HORIZON_SYSTEMS}  # fmt: skip


def _floor_counts(per: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
    """How many horizons pass the utility floor, per system."""
    return ", ".join(f"{lab} {sum(bool(x['T3_pass']) for x in per[s])} of {len(per[s])}"
                     for s, lab in HORIZON_SYSTEMS if per[s])  # fmt: skip


def _diagnostic_tests(inp: Mapping[str, Any]) -> list[DiagnosticTest]:
    """The four diagnostic tests with their thresholds, per tier, switch, horizon and system:
    retention (per tier and per switch to another writer), leaving, utility floor and cost
    ceiling (main-study systems at three horizons, native systems and the two baselines), and
    the retention test, utility floor and cost ceiling on the fresh worlds of the replication."""
    out: list[DiagnosticTest] = []
    m1 = inp.get("m1")
    xs, oks = _retention(inp.get("e2"))
    out += _test(r"Retention test, \holdmem{}(1,1), three tiers (change)",
                 r"90\% interval above $-$3\,pp", xs, oks, _passes(oks))  # fmt: skip
    cross = [
        (t.label, _get(m1, "t1", k) or {}) for t in TIERS if t.marker and (k := tier_key(m1, t))
    ]
    xs, oks = _ests((t.get("delta_90"), t.get("equivalent")) for _, t in cross)
    names = [lab for (lab, t) in cross if t.get("equivalent")]
    out += _test(r"Retention test, \holdmem{}(1,1), five other writers (change)",
                 r"90\% interval above $-$3\,pp", xs, oks, _passes(oks, names))  # fmt: skip
    t2 = _get(inp.get("e6"), "imports", "imported_with_ext")
    if t2 and _ok(_get(t2, "diff_90", "post")):
        out += _test(r"Leaving test, \holdmem{}(1,1) with its keys",
                     r"90\% interval within $\pm$3\,pp", [float(t2["diff_90"]["post"]["est"])],
                     [bool(t2.get("t2_equivalent"))],
                     "passes" if t2.get("t2_equivalent") else "does not pass")  # fmt: skip
    per = _per_horizon(inp.get("e3"))
    if any(per.values()):
        xs = [float(x["minus_oracle"]["est"]) for s, _ in HORIZON_SYSTEMS for x in per[s]
              if _ok(x.get("minus_oracle"))]  # fmt: skip
        out += _test(r"Utility floor, \holdmem{}(1,1), no-store BM25 and plain RAG, three "
                     "horizons", r"$\ge-3$\,pp", xs,
                     [bool(x["T3_pass"]) for s, _ in HORIZON_SYSTEMS for x in per[s]],
                     "passes: " + _floor_counts(per))  # fmt: skip
        cost = [x for s, _ in HORIZON_SYSTEMS for x in per[s] if "T4_pass" in x]
        if cost:
            ratio = max(float(x["serve_ratio_to_full_reread"]) for x in cost)
            out.append(DiagnosticTest("Cost ceiling, the same systems and horizons",
                                      "at most half", f"at most {100 * ratio:.1f}\\%",
                                      _passes([bool(x["T4_pass"]) for x in cost])))  # fmt: skip
    mem = _get(inp.get("nat"), "membership") or {}
    named = [(n.label, n.key) for n in NATIVES] + [("no-store BM25", "zero"), ("plain RAG", "rag")]
    systems: list[tuple[str, dict[str, Any]]] = [
        (lab, m) for lab, key in named if isinstance(m := mem.get(key), dict) and "T3_pass" in m
    ]
    if systems:
        passing = [lab for lab, m in systems if m["T3_pass"]]
        oks = [bool(m["T3_pass"]) for _, m in systems]
        out += _test("Utility floor, native systems and the two baselines", r"$\ge-3$\,pp",
                     [float(m["minus_oracle"]["est"]) for _, m in systems
                      if _ok(m.get("minus_oracle"))],
                     oks, _passes(oks, passing))  # fmt: skip
        ratio = max(float(m["serve_ratio_to_full_reread"]) for _, m in systems)
        out.append(DiagnosticTest("Cost ceiling, the same systems", "at most half",
                                  f"at most {100 * ratio:.1f}\\%",
                                  _passes([bool(m["T4_pass"]) for _, m in systems])))  # fmt: skip
    return out + _fresh_tests(inp)


def _fresh_tests(inp: Mapping[str, Any]) -> list[DiagnosticTest]:
    """The retention test, utility floor and cost ceiling on the fresh worlds of the replication,
    with the systems, tiers and horizons of the main study, one row per test: the estimate (or
    the range over the units) and how many units pass."""
    out: list[DiagnosticTest] = []
    ret, ret_ok = _retention(inp.get("rep_e2"))
    out += _test(r"Retention test, \holdmem{}(1,1), three tiers (change)",
                 r"90\% interval above $-$3\,pp", ret, ret_ok, _passes(ret_ok))  # fmt: skip
    per = _per_horizon(inp.get("rep_e3"))
    floor = [float(x["minus_oracle"]["est"]) for s, _ in HORIZON_SYSTEMS for x in per[s]
             if _ok(x.get("minus_oracle"))]  # fmt: skip
    if floor:
        out += _test(r"Utility floor, \holdmem{}(1,1), no-store BM25 and plain RAG, three "
                     "horizons", r"$\ge-3$\,pp", floor,
                     [bool(x["T3_pass"]) for s, _ in HORIZON_SYSTEMS for x in per[s]],
                     "passes: " + _floor_counts(per))  # fmt: skip
    cost = [x for s, _ in HORIZON_SYSTEMS for x in per[s] if "T4_pass" in x]
    if cost:
        ratio = max(float(x["serve_ratio_to_full_reread"]) for x in cost)
        out.append(DiagnosticTest("Cost ceiling, the same systems and horizons", "at most half",
                                  f"at most {100 * ratio:.1f}\\%",
                                  _passes([bool(x["T4_pass"]) for x in cost])))  # fmt: skip
    return [DiagnosticTest(t.label, t.threshold, t.estimate, t.outcome, fresh=True) for t in out]


def diagnostic_tests(inp: Mapping[str, Any]) -> list[DiagnosticTest]:
    """The data of `tab_diagnostic_tests`: the four diagnostic tests per set of systems or
    units."""
    return _diagnostic_tests(inp)


def tab_diagnostic_tests(tests: Sequence[DiagnosticTest]) -> str:
    """The table of the four diagnostic tests (paper/generated/tab_diagnostic_tests.tex): one row
    per test and set of systems, with its threshold, its estimate(s) and how many units pass, the
    test worlds above the fresh worlds of the replication."""
    lines = [
        "% generated by memshift.analysis.figures --only tab_diagnostic_tests; do not edit",
        r"\begin{table}[t]",
        r"\caption{The four diagnostic tests (\cref{app:experiments}) per tier, switch, horizon "
        r"and system, on the test worlds and on the fresh worlds of the replication. Estimate: "
        r"the change in retention, the leaving test's difference in carryover and the margin "
        r"over the utility floor in pp, and the serving cost as a share of a full re-read. A "
        r"range spans the units (tiers, writers, horizons or systems).}",
        r"\label{tab:diagnostic-tests}",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\newcolumntype{Y}[1]{>{\raggedright\arraybackslash}p{#1}}",
        r"\begin{tabular}{@{}l Y{4.5cm} l l Y{4.5cm}@{}}",
        r"\toprule",
        r"test & systems or units & threshold & estimate & outcome \\",
    ]
    for fresh, title in ((False, "test worlds"), (True, "fresh worlds of the replication")):
        rows = [t for t in tests if t.fresh == fresh]
        if not rows:
            continue
        lines += [r"\midrule", rf"\multicolumn{{5}}{{@{{}}l}}{{\emph{{{title}}}}} \\"]
        for t in rows:
            test, units = t.label.split(", ", 1)
            lines.append(rf"{test} & {units} & {t.threshold} & {t.estimate} & {t.outcome} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


FIGURES: dict[str, Callable[[Mapping[str, Any], Path], Path]] = {
    "fig_writers": fig_writers,
    "fig_reader": fig_reader,
    "fig_misses": fig_misses,
}
