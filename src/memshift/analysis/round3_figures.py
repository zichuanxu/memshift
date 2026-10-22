"""Figures of the round-3 results and the diagnostic-tests table (regenerated from the staged
analyses named in results/sources.json; nothing here calls a model or reads a raw run).

fig_writers     Section 4.2: (a) per program set (round3 g_b.json), B x D at D_m on definition
                items after S_def against how often the set's programs computed the right value at
                the fork, before any shift (ex_round3 writers.json), with the authored
                derivation's B x D on the same items as reference (main study); (b) per
                definition-item writer (Qwen3.5-9B, Claude Sonnet 5.5, Claude Opus 5.5) B x D with
                the authored derivation against r_0, both on the same worlds, after S_def and
                S_src, the two marks of a writer joined (writers.json)
fig_reader      appendix: bullet bars, the store part of B x D on Update behind B x D with each
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
from dataclasses import dataclass
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


def _dodge(
    ax: Any, pts: Sequence[Mapping[str, Any]], size_pt: float, diagonal: bool = False
) -> list[tuple[float, float]]:
    """Offsets (dx, dy in points) that set apart markers drawn on top of each other: markers
    closer than one marker width form a cluster, whose members are drawn side by side in the
    order of their values, centred on the cluster. With `diagonal` (marks with intervals on both
    axes) they step as far down as across, so that both their horizontal and their vertical
    intervals stay apart: the first member up and left, the last down and right."""
    pos = [ax.transData.transform((100 * p["x"]["est"], 100 * p["y"]["est"])) for p in pts]
    scale = float(ax.figure.dpi) / 72.0
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


DODGE = 1.3  # spacing of side-by-side markers, in marker widths: a thin white gap between them

WRITERS_IN = (486 / 72, 118.8 / 72)  # printed size (in): full width
WRITERS_NAME_PT = 7.5  # the writers' direct labels (pt)
WRITERS_MUTED = "#5A626D"  # tick labels and the shifts' key
WRITERS_CARD, WRITERS_ZERO = "#F2F4F7", "#8A929C"  # a panel's card, the zero line
WRITERS_MS, WRITERS_HALO = 5.4, 0.9  # marker size and its white edge (pt)
WRITERS_BAR, WRITERS_BAR_ALPHA = 2.3, 0.34  # a soft interval bar: width (pt), opacity
WRITERS_RC = {**fs.BAR_RC, "text.color": fs.INK}
OPUS, SONNET = "claude-opus-5-5", "claude-sonnet-5-5"
GEMMA, GPT = "google/gemma-4-12B-it", "openai/gpt-oss-20b"
LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
WRITERS_KEY = (("def", "s", "redefinition", False), ("src", "o", "source revision", True))


def _name_ink(writer: str) -> str:
    """The text colour of a writer's name: its mark colour, darkened just enough to read as text."""
    from matplotlib.colors import to_rgb

    c = fs.model_style(writer)[1]
    r, g, b = to_rgb(c)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return fs.mix(c, "#000000", 0.0 if lum < 0.25 else 0.22 if lum < 0.45 else 0.32)


@dataclass
class _Mark:
    """A drawn writer mark: its marker, its interval bars (y, then x), its point and its offset
    (points) from the dodge."""

    marker: Any
    intervals: list[Any]
    point: Mapping[str, Any]
    offset: tuple[float, float]


def _writer_card(ax: Any, ylabels: bool = True) -> Any:
    """A quiet panel on a pale rounded card: white grid lines at 0/20/40 pp, the zero line darker,
    no spines or tick marks, muted tick labels. Returns the zero line."""
    from matplotlib.patches import FancyBboxPatch

    w_mm, h_mm = WRITERS_IN[0] * 25.4, WRITERS_IN[1] * 25.4
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_facecolor("none")
    pos = ax.get_position()
    card = FancyBboxPatch((pos.x0, pos.y0), pos.width, pos.height,
                          boxstyle=f"round,pad=0,rounding_size={1.1 / w_mm}",
                          mutation_aspect=w_mm / h_mm, transform=ax.figure.transFigure,
                          facecolor=WRITERS_CARD, edgecolor="none", zorder=-10)  # fmt: skip
    ax.figure.patches.append(card)
    ax.tick_params(axis="x", length=0, labelsize=fs.T_TICK, labelcolor=WRITERS_MUTED, pad=2.6)
    ax.tick_params(axis="y", length=0, labelsize=fs.T_TICK, labelcolor=WRITERS_MUTED, pad=2.6,
                   labelleft=ylabels)  # fmt: skip
    ax.grid(axis="y", color="white", lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(-9, 53)
    ax.set_yticks([0, 20, 40])
    return ax.axhline(0, color=WRITERS_ZERO, lw=0.7, zorder=1)


def _writer_title(ax: Any, text: str, letter: str) -> None:
    ax.set_title(rf"$\bf{{({letter})}}$ {text}", loc="left", fontsize=fs.T_TITLE, pad=4.0,
                 color=fs.INK)  # fmt: skip


def _soft_bar(ax: Any, xs: list[float], ys: list[float], c: str, tr: Any) -> Any:
    """A 95% interval as a soft pill in the writer's colour whose rounded ends stop exactly at
    the interval's ends (the line is shortened by the cap radius)."""
    import numpy as np

    rad = WRITERS_BAR / 2 * ax.figure.dpi / 72  # cap radius in pixels
    (ax0, ay0), (ax1, ay1) = ax.transData.transform([(xs[0], ys[0]), (xs[1], ys[1])])
    length = np.hypot(ax1 - ax0, ay1 - ay0)
    cap = "butt"
    if length > 2 * rad:
        f = rad / length
        p0 = (ax0 + f * (ax1 - ax0), ay0 + f * (ay1 - ay0))
        p1 = (ax1 - f * (ax1 - ax0), ay1 - f * (ay1 - ay0))
        (xs[0], ys[0]), (xs[1], ys[1]) = ax.transData.inverted().transform([p0, p1])
        cap = "round"
    (ln,) = ax.plot(xs, ys, color=c, lw=WRITERS_BAR, alpha=WRITERS_BAR_ALPHA, zorder=3,
                    transform=tr, solid_capstyle=cap)  # fmt: skip
    return ln


def _writer_marks(ax: Any, pts: Sequence[Mapping[str, Any]], xerr: bool,
                  hollow: Sequence[str] = ()) -> dict[str, list[_Mark]]:  # fmt: skip
    """Soft interval bars, then markers with a white halo (colour = writer, shape = shift, open
    for a shift in `hollow`); writers on one spot dodged apart (`_dodge`). Returns per writer its
    marks in the order of `pts`."""
    from matplotlib.transforms import ScaledTranslation

    fig = ax.figure
    drawn: dict[str, list[_Mark]] = {}
    for p, (dx, dy) in zip(pts, _dodge(ax, pts, WRITERS_MS, diagonal=xerr), strict=True):
        c = fs.model_style(p["writer"])[1]
        tr = ax.transData + ScaledTranslation(dx / 72, dy / 72, fig.dpi_scale_trans)
        x, y = 100 * p["x"]["est"], 100 * p["y"]["est"]
        segs = [([x, x], [100 * p["y"]["lo"], 100 * p["y"]["hi"]])]
        if xerr:
            segs.append(([100 * p["x"]["lo"], 100 * p["x"]["hi"]], [y, y]))
        bars = [_soft_bar(ax, xs, ys, c, tr) for xs, ys in segs]
        open_ = p["shift"] in hollow
        shape = fs.SHIFTS[p["shift"]][1]
        (mk,) = ax.plot([x], [y], marker=shape, ms=WRITERS_MS, mfc="white" if open_ else c,
                        mec=c if open_ else "white", mew=1.25 if open_ else WRITERS_HALO,
                        ls="none", zorder=4, transform=tr)  # fmt: skip
        if open_:  # the white halo outside the coloured ring
            ax.plot([x], [y], marker=shape, ms=WRITERS_MS + 2 * 1.25, mfc="none", mec="white",
                    mew=WRITERS_HALO, ls="none", zorder=3.9, transform=tr)  # fmt: skip
        drawn.setdefault(p["writer"], []).append(_Mark(mk, bars, p, (dx, dy)))
    return drawn


def _box(artists: Sequence[Any]) -> Any:
    """The display-space union of artists; interval bars count with their painted width."""
    from matplotlib.transforms import Bbox

    fig = artists[0].figure
    r = fig.canvas.get_renderer()
    out = []
    for a in artists:
        bb = a.get_window_extent(r)
        if not a.get_marker() or a.get_marker() == "None":
            bb = bb.padded(WRITERS_BAR / 2 * fig.dpi / 72)
        out.append(bb)
    return Bbox.union(out)


def _put(ax: Any, text: str, anchor: Any, side: str, gap: float, color: str) -> Any:
    """A direct label beside a display-space box, `gap` points away on `side`."""
    fig = ax.figure
    g = gap * fig.dpi / 72
    x0, y0, x1, y1 = anchor.x0, anchor.y0, anchor.x1, anchor.y1
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    px, py, ha, va = {
        "left": (x0 - g, cy, "right", "center_baseline"),
        "right": (x1 + g, cy, "left", "center_baseline"),
        "above": (cx, y1 + g, "center", "bottom"),
        "below": (cx, y0 - g, "center", "top"),
        "above-left": (x0 - g, y1 + g * 0.3, "right", "bottom"),
        "below-left": (x0 - g, y0 - g * 0.3, "right", "top"),
        "above-right": (x1, y1 + g, "right", "bottom"),
        "below-start": (x0, y0 - g, "left", "top"),
    }[side]
    return ax.annotate(text, (px / fig.bbox.width, py / fig.bbox.height),
                       xycoords="figure fraction", ha=ha, va=va, fontsize=WRITERS_NAME_PT,
                       color=color, annotation_clip=False, zorder=6)  # fmt: skip


def _name_label(ax: Any, writer: str, anchor: Any, side: str, gap: float) -> Any:
    return _put(ax, fs.model_style(writer)[0], anchor, side, gap, _name_ink(writer))


def _exact_band(ax: Any, ref: Mapping[str, float], start_x: float) -> None:
    """Exact recomputation's 95% interval as a soft vertical gradient (densest at the estimate)
    with hairline edges at the interval's ends, and its estimate as a blue dashed line from
    `start_x` on."""
    import numpy as np
    from matplotlib.colors import to_rgb

    blue = fs.ROLE_BLUE
    lo, est, hi = (100 * ref[k] for k in ("lo", "est", "hi"))
    x0, x1 = ax.get_xlim()
    ys = np.linspace(lo, hi, 256)
    w = np.where(ys <= est, (ys - lo) / (est - lo), (hi - ys) / (hi - est))
    alpha = 0.07 + 0.20 * w**1.6
    # an opaque blend over the card, so the grid line at 40 pp does not show through the band
    rgb = (1 - alpha[:, None]) * np.array(to_rgb(WRITERS_CARD)) + alpha[:, None] * np.array(
        to_rgb(blue["mark"]))  # fmt: skip
    rgba = np.ones((256, 2, 4))
    rgba[..., :3] = rgb[:, None, :]
    ax.imshow(rgba, extent=(x0, x1, lo, hi), origin="lower", aspect="auto",
              interpolation="bilinear", zorder=0.4)  # fmt: skip
    ax.set_xlim(x0, x1)
    for v in (lo, hi):
        ax.axhline(v, color=blue["stroke"], lw=0.4, alpha=0.20, zorder=0.6)
    ax.plot([start_x, x1], [est, est], color=blue["mark"], lw=1.1, ls=(0, (4.2, 2.0)), zorder=1.2,
            solid_capstyle="butt", dash_capstyle="butt")  # fmt: skip


def _program_sets(fig: Any, axa: Any, data: Mapping[str, Any]) -> None:
    """Panel (a): the program sets after a redefinition, against how often their programs are
    right before the shift, over exact recomputation's band; each writer named once."""
    from matplotlib.transforms import Bbox

    _writer_card(axa)
    axa.set_xlim(-7, 106)
    axa.set_xticks([0, 25, 50, 75, 100])
    axa.set_ylabel(r"$B{\times}D$, definition items (pp)", fontsize=fs.T_LABEL, labelpad=3.0)
    axa.set_xlabel("programs right before the shift (%)", fontsize=fs.T_LABEL, labelpad=2.0)
    _writer_title(axa, "program sets: redefinition", "a")
    ref = data["reference"]
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    if ref:  # the hero's name on its own line at the left, the dashed line after it
        t_exact = axa.text(-5.0, 100 * ref["est"], "exact recomputation", ha="left",
                           va="center_baseline", fontsize=WRITERS_NAME_PT,
                           color=fs.ROLE_BLUE["ink"], fontstyle="italic", zorder=6)  # fmt: skip
        fig.canvas.draw()
        end = axa.transData.inverted().transform(
            (t_exact.get_window_extent(r).x1 + 2.2 * fig.dpi / 72, 0))[0]  # fmt: skip
        _exact_band(axa, ref, end)
        fig.canvas.draw()
    da = _writer_marks(axa, [{**g, "shift": "def"} for g in data["groups"]], False)
    fig.canvas.draw()
    if not da:
        axa.text(0.5, 0.5, "no program set staged", transform=axa.transAxes, ha="center",
                 fontsize=WRITERS_NAME_PT, color=WRITERS_MUTED)  # fmt: skip
        return

    def markers(who: str, keep: Any = None) -> list[Any]:
        return [m.marker for m in da.get(who, []) if keep is None or keep(m.point)]

    # the Claude writers: their names side by side above the band, right-aligned over their
    # shared spots, in the dodge's left-to-right order (Opus, then Sonnet)
    top = 100 * ref["hi"] if ref else max(100 * m.point["y"]["hi"] for ms in da.values()
                                          for m in ms)  # fmt: skip
    band_top = axa.transData.transform((0, top))[1]
    right = axa.transData.transform((104.6, 0))[0]
    x_end = Bbox.from_extents(right, band_top, right, band_top)
    if SONNET in da:
        t_son = _put(axa, fs.model_style(SONNET)[0], x_end, "above-right", 1.4,
                     _name_ink(SONNET))  # fmt: skip
        fig.canvas.draw()
        sx0 = t_son.get_window_extent(r).x0 - 5.0 * fig.dpi / 72
        x_end = Bbox.from_extents(sx0, band_top, sx0, band_top)
    if OPUS in da:
        _put(axa, fs.model_style(OPUS)[0], x_end, "above-right", 1.4, _name_ink(OPUS))
    # Qwen left of its first set; gemma below its sets under 75 %; gpt-oss right of its
    # right-most set; Llama under the zero line, from its sets above 5 %
    if QWEN in da:
        first = min(da[QWEN], key=lambda m: m.point["x"]["est"])
        _name_label(axa, QWEN, _box([first.marker]), "left", 2.6)
    if g_left := [m for m in da.get(GEMMA, []) if m.point["x"]["est"] < 0.75]:
        _name_label(axa, GEMMA, _box([m.marker for m in g_left] +
                                     [m.intervals[0] for m in g_left]), "below", 1.6)  # fmt: skip
    if GPT in da:
        right = max(da[GPT], key=lambda m: m.point["x"]["est"])
        _name_label(axa, GPT, _box([right.marker]), "right", 2.8)
    if ll := markers(LLAMA, lambda p: p["x"]["est"] > 0.05) or markers(LLAMA):
        anc = _box(ll)
        zero_y = axa.transData.transform((0, 0))[1]
        _put(axa, fs.model_style(LLAMA)[0], Bbox.from_extents(anc.x0, zero_y, anc.x1, zero_y),
             "below-start", 2.2, _name_ink(LLAMA))  # fmt: skip
    for who in sorted(set(da) - {QWEN, OPUS, SONNET, GEMMA, GPT, LLAMA}):  # any other writer
        right = max(da[who], key=lambda m: m.point["x"]["est"])
        _name_label(axa, who, _box([right.marker]), "right", 2.8)


def _shift_key(fig: Any, ax: Any, x_data: float, y_data: float) -> None:
    """A small inline key for the two shifts: a neutral filled square and a neutral open circle,
    each followed by its name."""
    from matplotlib.lines import Line2D

    r = fig.canvas.get_renderer()
    x, y = ax.transData.transform((x_data, y_data))
    px = fig.dpi / 72
    for _, m, label, open_ in WRITERS_KEY:
        fig.add_artist(Line2D([x / fig.bbox.width], [y / fig.bbox.height],
                              transform=fig.transFigure, marker=m, ms=WRITERS_MS - 0.6,
                              ls="none", mfc="white" if open_ else WRITERS_MUTED,
                              mec=WRITERS_MUTED, mew=1.1 if open_ else 0.0,
                              zorder=6))  # fmt: skip
        t = fig.text((x + 4.6 * px) / fig.bbox.width, y / fig.bbox.height, label, ha="left",
                     va="center_baseline", fontsize=fs.T_TICK, color=WRITERS_MUTED,
                     zorder=6)  # fmt: skip
        fig.canvas.draw()
        x = t.get_window_extent(r).x1 + 9.0 * px


def _shift_links(ax: Any, drawn: Mapping[str, Sequence[_Mark]]) -> None:
    """Per writer a thin line in its colour from its redefinition mark to its source-revision
    mark, between the drawn (dodged) marker centres, under the marks."""
    from matplotlib.lines import Line2D

    fig = ax.figure
    for who, marks in drawn.items():
        ends = {}
        for m in marks:
            cx, cy = ax.transData.transform((100 * m.point["x"]["est"], 100 * m.point["y"]["est"]))
            dx, dy = m.offset
            ends[m.point["shift"]] = (cx + dx * fig.dpi / 72, cy + dy * fig.dpi / 72)
        if "def" not in ends or "src" not in ends:
            continue
        (x0, y0), (x1, y1) = ends["def"], ends["src"]
        ax.add_artist(Line2D([x0 / fig.bbox.width, x1 / fig.bbox.width],
                             [y0 / fig.bbox.height, y1 / fig.bbox.height],
                             transform=fig.transFigure, color=fs.model_style(who)[1], lw=1.0,
                             alpha=0.9, zorder=3.5, solid_capstyle="round"))  # fmt: skip


def fig_writers(inp: Mapping[str, Any], out: Path) -> Path:
    """(a) G-b program sets after S_def; (b) the definition-item writers after both shifts, each
    writer's redefinition mark (filled square) joined to its source-revision mark (open circle):
    see the module docstring. Hue is the writer, marker shape the shift; intervals are soft bars
    in the writer's colour; identical sets of one writer draw as one marker, and marks of
    different writers on one spot are set apart (the caption says both). Each writer is named
    once per panel, in its own colour."""
    from matplotlib.transforms import Bbox

    data = writer_points(inp)
    w_mm, h_mm = WRITERS_IN[0] * 25.4, WRITERS_IN[1] * 25.4
    left, right, top, bottom, gap = 11.6, 1.0, 4.6, 8.6, 6.6  # mm
    widths = (1.8, 1.0)
    unit = (w_mm - left - right - gap) / sum(widths)
    ph = h_mm - top - bottom
    with fs.printed(WRITERS_RC) as plt:
        fig = plt.figure(figsize=(w_mm * fs.MM, h_mm * fs.MM), dpi=300)
        axa = fig.add_axes((left / w_mm, bottom / h_mm, widths[0] * unit / w_mm, ph / h_mm))
        x_b = left + (widths[0] * unit + gap)
        axb = fig.add_axes((x_b / w_mm, bottom / h_mm, widths[1] * unit / w_mm, ph / h_mm))
        _program_sets(fig, axa, data)
        _writer_card(axb, ylabels=False)
        axb.set_xlim(-3, 63)
        axb.set_xticks([0, 20, 40, 60])
        _writer_title(axb, "writers: both shifts", "b")
        fig.canvas.draw()
        db = _writer_marks(axb, data["defwriters"], True, hollow=("src",))
        fig.canvas.draw()
        _shift_links(axb, db)

        def at(who: str, shift: str) -> _Mark | None:
            return next((m for m in db.get(who, []) if m.point["shift"] == shift), None)

        # Qwen right of its redefinition mark (top left, alone); Opus above its source-revision
        # mark's vertical bar; Sonnet left of that mark's horizontal bar, below it
        if q := at(QWEN, "def"):
            qb, mb = _box([q.intervals[1]]), _box([q.marker])
            _name_label(axb, QWEN, Bbox.from_extents(qb.x0, mb.y0, qb.x1, mb.y1), "right", 2.4)
        if o := at(OPUS, "src"):
            _name_label(axb, OPUS, _box([o.intervals[0]]), "above", 1.2)
        if s := at(SONNET, "src"):
            _name_label(axb, SONNET, _box([s.intervals[1]]), "below-left", 1.8)
        _shift_key(fig, axb, 1.0, -4.6)  # under the zero line, where (b) has no data
        # the x title of (b), centred under it on (a)'s x-title baseline
        r = fig.canvas.get_renderer()
        xl = axa.xaxis.label.get_window_extent(r)
        bb = axb.get_window_extent(r)
        fig.text((bb.x0 + bb.x1) / 2 / fig.bbox.width, xl.y0 / fig.bbox.height,
                 r"$r_0$: repair correctness at $D\!=\!0$ (%)", ha="center", va="bottom",
                 fontsize=fs.T_LABEL)  # fmt: skip
        return fs.save_pdf(fig, out / "fig_writers.pdf")


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


READER_PT = (486, 158.4)  # printed size (pt): full width
READER_STORE = {"fill": "#E3E6EA", "top": "#8C95A1", "ink": "#6B7380"}  # a wide pale grey bar
READERS = (  # key, model, x offset (shifts), bar fill, dark ink (interval, reader part)
    ("qwen", QWEN, -0.14, "#4A5260", "#0E1116"),  # Qwen3.5-9B's near black as a dark slate
    ("sonnet", "claude-sonnet-5-5", 0.14, fs.model_style("claude-sonnet-5-5")[1], "#6E2E00"),
)
# the bars of each shift, back to front: key, label, fill
READER_BARS = (
    ("store", "store part (both readers)", READER_STORE["fill"]),
    *((key, f"{fs.model_style(model)[0]} reader", fill) for key, model, _, fill, _ in READERS),
)
READER_STORE_W, READER_W = 0.66, 0.235  # bar widths, in shifts
READER_GAP_CELL = ("other", "def")  # the cell whose reader parts are printed (the text cites them)


def fig_reader(inp: Mapping[str, Any], out: Path) -> Path:
    """Bullet bars per item class (panel) and shift (group): the store part of B x D on Update as
    a wide pale bar (the same for both readers, its interval in the slit between the readers)
    and B x D with each reader of the same stores as a narrower bar in front of it, in the
    reader's colour, each with its 95% interval. What a reader adds or loses is the gap between
    the pale bar and its own; in the cell the text cites, a bracket gives that reader part. The
    panels share one pp axis."""
    from matplotlib.ticker import MultipleLocator

    rows = reader_rows(inp)
    w_mm, h_mm = READER_PT[0] * fs.PT, READER_PT[1] * fs.PT
    left, right, gutter, bottom, top = 11.4, 0.6, 4.4, 5.8, 10.4
    pw = (w_mm - left - right - gutter) / 2
    ph = h_mm - bottom - top
    vals = [100 * r[k][e] for r in rows for k in ("store", "qwen", "sonnet") if r[k]
            for e in ("lo", "hi")]  # fmt: skip
    lo, hi = min([0.0, *vals]) - 1.5, max([10.0, *vals]) + 1.5
    store = READER_STORE
    with fs.printed(fs.BAR_RC) as plt:
        fig = plt.figure(figsize=(w_mm * fs.MM, h_mm * fs.MM))
        axes = []
        for p, (cls, title) in enumerate(CLASS_NAMES):
            ax = fig.add_axes(((left + p * (pw + gutter)) / w_mm, bottom / h_mm, pw / w_mm,
                               ph / h_mm))  # fmt: skip
            fs.bar_axes(ax)
            ax.set_ylim(lo, hi)
            ax.yaxis.set_major_locator(MultipleLocator(10))
            ax.set_xlim(-0.52, len(SHIFT_ROWS) - 0.48)
            ax.set_xticks(range(len(SHIFT_ROWS)), [t for _, t in SHIFT_ROWS])
            ax.set_title(title, fontsize=fs.T_TITLE, pad=3.2)
            if p:
                ax.tick_params(axis="y", labelleft=False)
            axes.append(ax)
            cap = fs.CAP_MM * fs.x_per_mm(ax, pw)
            for i, (b, _) in enumerate(SHIFT_ROWS):
                r = next((r for r in rows if r["cls"] == cls and r["branch"] == b), None)
                if r is None:
                    continue
                if s := r["store"]:
                    fs.bar_from_zero(ax, i, READER_STORE_W, 100 * s["est"], store["fill"], z=1.2)
                    ax.plot([i - READER_STORE_W / 2, i + READER_STORE_W / 2], [100 * s["est"]] * 2,
                            color=store["top"], lw=0.9, solid_capstyle="butt",
                            zorder=1.4)  # fmt: skip
                    # its interval sits in the slit between the readers, behind them
                    fs.interval_line(ax, i, 100 * s["lo"], 100 * s["hi"], store["ink"], 0.7 * cap,
                                     z=1.6)  # fmt: skip
                for key, _, dx, fill, ink in READERS:
                    if e := r[key]:
                        fs.bar_from_zero(ax, i + dx, READER_W, 100 * e["est"], fill, z=2.0)
                        fs.interval_line(ax, i + dx, 100 * e["lo"], 100 * e["hi"], ink, cap)
        axes[0].set_ylabel(r"$B{\times}D$ on Update (pp)", fontsize=fs.T_LABEL, labelpad=2.0)
        # each reader's part in the cell the text cites: a bracket over the uncovered part of
        # the store bar, from the store part to the reader, and its value
        ax = axes[[c for c, _ in CLASS_NAMES].index(READER_GAP_CELL[0])]
        i = [b for b, _ in SHIFT_ROWS].index(READER_GAP_CELL[1])
        cell = next((r for r in rows if (r["cls"], r["branch"]) == READER_GAP_CELL), None)
        for key, _, dx, _, ink in READERS:
            if not cell or not cell["store"] or not cell[key]:
                continue
            part = round(100 * (cell[key]["est"] - cell["store"]["est"]), 1)
            side = 1 if dx > 0 else -1
            ys, yr = 100 * cell["store"]["est"], 100 * cell[key]["est"]
            xb, tick = i + side * (READER_STORE_W / 2 + 0.035), 0.03
            ax.plot([xb - side * tick, xb, xb, xb - side * tick], [ys, ys, yr, yr], color=ink,
                    lw=0.7, solid_joinstyle="miter", solid_capstyle="butt", zorder=3)  # fmt: skip
            ax.annotate(f"{part:+.1f}".replace("-", "−"), (xb, (ys + yr) / 2),
                        xytext=(side * 2.2, 0), textcoords="offset points",
                        ha="left" if side > 0 else "right", va="center", fontsize=fs.T_NOTE,
                        color=ink)  # fmt: skip
        fig.canvas.draw()
        entries: list[tuple[fs.Glyph | None, str]] = [
            (fs.key_swatch(store["fill"], store["top"]), READER_BARS[0][1])
        ]
        entries += [(fs.key_swatch(fill), label) for _, label, fill in READER_BARS[1:]]
        fs.key_row(fig, entries, h_mm - 2.9, w_mm, h_mm)
        return fs.save_pdf(fig, out / "fig_reader.pdf")


# ==================================================================================== misses
# key -> (label, fill, a write-time key could have covered it), the coverable causes first: cool
# greys for the causes a write-time set could have contained, warm colours for the two it cannot,
# the strong one on the dominant cause (b new)
MISS_CAUSES = (
    ("a", r"$a$: an uncited record changed", "#8C95A1", True),
    ("b_old", r"$b$ old: existing record entered scope", "#CDD2D9", True),
    ("c_rest", r"$c$: other", "#E9EBEE", True),
    ("b_new", r"$b$ new: a record created after the derivation", fs.VERMILION, False),
    ("c_left", r"$c$: records left window or scope", "#F0A877", False),
)
MISS_STAGES = (("main", "main study"), ("rep", "replication"))
MISS_IN = (486 / 72, 140.4 / 72)  # printed size (in): full width
MISS_TICK = "#3F4650"  # tick labels
MISS_TEXT = 7.5  # counts, row labels and the key (pt)
MISS_WARM_INK = fs.mix(fs.VERMILION, "#000000", 0.22)  # the uncoverable share and its header


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
    main study and replication (panels); right of each panel two columns, the number missed
    ($n$) and the share no write-time set can contain (%, bold, under the two warm swatches);
    the key is one row per colour group, the group named in the row-label column."""
    from matplotlib.patches import Rectangle

    rows = miss_rows(inp)
    labels = {k: lab for k, lab, _, _ in MISS_CAUSES}
    fills = {k: fill for k, _, fill, _ in MISS_CAUSES}
    coverable = [k for k, _, _, covered in MISS_CAUSES if covered]
    uncoverable = [k for k, _, _, covered in MISS_CAUSES if not covered]
    causes = [k for k in (*coverable, *uncoverable)
              if k != "c_rest" or any(r["shares"]["c_rest"] for r in rows)]  # fmt: skip
    warm, warm_light = (fills[k] for k in uncoverable)
    w_mm, h_mm = MISS_IN[0] * 25.4, MISS_IN[1] * 25.4
    left, bottom, title_h, key_h = 35.0, 8.6, 4.6, 8.4  # mm
    col_n, col_s, gutter, right = 7.0, 15.0, 3.2, 0.4  # the n and share columns, right of the bars
    pw = (w_mm - left - 2 * col_s - gutter - right) / 2
    ph = h_mm - bottom - title_h - key_h
    x_panels = (left, left + pw + col_s + gutter)
    ypos: dict[tuple[str, str], float] = {}
    y = 0.0
    for b, _ in SHIFT_ROWS:
        for cls, _ in CLASS_NAMES:
            ypos[b, cls] = y
            y -= 1.0
        y -= 0.5

    def fx(mm: float) -> float:  # figure fraction of a horizontal position in mm
        return mm / w_mm

    def fy(mm: float) -> float:
        return mm / h_mm

    with fs.printed(fs.BAR_RC) as plt:
        fig = plt.figure(figsize=(w_mm * fs.MM, h_mm * fs.MM), dpi=300)
        axes = [fig.add_axes((x / w_mm, bottom / h_mm, pw / w_mm, ph / h_mm)) for x in x_panels]
        to_fig = fig.transFigure.inverted().transform
        for ax, x0, (stage, stitle) in zip(axes, x_panels, MISS_STAGES, strict=True):
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.spines["bottom"].set_color(fs.SPINE)
            ax.spines["bottom"].set_linewidth(0.6)
            ax.spines["left"].set_color(fs.SPINE)
            ax.spines["left"].set_linewidth(0.6)
            ax.tick_params(length=2.2, width=0.6, color=fs.SPINE, labelsize=fs.T_TICK,
                           labelcolor=MISS_TICK, pad=1.6)  # fmt: skip
            ax.grid(axis="x", color=fs.GRID, lw=0.5, zorder=0)
            ax.set_axisbelow(True)
            ax.tick_params(axis="y", length=0, labelleft=False)
            ax.set_xlim(0, 100)
            ax.set_ylim(min(ypos.values()) - 0.62, 0.62)
            ax.set_xticks([0, 25, 50, 75, 100])
            mine = [r for r in rows if r["stage"] == stage]
            for r in mine:
                yy, lft = ypos[r["branch"], r["cls"]], 0.0
                for k in causes:
                    w = 100 * r["shares"][k]
                    if w > 0:
                        ax.barh([yy], [w], left=[lft], height=0.7, color=fills[k],
                                edgecolor="white", lw=0.5, zorder=2)  # fmt: skip
                    lft += w
                # n, then the share no write-time set can contain (the claim), as two columns
                yfig = to_fig(ax.transData.transform((0, yy)))[1]
                fig.text(fx(x0 + pw + col_n), yfig, f"{r['missed']}", ha="right", va="center",
                         fontsize=MISS_TEXT, color=fs.MUTED)  # fmt: skip
                fig.text(fx(x0 + pw + col_s - 0.6), yfig, f"{100 * r['uncoverable']:.0f}",
                         ha="right", va="center", fontsize=MISS_TEXT, color=MISS_WARM_INK,
                         weight="bold")  # fmt: skip
            ax_top = bottom + ph
            fig.text(fx(x0), fy(ax_top + 1.3), stitle, ha="left", va="bottom",
                     fontsize=fs.T_TITLE, color=fs.INK)  # fmt: skip
            if mine:  # the columns' headers on the title line: n, and the two warm swatches + %
                fig.text(fx(x0 + pw + col_n), fy(ax_top + 1.3), "$n$", ha="right", va="bottom",
                         fontsize=MISS_TEXT, color=fs.MUTED)  # fmt: skip
                sw, sh = 1.7, 1.9  # mm
                sx = x0 + pw + col_s - 0.6 - 3.3 - 2 * sw
                for j, c in enumerate((warm, warm_light)):
                    fig.add_artist(Rectangle((fx(sx + j * sw), fy(ax_top + 1.55)), fx(sw), fy(sh),
                                             transform=fig.transFigure, fc=c, ec="white",
                                             lw=0.4))  # fmt: skip
                fig.text(fx(x0 + pw + col_s - 0.6), fy(ax_top + 1.3), "%", ha="right",
                         va="bottom", fontsize=MISS_TEXT, color=MISS_WARM_INK,
                         weight="bold")  # fmt: skip
        # row labels: the shift once per pair of rows, the item class per row
        for b, btitle in SHIFT_ROWS:
            for cls, cname in CLASS_NAMES:
                yfig = to_fig(axes[0].transData.transform((0, ypos[b, cls])))[1]
                fig.text(fx(left - 1.6), yfig, cname.split()[0], ha="right", va="center",
                         fontsize=MISS_TEXT, color=fs.MUTED)  # fmt: skip
            mid = (ypos[b, "definition"] + ypos[b, "other"]) / 2
            yfig = to_fig(axes[0].transData.transform((0, mid)))[1]
            fig.text(fx(0.6), yfig, btitle, ha="left", va="center", fontsize=fs.T_LABEL,
                     color=fs.INK)  # fmt: skip
        fig.text(fx((x_panels[0] + x_panels[1] + pw) / 2), fy(0.6),
                 "share of the missed changed item-times (%)", ha="center", va="bottom",
                 fontsize=fs.T_LABEL)  # fmt: skip
        # the key: one row per colour group, the group named on the left in the row-label column
        groups = [(heading, keys) for heading, keys in
                  (("a write-time set could contain", [k for k in causes if k in coverable]),
                   ("no write-time set can contain", [k for k in causes if k in uncoverable]))
                  if keys]  # fmt: skip
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        col_w = 0.0
        for _, keys in groups:  # the first column is as wide as its widest entry
            t = fig.text(0, 0, labels[keys[0]], fontsize=MISS_TEXT)
            col_w = max(col_w, t.get_window_extent(rend).width / fig.dpi * 25.4)
            t.remove()
        sw = 2.6  # swatch width, mm
        for (heading, keys), ym in zip(groups, (h_mm - 2.3, h_mm - 5.9), strict=False):
            fig.text(fx(left - 1.6), fy(ym), heading, ha="right", va="center",
                     fontsize=MISS_TEXT, style="italic",
                     color=MISS_WARM_INK if keys[0] in uncoverable else fs.MUTED)  # fmt: skip
            x = left
            for k in keys:
                fig.add_artist(Rectangle((fx(x), fy(ym - 0.95)), fx(sw), fy(1.9),
                                         transform=fig.transFigure, fc=fills[k],
                                         ec="none"))  # fmt: skip
                fig.text(fx(x + sw + 1.2), fy(ym), labels[k], ha="left", va="center",
                         fontsize=MISS_TEXT, color=fs.INK)  # fmt: skip
                x += sw + 1.2 + col_w + 5.0
        return fs.save_pdf(fig, out / "fig_misses.pdf")


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


def _of(passed: int, n: int) -> str:
    """How many of n units pass, tied ("3~of~3") so that it never breaks across lines."""
    return f"{passed}~of~{n}"


def _passes(oks: Sequence[bool], names: Sequence[str] = ()) -> str:
    """The outcome cell of a diagnostic test: how many units pass, and which where named."""
    return f"passes {_of(sum(oks), len(oks))}" + (f" ({', '.join(names)})" if names else "")


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
    """How many horizons pass the utility floor, per system; a name of several words is tied
    together with its count ("no-store~BM25~0~of~3"), so it does not break across lines."""
    return ", ".join(
        lab.replace(" ", "~") + ("~" if " " in lab else " ")
        + _of(sum(bool(x["T3_pass"]) for x in per[s]), len(per[s]))
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
    test worlds above the fresh worlds of the replication (each under an italic banner, in the
    paper's table style). A 2 pt gap separates the tests; a cost ceiling stays with the utility
    floor before it, whose systems it refers to ("the same systems")."""
    lines = [
        "% generated by memshift.analysis.figures --only tab_diagnostic_tests; do not edit",
        r"\begin{table}[t]",
        r"\caption{The four diagnostic tests (\cref{app:experiments}) per tier, switch, horizon "
        r"and system, on the test worlds and on the fresh worlds of the replication. Estimate: "
        r"the change in retention, the leaving test's difference in carryover and the margin "
        r"over the utility floor in pp, and the serving cost as a share of a full re-read. A "
        r"range spans the units (tiers, writers, horizons or systems).}",
        r"\label{tab:diagnostic-tests}",
        r"\centering\mtsetup",
        r"\setlength{\tabcolsep}{4pt}",
        r"\newcolumntype{Y}[1]{>{\raggedright\arraybackslash}p{#1}}",
        r"\begin{mttabular*}{\linewidth}{l Y{4.5cm} l l Y{4.2cm}@{}}",
        r"\toprule",
        r"Test & Systems or units & Threshold & Estimate & Outcome \\",
    ]
    for fresh, title in ((False, "Test worlds"), (True, "Fresh worlds of the replication")):
        rows = [t for t in tests if t.fresh == fresh]
        if not rows:
            continue
        lines += [r"\midrule", rf"\mtgroup{{5}}{{{title}}}"]
        before = None
        for t in rows:
            test, units = t.label.split(", ", 1)
            if before is not None and test not in (before, "Cost ceiling"):
                lines.append(r"\addlinespace[2pt]")
            before = test
            lines.append(rf"\mtin {test} & {units} & {t.threshold} & {t.estimate} & {t.outcome} \\")
    lines += [r"\bottomrule", r"\end{mttabular*}", r"\end{table}"]
    return "\n".join(lines) + "\n"


FIGURES: dict[str, Callable[[Mapping[str, Any], Path], Path]] = {
    "fig_writers": fig_writers,
    "fig_reader": fig_reader,
    "fig_misses": fig_misses,
}
