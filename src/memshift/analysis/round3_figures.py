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
def writer_points(inp: Mapping[str, Any]) -> dict[str, Any]:
    """The data of `fig_writers`.

    `programs`: per G-b program set (S_def, definition items): writer, x = share of definition
    slots whose program value at the fork equals gold (ex_round3 writers.json, S_def branch), y =
    B x D at D_m (g_b.json, world-bootstrap interval); a set without x or y keeps None there.
    `reference`: the authored derivation's B x D on the same item-times (main study, e1 class
    split). `defwriters`: per writer and shift (S_src, S_def), x = r_0 and y = B x D with the
    authored D=1 cells, Qwen3.5-9B on the same worlds and item-times as the other writers."""
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
        "reference": _est(ref),
        "defwriters": defw,
    }


WRITERS_PT = (486, 141.7)  # printed size (pt): full width
WRITERS_NAME_PT = 7.5  # the writers' direct labels (pt)
WRITERS_MUTED = "#5A626D"  # the subtitles under the panel titles
WRITERS_RC = {**fs.BAR_RC, "text.color": fs.INK}
OPUS, SONNET = "claude-opus-5-5", "claude-sonnet-5-5"
GEMMA, GPT = "google/gemma-4-12B-it", "openai/gpt-oss-20b"
LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
# layout (mm): margins of the plots, the gutter between the panels and the width of (a)'s plot;
# one value axis for both panels, B x D in pp; (a)'s share axis ends at WRITERS_XA_END, past 100
WRITERS_LEFT, WRITERS_RIGHT, WRITERS_TOP, WRITERS_BOTTOM, WRITERS_GAP = 11.6, 0.9, 9.6, 9.4, 8.4
WRITERS_WA = 99.6
WRITERS_YLIM, WRITERS_YTICKS = (-4.5, 56.5), (0, 10, 20, 30, 40)
WRITERS_XA, WRITERS_XA_END, WRITERS_XB = (-4.5, 107.0), 104.4, (-2.0, 61.5)
# marks (mm): squares and open circles of (a), smaller ones in (b), the pitch of sets side by
# side (a hairline white gap) and the white edge of a mark (pt)
WRITERS_SQ, WRITERS_CIRC, WRITERS_SQ_B, WRITERS_CIRC_B = 2.05, 2.25, 1.75, 1.95
WRITERS_SLOT, WRITERS_HALO = 2.05 + 0.32, 0.55
WRITERS_ORDER = (OPUS, SONNET, GEMMA, QWEN, LLAMA, GPT)  # left to right among sets on one spot
WRITERS_HULLS = ((LLAMA, GPT), (QWEN,), (GEMMA,))  # (a): one grey hull per group of writers
WRITERS_HULL, WRITERS_BAND = "#EDEFF2", "#E3ECF8"  # hull fill; exact recomputation's band
WRITERS_KEY = (("def", "redefinition"), ("src", "source revision"))  # (b): the shapes' key
# the nudge search of (b) (mm): half a cap's length, the clearances, the grid and the largest
# nudge; a nudge moves a mark and the lines through it, never the ends of its intervals
WRITERS_CAPH = 0.45
WRITERS_GAPS = {"mark": 0.15, "cap": 0.2, "graze": 0.2, "deep": 0.25, "parallel": 0.7,
                "tick": 0.35, "link": 0.12, "cap_link": 0.25, "cap_cap": 0.25}  # fmt: skip
WRITERS_STEP, WRITERS_NUDGE_MAX = 0.1, 2.4


def _name_ink(writer: str) -> str:
    """The text colour of a writer's name: its mark colour, darkened just enough to read as text."""
    from matplotlib.colors import to_rgb

    c = fs.model_style(writer)[1]
    r, g, b = to_rgb(c)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return fs.mix(c, "#000000", 0.0 if lum < 0.25 else 0.22 if lum < 0.45 else 0.32)


class _Plot:
    """A data axes of the writer figure placed in mm (`x0`, `w`; the height is the figure's
    minus its margins), with conversions between mm and data units."""

    def __init__(self, fig: Any, x0: float, w: float, xlim: tuple[float, float]) -> None:
        w_mm, h_mm = WRITERS_PT[0] * fs.PT, WRITERS_PT[1] * fs.PT
        self.x0, self.w = x0, w
        self.y0, self.h = WRITERS_BOTTOM, h_mm - WRITERS_BOTTOM - WRITERS_TOP
        self.ax = fig.add_axes((x0 / w_mm, self.y0 / h_mm, w / w_mm, self.h / h_mm))
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*WRITERS_YLIM)
        self.kx = w / (xlim[1] - xlim[0])  # mm per x unit
        self.ky = self.h / (WRITERS_YLIM[1] - WRITERS_YLIM[0])  # mm per y unit

    def dx(self, mm: float) -> float:
        return mm / self.kx

    def dy(self, mm: float) -> float:
        return mm / self.ky

    def mm(self, x: float, y: float) -> tuple[float, float]:
        xl, yl = self.ax.get_xlim()[0], self.ax.get_ylim()[0]
        return self.x0 + (x - xl) * self.kx, self.y0 + (y - yl) * self.ky

    def data(self, mx: float, my: float) -> tuple[float, float]:
        return (self.ax.get_xlim()[0] + (mx - self.x0) / self.kx,
                self.ax.get_ylim()[0] + (my - self.y0) / self.ky)  # fmt: skip


def _writers_axes(p: _Plot, ylabels: bool, xbounds: tuple[float, float]) -> None:
    """The apparatus of the bar figures: left and bottom spines, a light grid, a dark zero."""
    ax = p.ax
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(fs.BAR_AXIS)
        ax.spines[s].set_linewidth(0.6)
    ax.spines["bottom"].set_bounds(*xbounds)
    ax.tick_params(length=2.0, width=0.6, color=fs.BAR_AXIS, labelsize=fs.T_TICK,
                   labelcolor=fs.TICK, pad=1.6)  # fmt: skip
    ax.tick_params(axis="y", labelleft=ylabels)
    ax.set_yticks(WRITERS_YTICKS)
    for v in WRITERS_YTICKS[1:]:
        ax.axhline(v, color=fs.BAR_GRID, lw=0.5, zorder=0)
    ax.axhline(0, color=fs.BAR_ZERO, lw=0.65, zorder=2.6)
    ax.set_facecolor("none")


def _w_interval(p: _Plot, at: float, lo: float, hi: float, c: str, lw: float = 0.7,
                cap_mm: float = 1.1, vertical: bool = True) -> None:  # fmt: skip
    """A 95% interval in data units with caps `cap_mm` long: vertical at x=`at`, or horizontal
    at y=`at`."""
    if vertical:
        p.ax.plot([at, at], [lo, hi], color=c, lw=lw, solid_capstyle="butt", zorder=4.0)
        cw = p.dx(cap_mm) / 2
        for v in (lo, hi):
            p.ax.plot([at - cw, at + cw], [v, v], color=c, lw=lw, solid_capstyle="butt", zorder=4.0)
    else:
        p.ax.plot([lo, hi], [at, at], color=c, lw=lw, solid_capstyle="butt", zorder=4.0)
        ch = p.dy(cap_mm) / 2
        for v in (lo, hi):
            p.ax.plot([v, v], [at - ch, at + ch], color=c, lw=lw, solid_capstyle="butt", zorder=4.0)


def _w_square(p: _Plot, x: float, y: float, c: str, z: float = 5.0,
              side: float = WRITERS_SQ) -> None:  # fmt: skip
    from matplotlib.patches import Rectangle

    w, h = p.dx(side), p.dy(side)
    p.ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fc=c, ec="white", lw=WRITERS_HALO,
                             zorder=z, joinstyle="miter"))  # fmt: skip


def _w_ring(p: _Plot, x: float, y: float, c: str, z: float = 5.0,
            d: float = WRITERS_CIRC) -> None:  # fmt: skip
    from matplotlib.patches import Ellipse

    p.ax.add_patch(Ellipse((x, y), p.dx(d + 0.5), p.dy(d + 0.5), fc="white", ec="none", zorder=z))
    p.ax.add_patch(Ellipse((x, y), p.dx(d - 0.45), p.dy(d - 0.45), fc="white", ec=c, lw=1.25,
                           zorder=z + 0.1))  # fmt: skip


def _side_by_side(p: _Plot, pts: Sequence[Mapping[str, Any]]) -> list[float]:
    """Drawn x (data units) of each program set in (a): sets whose squares would touch, or whose
    interval would run behind another set's square, sit side by side as one compound mark
    centred on their mean x, ordered by share and then by `WRITERS_ORDER`; each keeps its y."""
    n, sq = len(pts), WRITERS_SQ
    xs = [p.mm(q["x"]["est"] * 100, q["y"]["est"] * 100) for q in pts]
    ends = [tuple(p.mm(0, 100 * q["y"][k])[1] for k in ("lo", "hi")) for q in pts]
    parent = list(range(n))

    def root(i: int) -> int:
        while parent[i] != i:
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i):
            (xi, yi), (xj, yj) = xs[i], xs[j]
            close = abs(xi - xj) < sq and abs(yi - yj) < sq
            behind = abs(xi - xj) < sq and (
                ends[i][0] - sq / 2 < yj < ends[i][1] + sq / 2
                or ends[j][0] - sq / 2 < yi < ends[j][1] + sq / 2
            )
            if close or behind:
                parent[root(i)] = root(j)

    def order(w: str) -> int:
        return WRITERS_ORDER.index(w) if w in WRITERS_ORDER else len(WRITERS_ORDER)

    out = [0.0] * n
    for r in {root(i) for i in range(n)}:
        mem = sorted((i for i in range(n) if root(i) == r),
                     key=lambda i: (round(pts[i]["x"]["est"], 4),
                                    order(pts[i]["writer"])))  # fmt: skip
        cx = sum(xs[i][0] for i in mem) / len(mem)
        for k, i in enumerate(mem):
            out[i] = p.data(cx + (k - (len(mem) - 1) / 2) * WRITERS_SLOT, 0.0)[0]
    return out


def _cross(o: Any, a: Any, b: Any) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def _convex(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The convex hull of points (monotone chain), counter-clockwise."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts
    lower: list[Any] = []
    for q in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    upper: list[Any] = []
    for q in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return lower[:-1] + upper[:-1]


def _hull(p: _Plot, centres: list[tuple[float, float]], pad: float) -> None:
    """A rounded grey hull (`pad` mm beyond the squares) around squares centred at `centres`."""
    import numpy as np
    from matplotlib.patches import Polygon

    cloud = []
    r = WRITERS_SQ / 2 * math.sqrt(2) * 0.82 + pad
    for x, y in centres:
        mx, my = p.mm(x, y)
        for t in np.linspace(0, 2 * math.pi, 48, endpoint=False):
            cloud.append((mx + r * math.cos(t), my + r * math.sin(t)))
    poly = [p.data(a, b) for a, b in _convex(cloud)]
    p.ax.add_patch(Polygon(poly, closed=True, fc=WRITERS_HULL, ec="none", lw=0.0, zorder=1.0,
                           joinstyle="round"))  # fmt: skip


def _program_sets(p: _Plot, data: Mapping[str, Any]) -> None:
    """Panel (a): one square per program set after a redefinition, with its interval, at the
    share of its programs right before the shift; a grey hull per group of `WRITERS_HULLS` (the
    Claude writers' sets lie in exact recomputation's band, without a hull), each writer named
    beside its sets; exact recomputation as a line over its band, its value at the right end."""
    from matplotlib.patches import Rectangle

    ax, blue = p.ax, fs.ROLE_BLUE
    xa, x_end = WRITERS_XA[0], WRITERS_XA_END
    _writers_axes(p, True, (xa, x_end))
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel(r"$B{\times}D$ on definition items (pp)", fontsize=fs.T_LABEL, color=fs.INK,
                  labelpad=2.4)  # fmt: skip
    ref = data["reference"]
    if ref:  # exact recomputation: its value at the line's right end, its name in the band
        lo, est, hi = (100 * ref[k] for k in ("lo", "est", "hi"))
        ax.add_patch(Rectangle((xa, lo), x_end - xa, hi - lo, fc=WRITERS_BAND, ec="none",
                               zorder=0.5))  # fmt: skip
        for v in (lo, hi):
            ax.plot([xa, x_end], [v, v], color=blue["stroke"], lw=0.4, alpha=0.5, zorder=0.6,
                    solid_capstyle="butt")  # fmt: skip
        ax.plot([xa, x_end], [est, est], color=blue["mark"], lw=1.3, zorder=1.2,
                solid_capstyle="butt")  # fmt: skip
        ax.text(x_end + p.dx(0.9), est, f"{est:+.1f}".replace("-", "−"), ha="left",
                va="center_baseline", fontsize=fs.T_LABEL, color=blue["ink"], weight="bold",
                zorder=6)  # fmt: skip
        ax.text(xa + p.dx(1.4), (est + hi) / 2, "exact recomputation, 95% interval", ha="left",
                va="center_baseline", fontsize=fs.T_NOTE, color=blue["ink"], style="italic",
                zorder=6)  # fmt: skip
    pts = [q for q in data["programs"] if q["x"] and q["y"]]
    if not pts:
        ax.text(0.5, 0.5, "no program set staged", transform=ax.transAxes, ha="center",
                fontsize=WRITERS_NAME_PT, color=WRITERS_MUTED)  # fmt: skip
        return
    xd = _side_by_side(p, pts)
    for group in WRITERS_HULLS:
        cs = [(xd[i], 100 * q["y"]["est"]) for i, q in enumerate(pts) if q["writer"] in group]
        if cs:
            _hull(p, cs, 1.0)
    for i, q in enumerate(pts):
        _w_interval(p, xd[i], 100 * q["y"]["lo"], 100 * q["y"]["hi"], _name_ink(q["writer"]))
        _w_square(p, xd[i], 100 * q["y"]["est"], fs.model_style(q["writer"])[1])

    def at(w: str) -> list[tuple[float, float]]:
        return [(xd[i], 100 * q["y"]["est"]) for i, q in enumerate(pts) if q["writer"] == w]

    def name(x: float, y: float, w: str, ha: str = "left", va: str = "center_baseline") -> None:
        ax.text(x, y, fs.model_style(w)[0], ha=ha, va=va, fontsize=WRITERS_NAME_PT,
                color=_name_ink(w), zorder=6)  # fmt: skip

    # Qwen and gemma left of their first set (gemma at its sets under 75%), gpt-oss above and
    # Llama below right of their shared hull, the Claude writers stacked over their sets
    gap = p.dx(WRITERS_SQ / 2 + 1.9)
    if q_at := at(QWEN):
        qx, qy = min(q_at)
        name(qx - gap, qy, QWEN, ha="right")
    if g_at := at(GEMMA):
        left = [b for a, b in g_at if a < 75] or [b for _, b in g_at]
        name(min(g_at)[0] - gap, sum(left) / len(left), GEMMA, ha="right")
    if low := at(LLAMA) + at(GPT):
        lx = max(a for a, _ in low) + gap
        if at(GPT):
            name(lx, max(b for _, b in at(GPT)) + p.dy(1.0), GPT)
        if at(LLAMA):
            ll = [b for a, b in at(LLAMA) if a > 5] or [b for _, b in at(LLAMA)]
            name(lx, min(ll) - p.dy(2.2), LLAMA)
    if claude := at(OPUS) + at(SONNET):
        cl = min(a for a, _ in claude) - p.dx(WRITERS_SQ / 2)
        top = 100 * ref["hi"] if ref else max(100 * q["y"]["hi"] for q in pts
                                             if q["writer"] in (OPUS, SONNET))  # fmt: skip
        base = top + p.dy(1.5)
        for k, who in enumerate(w for w in (SONNET, OPUS) if at(w)):
            name(cl, base + k * p.dy(3.05), who, va="baseline")
    for who in sorted({q["writer"] for q in pts} - set(WRITERS_ORDER)):  # any other writer
        x, y = max(at(who))
        name(x + gap, y, who)


def _b_marks(p: _Plot, by: Mapping[Any, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The marks of (b) in mm: centre, half size (a ring's white halo counts), interval ends."""
    out = []
    for (w, sh), q in by.items():
        cx, cy = p.mm(100 * q["x"]["est"], 100 * q["y"]["est"])
        xlo, ylo = p.mm(100 * q["x"]["lo"], 100 * q["y"]["lo"])
        xhi, yhi = p.mm(100 * q["x"]["hi"], 100 * q["y"]["hi"])
        h = WRITERS_SQ_B / 2 + 0.1 if sh == "def" else WRITERS_CIRC_B / 2 + 0.1
        out.append({"key": (w, sh), "w": w, "cx": cx, "cy": cy, "h": h, "round": sh == "src",
                    "xlo": xlo, "xhi": xhi, "ylo": ylo, "yhi": yhi})  # fmt: skip
    return out


def _to_mark(o: Mapping[str, Any], ox: Any, oy: Any, horizontal: bool, at: Any, lo: Any,
             hi: Any) -> Any:  # fmt: skip
    """Distance (mm) from an axis-parallel segment to the drawn shape of mark `o` centred at
    (ox, oy): a horizontal segment at y=`at` from x=`lo` to `hi`, or a vertical one at x=`at`."""
    import numpy as np

    if horizontal:
        px, py = np.clip(ox, lo, hi), at
    else:
        px, py = at, np.clip(oy, lo, hi)
    if o["round"]:
        return np.hypot(px - ox, py - oy) - o["h"]
    gx = np.maximum(0.0, np.abs(px - ox) - o["h"])
    gy = np.maximum(0.0, np.abs(py - oy) - o["h"])
    return np.hypot(gx, gy)


def _caps(m: Mapping[str, Any], x: Any, y: Any) -> list[tuple[Any, Any, Any, Any]]:
    """The four caps of a mark drawn at (x, y), as (x0, x1, y0, y1) segments."""
    h = WRITERS_CAPH
    out = [(xc, xc, y - h, y + h) for xc in (m["xlo"], m["xhi"])]
    return out + [(x - h, x + h, yc, yc) for yc in (m["ylo"], m["yhi"])]


def _lines(m: Mapping[str, Any], x: Any, y: Any) -> list[tuple[Any, Any, Any, Any]]:
    """The two interval lines of a mark drawn at (x, y), as (x0, x1, y0, y1) segments."""
    return [(m["xlo"], m["xhi"], y, y), (x, x, m["ylo"], m["yhi"])]


def _box_dist(a: Any, b: Any) -> Any:
    """Distance between two axis-parallel segments (x0, x1, y0, y1)."""
    import numpy as np

    dx = np.maximum(0.0, np.maximum(a[0] - b[1], b[0] - a[1]))
    dy = np.maximum(0.0, np.maximum(a[2] - b[3], b[2] - a[3]))
    return np.hypot(dx, dy)


def _clash(a: Mapping[str, Any], oa: Any, b: Mapping[str, Any], ob: Any) -> Any:
    """Whether mark `a` nudged by `oa` and mark `b` nudged by `ob` collide ((n, 2) and (m, 2)
    arrays of mm offsets; returns (n, m) booleans). Collisions (`WRITERS_GAPS`): marks closer
    than "mark"; a cap within "cap" of another mark; a line grazing another mark's edge (closer
    than "graze" outside it or "deep" inside it); two parallel lines that share a stretch closer
    than "parallel"; a cap closer than "cap_cap" to another cap or "tick" to another line."""
    import numpy as np

    g, caph = WRITERS_GAPS, WRITERS_CAPH
    ax_, ay_ = (a["cx"] + oa[:, 0])[:, None], (a["cy"] + oa[:, 1])[:, None]
    bx_, by_ = (b["cx"] + ob[:, 0])[None, :], (b["cy"] + ob[:, 1])[None, :]
    if a["round"] and b["round"]:
        bad = np.hypot(ax_ - bx_, ay_ - by_) < a["h"] + b["h"] + g["mark"]
    elif a["round"] or b["round"]:
        c, s_ = (a, b) if a["round"] else (b, a)
        cx, cy, sx, sy = (ax_, ay_, bx_, by_) if a["round"] else (bx_, by_, ax_, ay_)
        gx = np.maximum(0.0, np.abs(cx - sx) - s_["h"])
        gy = np.maximum(0.0, np.abs(cy - sy) - s_["h"])
        bad = np.hypot(gx, gy) < c["h"] + g["mark"]
    else:
        lim = a["h"] + b["h"] + g["mark"]
        bad = (np.abs(ax_ - bx_) < lim) & (np.abs(ay_ - by_) < lim)
    for (m, mx, my), (o, ox, oy) in (((a, ax_, ay_), (b, bx_, by_)),
                                     ((b, bx_, by_), (a, ax_, ay_))):  # fmt: skip
        for xc in (m["xlo"], m["xhi"]):  # caps of the horizontal interval: vertical ticks
            bad |= _to_mark(o, ox, oy, False, xc, my - caph, my + caph) < g["cap"]
        for yc in (m["ylo"], m["yhi"]):  # caps of the vertical interval: horizontal ticks
            bad |= _to_mark(o, ox, oy, True, yc, mx - caph, mx + caph) < g["cap"]
        h = o["h"]  # the lines of m grazing the edge of o
        over = (m["xlo"] < ox + h) & (m["xhi"] > ox - h)
        d = np.abs(my - oy)
        bad |= over & (d > h - g["deep"]) & (d < h + g["graze"])
        over = (m["ylo"] < oy + h) & (m["yhi"] > oy - h)
        d = np.abs(mx - ox)
        bad |= over & (d > h - g["deep"]) & (d < h + g["graze"])
        for seg in _caps(m, mx, my):  # caps of m touching a cap or a line of o
            for other in _caps(o, ox, oy):
                bad |= _box_dist(seg, other) < g["cap_cap"]
            for other in _lines(o, ox, oy):
                bad |= _box_dist(seg, other) < g["tick"]
    if min(a["xhi"], b["xhi"]) - max(a["xlo"], b["xlo"]) > 0.3:  # parallel lines, one stretch
        bad |= np.abs(ay_ - by_) < g["parallel"]
    if min(a["yhi"], b["yhi"]) - max(a["ylo"], b["ylo"]) > 0.3:
        bad |= np.abs(ax_ - bx_) < g["parallel"]
    return bad


def _link_hits(p0: Any, p1: Any, c: Any, h: float) -> Any:
    """Whether segments p0-p1 pass through the box of half size h (plus a gap) around c."""
    import numpy as np

    t = np.linspace(0.08, 0.92, 24)
    xs = p0[0][..., None] + (p1[0] - p0[0])[..., None] * t
    ys = p0[1][..., None] + (p1[1] - p0[1])[..., None] * t
    r = h + WRITERS_GAPS["link"]
    return ((np.abs(xs - c[0][..., None]) < r) & (np.abs(ys - c[1][..., None]) < r)).any(-1)


def _seg_dist(q: Any, p0: Any, p1: Any) -> Any:
    """Distance from points q to segments p0-p1 (arrays of x and y)."""
    import numpy as np

    vx, vy = p1[0] - p0[0], p1[1] - p0[1]
    t = np.clip(((q[0] - p0[0]) * vx + (q[1] - p0[1]) * vy) / np.maximum(vx**2 + vy**2, 1e-9),
                0, 1)  # fmt: skip
    return np.hypot(q[0] - p0[0] - t * vx, q[1] - p0[1] - t * vy)


def _expand(v: Any, a: int, b: int, k: int) -> Any:
    """A (len a, len b) table broadcast over k axes, with its axes at positions a and b."""
    if a > b:
        v, a, b = v.T, b, a
    shape = [1] * k
    shape[a], shape[b] = v.shape
    return v.reshape(shape)


def _nudges(marks: list[dict[str, Any]]) -> dict[Any, tuple[float, float]]:
    """The smallest nudges (mm) that leave no collision in (b): among the marks that collide
    unnudged, each moves by at most a limit, the limit as small as possible (0.1 mm steps), then
    the least total movement. A nudge moves a mark and the lines through it; its interval ends
    stay exact. Before the joint search a candidate is dropped when it clashes with a mark that
    stays or with every candidate of another moving mark; the rest keep their order, so the
    choice is that of the search over every candidate."""
    import numpy as np

    zero = np.zeros((1, 2))
    n = len(marks)
    ends: dict[str, dict[str, int]] = {}
    for i, m in enumerate(marks):
        ends.setdefault(m["w"], {})[m["key"][1]] = i
    links = {w: (v["def"], v["src"]) for w, v in ends.items() if len(v) == 2}
    move = sorted({k for i in range(n) for j in range(i)
                   if _clash(marks[i], zero, marks[j], zero)[0, 0] for k in (i, j)})  # fmt: skip
    if not move:
        return {}
    grid = np.round(np.arange(-WRITERS_NUDGE_MAX, WRITERS_NUDGE_MAX + 0.01, WRITERS_STEP), 2)
    cand = np.array([(dx, dy) for dx in grid for dy in grid])
    fixed = [i for i in range(n) if i not in move]

    def at(i: int, o: Any) -> tuple[Any, Any]:
        return marks[i]["cx"] + o[..., 0], marks[i]["cy"] + o[..., 1]

    for limit in np.arange(0.0, 4.01, 0.1):
        sub = []
        for i in move:  # each moving mark's candidates clear of the marks and links that stay
            c = cand[np.hypot(cand[:, 0], cand[:, 1]) <= limit + 1e-9]
            u = np.ones(len(c), bool)
            for j in fixed:
                u &= ~_clash(marks[i], c, marks[j], zero)[:, 0]
            for w, (d, s_) in links.items():
                if d in fixed and s_ in fixed and marks[i]["w"] != w:
                    pa = (np.full(len(c), marks[d]["cx"]), np.full(len(c), marks[d]["cy"]))
                    pb = (np.full(len(c), marks[s_]["cx"]), np.full(len(c), marks[s_]["cy"]))
                    u &= ~_link_hits(pa, pb, at(i, c), marks[i]["h"])
            sub.append(c[u])
        if any(len(c) == 0 for c in sub):
            continue
        pairs = {(a, b): ~_clash(marks[move[a]], sub[a], marks[move[b]], sub[b])
                 for a in range(len(move)) for b in range(a)}  # fmt: skip
        idx = [np.arange(len(c)) for c in sub]
        pruned = True
        while pruned:  # keep the candidates with a partner in every other moving mark
            pruned = False
            for (a, b), t in pairs.items():
                t = t[np.ix_(idx[a], idx[b])]
                ka, kb = t.any(axis=1), t.any(axis=0)
                if not (ka.all() and kb.all()):
                    idx[a], idx[b], pruned = idx[a][ka], idx[b][kb], True
        if any(len(i) == 0 for i in idx):
            continue
        ok = np.ones([len(i) for i in idx], bool)
        for (a, b), t in pairs.items():
            ok &= _expand(t[np.ix_(idx[a], idx[b])], a, b, len(move))
        combo = np.nonzero(ok)
        if not len(combo[0]):
            continue
        offs = [sub[a][idx[a][combo[a]]] for a in range(len(move))]
        size = len(combo[0])

        def where(i: int, offs: Any = offs, size: int = size) -> Any:
            if i in move:
                o = offs[move.index(i)]
                return marks[i]["cx"] + o[:, 0], marks[i]["cy"] + o[:, 1]
            return (np.full(size, marks[i]["cx"]), np.full(size, marks[i]["cy"]))

        good = np.ones(size, bool)  # links with a moving end, against every other mark
        for w, (d, s_) in links.items():
            pa, pb = where(d), where(s_)
            for j in range(n):
                cx, cy = where(j)
                if marks[j]["w"] != w and (d in move or s_ in move or j in move):
                    good &= ~_link_hits(pa, pb, (cx, cy), marks[j]["h"])
                for xc in (marks[j]["xlo"], marks[j]["xhi"]):  # caps near a link
                    for t in np.linspace(-WRITERS_CAPH, WRITERS_CAPH, 5):
                        good &= (_seg_dist((np.full(size, xc), cy + t), pa, pb)
                                 >= WRITERS_GAPS["cap_link"])  # fmt: skip
                for yc in (marks[j]["ylo"], marks[j]["yhi"]):
                    for t in np.linspace(-WRITERS_CAPH, WRITERS_CAPH, 5):
                        good &= (_seg_dist((cx + t, np.full(size, yc)), pa, pb)
                                 >= WRITERS_GAPS["cap_link"])  # fmt: skip
        if good.any():
            cost = np.where(good, sum(np.hypot(o[:, 0], o[:, 1]) for o in offs), np.inf)
            k = int(np.argmin(cost))
            return {marks[i]["key"]: (float(offs[a][k, 0]), float(offs[a][k, 1]))
                    for a, i in enumerate(move)}  # fmt: skip
    return {}  # no clean layout: the marks stay at their values


def _reading_cue(p: _Plot) -> None:
    """The reading of (b) as a muted note on a thin arrow from lower right to upper left, in the
    empty band above the data."""
    from matplotlib.patches import FancyArrowPatch

    x0, y0, x1, y1 = 60.5, 43.6, 31.0, 49.6
    p.ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                   arrowstyle="-|>,head_length=2.6,head_width=1.2",
                                   color=fs.QUIET, lw=0.6, shrinkA=0, shrinkB=0, zorder=3,
                                   mutation_scale=1.0))  # fmt: skip
    ang = math.degrees(math.atan2((y1 - y0) * p.ky, (x1 - x0) * p.kx)) + 180
    p.ax.text((x0 + x1) / 2, (y0 + y1) / 2 + p.dy(0.8), r"poorer $r_0$, larger $B{\times}D$",
              rotation=ang, rotation_mode="anchor", ha="center", va="bottom", fontsize=fs.T_NOTE,
              color=fs.QUIET, style="italic", zorder=6)  # fmt: skip


def _definition_writers(p: _Plot, data: Mapping[str, Any]) -> None:
    """Panel (b): per writer of definition items its redefinition (square) and source revision
    (open circle) at r_0 and B x D, each with thin interval lines in both directions, the two
    joined; marks whose crosshairs would collide are nudged apart (`_nudges`)."""
    ax = p.ax
    _writers_axes(p, False, WRITERS_XB)
    ax.set_xticks([0, 20, 40, 60])
    by = {(q["writer"], q["shift"]): q for q in data["defwriters"]}
    if not by:
        return
    nud = _nudges(_b_marks(p, by))
    pos = {}
    for (w, sh), q in by.items():
        dx, dy = nud.get((w, sh), (0.0, 0.0))
        pos[w, sh] = (100 * q["x"]["est"] + p.dx(dx), 100 * q["y"]["est"] + p.dy(dy))
    for (w, sh), q in by.items():
        x, y = pos[w, sh]
        c = _name_ink(w)
        cap = 2 * WRITERS_CAPH
        _w_interval(p, x, 100 * q["y"]["lo"], 100 * q["y"]["hi"], c, lw=0.6, cap_mm=cap)
        _w_interval(p, y, 100 * q["x"]["lo"], 100 * q["x"]["hi"], c, lw=0.6, cap_mm=cap,
                    vertical=False)  # fmt: skip
    for w in dict.fromkeys([QWEN, OPUS, SONNET, *(w for w, _ in by)]):
        if (w, "def") in pos and (w, "src") in pos:
            (xa, ya), (xb, yb) = pos[w, "def"], pos[w, "src"]
            ax.plot([xa, xb], [ya, yb], color=fs.model_style(w)[1], lw=1.3, zorder=4.6,
                    solid_capstyle="round")  # fmt: skip
    for (w, sh), (x, y) in pos.items():
        if sh == "def":
            _w_square(p, x, y, fs.model_style(w)[1], z=5.2, side=WRITERS_SQ_B)
        else:
            _w_ring(p, x, y, fs.model_style(w)[1], z=5.2, d=WRITERS_CIRC_B)

    def name(x: float, y: float, w: str, ha: str = "left", va: str = "center_baseline") -> None:
        ax.text(x, y, fs.model_style(w)[0], ha=ha, va=va, fontsize=WRITERS_NAME_PT,
                color=_name_ink(w), zorder=6)  # fmt: skip

    # Qwen right of its redefinition's r_0 interval; Opus above its source revision's B x D
    # interval; Sonnet left of its source revision's r_0 interval, a little lower
    if q := by.get((QWEN, "def")):
        name(100 * q["x"]["hi"] + p.dx(1.4), pos[QWEN, "def"][1], QWEN)
    if q := by.get((OPUS, "src")):
        name(pos[OPUS, "src"][0], 100 * q["y"]["hi"] + p.dy(0.9), OPUS, ha="center", va="bottom")
    if q := by.get((SONNET, "src")):
        name(100 * q["x"]["lo"] - p.dx(WRITERS_CAPH + 0.9), pos[SONNET, "src"][1] - p.dy(1.2),
             SONNET, ha="right")  # fmt: skip
    _reading_cue(p)


def _subtitle(fig: Any, ov: Any, x: float, y: float, parts: Sequence[tuple[Any, str]]) -> None:
    """A muted line under a panel title (mm on the overlay `ov`): (glyph or None, text) pieces
    from x on; a piece with a glyph is set upright, one without in italics."""
    r = fig.canvas.get_renderer()
    for g, s in parts:
        if g is not None:
            g(ov, x, y + 0.62)
            x += WRITERS_SQ + 0.9
        t = ov.text(x, y, s, ha="left", va="baseline", fontsize=fs.T_NOTE, color=WRITERS_MUTED,
                    style="italic" if g is None else "normal")  # fmt: skip
        fig.canvas.draw()
        x = t.get_window_extent(r).x1 / fig.dpi * 25.4 + 3.4


def _key_glyph(shift: str) -> Callable[[Any, float, float], None]:
    """The key's glyph of a shift: a neutral square (redefinition) or open circle."""

    def draw(ov: Any, x: float, y: float) -> None:
        from matplotlib.patches import Circle, Rectangle

        c = "#59606B"
        if shift == "def":
            s = WRITERS_SQ - 0.25
            ov.add_patch(Rectangle((x, y - s / 2), s, s, fc=c, ec="none"))
        else:
            d = WRITERS_CIRC - 0.45
            ov.add_patch(Circle((x + d / 2, y), d / 2, fc="white", ec=c, lw=1.15))

    return draw


def fig_writers(inp: Mapping[str, Any], out: Path) -> Path:
    """(a) G-b program sets after S_def; (b) the definition-item writers after both shifts, each
    writer's redefinition mark (square) joined to its source-revision mark (open circle): see the
    module docstring. One value axis (B x D on definition items) for both panels; colour is the
    writer, intervals thin capped lines in the writer's ink. In (a) each set is one square, sets
    on one spot side by side, a grey hull per group of writers and exact recomputation's line
    over its band; in (b) marks whose crosshairs would collide are nudged apart by the least that
    clears them, the interval ends exact (the caption says both). Each writer is named once per
    panel, in its own colour."""
    data = writer_points(inp)
    w_mm, h_mm = WRITERS_PT[0] * fs.PT, WRITERS_PT[1] * fs.PT
    with fs.printed(WRITERS_RC) as plt:
        fig = plt.figure(figsize=(w_mm * fs.MM, h_mm * fs.MM), dpi=300)
        pa = _Plot(fig, WRITERS_LEFT, WRITERS_WA, WRITERS_XA)
        xb0 = WRITERS_LEFT + WRITERS_WA + WRITERS_GAP
        pb = _Plot(fig, xb0, w_mm - WRITERS_RIGHT - xb0, WRITERS_XB)
        _program_sets(pa, data)
        _definition_writers(pb, data)
        ov = fig.add_axes((0, 0, 1, 1), zorder=10)  # titles, subtitles and x titles, in mm
        ov.set_xlim(0, w_mm)
        ov.set_ylim(0, h_mm)
        ov.axis("off")
        x_titles = ((pa, (0, 100), "programs right before the shift (%)"),
                    (pb, (0, 60), r"$r_0$: repair correctness at $D\!=\!0$ (%)"))  # fmt: skip
        for p, (lo, hi), text in x_titles:
            cx = (p.mm(lo, 0)[0] + p.mm(hi, 0)[0]) / 2
            ov.text(cx, 1.1, text, ha="center", va="bottom", fontsize=fs.T_LABEL, color=fs.INK)
        for x, title in ((0.9, "(a) Program sets after a redefinition"),
                         (pb.x0, "(b) Writers of definition items")):  # fmt: skip
            ov.text(x, h_mm - 0.8, title, ha="left", va="top", fontsize=fs.T_TITLE, color=fs.INK,
                    weight="bold")  # fmt: skip
        fig.canvas.draw()
        _subtitle(fig, ov, 0.9, h_mm - 6.9, [(None, "one square per program set")])
        _subtitle(fig, ov, pb.x0, h_mm - 6.9, [(_key_glyph(s), lab) for s, lab in WRITERS_KEY])
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
    the range over the units), how many units pass and the passing units named where only some
    pass; `fresh` on the fresh worlds of the replication. The label is the test's name, a comma,
    then the systems or units."""

    label: str
    threshold: str
    estimate: str
    outcome: str
    fresh: bool = False
    passing: tuple[str, ...] = ()


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
    """How many of n units pass ("3/3")."""
    return f"{passed}/{n}"


def _passes(oks: Sequence[bool]) -> str:
    """The outcome cell of a diagnostic test: how many units pass."""
    return _of(sum(oks), len(oks))


def _test(label: str, threshold: str, xs: Sequence[float], oks: Sequence[bool], outcome: str,
          passing: Sequence[str] = ()) -> list[DiagnosticTest]:  # fmt: skip
    """One row in pp, or none when no unit is staged."""
    if not oks:
        return []
    return [DiagnosticTest(label, threshold, _pp_range(xs), outcome, passing=tuple(passing))]


TIER_KEYS = ("same", "upgrade", "downgrade")
HORIZON_SYSTEMS = (("holdmem_D1B1_source", r"\holdmem{}(1,1)"), ("zero", "no-store BM25"),
                   ("rag", "plain RAG"))  # fmt: skip
DOT = r"\,$\cdot$\,"  # between the systems of a row and between their counts, in the same order
RETENTION = r"90\% interval above $-$3\,pp"


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
    """How many horizons pass the utility floor, per staged system of HORIZON_SYSTEMS, in its
    order ("3/3 . 0/3 . 0/3")."""
    return DOT.join(_of(sum(bool(x["T3_pass"]) for x in per[s]), len(per[s]))
                    for s, _ in HORIZON_SYSTEMS if per[s])  # fmt: skip


def _floor_units(per: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
    """The units of a horizon row: its staged systems in the order of `_floor_counts`, each name
    in one box so that a line breaks only after a dot."""
    names = r"~$\cdot$ ".join(rf"\mbox{{{lab}}}" for s, lab in HORIZON_SYSTEMS if per[s])
    return f"{names}, three horizons"


def _horizon_tests(per: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[DiagnosticTest]:
    """The utility floor and the cost ceiling of HORIZON_SYSTEMS at the three horizons."""
    out: list[DiagnosticTest] = []
    xs = [float(x["minus_oracle"]["est"]) for s, _ in HORIZON_SYSTEMS for x in per[s]
          if _ok(x.get("minus_oracle"))]  # fmt: skip
    if xs:
        out += _test(f"Utility floor, {_floor_units(per)}", r"$\ge-3$\,pp", xs,
                     [bool(x["T3_pass"]) for s, _ in HORIZON_SYSTEMS for x in per[s]],
                     _floor_counts(per))  # fmt: skip
    cost = [x for s, _ in HORIZON_SYSTEMS for x in per[s] if "T4_pass" in x]
    if cost:
        ratio = max(float(x["serve_ratio_to_full_reread"]) for x in cost)
        out.append(DiagnosticTest("Cost ceiling, the same systems and horizons", "at most half",
                                  f"at most {100 * ratio:.1f}\\%",
                                  _passes([bool(x["T4_pass"]) for x in cost])))  # fmt: skip
    return out


def _diagnostic_tests(inp: Mapping[str, Any]) -> list[DiagnosticTest]:
    """The four diagnostic tests with their thresholds, per tier, switch, horizon and system:
    retention (per tier and per switch to another writer), leaving, utility floor and cost
    ceiling (main-study systems at three horizons, native systems and the two baselines), and
    the retention test, utility floor and cost ceiling on the fresh worlds of the replication."""
    out: list[DiagnosticTest] = []
    m1 = inp.get("m1")
    xs, oks = _retention(inp.get("e2"))
    out += _test(r"Retention test, \holdmem{}(1,1), three tiers", RETENTION, xs, oks,
                 _passes(oks))  # fmt: skip
    cross = [
        (t.label, _get(m1, "t1", k) or {}) for t in TIERS if t.marker and (k := tier_key(m1, t))
    ]
    xs, oks = _ests((t.get("delta_90"), t.get("equivalent")) for _, t in cross)
    names = [lab for (lab, t) in cross if t.get("equivalent")]
    out += _test(r"Retention test, \holdmem{}(1,1), five other writers", RETENTION, xs, oks,
                 _passes(oks), names if len(names) < len(oks) else ())  # fmt: skip
    t2 = _get(inp.get("e6"), "imports", "imported_with_ext")
    if t2 and _ok(_get(t2, "diff_90", "post")):
        ok = [bool(t2.get("t2_equivalent"))]
        out += _test(r"Leaving test, \holdmem{}(1,1) with its keys",
                     r"90\% interval within $\pm$3\,pp", [float(t2["diff_90"]["post"]["est"])],
                     ok, _passes(ok))  # fmt: skip
    out += _horizon_tests(_per_horizon(inp.get("e3")))
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
                     oks, _passes(oks), passing if len(passing) < len(oks) else ())  # fmt: skip
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
    out += _test(r"Retention test, \holdmem{}(1,1), three tiers", RETENTION, ret, ret_ok,
                 _passes(ret_ok))  # fmt: skip
    out += _horizon_tests(_per_horizon(inp.get("rep_e3")))
    return [DiagnosticTest(t.label, t.threshold, t.estimate, t.outcome, True, t.passing)
            for t in out]  # fmt: skip


def diagnostic_tests(inp: Mapping[str, Any]) -> list[DiagnosticTest]:
    """The data of `tab_diagnostic_tests`: the four diagnostic tests per set of systems or
    units."""
    return _diagnostic_tests(inp)


NOT_RUN = r"\mtna"  # a test the replication did not run on these units


def tab_diagnostic_tests(tests: Sequence[DiagnosticTest]) -> str:
    """The table of the four diagnostic tests (paper/generated/tab_diagnostic_tests.tex): one row
    per test and set of systems, its estimate(s) and how many units pass in the main study and,
    beside them, in the replication (a dash where the replication did not run it), in the paper's
    table style. The thresholds, the passing units where only some pass and the definitions are
    in the notes. A 2 pt gap separates the tests; a cost ceiling stays with the utility floor
    before it, whose systems it refers to ("the same systems")."""
    main = [t for t in tests if not t.fresh]
    fresh = {t.label: t for t in tests if t.fresh}
    rows: list[tuple[DiagnosticTest | None, DiagnosticTest | None]] = [
        (t, fresh.pop(t.label, None)) for t in main
    ]
    rows += [(None, t) for t in fresh.values()]  # a test of the replication only
    lines = [
        "% generated by memshift.analysis.figures --only tab_diagnostic_tests; do not edit",
        r"\begin{table}[t]",
        r"\caption{The four diagnostic tests (\cref{app:experiments}) per tier, switch, horizon "
        r"and system, in the main study and the replication.}",
        r"\label{tab:diagnostic-tests}",
        r"\centering\mtsetup",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{mttabular}{@{}l >{\raggedright\arraybackslash}p{5cm} c c @{\hspace{14pt}} c c@{}}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{Main study} & \multicolumn{2}{c@{}}{Replication} \\",
        r"\cmidrule(lr){3-4}\cmidrule(l){5-6}",
        r"Test & Systems or units & Estimate & Passes & Estimate & Passes \\",
        r"\midrule",
    ]
    before = None
    for m, r in rows:
        t = m or r
        assert t is not None
        test, units = t.label.split(", ", 1)
        if before is not None and test not in (before, "Cost ceiling"):
            lines.append(r"\addlinespace[2pt]")
        name = "" if test == before else test
        before = test
        cells = [(x.estimate, x.outcome) if x else (NOT_RUN, NOT_RUN) for x in (m, r)]
        lines.append(rf"{name} & {units} & " + " & ".join(c for pair in cells for c in pair)
                     + r" \\")  # fmt: skip
    thresholds: dict[str, str] = {}
    passing: list[str] = []
    for t in tests:
        test, units = t.label.split(", ", 1)
        thresholds.setdefault(test, t.threshold)
        if t.passing:  # named by the last part of its units ("five other writers")
            study = " in the replication" if t.fresh else ""
            passing.append(f"{' and '.join(t.passing)} of the {units.split(', ')[-1]}{study}")
    notes = (
        r"Estimate: point estimates of the change in retention, the leaving test's difference in "
        r"carryover and the margin over the utility floor in pp, and of the serving cost as a "
        r"share of a full re-read, a range over the units. Passes: passing units, per system where "
        r"a row names several. "
        "Thresholds: " + ", ".join(f"{k[0].lower()}{k[1:]} {v}" for k, v in thresholds.items())
        + ". " + (f"Passing: {', '.join(passing)}. " if passing else "")
        + "---: not run in the replication."
    )  # fmt: skip
    lines += [r"\bottomrule", r"\end{mttabular}", rf"\mtnotes{{{notes}}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


FIGURES: dict[str, Callable[[Mapping[str, Any], Path], Path]] = {
    "fig_writers": fig_writers,
    "fig_reader": fig_reader,
    "fig_misses": fig_misses,
}
