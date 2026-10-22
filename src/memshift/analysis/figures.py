"""Paper figures regenerated from the committed run analyses (figures come from runs/).

Sources are the analysis directories listed in results/sources.json, the same ones the
paper's number macros are generated from.

fig_calibration    E2: survival of the D=0, B=1 items after the switch per bin of writer agreement
                   on the calibration set, pooled over the three tiers, with the pooled fit.
fig_zero_cost      E3: ContractValid before any shift by horizon; tokens per query against one
                   full re-read.
fig_native_systems Block N: every native system against the HoldMem cell with its inferred
                   label, per branch and metric (Update, Stability); systems that were not run
                   show as such.
fig_classsplit     `ex_e1_paper`: B and B x D on Update by item class and branch
fig_account        `ex_review_paper`: B x D against the repair gap r1 - r0 by writer
fig_writers        round 3 (`round3_figures`): B x D against how often the programs are right (G-b
                   program sets) and against r_0 (definition-item writers of block H)
fig_reader         round 3: B x D with each reader of the same stores against the store part
fig_misses         why the cited-cluster key misses changed item-times
tab_diagnostic_tests  not a figure: paper/generated/tab_diagnostic_tests.tex (in --tables), the
                   four diagnostic tests per set of systems (`round3_figures`)
Usage: python -m memshift.analysis.figures [--root .] [--out paper/figures] [--only NAME ...]
       [--tables paper/generated]
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeGuard

from memshift.analysis import figstyle as fs
from memshift.analysis.paper_data import (
    NATIVES,
    Native,
    label_text,
    load_stage4,
    native_results,
)
from memshift.analysis.paper_data import get as _get_path


def _ok(e: dict[str, Any] | None) -> TypeGuard[dict[str, Any]]:
    return e is not None and e.get("est") is not None and not math.isnan(e["est"])


def _spread(ys: list[float], gap: float) -> list[float]:
    """Positions for labels aimed at `ys`: in the same order, at least `gap` apart, each run of
    labels that would touch centred on the mean of its targets."""
    groups: list[list[int]] = []

    def start(g: list[int]) -> float:
        return sum(ys[i] for i in g) / len(g) - gap * (len(g) - 1) / 2

    for i in sorted(range(len(ys)), key=ys.__getitem__):
        groups.append([i])
        while len(groups) > 1 and start(groups[-2]) + gap * len(groups[-2]) > start(groups[-1]):
            top = groups.pop()
            groups[-1] += top
    out = [0.0] * len(ys)
    for g in groups:
        for k, i in enumerate(g):
            out[i] = start(g) + k * gap
    return out


def _rich(cv: fs.MmCanvas, x: float, y: float, segs: Sequence[tuple[str, ...]], size: float,
          ha: str = "left") -> float:  # fmt: skip
    """Inline segments (text, colour[, weight]) on one baseline, placed by their exact advance
    widths (spaces kept); a segment with a dollar sign is math. (x, y): the anchor and the
    vertical centre (mm). Returns the right end."""
    weights = [s[2] if len(s) > 2 else "normal" for s in segs]
    widths = [fs.advance(s[0], size, w) for s, w in zip(segs, weights, strict=True)]
    x0 = x - (sum(widths) if ha == "right" else sum(widths) / 2 if ha == "center" else 0.0)
    base = y - 0.33 * size * fs.PT
    for s, weight, w in zip(segs, weights, widths, strict=True):
        cv.text(x0, base, s[0], size=size, color=s[1], weight=weight, ha="left", va="baseline")
        x0 += w
    return x0


CALIBRATION_BINS = (0.0, 0.5, 0.8, 0.9, 0.95, 1.0)  # agreement bins of the calibration figure
AREA_PER_ITEM = 0.1  # marker area (pt^2) per item of a bin


def calibration_counts(e2: dict[str, Any]) -> list[tuple[float, int, int]]:
    """(agreement, surviving items, failed items) per distinct writer agreement, ascending, pooled
    over the tiers: the D=0, B=1 items of the per-type calibration records, whose survivals
    (observed x n) are whole items."""
    counts: dict[float, list[int]] = {}
    for tier in (e2.get("calibration") or {}).values():
        for p in ((tier or {}).get("per_type") or {}).values():
            n, alive = int(p["n"]), round(p["observed"] * p["n"])
            if abs(alive - p["observed"] * p["n"]) > 1e-6:
                raise ValueError(f"survivals not a whole number of items: {p}")
            c = counts.setdefault(round(p["predicted"], 9), [0, 0])
            c[0] += alive
            c[1] += n - alive
    return [(a, s, f) for a, (s, f) in sorted(counts.items())]


def calibration_bins(e2: dict[str, Any]) -> list[dict[str, float]]:
    """Per bin of `CALIBRATION_BINS` with items (the last bin closed): the item-weighted mean
    agreement, the share of the items that survive and the number of items."""
    edges, counts = CALIBRATION_BINS, calibration_counts(e2)
    out = []
    for i, (lo, hi) in enumerate(zip(edges, edges[1:], strict=False)):
        last = i == len(edges) - 2
        inside = [c for c in counts if lo <= c[0] and (c[0] < hi or last)]
        n = sum(s + f for _, s, f in inside)
        if n:
            out.append(
                {
                    "lo": lo,
                    "hi": hi,
                    "agreement": sum(a * (s + f) for a, s, f in inside) / n,
                    "survival": sum(s for _, s, _ in inside) / n,
                    "n": n,
                }
            )
    return out


CALIBRATION_PT = (324, 194.4)  # printed size (pt): `figstyle.NARROW` wide
CAL_FRAME = (12.6, 110.6, 9.2, 63.6)  # the plot frame (mm): left, right, bottom, top
CAL_REF = "#8E99A8"  # the diagonal of perfect calibration
CAL_BIN_TINT = "#F3F5F8"  # every other agreement bin
CAL_LARGE_BIN = 1000  # a bin with more items: count below-left of its marker, marker translucent


def fig_calibration(e2: dict[str, Any], out: Path) -> Path:
    """Survival of a switch against writer agreement for the D=0, B=1 items pooled over the
    tiers: the agreement bins as alternate shaded columns, each bin's survival at its mean
    agreement (area = items, the number beside it), the pooled logistic fit with its 90% band in
    the same blue, the dashed diagonal of perfect calibration, and the calibration slope and
    AUROC with their 90% intervals."""
    from matplotlib.patches import Circle, Polygon, Rectangle

    blue = fs.ROLE_BLUE
    x0, x1, y0, y1 = CAL_FRAME

    def px(v: float) -> float:
        return x0 + v * (x1 - x0)

    def py(v: float) -> float:
        return y0 + v * (y1 - y0)

    with fs.printed() as plt:
        cv = fs.MmCanvas(plt, CALIBRATION_PT[0] * fs.PT, CALIBRATION_PT[1] * fs.PT, size=7.5)
        ax = cv.ax
        edges = CALIBRATION_BINS
        for i, (lo, hi) in enumerate(zip(edges, edges[1:], strict=False)):
            if i % 2 == 1:
                ax.add_patch(Rectangle((px(lo), y0), px(hi) - px(lo), y1 - y0, fc=CAL_BIN_TINT,
                                       ec="none", zorder=0.2))  # fmt: skip
        for v in (0.2, 0.4, 0.6, 0.8, 1.0):
            ax.plot([x0, x1], [py(v)] * 2, color=fs.GRID, lw=0.5, zorder=0.5)
        ax.plot([x0, x1], [y0, y0], color=fs.SPINE, lw=0.6, zorder=1)
        ax.plot([x0, x0], [y0, y1], color=fs.SPINE, lw=0.6, zorder=1)
        for v in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            ax.plot([px(v)] * 2, [y0, y0 - 0.8], color=fs.SPINE, lw=0.6, zorder=1)
            cv.text(px(v), y0 - 2.4, f"{100 * v:.0f}", size=7.2, color=fs.TICK)
            ax.plot([x0, x0 - 0.8], [py(v)] * 2, color=fs.SPINE, lw=0.6, zorder=1)
            cv.text(x0 - 1.5, py(v), f"{100 * v:.0f}", size=7.2, color=fs.TICK, ha="right")
        cv.text((x0 + x1) / 2, 2.5, "writer agreement (%)", size=8.0, color=fs.INK)
        cv.text(3.4, (y0 + y1) / 2, "survival after the switch (%)", size=8.0, color=fs.INK,
                rotation=90)  # fmt: skip
        cv.text(px(0.65), y1 - 1.9, "agreement bins", size=7.0, color=fs.MUTED, style="italic")
        # perfect calibration, named along the line
        ax.plot([px(0), px(1)], [py(0), py(1)], color=CAL_REF, lw=0.75, ls=(0, (3.2, 2.0)),
                zorder=1.5)  # fmt: skip
        ang = math.degrees(math.atan2(y1 - y0, x1 - x0))
        cv.text(px(0.075), py(0.075) - 1.25, "perfect calibration", size=7.0, color=fs.MUTED,
                rotation=ang, rotation_mode="anchor", ha="left", va="top",
                style="italic")  # fmt: skip
        # the pooled logistic fit and its 90% band, named where it starts
        curve = e2.get("explore_pooled_curve") or {}
        if curve.get("est") and not any(math.isnan(v) for v in curve["est"]):
            g, est, lo90, hi90 = curve["grid"], curve["est"], curve["lo90"], curve["hi90"]
            band = [(px(x), py(v)) for x, v in zip(g, hi90, strict=True)]
            band += [(px(x), py(v)) for x, v in zip(reversed(g), reversed(lo90), strict=True)]
            ax.add_patch(Polygon(band, closed=True, fc=blue["soft"], ec="none", alpha=0.75,
                                 zorder=2))  # fmt: skip
            ax.plot([px(x) for x in g], [py(v) for v in est], color=blue["ink"], lw=1.1,
                    zorder=3, solid_capstyle="round")  # fmt: skip
            cv.text(px(g[0]) - 1.2, py(est[0]) + 1.25, "logistic fit", size=7.4,
                    color=blue["ink"], ha="right")  # fmt: skip
            cv.text(px(g[0]) - 1.2, py(est[0]) - 1.6, "90% band", size=7.0, color=blue["ink"],
                    ha="right", style="italic")  # fmt: skip
        # one marker per bin, area = items
        placed = []
        for b in calibration_bins(e2):
            r = math.sqrt(AREA_PER_ITEM * b["n"]) * fs.PT / 2  # a scatter marker of that area
            cx, cy = px(b["agreement"]), py(b["survival"])
            ax.add_patch(Circle((cx, cy), r, fc=blue["mark"], ec=blue["ink"], lw=0.45, zorder=6,
                                alpha=0.92 if b["n"] > CAL_LARGE_BIN else 1.0))  # fmt: skip
            placed.append((b, cx, cy, r))
        # the counts: above a marker on or above the diagonal, below one under it; a large bin's
        # count below-left of its marker, clear of the top edge
        for b, cx, cy, r in placed:
            lab = f"{b['n']:,}"
            if b["n"] > CAL_LARGE_BIN:
                cv.text(cx + r, cy - r - 1.3, lab, size=7.2, color=blue["ink"], ha="right",
                        va="top")  # fmt: skip
            elif b["survival"] >= b["agreement"]:
                cv.text(cx, cy + r + 0.6, lab, size=7.2, color=blue["ink"], va="bottom")
            else:
                cv.text(cx, cy - r - 0.7, lab, size=7.2, color=blue["ink"], va="top")
        # the pooled calibration slope and AUROC with their 90% intervals
        p3 = e2.get("path3") or {}
        sl, au = p3.get("pooled_slope") or {}, p3.get("pooled_auroc") or {}
        if sl and au:
            xr, y_top = x1 - 0.4, py(0.2) - 1.9  # below the 20% gridline, clear of the data
            for k, (name, e) in enumerate((("calibration slope ", sl), ("AUROC ", au))):
                _rich(cv, xr, y_top - 3.4 * k,
                      [(name, fs.INK), (f"{e['est']:.2f}", blue["ink"], "bold"),
                       (f"  [{e['lo90']:.2f}, {e['hi90']:.2f}]", fs.INK)],
                      size=7.4, ha="right")  # fmt: skip
            cv.text(xr, y_top - 6.6, "90% intervals", size=7.0, color=fs.MUTED, ha="right",
                    style="italic")  # fmt: skip
        return fs.save_pdf(cv.fig, out / "fig_calibration.pdf")


ZERO_PT = (486, 151.2)  # printed size of the zero-system figure (pt): full width
ZERO_SYSTEMS = {  # system -> (label, colour, dash, marker); HoldMem(1,1) is the hero
    "holdmem_D1B1_source": ("HOLDMEM(1,1)", fs.ROLE_BLUE["mark"], "-", "o"),
    "oracle": ("oracle context", "#5E6875", (0, (1.0, 1.4)), "D"),
    "zero": ("no-store BM25", "#262B33", "-", "s"),
    "rag": ("plain RAG", "#8E99A8", (0, (3.4, 1.8)), "^"),
}
ZERO_A = (12.4, 81.0, 9.6, 47.4)  # panel (a) frame (mm): left, right, bottom, top
ZERO_B = (110.5, 169.6)  # panel (b) left and right (mm); bottom and top as (a)
ZERO_FAIL = "#ECEEF2"  # the region that fails the cost ceiling
ZERO_GAP_AT = 0.735  # accuracy at which the 12-month gap is printed, inside its bracket


def _system_name(cv: fs.MmCanvas, x: float, y: float, label: str, size: float, color: str, *,
                 weight: str = "normal") -> float:  # fmt: skip
    """A system's name with (x, y) its left end and vertical centre (mm); `HOLDMEM(...)` is set
    in small capitals, as \\textsc{HoldMem} in the paper. Returns the right end."""
    if label.startswith("HOLDMEM"):
        segs = [("H", 1.0), ("OLD", 0.78), ("M", 1.0), ("EM", 0.78), (label[7:], 1.0)]
    else:
        segs = [(label, 1.0)]
    widths = [fs.advance(s, size * k, weight) for s, k in segs]
    base = y - 0.33 * size * fs.PT
    for (s, k), w in zip(segs, widths, strict=True):
        cv.text(x, base, s, size=size * k, color=color, weight=weight, ha="left", va="baseline")
        x += w
    return x


def _system_marker(ax: Any, x: float, y: float, kind: str, color: str, d: float, *,
                   hollow: bool = False, z: float = 6) -> None:  # fmt: skip
    """A marker `d` mm wide at (x, y) mm: circle (o), square (s), triangle (^) or diamond (D)."""
    from matplotlib.patches import Circle, Rectangle, RegularPolygon

    fc, lw = ("white", 0.75) if hollow else (color, 0.4)
    p: Any
    if kind == "o":
        p = Circle((x, y), d / 2, fc=fc, ec=color, lw=lw, zorder=z)
    elif kind == "s":
        s = d * 0.86
        p = Rectangle((x - s / 2, y - s / 2), s, s, fc=fc, ec=color, lw=lw, zorder=z)
    elif kind == "^":
        p = RegularPolygon((x, y - d * 0.08), 3, radius=d * 0.62, fc=fc, ec=color, lw=lw,
                           zorder=z)  # fmt: skip
    else:
        p = RegularPolygon((x, y), 4, radius=d * 0.6, fc=fc, ec=color, lw=lw, zorder=z)
    ax.add_patch(p)


def _zero_accuracy(cv: fs.MmCanvas, e3: dict[str, Any]) -> None:
    """Panel (a): ContractValid before any shift by horizon (log scale), 95% bands, each line
    named at its right end, and the 12-month gap of HoldMem(1,1) over no-store BM25 with its 95%
    interval beside a bracket."""
    from matplotlib.patches import Polygon

    ax, blue = cv.ax, fs.ROLE_BLUE
    x0, x1, y0, y1 = ZERO_A
    hs = sorted(int(h) for h in e3["horizons"])
    x_lo, x_hi = hs[0] / 1.25, hs[-1] * 1.25

    def px(h: float) -> float:
        span = math.log10(x_hi) - math.log10(x_lo)
        return x0 + (math.log10(h) - math.log10(x_lo)) / span * (x1 - x0)

    lows: list[float] = []
    highs: list[float] = []
    for s in ZERO_SYSTEMS:
        for h in hs:
            e = e3["horizons"][str(h)].get(s, {}).get("contract_valid")
            if e:
                lows.append(e["lo"])
                highs.append(e["hi"])
    lo, hi = math.floor(20 * min(lows)) / 20, math.ceil(20 * max(highs)) / 20  # 5-point steps

    def py(v: float) -> float:
        return y0 + (v - lo) / (hi - lo) * (y1 - y0)

    cv.label(0.6, cv.h - 0.6, "a", "Accuracy before any shift", size=8.2)
    v = math.ceil(lo * 10) / 10
    while v <= hi + 1e-9:
        ax.plot([x0, x1], [py(v)] * 2, color=fs.GRID, lw=0.5, zorder=0.5)
        ax.plot([x0 - 0.8, x0], [py(v)] * 2, color=fs.SPINE, lw=0.6, zorder=1)
        cv.text(x0 - 1.5, py(v), f"{100 * v:.0f}", size=7.2, color=fs.TICK, ha="right")
        v += 0.1
    ax.plot([x0, x1], [y0, y0], color=fs.SPINE, lw=0.6, zorder=1)
    ax.plot([x0, x0], [y0, y1], color=fs.SPINE, lw=0.6, zorder=1)
    for h in hs:
        ax.plot([px(h)] * 2, [y0, y0 - 0.8], color=fs.SPINE, lw=0.6, zorder=1)
        cv.text(px(h), y0 - 2.4, str(h), size=7.2, color=fs.TICK)
    cv.text((x0 + x1) / 2, 2.6, "horizon (months, log scale)", size=8.0)
    cv.text(3.3, (y0 + y1) / 2, "accuracy (%)", size=8.0, rotation=90)
    ends: list[tuple[str, float]] = []
    last: dict[str, float] = {}
    for s, (_, colour, dash, mk) in ZERO_SYSTEMS.items():
        pts = [(h, e3["horizons"][str(h)][s]["contract_valid"]) for h in hs
               if s in e3["horizons"][str(h)]]  # fmt: skip
        if not pts:
            continue
        hero = s.startswith("holdmem")
        band = [(px(h), py(e["hi"])) for h, e in pts]
        band += [(px(h), py(e["lo"])) for h, e in reversed(pts)]
        ax.add_patch(Polygon(band, closed=True, fc=blue["soft"] if hero else colour,
                             alpha=0.8 if hero else 0.13, ec="none",
                             zorder=2 if hero else 1.6))  # fmt: skip
        ax.plot([px(h) for h, _ in pts], [py(e["est"]) for _, e in pts], color=colour,
                lw=1.6 if hero else 1.0, ls=dash, zorder=4 if hero else 3,
                solid_capstyle="round", dash_capstyle="butt")  # fmt: skip
        for h, e in pts:
            _system_marker(ax, px(h), py(e["est"]), mk, colour, 1.75 if hero else 1.45,
                           z=5 if hero else 4.5)  # fmt: skip
        ends.append((s, py(pts[-1][1]["est"])))
        last[s] = pts[-1][1]["est"]
    for (s, _), y in zip(ends, _spread([y for _, y in ends], 3.1), strict=True):
        label, colour, _, _ = ZERO_SYSTEMS[s]
        hero = s.startswith("holdmem")
        _system_name(cv, x1 + 1.4, y, label, 7.6, blue["ink"] if hero else colour,
                     weight="bold" if hero else "normal")  # fmt: skip
    # the gap at the longest horizon: HoldMem(1,1) over no-store BM25 (paired, 95% interval)
    gap = e3["horizons"][str(hs[-1])].get("zero_minus_holdmem")
    hold, zero = last.get("holdmem_D1B1_source"), last.get("zero")
    if _ok(gap) and hold is not None and zero is not None:
        yh, yz = py(hold), py(zero)
        xb = px(hs[-1] * 1.105)
        cv.line([(xb - 0.9, yh), (xb, yh), (xb, yz), (xb - 0.9, yz)], color=blue["ink"], lw=0.7,
                radius=0.3)  # fmt: skip
        ym = py(min(max(ZERO_GAP_AT, min(hold, zero)), max(hold, zero)))
        cv.text(xb - 1.2, ym + 1.3, f"{-100 * gap['est']:.1f} pp", size=7.6, color=blue["ink"],
                weight="bold", ha="right")  # fmt: skip
        cv.text(xb - 1.2, ym - 1.6, f"[{-100 * gap['hi']:.1f}, {-100 * gap['lo']:.1f}]",
                size=7.0, color=blue["ink"], ha="right")  # fmt: skip


def _zero_cost(cv: fs.MmCanvas, e3: dict[str, Any]) -> None:
    """Panel (b): tokens per query at the longest horizon (log scale), serving only (filled) and
    with amortised ingestion and maintenance (hollow), against one full re-read and the region
    that fails the cost ceiling; the ratio of HoldMem(1,1) to no-store BM25 with maintenance."""
    from matplotlib.patches import Rectangle

    from memshift.analysis.e3 import COST_CEILING

    ax, blue = cv.ax, fs.ROLE_BLUE
    x0, x1 = ZERO_B
    y0, y1 = ZERO_A[2:]
    hs = sorted(int(h) for h in e3["horizons"])
    last = e3["horizons"][str(hs[-1])]
    rows = [s for s in ZERO_SYSTEMS if "cost_mean" in last.get(s, {})]
    if not rows:
        return
    reread = last[rows[0]]["full_reread_tokens_per_query"]
    ceiling = COST_CEILING * reread
    x_lo = min(last[s]["serve_tokens_per_query"] for s in rows) / 1.8
    x_hi = 10 ** math.ceil(math.log10(reread * 1.5))

    def px(v: float) -> float:
        span = math.log10(x_hi) - math.log10(x_lo)
        return x0 + (math.log10(v) - math.log10(x_lo)) / span * (x1 - x0)

    cv.label(x0 - 7.0, cv.h - 0.6, "b", "Tokens per query at twelve months", size=8.2)
    ax.add_patch(Rectangle((px(ceiling), y0), x1 - px(ceiling), y1 - y0, fc=ZERO_FAIL, ec="none",
                           zorder=0.3))  # fmt: skip
    ax.plot([px(ceiling)] * 2, [y0, y1], color="#B4BBC5", lw=0.6, ls=(0, (2.4, 1.6)), zorder=0.6)
    ax.plot([px(reread)] * 2, [y0, y1], color=fs.INK, lw=0.8, zorder=1.2)
    cv.text(px(ceiling) - 0.7, y1 - 0.6, "cost ceiling", size=7.0, color=fs.MUTED, ha="right",
            va="top", rotation=90, style="italic",
            bbox={"fc": "white", "ec": "none", "pad": 0.6})  # fmt: skip
    cv.text(px(reread) + 1.0, y0 + 1.0, "full re-read", size=7.0, color=fs.INK, ha="left",
            va="bottom", rotation=90)  # fmt: skip
    cv.text((px(reread) + x1) / 2 + 0.4, y1 - 1.2, "fails", size=7.0, color=fs.MUTED, va="top",
            style="italic")  # fmt: skip
    ax.plot([x0, x1], [y0, y0], color=fs.SPINE, lw=0.6, zorder=1)
    for k in range(math.ceil(math.log10(x_lo)), round(math.log10(x_hi)) + 1):
        ax.plot([px(10**k)] * 2, [y0, y1], color=fs.GRID, lw=0.5, zorder=0.5)
        ax.plot([px(10**k)] * 2, [y0, y0 - 0.9], color=fs.SPINE, lw=0.6, zorder=1)
        cv.text(px(10**k), y0 - 2.6, f"$10^{{{k}}}$", size=7.2, color=fs.TICK)
    for k in range(math.floor(math.log10(x_lo)), round(math.log10(x_hi))):
        for m in range(2, 10):
            if x_lo <= (v := m * 10**k) <= x_hi:
                ax.plot([px(v)] * 2, [y0, y0 - 0.5], color=fs.SPINE, lw=0.45, zorder=1)
    cv.text((x0 + x1) / 2, 2.6, "tokens per query (log scale)", size=8.0)
    ys = {s: y1 - 8.2 - 11.0 * k for k, s in enumerate(rows)}
    vals: dict[str, tuple[float, float]] = {}
    for s in rows:
        _, colour, _, mk = ZERO_SYSTEMS[s]
        hero = s.startswith("holdmem")
        c = last[s]
        serve = c["serve_tokens_per_query"]
        queries = c["cost_mean"]["serve_tokens"] / max(serve, 1e-9)
        parts = ("ingest_tokens", "maintain_tokens", "serve_tokens")
        total = sum(c["cost_mean"][p] for p in parts) / queries
        vals[s] = (serve, total)
        ax.plot([px(serve), px(total)], [ys[s]] * 2, color=colour, lw=1.0 if hero else 0.8,
                zorder=4, solid_capstyle="butt")  # fmt: skip
        d = 2.0 if hero else 1.7
        _system_marker(ax, px(total), ys[s], mk, colour, d, hollow=True, z=5)
        _system_marker(ax, px(serve), ys[s], mk, colour, d, z=6)
    for s in rows:  # each name right of its row's last marker
        label, colour, _, _ = ZERO_SYSTEMS[s]
        hero = s.startswith("holdmem")
        _system_name(cv, px(vals[s][1]) + 2.4, ys[s], label, 7.6,
                     blue["ink"] if hero else colour,
                     weight="bold" if hero else "normal")  # fmt: skip
    if "holdmem_D1B1_source" not in vals:
        return
    # what the two markers are, on the hero's row
    hs_, ht = vals["holdmem_D1B1_source"]
    yh = ys["holdmem_D1B1_source"]
    cv.text(px(hs_), yh + 2.3, "serving", size=7.0, color=fs.MUTED, style="italic", va="bottom")
    cv.text(px(ht) - 1.0, yh + 2.3, "+ ingestion, maintenance", size=7.0, color=fs.MUTED,
            style="italic", va="bottom", ha="left")  # fmt: skip
    if "zero" not in vals:
        return
    # the cost model's ratio: HoldMem(1,1) against no-store BM25, both with maintenance
    zt, yz = vals["zero"][1], ys["zero"]
    yb = (yh + yz) / 2 + 0.6
    cv.line([(px(zt), yz + 1.4), (px(zt), yb)], color=blue["ink"], lw=0.5)
    cv.line([(px(zt), yb), (px(ht), yb), (px(ht), yh - 1.3)], color=blue["ink"], lw=0.7,
            radius=0.3)  # fmt: skip
    _rich(cv, px(ht) + 1.2, yb, [(f"{ht / zt:.2f}×", blue["ink"], "bold"),
                                 (" no-store BM25", blue["ink"])], size=7.2)  # fmt: skip


def fig_zero_cost(e3: dict[str, Any], out: Path) -> Path:
    """(a) ContractValid before any shift by horizon with 95% bands, each line named at its end,
    and the gap of HoldMem(1,1) over no-store BM25 at the longest horizon; (b) tokens per query
    at the longest horizon, serving (filled) and serving with amortised ingestion and maintenance
    (hollow), against one full re-read and the cost ceiling, with the cost ratio of HoldMem(1,1)
    to no-store BM25."""
    with fs.printed() as plt:
        cv = fs.MmCanvas(plt, ZERO_PT[0] * fs.PT, ZERO_PT[1] * fs.PT, size=7.5)
        _zero_accuracy(cv, e3)
        _zero_cost(cv, e3)
        return fs.save_pdf(cv.fig, out / "fig_zero_cost.pdf")


NATIVE_BRANCHES = (
    ("src", r"$S_{\mathrm{src}}$"),
    ("def", r"$S_{\mathrm{def}}$"),
    ("orphan", r"$S_{\mathrm{del}}$"),  # the deletion branch ("orphan" in the analyses)
)
NATIVE_METRICS = ("Update", "Stability")


def native_panels(e5: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The data of `fig_native_systems`: one record per expected system (Mem0 and the five of
    block N, then any other system in the analysis) with its inferred label and, per branch and
    metric, the native estimate, the interval of the HoldMem cell with the same label (`cell`)
    and whether the estimate lies inside it. A system that is not in the analysis has `staged`
    False and no values."""
    systems = (e5 or {}).get("systems") or {}
    extra = [Native(s, s, s) for s in sorted(systems) if s not in {n.key for n in NATIVES}]
    out = []
    for nat in (*NATIVES, *extra):
        entry = systems.get(nat.key)
        lab = _get_path(e5, "labels", nat.key) or {}
        d = 0 if (lab.get("D") or {}).get("label") is None else lab["D"]["label"]
        b = (lab.get("B") or {}).get("label") if lab.get("B") is not None else None
        rec: dict[str, Any] = {
            "system": nat.key,
            "label": nat.label,
            "staged": entry is not None,
            "inferred": label_text(d, b) if entry is not None and lab.get("B") else None,
            "cell": (entry or {}).get("predicted_by"),
            "values": {},
        }
        for br, _name in NATIVE_BRANCHES:
            for m in NATIVE_METRICS:
                nat_est = _get_path(entry, "branches", br, "native", m)
                pred = _get_path(entry, "branches", br, "white_box", m)
                rec["values"][br, m] = {
                    "native": nat_est if _ok(nat_est) else None,
                    "pred": pred if _ok(pred) else None,
                    "inside": _get_path(entry, "branches", br, "consistent", m),
                }
        out.append(rec)
    return out


NATIVE_PT = (486, 230.4)  # printed size of the native-systems figure (pt): full width
NATIVE_SHIFT_NAMES = {"src": "source revision", "def": "redefinition", "orphan": "deletion"}
NATIVE_PALE = "#E3E6EA"  # a HoldMem cell's bar to its estimate (the store-part grey of fig_reader)
NATIVE_ARMS = {  # binding arm -> its cell's interval band, estimate line, label ink and rail
    "none": {"band": fs.tint(fs.ORANGE, 0.8), "line": "#9A6400", "ink": "#8A5A0B",
             "rail": fs.ORANGE},
    "cited": {"band": fs.tint(fs.SKY, 0.8), "line": "#1F6FA8", "ink": "#1C5F95",
              "rail": fs.SKY},
    "source": {"band": fs.tint(fs.BLUE, 0.8), "line": fs.ROLE_BLUE["ink"],
               "ink": fs.ROLE_BLUE["ink"], "rail": fs.BLUE},
}  # fmt: skip
NATIVE_INSIDE = {"fill": "#4A5260", "edge": None, "ink": "#0E1116", "inner": "white"}
NATIVE_OUTSIDE = {"fill": "white", "edge": "#4A5260", "ink": "#4A5260", "inner": None}
NATIVE_SC = 0.84  # small capitals of HoldMem: scale of the lower letters
# layout (mm): the names end at NAME_R; a rail at RAIL in the cell's colour binds a group's names
# to its label; three panels from P0 with gutters GUT; rows PITCH apart, groups GROUP_GAP more; a
# native bar BAR_T thick inside its cell's block (BLOCK_PAD beyond the bars); interval caps CAP
# long; the bottom panels' axis at BOT, each panel PH high under a STRIP for its counts
NATIVE_NAME_R, NATIVE_RAIL, NATIVE_P0, NATIVE_GUT, NATIVE_RIGHT = 26.4, 9.6, 28.0, 4.2, 2.3
NATIVE_PITCH, NATIVE_GROUP_GAP, NATIVE_BAR_T, NATIVE_BLOCK_PAD = 3.9, 2.3, 2.1, 0.8
NATIVE_CAP, NATIVE_LINE_LW, NATIVE_EDGE_LW = 1.3, 0.9, 0.6
NATIVE_BOT, NATIVE_PH, NATIVE_STRIP = 7.9, 26.7, 5.0
NATIVE_XMAX = 101.5  # the axis runs 0-100; the room beyond keeps a band at 99.8 off the edge


def native_cell_style(cell: str | None) -> dict[str, Any] | None:
    """The style of a HoldMem cell id such as `holdmem_D0B1_cited`: its label as segments (text,
    1 full size or 0 small capital), its short label (d,b), and the colours of its binding arm
    (`NATIVE_ARMS`); None for no cell."""
    if not cell or not cell.startswith("holdmem_D") or len(cell) < 12:
        return None
    d, b = cell[9], cell[11]
    arm = "none" if b == "0" else ("cited" if cell.endswith("cited") else "source")
    where = {"cited": ", citation key", "source": ", source key", "none": ""}[arm]
    segs = [("H", 1), ("OLD", 0), ("M", 1), ("EM", 0), (f"({d},{b}){where}", 1)]
    return {"segments": segs, "short": f"({d},{b})", "pale": NATIVE_PALE, **NATIVE_ARMS[arm]}


def native_groups(panels: Sequence[dict[str, Any]]) -> list[tuple[str | None, list[int]]]:
    """Consecutive systems with the same inferred cell, in the order of `native_panels`. A cell's
    systems are drawn in front of one bar, so they must be contiguous and share one cell estimate
    per branch and metric; ValueError otherwise."""
    groups: list[tuple[str | None, list[int]]] = []
    for k, rec in enumerate(panels):
        if not groups or groups[-1][0] != rec["cell"]:
            groups.append((rec["cell"], []))
        groups[-1][1].append(k)
    cells = [c for c, _ in groups if c is not None]
    if len(set(cells)) != len(cells):
        raise ValueError(f"the systems of a cell are not contiguous: {cells}")
    for cell, ks in groups:
        for key in panels[ks[0]]["values"]:
            preds = {tuple(round(p[f], 12) for f in ("est", "lo", "hi"))
                     for k in ks if (p := panels[k]["values"][key]["pred"])}  # fmt: skip
            if len(preds) > 1:
                raise ValueError(f"{cell} {key}: its systems differ in the cell's estimate")
    return groups


def _segments_mm(ov: Any, x: float, y: float, segs: Sequence[tuple[str, float]],
                 color: str) -> float:  # fmt: skip
    """Segments (text, size pt) on one baseline from x (mm), vertically centred on y (mm) by the
    first segment; returns the right end."""
    base = y - 0.34 * segs[0][1] * fs.PT
    for s, size in segs:
        ov.text(x, base, s, fontsize=size, color=color, va="baseline", ha="left")
        x += fs.advance(s, size)
    return x


def fig_native_systems(e5: dict[str, Any] | None, out: Path) -> Path:
    """Per metric (rows) and shift (columns), bullet bars on a 0-100% axis: per group of systems
    with the same inferred HoldMem cell, one wide pale bar for that cell (its 95% interval as a
    band in the colour of its binding arm, a dark line at its estimate), and each native system
    as a narrower bar in front of it with its 95% interval, dark slate when its estimate lies
    inside the cell's interval, outlined when outside; above each panel how many lie inside.
    Systems that were not run show as such."""
    from matplotlib.patches import Rectangle

    panels = native_panels(e5)
    groups = native_groups(panels)
    w_mm, h_mm = NATIVE_PT[0] * fs.PT, NATIVE_PT[1] * fs.PT
    p0, gut, ph = NATIVE_P0, NATIVE_GUT, NATIVE_PH
    pw = (w_mm - p0 - NATIVE_RIGHT - 2 * gut) / 3
    bar_t = NATIVE_BAR_T
    # row centres, measured down from a panel's top (mm)
    inner = len(panels) * NATIVE_PITCH + (len(groups) - 1) * NATIVE_GROUP_GAP
    yc: dict[int, float] = {}
    y = ph - (ph - inner) / 2 - NATIVE_PITCH / 2
    for _, ks in groups:
        for k in ks:
            yc[k] = y
            y -= NATIVE_PITCH
        y -= NATIVE_GROUP_GAP
    styles = {cell: native_cell_style(cell) for cell, _ in groups}
    with fs.printed(fs.BAR_RC) as plt:
        fig = plt.figure(figsize=(w_mm * fs.MM, h_mm * fs.MM))
        ov = fig.add_axes((0, 0, 1, 1), zorder=10)  # names, headers and the key, in mm
        ov.set_xlim(0, w_mm)
        ov.set_ylim(0, h_mm)
        ov.axis("off")
        fig.canvas.draw()
        for i, m in enumerate(NATIVE_METRICS):
            bot = NATIVE_BOT  # the last metric's row; each row above it one panel and strip up
            for _ in range(len(NATIVE_METRICS) - 1 - i):
                bot = bot + ph + NATIVE_STRIP
            top = bot + ph
            ov.text(1.0, top + 1.8, m, fontsize=fs.T_TITLE, color=fs.INK, ha="left",
                    va="center")  # fmt: skip
            for cell, ks in groups:
                if (sty := styles[cell]) is not None:
                    ymid = bot + (yc[ks[0]] + yc[ks[-1]]) / 2
                    ov.text(NATIVE_RAIL - 1.1, ymid, sty["short"], fontsize=fs.T_LABEL,
                            color=sty["ink"], ha="right", va="center")  # fmt: skip
                    ov.plot([NATIVE_RAIL] * 2, [bot + yc[ks[-1]] - 1.5, bot + yc[ks[0]] + 1.5],
                            color=sty["rail"], lw=1.1, solid_capstyle="butt")  # fmt: skip
                for k in ks:
                    ov.text(NATIVE_NAME_R, bot + yc[k], panels[k]["label"], fontsize=fs.T_LABEL,
                            color=fs.INK, ha="right", va="center")  # fmt: skip
            for j, (br, _) in enumerate(NATIVE_BRANCHES):
                x0 = p0 + j * (pw + gut)
                ax = fig.add_axes((x0 / w_mm, bot / h_mm, pw / w_mm, ph / h_mm))
                for s in ("top", "right", "left"):
                    ax.spines[s].set_visible(False)
                ax.spines["bottom"].set_color(fs.BAR_AXIS)
                ax.spines["bottom"].set_linewidth(0.6)
                ax.spines["bottom"].set_bounds(0, 100)
                ax.set_xlim(0, NATIVE_XMAX)
                ax.set_ylim(0, ph)
                ax.set_xticks(range(0, 101, 20))
                ax.set_yticks([])
                ax.tick_params(axis="x", length=2.2, width=0.6, color=fs.BAR_AXIS,
                               labelsize=fs.T_TICK, labelcolor=fs.TICK, pad=1.6,
                               labelbottom=m == NATIVE_METRICS[-1])  # fmt: skip
                ax.grid(axis="x", color=fs.BAR_GRID, lw=0.5, zorder=0)
                ax.set_axisbelow(True)
                ax.axvline(0, color=fs.BAR_ZERO, lw=0.6, zorder=2.6)
                ax.patch.set_alpha(0)
                for cell, ks in groups:
                    pred = panels[ks[0]]["values"][br, m]["pred"]
                    if (sty := styles[cell]) is None or not pred:
                        continue
                    est, lo, hi = (100 * pred[f] for f in ("est", "lo", "hi"))
                    ya = yc[ks[-1]] - bar_t / 2 - NATIVE_BLOCK_PAD
                    yb = yc[ks[0]] + bar_t / 2 + NATIVE_BLOCK_PAD
                    ax.add_patch(Rectangle((0, ya), est, yb - ya, fc=sty["pale"], ec="none",
                                           zorder=1.0))  # fmt: skip
                    ax.add_patch(Rectangle((lo, ya), hi - lo, yb - ya, fc=sty["band"],
                                           ec="none", zorder=1.1))  # fmt: skip
                    ax.plot([est, est], [ya, yb], color=sty["line"], lw=NATIVE_LINE_LW,
                            solid_capstyle="butt", zorder=1.3)  # fmt: skip
                inside, drawn = 0, 0
                for k, rec in enumerate(panels):
                    v = rec["values"][br, m]
                    if not rec["staged"]:  # where its bar would start
                        ax.annotate("not run", xy=(0, yc[k]), xytext=(4, 0),
                                    textcoords="offset points", ha="left", va="center",
                                    fontsize=fs.T_NOTE, color=fs.QUIET)  # fmt: skip
                    if not v["native"]:
                        continue
                    nat = NATIVE_INSIDE if v["inside"] else NATIVE_OUTSIDE
                    est, lo, hi = (100 * v["native"][f] for f in ("est", "lo", "hi"))
                    bar = ax.add_patch(Rectangle((0, yc[k] - bar_t / 2), est, bar_t,
                                                 fc=nat["fill"], ec=nat["edge"] or "none",
                                                 lw=NATIVE_EDGE_LW, zorder=2.0))  # fmt: skip
                    lines = ax.plot([lo, hi], [yc[k]] * 2, color=nat["ink"], lw=fs.IV_LW,
                                    solid_capstyle="butt", zorder=4)  # fmt: skip
                    for e in (lo, hi):
                        lines += ax.plot([e, e], [yc[k] - NATIVE_CAP / 2, yc[k] + NATIVE_CAP / 2],
                                         color=nat["ink"], lw=fs.IV_LW, solid_capstyle="butt",
                                         zorder=4)  # fmt: skip
                    if nat["inner"]:  # where the interval runs over the dark bar, it turns light
                        for ln in lines:
                            (light,) = ax.plot(*ln.get_data(), color=nat["inner"], lw=fs.IV_LW,
                                               solid_capstyle="butt", zorder=4.1)  # fmt: skip
                            light.set_clip_path(bar)
                    if v["pred"] and v["inside"] is not None:
                        inside += bool(v["inside"])
                        drawn += 1
                if drawn:
                    ov.text(x0 + pw * 100 / NATIVE_XMAX, top + 1.8, f"{inside} of {drawn} inside",
                            fontsize=fs.T_NOTE, color=fs.QUIET, ha="right",
                            va="center")  # fmt: skip
        for j, (br, sym) in enumerate(NATIVE_BRANCHES):  # column headers: the shift
            ov.text(p0 + j * (pw + gut) + pw * 50 / NATIVE_XMAX, h_mm - 8.4,
                    f"{NATIVE_SHIFT_NAMES[br]} {sym}", fontsize=fs.T_TITLE, color=fs.INK,
                    ha="center", va="center")  # fmt: skip
        ov.text(p0 + 1.5 * pw + gut, 1.7, "accuracy (%)", fontsize=fs.T_LABEL, color=fs.INK,
                ha="center", va="center")  # fmt: skip
        fig.canvas.draw()
        _native_key(ov, [s for s in styles.values() if s is not None], w_mm, h_mm - 2.9)
        return fs.save_pdf(fig, out / "fig_native_systems.pdf")


def _native_key(ov: Any, cells: Sequence[dict[str, Any]], w_mm: float, y: float,
                sep: float = 4.8) -> None:  # fmt: skip
    """The key row, centred at the top: each cell drawn (its pale bar ending at its estimate line
    inside its band, then its name), then the native system inside and outside the interval."""
    from matplotlib.patches import Rectangle

    def cell_glyph(sty: dict[str, Any]) -> fs.Glyph:
        def draw(ov: Any, x: float, y: float) -> None:
            h = 2.3
            ov.add_patch(Rectangle((x, y - h / 2), 2.0, h, fc=sty["pale"], ec="none"))
            ov.add_patch(Rectangle((x + 1.35, y - h / 2), 1.65, h, fc=sty["band"], ec="none"))
            ov.plot([x + 2.0] * 2, [y - h / 2, y + h / 2], color=sty["line"],
                    lw=NATIVE_LINE_LW, solid_capstyle="butt")  # fmt: skip

        return draw

    def native_glyph(nat: dict[str, Any]) -> fs.Glyph:
        def draw(ov: Any, x: float, y: float) -> None:
            ov.add_patch(Rectangle((x, y - NATIVE_BAR_T / 2), 3.0, NATIVE_BAR_T, fc=nat["fill"],
                                   ec=nat["edge"] or "none", lw=NATIVE_EDGE_LW))  # fmt: skip

        return draw

    key, small = fs.T_KEY, NATIVE_SC * fs.T_KEY
    entries = [(cell_glyph(sty), [(s, key if full else small) for s, full in sty["segments"]])
               for sty in cells]  # fmt: skip
    entries += [(native_glyph(NATIVE_INSIDE), [("native system, inside the cell’s 95% interval",
                                                key)]),
                (native_glyph(NATIVE_OUTSIDE), [("outside it", key)])]  # fmt: skip
    widths = [sum(fs.advance(s, size) for s, size in segs) for _, segs in entries]
    lead = 3.0 + 1.1
    total = sum(widths) + lead * len(entries) + sep * (len(entries) - 1)
    x = (w_mm - total) / 2
    for (glyph, segs), w in zip(entries, widths, strict=True):
        glyph(ov, x, y)
        _segments_mm(ov, x + lead, y, segs, fs.INK)
        x += lead + w + sep


def _analysis(root: Path, sources: dict[str, str], group: str, name: str) -> dict[str, Any] | None:
    p = root / sources.get(group, "__missing__") / "analysis" / name
    return dict(json.loads(p.read_text())) if p.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("paper/figures"))
    ap.add_argument("--only", nargs="*", default=[], help="figure names (fig_calibration, ...)")
    ap.add_argument(
        "--tables",
        type=Path,
        default=Path("paper/generated"),
        help="the directory of tab_diagnostic_tests.tex",
    )
    a = ap.parse_args()
    sources = json.loads((a.root / "results" / "sources.json").read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = [
        (fig_calibration, "e2", "e2_metrics.json"),
        (fig_zero_cost, "e3", "e3_metrics.json"),
    ]
    for fn, group, name in jobs:
        data = _analysis(a.root, sources, group, name)
        if data is not None and (not a.only or fn.__name__ in a.only):
            print(fn(data, a.out))
    if not a.only or "fig_native_systems" in a.only:  # draws the systems that are not staged, too
        print(fig_native_systems(native_results(load_stage4(a.root, sources)), a.out))
    if not a.only or "fig_classsplit" in a.only:  # contrasts by item class
        from memshift.analysis import ex_e1_paper

        print(ex_e1_paper.fig_classsplit(ex_e1_paper.load(a.root, sources), a.out))
    if not a.only or "fig_account" in a.only:  # the account across writers
        from memshift.analysis import ex_review_paper

        print(ex_review_paper.fig_account(ex_review_paper.load(a.root, sources), a.out))
    from memshift.analysis import round3_figures  # round 3 and the diagnostic tests

    wanted = [n for n in round3_figures.FIGURES if not a.only or n in a.only]
    diagnostic = not a.only or "tab_diagnostic_tests" in a.only
    if wanted or diagnostic:
        inp = round3_figures.load(a.root, sources)
        for name in wanted:
            print(round3_figures.FIGURES[name](inp, a.out))
        if diagnostic:
            a.tables.mkdir(parents=True, exist_ok=True)
            table = a.tables / "tab_diagnostic_tests.tex"
            table.write_text(
                round3_figures.tab_diagnostic_tests(round3_figures.diagnostic_tests(inp))
            )
            print(table)


if __name__ == "__main__":
    main()
