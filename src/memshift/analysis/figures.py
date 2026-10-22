"""Paper figures regenerated from the committed run analyses (figures come from runs/).

Sources are the analysis directories listed in results/sources.json, the same ones the
paper's number macros are generated from.

fig_concept        Figure 1 (`concept_figure`): one item and the five shifts, and the HoldMem
                   cells after a redefinition, detection by repair (ex_round4_detection, e1)
fig_calibration    E2: survival of the D=0, B=1 items after the switch per bin of writer agreement
                   on the calibration set, pooled over the three tiers, with the pooled fit, and
                   agreement from 80% magnified.
fig_zero_cost      E3: ContractValid before any shift by horizon; tokens per query as stacked
                   bars, with the cost ceiling and one full re-read beyond an axis break.
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


CALIBRATION_BINS = (0.0, 0.5, 0.8, 0.9, 0.95, 1.0)  # agreement bins of the calibration figure


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


CALIBRATION_PT = (486, 182.8)  # printed size (pt): full width
CAL_A = (13.0, 105.0)  # panel (a): left and right of the plot frame (mm)
CAL_B = (119.0, 165.0)  # panel (b), the magnified range: left and right of its frame (mm)
CAL_Y = (13.5, 57.5)  # bottom and top of both frames (mm)
CAL_TOP = 1.04  # survival at the top of both frames: points at 100% clear the frame
CAL_ZOOM = (0.80, 0.85)  # (b) magnifies agreement from 80% and survival from 85%: equal scales
CAL_STRIP = (0.9, 3.6, 0.45)  # count strip: gap under the x axis, height, white gap between cells
CAL_DOT_R = 1.0  # point radius (mm)
CAL_REF = "#8E99A8"  # the diagonal of perfect calibration
CAL_WEDGE = fs.tint(fs.SPINE, 0.018)  # the barely visible fill between the zoom rays
CAL_TICK, CAL_NOTE, CAL_SMALL = 7.0, 7.0, 6.5  # ticks and counts; notes; counts of narrow cells


class _CalPanel:
    """A plot frame of the calibration figure: agreement in [lo, 1] across x0..x1 (mm), survival
    in [ylo, CAL_TOP] across `CAL_Y`."""

    def __init__(self, x0: float, x1: float, lo: float, ylo: float) -> None:
        self.x0, self.x1, self.lo, self.ylo = x0, x1, lo, ylo

    def px(self, v: float) -> float:
        return self.x0 + (v - self.lo) / (1 - self.lo) * (self.x1 - self.x0)

    def py(self, v: float) -> float:
        return CAL_Y[0] + (v - self.ylo) / (CAL_TOP - self.ylo) * (CAL_Y[1] - CAL_Y[0])


def _cal_strip_tint(share: float, top: float) -> str:
    """A strip cell's colour: the role-blue mark mixed with white, its share of the mark growing
    with the square root of the bin's share of the items (relative to the largest bin)."""
    return fs.tint(fs.ROLE_BLUE["mark"], 0.12 + 0.88 * math.sqrt(share / top))


def fig_calibration(e2: dict[str, Any], out: Path) -> Path:
    """Survival of a switch against writer agreement for the D=0, pb=1 items pooled over the
    tiers. (a) The whole range: each bin's survival at its mean agreement as a point, the pooled
    logistic fit with its 90% band, the dashed diagonal of perfect calibration, and a card with
    the calibration slope and AUROC and their 90% intervals; (b) agreement from 80% magnified at
    equal scales, joined to its box in (a) by zoom rays, with each bin's survival printed and the
    share of the items it holds. Under each axis a strip gives the items per bin, its tint
    deepening with the bin's share of the items."""
    from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

    blue = fs.ROLE_BLUE
    y0, y1 = CAL_Y
    zlo, zlo_y = CAL_ZOOM
    s_gap, s_h, sep = CAL_STRIP
    bins = calibration_bins(e2)
    total = sum(b["n"] for b in bins)
    with fs.printed() as plt:
        cv = fs.MmCanvas(plt, CALIBRATION_PT[0] * fs.PT, CALIBRATION_PT[1] * fs.PT,
                         size=fs.T_LABEL)  # fmt: skip
        ax = cv.ax
        pa, pb = _CalPanel(*CAL_A, 0.0, 0.0), _CalPanel(*CAL_B, zlo, zlo_y)
        clip_a = Rectangle((pa.x0, y0), pa.x1 - pa.x0, y1 - y0, transform=ax.transData)
        clip_b = Rectangle((pb.x0, y0), pb.x1 - pb.x0, y1 - y0, transform=ax.transData)
        # frames, grids and ticks: (a) y on the left; (b) y on the right, clear of the zoom rays
        for v in (0.2, 0.4, 0.6, 0.8, 1.0):
            ax.plot([pa.x0, pa.x1], [pa.py(v)] * 2, color=fs.GRID, lw=0.5, zorder=0.5)
        ax.plot([pa.x0, pa.x0], [y0, y1], color=fs.SPINE, lw=0.6, zorder=1)
        ax.plot([pa.x0, pa.x1], [y0, y0], color=fs.SPINE, lw=0.6, zorder=1)
        for v in (0.90, 0.95, 1.0):
            ax.plot([pb.x0, pb.x1], [pb.py(v)] * 2, color=fs.GRID, lw=0.5, zorder=0.5)
        ax.add_patch(Rectangle((pb.x0, y0), pb.x1 - pb.x0, y1 - y0, fc="none", ec=fs.SPINE, lw=0.6,
                               zorder=1))  # fmt: skip
        s_top, s_bot = y0 - s_gap, y0 - s_gap - s_h
        for v in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            ax.plot([pa.x0 - 0.8, pa.x0], [pa.py(v)] * 2, color=fs.SPINE, lw=0.6, zorder=1)
            cv.text(pa.x0 - 1.4, pa.py(v), f"{100 * v:.0f}", size=CAL_TICK, color=fs.TICK,
                    ha="right")  # fmt: skip
            ax.plot([pa.px(v)] * 2, [s_bot, s_bot - 0.7], color=fs.SPINE, lw=0.6, zorder=1)
            cv.text(pa.px(v), s_bot - 2.2, f"{100 * v:.0f}", size=CAL_TICK, color=fs.TICK)
        for v in (0.85, 0.90, 0.95, 1.0):
            ax.plot([pb.x1, pb.x1 + 0.8], [pb.py(v)] * 2, color=fs.SPINE, lw=0.6, zorder=1)
            cv.text(pb.x1 + 1.4, pb.py(v), f"{100 * v:.0f}", size=CAL_TICK, color=fs.TICK,
                    ha="left")  # fmt: skip
        for v in (0.80, 0.85, 0.90, 0.95, 1.0):
            ax.plot([pb.px(v)] * 2, [s_bot, s_bot - 0.7], color=fs.SPINE, lw=0.6, zorder=1)
            cv.text(pb.px(v), s_bot - 2.2, f"{100 * v:.0f}", size=CAL_TICK, color=fs.TICK)
        for p in (pa, pb):
            cv.text((p.x0 + p.x1) / 2, 2.6, "writer agreement (%)", size=fs.T_LABEL)
        cv.text(3.2, (y0 + y1) / 2, "survival after the switch (%)", size=fs.T_LABEL, rotation=90)
        cv.label(pa.x0 - 0.2, cv.h - 0.9, "a", "Survival against writer agreement",
                 size=fs.T_TITLE)  # fmt: skip
        cv.label(pb.x0 - 0.2, cv.h - 0.9, "b", f"Agreement {100 * zlo:.0f}–100%, magnified",
                 size=fs.T_TITLE)  # fmt: skip
        # count strips: one cell per bin, its tint the bin's share of the items; a count that
        # does not fit its cell of (a) sits just above the axis, on a leader to the cell
        top_share = max((b["n"] for b in bins), default=1) / max(total, 1)
        narrow: list[tuple[float, str]] = []
        for p, x_lo in ((pa, 0.0), (pb, zlo)):
            for b in bins:
                if b["hi"] <= x_lo + 1e-9:
                    continue
                xa, xb = p.px(max(b["lo"], x_lo)) + sep / 2, p.px(b["hi"]) - sep / 2
                share = b["n"] / total
                deep = share / top_share > 0.5
                ax.add_patch(Rectangle((xa, s_bot), xb - xa, s_h,
                                       fc=_cal_strip_tint(share, top_share), ec="none",
                                       zorder=1))  # fmt: skip
                lab = f"{b['n']:,}"
                if fs.advance(lab, CAL_TICK, "bold" if deep else "normal") + 1.2 > xb - xa:
                    if p is pa:
                        narrow.append(((xa + xb) / 2, lab))
                else:
                    cv.text((xa + xb) / 2, (s_top + s_bot) / 2 - 0.05, lab, size=CAL_TICK,
                            color="white" if deep else blue["ink"],
                            weight="bold" if deep else "normal")  # fmt: skip
        right, y_lab = pa.x1, y0 + 2.1  # right to left from the frame's edge, 1 mm apart
        for cx, lab in reversed(narrow):
            w = fs.advance(lab, CAL_SMALL)
            cv.text(right, y_lab, lab, size=CAL_SMALL, color=blue["ink"], ha="right")
            ax.plot([right - w / 2, cx], [y_lab - 1.05, s_top], color=blue["ink"], lw=0.4,
                    zorder=5, solid_capstyle="butt")  # fmt: skip
            right -= w + 1.0
        if bins:
            cv.text(pa.x0 - 1.4, (s_top + s_bot) / 2, "items", size=CAL_NOTE, color=fs.MUTED,
                    ha="right", style="italic")  # fmt: skip
        # the zoom box in (a), and rays from its right corners to the corners of (b)
        zx0, zx1, zy0, zy1 = pa.px(zlo), pa.px(1.0), pa.py(zlo_y), pa.py(CAL_TOP)
        ax.add_patch(Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, fc="none", ec=fs.SPINE, lw=0.6,
                               zorder=4.5))  # fmt: skip
        ax.add_patch(Polygon([(zx1, zy1), (pb.x0, y1), (pb.x0, y0), (zx1, zy0)], closed=True,
                             fc=CAL_WEDGE, ec="none", zorder=0.2))  # fmt: skip
        for (xa, ya), (xb, yb) in (((zx1, zy1), (pb.x0, y1)), ((zx1, zy0), (pb.x0, y0))):
            ax.plot([xa, xb], [ya, yb], color=fs.SPINE, lw=0.5, zorder=1.1, solid_capstyle="butt")
        # perfect calibration in both panels, named once in (a)
        for p, clip in ((pa, clip_a), (pb, clip_b)):
            (ln,) = ax.plot([p.px(p.lo), p.px(1)], [p.py(p.lo), p.py(1)], color=CAL_REF,
                            lw=0.75, ls=(0, (3.2, 2.0)), zorder=1.5)  # fmt: skip
            ln.set_clip_path(clip)
        ang = math.degrees(math.atan2(pa.py(1) - pa.py(0), pa.px(1) - pa.px(0)))
        cv.text(pa.px(0.11), pa.py(0.11) - 1.0, "perfect calibration", size=CAL_NOTE,
                color=fs.MUTED, rotation=ang, rotation_mode="anchor", ha="left", va="top",
                style="italic")  # fmt: skip
        # the pooled logistic fit and its 90% band in both panels, named where it starts
        curve = e2.get("explore_pooled_curve") or {}
        if curve.get("est") and not any(math.isnan(v) for v in curve["est"]):
            g, est, lo90, hi90 = curve["grid"], curve["est"], curve["lo90"], curve["hi90"]
            for p, clip in ((pa, clip_a), (pb, clip_b)):
                keep = [i for i, x in enumerate(g) if x >= p.lo - 1e-9]
                band = [(p.px(g[i]), p.py(hi90[i])) for i in keep]
                band += [(p.px(g[i]), p.py(lo90[i])) for i in reversed(keep)]
                poly = Polygon(band, closed=True, fc=blue["soft"], ec="none", alpha=0.8, zorder=2)
                ax.add_patch(poly)
                poly.set_clip_path(clip)
                (ln,) = ax.plot([p.px(g[i]) for i in keep], [p.py(est[i]) for i in keep],
                                color=blue["ink"], lw=1.2, zorder=3,
                                solid_capstyle="round")  # fmt: skip
                ln.set_clip_path(clip)
            cv.text(pa.px(g[0]) - 1.2, pa.py(est[0]) + 1.3, "logistic fit", size=7.5,
                    color=blue["ink"], ha="right")  # fmt: skip
            cv.text(pa.px(g[0]) - 1.2, pa.py(est[0]) - 1.6, "90% band", size=CAL_NOTE,
                    color=blue["ink"], ha="right", style="italic")  # fmt: skip
        # one point per bin at its mean agreement, in both panels
        for p, x_lo in ((pa, 0.0), (pb, zlo)):
            for b in bins:
                if b["lo"] >= x_lo - 1e-9:
                    ax.add_patch(Circle((p.px(b["agreement"]), p.py(b["survival"])), CAL_DOT_R,
                                        fc=blue["mark"], ec="white", lw=0.6,
                                        zorder=6))  # fmt: skip
        # (b): each bin's survival at its point, above it near 100%, else below it; and the
        # share of the items in the magnified range
        top = [b for b in bins if b["lo"] >= zlo - 1e-9]
        for b in top:
            x, y = pb.px(b["agreement"]), pb.py(b["survival"])
            lab = f"{100 * b['survival']:.1f}" if b["survival"] < 1 else "100"
            ty = y + CAL_DOT_R + 1.5 if b["survival"] >= 0.99 else y - CAL_DOT_R - 1.6
            if x + fs.advance(lab, CAL_NOTE) / 2 > pb.x1 - 0.5:  # inside the frame
                cv.text(pb.x1 - 0.5, ty, lab, size=CAL_NOTE, color=blue["ink"], ha="right")
            else:
                cv.text(x, ty, lab, size=CAL_NOTE, color=blue["ink"])
        if top:
            n_top = sum(b["n"] for b in top)
            cv.text(pb.x1 - 1.4, y0 + 5.0, f"{100 * n_top / total:.1f}% of the items", size=7.5,
                    color=blue["ink"], ha="right")  # fmt: skip
            cv.text(pb.x1 - 1.4, y0 + 2.0, f"{n_top:,} of {total:,}", size=CAL_NOTE,
                    color=fs.MUTED, ha="right", style="italic")  # fmt: skip
        # the pooled calibration slope and AUROC with their 90% intervals, on a card in the
        # empty lower right of (a)
        p3 = e2.get("path3") or {}
        sl, au = p3.get("pooled_slope") or {}, p3.get("pooled_auroc") or {}
        if sl and au:
            rows = [("calibration slope", sl), ("AUROC", au)]
            lab_w = max(fs.advance(r[0], 7.5) for r in rows)
            val_w = max(fs.advance(f"{r[1]['est']:.2f}", 8.5, "bold") for r in rows)
            iv_w = max(fs.advance(f"[{r[1]['lo90']:.2f}, {r[1]['hi90']:.2f}]", 7.5) for r in rows)
            pad, gap, row_h = 1.8, 1.6, 3.5
            cw = pad + lab_w + gap + val_w + gap + iv_w + pad
            ch = pad + 2.6 + 2 * row_h + pad - 0.6
            cx0, cy0 = pa.x1 - 1.8 - cw, y0 + 4.7
            ax.add_patch(FancyBboxPatch((cx0, cy0), cw, ch,
                                        boxstyle="round,pad=0,rounding_size=1.4", fc=blue["fill"],
                                        ec=blue["stroke"], lw=0.6, zorder=7))  # fmt: skip
            xl, xv = cx0 + pad, cx0 + pad + lab_w + gap
            xi = xv + val_w + gap
            yh = cy0 + ch - pad - 0.9
            cv.text(xi, yh, "90% interval", size=CAL_NOTE, color=fs.MUTED, ha="left",
                    style="italic")  # fmt: skip
            for k, (name, e) in enumerate(rows):
                y = yh - 3.2 - k * row_h
                cv.text(xl, y, name, size=7.5, color=fs.INK, ha="left")
                cv.text(xv, y, f"{e['est']:.2f}", size=8.5, color=blue["ink"], weight="bold",
                        ha="left")  # fmt: skip
                cv.text(xi, y, f"[{e['lo90']:.2f}, {e['hi90']:.2f}]", size=7.5, color=fs.TICK,
                        ha="left")  # fmt: skip
        return fs.save_pdf(cv.fig, out / "fig_calibration.pdf")


ZERO_PT = (486, 151.2)  # printed size of the zero-system figure (pt): full width
HOLD = "holdmem_D1B1_source"  # the hero of the zero-system figure
ZERO_SYSTEMS = {  # system -> (label, colour, dash, marker, light part of its cost bar)
    HOLD: ("HOLDMEM(1,1)", fs.ROLE_BLUE["mark"], "-", "o", "#B3CBEA"),
    "oracle": ("oracle context", "#5E6875", (0, (1.0, 1.4)), "D", None),
    "zero": ("no-store BM25", "#262B33", "-", "s", "#B9BEC6"),
    "rag": ("plain RAG", "#8E99A8", (0, (3.4, 1.8)), "^", "#D5DAE0"),
}
ZERO_COST_ROWS = (HOLD, "zero", "rag")  # the cost bars, in the order of the line labels of (a)
ZERO_NAME_INK = {"rag": "#6E7987"}  # a name darker than its pale line or bar
ZERO_NAME_PT, ZERO_HERO_PT = 7.8, 8.4  # system names; the hero's small capitals stay >= 6.5 pt
ZERO_Y = (9.6, 46.4)  # bottom and top of both plots (mm)
ZERO_A = (11.6, 54.0)  # panel (a): left and right of the plot (mm)
# panel (a): the horizons in evenly spaced slots 0, 1, ...; room before the first slot and after
# the last, and the bracket of the gap that far past the last slot (in slots)
ZERO_SLOT_LO, ZERO_SLOT_HI, ZERO_SLOT_BRACKET = 0.30, 0.42, 0.19
ZERO_GAP_AT = 0.735  # accuracy at which the gap is printed, inside its bracket
ZERO_B0 = 80.6  # panel (b): where its title and its row names start (mm)
ZERO_BAR_H = 6.4  # bar height (mm)
ZERO_STEP = 5000  # tick step of the bars' linear axis (tokens per query)
ZERO_SEG = (0.62e6, 2.62e6)  # the far segment beyond the axis break (tokens per query)
ZERO_SEG_W, ZERO_BREAK_W = 18.6, 3.4  # widths of the far segment and of the break (mm)
ZERO_FAIL = "#ECEEF2"  # the region that fails the cost ceiling


def zero_accuracy(e3: dict[str, Any]) -> dict[str, list[tuple[int, dict[str, float]]]]:
    """Per system of `ZERO_SYSTEMS` that reports it, ContractValid before any shift as (slot of
    the horizon, estimate with its 95% interval), horizons ascending."""
    horizons = e3.get("horizons") or {}
    hs = sorted(int(h) for h in horizons)
    out: dict[str, list[tuple[int, dict[str, float]]]] = {}
    for s in ZERO_SYSTEMS:
        pts = [(i, e) for i, h in enumerate(hs)
               if (e := (horizons[str(h)].get(s) or {}).get("contract_valid"))]  # fmt: skip
        if pts:
            out[s] = pts
    return out


def zero_costs(e3: dict[str, Any]) -> dict[str, Any]:
    """Tokens per query at the longest horizon, per system of `ZERO_COST_ROWS` with a cost:
    serving, and ingestion and maintenance amortised over the queries (queries = serve tokens /
    serve tokens per query), and their total; one full re-read and the cost ceiling."""
    from memshift.analysis.e3 import COST_CEILING

    horizons = e3.get("horizons") or {}
    if not horizons:
        return {"systems": {}, "reread": None, "ceiling": None}
    last = horizons[str(max(int(h) for h in horizons))]
    systems: dict[str, dict[str, float]] = {}
    for s in ZERO_COST_ROWS:
        c = last.get(s) or {}
        if "cost_mean" not in c:
            continue
        serve = c["serve_tokens_per_query"]
        queries = c["cost_mean"]["serve_tokens"] / max(serve, 1e-9)
        parts = {p: c["cost_mean"][f"{p}_tokens"] / queries
                 for p in ("serve", "ingest", "maintain")}  # fmt: skip
        systems[s] = {"serve": serve, "ingest": parts["ingest"], "maintain": parts["maintain"],
                      "total": sum(parts.values())}  # fmt: skip
    reread = next((last[s]["full_reread_tokens_per_query"] for s in systems), None)
    return {"systems": systems, "reread": reread,
            "ceiling": None if reread is None else COST_CEILING * reread}  # fmt: skip


def _system_segments(label: str) -> list[tuple[str, float]]:
    """A system's name as (text, scale): `HOLDMEM(...)` in small capitals, as \\textsc{HoldMem}."""
    if label.startswith("HOLDMEM"):
        return [("H", 1.0), ("OLD", 0.78), ("M", 1.0), ("EM", 0.78), (label[7:], 1.0)]
    return [(label, 1.0)]


def _system_name_width(label: str, size: float, weight: str = "normal") -> float:
    """The printed width (mm) of a system's name as `_system_name` sets it."""
    return sum(fs.advance(s, size * k, weight) for s, k in _system_segments(label))


def _system_name(cv: fs.MmCanvas, x: float, y: float, label: str, size: float, color: str, *,
                 weight: str = "normal") -> float:  # fmt: skip
    """A system's name with (x, y) its left end and vertical centre (mm); `HOLDMEM(...)` is set
    in small capitals, as \\textsc{HoldMem} in the paper. Returns the right end."""
    segs = _system_segments(label)
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


def _zero_name(cv: fs.MmCanvas, x: float, y: float, s: str) -> None:
    """A system's name with its left end at x: the hero bold, larger and in its blue ink."""
    label, colour, *_ = ZERO_SYSTEMS[s]
    hero = s == HOLD
    _system_name(cv, x, y, label, ZERO_HERO_PT if hero else ZERO_NAME_PT,
                 fs.ROLE_BLUE["ink"] if hero else ZERO_NAME_INK.get(s, colour),
                 weight="bold" if hero else "normal")  # fmt: skip


def _zero_name_width(s: str) -> float:
    hero = s == HOLD
    return _system_name_width(ZERO_SYSTEMS[s][0], ZERO_HERO_PT if hero else ZERO_NAME_PT,
                              "bold" if hero else "normal")  # fmt: skip


def _zero_accuracy(cv: fs.MmCanvas, e3: dict[str, Any]) -> None:
    """Panel (a): ContractValid before any shift by horizon (evenly spaced), 95% bands, each line
    named at its right end, and the paired gap of HoldMem(1,1) over no-store BM25 at the longest
    horizon with its 95% interval beside a bracket."""
    from matplotlib.patches import Polygon

    ax, blue = cv.ax, fs.ROLE_BLUE
    x0, x1 = ZERO_A
    y0, y1 = ZERO_Y
    cv.label(0.4, cv.h - 0.5, "a", "Accuracy before any shift", size=fs.T_TITLE)
    acc = zero_accuracy(e3)
    if not acc:
        return
    hs = sorted(int(h) for h in e3["horizons"])
    u_lo, u_hi = -ZERO_SLOT_LO, len(hs) - 1 + ZERO_SLOT_HI

    def px(u: float) -> float:
        return x0 + (u - u_lo) / (u_hi - u_lo) * (x1 - x0)

    lows = [e["lo"] for pts in acc.values() for _, e in pts]
    highs = [e["hi"] for pts in acc.values() for _, e in pts]
    lo, hi = math.floor(20 * min(lows)) / 20, math.ceil(20 * max(highs)) / 20  # 5-point steps

    def py(v: float) -> float:
        return y0 + (v - lo) / (hi - lo) * (y1 - y0)

    v = math.ceil(lo * 10) / 10
    while v <= hi + 1e-9:
        ax.plot([x0, x1], [py(v)] * 2, color=fs.BAR_GRID, lw=0.5, zorder=0.5)
        ax.plot([x0 - 0.8, x0], [py(v)] * 2, color=fs.BAR_AXIS, lw=0.6, zorder=1)
        cv.text(x0 - 1.5, py(v), f"{100 * v:.0f}", size=fs.T_TICK, color=fs.TICK, ha="right")
        v += 0.1
    ax.plot([x0, x0], [y0, y1], color=fs.BAR_AXIS, lw=0.6, zorder=1)
    ax.plot([x0, x1], [y0, y0], color=fs.BAR_AXIS, lw=0.6, zorder=1)
    for i, h in enumerate(hs):
        ax.plot([px(i)] * 2, [y0, y0 - 0.8], color=fs.BAR_AXIS, lw=0.6, zorder=1)
        cv.text(px(i), y0 - 2.6, str(h), size=fs.T_TICK, color=fs.TICK)
    cv.text((x0 + x1) / 2, 2.4, "horizon (months)", size=fs.T_LABEL)
    cv.text(2.9, (y0 + y1) / 2, "accuracy (%)", size=fs.T_LABEL, rotation=90)
    # bands, lines and markers: the baselines first, the hero on top
    ends: dict[str, float] = {}
    for s in ("oracle", "zero", "rag", HOLD):
        if s not in acc:
            continue
        _, colour, dash, mk, _ = ZERO_SYSTEMS[s]
        pts, hero = acc[s], s == HOLD
        band = [(px(i), py(e["hi"])) for i, e in pts]
        band += [(px(i), py(e["lo"])) for i, e in reversed(pts)]
        ax.add_patch(Polygon(band, closed=True, fc=blue["soft"] if hero else colour,
                             alpha=0.85 if hero else 0.13, ec="none",
                             zorder=2.2 if hero else 1.6))  # fmt: skip
        ax.plot([px(i) for i, _ in pts], [py(e["est"]) for _, e in pts], color=colour,
                lw=1.9 if hero else 1.0, ls=dash, zorder=4 if hero else 3,
                solid_capstyle="round", dash_capstyle="butt")  # fmt: skip
        for i, e in pts:
            _system_marker(ax, px(i), py(e["est"]), mk, colour, 1.9 if hero else 1.45,
                           z=5 if hero else 4.5)  # fmt: skip
        ends[s] = pts[-1][1]["est"]
    names = [s for s in (HOLD, "oracle", "zero", "rag") if s in ends]
    for s, y in zip(names, _spread([py(ends[s]) for s in names], 3.3), strict=True):
        _zero_name(cv, x1 + 1.2, y, s)
    # the gap at the longest horizon: HoldMem(1,1) over no-store BM25 (paired, 95% interval)
    gap = e3["horizons"][str(hs[-1])].get("zero_minus_holdmem")
    hold, zero = ends.get(HOLD), ends.get("zero")
    if _ok(gap) and hold is not None and zero is not None:
        xb, yh, yz = px(len(hs) - 1 + ZERO_SLOT_BRACKET), py(hold), py(zero)
        arm = xb - px(len(hs) - 1) - 1.4
        cv.line([(xb - arm, yh), (xb, yh), (xb, yz), (xb - arm, yz)], color=blue["ink"], lw=1.0,
                radius=0.35)  # fmt: skip
        ym = py(min(max(ZERO_GAP_AT, min(hold, zero)), max(hold, zero)))
        cv.text(xb - 1.3, ym + 1.35, f"{-100 * gap['est']:.1f} pp", size=8.2, color=blue["ink"],
                weight="bold", ha="right")  # fmt: skip
        cv.text(xb - 1.3, ym - 1.75, f"[{-100 * gap['hi']:.1f}, {-100 * gap['lo']:.1f}]",
                size=fs.T_NOTE, color=blue["ink"], ha="right")  # fmt: skip


def _zero_cost(cv: fs.MmCanvas, e3: dict[str, Any]) -> None:
    """Panel (b): tokens per query at the longest horizon as stacked bars on a linear axis,
    serving (dark) and amortised ingestion and maintenance (light, the same hue), with the total
    above each bar's end and the ratio of HoldMem(1,1) to no-store BM25 as a bracket; beyond an
    axis break, the cost ceiling, one full re-read and the region that fails the ceiling."""
    from matplotlib.patches import Rectangle

    ax, blue = cv.ax, fs.ROLE_BLUE
    y0, y1 = ZERO_Y
    cv.label(ZERO_B0, cv.h - 0.5, "b", "Tokens per query at twelve months", size=fs.T_TITLE)
    cost = zero_costs(e3)
    rows = [s for s in ZERO_COST_ROWS if s in cost["systems"]]
    if not rows:
        return
    c = cost["systems"]
    widths = [_zero_name_width(s) for s in rows]
    b0 = ZERO_B0 + max(widths) + 1.8  # the bars' zero
    f1 = cv.w - 0.6
    f0 = f1 - ZERO_SEG_W
    l1 = f0 - ZERO_BREAK_W
    l_max = ZERO_STEP * math.ceil(float(max(c[s]["total"] for s in rows)) / ZERO_STEP) + 1000

    def px(v: float) -> float:
        return b0 + v / l_max * (l1 - b0)

    def qx(v: float) -> float:
        return f0 + (v - ZERO_SEG[0]) / (ZERO_SEG[1] - ZERO_SEG[0]) * (f1 - f0)

    # the linear axis: grid, zero line, ticks
    for v in range(0, l_max + 1, ZERO_STEP):
        if v:
            ax.plot([px(v)] * 2, [y0, y1], color=fs.BAR_GRID, lw=0.5, zorder=0.5)
        ax.plot([px(v)] * 2, [y0, y0 - 0.8], color=fs.BAR_AXIS, lw=0.6, zorder=1)
        cv.text(px(v), y0 - 2.6, "0" if not v else f"{v // 1000}k", size=fs.T_TICK,
                color=fs.TICK)  # fmt: skip
    ax.plot([b0, l1], [y0, y0], color=fs.BAR_AXIS, lw=0.6, zorder=1)
    ax.plot([b0, b0], [y0, y1], color=fs.BAR_ZERO, lw=0.6, zorder=2.6)
    cv.text((b0 + l1) / 2, 2.4, "tokens per query", size=fs.T_LABEL)
    # the break: the axis stops, two slanted strokes, then the far segment
    for dx in (-0.6, 0.6):
        xc = (l1 + f0) / 2 + dx
        ax.plot([xc - 0.6, xc + 0.6], [y0 - 1.0, y0 + 1.4], color=fs.TICK, lw=0.75, zorder=3,
                solid_capstyle="butt")  # fmt: skip
    # the far segment: the region that fails the cost ceiling, the ceiling and one full re-read
    xc, xr = qx(cost["ceiling"]), qx(cost["reread"])
    ax.plot([f0, f1], [y0, y0], color=fs.BAR_AXIS, lw=0.6, zorder=1)
    ax.add_patch(Rectangle((xc, y0), f1 - xc, y1 - y0, fc=ZERO_FAIL, ec="none", zorder=0.3))
    ax.plot([xc] * 2, [y0, y1], color=fs.MUTED, lw=0.8, ls=(0, (2.4, 1.5)), zorder=1.2)
    ax.plot([xr] * 2, [y0, y1], color=fs.INK, lw=0.85, zorder=1.2)
    for x, v in ((xc, cost["ceiling"]), (xr, cost["reread"])):
        ax.plot([x] * 2, [y0, y0 - 0.8], color=fs.BAR_AXIS, lw=0.6, zorder=1)
        cv.text(x, y0 - 2.6, f"{v / 1e6:.2f}M", size=fs.T_TICK, color=fs.TICK)
    cv.text(xc + 0.8, y1 - 0.5, "cost ceiling", size=fs.T_NOTE, color=fs.MUTED, rotation=90,
            ha="left", va="top", style="italic")  # fmt: skip
    cv.text(xr + 0.8, y0 + 0.8, "full re-read", size=fs.T_NOTE, color=fs.INK, rotation=90,
            ha="left", va="bottom")  # fmt: skip
    cv.text((xc + 3.3 + xr) / 2, (y0 + y1) / 2, "fails", size=fs.T_NOTE, color=fs.MUTED,
            style="italic")  # fmt: skip
    # the bars: serving (dark), then ingestion and maintenance (light, the same hue)
    pitch = (y1 - y0) / len(rows)
    ys = {s: y1 - pitch * (k + 0.5) for k, s in enumerate(rows)}
    for s, w in zip(rows, widths, strict=True):
        _, dark, _, _, light = ZERO_SYSTEMS[s]
        hero, y = s == HOLD, ys[s]
        xs, xt = px(c[s]["serve"]), px(c[s]["total"])
        ax.add_patch(Rectangle((b0, y - ZERO_BAR_H / 2), xs - b0, ZERO_BAR_H, fc=dark, ec="none",
                               zorder=3))  # fmt: skip
        if xt > xs:
            ax.add_patch(Rectangle((xs, y - ZERO_BAR_H / 2), xt - xs, ZERO_BAR_H, fc=light,
                                   ec="none", zorder=3))  # fmt: skip
        _zero_name(cv, b0 - 1.8 - w, y, s)
        cv.text(xt, y + ZERO_BAR_H / 2 + 0.5, f"{c[s]['total']:,.0f}", size=fs.T_TICK + 0.3,
                color=blue["ink"] if hero else fs.INK, weight="bold" if hero else "normal",
                ha="right", va="bottom")  # fmt: skip
    if HOLD not in c:
        return
    # the two parts, named on the hero's bar
    y = ys[HOLD]
    cv.text((b0 + px(c[HOLD]["serve"])) / 2, y, "serving", size=fs.T_NOTE, color="white")
    cv.text((px(c[HOLD]["serve"]) + px(c[HOLD]["total"])) / 2, y, "ingestion + maintenance",
            size=fs.T_NOTE, color=blue["ink"])  # fmt: skip
    if "zero" not in c:
        return
    # the ratio of the totals: HoldMem(1,1) against no-store BM25
    xh, xz = px(c[HOLD]["total"]), px(c["zero"]["total"])
    yh, yz = ys[HOLD], ys["zero"]
    xb = xh + 1.7
    cv.line([(xh + 0.35, yh), (xb, yh), (xb, yz), (xb - 1.0, yz)], color=blue["ink"], lw=1.0,
            radius=0.35)  # fmt: skip
    ax.plot([xz + 0.6, xb - 1.0], [yz, yz], color=blue["ink"], lw=0.6, ls=(0, (1.0, 1.3)),
            zorder=7, dash_capstyle="round")  # fmt: skip
    cv.text(xb - 1.2, (yh + yz) / 2 - 0.1, f"{c[HOLD]['total'] / c['zero']['total']:.2f}×",
            size=8.6, color=blue["ink"], weight="bold", ha="right")  # fmt: skip


def fig_zero_cost(e3: dict[str, Any], out: Path) -> Path:
    """(a) ContractValid before any shift by horizon with 95% bands, each line named at its end,
    and the gap of HoldMem(1,1) over no-store BM25 at the longest horizon; (b) tokens per query
    at the longest horizon as stacked bars, serving and amortised ingestion and maintenance, with
    the cost ratio of HoldMem(1,1) to no-store BM25, and the cost ceiling and one full re-read
    beyond an axis break."""
    with fs.printed() as plt:
        cv = fs.MmCanvas(plt, ZERO_PT[0] * fs.PT, ZERO_PT[1] * fs.PT, size=fs.T_LABEL)
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
    if not a.only or "fig_concept" in a.only:  # Figure 1
        from memshift.analysis import concept_figure

        print(concept_figure.fig_concept(concept_figure.load(a.root, sources), a.out))
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
