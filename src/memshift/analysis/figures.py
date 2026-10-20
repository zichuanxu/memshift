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


ZERO_H = 2.1  # height of the zero-system figure (in), full width
CALIBRATION_H = 2.7  # height of the calibration figure (in), `figstyle.NARROW` wide
NOTE_PT = 8  # direct labels and notes inside a figure (pt)
CLEAR = {"facecolor": "white", "edgecolor": "none", "pad": 0.4}  # no gridline through a note


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


def _height_pt(ax: Any) -> float:
    """The height of an axes in points, after the layout of its figure."""
    ax.figure.draw_without_rendering()
    return float(ax.get_window_extent().height * 72 / ax.figure.dpi)


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


def fig_calibration(e2: dict[str, Any], out: Path) -> Path:
    """Survival of a switch against writer agreement for the D=0, B=1 items pooled over the
    tiers: each agreement bin's survival at its mean agreement (area = items, the number beside
    it), the pooled logistic fit with its 90% band, the diagonal, and the pooled slope and AUROC
    with their 90% intervals."""
    from matplotlib.transforms import ScaledTranslation

    plt = fs.plt()
    fig, ax = plt.subplots(figsize=(fs.NARROW, CALIBRATION_H))
    ax.plot([0, 1], [0, 1], color=fs.GREY, lw=0.8, ls=(0, (3, 2)), zorder=1)
    below = ScaledTranslation(1.5 / 72, -1.5 / 72, fig.dpi_scale_trans)
    ax.text(0.06, 0.06, "perfect calibration", transform=ax.transData + below, rotation=45,
            transform_rotates_text=True, rotation_mode="anchor", ha="left", va="top",
            fontsize=NOTE_PT, color=fs.GREY)  # fmt: skip
    curve = e2.get("explore_pooled_curve") or {}
    if curve.get("est") and not any(math.isnan(v) for v in curve["est"]):
        ax.fill_between(curve["grid"], curve["lo90"], curve["hi90"], color=fs.LIGHT, lw=0,
                        zorder=0.5)  # fmt: skip
        ax.plot(curve["grid"], curve["est"], color=fs.GREY, lw=1.0, zorder=2)
        # the fit's name left of where it starts
        ax.annotate("logistic fit", (curve["grid"][0], curve["est"][0]), xytext=(-3, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=NOTE_PT,
                    color=fs.GREY)  # fmt: skip
    for b in calibration_bins(e2):
        area = AREA_PER_ITEM * b["n"]
        ax.scatter([b["agreement"]], [b["survival"]], s=[area], color=fs.BLUE,
                   edgecolors=fs.darker(fs.BLUE), linewidths=0.5, zorder=3,
                   clip_on=False)  # fmt: skip
        # the number of items on the side of the marker away from the diagonal
        up = b["survival"] >= b["agreement"]
        gap = math.sqrt(area / math.pi) + 2.0
        ax.annotate(f"{b['n']:,}".replace(",", "{,}"), (b["agreement"], b["survival"]),
                    xytext=(0, gap if up else -gap), textcoords="offset points", ha="center",
                    va="bottom" if up else "top", fontsize=NOTE_PT, color=fs.BLUE,
                    annotation_clip=False)  # fmt: skip
    p3 = e2.get("path3") or {}
    sl, au = p3.get("pooled_slope") or {}, p3.get("pooled_auroc") or {}
    if sl and au:
        ax.text(0.97, 0.05,
                f"slope {sl['est']:.2f} [{sl['lo90']:.2f}, {sl['hi90']:.2f}]\n"
                f"AUROC {au['est']:.2f} [{au['lo90']:.2f}, {au['hi90']:.2f}]",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=NOTE_PT,
                linespacing=1.3)  # fmt: skip
    fs.pct_axis(ax, "x", 0.0, 1.0)
    fs.pct_axis(ax, "y", 0.0, 1.0)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True, axis="both")
    ax.set_xlabel(r"writer agreement (\%)")
    ax.set_ylabel(r"survival after the switch (\%)")
    path = out / "fig_calibration.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_zero_cost(e3: dict[str, Any], out: Path) -> Path:
    """(a) ContractValid before any shift by horizon with 95% bands, each line named at its end;
    (b) tokens per query at the longest horizon as a dot plot, serving (filled) and serving with
    amortised ingestion and maintenance (hollow), against one full re-read and the cost ceiling."""
    from matplotlib.ticker import LogLocator, NullFormatter, NullLocator

    from memshift.analysis.e3 import COST_CEILING

    plt = fs.plt()
    hs = sorted(int(h) for h in e3["horizons"])
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(fs.TEXT, ZERO_H), gridspec_kw={"width_ratios": [1.4, 1]}
    )
    order = ("holdmem_D1B1_source", "oracle", "zero", "rag")
    ends: list[tuple[str, str, float]] = []  # (label, colour, last estimate) of every line
    lows, highs = [], []
    for s in order:
        label, colour, ls, marker = fs.SYSTEMS[s]
        pts = [
            (h, e3["horizons"][str(h)][s]["contract_valid"])
            for h in hs
            if s in e3["horizons"][str(h)]
        ]
        if not pts:
            continue
        x = [h for h, _ in pts]
        ax1.fill_between(
            x, [e["lo"] for _, e in pts], [e["hi"] for _, e in pts], color=colour, alpha=0.15, lw=0
        )
        ax1.plot(x, [e["est"] for _, e in pts], ls=ls, color=colour, marker=marker, ms=3.6)
        ends.append((label, colour, pts[-1][1]["est"]))
        lows += [e["lo"] for _, e in pts]
        highs += [e["hi"] for _, e in pts]
    ax1.set_xscale("log")
    ax1.set_xticks(hs, [str(h) for h in hs])
    ax1.xaxis.set_minor_locator(NullLocator())
    ax1.set_xlim(hs[0] / 1.2, hs[-1] * 1.2)
    lo, hi = math.floor(20 * min(lows)) / 20, math.ceil(20 * max(highs)) / 20  # 5-point steps
    fs.pct_axis(ax1, "y", lo, hi)
    ax1.set_xlabel("horizon (months, log scale)")
    ax1.set_ylabel("accuracy (\\%)")
    fs.panel_label(ax1, "a")
    last = e3["horizons"][str(hs[-1])]
    rows = [s for s in order if "cost_mean" in last.get(s, {})]
    reread = last[rows[0]]["full_reread_tokens_per_query"]
    ceiling = COST_CEILING * reread
    serve_lo = min(last[s]["serve_tokens_per_query"] for s in rows)
    x_lo = serve_lo / 1.8
    x_hi = 10 ** math.ceil(math.log10(reread * 1.5))
    ax2.axvspan(ceiling, x_hi, color=fs.LIGHT, alpha=0.6, lw=0, zorder=0)
    ax2.axvline(reread, color=fs.BLACK, lw=0.8, zorder=1)
    for k, s in enumerate(rows):  # top to bottom in the order of the lines of (a)
        _, colour, _, marker = fs.SYSTEMS[s]  # colour and marker as in (a)
        y = len(rows) - 1 - k
        c = last[s]
        serve = c["serve_tokens_per_query"]
        queries = c["cost_mean"]["serve_tokens"] / max(serve, 1e-9)
        parts = ("ingest_tokens", "maintain_tokens", "serve_tokens")
        total = sum(c["cost_mean"][p] for p in parts) / queries
        ax2.plot([serve, total], [y, y], color=colour, lw=0.9, zorder=2)
        ax2.plot([total], [y], marker=marker, ms=4.8, mfc="white", mec=colour, mew=1.1, zorder=3)
        ax2.plot([serve], [y], marker=marker, ms=4.8, color=colour, mec=colour, zorder=4)
    ax2.set_xscale("log")
    ax2.set_xlim(x_lo, x_hi)
    ax2.xaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax2.xaxis.set_minor_locator(LogLocator(base=10, subs=tuple(range(2, 10)), numticks=12))
    ax2.xaxis.set_minor_formatter(NullFormatter())
    ax2.set_yticks(range(len(rows)), [fs.SYSTEMS[s][0] for s in reversed(rows)])
    ax2.tick_params(axis="y", length=0)
    ax2.set_ylim(-0.6, len(rows) - 0.4)
    ax2.grid(False, axis="y")
    ax2.grid(True, axis="x", which="major")
    ax2.set_xlabel("tokens per query (log scale)")
    # the region's name above its top edge, the line's name along it
    ax2.annotate("fails the cost ceiling", xy=(1, 1), xycoords="axes fraction", xytext=(0, 3),
                 textcoords="offset points", ha="right", va="bottom", fontsize=NOTE_PT,
                 color=fs.GREY)  # fmt: skip
    ax2.annotate("full re-read", xy=(reread, 0), xycoords=("data", "axes fraction"),
                 xytext=(2.5, 3), textcoords="offset points", rotation=90, ha="left",
                 va="bottom", fontsize=NOTE_PT)  # fmt: skip
    fs.panel_label(ax2, "b")
    # the name of each line of (a) at its right end, outside the axes, kept apart vertically
    gap = (hi - lo) * (NOTE_PT + 1.5) / _height_pt(ax1)
    for (label, colour, _), at in zip(ends, _spread([e for *_, e in ends], gap), strict=True):
        ax1.annotate(label, xy=(1, at), xycoords=("axes fraction", "data"), xytext=(3, 0),
                     textcoords="offset points", ha="left", va="center", fontsize=NOTE_PT,
                     color=colour)  # fmt: skip
    path = out / "fig_zero_cost.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


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


NATIVE_H = 3.2  # height of the native-systems figure (in), full width
NATIVE_BAR = 0.34  # thickness of one bar (one system per unit of the system axis)
NATIVE_ERROR = {"fmt": "none", "ecolor": fs.BLACK, "elinewidth": 0.6, "capsize": 1.6,
             "capthick": 0.6}  # fmt: skip


def _cell_style(cell: str | None) -> tuple[str, str]:
    """(label, colour) of a HoldMem cell id such as `holdmem_D0B1_cited`: the colour of its binding
    arm (`figstyle.ARMS`), grey for an unknown cell."""
    if not cell or not cell.startswith("holdmem_D") or len(cell) < 12:
        return (cell or "", fs.GREY)
    d, b = cell[9], cell[11]
    arm = "none" if b == "0" else ("cited" if cell.endswith("cited") else "source")
    where = {"cited": ", citation key", "source": ", source key", "none": ""}[arm]
    return (rf"\textsc{{HoldMem}}({d},{b}){where}", fs.ARMS[arm][1])


def _native_bar(ax: Any, y: float, e: dict[str, Any], face: str, edge: str) -> None:
    """One horizontal bar at `y` from 0 to the estimate, its 95% interval as a thin dark error
    bar; on a dark bar the part of the error bar inside it is white."""
    from matplotlib.colors import to_rgb

    (bar,) = ax.barh([y], [e["est"]], height=NATIVE_BAR, color=face, edgecolor=edge,
                     lw=fs.EDGE_PT, zorder=2)  # fmt: skip
    xerr = [[e["est"] - e["lo"]], [e["hi"] - e["est"]]]
    ax.errorbar([e["est"]], [y], xerr=xerr, zorder=3, **NATIVE_ERROR)
    if sum(to_rgb(face)) < 1.5:  # a dark face: the black error bar would vanish inside it
        white = NATIVE_ERROR | {"ecolor": "white"}
        inner = ax.errorbar([e["est"]], [y], xerr=xerr, zorder=3.1, **white)
        for part in (*inner.lines[1], *inner.lines[2]):
            part.set_clip_path(bar)


def fig_native_systems(e5: dict[str, Any] | None, out: Path) -> Path:
    """Per metric (rows) and branch (columns), grouped bars on a 0-100% axis: per system the
    HoldMem cell with its inferred label (in the colour of the cell's binding arm) above the native
    system (dark grey, filled when its estimate lies inside the cell's 95% interval, outlined when
    outside), each with its 95% interval as an error bar."""
    from matplotlib.patches import Patch
    from matplotlib.ticker import MultipleLocator

    plt = fs.plt()
    native = fs.SYSTEMS["native"][1]
    panels = native_panels(e5)
    n = len(panels)
    fig, axes = plt.subplots(
        len(NATIVE_METRICS),
        len(NATIVE_BRANCHES),
        figsize=(fs.TEXT, NATIVE_H),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    fig.get_layout_engine().set(wspace=0.08)  # "100" of one panel clears "0" of the next
    cells: dict[str, tuple[str, str]] = {}  # the HoldMem cells drawn, for the legend
    for i, m in enumerate(NATIVE_METRICS):
        for j, (br, br_name) in enumerate(NATIVE_BRANCHES):
            ax = axes[i][j]
            for k, rec in enumerate(panels):
                y = n - 1 - k
                v = rec["values"][br, m]
                if not rec["staged"]:  # where its bars would start
                    ax.annotate("not run", xy=(0, y), xytext=(4, 0), textcoords="offset points",
                                ha="left", va="center", fontsize=NOTE_PT, color=fs.GREY,
                                bbox=CLEAR)  # fmt: skip
                    continue
                if v["pred"]:
                    label, colour = _cell_style(rec["cell"])
                    cells.setdefault(rec["cell"] or "", (label, colour))
                    _native_bar(ax, y + NATIVE_BAR / 2, v["pred"], colour, colour)
                if v["native"]:  # filled inside the cell's interval, outlined outside it
                    face = native if v["inside"] else "white"
                    _native_bar(ax, y - NATIVE_BAR / 2, v["native"], face, native)
            ax.set_title(f"{m}, {br_name}", pad=3)
            fs.pct_axis(ax, "x", 0.0, 1.0)
            ax.xaxis.set_major_locator(MultipleLocator(0.2))
            ax.grid(True, axis="x")
            ax.grid(False, axis="y")
            ax.set_ylim(-0.6, n - 0.4)
            ax.tick_params(axis="y", length=0)
            if i == len(NATIVE_METRICS) - 1:
                ax.set_xlabel("accuracy (\\%)")
    ticks = [
        f"{rec['label']} {rec['inferred']}" if rec["inferred"] else rec["label"] for rec in panels
    ]
    axes[0][0].set_yticks(range(n), list(reversed(ticks)))
    keys = [(label, colour, colour) for label, colour in cells.values()]  # (label, face, edge)
    keys += [("native system, inside the cell's interval", native, native),
             ("outside it", "white", native)]  # fmt: skip
    handles = [Patch(fc=face, ec=edge, lw=fs.EDGE_PT) for _, face, edge in keys]
    labels = [label for label, _, _ in keys]
    fig.legend(handles, labels, loc="outside upper center", ncol=len(handles), columnspacing=1.4,
               handlelength=1.4)  # fmt: skip
    path = out / "fig_native_systems.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


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
