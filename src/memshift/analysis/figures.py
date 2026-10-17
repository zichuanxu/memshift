"""Paper figures regenerated from the committed run analyses (plan §10: figures come from runs/).

Sources are the analysis directories listed in results/sources.json, the same ones the
paper's number macros are generated from.

fig_dissociation   E1: detection (missed invalidation), repair (correct re-derivations) and the
                   outcome (Update) against D, one line per binding arm, S_src and S_def.
fig_calibration    E2: per-type writer agreement on the calibration set against observed survival
                   of D=0, B=1 items after the switch, all three tiers.
fig_zero_cost      E3: static ContractValid by horizon; tokens per query against one full re-read.
fig_native         E5: Mem0 against the interval predicted by the white-box cell with its labels.
fig_native_systems Block N (prereg v2 §8): predicted against observed for every native system, per
                   branch and metric (Update, Stability); systems that were not run show as such.
fig_classsplit     exploratory (`ex_e1_paper`): B and B x D on Update by item class and branch
fig_account        exploratory (`ex_review_paper`): B x D against the repair gap r1 - r0 by writer
fig_writers        round 3 (`round3_figures`): B x D against how often the programs are right (G-b
                   program sets) and against r_0 (definition-item writers of block H)
fig_reader         round 3: B x D with each reader of the same stores against the store part
fig_misses         exploratory: why the cited-cluster key misses changed item-times
fig_registered     every registered prediction with an estimate against its registered bar; with it
                   paper/generated/tab_registered_tests.tex (the other registered verdicts) and
                   the counts (held, failed, mixed) as paper/generated/registered_counts.tex
Usage: python -m memshift.analysis.figures [--root .] [--out paper/figures] [--only NAME ...]
       [--counts paper/generated/registered_counts.tex]
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

ARM_CELLS = {  # binding arm -> (cell with D=0, cell with D=1)
    "none": ("holdmem_D0B0", "holdmem_D1B0"),
    "cited": ("holdmem_D0B1_cited", "holdmem_D1B1_cited"),
    "source": ("holdmem_D0B1_source", "holdmem_D1B1_source"),
}
BY_DESIGN = {  # detection lines that are 0 or 100 by construction (fig_dissociation)
    "none": "100 by design",
    "source": "0 by design",
}
BRANCH_LABELS = {
    "src": r"source revision $S_{\mathrm{src}}$",
    "def": r"definition change $S_{\mathrm{def}}$",
}


def _ok(e: dict[str, Any] | None) -> TypeGuard[dict[str, Any]]:
    return e is not None and e.get("est") is not None and not math.isnan(e["est"])


def _err(e: dict[str, Any]) -> list[list[float]]:
    return [[max(0.0, e["est"] - e["lo"])], [max(0.0, e["hi"] - e["est"])]]


def fig_dissociation(e1: dict[str, Any], out: Path) -> Path:
    plt = fs.plt()
    metrics = [
        ("MissedInvalidation", "Detection:\nmissed invalidation (\\%)"),
        ("RepairCorrectness", "Repair:\ncorrect re-derivations (\\%)"),
        ("Update", "Outcome:\nUpdate (\\%)"),
    ]
    branches = [b for b in ("src", "def") if b in e1["cells"]]
    fig, axes = plt.subplots(
        len(branches),
        len(metrics),
        figsize=(fs.TEXT, 1.62 * len(branches) + 0.55),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for i, b in enumerate(branches):
        cells = e1["cells"][b]
        for j, (metric, title) in enumerate(metrics):
            ax = axes[i][j]
            for arm in ("none", "cited", "source"):
                label, colour, ls = fs.ARMS[arm]
                xs, ys, lo, hi = [], [], [], []
                for d, cell in enumerate(ARM_CELLS[arm]):
                    e = cells.get(cell, {}).get(metric)
                    if _ok(e):
                        xs.append(d)
                        ys.append(e["est"])
                        lo.append(e["est"] - e["lo"])
                        hi.append(e["hi"] - e["est"])
                if not xs:
                    continue
                # 0 or 100 by construction: a source-epoch key invalidates every changed object,
                # B=0 has no key to invalidate; drawn lighter and named in the panel
                design = metric == "MissedInvalidation" and arm in BY_DESIGN
                alpha = 0.5 if design else 1.0
                ax.plot([], [], ls=ls, color=colour, lw=1.4, label=label)  # full-colour legend key
                ax.plot(xs, ys, ls=ls, color=colour, lw=1.4, zorder=2, alpha=alpha)
                ax.errorbar(
                    xs,
                    ys,
                    yerr=[lo, hi],
                    fmt="none",
                    ecolor=colour,
                    elinewidth=1.0,
                    zorder=3,
                    alpha=alpha,
                )
                for x, y in zip(xs, ys, strict=True):  # hollow marker: D=0, filled: D=1
                    ax.plot(
                        [x],
                        [y],
                        marker="o",
                        ms=5.2,
                        mec=colour,
                        mew=1.2,
                        mfc=colour if x == 1 else "white",
                        ls="none",
                        zorder=4,
                        alpha=alpha,
                    )
                if design:
                    ax.text(
                        0.5,
                        0.93 if arm == "none" else 0.015,  # clear of the cited line above 0
                        BY_DESIGN[arm],
                        ha="center",
                        va="top" if arm == "none" else "bottom",
                        fontsize=7.5,
                        color=colour,
                    )
            if metric == "Update" and _ok(cells.get("zero", {}).get("Update")):
                z = cells["zero"]["Update"]["est"]
                ax.axhline(
                    z, color=fs.GREY, lw=1.0, ls=(0, (1, 1.5)), zorder=1, label="zero system"
                )
            if metric == "RepairCorrectness":
                ax.text(
                    0.5,
                    0.07,
                    r"$B{=}0$ never re-derives",
                    transform=ax.transAxes,
                    ha="center",
                    fontsize=8,
                    color=fs.ORANGE,
                )
            ax.set_xlim(-0.3, 1.3)
            ax.set_xticks([0, 1], [r"$D{=}0$", r"$D{=}1$"])
            ax.tick_params(axis="x", length=0, pad=4)
            fs.pct_axis(ax, "y", -0.06, 1.07)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.spines["left"].set_bounds(0, 1)
            if i == 0:
                ax.set_title(title, pad=6)
            if j == 0:
                ax.set_ylabel(BRANCH_LABELS[b])
    handles, labels = axes[0][0].get_legend_handles_labels()
    zh, zl = axes[0][-1].get_legend_handles_labels()
    handles, labels = handles + zh, labels + zl
    wanted = [fs.ARMS[a][0] for a in ("source", "cited", "none")] + ["zero system"]
    order = [labels.index(w) for w in wanted if w in labels]
    fig.legend(
        [handles[k] for k in order],
        [labels[k] for k in order],
        loc="outside upper center",
        ncol=4,
        columnspacing=2.4,
    )
    fig.get_layout_engine().set(h_pad=0.05)
    path = out / "fig_dissociation.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


RUG_BASE, RUG_HEIGHT = -0.13, 0.10  # the item-count rug of the calibration figure (axis units)
CALIBRATION_W = 2.6  # the calibration and zero-system figures share one row of the page
ZERO_W = fs.TEXT - CALIBRATION_W - 0.1
PAIR_H = 2.55


def fig_calibration(e2: dict[str, Any], out: Path) -> Path:
    plt = fs.plt()
    fig, ax = plt.subplots(figsize=(CALIBRATION_W, PAIR_H))
    styles = {
        "same": ("same tier", fs.BLUE, "o"),
        "upgrade": ("upgrade", fs.GREEN, "s"),
        "downgrade": ("downgrade", fs.VERMILION, "^"),
    }
    caveats = set((e2.get("path3") or {}).get("tier_caveats", []))
    ax.plot([0, 1], [0, 1], color=fs.GREY, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.text(
        0.52,
        0.45,
        "perfect calibration",
        rotation=41,
        rotation_mode="anchor",
        color=fs.GREY,
        fontsize=7.5,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3},  # no gridline through it
        zorder=1.5,
    )
    curve = e2.get("explore_pooled_curve") or {}
    fit = bool(curve.get("est")) and not any(math.isnan(v) for v in curve["est"])
    if fit:  # the pooled logistic fit whose slope and AUROC are printed below, with its 90% band
        ax.fill_between(
            curve["grid"],
            curve["lo90"],
            curve["hi90"],
            color=fs.LIGHT,
            alpha=0.85,
            lw=0,
            zorder=0.5,
        )
        ax.plot(curve["grid"], curve["est"], color=fs.BLACK, lw=1.1, zorder=2)
    named: dict[str, tuple[float, float, int]] = {}
    counts: list[tuple[float, int, str]] = []  # (agreement, items, colour) of every plotted type
    for tier, (label, colour, marker) in styles.items():
        pts = (e2["calibration"].get(tier) or {}).get("per_type", {})
        if not pts:
            continue
        hollow = tier in caveats
        counts += [(p["predicted"], p["n"], colour) for p in pts.values()]
        ax.scatter(
            [p["predicted"] for p in pts.values()],
            [p["observed"] for p in pts.values()],
            s=[6 + 4.0 * math.sqrt(p["n"]) for p in pts.values()],
            marker=marker,
            facecolor="none" if hollow else colour,
            edgecolor=colour,
            linewidth=1.0,
            alpha=0.85,
            zorder=3,
            label=label,
        )
        for name, p in pts.items():  # name every item type that misses calibration by 20+ points
            if abs(p["observed"] - p["predicted"]) > 0.2 and not hollow:
                named[name] = (p["predicted"], p["observed"], p["n"])
    for name, (x, y, n) in named.items():
        short = name.split(".")[-1].replace("_", " ") + (f" ($n{{=}}{n}$)" if n < 7 else "")
        right = x > 0.8
        ax.annotate(
            short,
            (x, y),
            xytext=(-6 if right else 5, (-9 if y > 0.95 else 5) if not right or y < 0.05 else -3),
            textcoords="offset points",
            fontsize=7.5,
            color=fs.GREY,
            ha="right" if right else "left",
        )
    p3 = e2.get("path3") or {}
    sl, au = p3.get("pooled_slope") or {}, p3.get("pooled_auroc") or {}
    if sl and au:
        ax.text(
            0.98,
            0.40,
            f"slope {sl['est']:.2f} [{sl['lo90']:.2f}, {sl['hi90']:.2f}]\n"
            f"AUROC {au['est']:.2f} [{au['lo90']:.2f}, {au['hi90']:.2f}]",
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=8,
            linespacing=1.3,
        )
    if counts:  # rug of the items behind each point: tick height grows with sqrt(items)
        n_max = max(n for _, n, _ in counts)
        for x, n, colour in counts:
            top = RUG_BASE + RUG_HEIGHT * math.sqrt(n / n_max)
            ax.plot([x, x], [RUG_BASE, top], color=colour, lw=0.9, alpha=0.7, zorder=2)
        ax.text(0.205, RUG_BASE + 0.05, "items", fontsize=7, color=fs.GREY, va="center")
    fs.pct_axis(ax, "x", 0.2, 1.04)
    fs.pct_axis(ax, "y", RUG_BASE - 0.02, 1.07)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.spines["left"].set_bounds(0, 1)
    ax.grid(True, axis="both")
    ax.set_xlabel("writer agreement, calibration set (\\%)")
    ax.set_ylabel("survival after the switch (\\%)")
    handles, labels = ax.get_legend_handles_labels()
    if fit:
        from matplotlib.lines import Line2D

        handles.append(Line2D([], [], color=fs.BLACK, lw=1.1))
        labels.append("pooled fit, 90\\% band")
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=2 if fit else 3,
        handletextpad=0.3,
        columnspacing=0.8,
        markerscale=0.8,
        handlelength=1.4,
    )
    path = out / "fig_calibration.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_zero_cost(e3: dict[str, Any], out: Path) -> Path:
    from matplotlib.lines import Line2D

    plt = fs.plt()
    hs = sorted(int(h) for h in e3["horizons"])
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(ZERO_W, PAIR_H), gridspec_kw={"width_ratios": [1.35, 1]}
    )
    order = ("holdmem_D1B1_source", "oracle", "zero", "rag")
    for s in order:
        label, colour, ls, marker = fs.SYSTEMS[s]
        pts = [
            (h, e3["horizons"][str(h)][s]["contract_valid"])
            for h in hs
            if s in e3["horizons"][str(h)]
        ]
        if not pts:
            continue
        x = [math.log(h) for h, _ in pts]
        ax1.fill_between(
            x, [e["lo"] for _, e in pts], [e["hi"] for _, e in pts], color=colour, alpha=0.15, lw=0
        )
        ax1.plot(
            x, [e["est"] for _, e in pts], ls=ls, color=colour, marker=marker, ms=3.6, label=label
        )
    ax1.set_xticks([math.log(h) for h in hs], [str(h) for h in hs])
    ax1.set_xlim(-0.2, math.log(hs[-1]) + 0.2)
    ax1.set_xlabel("horizon (months)")
    fs.pct_axis(ax1, "y", 0.55, 0.97)
    ax1.set_ylabel("static ContractValid (\\%)")
    last = e3["horizons"][str(hs[-1])]
    systems = [s for s in ("holdmem_D1B1_source", "rag", "zero") if "cost_mean" in last.get(s, {})]
    reread = last[systems[0]]["full_reread_tokens_per_query"]
    top = 2e8
    ax2.axhspan(0.5 * reread, top, color=fs.LIGHT, alpha=0.7, lw=0, zorder=0)
    ax2.axhline(reread, color=fs.BLACK, lw=0.8, zorder=1)
    ax2.text(1.0, reread * 1.3, "full re-read", fontsize=7.5, ha="center", va="bottom")
    ax2.text(
        1.0, top / 1.6, "fails cost ceiling", fontsize=7.5, ha="center", va="top", color=fs.GREY
    )
    for k, s in enumerate(systems):  # colour and marker as in panel (a), whose legend names them
        _, colour, _, marker = fs.SYSTEMS[s]
        c = last[s]
        queries = c["cost_mean"]["serve_tokens"] / max(c["serve_tokens_per_query"], 1e-9)
        parts = ("ingest_tokens", "maintain_tokens", "serve_tokens")
        total = sum(c["cost_mean"][p] for p in parts) / queries
        serve = c["serve_tokens_per_query"]
        ax2.plot([k, k], [serve, total], color=colour, lw=1.0, zorder=2)
        ax2.plot([k], [total], marker=marker, ms=5, mfc="white", mec=colour, mew=1.2, zorder=3)
        ax2.plot([k], [serve], marker=marker, ms=5, color=colour, zorder=4)
    ax2.set_yscale("log")
    ax2.set_ylim(1e3, top)
    ax2.set_yticks([1e3, 1e4, 1e5, 1e6, 1e7, 1e8])
    # no system names under the narrow panel: the figure legend above names the markers
    ax2.set_xticks(range(len(systems)), [""] * len(systems))
    ax2.tick_params(axis="x", length=0)
    ax2.set_xlim(-0.6, len(systems) - 0.4)
    ax2.set_ylabel("tokens per query (12 months)")
    ax2.grid(True, axis="y", which="major")
    ax2.legend(
        handles=[
            Line2D([], [], marker="o", color=fs.GREY, ls="none", ms=5, label="serving"),
            Line2D(
                [],
                [],
                marker="o",
                mfc="white",
                mec=fs.GREY,
                mew=1.2,
                ls="none",
                ms=5,
                label="+ ingest,\nmaintain",
            ),
        ],
        loc="center",
        bbox_to_anchor=(0.5, 0.47),
        handletextpad=0.2,
        handlelength=1.0,
        fontsize=7.5,
        borderaxespad=0,
    )
    fig.legend(
        *ax1.get_legend_handles_labels(),
        loc="outside upper center",
        ncol=2,
        columnspacing=1.2,
        handlelength=1.8,
    )
    fs.panel_label(ax1, "a")
    fs.panel_label(ax2, "b")
    path = out / "fig_zero_cost.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_native(e5: dict[str, Any], out: Path) -> Path:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt = fs.plt()
    mem = (e5.get("systems") or {}).get("mem0") or {}
    cells = (e5.get("exploratory") or {}).get("white_box_cells") or {}
    rows = [(b, m) for b in ("src", "def", "orphan") for m in ("Update", "Stability")]
    names = {
        "src": r"$S_{\mathrm{src}}$",
        "def": r"$S_{\mathrm{def}}$",
        "orphan": r"$S_{\mathrm{del}}$",
    }
    fig, ax = plt.subplots(figsize=(fs.TEXT, 1.95))
    for k, (b, m) in enumerate(rows):
        entry = (mem.get("branches") or {}).get(b) or {}
        pred, nat = (entry.get("white_box") or {}).get(m), (entry.get("native") or {}).get(m)
        y = len(rows) - 1 - k
        if _ok(pred):
            ax.barh(
                [y],
                [pred["hi"] - pred["lo"]],
                left=[pred["lo"]],
                height=0.56,
                color=fs.LIGHT,
                lw=0,
                zorder=1,
            )
            ax.plot([pred["est"]] * 2, [y - 0.28, y + 0.28], color=fs.GREY, lw=1.0, zorder=2)
        other = (cells.get("holdmem_D0B1_source") or {}).get(b, {}).get(m)
        if _ok(other):
            ax.plot([other["est"]], [y], marker="|", ms=9, mew=1.4, color=fs.BLUE, zorder=2)
        if _ok(nat):
            inside = (entry.get("consistent") or {}).get(m)
            ax.errorbar(
                [nat["est"]],
                [y],
                xerr=_err(nat),
                fmt="o",
                ms=4.8,
                color=fs.VERMILION,
                mfc=fs.VERMILION if inside else "white",
                mew=1.2,
                elinewidth=1.0,
                zorder=4,
            )
    ax.set_yticks(range(len(rows)), [f"{names[b]} {m}" for b, m in reversed(rows)])
    ax.tick_params(axis="y", length=0)
    fs.pct_axis(ax, "x", 0.2, 1.02)
    ax.set_xlabel("ContractValid on the measurement half of the items (\\%)")
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.set_ylim(-0.6, len(rows) - 0.4)
    handles = [
        Patch(color=fs.LIGHT, label=r"prediction: \textsc{HoldMem}(0,0), 95\% interval"),
        Line2D([], [], marker="o", color=fs.VERMILION, ls="none", ms=4.8, label="Mem0, inside"),
        Line2D(
            [],
            [],
            marker="o",
            mfc="white",
            mec=fs.VERMILION,
            mew=1.2,
            ls="none",
            ms=4.8,
            label="Mem0, outside",
        ),
        Line2D(
            [],
            [],
            marker="|",
            color=fs.BLUE,
            ls="none",
            ms=9,
            mew=1.4,
            label=r"\textsc{HoldMem}(0,1), reference",
        ),
    ]
    fig.legend(handles=handles, loc="outside right center", handletextpad=0.5, labelspacing=0.7)
    path = out / "fig_native.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


NATIVE_BRANCHES = (
    ("src", r"$S_{\mathrm{src}}$"),
    ("def", r"$S_{\mathrm{def}}$"),
    ("orphan", r"$S_{\mathrm{del}}$"),  # the deletion (orphan) branch, as in Figures 1 and 2
)
NATIVE_METRICS = ("Update", "Stability")


def native_panels(e5: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The data of `fig_native_systems`: one record per expected system (Mem0 and the five of
    block N, then any other system in the analysis) with its inferred label and, per branch and
    metric, the native estimate, the white-box prediction interval and whether the estimate lies
    inside it. A system that is not in the analysis has `staged` False and no values."""
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


def fig_native_systems(e5: dict[str, Any] | None, out: Path) -> Path:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt = fs.plt()
    panels = native_panels(e5)
    n = len(panels)
    fig, axes = plt.subplots(
        len(NATIVE_METRICS),
        len(NATIVE_BRANCHES),
        figsize=(fs.TEXT, len(NATIVE_METRICS) * (0.22 * n + 0.5) + 0.4),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for i, m in enumerate(NATIVE_METRICS):
        for j, (br, br_name) in enumerate(NATIVE_BRANCHES):
            ax = axes[i][j]
            for k, rec in enumerate(panels):
                y = n - 1 - k
                v = rec["values"][br, m]
                if not rec["staged"]:
                    ax.text(0.5, y, "not run", ha="center", va="center", fontsize=7, color=fs.GREY)
                    continue
                pred, nat = v["pred"], v["native"]
                if pred:
                    ax.barh(
                        [y],
                        [pred["hi"] - pred["lo"]],
                        left=[pred["lo"]],
                        height=0.6,
                        color=fs.LIGHT,
                        lw=0,
                        zorder=1,
                    )
                    ax.plot([pred["est"]] * 2, [y - 0.3, y + 0.3], color=fs.GREY, lw=0.9, zorder=2)
                if nat:  # filled inside the predicted interval, outlined outside it
                    fs.interval_bar(ax, y, nat["est"], nat["lo"], nat["hi"], fs.VERMILION,
                                    filled=bool(v["inside"]), size=4.0, zorder=4)  # fmt: skip
            ax.set_title(f"{m}, {br_name}", pad=3)
            # padded so that the 100 of one panel and the 0 of the next do not run together
            fs.pct_axis(ax, "x", -0.05, 1.07)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.grid(True, axis="x")
            ax.grid(False, axis="y")
            ax.set_ylim(-0.6, n - 0.4)
            ax.tick_params(axis="y", length=0)
            if i == len(NATIVE_METRICS) - 1:
                ax.set_xlabel("ContractValid (\\%)")
    ticks = [
        f"{rec['label']} {rec['inferred']}" if rec["inferred"] else rec["label"] for rec in panels
    ]
    axes[0][0].set_yticks(range(n), list(reversed(ticks)))
    for i in range(len(NATIVE_METRICS)):
        axes[i][0].tick_params(axis="y", labelsize=7.5)
    handles = [  # every mark of the panels: bar and tick of the prediction, native estimate
        Patch(color=fs.LIGHT, label="predicted, 95\\% interval"),
        Line2D([], [], marker="|", color=fs.GREY, ls="none", ms=8, mew=0.9,
               label="predicted estimate"),
        fs.interval_handle(fs.VERMILION, True, "native, 95\\% interval: inside"),
        fs.interval_handle(fs.VERMILION, False, "outside"),
    ]  # fmt: skip
    fig.legend(
        handles=handles,
        handler_map=fs.INTERVAL_HANDLERS,
        loc="outside upper center",
        ncol=4,
        columnspacing=1.2,
    )
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
    ap.add_argument("--only", nargs="*", default=[], help="figure names (fig_native, ...)")
    ap.add_argument(
        "--counts",
        type=Path,
        default=Path("paper/generated/registered_counts.tex"),
        help="where fig_registered writes the counts the paper quotes (registered_counts)",
    )
    a = ap.parse_args()
    sources = json.loads((a.root / "results" / "sources.json").read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = [
        (fig_dissociation, "e1", "e1_metrics.json"),
        (fig_calibration, "e2", "e2_metrics.json"),
        (fig_zero_cost, "e3", "e3_metrics.json"),
        (fig_native, "e5", "e5_natives.json"),
    ]
    for fn, group, name in jobs:
        data = _analysis(a.root, sources, group, name)
        if data is not None and (not a.only or fn.__name__ in a.only):
            print(fn(data, a.out))
    if not a.only or "fig_native_systems" in a.only:  # draws the systems that are not staged, too
        print(fig_native_systems(native_results(load_stage4(a.root, sources)), a.out))
    if (
        not a.only or "fig_classsplit" in a.only
    ):  # exploratory class split (decisions.md 2026-09-30)
        from memshift.analysis import ex_e1_paper

        print(ex_e1_paper.fig_classsplit(ex_e1_paper.load(a.root, sources), a.out))
    if not a.only or "fig_account" in a.only:  # exploratory account across writers (2026-10-01)
        from memshift.analysis import ex_review_paper

        print(ex_review_paper.fig_account(ex_review_paper.load(a.root, sources), a.out))
    from memshift.analysis import round3_figures  # round 3 and the registration record

    wanted = [n for n in round3_figures.FIGURES if not a.only or n in a.only]
    if wanted:
        inp = round3_figures.load(a.root, sources)
        for name in wanted:
            print(round3_figures.FIGURES[name](inp, a.out))
        if "fig_registered" in wanted:  # the counts quoted in the paper, and the other verdicts
            a.counts.parent.mkdir(parents=True, exist_ok=True)
            a.counts.write_text(round3_figures.counts_tex(round3_figures.paper_counts(inp)))
            print(a.counts)
            table = a.counts.parent / "tab_registered_tests.tex"
            table.write_text(
                round3_figures.tab_registered_tests(round3_figures.registered_tests(inp))
            )
            print(table)


if __name__ == "__main__":
    main()
