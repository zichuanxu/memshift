"""One visual language for every paper figure (single-column preprint layout).

Text is typeset by LaTeX with the paper's own fonts (newtx Times text and math), so figures match
the body and embed only Type-1 fonts. Figures are saved at their printed size (no tight bounding
box) and included at 100%, so the 9/8 pt figure text sits next to the 10 pt body at a fixed ratio.
A full-width figure is 5.5 in wide, centred in the 6.5 in text block. The encoding is fixed across
figures: hue = binding (blue B=1
source epoch, light blue B=1 cited cluster, orange B=0), filled vs. hollow marker = D. The round-3
figures have no binding arm: there hue = the model (`MODELS`) and, in the scatter of writers,
marker shape = the shift (`SHIFTS`).

An estimate with its 95% interval is drawn as an interval bar (`interval_bar`): the interval is a
soft translucent bar with exact ends, the estimate a short dark stroke across it. A second series
that must stay apart in print and greyscale (the replication, a failed prediction, a value outside
its predicted interval) is an outlined bar instead of a filled one. Lines with bands over time keep
their bands.
"""

from __future__ import annotations

from typing import Any

TEXT = 5.5  # width of a full-width figure (in); the text block is 6.5 in

# Okabe-Ito, colour-blind safe
BLUE = "#0072B2"
SKY = "#56B4E9"
ORANGE = "#E69F00"
VERMILION = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GREY = "#6E6E6E"
LIGHT = "#DCDCDC"
BLACK = "#222222"

ARMS = {  # binding arm -> (label, colour, line style)
    "source": (r"$B{=}1$, source epoch", BLUE, "-"),
    "cited": (r"$B{=}1$, cited cluster", SKY, "--"),
    "none": (r"$B{=}0$", ORANGE, "-"),
}
SYSTEMS = {  # baseline -> (label, colour, line style, marker)
    "oracle": ("gold-dependency context", GREEN, ":", "D"),
    "holdmem_D1B1_source": (r"\textsc{HoldMem}(1,1)", BLUE, "-", "o"),
    "zero": ("zero system", BLACK, "-", "s"),
    "rag": ("plain RAG", PURPLE, "-", "^"),
    "mem0": ("Mem0", VERMILION, "-", "*"),
}
# round-3 figures: hue = the model (writer of programs or definition items, or reader), marker
# shape = the shift; a model without an entry is drawn in GREY under its own name
MODELS = {  # model id -> (label, colour)
    "Qwen/Qwen3.5-9B": ("Qwen3.5-9B", BLACK),
    "claude-sonnet-5-5": ("Claude Sonnet 5.5", VERMILION),
    "claude-opus-5-5": ("Claude Opus 5.5", PURPLE),
    "google/gemma-4-12B-it": ("gemma-4-12B", GREEN),
    "meta-llama/Llama-3.1-8B-Instruct": ("Llama-3.1-8B", ORANGE),
    "openai/gpt-oss-20b": ("gpt-oss-20b", SKY),
}
SHIFTS = {  # branch -> (label, marker)
    "src": (r"$S_{\mathrm{src}}$", "o"),
    "def": (r"$S_{\mathrm{def}}$", "s"),
    "orphan": (r"$S_{\mathrm{del}}$", "^"),
}


def model_style(model: str) -> tuple[str, str]:
    """(label, colour) of a model; an unknown model keeps its name after the last slash."""
    return MODELS.get(model, (model.rsplit("/", 1)[-1], GREY))


def plt() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    pyplot.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{newtxtext,newtxmath}",
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.6,
            "axes.edgecolor": BLACK,
            "axes.labelcolor": BLACK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": "#EBEBEB",
            "grid.linewidth": 0.5,
            "axes.axisbelow": True,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.color": BLACK,
            "ytick.color": BLACK,
            "lines.linewidth": 1.3,
            "lines.markersize": 4.5,
            "errorbar.capsize": 0,
            "legend.frameon": False,
            "legend.handlelength": 1.8,
            "figure.constrained_layout.use": True,
            "figure.constrained_layout.h_pad": 0.02,
            "figure.constrained_layout.w_pad": 0.02,
            "savefig.dpi": 300,
            "savefig.bbox": "standard",
            "pdf.fonttype": 42,
        }
    )
    return pyplot


def pct_axis(ax: Any, axis: str = "y", lo: float = 0.0, hi: float = 1.0) -> None:
    from matplotlib.ticker import FuncFormatter

    fmt = FuncFormatter(lambda v, _: f"{100 * v:.0f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)
    (ax.set_ylim if axis == "y" else ax.set_xlim)(lo, hi)


def panel_label(ax: Any, text: str) -> None:
    ax.set_title(rf"\textbf{{({text})}}", loc="left", fontsize=9, pad=4)


BAR_PT = 4.5  # thickness of an interval bar (points)
STROKE_PT = 1.7  # width of the estimate's stroke (points)
EDGE_PT = 0.7  # outline of an outlined interval bar (points)
BAR_ALPHA = 0.4


def darker(colour: str, f: float = 0.3) -> tuple[float, float, float]:
    """The colour mixed with black: the estimate's stroke over its own soft bar."""
    from matplotlib.colors import to_rgb

    r, g, b = to_rgb(colour)
    return (r * (1 - f), g * (1 - f), b * (1 - f))


def interval_bar(
    ax: Any,
    pos: float,
    est: float | None,
    lo: float,
    hi: float,
    colour: str,
    *,
    filled: bool = True,
    vertical: bool = False,
    size: float = BAR_PT,
    alpha: float = BAR_ALPHA,
    zorder: float = 3.0,
) -> None:
    """An estimate with its interval at `pos` (y for a horizontal bar, x for a vertical one): the
    interval [lo, hi] as a bar `size` points thick whose ends are exactly lo and hi (butt caps),
    soft (translucent) when `filled`, else outlined; the estimate as a short dark stroke a little
    longer than the bar is thick (none when `est` is None)."""
    from matplotlib.transforms import ScaledTranslation

    def at(a: float, b: float) -> tuple[list[float], list[float]]:
        return ([pos, pos], [a, b]) if vertical else ([a, b], [pos, pos])

    if filled:
        ax.plot(*at(lo, hi), color=colour, alpha=alpha, lw=size, solid_capstyle="butt",
                zorder=zorder)  # fmt: skip
    else:  # two long edges shifted half a bar off the centre line, and the two ends
        half = (size - EDGE_PT) / 2
        for off in (-half, half):
            shift = ScaledTranslation(off / 72 if vertical else 0, 0 if vertical else off / 72,
                                      ax.figure.dpi_scale_trans)  # fmt: skip
            ax.plot(*at(lo, hi), color=colour, lw=EDGE_PT, solid_capstyle="butt",
                    transform=ax.transData + shift, zorder=zorder)  # fmt: skip
        for end in (lo, hi):
            x, y = at(end, end)
            ax.plot(x[:1], y[:1], marker="_" if vertical else "|", ms=size, mew=EDGE_PT,
                    color=colour, ls="none", zorder=zorder)  # fmt: skip
    if est is not None:
        x, y = at(est, est)
        ax.plot(x[:1], y[:1], marker="_" if vertical else "|", ms=size * 1.55, mew=STROKE_PT,
                color=darker(colour), ls="none", zorder=zorder + 0.1)  # fmt: skip


def interval_handle(colour: str, filled: bool = True, label: str = "") -> Any:
    """A legend handle that looks like `interval_bar` (pass `handler_map=INTERVAL_HANDLERS`)."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    stroke = Line2D([], [], marker="|", ms=BAR_PT * 1.55, mew=STROKE_PT, color=darker(colour),
                    ls="none")  # fmt: skip
    if filled:
        bar = Rectangle((0, 0), 1, 1, fc=colour, alpha=BAR_ALPHA, ec="none")
    else:
        bar = Rectangle((0, 0), 1, 1, fc="none", ec=colour, lw=EDGE_PT)
    return _IntervalHandle(bar, stroke, label)


class _IntervalHandle:
    def __init__(self, bar: Any, stroke: Any, label: str) -> None:
        self.bar, self.stroke, self.label = bar, stroke, label

    def get_label(self) -> str:
        return self.label


class _IntervalHandler:
    """Draws an `interval_handle`: a bar as thick as `BAR_PT` across the handle box, and the
    stroke in its middle."""

    def legend_artist(self, legend: Any, orig: _IntervalHandle, fontsize: float, box: Any) -> Any:
        from matplotlib.lines import Line2D
        from matplotlib.patches import Rectangle

        x0, y0, w, h = box.xdescent, box.ydescent, box.width, box.height
        tall = BAR_PT  # the handle box is laid out in points
        bar = Rectangle((-x0, -y0 + (h - tall) / 2), w, tall, transform=box.get_transform())
        bar.update_from(orig.bar)
        bar.set_transform(box.get_transform())
        stroke = Line2D([-x0 + w / 2], [-y0 + h / 2], transform=box.get_transform())
        stroke.update_from(orig.stroke)
        stroke.set_transform(box.get_transform())
        box.add_artist(bar)
        box.add_artist(stroke)
        return bar


INTERVAL_HANDLERS = {_IntervalHandle: _IntervalHandler()}
