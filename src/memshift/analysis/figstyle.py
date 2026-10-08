"""One visual language for every paper figure (NeurIPS 2025 single-column layout).

Text is typeset by LaTeX with the paper's own fonts (newtx Times text and math), so figures match
the body and embed only Type-1 fonts. Figures are saved at their printed size (no tight bounding
box) and included at 100%, so the 9/8 pt figure text sits next to the 10 pt body at a fixed ratio.
The text block is 5.5 in wide. The encoding is fixed across figures: hue = binding (blue B=1
source epoch, light blue B=1 cited cluster, orange B=0), filled vs. hollow marker = D.
"""

from __future__ import annotations

from typing import Any

TEXT = 5.5  # text width of the paper (in), NeurIPS 2025

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
