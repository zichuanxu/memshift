"""One visual language for every paper figure (two-column preprint, one-column appendix).

Figures are drawn at their printed size and included at 100%: a full-width figure is `TEXT` =
6.75 in (486 pt) wide, the whole text block; a single-panel appendix figure is `NARROW` = 4.5 in
(324 pt) wide. Positions are laid out in millimetres (`PT` mm per point), text sizes in points.
Text is set by matplotlib in the serif face (`SERIF`: Times New Roman where it is installed, else
matplotlib's own STIXGeneral, so drawing never fails) with STIX math, and the PDF embeds TrueType
fonts and carries no creation date, so the same data give the same file (`printed`, `save_pdf`).

Colours have one meaning each across the figure set: no key (B=0) amber, key (B=1) blue, exact
recomputation (D=1) a deep blue (`ROLE_BLUE`), the writer re-derives (D=0) slate grey, native
systems and store parts cool greys. Where models are compared, hue is the model (`MODELS`) and
marker shape the shift (`SHIFTS`). Intervals are thin lines in their mark's own dark ink.

The bar figures (fig_classsplit, fig_reader, fig_native_systems) share one type scale (`T_*`), one
value axis (`bar_axes`), one interval mark (`interval_line`) and one key row (`key_row`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

TEXT = 6.75  # width of a full-width figure (in): the text block
NARROW = 4.5  # largest width of a single-panel appendix figure (in)
PT = 25.4 / 72  # mm per point
MM = 1 / 25.4  # inches per mm

# Okabe-Ito, colour-blind safe
BLUE = "#0072B2"
SKY = "#56B4E9"
ORANGE = "#E69F00"
VERMILION = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GREY = "#6E6E6E"
BLACK = "#222222"

# text and apparatus
INK = "#262B33"  # text
MUTED = "#6B7480"  # secondary text
SPINE = "#9AA0A8"  # axis lines and tick marks
TICK = "#4A5059"  # tick labels
GRID = "#E6E9ED"  # grid lines
# the blue family of key / exact recomputation: pale fill, soft tint, mid stroke, dark ink (text
# on a tint) and the saturated mark of data
ROLE_BLUE = {"fill": "#EAF1FB", "soft": "#D3E2F6", "stroke": "#5B8AC9", "ink": "#1F4E8C",
             "mark": "#2F6FC0"}  # fmt: skip

# figures that compare models: hue = the model (writer of programs or definition items, or
# reader), marker shape = the shift; a model without an entry is drawn in GREY under its own name
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


# ------------------------------------------------------------------------------ fonts, files
SERIF = ("Times New Roman", "STIXGeneral")  # the paper's Times, then matplotlib's own Times face


def serif_fonts() -> list[str]:
    """The faces of `SERIF` that are installed; STIXGeneral ships with matplotlib."""
    from matplotlib import font_manager

    names = {f.name for f in font_manager.fontManager.ttflist}
    return [n for n in SERIF if n in names] or ["STIXGeneral"]


@contextmanager
def printed(rc: Mapping[str, Any] | None = None) -> Iterator[Any]:
    """pyplot (Agg) with matplotlib's default style, the serif text face with STIX math,
    TrueType fonts embedded, and `rc` on top; the caller's settings come back on exit, so a
    figure does not depend on what was drawn before it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    params: dict[str, Any] = {
        "font.family": "serif",
        "font.serif": serif_fonts(),
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
    }
    params.update(rc or {})
    with matplotlib.rc_context():
        matplotlib.rcdefaults()
        current: Any = matplotlib.rcParams  # its keys are typed as literals
        current.update(params)
        yield pyplot


def save_pdf(fig: Any, path: Path) -> Path:
    """The figure as a PDF without creation date, creator or producer, closed after saving."""
    import matplotlib.pyplot as pyplot

    fig.savefig(path, metadata={"Creator": None, "Producer": None, "CreationDate": None})
    pyplot.close(fig)
    return path


def mix(c1: str, c2: str, t: float) -> str:
    """Linear RGB mix of two colours: t=0 is `c1`, t=1 is `c2`."""
    import numpy as np
    from matplotlib import colors

    a, b = np.array(colors.to_rgb(c1)), np.array(colors.to_rgb(c2))
    return str(colors.to_hex(a + (b - a) * t))


def tint(colour: str, a: float) -> str:
    """A hex colour mixed with white, `a` the share of the colour."""
    rgb = [int(colour[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(255 - a * (255 - c)):02X}" for c in rgb)


def advance(s: str, size: float, weight: str = "normal") -> float:
    """Advance width (mm) of `s` at `size` pt in the serif face, from the glyph outlines (exact,
    unlike a raster extent), so that the pieces of a small-capital word join without a gap; a
    string with a dollar sign is measured as math."""
    import matplotlib
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import text_to_path

    prop = FontProperties(family=matplotlib.rcParams["font.serif"], weight=weight, size=size)
    return float(text_to_path.get_text_width_height_descent(s, prop, ismath="$" in s)[0]) * PT


# --------------------------------------------------------------------- the millimetre canvas
class MmCanvas:
    """A figure drawn in millimetres at its printed size: one frameless axes over the whole figure
    with the origin at the bottom left; text sizes in points (default `size`)."""

    def __init__(self, plt: Any, width_mm: float, height_mm: float, size: float = 7.5) -> None:
        self.w, self.h, self.size = width_mm, height_mm, size
        self.fig = plt.figure(figsize=(width_mm * MM, height_mm * MM))
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set_xlim(0, width_mm)
        self.ax.set_ylim(0, height_mm)
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self._renderer: Any = None

    def text(
        self,
        x: float,
        y: float,
        s: str,
        size: float | None = None,
        color: str = INK,
        weight: str = "normal",
        ha: str = "center",
        va: str = "center",
        style: str = "normal",
        z: float = 20,
        **kw: Any,
    ) -> Any:
        return self.ax.text(x, y, s, fontsize=size or self.size, color=color, weight=weight,
                            ha=ha, va=va, style=style, zorder=z, **kw)  # fmt: skip

    def text_width(self, s: str, size: float, weight: str = "normal") -> float:
        """The rendered width (mm) of a text."""
        if self._renderer is None:
            self.fig.canvas.draw()
            self._renderer = self.fig.canvas.get_renderer()
        t = self.ax.text(0, 0, s, fontsize=size, weight=weight)
        bb = t.get_window_extent(self._renderer)
        t.remove()
        return float(bb.width / self.fig.dpi * 25.4)

    def label(self, x: float, y: float, letter: str, title: str, size: float = 8.0) -> None:
        """A panel label: the bold letter in parentheses, then the bold title, top-left anchored."""
        self.text(x, y, f"({letter})", size=size, weight="bold", ha="left", va="top")
        dx = self.text_width(f"({letter})", size, "bold") + 1.0
        self.text(x + dx, y, title, size=size, weight="bold", ha="left", va="top")

    def line(
        self,
        pts: Sequence[tuple[float, float]],
        color: str,
        lw: float = 0.6,
        radius: float = 1.0,
        z: float = 7,
    ) -> None:
        """A polyline with rounded corners (`radius` mm)."""
        from matplotlib.patches import PathPatch

        self.ax.add_patch(PathPatch(rounded_path(pts, radius), fc="none", ec=color, lw=lw,
                                    zorder=z, ls="-", alpha=1, capstyle="round",
                                    joinstyle="round"))  # fmt: skip


def rounded_path(pts: Sequence[tuple[float, float]], r: float) -> Any:
    """A polyline through `pts` whose corners are quadratic curves of radius `r`."""
    import numpy as np
    from matplotlib.path import Path as MplPath

    points = [tuple(p) for p in pts]
    if len(points) < 3 or r <= 0:
        return MplPath(points)
    verts, codes = [points[0]], [MplPath.MOVETO]
    for i in range(1, len(points) - 1):
        p0, p1, p2 = (np.array(points[j], float) for j in (i - 1, i, i + 1))
        d0, d1 = p1 - p0, p2 - p1
        l0, l1 = np.linalg.norm(d0), np.linalg.norm(d1)
        rr = min(r, l0 / 2, l1 / 2)
        a, b = p1 - d0 / l0 * rr, p1 + d1 / l1 * rr
        verts += [tuple(a), tuple(p1), tuple(b)]
        codes += [MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3]
    verts.append(points[-1])
    codes.append(MplPath.LINETO)
    return MplPath(verts, codes)


def chart_axes(ax: Any, size: float) -> Any:
    """Quiet chart axes: no top or right spine, short outward ticks, muted tick labels."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(SPINE)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(length=2.2, width=0.6, color=SPINE, labelsize=size - 0.5, labelcolor=TICK,
                   pad=1.5)  # fmt: skip
    return ax


# ---------------------------------------------------------------------------- bar figures
T_TITLE, T_LABEL, T_TICK, T_KEY, T_NOTE = 8.5, 8.0, 7.5, 7.8, 7.2  # pt at print
QUIET = "#6B7380"  # quiet text of the bar figures (shares, counts)
BAR_AXIS, BAR_GRID, BAR_ZERO = "#A3A9B1", "#E9EBEE", "#3F4650"  # value axis, grid, zero line
IV_LW, CAP_MM = 0.75, 1.9  # interval line (pt); cap width (mm at print)
BAR_RC = {"font.size": T_LABEL, "axes.linewidth": 0.6, "axes.titlecolor": INK,
          "axes.labelcolor": INK}  # fmt: skip


def bar_axes(ax: Any) -> None:
    """Value axis on the left with a light grid and a dark zero line; no other spine."""
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(BAR_AXIS)
    ax.spines["left"].set_linewidth(0.6)
    ax.tick_params(axis="y", length=2.2, width=0.6, color=BAR_AXIS, labelsize=T_TICK,
                   labelcolor=TICK, pad=1.6)  # fmt: skip
    ax.tick_params(axis="x", length=0, labelsize=T_LABEL, labelcolor=INK, pad=2.4)
    ax.grid(axis="y", color=BAR_GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(0, color=BAR_ZERO, lw=0.6, zorder=2.6)


def bar_from_zero(ax: Any, x: float, w: float, est: float, fill: str, z: float = 2.0) -> Any:
    """A vertical bar from zero to `est`, `w` data units wide, centred on `x`."""
    from matplotlib.patches import Rectangle

    return ax.add_patch(Rectangle((x - w / 2, 0), w, est, fc=fill, ec="none", lw=0, zorder=z))


def x_per_mm(ax: Any, panel_mm: float) -> float:
    """Data units of the x axis per printed mm, for a panel `panel_mm` wide."""
    x0, x1 = ax.get_xlim()
    return float((x1 - x0) / panel_mm)


def interval_line(
    ax: Any, x: float, lo: float, hi: float, ink: str, cap: float, lw: float = IV_LW,
    z: float = 4.0,
) -> list[Any]:  # fmt: skip
    """A 95% interval at `x`: a thin vertical line with light caps (`cap`: their width in data
    units) in its mark's own dark ink. Returns its lines."""
    lines: list[Any] = ax.plot([x, x], [lo, hi], color=ink, lw=lw, solid_capstyle="butt",
                               zorder=z)  # fmt: skip
    for y in (lo, hi):
        lines += ax.plot([x - cap / 2, x + cap / 2], [y, y], color=ink, lw=lw,
                         solid_capstyle="butt", zorder=z)  # fmt: skip
    return lines


Glyph = Callable[[Any, float, float], None]


def key_row(
    fig: Any,
    entries: Sequence[tuple[Glyph | None, str]],
    y_mm: float,
    w_mm: float,
    h_mm: float,
    gap: float = 1.1,
    sep: float = 4.6,
    breaks: Sequence[int] = (),
) -> Any:
    """One compact key row centred at the top: entries are (glyph, label), where glyph(ov, x, y)
    draws a 3.0 x 2.3 mm glyph whose left-centre is (x, y), in mm on an overlay axes (None: text
    only); the entries whose index is in `breaks` start a second group, 3 mm further on."""
    ov = fig.add_axes((0, 0, 1, 1), zorder=10)
    ov.set_xlim(0, w_mm)
    ov.set_ylim(0, h_mm)
    ov.axis("off")
    r = fig.canvas.get_renderer()
    texts = [ov.text(0, y_mm, lab, fontsize=T_KEY, color=INK, va="center", ha="left")
             for _, lab in entries]  # fmt: skip
    widths = [t.get_window_extent(r).width / fig.dpi * 25.4 for t in texts]
    pads = [3.0 * (n in breaks) for n in range(len(entries))]
    leads = [0.0 if g is None else 3.0 + gap for g, _ in entries]
    total = sum(pads) + sum(leads) + sum(widths) + sep * (len(entries) - 1)
    x = (w_mm - total) / 2
    for (glyph, _), t, w, pad, lead in zip(entries, texts, widths, pads, leads, strict=True):
        x += pad
        if glyph:
            glyph(ov, x, y_mm)
        t.set_x(x + lead)
        x += lead + w + sep
    return ov


def key_swatch(fill: str, top: str | None = None) -> Glyph:
    """Key glyph of a bar: a filled block, with its top edge line if the bars have one."""

    def draw(ov: Any, x: float, y: float) -> None:
        from matplotlib.patches import Rectangle

        ov.add_patch(Rectangle((x, y - 1.15), 3.0, 2.3, fc=fill, ec="none"))
        if top:
            ov.plot([x, x + 3.0], [y + 1.15] * 2, color=top, lw=0.9, solid_capstyle="butt")

    return draw
