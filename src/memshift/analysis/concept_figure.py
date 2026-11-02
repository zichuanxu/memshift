"""Figure 1 of the paper (fig_concept): one item and its five shifts (a), and the 2 x 2 grid of
HoldMem cells (D, B) after a redefinition (b), detection by repair, regenerated from the staged
analyses named in results/sources.json (nothing here calls a model or reads a raw run).

(a) is a drawing: the calendar records, the definition (v1, v2), the writer and the memory item,
each shift attached to what it changes. (b) gives per cell its detection and repair steps, the
lists it serves and a bar that splits the changed definition items as Eq. (1) does: right (in the
colour of the repair), wrong after a re-derivation (red tint) and wrong after a missed change
(amber tint), with the 95% interval of Update. Between the rows, the key's effect on Update at
each D and their difference, B x D.

Data (`concept_cells`): m, r, a and Update of the B=1 cells (source key, detection counted from
the latest change) and Update of the B=0 cells, on the changed definition items after a
redefinition in the main study (ex_round4_detection event.json, whose macros the paper prints as
ExDetMainDefDefinition...); the key's effect at D=0 and D=1 and B x D on the same item-times (e1
class split, the macros ExClsMainDefDefinitionBDzero, ...BDone and ExAcctMainDefDefinitionBxD).
Every printed number is formatted as its macro is. Without a stage the figure leaves out what it
would draw from it.

Icons: the raster icons of paper/figures/icons/ (`load` names the directory). The key is recoloured
to amber for the no-key tile and the writer to a darker slate; an icon that is not there is left
out, so the figure also draws without paper/.

Usage: python -m memshift.analysis.figures --only fig_concept
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from memshift.analysis import figstyle as fs
from memshift.analysis.paper_data import get as _get
from memshift.analysis.paper_data import stage_json
from memshift.analysis.paper_numbers import pct, pp

CONCEPT_PT = (486, 161.6)  # printed size (pt): full width
ICONS = "paper/figures/icons"  # the icon set, relative to the repository root
STAGE_KEY = "ex_round4_detection"  # event.json: m, r, a and Update per cell

# palette: one meaning per role (fill, soft, stroke, ink, mark)
INK, MUTED = fs.INK, fs.MUTED
LINE = "#4A4F57"  # connectors
NEU = {"fill": "#F4F5F7", "soft": "#E6E8EC", "stroke": "#8C95A1", "ink": "#3A4250",
       "mark": "#5E6875"}  # the item and the shifts  # fmt: skip
AMB = {"fill": "#FFF5E6", "soft": "#FCE3BD", "stroke": "#D9A04A", "ink": "#7A4A0E",
       "mark": "#E69F00"}  # no key, B=0  # fmt: skip
KEY = {"fill": "#E9F3FA", "soft": "#D2E7F5", "stroke": "#4A9AD0", "ink": "#0B527F",
       "mark": "#0072B2"}  # the key, B=1  # fmt: skip
EXA = fs.ROLE_BLUE  # exact recomputation, D=1
SLA = {"fill": "#F3F4F6", "soft": "#E3E6EA", "stroke": "#9AA3AF", "ink": "#3F4753",
       "mark": "#9AA3AF"}  # the writer re-derives, D=0  # fmt: skip
PLUM = {"ink": "#4F1838", "mark": "#962F6E"}  # B x D
RED = "#C0392B"  # a step that fails
WRONG = {"fill": "#F6D2CD", "ink": "#9C2A1F"}  # bar: wrong after a re-derivation
MISS = {"fill": "#FCE3BD", "ink": "#7A4A0E"}  # bar: wrong after a missed change
HEAD_D = {0: {**SLA, "solid": "#6F7986"}, 1: {**EXA, "solid": EXA["mark"]}}  # column headers
CARD_EDGE, CARD_BAND, ZONE_EDGE = "#AEB5BF", "#ECEEF1", "#B8BFC9"

# type sizes at print (pt): titles, axis headers, labels, small text, Update
S_TITLE, S_HEAD, S_LAB, S_SMALL, S_NUM = 8.5, 7.6, 7.0, 6.6, 10.0
BAND = 5.0  # height of a card's header band (mm)
PANEL_A_W = 43.0  # width of panel (a) (mm)
TILE = 6.6  # a detection or repair tile (mm)
CELL_H = 17.4  # a cell of (b) (mm)
RASTER_DPI = 1200  # icons are resampled once to their printed size at this resolution


# ===================================================================================== data
def load(root: Path, sources: Mapping[str, str]) -> dict[str, Any]:
    """The staged inputs (None where a stage is not there) and the icon directory."""
    src = dict(sources)
    return {
        "event": stage_json(root, src, STAGE_KEY, "event.json"),
        "e1": stage_json(root, src, "e1", "e1_metrics.json"),
        "icons": root / ICONS,
    }


def concept_cells(inp: Mapping[str, Any]) -> dict[str, Any]:
    """The data of `fig_concept`, on the changed definition items after a redefinition in the
    main study. `cells`: per staged cell (D, B) Update (estimate and 95% interval) and the terms
    of Eq. (1), m, r and a (B=1: source key, detection counted from the latest change; B=0: m = 1,
    no r, a = Update). `key`: per D the key's effect on Update; `bxd`: B x D (e1 class split).
    `items`: the changed item-times; `worlds`: the test worlds. ValueError when a B=1 cell's
    terms do not give its Update by Eq. (1)."""
    main = _get(inp.get("event"), "main") or {}
    terms = _get(main, "terms", "def", "definition") or {}
    cells: dict[tuple[int, int], dict[str, Any]] = {}
    items = None
    for d in (0, 1):
        u = _est(_get(terms, "b0", f"D{d}B0"))
        if u:
            cells[d, 0] = {"U": u, "m": 1.0, "r": None, "a": u["est"]}
        lc = _get(terms, "source", "cells", f"D{d}B1", "latest_change") or {}
        u = _est(lc.get("U"))
        if u and all(_est(lc.get(k)) for k in "mra"):
            m, r, a = (float(lc[k]["est"]) for k in "mra")
            if abs((1 - m) * r + m * a - u["est"]) > 1e-9:
                raise ValueError(f"cell ({d},1): Update is not (1 - m) r + m a: {lc}")
            cells[d, 1] = {"U": u, "m": m, "r": r, "a": a}
            items = lc.get("n", items)
    split = _get(inp.get("e1"), "explore_class_split", "D1", "def", "source", "definition") or {}
    return {
        "cells": cells,
        "key": {d: _est(split.get(f"B_effect_update_D{d}")) for d in (0, 1)},
        "bxd": _est(split.get("BxD_interaction_update")),
        "items": items,
        "worlds": main.get("n_worlds"),
    }


def _est(e: Any) -> dict[str, float] | None:
    """`{est, lo, hi}` of an estimate with a finite value, or None."""
    if not isinstance(e, Mapping) or not isinstance(e.get("est"), int | float):
        return None
    if not math.isfinite(e["est"]):
        return None
    return {k: float(e.get(k, e["est"])) for k in ("est", "lo", "hi")}


def _printed(s: str) -> str:
    """A macro's text as the figure prints it: signs and units in Unicode, a thin space before
    pp."""
    return (s.replace("$+$", "+").replace("$-$", "−").replace("\\%", "%")
            .replace("\\,pp", "\u2009pp"))  # fmt: skip


# ==================================================================================== icons
def _recolour(im: Any, dark: str, light: str) -> Any:
    """An icon's lightness mapped onto a two-stop ramp of one role (dark outline, light fill)."""
    import numpy as np
    from matplotlib.colors import to_rgb
    from PIL import Image

    a = np.asarray(im.convert("RGBA")).astype(float) / 255
    lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    lo, hi = np.percentile(lum[a[..., 3] > 0.5], [3, 97])
    t = np.clip((lum - lo) / max(hi - lo, 1e-6), 0, 1)[..., None]
    c0, c1 = np.array(to_rgb(dark)), np.array(to_rgb(light))
    out = np.dstack([c0 + (c1 - c0) * t, a[..., 3]])
    return Image.fromarray((out * 255).round().astype(np.uint8), "RGBA")


def _icons(folder: Path | None) -> dict[str, Any]:
    """The icons of the figure by role, as RGBA images; an icon whose file is missing is left
    out."""
    from PIL import Image

    files = {"key": "key-blue", "robot": "writer-slate", "calc": "recompute-darkblue",
             "contacts": "contacts-grey", "revision": "revision-grey",
             "deletion": "deletion-grey", "redef_w": "redefinition-white",
             "switch": "switch-grey", "export": "export-grey"}  # fmt: skip
    out: dict[str, Any] = {}
    for role, name in files.items():
        path = None if folder is None else Path(folder) / f"{name}.png"
        if path is not None and path.is_file():
            with Image.open(path) as im:
                out[role] = im.convert("RGBA")
    if "key" in out:
        out["key_amber"] = _recolour(out["key"], "#9A6A1E", "#FBE7C6")
    if "robot" in out:
        out["robot"] = _recolour(out["robot"], "#3F4753", "#D9DDE3")
    return out


def _image(cv: fs.MmCanvas, im: Any, cx: float, cy: float, w: float, z: float) -> None:
    """An icon centred at (cx, cy) whose longer side is `w` mm: cropped to its drawing, resampled
    once (Lanczos and a light unsharp mask) to its printed size at `RASTER_DPI`, its white
    surround made transparent, and placed on the pixel grid of that resolution, so that it is
    never resampled again."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter

    im = Image.alpha_composite(Image.new("RGBA", im.size, "white"), im).convert("RGB")
    box = im.convert("L").point(lambda v: 255 if v < 245 else 0).getbbox()
    if box:
        pad = 8
        im = im.crop((max(box[0] - pad, 0), max(box[1] - pad, 0),
                      min(box[2] + pad, im.width), min(box[3] + pad, im.height)))  # fmt: skip
    ppm = RASTER_DPI / 25.4  # pixels per mm
    tall = im.height > im.width
    tw = max(8, round((w * im.width / im.height if tall else w) * ppm))
    th = max(8, round(tw * im.height / im.width))
    im = im.resize((tw, th), Image.Resampling.LANCZOS).filter(
        ImageFilter.UnsharpMask(radius=1.0, percent=60, threshold=2))  # fmt: skip
    a = np.asarray(im).astype(float) / 255.0
    # the white connected to the border becomes transparent, with a soft rim of one pixel
    m = Image.new("L", (tw + 2, th + 2), 255)
    m.paste(Image.fromarray(((a.min(axis=2) > 0.975) * 255).astype(np.uint8)), (1, 1))
    ImageDraw.floodfill(m, (0, 0), 128)
    outside = np.asarray(m)[1:-1, 1:-1] == 128
    rim = np.asarray(Image.fromarray((outside * 255).astype(np.uint8))
                     .filter(ImageFilter.MaxFilter(3))) > 0  # fmt: skip
    soft = np.clip((1.0 - a.min(axis=2)) / 0.25, 0.0, 1.0)
    alpha = np.where(outside, 0.0, np.where(rim, soft, 1.0))
    a = np.clip((a - (1.0 - alpha[..., None])) / np.maximum(alpha, 1e-3)[..., None], 0.0, 1.0)
    x0, y0 = round((cx - tw / ppm / 2) * ppm) / ppm, round((cy - th / ppm / 2) * ppm) / ppm
    cv.ax.imshow(np.dstack([a, alpha]), extent=(x0, x0 + tw / ppm, y0, y0 + th / ppm), zorder=z,
                 interpolation="none")  # fmt: skip
    cv.ax.set_xlim(0, cv.w)
    cv.ax.set_ylim(0, cv.h)


def _icon(cv: fs.MmCanvas, ic: Mapping[str, Any], role: str, cx: float, cy: float, w: float,
          z: float) -> None:  # fmt: skip
    if role in ic:
        _image(cv, ic[role], cx, cy, w, z)


# ================================================================================ primitives
def _rbox(cv: fs.MmCanvas, x: float, y: float, w: float, h: float, fc: str, ec: str,
          lw: float = 0.6, r: float = 1.2, z: float = 5, ls: Any = "-",
          alpha: float = 1.0) -> None:  # fmt: skip
    from matplotlib.patches import FancyBboxPatch

    cv.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                   fc=fc, ec=ec, lw=lw, zorder=z, ls=ls,
                                   alpha=alpha))  # fmt: skip


def _round_top(x: float, y: float, w: float, h: float, r: float) -> Any:
    """A rectangle whose top corners are rounded (radius r): a card's header band."""
    from matplotlib.path import Path as MplPath

    r = min(r, h)
    v = [(x, y), (x + w, y), (x + w, y + h - r), (x + w, y + h), (x + w - r, y + h),
         (x + r, y + h), (x, y + h), (x, y + h - r), (x, y)]  # fmt: skip
    c = [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3,
         MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.CLOSEPOLY]  # fmt: skip
    return MplPath(v, c)


def _arrow(cv: fs.MmCanvas, pts: Sequence[tuple[float, float]], color: str = LINE,
           lw: float = 0.7, head: float = 1.4, z: float = 8) -> None:  # fmt: skip
    """A straight or elbowed arrow with a filled head `head` mm long, ending on its last point."""
    from matplotlib.patches import FancyArrowPatch

    style = f"-|>,head_length={head * 0.95},head_width={head * 0.42}"
    cv.ax.add_patch(FancyArrowPatch(path=fs.rounded_path(pts, 1.2), arrowstyle=style,
                                    mutation_scale=72 / 25.4, lw=lw, color=color, zorder=z,
                                    linestyle="-", shrinkA=0, shrinkB=0, joinstyle="miter",
                                    capstyle="butt"))  # fmt: skip


def _check(cv: fs.MmCanvas, x: float, y: float, s: float, color: str, lw: float,
           z: float) -> None:  # fmt: skip
    cv.ax.plot([x - s * 0.45, x - s * 0.1, x + s * 0.5], [y, y - s * 0.38, y + s * 0.42],
               color=color, lw=lw, zorder=z, solid_capstyle="round",
               solid_joinstyle="round")  # fmt: skip


def _cross(cv: fs.MmCanvas, x: float, y: float, s: float, color: str, lw: float,
           z: float) -> None:  # fmt: skip
    for sg in (1, -1):
        cv.ax.plot([x - s * 0.4, x + s * 0.4], [y - sg * s * 0.4, y + sg * s * 0.4], color=color,
                   lw=lw, zorder=z, solid_capstyle="round")  # fmt: skip


def _card(cv: fs.MmCanvas, x: float, y: float, w: float, h: float, title: str,
          sub: str | None = None) -> None:  # fmt: skip
    """A panel card: white body, neutral outline, a tinted header band with the bold title."""
    from matplotlib.patches import PathPatch

    _rbox(cv, x, y, w, h, "white", CARD_EDGE, lw=0.7, r=2.0, z=1)
    cv.ax.add_patch(PathPatch(_round_top(x, y + h - BAND, w, BAND, 2.0), fc=CARD_BAND, ec="none",
                              zorder=1.1))  # fmt: skip
    cv.ax.plot([x, x + w], [y + h - BAND] * 2, color=CARD_EDGE, lw=0.5, zorder=1.2)
    cv.text(x + 2.2, y + h - BAND / 2, title, size=S_TITLE, weight="bold", ha="left", z=3)
    if sub:
        cv.text(x + w - 2.2, y + h - BAND / 2, sub, size=S_SMALL, style="italic", color=MUTED,
                ha="right", z=3)  # fmt: skip


def _badge(cv: fs.MmCanvas, x: float, y: float, ok: bool, d: float = 2.5) -> None:
    """A step's verdict on a tile's corner: a check in ink on white, or a white cross on red."""
    from matplotlib.patches import Circle

    if ok:
        cv.ax.add_patch(Circle((x, y), d / 2, fc="white", ec=INK, lw=0.6, zorder=14))
        _check(cv, x + 0.02, y + 0.02, d * 0.52, INK, 0.75, 15)
    else:
        cv.ax.add_patch(Circle((x, y), d / 2, fc=RED, ec="white", lw=0.5, zorder=14))
        _cross(cv, x, y, d * 0.5, "white", 0.8, 15)


def _record(cv: fs.MmCanvas, x: float, y: float, w: float, h: float, ec: str = "#6F7986",
            ls: Any = "-", alpha: float = 1.0, edit: bool = False) -> None:  # fmt: skip
    """A calendar record: a page with a darker binding band; `edit` adds a pencil (revised)."""
    import numpy as np
    from matplotlib.patches import FancyBboxPatch, PathPatch, Polygon

    r, z = min(w, h) * 0.18, 8
    cv.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                   fc="white", ec=ec, lw=0.45, zorder=z, ls=ls,
                                   alpha=alpha))  # fmt: skip
    cv.ax.add_patch(PathPatch(_round_top(x, y + h * 0.70, w, h * 0.30, r), fc=ec, ec="none",
                              zorder=z + 0.1, alpha=alpha))  # fmt: skip
    for i in (1, 2):
        yy = y + h * (0.70 - 0.22 * i)
        cv.ax.plot([x + w * 0.2, x + w * 0.8], [yy, yy], color=ec, lw=0.35, zorder=z + 0.1,
                   alpha=alpha * 0.8)  # fmt: skip
    if edit:
        length, wd = min(w, h) * 1.05, min(w, h) * 0.24
        tip = np.array([x + w * 0.28, y + h * 0.12])
        u = np.array([math.cos(math.radians(50)), math.sin(math.radians(50))])
        nrm = np.array([-u[1], u[0]])
        a0 = tip + u * wd * 1.1
        body = [a0 + nrm * wd / 2, a0 + u * length + nrm * wd / 2, a0 + u * length - nrm * wd / 2,
                a0 - nrm * wd / 2]  # fmt: skip
        cv.ax.add_patch(Polygon(body, closed=True, fc="#AEB5BF", ec=INK, lw=0.35, zorder=z + 0.4))
        cv.ax.add_patch(Polygon([tip, a0 + nrm * wd / 2, a0 - nrm * wd / 2], closed=True,
                                fc="white", ec=INK, lw=0.35, zorder=z + 0.4))  # fmt: skip


def _names(cv: fs.MmCanvas, x: float, y: float, names: Sequence[str], role: Mapping[str, str],
           verdict: bool | None = None) -> float:  # fmt: skip
    """A served list: one pill per name in its producer's role colour, then a bare check or
    cross. Returns the right end."""
    from matplotlib.patches import FancyBboxPatch

    h = 2.8
    for s in names:
        w = cv.text_width(s, S_SMALL) + 2 * 0.75
        cv.ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h,
                                       boxstyle=f"round,pad=0,rounding_size={h / 2}", lw=0.5,
                                       ec=role["stroke"], fc=role["soft"], zorder=9))  # fmt: skip
        cv.text(x + w / 2, y, s, size=S_SMALL, color=role["ink"], z=10)
        x += w + 0.5
    if verdict is not None:
        d = 2.0
        vx = x + d / 2 + 0.3
        if verdict:
            _check(cv, vx, y + 0.05, d * 0.95, role["ink"], 1.0, 12)
        else:
            _cross(cv, vx, y, d * 0.85, RED, 1.0, 12)
        x += d + 0.6
    return x


def _pill(cv: fs.MmCanvas, ic: Mapping[str, Any], x: float, y: float, text: str, icon: str,
          solid: bool = False, ha: str = "left", z: float = 9) -> None:  # fmt: skip
    """A shift's badge: its icon and its name in a rounded pill (solid: the active shift)."""
    h = 2.9
    iw = h * 0.78
    w = 0.8 + iw + 0.6 + cv.text_width(text, S_SMALL) + 0.9
    x0 = x if ha == "left" else x - w
    _rbox(cv, x0, y - h / 2, w, h, NEU["ink"] if solid else "white",
          NEU["ink"] if solid else NEU["stroke"], lw=0.5, r=h / 2, z=z)  # fmt: skip
    _icon(cv, ic, icon, x0 + 0.8 + iw / 2, y, iw, z + 1)
    cv.text(x0 + 0.8 + iw + 0.6, y - 0.05, text, size=S_SMALL, ha="left", z=z + 1,
            color="white" if solid else NEU["ink"],
            weight="bold" if solid else "normal")  # fmt: skip


# =================================================================================== panel (a)
def _panel_items(cv: fs.MmCanvas, ic: Mapping[str, Any], x: float, y: float, w: float,
                 h: float) -> None:  # fmt: skip
    """(a): the records and the two shifts that change them; the definition (redefinition as the
    active tab), the writer (writer switch) and the memory item (export), which add no text."""
    _card(cv, x, y, w, h, "(a) One item, five shifts")
    top = y + h - BAND
    left, right = x + 2.0, x + w - 2.0
    dashed = (0, (3, 2))
    # the records, and the two shifts that change them
    z1b, z1t = top - 13.6, top - 1.2
    _rbox(cv, left - 0.6, z1b, right - left + 1.2, z1t - z1b, NEU["fill"], ZONE_EDGE, lw=0.5,
          r=1.6, z=2, ls=dashed)  # fmt: skip
    cv.text(left + 0.6, z1t - 1.7, "changes records", size=S_SMALL, style="italic", color=MUTED,
            ha="left", z=4)  # fmt: skip
    cv.text(right - 0.6, z1t - 1.7, "calendar, 12 months", size=S_SMALL, color=MUTED, ha="right",
            z=4)  # fmt: skip
    n, tw, th = 12, 2.6, 3.2
    pitch = (right - left - 1.2 - tw) / (n - 1)
    ty = z1t - 7.2
    tops = []
    for k in range(n):
        tx = left + 0.6 + k * pitch
        if k == 10:  # the deleted record
            _record(cv, tx, ty, tw, th, ec="#A9B1BB", ls=(0, (1.6, 1.0)), alpha=0.55)
        else:
            _record(cv, tx, ty, tw, th, ec=NEU["ink"] if k == 3 else "#7D8794", edit=k == 3)
        tops.append(tx + tw / 2)
    py = z1b + 2.0
    for k, label, icon in ((3, "source revision", "revision"), (10, "deletion", "deletion")):
        cx = tops[k]
        width = 0.8 + 2.9 * 0.78 + 0.6 + cv.text_width(label, S_SMALL) + 0.9
        _pill(cv, ic, min(max(cx - width / 2, left + 3.0), right - width), py, label, icon)
        cv.ax.plot([cx, cx], [py + 1.45, ty - 0.05], color=NEU["stroke"], lw=0.5, zorder=7)
    # the definition, the writer and the memory item: these shifts add no text
    z2b, z2t = y + 1.2, z1b - 1.1
    _rbox(cv, left - 0.6, z2b, right - left + 1.2, z2t - z2b, "white", ZONE_EDGE, lw=0.5, r=1.6,
          z=2, ls=dashed)  # fmt: skip
    rec_x = tops[0]
    cv.text(rec_x + 1.6, z2t - 1.7, "adds no text", size=S_SMALL, style="italic", color=MUTED,
            ha="left", z=4)  # fmt: skip
    dx0, dx1 = rec_x + 1.9, right - 0.6
    dy1 = z2t - 4.3
    dy0 = dy1 - 13.0
    _rbox(cv, dx0, dy0, dx1 - dx0, dy1 - dy0, "white", NEU["stroke"], lw=0.6, r=1.3, z=5)
    _pill(cv, ic, dx1 - 1.0, dy1, "redefinition", "redef_w", solid=True, ha="right", z=10)
    cv.text(dx0 + 1.3, dy1 - 2.3, "definition", size=S_SMALL, weight="bold", color=NEU["ink"],
            ha="left", z=7)  # fmt: skip
    vy1, vy2 = dy1 - 5.3, dy0 + 2.1
    for vy, tag, solid in ((vy1, "v1", False), (vy2, "v2", True)):
        _rbox(cv, dx0 + 1.3, vy - 1.3, 4.0, 2.6, NEU["ink"] if solid else "white", NEU["ink"],
              lw=0.5, r=0.5, z=7)  # fmt: skip
        cv.text(dx0 + 3.3, vy - 0.05, tag, size=S_SMALL, weight="bold",
                color="white" if solid else NEU["ink"], z=8)  # fmt: skip
    cv.text(dx0 + 6.2, vy1, r"$\geq n$ one-off meetings", size=S_SMALL, ha="left", z=7)
    cv.text(dx0 + 6.2, vy1 - 2.8, r"in the last $w$ days", size=S_SMALL, ha="left", z=7)
    cv.text(dx0 + 6.2, vy2, r"smaller $n$ and $w$", size=S_SMALL, ha="left", weight="bold", z=7)
    # the writer: records arrive from above, the definition from its card
    ww, wh = 9.4, 10.6
    wx0 = left + 0.6
    wy1 = dy0 - 3.0
    wy0 = wy1 - wh
    _rbox(cv, wx0, wy0, ww, wh, SLA["fill"], SLA["stroke"], lw=0.7, r=1.4, z=6)
    _icon(cv, ic, "robot", wx0 + ww / 2, wy0 + wh / 2 + 1.3, 6.0, 8)
    cv.text(wx0 + ww / 2, wy0 + 1.7, "writer", size=S_SMALL, color=SLA["ink"], z=8)
    _arrow(cv, [(rec_x, ty - 0.15), (rec_x, wy1)], lw=0.6, head=1.2)
    dfx = wx0 + ww - 2.2
    _arrow(cv, [(dfx, dy0), (dfx, wy1)], lw=0.6, head=1.2)
    sy = z2b + 2.2
    _pill(cv, ic, wx0, sy, "writer switch", "switch")
    # the memory item, right of the writer, with export on its lower edge
    ix0, ix1 = wx0 + ww + 3.4, right - 0.6
    iy1 = wy1 + 0.2
    iy0 = sy + 1.45 + 1.2
    _rbox(cv, ix0, iy0, ix1 - ix0, iy1 - iy0, NEU["fill"], NEU["stroke"], lw=0.7, r=1.4, z=5)
    _arrow(cv, [(wx0 + ww, wy0 + wh / 2), (ix0, wy0 + wh / 2)], lw=0.6, head=1.2)
    hy = iy1 - 2.4
    _icon(cv, ic, "contacts", ix0 + 2.5, hy, 3.0, 8)
    cv.text(ix0 + 4.6, hy, "important contacts", size=S_SMALL, weight="bold", color=NEU["ink"],
            ha="left", z=7)  # fmt: skip
    _names(cv, ix0 + 1.3, hy - 3.9, ["Ana", "Ben", "Chloé"], NEU)
    _pill(cv, ic, ix1 - 1.0, iy0, "export", "export", ha="right", z=10)


# =================================================================================== panel (b)
def _bar(cv: fs.MmCanvas, x0: float, x1: float, yc: float, h: float, cell: Mapping[str, Any],
         d: int) -> None:  # fmt: skip
    """A cell's outcome bar, the changed items as Eq. (1) splits them: detected (1 - m), then
    missed (m); the right share in the colour of the repair, wrong after a re-derivation in a red
    tint and wrong after a miss in an amber tint (each named where it fits); a thin whisker gives
    the 95% interval of Update."""
    from matplotlib.patches import Rectangle

    span = x1 - x0
    m, r, a = cell["m"], cell["r"], cell["a"]
    det = 1 - m
    parts = [("right", det * r), ("wrong", det * (1 - r))] if det > 0 else []
    parts += [("right", m * a), ("missed", m * (1 - a))]
    fill = {"right": EXA["mark"] if d else SLA["mark"], "wrong": WRONG["fill"],
            "missed": MISS["fill"]}  # fmt: skip
    xx, segs = x0, {}
    for kind, share in parts:
        if share <= 0:
            continue
        cv.ax.add_patch(Rectangle((xx, yc - h / 2), share * span, h, fc=fill[kind], ec="none",
                                  zorder=6))  # fmt: skip
        segs[kind] = (xx, xx + share * span)
        xx += share * span
    cv.ax.add_patch(Rectangle((x0, yc - h / 2), span, h, fc="none", ec="#8E99A8", lw=0.5,
                              zorder=7))  # fmt: skip
    for kind, txt, col in (("wrong", "re-derived wrong", WRONG["ink"]),
                           ("missed", "missed", MISS["ink"])):  # fmt: skip
        if kind in segs:
            s0, s1 = segs[kind]
            if s1 - s0 > cv.text_width(txt, S_SMALL) + 1.2:
                nudge = 0.8 if kind == "missed" and d == 0 else 0  # clear of the whisker
                cv.text((s0 + s1) / 2 + nudge, yc - 0.05, txt, size=S_SMALL, style="italic",
                        color=col, z=9)  # fmt: skip
    u = cell["U"]
    lo, hi = x0 + u["lo"] * span, x0 + u["hi"] * span
    ink = "#14335C" if d == 1 else "#2E343D"
    cv.ax.plot([lo, hi], [yc, yc], color=ink, lw=0.7, zorder=8, solid_capstyle="butt")
    for xv in (lo, hi):
        cv.ax.plot([xv, xv], [yc - 1.0, yc + 1.0], color=ink, lw=0.7, zorder=8,
                   solid_capstyle="butt")  # fmt: skip


def _cell(cv: fs.MmCanvas, ic: Mapping[str, Any], data: Mapping[str, Any], d: int, b: int,
          x0: float, top: float, w: float, h: float) -> dict[str, float]:  # fmt: skip
    """A HoldMem cell (D, B): its detection tile, what triggers repair, its repair tile, the lists
    it serves, the step measures m and r, and its outcome bar with its Update."""
    from matplotlib.patches import Circle, FancyBboxPatch

    hero = (d, b) == (1, 1)
    cell = data["cells"].get((d, b))
    if hero:  # the hero's shadow
        cv.ax.add_patch(FancyBboxPatch((x0 + 0.35, top - h - 0.35), w, h,
                                       boxstyle="round,pad=0,rounding_size=1.6", lw=0,
                                       fc="#1A2230", alpha=0.10, zorder=2.8))  # fmt: skip
    _rbox(cv, x0, top - h, w, h, "#F5F9FE" if hero else "white",
          EXA["mark"] if hero else CARD_EDGE, lw=1.1 if hero else 0.6, r=1.6, z=3)  # fmt: skip
    ts = TILE
    yc = top - 1.5 - ts / 2
    dx, rx = x0 + 2.0, x0 + 14.6
    # detection: the key (B=1) or no key (B=0)
    role_b = KEY if b else AMB
    _rbox(cv, dx, yc - ts / 2, ts, ts, role_b["fill"], role_b["stroke"], lw=0.7, r=1.3, z=6)
    if b:
        _icon(cv, ic, "key", dx + ts / 2, yc, ts * 0.72, 8)
    else:
        _icon(cv, ic, "key_amber", dx + ts / 2, yc, ts * 0.62, 8)
        rr, t = ts * 0.36, math.radians(135)
        cv.ax.add_patch(Circle((dx + ts / 2, yc), rr, fc="none", ec=AMB["ink"], lw=0.9, zorder=9))
        cv.ax.plot([dx + ts / 2 + rr * math.cos(t), dx + ts / 2 - rr * math.cos(t)],
                   [yc + rr * math.sin(t), yc - rr * math.sin(t)], color=AMB["ink"], lw=0.9,
                   zorder=9)  # fmt: skip
    _badge(cv, dx + ts - 0.2, yc + ts / 2 - 0.2, bool(b))
    # repair: the writer (D=0) or exact recomputation (D=1); without a key nothing is
    # invalidated, so no repair of Eq. (1) runs: a dashed tile without a badge, which derives a
    # fresh item only when a new record arrives
    role_d = EXA if d else SLA
    _rbox(cv, rx, yc - ts / 2, ts, ts, role_d["fill"], role_d["stroke"], lw=0.9 if d else 0.7,
          r=1.3, z=6, ls="-" if b else (0, (2.4, 1.3)))  # fmt: skip
    _icon(cv, ic, "calc" if d else "robot", rx + ts / 2, yc + 0.1, ts * 0.68, 8)
    if b:
        _badge(cv, rx + ts - 0.2, yc + ts / 2 - 0.2, bool(d))
    # what triggers repair: the key invalidates (B=1), or a new record arrives (B=0)
    if b:
        _arrow(cv, [(dx + ts, yc), (rx, yc)], color=KEY["mark"], lw=0.8, head=1.3)
    else:
        gw, gh = 2.6, 3.1
        gx = dx + ts + 0.6
        _record(cv, gx, yc - gh / 2, gw, gh, ec="#7D8794")
        _arrow(cv, [(gx + gw + 0.15, yc), (rx, yc)], lw=0.6, head=1.1)
    # the lists it serves: the re-derived item (B=1), or the old list kept and the fresh item
    # derived on a new record (B=0)
    sx = rx + ts + 3.0
    fresh = ["Ana", "Dev"] if d else ["Ana", "Ben", "Dev"]
    ink = role_d["mark"] if d else role_d["ink"]
    if b:
        _arrow(cv, [(rx + ts, yc), (sx - 0.35, yc)], color=ink, lw=0.7, head=1.2)
        _names(cv, sx, yc, fresh, role_d, verdict=bool(d))
    else:
        y_old, y_new = yc + 1.75, yc - 1.75
        x_old = _names(cv, sx, y_old, ["Ana", "Ben", "Chloé"], AMB, verdict=False)
        _arrow(cv, [(rx + ts, y_new), (sx - 0.35, y_new)], color=ink, lw=0.7, head=1.2)
        x_new = _names(cv, sx, y_new, fresh, role_d, verdict=bool(d))
        for yy, word, xe in ((y_old, "old", x_old), (y_new, "fresh", x_new)):
            cv.text(xe + 0.5, yy - 0.1, word, size=S_SMALL, style="italic", color=MUTED,
                    ha="left", z=7)  # fmt: skip
    # the step measures under the tiles
    ly = yc - ts / 2 - 1.75
    if not b:
        cv.text(dx + ts / 2, ly, r"$m$ = 1", size=S_SMALL, color=role_b["ink"], z=7)
        cv.text(dx + ts + 0.7, ly, "on a new record", size=S_SMALL, color=MUTED, style="italic",
                ha="left", z=7)  # fmt: skip
    elif cell:
        cv.text(dx + ts / 2, ly, rf"$m$ = {_printed(pct(cell['m']))}", size=S_SMALL,
                color=role_b["ink"], z=7)  # fmt: skip
        cv.text(rx + ts / 2, ly, rf"$r$ = {_printed(pct(cell['r']))}", size=S_SMALL,
                color=role_d["ink"], z=7)  # fmt: skip
    # the outcome bar, the cell's name and its Update
    by = top - h + 3.0
    cv.text(x0 + 1.8, by, f"({d},{b})", size=S_LAB, weight="bold", ha="left", z=7,
            color=EXA["ink"] if hero else INK)  # fmt: skip
    if cell:
        _bar(cv, x0 + 9.4, x0 + w - 14.4, by, 3.4, cell, d)
        cv.text(x0 + w - 1.4, by - 0.1, _printed(pct(cell["U"]["est"])),
                size=S_NUM + (1.0 if hero else 0), ha="right",
                weight="bold" if hero else "normal", color=EXA["ink"] if hero else INK,
                z=7)  # fmt: skip
    return {"dx": dx, "rx": rx, "sx": sx}


def _panel_cells(cv: fs.MmCanvas, ic: Mapping[str, Any], data: Mapping[str, Any], x: float,
                 y: float, w: float, h: float) -> None:  # fmt: skip
    """(b): the cells as a grid, repair D across (column headers) and detection B down (row
    tabs); between the rows, the key's effect in each column and B x D, their difference."""
    items = data["items"]
    sub = "Update after a redefinition, changed definition items"
    _card(cv, x, y, w, h, "(b) Detection × repair",
          sub=f"{sub} ({items:,} item-times)" if items else sub)  # fmt: skip
    top = y + h - BAND
    rh_w = 10.4  # the row tabs
    cx0 = x + 1.6 + rh_w + 1.2
    gut = 2.6
    cw = (x + w - 1.6 - cx0 - gut) / 2
    col_x = {0: cx0, 1: cx0 + cw + gut}
    # column headers: repair D (solid at D=1)
    hy1 = top - 0.9
    hy0 = hy1 - 3.9
    for d in (0, 1):
        hd = HEAD_D[d]
        fc, ec, tc = (hd["solid"], hd["solid"], "white") if d else (hd["soft"], hd["stroke"],
                                                                    hd["ink"])  # fmt: skip
        _rbox(cv, col_x[d], hy0, cw, hy1 - hy0, fc, ec, lw=0.6, r=1.0, z=3)
        label = "exact recomputation" if d else "writer re-derives"
        lw_ = cv.text_width(label, S_HEAD, "bold")
        xs = col_x[d] + cw / 2 - (lw_ + 1.8 + cv.text_width(r"$D$=0", S_HEAD)) / 2
        cv.text(xs, (hy0 + hy1) / 2, label, size=S_HEAD, weight="bold", color=tc, ha="left", z=5)
        cv.text(xs + lw_ + 1.8, (hy0 + hy1) / 2, rf"$D$={d}", size=S_HEAD, color=tc, ha="left",
                z=5)  # fmt: skip
    sy = hy0 - 1.75
    ch_top = sy - 1.3
    gap = (ch_top - (y + 1.3)) - 2 * CELL_H
    rows = {0: ch_top, 1: ch_top - CELL_H - gap}
    geo = {}
    for b in (0, 1):
        for d in (0, 1):
            geo[d, b] = _cell(cv, ic, data, d, b, col_x[d], rows[b], cw, CELL_H)
    g = geo[0, 0]  # the steps, named once per column
    for d in (0, 1):
        ox = col_x[d] - col_x[0]
        for xx, s in ((g["dx"] + TILE / 2, "detection"), (g["rx"] + TILE / 2, "repair"),
                      (g["sx"] + 10.5, "served")):  # fmt: skip
            cv.text(xx + ox, sy, s, size=S_SMALL, style="italic", color=MUTED, z=5)
    # row tabs: detection B
    rx0 = x + 1.6
    for b in (0, 1):
        rl = KEY if b else AMB
        t = rows[b]
        _rbox(cv, rx0, t - CELL_H, rh_w, CELL_H, rl["soft"], rl["stroke"], lw=0.6, r=1.4, z=3)
        cv.text(rx0 + rh_w / 2, t - CELL_H / 2 + 1.7, "key" if b else "no key", size=S_HEAD,
                weight="bold", color=rl["ink"], z=5)  # fmt: skip
        cv.text(rx0 + rh_w / 2, t - CELL_H / 2 - 1.7, rf"$B$={b}", size=S_HEAD, color=rl["ink"],
                z=5)  # fmt: skip
    # the key, column by column: an arrow with its effect, and B x D between them
    g_top, g_bot = rows[0] - CELL_H, rows[1]
    gy = (g_top + g_bot) / 2
    ax_ = rx0 + rh_w / 2 - 2.6
    _arrow(cv, [(ax_, g_top - 0.3), (ax_, g_bot + 0.3)], color=KEY["mark"], lw=1.1, head=1.5)
    cv.text(ax_ + 1.4, gy, "+ key", size=S_LAB, weight="bold", color=KEY["ink"], ha="left", z=5)
    xa, xb = col_x[0] + cw * 0.56, col_x[1] + cw * 0.44
    for xv in (xa, xb):
        _arrow(cv, [(xv, g_top - 0.3), (xv, g_bot + 0.3)], color=KEY["mark"], lw=1.1, head=1.5)
    key = data["key"]
    if not (key[0] and key[1]):
        return
    pill_h = 3.2
    eff = {d: _printed(pp(key[d]["est"])) for d in (0, 1)}
    pw = {d: cv.text_width(eff[d], S_LAB, "bold") + 2.4 for d in (0, 1)}
    p0 = (xa + 1.4, xa + 1.4 + pw[0])  # right of its arrow
    p1 = (xb - 1.4 - pw[1], xb - 1.4)  # left of its arrow
    for d, (a0, a1) in ((0, p0), (1, p1)):
        _rbox(cv, a0, gy - pill_h / 2, a1 - a0, pill_h, KEY["soft"], KEY["stroke"], lw=0.5,
              r=pill_h / 2, z=6)  # fmt: skip
        cv.text((a0 + a1) / 2, gy - 0.05, eff[d], size=S_LAB, weight="bold", color=KEY["ink"],
                z=7)  # fmt: skip
    bxd = data["bxd"]
    if not bxd:
        return
    text = rf"$B{{\times}}D$ = {_printed(pp(bxd['est']))}"
    bw = cv.text_width(text, S_LAB, "bold") + 3.0
    bcx = (p0[1] + p1[0]) / 2
    bh = 3.4
    byc = gy + 0.95
    _rbox(cv, bcx - bw / 2, byc - bh / 2, bw, bh, PLUM["mark"], PLUM["mark"], lw=0.5, r=bh / 2,
          z=7)  # fmt: skip
    cv.text(bcx, byc - 0.05, text, size=S_LAB, weight="bold", color="white", z=8)
    lo, hi = (_printed(pp(bxd[k]).replace("\\,pp", "")) for k in ("lo", "hi"))
    cv.text(bcx, byc - bh / 2 - 1.15, f"[{lo}, {hi}]", size=S_SMALL, color=PLUM["ink"], z=8)
    for xa_, xb_ in ((p0[1], bcx - bw / 2), (bcx + bw / 2, p1[0])):  # B x D joins the two
        cv.ax.plot([xa_ + 0.15, xb_ - 0.15], [gy, gy], color=PLUM["mark"], lw=0.8, zorder=5)


def fig_concept(inp: Mapping[str, Any], out: Path) -> Path:
    """Figure 1: (a) one item and the five shifts, (b) the four HoldMem cells after a
    redefinition (see the module docstring)."""
    data = concept_cells(inp)
    ic = _icons(inp.get("icons"))
    with fs.printed() as plt:
        cv = fs.MmCanvas(plt, CONCEPT_PT[0] * fs.PT, CONCEPT_PT[1] * fs.PT, size=S_LAB)
        _panel_items(cv, ic, 0.25, 0.25, PANEL_A_W, cv.h - 0.5)
        _panel_cells(cv, ic, data, PANEL_A_W + 2.6, 0.25, cv.w - PANEL_A_W - 2.6 - 0.25,
                     cv.h - 0.5)  # fmt: skip
        return fs.save_pdf(cv.fig, out / "fig_concept.pdf")
