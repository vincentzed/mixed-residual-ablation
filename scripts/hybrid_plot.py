# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "typer", "pydantic", "pillow"]
# ///
"""Generic Meta-layout chart with humans& styling — the hybrid plotter.

Composition is Meta's DFlash card: 1080x1080 dark square, touching bar
pairs, range whiskers on the hero bar, bracket + speed-up pill per group,
manual right-hand legend, footnote. Skin is humansand (tokens.css):
Newsreader serif title at weight 400 with one accent-italic word, Inter
labels, monospace numerals, teal hero / faint context bars, rust pills,
warm paper grounds. Themes: dark (default, matches Meta's mood) and light.
"""

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import matplotlib
import typer
from pydantic import BaseModel

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

CANVAS = 1080.0  # design space; everything is placed in these pixels
AX_L, AX_R, AX_T, AX_B = 180.0, 1000.0, 200.0, 895.0

SERIF = "Newsreader"
SANS = "Inter"
MONO = "GoogleSansCode Nerd Font Mono"


class Theme(BaseModel):
    """humansand tokens (assets/tokens.css) mapped onto Meta's chart roles."""

    paper: str
    ink: str
    muted: str
    faint: str
    line: str
    accent: str  # hero bars
    accent_strong: str  # speed-up pill


DARK = Theme(
    paper="#1e1a14",
    ink="#f1ece0",
    muted="#b3a897",
    faint="#837a6a",
    line="#3c352a",
    accent="#5cc9b8",
    accent_strong="#e2836a",
)
LIGHT = Theme(
    paper="#fffdf8",
    ink="#161513",
    muted="#6d6860",
    faint="#918a7c",
    line="#d8d0c4",
    accent="#176b64",
    accent_strong="#a33e2d",
)


class Group(BaseModel):
    """One x-axis group: a hero/context bar pair with optional extras."""

    label: str
    hero: float
    context: float
    range_low: float | None = None
    range_high: float | None = None
    badge: str | None = None


class ThemeName(StrEnum):
    dark = "dark"
    light = "light"
    both = "both"


def _register_fonts(fonts_dir: Path) -> None:
    for ttf in fonts_dir.glob("*.ttf"):
        font_manager.fontManager.addfont(str(ttf))
    home_fonts = Path.home() / "Library" / "Fonts"
    for name in (
        "GoogleSansCodeNerdFontMono-Regular.ttf",
        "GoogleSansCodeNerdFontMono-SemiBold.ttf",
        "GoogleSansCodeNerdFontMono-Bold.ttf",
    ):
        ttf = home_fonts / name
        if ttf.exists():
            font_manager.fontManager.addfont(str(ttf))


def _nice_axis(vmax: float) -> tuple[float, float]:
    """Return (ymax, step) giving 5-8 zero-anchored ticks above vmax."""
    for mag in (1, 10, 100, 1000, 10000):
        for step in (1 * mag, 2 * mag, 5 * mag):
            ticks = int(vmax // step) + 1
            if 4 <= ticks <= 7:
                return step * ticks, step
    return vmax, vmax / 5


def _title_with_accent(
    fig,
    x_px: float,
    y_px: float,
    parts: list[tuple[str, bool]],
    theme: Theme,
    fontsize: float,
) -> None:
    """Sequential serif title spans; accented parts go italic teal."""
    fig.canvas.draw()
    x = x_px / CANVAS
    for text, accented in parts:
        t = fig.text(
            x,
            1 - y_px / CANVAS,
            text,
            fontfamily=SERIF,
            fontsize=fontsize,
            fontweight=400,
            color=theme.accent if accented else theme.ink,
            fontstyle="italic" if accented else "normal",
            ha="left",
            va="baseline",
        )
        fig.canvas.draw()
        bb = t.get_window_extent()
        x = fig.transFigure.inverted().transform((bb.x1, 0))[0]


def meta_card(
    groups: list[Group],
    out: Path,
    title_parts: list[tuple[str, bool]],
    subtitle: str,
    ylabel: str,
    hero_name: tuple[str, str],
    context_name: tuple[str, str],
    theme: Theme = DARK,
    kicker: str = "",
    footnote: str = "",
    range_legend: tuple[str, str] = ("Range across", "prompt categories"),
    badge_legend: tuple[str, str] = ("Speed-up over", "baseline"),
    legend_loc: str = "right",
    value_fmt: str = "{:g}",
    show_badge_legend: bool = True,
    xlabel: str = "",
) -> Path:
    """Render one Meta-composition card with humansand styling.

    legend_loc: "right" puts the legend inside the plot's upper right (the
    Meta original — needs short bars there); "bottom" lays it in a row under
    the x labels, for cards whose bars fill the frame.
    """
    vmax = max(max(g.hero, g.context, g.range_high or 0) for g in groups)
    ymax, step = _nice_axis(vmax)
    if any(g.badge for g in groups) and vmax > ymax - 0.55 * step:
        ymax += step  # headroom so a clamped pill never lands on a value label
    ax_b = AX_B if legend_loc == "right" else 858.0

    fig = plt.figure(figsize=(CANVAS / 100, CANVAS / 100), dpi=100)
    fig.patch.set_facecolor(theme.paper)
    ax = fig.add_axes(
        (
            AX_L / CANVAS,
            1 - ax_b / CANVAS,
            (AX_R - AX_L) / CANVAS,
            (ax_b - AX_T) / CANVAS,
        )
    )
    ax.set_facecolor(theme.paper)
    ax.set_xlim(AX_L, AX_R)
    ax.set_ylim(0, ymax)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme.ink)
        ax.spines[side].set_linewidth(1.6)

    n_ticks = int(ymax // step) + 1
    ax.set_yticks([i * step for i in range(n_ticks)])
    ax.tick_params(axis="y", length=0, pad=12, labelsize=14, labelcolor=theme.muted)
    for lab in ax.get_yticklabels():
        lab.set_fontfamily(MONO)

    n = len(groups)
    slot = (AX_R - AX_L) / n
    centers = [AX_L + slot * (i + 0.5) for i in range(n)]
    bar_w = min(85.0, slot * 0.31)

    ax.set_xticks(centers)
    ax.set_xticklabels([g.label for g in groups])
    ax.tick_params(axis="x", length=0, pad=16, labelsize=15.5, labelcolor=theme.ink)
    for lab in ax.get_xticklabels():
        lab.set_fontfamily(SANS)
        lab.set_fontweight("semibold")

    # LOCAL EXTENSION: meta_line_card labels its x axis and meta_card does not.
    # Category names are usually self-describing, but numeric ones are not.
    if xlabel:
        ax.set_xlabel(
            xlabel,
            fontsize=14,
            fontfamily=SANS,
            fontweight="medium",
            color=theme.muted,
            labelpad=14,
        )

    fig.text(
        90 / CANVAS,
        1 - ((AX_T + ax_b) / 2) / CANVAS,
        ylabel,
        rotation=90,
        ha="center",
        va="center",
        fontsize=14,
        fontfamily=SANS,
        fontweight="medium",
        color=theme.muted,
    )

    lift = ymax * 0.02  # gap between a bar/whisker top and its value label
    for g, cx in zip(groups, centers):
        bx, hx = cx - bar_w / 2, cx + bar_w / 2
        ax.bar(bx, g.context, width=bar_w, color=theme.faint, zorder=3)
        ax.bar(hx, g.hero, width=bar_w, color=theme.accent, zorder=3)

        top = g.hero
        if g.range_low is not None and g.range_high is not None:
            top = max(top, g.range_high)
            cap = bar_w * 0.24
            ax.plot(
                [hx, hx],
                [g.range_low, g.range_high],
                color=theme.ink,
                lw=1.8,
                zorder=5,
            )
            for yv in (g.range_low, g.range_high):
                ax.plot(
                    [hx - cap, hx + cap],
                    [yv, yv],
                    color=theme.ink,
                    lw=1.8,
                    zorder=5,
                )

        ax.text(
            cx - bar_w * 0.06,
            g.context + lift,
            value_fmt.format(g.context),
            ha="right",
            va="bottom",
            fontsize=15,
            fontfamily=MONO,
            fontweight="semibold",
            color=theme.muted,
            zorder=5,
        )
        ax.text(
            cx + bar_w * 0.06,
            top + lift,
            value_fmt.format(g.hero),
            ha="left",
            va="bottom",
            fontsize=15,
            fontfamily=MONO,
            fontweight="semibold",
            color=theme.accent,
            zorder=5,
            clip_on=False,
        )

        if g.badge:
            # clamp so the pill never rises into the title/subtitle zone
            ceiling = ymax * (ax_b - 190.0) / (ax_b - AX_T)
            by = min(top + ymax * 0.115, ceiling)
            tick = ymax * 0.02
            left, right = cx - bar_w * 1.18, cx + bar_w * 1.18
            for xs in ([left, left], [right, right]):
                ax.plot(
                    xs,
                    [by - tick, by],
                    color=theme.faint,
                    lw=1.6,
                    clip_on=False,
                    zorder=4,
                )
            ax.plot(
                [left, right],
                [by, by],
                color=theme.faint,
                lw=1.6,
                clip_on=False,
                zorder=4,
            )
            ax.text(
                cx,
                by,
                g.badge,
                ha="center",
                va="center",
                fontsize=14.5,
                fontfamily=MONO,
                fontweight="bold",
                color=theme.paper,
                zorder=7,
                clip_on=False,
                bbox={
                    "boxstyle": "round,pad=0.38,rounding_size=0.65",
                    "facecolor": theme.accent_strong,
                    "edgecolor": "none",
                },
            )

    if kicker:
        fig.text(
            75 / CANVAS,
            1 - 62 / CANVAS,
            kicker,
            fontsize=11.5,
            fontfamily=MONO,
            fontweight="bold",
            color=theme.accent,
            ha="left",
            va="baseline",
        )
    _title_with_accent(fig, 75, 112, title_parts, theme, fontsize=29)
    fig.text(
        75 / CANVAS,
        1 - 152 / CANVAS,
        subtitle,
        fontsize=15.5,
        fontfamily=SERIF,
        fontweight=400,
        color=theme.muted,
        ha="left",
        va="baseline",
    )
    if footnote:
        fig.text(
            75 / CANVAS,
            1 - (1000 if legend_loc == "right" else 1012) / CANVAS,
            footnote,
            fontsize=12,
            fontfamily=SANS,
            color=theme.faint,
            ha="left",
            va="baseline",
        )

    sw_w, sw_h = 48.0, 24.0
    entries = [
        ("swatch", theme.accent, 240.0, *hero_name),
        ("swatch", theme.faint, 305.0, *context_name),
    ]
    if any(g.range_low is not None for g in groups):
        entries.append(("whisker", theme.ink, 372.0, *range_legend))
    # LOCAL EXTENSION: a bottom legend fits two entries at this canvas width; a
    # third runs off the edge. Opt out when the subtitle already says what the
    # badge is.
    if show_badge_legend and any(g.badge for g in groups):
        entries.append(("swatch", theme.accent_strong, 438.0, *badge_legend))
    if legend_loc == "bottom":
        _bottom_legend(fig, entries, theme)
        entries = []
    for kind, color, cy, line1, line2 in entries:
        if kind == "swatch":
            fig.add_artist(
                FancyBboxPatch(
                    (812 / CANVAS, 1 - (cy + sw_h / 2) / CANVAS),
                    sw_w / CANVAS,
                    sw_h / CANVAS,
                    boxstyle="round,pad=0,rounding_size=0.004",
                    facecolor=color,
                    edgecolor="none",
                    transform=fig.transFigure,
                )
            )
        else:
            gx = (812 + sw_w / 2) / CANVAS
            fig.add_artist(
                Line2D(
                    [gx, gx],
                    [1 - (cy + 16) / CANVAS, 1 - (cy - 16) / CANVAS],
                    color=color,
                    lw=1.8,
                    transform=fig.transFigure,
                )
            )
            for dy in (-16, 16):
                fig.add_artist(
                    Line2D(
                        [gx - 10 / CANVAS, gx + 10 / CANVAS],
                        [1 - (cy + dy) / CANVAS] * 2,
                        color=color,
                        lw=1.8,
                        transform=fig.transFigure,
                    )
                )
        fig.text(
            878 / CANVAS,
            1 - (cy - 6) / CANVAS,
            line1,
            fontsize=14,
            fontfamily=SANS,
            fontweight="semibold",
            color=theme.ink,
            va="center",
        )
        fig.text(
            878 / CANVAS,
            1 - (cy + 20) / CANVAS,
            line2,
            fontsize=12.5,
            fontfamily=SANS,
            fontweight="medium",
            color=theme.muted,
            va="center",
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, facecolor=theme.paper)
    plt.close(fig)
    _round_card(out, theme)
    return out


def _bottom_legend(fig, entries, theme: Theme) -> None:
    """One-row legend under the x labels, entries placed by measured width."""
    y_c = 952.0
    fig.canvas.draw()
    x_cur = AX_L
    for kind, color, _cy, line1, line2 in entries:
        if kind == "swatch":
            fig.add_artist(
                FancyBboxPatch(
                    (x_cur / CANVAS, 1 - (y_c + 9) / CANVAS),
                    36 / CANVAS,
                    18 / CANVAS,
                    boxstyle="round,pad=0,rounding_size=0.003",
                    facecolor=color,
                    edgecolor="none",
                    transform=fig.transFigure,
                )
            )
        else:
            gx = (x_cur + 18) / CANVAS
            fig.add_artist(
                Line2D(
                    [gx, gx],
                    [1 - (y_c + 12) / CANVAS, 1 - (y_c - 12) / CANVAS],
                    color=color,
                    lw=1.8,
                    transform=fig.transFigure,
                )
            )
            for dy in (-12, 12):
                fig.add_artist(
                    Line2D(
                        [gx - 8 / CANVAS, gx + 8 / CANVAS],
                        [1 - (y_c + dy) / CANVAS] * 2,
                        color=color,
                        lw=1.8,
                        transform=fig.transFigure,
                    )
                )
        t1 = fig.text(
            (x_cur + 48) / CANVAS,
            1 - y_c / CANVAS,
            line1,
            fontsize=13.5,
            fontfamily=SANS,
            fontweight="semibold",
            color=theme.ink,
            va="center",
        )
        fig.canvas.draw()
        x1 = t1.get_window_extent().x1 * 100 / fig.dpi
        t2 = fig.text(
            (x1 + 7) / CANVAS,
            1 - y_c / CANVAS,
            line2,
            fontsize=12.5,
            fontfamily=SANS,
            fontweight="medium",
            color=theme.muted,
            va="center",
        )
        fig.canvas.draw()
        x_cur = t2.get_window_extent().x1 * 100 / fig.dpi + 46


def meta_heatmap_card(
    values: list[list[float]],
    row_labels: list[str],
    out: Path,
    title_parts: list[tuple[str, bool]],
    subtitle: str,
    xlabel: str,
    cbar_label: str,
    theme: Theme = DARK,
    kicker: str = "",
    footnote: str = "",
    vmax: float = 90.0,
    x_ticks: list[int] | None = None,
) -> Path:
    """Heatmap sibling of meta_card: paper-to-accent ramp, same card chrome."""
    from matplotlib.colors import LinearSegmentedColormap

    fig = plt.figure(figsize=(CANVAS / 100, CANVAS / 100), dpi=100)
    fig.patch.set_facecolor(theme.paper)
    top, bottom = 280.0, 760.0
    ax = fig.add_axes(
        (
            AX_L / CANVAS,
            1 - bottom / CANVAS,
            (AX_R - AX_L) / CANVAS,
            (bottom - top) / CANVAS,
        )
    )
    ax.set_facecolor(theme.paper)

    cmap = LinearSegmentedColormap.from_list("hs", [theme.line, theme.accent])
    im = ax.imshow(
        values, aspect="auto", cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest"
    )

    n_rows, n_cols = len(values), len(values[0])
    ax.set_xticks([i - 0.5 for i in range(n_cols + 1)], minor=True)
    ax.set_yticks([i - 0.5 for i in range(n_rows + 1)], minor=True)
    ax.grid(which="minor", color=theme.paper, linewidth=2.5)
    ax.tick_params(which="both", length=0)

    cols = x_ticks or list(range(1, n_cols + 1, 4))
    ax.set_xticks([c - 1 for c in cols])
    ax.set_xticklabels([str(c) for c in cols])
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels)
    ax.tick_params(axis="x", pad=10, labelsize=13, labelcolor=theme.muted)
    ax.tick_params(axis="y", pad=10, labelsize=14, labelcolor=theme.ink)
    for lab in ax.get_xticklabels():
        lab.set_fontfamily(MONO)
    for lab in ax.get_yticklabels():
        lab.set_fontfamily(SANS)
        lab.set_fontweight("semibold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel(
        xlabel,
        fontsize=14,
        fontfamily=SANS,
        fontweight="medium",
        color=theme.muted,
        labelpad=14,
    )

    cax = fig.add_axes((AX_L / CANVAS, 1 - 880 / CANVAS, 250 / CANVAS, 16 / CANVAS))
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.set_ticks([0, vmax])
    cb.set_ticklabels(["0%", f"{vmax:g}%"])
    cax.tick_params(length=0, pad=8, labelsize=12.5, labelcolor=theme.muted)
    for lab in cax.get_xticklabels():
        lab.set_fontfamily(MONO)
    fig.text(
        (AX_L + 270) / CANVAS,
        1 - 872 / CANVAS,
        cbar_label,
        fontsize=13,
        fontfamily=SANS,
        fontweight="medium",
        color=theme.muted,
        va="center",
    )

    if kicker:
        fig.text(
            75 / CANVAS,
            1 - 62 / CANVAS,
            kicker,
            fontsize=11.5,
            fontfamily=MONO,
            fontweight="bold",
            color=theme.accent,
            ha="left",
            va="baseline",
        )
    _title_with_accent(fig, 75, 112, title_parts, theme, fontsize=29)
    fig.text(
        75 / CANVAS,
        1 - 152 / CANVAS,
        subtitle,
        fontsize=15.5,
        fontfamily=SERIF,
        fontweight=400,
        color=theme.muted,
        ha="left",
        va="baseline",
    )
    if footnote:
        fig.text(
            75 / CANVAS,
            1 - 1000 / CANVAS,
            footnote,
            fontsize=12,
            fontfamily=SANS,
            color=theme.faint,
            ha="left",
            va="baseline",
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, facecolor=theme.paper)
    plt.close(fig)
    _round_card(out, theme)
    return out


def meta_line_card(
    x: list[float],
    series: dict[str, list[float]],
    out: Path,
    title_parts: list[tuple[str, bool]],
    subtitle: str,
    ylabel: str,
    xlabel: str,
    hero: str,
    theme: Theme = DARK,
    kicker: str = "",
    footnote: str = "",
    y_suffix: str = "",
) -> Path:
    """Line-chart sibling of meta_card: same card chrome, direct end labels.

    The hero series gets the accent teal and a heavier stroke; the rest walk
    down [rust, muted, faint]. Labels sit at each line's right end (nudged
    apart on collision) instead of a legend box.
    """
    fig = plt.figure(figsize=(CANVAS / 100, CANVAS / 100), dpi=100)
    fig.patch.set_facecolor(theme.paper)
    right = 870.0  # leave room for the end labels
    ax = fig.add_axes(
        (
            AX_L / CANVAS,
            1 - AX_B / CANVAS,
            (right - AX_L) / CANVAS,
            (AX_B - AX_T) / CANVAS,
        )
    )
    ax.set_facecolor(theme.paper)

    lo = min(min(v) for v in series.values())
    hi = max(max(v) for v in series.values())
    pad = (hi - lo) * 0.12
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(lo - pad, hi + pad)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme.ink)
        ax.spines[side].set_linewidth(1.6)
    ax.grid(axis="y", color=theme.line, linewidth=1.0)
    ax.set_axisbelow(True)

    ax.tick_params(
        axis="both", length=0, pad=12, labelsize=13.5, labelcolor=theme.muted
    )
    ax.set_xticks(x[:: max(1, len(x) // 15)])
    for lab in [*ax.get_yticklabels(), *ax.get_xticklabels()]:
        lab.set_fontfamily(MONO)
    if y_suffix:
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}{y_suffix}")

    fig.text(
        90 / CANVAS,
        1 - ((AX_T + AX_B) / 2) / CANVAS,
        ylabel,
        rotation=90,
        ha="center",
        va="center",
        fontsize=14,
        fontfamily=SANS,
        fontweight="medium",
        color=theme.muted,
    )
    ax.set_xlabel(
        xlabel,
        fontsize=14,
        fontfamily=SANS,
        fontweight="medium",
        color=theme.muted,
        labelpad=14,
    )

    rest = [theme.accent_strong, theme.muted, theme.faint]
    colors, i = {}, 0
    for name in series:
        if name == hero:
            colors[name] = theme.accent
        else:
            colors[name] = rest[min(i, len(rest) - 1)]
            i += 1

    # end labels, nudged apart when lines finish too close together
    min_gap = (hi - lo + 2 * pad) * 0.055
    ends = sorted(((v[-1], k) for k, v in series.items()))
    placed: list[float] = []
    label_y = {}
    for yv, name in ends:
        y = yv if not placed else max(yv, placed[-1] + min_gap)
        placed.append(y)
        label_y[name] = y

    for name, vals in series.items():
        is_hero = name == hero
        ax.plot(
            x,
            vals,
            color=colors[name],
            lw=3.0 if is_hero else 2.0,
            solid_capstyle="round",
            zorder=5 if is_hero else 4,
        )
        ax.plot(
            x,
            vals,
            "o",
            color=colors[name],
            markersize=4.5 if is_hero else 3.5,
            zorder=5 if is_hero else 4,
        )
        ax.text(
            x[-1] + (max(x) - min(x)) * 0.025,
            label_y[name],
            name,
            ha="left",
            va="center",
            fontsize=13.5,
            fontfamily=SANS,
            fontweight="semibold" if is_hero else "medium",
            color=colors[name],
            clip_on=False,
        )

    if kicker:
        fig.text(
            75 / CANVAS,
            1 - 62 / CANVAS,
            kicker,
            fontsize=11.5,
            fontfamily=MONO,
            fontweight="bold",
            color=theme.accent,
            ha="left",
            va="baseline",
        )
    _title_with_accent(fig, 75, 112, title_parts, theme, fontsize=29)
    fig.text(
        75 / CANVAS,
        1 - 152 / CANVAS,
        subtitle,
        fontsize=15.5,
        fontfamily=SERIF,
        fontweight=400,
        color=theme.muted,
        ha="left",
        va="baseline",
    )
    if footnote:
        fig.text(
            75 / CANVAS,
            1 - 1000 / CANVAS,
            footnote,
            fontsize=12,
            fontfamily=SANS,
            color=theme.faint,
            ha="left",
            va="baseline",
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, facecolor=theme.paper)
    plt.close(fig)
    _round_card(out, theme)
    return out


def _round_card(png: Path, theme: Theme, radius: int = 48) -> None:
    """Rounded corners + thin warm border, matching the humansand cards."""
    from PIL import Image, ImageDraw

    img = Image.open(png).convert("RGBA")
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, img.width - 1, img.height - 1), radius=radius, fill=255
    )
    img.putalpha(mask)
    ImageDraw.Draw(img).rounded_rectangle(
        (1, 1, img.width - 2, img.height - 2),
        radius=radius,
        outline=theme.line,
        width=3,
    )
    img.save(png)


DFLASH_GROUPS = [
    Group(
        label="RTX-5090",
        hero=233,
        context=74.9,
        range_low=131,
        range_high=327,
        badge="3.1x",
    ),
    Group(
        label="M5-Max", hero=50, context=26.6, range_low=35, range_high=63, badge="1.8x"
    ),
    Group(
        label="M4-Max", hero=38, context=23.7, range_low=28, range_high=50, badge="1.5x"
    ),
]


def main(
    out: Annotated[Path, typer.Option(help="Output directory.")] = Path("media"),
    theme: Annotated[ThemeName, typer.Option(help="Card theme.")] = ThemeName.both,
    fonts: Annotated[Path, typer.Option(help="Dir with extra TTFs.")] = Path("fonts"),
) -> None:
    """Render the DFlash demo data through the hybrid plotter."""
    _register_fonts(fonts)
    wanted = (
        [("dark", DARK), ("light", LIGHT)]
        if theme is ThemeName.both
        else [(theme.value, DARK if theme is ThemeName.dark else LIGHT)]
    )
    for name, th in wanted:
        p = meta_card(
            DFLASH_GROUPS,
            out / f"dflash-hybrid-{name}.png",
            title_parts=[
                ("DFlash ", False),
                ("Speculative", True),
                (" Decoding Performance", False),
            ],
            subtitle="Decode speed (tok/s) — baseline vs DFlash across 7 prompt categories",
            ylabel="Decode Speed (tok/s)",
            hero_name=("DFlash", "(Speculative)"),
            context_name=("Baseline", "(No Speculation)"),
            theme=th,
            kicker="BENCHMARK · MOCK DATA",
            footnote="*M4/M5 numbers measured using ExecuTorch; RTX-5090 measured using llama.cpp.",
        )
        typer.echo(f"wrote {p}")


if __name__ == "__main__":
    typer.run(main)
