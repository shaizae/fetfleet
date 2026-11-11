"""One look for every report figure: theme, colors and figure factory.

Figures are plain `matplotlib.figure.Figure` objects (no pyplot): nothing is
registered globally, no GUI backend is touched, and they are safe to draw on a
background thread as long as each figure stays on one thread.

Colors: a fixed categorical order validated for color-vision deficiency (each model
keeps its slot in every plot of a report), a single-hue blue sequential ramp for
heatmaps, and neutral inks for text, axes and reference lines.
"""
from __future__ import annotations

import contextlib
from typing import Iterator, Sequence

import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

# categorical slots, in this fixed order (validated adjacent-pair CVD separation)
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
OVERFLOW = "#8c8b86"          # models past the 8th slot (always named in the legend)
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8c8b86"
GRID = "#e6e5e1"
REFERENCE = "#9a9994"         # chance / identity / baseline lines
SURFACE = "#ffffff"
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "ml_pipe_blue", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

THEME = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 9,
    "axes.labelcolor": INK_SECONDARY,
    "axes.edgecolor": GRID,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.prop_cycle": matplotlib.cycler(color=CATEGORICAL),
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "text.color": INK,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "lines.linewidth": 2,
    "lines.markersize": 5,
    "figure.titlesize": 11,
    "figure.titleweight": "bold",
}


@contextlib.contextmanager
def theme() -> Iterator[None]:
    """Apply the report theme while a figure is built and saved."""
    with matplotlib.rc_context(THEME):
        yield


def new_figure(width: float, height: float, nrows: int = 1, ncols: int = 1, squeeze: bool = True, **subplot_kw):
    """(Figure, axes) with constrained layout; axes as plt.subplots returns them."""
    fig = Figure(figsize=(width, height), layout="constrained")
    axes = fig.subplots(nrows, ncols, squeeze=squeeze, **subplot_kw)
    return fig, axes


def wrap_titles(fig: Figure) -> None:
    """Wrap axes titles that would be wider than their axes (long model names, many words)."""
    import textwrap
    for ax in fig.axes:
        for loc in ("left", "center", "right"):
            text = ax.get_title(loc=loc)
            if not text or "\n" in text:
                continue
            width_pt = ax.get_position().width * fig.get_figwidth() * 72
            capacity = max(16, int(width_pt / (ax.title.get_fontsize() * 0.6)))
            if len(text) > capacity:
                ax.set_title("\n".join(textwrap.wrap(text, capacity)), loc=loc)


def model_colors(models: Sequence[str]) -> dict[str, str]:
    """Each model keeps one color in every plot: slot i for the i-th model, gray past 8."""
    return {m: CATEGORICAL[i] if i < len(CATEGORICAL) else OVERFLOW for i, m in enumerate(models)}


def model_markers(models: Sequence[str]) -> dict[str, str]:
    """A second, color-independent identity channel for scatter plots."""
    return {m: MARKERS[i % len(MARKERS)] for i, m in enumerate(models)}


def reference_line(ax, xs, ys, label: str | None = None) -> None:
    """A thin neutral reference (chance, identity, baseline) line."""
    ax.plot(xs, ys, color=REFERENCE, linewidth=1, linestyle=(0, (4, 3)), label=label, zorder=1)


def annotate_heatmap(ax, values, fmt: str, vmax: float | None = None) -> None:
    """Write each cell's value in ink that stays readable on the sequential ramp."""
    import numpy as np
    values = np.asarray(values, dtype=float)
    top = np.nanmax(values) if vmax is None else vmax
    for (i, j), v in np.ndenumerate(values):
        if np.isnan(v):
            continue
        dark = top > 0 and v / top > 0.55
        ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=7,
                color="#ffffff" if dark else INK)
