"""Shared matplotlib styling for figures targeting an ACM sigconf (acmart) paper."""

import os
import shutil

import matplotlib.pyplot as plt
import scienceplots
from matplotlib.patches import Patch
from cycler import cycler

# Figure widths in inches, measured from acmart's sigconf layout
COLUMN_WIDTH = 3.335
TEXT_WIDTH = 7.03

PANEL_WIDTH = (TEXT_WIDTH - 0.62) / 4
PANEL_ASPECT = 0.80
ROW_LABEL_WIDTH = 0.62

PALETTE = [
    "#e22e4c",  # crimson
    "#9d6ac7",  # purple
    "#feb876",  # peach-orange
    "#db7eb1",  # pink
    "#56b4e9",  # sky blue 
    "#00af81",  # green
    "#1e64cc",  # blue
    "#e6a62e",  # gold
    "#f0785c",  # coral
    "#b85aa8",  # magenta
    "#745fc0",  # indigo
    "#35a6a0",  # turquoise
    "#8fba4a",  # yellow-green
    "#d95f8d",  # raspberry
    "#c8b63c",  # mustard
    "#5c9bd5",  # medium blue
    "#c875a6",  # mauve
]

INK = "#0b0b0b"
MUTED_INK = "#595959"
GRID_COLOUR = "#d9d9d9"

USE_TEX = bool(shutil.which("latex") and shutil.which("dvipng"))
if os.environ.get("PLOT_USETEX") == "0":
    USE_TEX = False

plt.style.use(["science"])
plt.rcParams.update({
    "text.usetex": USE_TEX,
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "axes.titlepad": 5,
    "axes.labelpad": 5,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.prop_cycle": cycler(color=PALETTE),
    "axes.linewidth": 0.6,
    "axes.edgecolor": MUTED_INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "lines.linewidth": 1.3,
    "lines.solid_capstyle": "round",
    "axes.grid": False,
    "xtick.color": MUTED_INK,
    "ytick.color": MUTED_INK,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "xtick.top": False,
    "ytick.right": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

if USE_TEX:
    plt.rcParams.update({
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{libertine}\usepackage[libertine]{newtxmath}",
    })
else:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
    })

plt.rcParams["figure.figsize"] = (COLUMN_WIDTH, COLUMN_WIDTH * 0.85)

def figure_size(width: float, aspect: float = 0.85) -> tuple[float, float]:
    """
    Build a figure size from a target width.

    :param width: Figure width in inches, typically COLUMN_WIDTH or TEXT_WIDTH.
    :param aspect: Height as a fraction of the width.
    :return: Figure size as a (width, height) pair.
    """
    return (width, width * aspect)

def legend_below(fig, ax, ncol: int) -> None:
    """
    Place a single legend in a horizontal strip below the axes.

    :param fig: Figure the legend belongs to.
    :param ax: Axes supplying the legend handles.
    :param ncol: Number of legend columns.
    """
    handles, labels = ax.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=ncol,
        frameon=True,
        fancybox=False,
        edgecolor="0.8",
    )

MODELS = ["roberta", "xlmr", "nguni-xlmr", "afriberta"]

MODEL_COLOURS = dict(zip(MODELS, PALETTE))

MODEL_DISPLAY_NAMES = {
    "roberta": "RoBERTa",
    "xlmr": "XLM-R",
    "nguni-xlmr": "Nguni-XLMR",
    "afriberta": "AfriBERTa",
}

TASKS = ["ner", "pos", "ntc"]

TASK_COLOURS = dict(zip(TASKS, PALETTE))

LANGUAGE_DISPLAY_NAMES = {
    "xho": "isiXhosa",
    "zul": "isiZulu",
}

def categorical_colours(n: int) -> list:
    """
    Assign palette colours to an unordered series, such as entity classes.

    :param n: Number of series to colour.
    :return: List of n colours drawn from the shared palette.
    """
    return [PALETTE[i % len(PALETTE)] for i in range(n)]

def fitting_ncol(labels: list[str], width: float) -> int:
    """
    Estimate how many legend entries fit across the figure on one row.

    :param labels: Legend labels.
    :param width: Figure width in inches.
    :return: Largest number of columns that fits, at least one.
    """
    size = plt.rcParams["legend.fontsize"] / 72
    longest = max(len(label) for label in labels)

    entry = size * (1.0 + 0.6 + 0.55 * longest)
    spacing = size * 1.6

    return max(1, int((width + spacing) / (entry + spacing)))

def balanced_ncol(n: int, max_cols: int) -> int:
    """
    Choose a legend column count that fills every legend row evenly, so the
    strip below a figure never trails off with one lonely entry.

    :param n: Number of legend entries.
    :param max_cols: Largest number of columns that fits across the figure.
    :return: Number of legend columns.
    """
    if n <= max_cols:
        return n

    for cols in range(max_cols, 1, -1):
        if n % cols == 0:
            return cols

    return max(range(2, max_cols + 1), key=lambda cols: (-((-n % cols)), cols))

def grid_figure(nrows: int, ncols: int, row_labels: bool = False,
                sharex: bool | str = True, sharey: bool | str = "row"):
    """
    Create a panel grid. Panels are the same size in every figure, so the figure
    width follows from the column count rather than being fixed to the text width.

    :param nrows: Number of panel rows.
    :param ncols: Number of panel columns.
    :param row_labels: Whether to reserve a left margin for row labels.
    :param sharex: Axis sharing for the x axis, e.g. True, "col", "row" or False.
    :param sharey: Axis sharing for the y axis, e.g. "row", "col", True or False.
    :return: The figure and its 2D array of axes.
    """
    width = ncols * PANEL_WIDTH + (ROW_LABEL_WIDTH if row_labels else 0.0)
    height = nrows * (PANEL_WIDTH * PANEL_ASPECT + 0.35)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(width, height),
        sharex=sharex,
        sharey=sharey,
        squeeze=False,
    )

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelbottom=True, labelleft=True)
        ax.grid(False)

    return fig, axes

# Longest x label that still fits under every column of a wide grid
MAX_REPEATED_XLABEL = 12

def label_grid(axes, column_titles: list[str] | None = None,
               xlabel: str | None = None,
               ylabels: list[str] | str | None = None,
               fig=None) -> None:
    """
    Label a panel grid once around its edges rather than per panel: titles along
    the top, the measured quantity down the left, the grouping down the right and
    the shared x label along the bottom.

    :param axes: 2D array of axes returned by grid_figure.
    :param column_titles: One title per column, drawn above the top row.
    :param xlabel: Shared x-axis label, drawn under the bottom row.
    :param ylabels: One y label per row, or a single label used for every row.
    :param fig: Figure holding the grid, needed to centre a long x label.
    """
    nrows, ncols = axes.shape

    if column_titles is not None:
        for col, title in enumerate(column_titles):
            axes[0][col].set_title(title)

    if xlabel is not None:
        if fig is not None and len(xlabel) > MAX_REPEATED_XLABEL and ncols > 2:
            fig.supxlabel(xlabel, fontsize=plt.rcParams["axes.labelsize"])
        else:
            for col in range(ncols):
                axes[nrows - 1][col].set_xlabel(xlabel)

    if ylabels is not None:
        if isinstance(ylabels, str):
            ylabels = [ylabels] * nrows
        for row, label in enumerate(ylabels):
            axes[row][0].set_ylabel(label)

def swatch_handles(handles: list) -> list:
    """
    Convert line handles into filled squares for the legend.

    :param handles: Line handles collected from the axes.
    :return: One square patch per handle, keeping each line's colour.
    """
    return [Patch(facecolor=handle.get_color(), edgecolor="none") for handle in handles]

def legend_entries(axes) -> tuple[list, list]:
    """
    Collect one legend entry per distinct label across every panel in a grid.

    :param axes: 2D array of axes.
    :return: The handles and labels for a single shared legend, in first-seen order.
    """
    seen = {}

    for ax in axes.flat:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(label, handle)

    return list(seen.values()), list(seen.keys())

def row_bands(labels: list[str]) -> list[tuple[str, list[int]]]:
    """
    Group consecutive rows carrying the same label into one band.

    :param labels: One label per row.
    :return: List of (label, row indices) pairs, in row order.
    """
    bands = []

    for row, label in enumerate(labels):
        if bands and bands[-1][0] == label:
            bands[-1][1].append(row)
        else:
            bands.append((label, [row]))

    return bands

def draw_row_labels(fig, axes, labels: list[str], left: float) -> None:
    """
    Draw a horizontal label beside each row, in the margin left of the y labels.

    :param fig: Figure holding the grid.
    :param axes: 2D array of axes.
    :param labels: One label per row; runs of the same label share one band label.
    :param left: Fraction of the figure width reserved for the labels.
    """
    for label, rows in row_bands(labels):
        top = axes[rows[0]][0].get_position().y1
        bottom = axes[rows[-1]][0].get_position().y0

        fig.text(
            left * 0.5,
            0.5 * (top + bottom),
            label,
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=plt.rcParams["axes.labelsize"],
        )

def finalise_grid(fig, axes, output_path, row_labels: list[str] | None = None,
                  ncol: int | None = None, max_legend_cols: int | None = None) -> None:
    """
    Lay out a panel grid, add its shared legend strip underneath and save it.

    :param fig: Figure holding the grid.
    :param axes: 2D array of axes.
    :param output_path: File path to save the figure to.
    :param row_labels: One horizontal label per row, drawn in the left margin.
    :param ncol: Legend column count; chosen to fill evenly when omitted.
    :param max_legend_cols: Widest legend strip to allow; measured from the figure
        width when omitted.
    """
    handles, labels = legend_entries(axes)

    if ncol is None:
        fits = max_legend_cols or fitting_ncol(labels, fig.get_figwidth())
        ncol = balanced_ncol(len(labels), fits)

    legend_rows = -(-len(labels) // ncol)

    supxlabel = fig._supxlabel
    legend_height = 0.20 * legend_rows + 0.10
    label_height = 0.24 if supxlabel is not None else 0.0

    fig.set_figheight(fig.get_figheight() + legend_height + label_height)

    legend_fraction = legend_height / fig.get_figheight()
    label_fraction = label_height / fig.get_figheight()

    left = ROW_LABEL_WIDTH / fig.get_figwidth() if row_labels else 0.0

    fig.tight_layout(
        rect=(left, legend_fraction + label_fraction, 1, 1), h_pad=1.2, w_pad=1.0
    )

    if row_labels:
        draw_row_labels(fig, axes, row_labels, left)

    if supxlabel is not None:
        supxlabel.set_position((0.5, legend_fraction + 0.35 * label_fraction))

    fig.legend(
        swatch_handles(handles),
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=ncol,
        frameon=False,
        columnspacing=1.6,
        handlelength=1.0,
        handleheight=1.0,
        handletextpad=0.6,
    )

    fig.savefig(output_path)
    plt.close(fig)

def configure_step_axis(ax, steps) -> None:
    """
    Configure a continued pretraining step axis, shared by every dynamics panel.

    :param ax: Axes to configure.
    :param steps: Checkpoint steps being plotted.
    """
    import matplotlib.ticker as ticker

    ax.set_xlim(min(steps), max(steps))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x / 1000:g}k" if x >= 1000 else f"{int(x)}")
    )

def configure_row_value_axis(row_axes, cap_at_one: bool = True) -> None:
    """
    Give every panel in a row one common y scale, running from zero (or below,
    for metrics that go negative) to a tick clear of the highest value in the row.

    :param row_axes: The axes making up a single grid row.
    :param cap_at_one: Whether the metric cannot exceed one, as F1 and cosine cannot.
    """
    import matplotlib.ticker as ticker

    drawn = [ax for ax in row_axes if ax.has_data()]
    if not drawn:
        return

    low = min(0.0, min(ax.dataLim.intervaly[0] for ax in drawn))
    high = max(ax.dataLim.intervaly[1] for ax in drawn)

    locator = ticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10])
    ticks = locator.tick_values(low, high)
    step = ticks[1] - ticks[0]

    bottom = max(t for t in ticks if t <= low)
    top = min(t for t in ticks if t >= high)

    if top - high < 0.25 * step:
        top += step

    if cap_at_one:
        top = min(top, 1.0)

    for ax in row_axes:
        ax.yaxis.set_major_locator(locator)
        ax.set_ylim(bottom, top)
