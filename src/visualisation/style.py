"""Shared matplotlib styling for figures targeting an ACM sigconf (acmart) paper."""

import os
import shutil

import matplotlib.pyplot as plt
import scienceplots 
from cycler import cycler

# Figure widths in inches, measured from acmart's sigconf layout
COLUMN_WIDTH = 3.335
TEXT_WIDTH = 7.03

PALETTE = [
    "#e22e4c",  # crimson
    "#01c993",  # green
    "#9d6ac7",  # purple
    "#feb876",  # vermillion
    "#56b4e9",  # sky blue
    "#db7eb1",  # pink
    "#1a84c1",  # blue
]

INK = "#0b0b0b"
MUTED_INK = "#595959"
GRID_COLOUR = "#d9d9d9"

USE_TEX = bool(shutil.which("latex") and shutil.which("dvipng"))
if os.environ.get("PLOT_USETEX") == "0":
    USE_TEX = False

plt.style.use(["science", "grid"])
plt.rcParams.update({
    "text.usetex": USE_TEX,
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "axes.titlepad": 10,
    "axes.labelpad": 5,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.prop_cycle": cycler(color=PALETTE),
    "axes.linewidth": 0.6,
    "axes.edgecolor": MUTED_INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "lines.linewidth": 1.2,
    "lines.solid_capstyle": "round",
    "grid.linestyle": ":",
    "grid.linewidth": 0.5,
    "grid.color": GRID_COLOUR,
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
