"""Style graphique unique : palette lisible par les daltoniens (ordre fixe), axes discrets."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

# palette réexportée pour les notebooks (viz.C, viz.MED_COLORS…)
from ..utils.palette import COHORT_COLORS, DIV, GRID, INK, INK2, MED_COLORS, MUTED, SEQ, SERIES, C  # noqa: F401


def set_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": GRID, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 9.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.frameon": False, "legend.fontsize": 8.5,
        "lines.linewidth": 2, "lines.markersize": 5, "font.size": 9.5,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
    })


def save(fig, folder: Path, name: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = Path(folder) / f"{name}.png"
    fig.savefig(path)
    return path


def bar(ax, labels, values, color=C["blue"], horizontal=False, fmt="{:.0f}", label_values=True):
    if horizontal:
        bars = ax.barh(labels, values, color=color, height=0.65, edgecolor="white", linewidth=1)
        ax.grid(axis="y", visible=False)
    else:
        bars = ax.bar(labels, values, color=color, width=0.65, edgecolor="white", linewidth=1)
        ax.grid(axis="x", visible=False)
    if label_values:
        ax.bar_label(bars, labels=[fmt.format(v) for v in values], padding=2, fontsize=8, color=INK2)
    return bars
