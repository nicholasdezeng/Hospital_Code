"""SCI 论文风格绘图配置。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


PALETTE = {
    "early": "#4C72B0",
    "delayed": "#C44E52",
    "no_ae": "#55A868",
    "yes_ae": "#DD8452",
    "microbiome": "#59A14F",
    "inflammation": "#F28E2B",
    "clinical": "#9DA5B4",
}

PHylum_COLORS = {
    "Firmicutes": "#4C72B0",
    "Proteobacteria": "#C44E52",
    "Bacteroidetes": "#55A868",
    "Actinobacteria": "#8172B3",
    "Fusobacteria": "#CCB974",
    "Other": "#BAB0AC",
}


def apply_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig, output_dir: Path, name: str, fmt: str = "png"):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    return path


def add_stat_box(ax, text: str, x: float = 0.98, y: float = 0.02):
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#CCCCCC"),
    )
