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
    "Bacillota": "#4C72B0",
    "Proteobacteria": "#C44E52",
    "Pseudomonadota": "#C44E52",
    "Bacteroidetes": "#55A868",
    "Bacteroidota": "#55A868",
    "Actinobacteria": "#8172B3",
    "Actinomycetota": "#8172B3",
    "Fusobacteria": "#CCB974",
    "Fusobacteriota": "#CCB974",
    "Unknown": "#E15759",
    "Other": "#BAB0AC",
}

# 多面板图默认边距（避免 suptitle / 图例 / 轴标签重叠）
FIG_MARGINS = dict(top=0.90, bottom=0.12, left=0.08, right=0.98, hspace=0.42, wspace=0.40)


def apply_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
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


def finalize_figure(
    fig,
    suptitle: str | None = None,
    y: float = 0.98,
    fontsize: float = 13,
    **margins,
):
    """统一设置总标题与子图间距，减少面板重叠。"""
    if suptitle:
        fig.suptitle(suptitle, y=y, fontsize=fontsize)
    fig.subplots_adjust(**{**FIG_MARGINS, **margins})


def format_taxon(label: str) -> str:
    """去掉 SILVA 前缀，缩短轴标签。"""
    text = str(label)
    for prefix in ("k__", "p__", "c__", "o__", "f__", "g__", "s__"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    return text.replace("_", " ")


def truncate_label(text: str, max_len: int = 24) -> str:
    text = str(text)
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def add_stat_box(ax, text: str, x: float = 0.98, y: float = 0.98, ha: str = "right", va: str = "top"):
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.92, edgecolor="#CCCCCC"),
    )


def legend_outside(ax, *, loc="upper left", bbox=(1.02, 1.0), fontsize=8, title=None, ncol=1, frameon=False):
    """将图例放到坐标轴外侧，避免遮挡数据。"""
    leg = ax.legend(
        loc=loc,
        bbox_to_anchor=bbox,
        fontsize=fontsize,
        title=title,
        ncol=ncol,
        frameon=frameon,
        borderaxespad=0,
    )
    return leg


def heatmap_cbar_kw(label: str = "", shrink: float = 0.82):
    return {"label": label, "shrink": shrink, "pad": 0.02}
