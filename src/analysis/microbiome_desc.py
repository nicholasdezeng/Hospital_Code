"""Figure 1: 菌群组成全景描述（SCI 论文三面板）。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from src.utils.microbiome import (
    aggregate_taxonomy,
    collapse_phyla_for_plot,
    rarefaction_curve,
    relative_abundance,
)
from src.visualization.style import (
    PHylum_COLORS,
    PALETTE,
    apply_style,
    finalize_figure,
    format_taxon,
    save_figure,
)

# 方案指定关键属（优先展示，若存在于数据中）
KEY_GENERA = [
    "Streptococcus",
    "Pseudomonas",
    "Klebsiella",
    "Veillonella",
    "Prevotella",
    "Acinetobacter",
    "Haemophilus",
    "Neisseria",
    "Rothia",
    "Lactobacillus",
]


def _italic_genus(name: str) -> str:
    """SCI 规范：属名斜体。"""
    clean = format_taxon(name)
    if clean.lower() in ("unknown", "unassigned", "other"):
        return clean
    return rf"$\it{{{clean}}}$"


def _pick_top_genera(rel_genus: pd.DataFrame, n: int = 15) -> list[str]:
    """Top N 属：排除 Unknown，并优先纳入方案关键属。"""
    named = [c for c in rel_genus.columns if format_taxon(c).lower() not in ("unknown", "unassigned", "")]
    pool = rel_genus[named] if named else rel_genus
    ranked = pool.mean().sort_values(ascending=False)

    selected: list[str] = []
    for g in KEY_GENERA:
        if g in ranked.index and g not in selected:
            selected.append(g)
    for g in ranked.index:
        if g not in selected:
            selected.append(g)
        if len(selected) >= n:
            break
    return selected[:n]


def _plot_panel_a(ax, plot_phylum: pd.DataFrame, groups: pd.Series) -> list[mpatches.Patch]:
    """门水平堆叠柱状图，纵轴 0–100%，按拔管时间排序。"""
    n = len(plot_phylum)
    x = np.arange(n)
    bottom = np.zeros(n)
    handles: list[mpatches.Patch] = []

    for phylum in plot_phylum.columns:
        vals = plot_phylum[phylum].values * 100
        color = PHylum_COLORS.get(phylum, PHylum_COLORS["Other"])
        ax.bar(x, vals, bottom=bottom, color=color, width=0.92, edgecolor="none", linewidth=0)
        bottom += vals
        handles.append(mpatches.Patch(facecolor=color, edgecolor="none", label=phylum))

    n_early = int((groups == "Early").sum())
    if 0 < n_early < n:
        ax.axvline(n_early - 0.5, color="#666666", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.axvspan(-0.5, n_early - 0.5, color=PALETTE["early"], alpha=0.04, linewidth=0)
        ax.axvspan(n_early - 0.5, n - 0.5, color=PALETTE["delayed"], alpha=0.04, linewidth=0)
        y_top = 103
        ax.text((n_early - 1) / 2, y_top, "Early", ha="center", va="bottom", fontsize=8, color=PALETTE["early"])
        ax.text(n_early + (n - n_early - 1) / 2, y_top, "Delayed", ha="center", va="bottom", fontsize=8, color=PALETTE["delayed"])

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_title("A. Phylum-level composition", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Patients (sorted by extubation time)", fontsize=10)
    ax.set_ylabel("Relative abundance (%)", fontsize=10)
    ax.tick_params(axis="both", labelsize=8)
    return handles


def _plot_panel_b(ax, rel_genus: pd.DataFrame, clinical: pd.DataFrame, top_genera: list[str]):
    """属水平 Top15 气泡图：Early vs Delayed。"""
    rows = []
    for genus in top_genera:
        for grp, xpos in [("Early", 0), ("Delayed", 1)]:
            mask = clinical["extubation_group"] == grp
            sub = rel_genus.loc[mask, genus]
            mean_ab = float(sub.mean()) if len(sub) else 0.0
            rows.append({"Genus": genus, "Group": grp, "x": xpos, "Abundance": mean_ab * 100})
    bubble_df = pd.DataFrame(rows)

    genus_order = top_genera[::-1]
    y_map = {g: i for i, g in enumerate(genus_order)}
    bubble_df["y"] = bubble_df["Genus"].map(y_map)
    bubble_df["bubble_size"] = np.sqrt(bubble_df["Abundance"].clip(lower=0)) * 55 + 15

    vmax = max(bubble_df["Abundance"].max(), 1.0)
    sc = ax.scatter(
        bubble_df["x"],
        bubble_df["y"],
        s=bubble_df["bubble_size"],
        c=bubble_df["Abundance"],
        cmap="YlOrRd",
        alpha=0.88,
        edgecolors="#444444",
        linewidths=0.4,
        vmin=0,
        vmax=vmax,
    )

    ax.set_yticks(range(len(genus_order)))
    ax.set_yticklabels([_italic_genus(g) for g in genus_order], fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Early", "Delayed"], fontsize=10)
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(-0.6, len(genus_order) - 0.4)
    ax.set_title("B. Top 15 genera (bubble plot)", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Extubation group", fontsize=10)
    ax.set_ylabel("")

    cbar = plt.colorbar(sc, ax=ax, shrink=0.72, pad=0.02)
    cbar.set_label("Mean relative abundance (%)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ref_vals = [1, 5, 15]
    size_legend = ax.scatter(
        [], [], s=0, c=[], alpha=0,
    )
    size_handles = [
        ax.scatter([], [], s=np.sqrt(v) * 55 + 15, c="#888888", alpha=0.65, edgecolors="#444444", linewidths=0.4)
        for v in ref_vals
    ]
    ax.legend(
        size_handles,
        [f"{v}%" for v in ref_vals],
        title="Bubble size",
        loc="lower left",
        fontsize=7,
        title_fontsize=7,
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
        labelspacing=1.2,
        borderpad=0.6,
    )
    del size_legend


def _plot_panel_c(ax, asv: pd.DataFrame, clinical: pd.DataFrame, ref_depth: int = 30000):
    """稀释曲线：每样本一条线，按拔管组着色。"""
    for sid in asv.index:
        if sid not in clinical.index:
            continue
        grp = clinical.loc[sid, "extubation_group"]
        color = PALETTE["early"] if grp == "Early" else PALETTE["delayed"]
        depths, obs = rarefaction_curve(asv.loc[sid].values, steps=35)
        ax.plot(depths, obs, color=color, alpha=0.38, linewidth=0.75)

    max_depth = float(asv.sum(axis=1).max())
    if max_depth >= ref_depth * 0.5:
        ax.axvline(ref_depth, color="#888888", linestyle=":", linewidth=1, alpha=0.8)
        ax.text(ref_depth, ax.get_ylim()[1] * 0.97, f"{ref_depth:,} reads", ha="right", va="top", fontsize=7, color="#666666")

    ax.set_title("C. Rarefaction curves", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Sequencing depth (reads)", fontsize=10)
    ax.set_ylabel("Observed ASVs", fontsize=10)
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(
        handles=[
            mpatches.Patch(facecolor=PALETTE["early"], label="Early extubation"),
            mpatches.Patch(facecolor=PALETTE["delayed"], label="Delayed extubation"),
        ],
        loc="lower right",
        fontsize=8,
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
    )


def run_figure1(clinical: pd.DataFrame, output_dir: Path) -> dict:
    apply_style()
    asv = clinical.attrs["asv"]
    taxonomy = clinical.attrs["taxonomy"]
    rel_genus = relative_abundance(aggregate_taxonomy(asv, taxonomy, "Genus"))
    rel_phylum = relative_abundance(aggregate_taxonomy(asv, taxonomy, "Phylum"))

    order = clinical.sort_values("extubation_time_min").index
    groups = clinical.loc[order, "extubation_group"]
    plot_phylum = collapse_phyla_for_plot(rel_phylum.loc[order])
    top_genera = _pick_top_genera(rel_genus, n=15)

    # SCI 双栏宽度 ~180 mm ≈ 7.1 inch；三面板横向排列
    fig = plt.figure(figsize=(11.5, 4.8))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 0.11], width_ratios=[1.5, 1.05, 1.1], hspace=0.05, wspace=0.48)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    phylum_handles = _plot_panel_a(ax_a, plot_phylum, groups)
    _plot_panel_b(ax_b, rel_genus, clinical, top_genera)
    _plot_panel_c(ax_c, asv.loc[order], clinical.loc[order])

    # 门水平图例：全图底部居中，不遮挡数据
    fig.legend(
        handles=phylum_handles,
        loc="lower center",
        bbox_to_anchor=(0.36, 0.01),
        ncol=len(phylum_handles),
        frameon=False,
        fontsize=8,
        columnspacing=1.2,
        handlelength=1.2,
    )

    finalize_figure(
        fig,
        "Figure 1. Overview of respiratory microbiome composition in AICU patients",
        top=0.88,
        bottom=0.20,
        left=0.07,
        right=0.98,
    )
    save_figure(fig, output_dir, "figure1_microbiome_overview")

    phylum_means = (plot_phylum.mean() * 100).round(2).to_dict()
    summary = {
        "n_asv": asv.shape[1],
        "n_genera": rel_genus.shape[1],
        "n_samples": len(order),
        "phylum_means_pct": phylum_means,
        "top_genera": top_genera,
    }
    pd.Series(summary).to_csv(output_dir / "figure1_summary.csv", encoding="utf-8-sig")
    return summary
