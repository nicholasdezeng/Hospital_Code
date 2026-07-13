"""Figure 1: 菌群组成全景描述。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.microbiome import aggregate_taxonomy, rarefaction_curve, relative_abundance
from src.visualization.style import (
    PHylum_COLORS,
    PALETTE,
    apply_style,
    finalize_figure,
    format_taxon,
    save_figure,
)


def _group_rarefaction(asv: pd.DataFrame, sample_ids: list[str], n_points: int = 30) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median rarefaction curve with IQR band for a sample set."""
    curves: list[tuple[np.ndarray, np.ndarray]] = []
    max_depth = 0
    for sid in sample_ids:
        if sid not in asv.index:
            continue
        depths, obs = rarefaction_curve(asv.loc[sid].values, steps=n_points)
        if len(depths) == 0:
            continue
        max_depth = max(max_depth, int(depths[-1]))
        curves.append((depths, obs))
    if not curves or max_depth <= 0:
        x = np.linspace(0, 1, n_points)
        return x, np.zeros(n_points), np.zeros(n_points)

    grid = np.linspace(1, max_depth, n_points, dtype=int)
    interp = []
    for depths, obs in curves:
        interp.append(np.interp(grid, depths, obs))
    arr = np.vstack(interp)
    return grid, np.median(arr, axis=0), np.percentile(arr, [25, 75], axis=0)


def run_figure1(clinical: pd.DataFrame, output_dir: Path) -> dict:
    apply_style()
    asv = clinical.attrs["asv"]
    taxonomy = clinical.attrs["taxonomy"]
    rel_genus = relative_abundance(aggregate_taxonomy(asv, taxonomy, "Genus"))
    rel_phylum = relative_abundance(aggregate_taxonomy(asv, taxonomy, "Phylum"))

    order = clinical.sort_values("extubation_time_min").index
    rel_phylum = rel_phylum.loc[order]
    top_phyla = rel_phylum.mean().sort_values(ascending=False).head(5).index.tolist()
    plot_phylum = rel_phylum[top_phyla].copy()
    plot_phylum["Other"] = (1 - plot_phylum.sum(axis=1)).clip(lower=0)

    fig = plt.figure(figsize=(18, 6.5))
    gs = gridspec.GridSpec(
        2,
        3,
        figure=fig,
        height_ratios=[1, 0.08],
        width_ratios=[1.55, 1.05, 1.15],
        hspace=0.08,
        wspace=0.42,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_leg = fig.add_subplot(gs[1, 0])
    ax_leg.axis("off")

    # 1A stacked bar — phylum composition
    bottom = np.zeros(len(plot_phylum))
    x = np.arange(len(plot_phylum))
    phylum_labels = [format_taxon(p) for p in plot_phylum.columns]
    handles = []
    for phylum, label in zip(plot_phylum.columns, phylum_labels):
        vals = plot_phylum[phylum].values
        color = PHylum_COLORS.get(phylum, PHylum_COLORS.get(format_taxon(phylum), PHylum_COLORS["Other"]))
        bar = ax_a.bar(x, vals, bottom=bottom, label=label, color=color, width=0.92, edgecolor="none")
        handles.append(bar)
        bottom += vals
    ax_a.set_title("A. Phylum-level composition", pad=10)
    ax_a.set_xlabel("Patients (sorted by extubation time)")
    ax_a.set_ylabel("Relative abundance")
    ax_a.set_xlim(-0.5, len(plot_phylum) - 0.5)
    ax_a.set_ylim(0, 1.02)
    ax_a.legend(
        handles=[h[0] for h in handles],
        labels=phylum_labels,
        loc="center",
        ncol=min(6, len(phylum_labels)),
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.5),
        borderaxespad=0,
    )

    # 1B heatmap — top genera by group (replaces crowded bubble plot)
    top15 = rel_genus.mean().sort_values(ascending=False).head(15).index
    heat = pd.DataFrame(index=top15, columns=["Early", "Delayed"], dtype=float)
    for g in top15:
        for grp in heat.columns:
            sub = rel_genus.loc[clinical["extubation_group"] == grp, g]
            heat.loc[g, grp] = sub.mean() if len(sub) else 0.0
    heat.index = [format_taxon(g) for g in heat.index]
    heat_pct = heat * 100

    sns.heatmap(
        heat_pct,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Mean relative abundance (%)", "shrink": 0.85},
        ax=ax_b,
        vmin=0,
    )
    ax_b.set_title("B. Top 15 genera", pad=10)
    ax_b.set_xlabel("")
    ax_b.set_ylabel("")
    ax_b.tick_params(axis="y", labelsize=8)
    ax_b.set_yticklabels(ax_b.get_yticklabels(), rotation=0)

    # 1C group rarefaction — median + IQR per extubation group
    for grp, color, label in [
        ("Early", PALETTE["early"], "Early extubation"),
        ("Delayed", PALETTE["delayed"], "Delayed extubation"),
    ]:
        ids = clinical.index[clinical["extubation_group"] == grp].tolist()
        grid, median, iqr = _group_rarefaction(asv, ids)
        ax_c.plot(grid, median, color=color, linewidth=2.2, label=label)
        ax_c.fill_between(grid, iqr[0], iqr[1], color=color, alpha=0.18, linewidth=0)
    ax_c.set_title("C. Rarefaction curves", pad=10)
    ax_c.set_xlabel("Sequencing depth (reads)")
    ax_c.set_ylabel("Observed ASVs")
    ax_c.legend(loc="lower right", fontsize=8, frameon=True, fancybox=False, edgecolor="#CCCCCC")

    finalize_figure(
        fig,
        "Figure 1. Overview of respiratory microbiome composition in AICU patients",
        left=0.06,
        bottom=0.14,
    )
    save_figure(fig, output_dir, "figure1_microbiome_overview")

    summary = {
        "n_asv": asv.shape[1],
        "n_genera": rel_genus.shape[1],
        "dominant_phyla": rel_phylum.mean().sort_values(ascending=False).head(3).to_dict(),
    }
    pd.Series(summary).to_csv(output_dir / "figure1_summary.csv", encoding="utf-8-sig")
    return summary
