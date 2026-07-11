"""Figure 1: 菌群组成全景描述。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.microbiome import aggregate_taxonomy, rarefaction_curve, relative_abundance
from src.visualization.style import PHylum_COLORS, PALETTE, add_stat_box, apply_style, save_figure


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
    other = 1 - plot_phylum.sum(axis=1)
    plot_phylum["Other"] = other.clip(lower=0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1A stacked bar
    bottom = np.zeros(len(plot_phylum))
    x = np.arange(len(plot_phylum))
    for phylum in plot_phylum.columns:
        vals = plot_phylum[phylum].values
        color = PHylum_COLORS.get(phylum, PHylum_COLORS["Other"])
        axes[0].bar(x, vals, bottom=bottom, label=phylum, color=color, width=1.0, edgecolor="none")
        bottom += vals
    axes[0].set_title("A. Phylum-level composition")
    axes[0].set_xlabel("Patients (sorted by extubation time)")
    axes[0].set_ylabel("Relative abundance")
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)

    # 1B bubble top genera
    top15 = rel_genus.mean().sort_values(ascending=False).head(15).index
    bubble = []
    for g in top15:
        for grp in ["Early", "Delayed"]:
            sub = rel_genus.loc[clinical["extubation_group"] == grp, g]
            bubble.append({"Genus": g, "Group": grp, "Abundance": sub.mean()})
    bubble_df = pd.DataFrame(bubble)
    sns.scatterplot(
        data=bubble_df,
        x="Group",
        y="Genus",
        size="Abundance",
        hue="Abundance",
        palette="viridis",
        sizes=(50, 800),
        ax=axes[1],
        legend=False,
    )
    axes[1].set_title("B. Top 15 genera")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")

    # 1C rarefaction
    for sid in asv.index[: min(15, len(asv))]:
        depths, obs = rarefaction_curve(asv.loc[sid].values)
        axes[2].plot(depths, obs, alpha=0.5, linewidth=1)
    axes[2].set_title("C. Rarefaction curves")
    axes[2].set_xlabel("Sequencing depth (reads)")
    axes[2].set_ylabel("Observed ASVs")

    fig.suptitle("Figure 1. Overview of respiratory microbiome composition in AICU patients", y=1.02)
    save_figure(fig, output_dir, "figure1_microbiome_overview")

    summary = {
        "n_asv": asv.shape[1],
        "n_genera": rel_genus.shape[1],
        "dominant_phyla": rel_phylum.mean().sort_values(ascending=False).head(3).to_dict(),
    }
    pd.Series(summary).to_csv(output_dir / "figure1_summary.csv", encoding="utf-8-sig")
    return summary
