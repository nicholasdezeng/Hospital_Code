"""Figure 5 & 6: 菌群-炎症轴分析。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.preprocessing import add_quadrant_groups
from src.utils.stats import format_p, sig_stars, spearman_with_fdr
from src.visualization.style import PALETTE, add_stat_box, apply_style, finalize_figure, heatmap_cbar_kw, save_figure


def run_figure5(clinical: pd.DataFrame, output_dir: Path, fdr_alpha: float = 0.05) -> pd.DataFrame:
    apply_style()
    rows = ["shannon", "chao1", "pielou_j"]
    cols = ["wbc", "crp", "pct", "nlr"]
    corr = spearman_with_fdr(clinical, rows, cols, alpha=fdr_alpha)
    pivot = corr.pivot(index="row", columns="col", values="rho")
    q_pivot = corr.pivot(index="row", columns="col", values="q")

    row_labels = {"shannon": "Shannon", "chao1": "Chao1", "pielou_j": "Pielou J"}
    col_labels = {"wbc": "WBC", "crp": "CRP", "pct": "PCT", "nlr": "NLR"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    annot = pivot.copy().astype(str)
    for i in pivot.index:
        for j in pivot.columns:
            rho = pivot.loc[i, j]
            q = q_pivot.loc[i, j]
            star = sig_stars(q if not pd.isna(q) else np.nan)
            annot.loc[i, j] = f"{rho:.2f}{star}" if not pd.isna(rho) else ""

    pivot_display = pivot.rename(index=row_labels, columns=col_labels)
    annot_display = annot.rename(index=row_labels, columns=col_labels)

    sns.heatmap(
        pivot_display,
        annot=annot_display,
        fmt="",
        cmap="RdBu_r",
        center=0,
        ax=axes[0],
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        cbar_kws=heatmap_cbar_kw("Spearman ρ"),
        annot_kws={"size": 9},
    )
    axes[0].set_title("A. Diversity-inflammation correlation matrix", pad=8)
    axes[0].set_xlabel("Inflammatory markers")
    axes[0].set_ylabel("Alpha diversity")

    sub = clinical.dropna(subset=["shannon", "log_crp"])
    sns.scatterplot(
        data=sub.reset_index(),
        x="shannon",
        y="log_crp",
        color=PALETTE["inflammation"],
        ax=axes[1],
        s=55,
        alpha=0.8,
        edgecolor="white",
        linewidth=0.4,
    )
    if len(sub) >= 5:
        rho, p = stats.spearmanr(sub["shannon"], sub["log_crp"])
        sns.regplot(data=sub, x="shannon", y="log_crp", scatter=False, ax=axes[1], color="#333333")
        axes[1].set_title(f"B. Shannon vs log(CRP) (ρ={rho:.2f}, P={format_p(p)})", pad=8)
    else:
        axes[1].set_title("B. Shannon vs log(CRP)", pad=8)
    axes[1].set_xlabel("Shannon diversity")
    axes[1].set_ylabel("log(CRP+1)")

    finalize_figure(fig, "Figure 5. Correlation between microbiome diversity and inflammatory markers", wspace=0.35)
    save_figure(fig, output_dir, "figure5_inflammation_correlation")
    corr.to_csv(output_dir / "figure5_correlations.csv", index=False, encoding="utf-8-sig")
    return corr


def run_figure6(clinical: pd.DataFrame, output_dir: Path, crp_threshold: float = 10.0) -> pd.DataFrame:
    apply_style()
    df = add_quadrant_groups(clinical, crp_threshold=crp_threshold)
    order = ["HighDiv_LowCRP", "HighDiv_HighCRP", "LowDiv_LowCRP", "LowDiv_HighCRP"]
    short_labels = ["HiDiv\nLoCRP", "HiDiv\nHiCRP", "LoDiv\nLoCRP", "LoDiv\nHiCRP"]
    df["quadrant_group"] = pd.Categorical(df["quadrant_group"], categories=order, ordered=True)
    label_map = dict(zip(order, short_labels))
    df["quadrant_short"] = df["quadrant_group"].map(label_map)

    fig = plt.figure(figsize=(16, 6.5))
    gs = gridspec.GridSpec(1, 4, figure=fig, width_ratios=[1.05, 1.05, 1.05, 1.15], wspace=0.42)
    panel_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_quad = fig.add_subplot(gs[0, 3])

    outcome_specs = [
        ("extubation_time_min", "Extubation time"),
        ("icu_stay_min", "ICU stay"),
        ("qor15", "QoR-15"),
    ]

    for ax, (col, label) in zip(panel_axes, outcome_specs):
        plot_df = df.dropna(subset=[col, "quadrant_short"]).copy()
        if plot_df.empty:
            ax.set_title(label, pad=8)
            continue
        sns.boxplot(
            data=plot_df,
            x="quadrant_short",
            y=col,
            hue="quadrant_short",
            order=short_labels,
            color="#DDDDDD",
            width=0.5,
            fliersize=0,
            ax=ax,
            legend=False,
            dodge=False,
        )
        sns.stripplot(
            data=plot_df,
            x="quadrant_short",
            y=col,
            order=short_labels,
            color="black",
            alpha=0.55,
            size=3,
            ax=ax,
            jitter=0.12,
        )
        ax.set_title(label, fontsize=10, pad=8)
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelsize=7.5, rotation=0)

    if df["quadrant_group"].notna().sum() >= 4:
        groups = [df.loc[df["quadrant_group"] == g, "extubation_time_min"].dropna().values for g in order]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) >= 2:
            _, p = stats.kruskal(*groups)
            add_stat_box(panel_axes[0], f"Kruskal-Wallis\nP={format_p(p)}", x=0.03, y=0.97, ha="left")

    panel_axes[0].text(-0.12, 1.12, "A.", transform=panel_axes[0].transAxes, fontsize=11, fontweight="bold", va="top")

    quad_layout = {
        "HighDiv_LowCRP": (0, 1),
        "HighDiv_HighCRP": (1, 1),
        "LowDiv_LowCRP": (0, 0),
        "LowDiv_HighCRP": (1, 0),
    }
    ae_rate = df.groupby("quadrant_group")["adverse_event"].mean().reindex(order)
    n_grp = df.groupby("quadrant_group").size().reindex(order)
    ax_quad.set_xlim(-0.55, 1.55)
    ax_quad.set_ylim(-0.55, 1.55)
    ax_quad.set_xticks([0, 1])
    ax_quad.set_xticklabels(["Low CRP", "High CRP"])
    ax_quad.set_yticks([0, 1])
    ax_quad.set_yticklabels(["Low diversity", "High diversity"])
    ax_quad.set_xlabel("Inflammation status")
    ax_quad.set_ylabel("Microbiome diversity")
    for grp, (x, y) in quad_layout.items():
        rate = ae_rate.get(grp, np.nan)
        n = n_grp.get(grp, 0)
        if pd.isna(rate):
            continue
        color = plt.cm.RdYlGn_r(rate)
        size = 280 + 520 * min(rate, 1.0)
        ax_quad.scatter(
            x, y, s=size, c=[color], alpha=0.88,
            edgecolors="#333333", linewidths=0.8,
        )
        ax_quad.text(x, y, f"AE {rate:.0%}\nn={int(n)}", ha="center", va="center", fontsize=8)
    ax_quad.set_title("B. Adverse event rate by quadrant", pad=8)

    finalize_figure(fig, "Figure 6. Combined stratification of microbiome diversity and inflammatory status", wspace=0.42)
    save_figure(fig, output_dir, "figure6_quadrant_analysis")

    summary = df.groupby("quadrant_group").agg(
        n=("extubation_time_min", "count"),
        extubation_median=("extubation_time_min", "median"),
        ae_rate=("adverse_event", "mean"),
        qor15_median=("qor15", "median"),
    )
    summary.to_csv(output_dir / "figure6_quadrant_summary.csv", encoding="utf-8-sig")
    return summary
