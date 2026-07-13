"""Figure 5 & 6: 菌群-炎症轴分析。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.preprocessing import add_quadrant_groups
from src.utils.stats import format_p, sig_stars, spearman_with_fdr
from src.visualization.style import PALETTE, apply_style, save_figure


def run_figure5(clinical: pd.DataFrame, output_dir: Path, fdr_alpha: float = 0.05) -> pd.DataFrame:
    apply_style()
    rows = ["shannon", "chao1", "pielou_j"]
    cols = ["wbc", "crp", "pct", "nlr"]
    corr = spearman_with_fdr(clinical, rows, cols, alpha=fdr_alpha)
    pivot = corr.pivot(index="row", columns="col", values="rho")
    q_pivot = corr.pivot(index="row", columns="col", values="q")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    annot = pivot.copy().astype(str)
    for i in pivot.index:
        for j in pivot.columns:
            rho = pivot.loc[i, j]
            q = q_pivot.loc[i, j]
            star = sig_stars(q if not pd.isna(q) else np.nan)
            annot.loc[i, j] = f"{rho:.2f}{star}" if not pd.isna(rho) else ""

    sns.heatmap(pivot, annot=annot, fmt="", cmap="RdBu_r", center=0, ax=axes[0], vmin=-1, vmax=1)
    axes[0].set_title("A. Diversity-inflammation correlation matrix")
    axes[0].set_xlabel("Inflammatory markers")
    axes[0].set_ylabel("Alpha diversity")

    sub = clinical.dropna(subset=["shannon", "log_crp"])
    sns.scatterplot(
        data=sub.reset_index(),
        x="shannon",
        y="log_crp",
        color=PALETTE["inflammation"],
        ax=axes[1],
        s=60,
        alpha=0.8,
    )
    if len(sub) >= 5:
        rho, p = stats.spearmanr(sub["shannon"], sub["log_crp"])
        sns.regplot(data=sub, x="shannon", y="log_crp", scatter=False, ax=axes[1], color="#333333")
        axes[1].set_title(f"B. Shannon vs log(CRP) (ρ={rho:.2f}, P={format_p(p)})")
    else:
        axes[1].set_title("B. Shannon vs log(CRP)")
    axes[1].set_xlabel("Shannon diversity")
    axes[1].set_ylabel("log(CRP+1)")

    fig.suptitle("Figure 5. Correlation between microbiome diversity and inflammatory markers", y=1.02)
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

    fig = plt.figure(figsize=(14, 5.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.2])
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
            ax.set_title(label)
            continue
        sns.boxplot(
            data=plot_df,
            x="quadrant_short",
            y=col,
            order=short_labels,
            color="#DDDDDD",
            width=0.55,
            fliersize=0,
            ax=ax,
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
            jitter=0.15,
        )
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelsize=8)

    if df["quadrant_group"].notna().sum() >= 4:
        groups = [df.loc[df["quadrant_group"] == g, "extubation_time_min"].dropna().values for g in order]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) >= 2:
            _, p = stats.kruskal(*groups)
            panel_axes[0].text(
                0.02, 0.98, f"Kruskal-Wallis P={format_p(p)}",
                transform=panel_axes[0].transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
            )
    fig.text(0.02, 0.98, "A. Outcomes by quadrant", fontsize=11, fontweight="bold", va="top")

    # 6B: 2×2 象限气泡图（按方案布局）
    quad_layout = {
        "HighDiv_LowCRP": (0, 1),
        "HighDiv_HighCRP": (1, 1),
        "LowDiv_LowCRP": (0, 0),
        "LowDiv_HighCRP": (1, 0),
    }
    ae_rate = df.groupby("quadrant_group")["adverse_event"].mean().reindex(order)
    n_grp = df.groupby("quadrant_group").size().reindex(order)
    ax_quad.set_xlim(-0.5, 1.5)
    ax_quad.set_ylim(-0.5, 1.5)
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
        ax_quad.scatter(
            x, y, s=400 + 1200 * rate, c=[color], alpha=0.85,
            edgecolors="#333333", linewidths=0.8,
        )
        ax_quad.text(x, y, f"{rate:.0%}\nn={int(n)}", ha="center", va="center", fontsize=8)
    ax_quad.set_title("B. Adverse event rate by quadrant")

    fig.suptitle("Figure 6. Combined stratification of microbiome diversity and inflammatory status", y=1.02)
    save_figure(fig, output_dir, "figure6_quadrant_analysis")

    summary = df.groupby("quadrant_group").agg(
        n=("extubation_time_min", "count"),
        extubation_median=("extubation_time_min", "median"),
        ae_rate=("adverse_event", "mean"),
        qor15_median=("qor15", "median"),
    )
    summary.to_csv(output_dir / "figure6_quadrant_summary.csv", encoding="utf-8-sig")
    return summary
