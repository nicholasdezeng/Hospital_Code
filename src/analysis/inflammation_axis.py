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

    sub = clinical.dropna(subset=["shannon", "log_crp", "extubation_group"])
    sns.scatterplot(
        data=sub.reset_index(),
        x="shannon",
        y="log_crp",
        hue="extubation_group",
        palette={"Early": PALETTE["early"], "Delayed": PALETTE["delayed"]},
        ax=axes[1],
        s=60,
    )
    if len(sub) >= 5:
        rho, p = stats.spearmanr(sub["shannon"], sub["log_crp"])
        sns.regplot(data=sub, x="shannon", y="log_crp", scatter=False, ax=axes[1], color="gray")
        axes[1].set_title(f"B. Shannon vs log(CRP) (ρ={rho:.2f}, P={format_p(p)})")
    else:
        axes[1].set_title("B. Shannon vs log(CRP)")

    fig.suptitle("Figure 5. Correlation between microbiome diversity and inflammatory markers", y=1.02)
    save_figure(fig, output_dir, "figure5_inflammation_correlation")
    corr.to_csv(output_dir / "figure5_correlations.csv", index=False, encoding="utf-8-sig")
    return corr


def run_figure6(clinical: pd.DataFrame, output_dir: Path, crp_threshold: float = 10.0) -> pd.DataFrame:
    apply_style()
    df = add_quadrant_groups(clinical, crp_threshold=crp_threshold)
    order = ["HighDiv_LowCRP", "HighDiv_HighCRP", "LowDiv_LowCRP", "LowDiv_HighCRP"]
    df["quadrant_group"] = pd.Categorical(df["quadrant_group"], categories=order, ordered=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    outcome_specs = [
        ("extubation_time_min", "Extubation time (min)"),
        ("icu_stay_min", "ICU stay (min)"),
        ("qor15", "QoR-15"),
    ]
    plot_rows = []
    for col, label in outcome_specs:
        for grp in order:
            vals = df.loc[df["quadrant_group"] == grp, col].dropna()
            plot_rows.append({"Quadrant": grp, "Outcome": label, "Value": vals.median() if len(vals) else np.nan})
    plot_df = pd.DataFrame(plot_rows)

    sns.barplot(data=plot_df, x="Quadrant", y="Value", hue="Outcome", ax=axes[0])
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set_title("A. Outcomes by quadrant groups")

    ae_rate = df.groupby("quadrant_group")["adverse_event"].mean().reindex(order)
    bubble = pd.DataFrame({"Quadrant": ae_rate.index, "AE_rate": ae_rate.values})
    sns.scatterplot(
        data=bubble,
        x="Quadrant",
        y="AE_rate",
        size="AE_rate",
        sizes=(100, 900),
        color=PALETTE["delayed"],
        ax=axes[1],
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_title("B. Adverse event rate by quadrant")
    axes[1].tick_params(axis="x", rotation=20)

    if df["quadrant_group"].notna().sum() >= 4:
        groups = [df.loc[df["quadrant_group"] == g, "extubation_time_min"].dropna().values for g in order]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) >= 2:
            h, p = stats.kruskal(*groups)
            axes[0].text(0.02, 0.95, f"Kruskal-Wallis P={format_p(p)}", transform=axes[0].transAxes)

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
