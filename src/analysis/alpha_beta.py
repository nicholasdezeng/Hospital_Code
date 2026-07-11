"""Figure 2 & 3: α/β 多样性分析。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA

from src.utils.stats import compare_continuous, format_p, permanova, sig_stars
from src.visualization.style import PALETTE, add_stat_box, apply_style, save_figure


def _boxplot_with_points(ax, data, x, y, palette):
    sns.boxplot(data=data, x=x, y=y, palette=palette, ax=ax, width=0.55, fliersize=0)
    sns.stripplot(data=data, x=x, y=y, color="black", alpha=0.55, size=4, ax=ax, jitter=0.15)


def run_figure2(clinical: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    apply_style()
    df = clinical.reset_index().rename(columns={"sample_id": "Sample"})
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # 2A Shannon vs extubation group
    _boxplot_with_points(
        axes[0, 0],
        df,
        "extubation_group",
        "shannon",
        {"Early": PALETTE["early"], "Delayed": PALETTE["delayed"]},
    )
    p, _ = compare_continuous(
        clinical.loc[clinical["extubation_group"] == "Early", "shannon"],
        clinical.loc[clinical["extubation_group"] == "Delayed", "shannon"],
    )
    axes[0, 0].set_title(f"A. Shannon × Extubation group ({sig_stars(p)})")
    add_stat_box(axes[0, 0], f"P = {format_p(p)}")

    # 2B Shannon vs adverse event
    df["AE"] = np.where(df["adverse_event"] == 1, "Yes", "No")
    _boxplot_with_points(
        axes[0, 1],
        df,
        "AE",
        "shannon",
        {"No": PALETTE["no_ae"], "Yes": PALETTE["yes_ae"]},
    )
    p2, _ = compare_continuous(
        clinical.loc[clinical["adverse_event"] == 0, "shannon"],
        clinical.loc[clinical["adverse_event"] == 1, "shannon"],
    )
    axes[0, 1].set_title(f"B. Shannon × Adverse events ({sig_stars(p2)})")
    add_stat_box(axes[0, 1], f"P = {format_p(p2)}")

    # 2C Chao1 vs extubation group
    _boxplot_with_points(
        axes[1, 0],
        df,
        "extubation_group",
        "chao1",
        {"Early": PALETTE["early"], "Delayed": PALETTE["delayed"]},
    )
    p3, _ = compare_continuous(
        clinical.loc[clinical["extubation_group"] == "Early", "chao1"],
        clinical.loc[clinical["extubation_group"] == "Delayed", "chao1"],
    )
    axes[1, 0].set_title(f"C. Chao1 × Extubation group ({sig_stars(p3)})")
    add_stat_box(axes[1, 0], f"P = {format_p(p3)}")

    # 2D Shannon vs QoR-15
    sub = df.dropna(subset=["shannon", "qor15"])
    sns.regplot(data=sub, x="shannon", y="qor15", ax=axes[1, 1], scatter_kws={"alpha": 0.7})
    if len(sub) >= 5:
        rho, p4 = stats.spearmanr(sub["shannon"], sub["qor15"])
        axes[1, 1].set_title(f"D. Shannon × QoR-15 (ρ={rho:.2f}, P={format_p(p4)})")
    else:
        axes[1, 1].set_title("D. Shannon × QoR-15")

    fig.suptitle("Figure 2. Alpha diversity comparisons between outcome groups", y=1.02)
    save_figure(fig, output_dir, "figure2_alpha_diversity")

    return pd.DataFrame(
        [
            {"comparison": "Shannon Early vs Delayed", "p": p},
            {"comparison": "Shannon AE yes vs no", "p": p2},
            {"comparison": "Chao1 Early vs Delayed", "p": p3},
        ]
    )


def run_figure3(clinical: pd.DataFrame, output_dir: Path, permutations: int = 999) -> dict:
    apply_style()
    dist = clinical.attrs["bray_curtis"]
    groups = clinical["extubation_group"].values
    sample_ids = clinical.index.tolist()

    # PCoA via PCA on distance matrix (classical MDS approximation)
    d_sq = -0.5 * (dist ** 2)
    d_sq -= d_sq.mean(axis=0)
    d_sq -= d_sq.mean(axis=1)
    d_sq += d_sq.mean()
    coords = PCA(n_components=2).fit_transform(d_sq)
    pcoa = pd.DataFrame(coords, index=sample_ids, columns=["PC1", "PC2"])
    pcoa = pcoa.join(clinical[["extubation_group", "adverse_event"]])

    perm = permanova(dist, groups, permutations=permutations)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    palette = {"Early": PALETTE["early"], "Delayed": PALETTE["delayed"]}
    markers = {0: "o", 1: "^"}
    for grp, sub in pcoa.groupby("extubation_group"):
        for ae, ss in sub.groupby("adverse_event"):
            axes[0].scatter(
                ss["PC1"],
                ss["PC2"],
                c=palette[grp],
                marker=markers[int(ae)],
                s=60,
                alpha=0.85,
                label=f"{grp}, AE={'Yes' if ae else 'No'}",
            )
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].set_title("A. PCoA (Bray-Curtis)")
    add_stat_box(
        axes[0],
        f"PERMANOVA\nF={perm['F']:.2f}, R²={perm['R2']:.3f}\nP={format_p(perm['p'])}",
        x=0.02,
        y=0.98,
    )
    axes[0].legend(fontsize=7, loc="best")

    # within vs between distances
    within, between = [], []
    for i, sid_i in enumerate(sample_ids):
        for j, sid_j in enumerate(sample_ids):
            if i >= j:
                continue
            if groups[i] == groups[j]:
                within.append(dist[i, j])
            else:
                between.append(dist[i, j])
    dist_df = pd.DataFrame(
        {"Distance": within + between, "Type": ["Within"] * len(within) + ["Between"] * len(between)}
    )
    sns.boxplot(data=dist_df, x="Type", y="Distance", palette=["#4C72B0", "#C44E52"], ax=axes[1])
    axes[1].set_title("B. Within vs between group distances")

    fig.suptitle("Figure 3. Beta diversity analysis of respiratory microbiome", y=1.02)
    save_figure(fig, output_dir, "figure3_beta_diversity")

    clinical.attrs["pcoa"] = pcoa
    return perm
