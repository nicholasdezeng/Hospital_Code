"""Figure 2 & 3: α/β 多样性分析。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA

from src.utils.stats import compare_continuous, format_p, permanova, sig_stars
from src.visualization.style import PALETTE, add_stat_box, apply_style, finalize_figure, save_figure


def _boxplot_with_points(ax, data, x, y, palette):
    sns.boxplot(data=data, x=x, y=y, hue=x, palette=palette, ax=ax, width=0.5, fliersize=0, legend=False, dodge=False)
    sns.stripplot(data=data, x=x, y=y, color="black", alpha=0.55, size=4, ax=ax, jitter=0.12)
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=9)


def run_figure2(clinical: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    apply_style()
    df = clinical.reset_index().rename(columns={"index": "Sample"})
    if "Sample" not in df.columns and clinical.index.name:
        df = clinical.reset_index().rename(columns={clinical.index.name: "Sample"})

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

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
    axes[0, 0].set_title(f"A. Shannon × Extubation group ({sig_stars(p)})", pad=8)
    axes[0, 0].set_ylabel("Shannon index")
    add_stat_box(axes[0, 0], f"P = {format_p(p)}", x=0.98, y=0.98)

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
    axes[0, 1].set_title(f"B. Shannon × Adverse events ({sig_stars(p2)})", pad=8)
    axes[0, 1].set_ylabel("Shannon index")
    add_stat_box(axes[0, 1], f"P = {format_p(p2)}", x=0.98, y=0.98)

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
    axes[1, 0].set_title(f"C. Chao1 × Extubation group ({sig_stars(p3)})", pad=8)
    axes[1, 0].set_ylabel("Chao1 index")
    add_stat_box(axes[1, 0], f"P = {format_p(p3)}", x=0.98, y=0.98)

    # 2D Shannon vs QoR-15
    sub = df.dropna(subset=["shannon", "qor15"])
    sns.regplot(
        data=sub,
        x="shannon",
        y="qor15",
        ax=axes[1, 1],
        scatter_kws={"alpha": 0.75, "s": 45, "color": PALETTE["clinical"]},
        line_kws={"color": "#333333"},
    )
    if len(sub) >= 5:
        rho, p4 = stats.spearmanr(sub["shannon"], sub["qor15"])
        axes[1, 1].set_title(f"D. Shannon × QoR-15 (ρ={rho:.2f}, P={format_p(p4)})", pad=8)
    else:
        axes[1, 1].set_title("D. Shannon × QoR-15", pad=8)
    axes[1, 1].set_xlabel("Shannon index")
    axes[1, 1].set_ylabel("QoR-15 score")

    finalize_figure(fig, "Figure 2. Alpha diversity comparisons between outcome groups")
    save_figure(fig, output_dir, "figure2_alpha_diversity")

    from src.utils.stats import cliffs_delta

    early = clinical[clinical["extubation_group"] == "Early"]
    delayed = clinical[clinical["extubation_group"] == "Delayed"]
    stats_df = pd.DataFrame([
        {"comparison": "Shannon Early vs Delayed", "p": p, "cliffs_delta": cliffs_delta(early["shannon"], delayed["shannon"])},
        {"comparison": "Shannon AE yes vs no", "p": p2, "cliffs_delta": cliffs_delta(
            clinical.loc[clinical["adverse_event"] == 0, "shannon"],
            clinical.loc[clinical["adverse_event"] == 1, "shannon"],
        )},
        {"comparison": "Chao1 Early vs Delayed", "p": p3, "cliffs_delta": cliffs_delta(early["chao1"], delayed["chao1"])},
    ])
    stats_df.to_csv(output_dir / "figure2_alpha_stats.csv", index=False, encoding="utf-8-sig")
    return stats_df


def run_figure3(clinical: pd.DataFrame, output_dir: Path, permutations: int = 999) -> dict:
    apply_style()
    dist = clinical.attrs["bray_curtis"]
    groups = clinical["extubation_group"].values
    sample_ids = clinical.index.tolist()

    d_sq = -0.5 * (dist ** 2)
    d_sq -= d_sq.mean(axis=0)
    d_sq -= d_sq.mean(axis=1)
    d_sq += d_sq.mean()
    coords = PCA(n_components=2).fit_transform(d_sq)
    pcoa = pd.DataFrame(coords, index=sample_ids, columns=["PC1", "PC2"])
    pcoa = pcoa.join(clinical[["extubation_group", "adverse_event"]])

    perm = permanova(dist, groups, permutations=permutations)

    perm_df = pd.DataFrame([{
        "F": perm["F"], "R2": perm["R2"], "p": perm["p"],
        "permutations": permutations, "n_samples": len(sample_ids),
    }])
    perm_df.to_csv(output_dir / "figure3_permanova.csv", index=False, encoding="utf-8-sig")

    from src.utils.stats import betadisper_test
    bd = betadisper_test(dist, groups)
    pd.DataFrame([{
        "test": "betadisper",
        "p": bd["p"],
        "statistic": bd["statistic"],
        "groups": ",".join(bd["groups"]),
    }]).to_csv(output_dir / "figure3_betadisper.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    palette = {"Early": PALETTE["early"], "Delayed": PALETTE["delayed"]}
    markers = {0: "o", 1: "^"}
    for grp, sub in pcoa.groupby("extubation_group"):
        for ae, ss in sub.groupby("adverse_event"):
            axes[0].scatter(
                ss["PC1"],
                ss["PC2"],
                c=palette[grp],
                marker=markers[int(ae)],
                s=55,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.4,
            )
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].set_title("A. PCoA (Bray-Curtis)", pad=8)
    add_stat_box(
        axes[0],
        f"PERMANOVA\nF={perm['F']:.2f}, R²={perm['R2']:.3f}\nP={format_p(perm['p'])}",
        x=0.03,
        y=0.97,
        ha="left",
    )
    group_handles = [
        mlines.Line2D([], [], color=palette["Early"], marker="o", linestyle="", markersize=7, label="Early"),
        mlines.Line2D([], [], color=palette["Delayed"], marker="o", linestyle="", markersize=7, label="Delayed"),
        mlines.Line2D([], [], color="#555555", marker="o", linestyle="", markersize=7, label="AE: No"),
        mlines.Line2D([], [], color="#555555", marker="^", linestyle="", markersize=7, label="AE: Yes"),
    ]
    axes[0].legend(handles=group_handles, loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=4, fontsize=8, frameon=False)

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
    sns.boxplot(
        data=dist_df,
        x="Type",
        y="Distance",
        hue="Type",
        palette={"Within": PALETTE["early"], "Between": PALETTE["delayed"]},
        ax=axes[1],
        width=0.45,
        fliersize=0,
        legend=False,
        dodge=False,
    )
    sns.stripplot(data=dist_df, x="Type", y="Distance", color="black", alpha=0.35, size=3, ax=axes[1], jitter=0.12)
    axes[1].set_title("B. Within vs between group distances", pad=8)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Bray-Curtis distance")

    finalize_figure(fig, "Figure 3. Beta diversity analysis of respiratory microbiome", bottom=0.22)
    save_figure(fig, output_dir, "figure3_beta_diversity")

    clinical.attrs["pcoa"] = pcoa
    return perm
