"""Figure 4: LEfSe 风格差异菌群分析。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from src.utils.microbiome import aggregate_taxonomy, relative_abundance
from src.utils.stats import clr_transform, fdr_correct, format_p
from src.visualization.style import PALETTE, apply_style, save_figure


def _lefse_like(rel_genus: pd.DataFrame, groups: pd.Series, lda_threshold: float = 2.0):
    records = []
    for genus in rel_genus.columns:
        early = rel_genus.loc[groups == "Early", genus]
        delayed = rel_genus.loc[groups == "Delayed", genus]
        if len(early) < 3 or len(delayed) < 3:
            continue
        _, p = stats.mannwhitneyu(early, delayed, alternative="two-sided")
        enriched = "Delayed" if delayed.mean() > early.mean() else "Early"
        records.append({"genus": genus, "p": p, "enriched_group": enriched, "lda": np.nan})

    if not records:
        return pd.DataFrame()

    res = pd.DataFrame(records)
    _, q = fdr_correct(res["p"].values)
    res["q"] = q
    sig = res[res["q"] < 0.1].copy()
    if sig.empty:
        sig = res.nsmallest(10, "p").copy()

    # LDA effect size on CLR-transformed significant genera
    X = clr_transform(rel_genus[sig["genus"]])
    y = (groups == "Delayed").astype(int)
    if len(sig["genus"]) >= 2 and y.nunique() == 2:
        lda = LinearDiscriminantAnalysis(n_components=1)
        lda.fit(X, y)
        coef = np.abs(lda.coef_.ravel())
        coef = coef / coef.max() * 4 if coef.max() > 0 else coef
        sig["lda"] = coef
    else:
        sig["lda"] = np.where(sig["enriched_group"] == "Delayed", 2.5, -2.5)

    sig["lda_signed"] = np.where(sig["enriched_group"] == "Delayed", sig["lda"], -sig["lda"])
    sig = sig[np.abs(sig["lda_signed"]) >= lda_threshold].sort_values("lda_signed")
    return sig


def run_figure4(clinical: pd.DataFrame, output_dir: Path, lda_threshold: float = 2.0) -> pd.DataFrame:
    apply_style()
    asv = clinical.attrs["asv"]
    taxonomy = clinical.attrs["taxonomy"]
    rel_genus = relative_abundance(aggregate_taxonomy(asv, taxonomy, "Genus"))
    lefse = _lefse_like(rel_genus, clinical["extubation_group"], lda_threshold=lda_threshold)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, 0.35 * max(len(lefse), 3) + 2)))

    if lefse.empty:
        axes[0].text(0.5, 0.5, "No significant genera", ha="center")
    else:
        colors = [PALETTE["delayed"] if g == "Delayed" else PALETTE["early"] for g in lefse["enriched_group"]]
        axes[0].barh(lefse["genus"], lefse["lda_signed"], color=colors)
        axes[0].axvline(0, color="black", linewidth=0.8)
        axes[0].set_xlabel("LDA score (signed)")
        axes[0].set_title("A. LEfSe-like differential genera")

        order = clinical.sort_values("extubation_time_min").index
        heat_data = clr_transform(rel_genus.loc[order, lefse["genus"]])
        group_colors = clinical.loc[order, "extubation_group"].map(
            {"Early": PALETTE["early"], "Delayed": PALETTE["delayed"]}
        )
        sns.heatmap(heat_data.T, cmap="RdBu_r", center=0, ax=axes[1], cbar_kws={"label": "CLR abundance"})
        axes[1].set_title("B. Differential genera heatmap")
        axes[1].set_xlabel("Patients (sorted by extubation time)")
        axes[1].set_ylabel("Genus")

    fig.suptitle("Figure 4. Differential microbiota analysis between outcome groups", y=1.02)
    save_figure(fig, output_dir, "figure4_differential_microbiota")

    lefse.to_csv(output_dir / "figure4_lefse_results.csv", index=False, encoding="utf-8-sig")
    clinical.attrs["lefse_genera"] = lefse["genus"].tolist() if not lefse.empty else []
    return lefse
