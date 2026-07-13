"""Figure 4: LEfSe 风格差异菌群分析。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from src.utils.microbiome import aggregate_taxonomy, relative_abundance
from src.utils.stats import clr_transform, fdr_correct, format_p
from src.visualization.style import PALETTE, apply_style, finalize_figure, format_taxon, heatmap_cbar_kw, save_figure


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

    n_rows = max(len(lefse), 3)
    fig = plt.figure(figsize=(15, max(5.5, 0.42 * n_rows + 2.2)))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.35], wspace=0.38)
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[0, 1])

    if lefse.empty:
        ax_bar.text(0.5, 0.5, "No significant genera", ha="center", va="center", transform=ax_bar.transAxes)
        ax_bar.set_title("A. LEfSe-like differential genera", pad=8)
        ax_heat.axis("off")
    else:
        lefse = lefse.copy()
        lefse["genus_label"] = lefse["genus"].map(format_taxon)
        colors = [PALETTE["delayed"] if g == "Delayed" else PALETTE["early"] for g in lefse["enriched_group"]]
        ax_bar.barh(lefse["genus_label"], lefse["lda_signed"], color=colors, height=0.65)
        ax_bar.axvline(0, color="black", linewidth=0.8)
        ax_bar.set_xlabel("LDA score (signed)")
        ax_bar.set_title("A. LEfSe-like differential genera", pad=8)
        ax_bar.tick_params(axis="y", labelsize=9)

        order = clinical.sort_values("extubation_time_min").index
        heat_data = clr_transform(rel_genus.loc[order, lefse["genus"]])
        heat_data.columns = lefse["genus_label"].tolist()
        sns.heatmap(
            heat_data.T,
            cmap="RdBu_r",
            center=0,
            ax=ax_heat,
            cbar_kws=heatmap_cbar_kw("CLR abundance"),
            xticklabels=False,
            yticklabels=True,
        )
        ax_heat.set_title("B. Differential genera heatmap", pad=8)
        ax_heat.set_xlabel(f"Patients (n={len(order)}, sorted by extubation time)")
        ax_heat.set_ylabel("Genus")
        ax_heat.tick_params(axis="y", labelsize=9)

    finalize_figure(fig, "Figure 4. Differential microbiota analysis between outcome groups", wspace=0.38)
    save_figure(fig, output_dir, "figure4_differential_microbiota")

    lefse.to_csv(output_dir / "figure4_lefse_results.csv", index=False, encoding="utf-8-sig")
    clinical.attrs["lefse_genera"] = lefse["genus"].tolist() if not lefse.empty else []
    return lefse
