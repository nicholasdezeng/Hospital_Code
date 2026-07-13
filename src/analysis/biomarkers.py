"""Figure 9: 关键预后菌群鉴定。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.microbiome import aggregate_taxonomy, relative_abundance
from src.utils.stats import fdr_correct
from src.visualization.style import PALETTE, apply_style, finalize_figure, format_taxon, heatmap_cbar_kw, save_figure


def _consensus_genera(clinical: pd.DataFrame, rel_genus: pd.DataFrame) -> pd.DataFrame:
    methods = {}

    lefse = clinical.attrs.get("lefse_genera", [])
    methods["LEfSe"] = set(lefse)

    y = clinical["delayed_extubation"].astype(int)
    X = rel_genus.fillna(0)
    rf = RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced")
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    methods["RandomForest"] = set(imp.head(10).index)

    corr_rows = []
    pvals = []
    for g in rel_genus.columns:
        sub = pd.concat([rel_genus[g], clinical["extubation_time_min"]], axis=1).dropna()
        if len(sub) < 5:
            rho, p = np.nan, 1.0
        else:
            rho, p = stats.spearmanr(sub.iloc[:, 0], sub.iloc[:, 1])
        corr_rows.append((g, rho, p))
        pvals.append(p)
    corr_df = pd.DataFrame(corr_rows, columns=["genus", "rho", "p"])
    _, q = fdr_correct(corr_df["p"].values)
    corr_df["q"] = q
    methods["Spearman"] = set(corr_df.loc[corr_df["q"] < 0.1, "genus"])

    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(penalty="l1", solver="saga", max_iter=5000, class_weight="balanced", C=0.5)),
        ]
    )
    pipe.fit(X, y)
    coef = pd.Series(pipe.named_steps["clf"].coef_.ravel(), index=X.columns)
    methods["L1-Logistic"] = set(coef[coef.abs() > 1e-6].sort_values(key=np.abs, ascending=False).head(10).index)

    all_genera = sorted(set().union(*methods.values()))
    records = []
    for g in all_genera:
        count = sum(g in s for s in methods.values())
        records.append(
            {
                "genus": g,
                "method_count": count,
                "LEfSe": int(g in methods["LEfSe"]),
                "RandomForest": int(g in methods["RandomForest"]),
                "Spearman": int(g in methods["Spearman"]),
                "L1-Logistic": int(g in methods["L1-Logistic"]),
            }
        )
    return pd.DataFrame(records).sort_values(["method_count", "genus"], ascending=[False, True])


def run_figure9(clinical: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    apply_style()
    asv = clinical.attrs["asv"]
    taxonomy = clinical.attrs["taxonomy"]
    rel_genus = relative_abundance(aggregate_taxonomy(asv, taxonomy, "Genus"))

    consensus = _consensus_genera(clinical, rel_genus)
    key_genera = consensus.loc[consensus["method_count"] >= 2, "genus"].tolist()
    if not key_genera:
        key_genera = consensus.head(4)["genus"].tolist()

    outcome_cols = ["extubation_time_min", "icu_stay_min", "adverse_event", "qor15"]
    outcome_labels = {
        "extubation_time_min": "Extubation",
        "icu_stay_min": "ICU stay",
        "adverse_event": "Adverse event",
        "qor15": "QoR-15",
    }
    corr_records = []
    for g in key_genera:
        for oc in outcome_cols:
            sub = pd.concat([rel_genus[g], clinical[oc]], axis=1).dropna()
            if len(sub) < 5:
                rho, p = np.nan, np.nan
            else:
                rho, p = stats.spearmanr(sub.iloc[:, 0], sub.iloc[:, 1])
            corr_records.append({"genus": g, "outcome": oc, "rho": rho, "p": p})
    corr_out = pd.DataFrame(corr_records)
    _, q = fdr_correct(corr_out["p"].fillna(1).values)
    corr_out["q"] = q

    fig = plt.figure(figsize=(18, 6.5))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.05, 1.15, 1.1], wspace=0.42)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    method_cols = ["LEfSe", "RandomForest", "Spearman", "L1-Logistic"]
    top = consensus.head(12).copy()
    if not top.empty:
        top["genus_label"] = top["genus"].map(format_taxon)
        matrix = top.set_index("genus_label")[method_cols].astype(int)
        sns.heatmap(
            matrix,
            cmap=["#F5F5F5", PALETTE["microbiome"]],
            cbar=False,
            linewidths=0.5,
            linecolor="white",
            ax=axes[0],
        )
        axes[0].set_xlabel("Analysis method")
        axes[0].set_ylabel("")
        axes[0].set_title("A. Cross-method identification matrix", pad=8)
        axes[0].tick_params(axis="y", labelsize=8)
        axes[0].tick_params(axis="x", labelsize=8, rotation=25)
    else:
        axes[0].text(0.5, 0.5, "No consensus genera", ha="center", va="center", transform=axes[0].transAxes)

    if key_genera:
        pivot = corr_out.pivot(index="genus", columns="outcome", values="rho").loc[key_genera]
        pivot.index = [format_taxon(g) for g in pivot.index]
        pivot = pivot.rename(columns=outcome_labels)
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            ax=axes[1],
            linewidths=0.5,
            linecolor="white",
            cbar_kws=heatmap_cbar_kw("Spearman ρ"),
            annot_kws={"size": 8},
        )
        axes[1].set_title("B. Genus-outcome correlations", pad=8)
        axes[1].tick_params(axis="y", labelsize=8)
        axes[1].tick_params(axis="x", labelsize=8, rotation=20)
    else:
        axes[1].text(0.5, 0.5, "No key genera", ha="center", va="center", transform=axes[1].transAxes)

    plot_genera = key_genera[:2] if len(key_genera) >= 2 else key_genera
    if plot_genera:
        plot_rows = []
        for g in plot_genera:
            high_mask = rel_genus[g] >= rel_genus[g].median()
            for sid in clinical.index:
                plot_rows.append({
                    "Genus": format_taxon(g),
                    "Group": "High" if high_mask.loc[sid] else "Low",
                    "ExtubationTime": clinical.loc[sid, "extubation_time_min"],
                    "AE": clinical.loc[sid, "adverse_event"],
                })
        plot_df = pd.DataFrame(plot_rows).dropna(subset=["ExtubationTime"])
        sns.boxplot(
            data=plot_df,
            x="Genus",
            y="ExtubationTime",
            hue="Group",
            palette={"Low": PALETTE["early"], "High": PALETTE["delayed"]},
            width=0.55,
            fliersize=0,
            ax=axes[2],
        )
        sns.stripplot(
            data=plot_df,
            x="Genus",
            y="ExtubationTime",
            hue="Group",
            dodge=True,
            palette={"Low": "#333333", "High": "#333333"},
            alpha=0.45,
            size=3,
            ax=axes[2],
            legend=False,
        )
        ymax = plot_df["ExtubationTime"].max()
        ypad = ymax * 0.06
        for i, g in enumerate([format_taxon(x) for x in plot_genera]):
            for j, ab in enumerate(["Low", "High"]):
                sub = plot_df[(plot_df["Genus"] == g) & (plot_df["Group"] == ab)]
                if len(sub):
                    axes[2].text(
                        i + (-0.22 if ab == "Low" else 0.22),
                        ymax + ypad,
                        f"AE {sub['AE'].mean():.0%}",
                        ha="center",
                        fontsize=7,
                    )
        axes[2].set_ylim(top=ymax + ypad * 2.5)
        handles, labels = axes[2].get_legend_handles_labels()
        if handles:
            axes[2].legend(handles[:2], labels[:2], title="Abundance", loc="upper right", fontsize=8, frameon=False)
        axes[2].set_title("C. High vs low genus abundance — extubation time", pad=8)
        axes[2].set_xlabel("")
        axes[2].set_ylabel("Extubation time (min)")
    else:
        axes[2].text(0.5, 0.5, "No comparison data", ha="center", va="center", transform=axes[2].transAxes)

    finalize_figure(fig, "Figure 9. Identification and characterization of key prognostic microbiota", wspace=0.42)
    save_figure(fig, output_dir, "figure9_key_biomarkers")

    consensus.to_csv(output_dir / "figure9_consensus_genera.csv", index=False, encoding="utf-8-sig")
    corr_out.to_csv(output_dir / "figure9_genus_outcome_correlations.csv", index=False, encoding="utf-8-sig")
    return consensus
