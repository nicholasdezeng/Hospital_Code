"""Figure 9: 关键预后菌群鉴定。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.microbiome import aggregate_taxonomy, relative_abundance
from src.utils.stats import compare_continuous, fdr_correct, format_p, spearman_with_fdr
from src.visualization.style import PALETTE, apply_style, save_figure


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

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 9A: 多方法交叉矩阵（区别于 Figure 4 的 LEfSe 条形图）
    method_cols = ["LEfSe", "RandomForest", "Spearman", "L1-Logistic"]
    top = consensus.head(12)
    if not top.empty:
        matrix = top.set_index("genus")[method_cols].astype(int)
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
        axes[0].set_title("A. Cross-method identification matrix")
    else:
        axes[0].text(0.5, 0.5, "No consensus genera", ha="center")

    # 9B: 共识菌属-预后关联热图（保留，与 Figure 5 的多样性-炎症矩阵区分）
    if key_genera:
        pivot = corr_out.pivot(index="genus", columns="outcome", values="rho").loc[key_genera]
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=axes[1])
        axes[1].set_title("B. Genus-outcome correlations")
    else:
        axes[1].text(0.5, 0.5, "No key genera", ha="center")

    # 9C: 关键菌属高/低丰度 — 拔管时间 + AE 率标注
    plot_genera = key_genera[:2] if len(key_genera) >= 2 else key_genera
    if plot_genera:
        plot_rows = []
        for g in plot_genera:
            high_mask = rel_genus[g] >= rel_genus[g].median()
            for sid in clinical.index:
                plot_rows.append({
                    "Genus": g,
                    "Group": "High" if high_mask.loc[sid] else "Low",
                    "ExtubationTime": clinical.loc[sid, "extubation_time_min"],
                    "AE": clinical.loc[sid, "adverse_event"],
                })
        plot_df = pd.DataFrame(plot_rows).dropna(subset=["ExtubationTime"])
        sns.boxplot(data=plot_df, x="Genus", y="ExtubationTime", hue="Group", ax=axes[2])
        ymax = plot_df["ExtubationTime"].max()
        for i, g in enumerate(plot_genera):
            for j, ab in enumerate(["Low", "High"]):
                sub = plot_df[(plot_df["Genus"] == g) & (plot_df["Group"] == ab)]
                if len(sub):
                    axes[2].text(
                        i + (-0.18 if ab == "Low" else 0.18), ymax * 1.02,
                        f"AE {sub['AE'].mean():.0%}", ha="center", fontsize=7,
                    )
        axes[2].set_title("C. High vs low genus abundance — extubation time")
    else:
        axes[2].text(0.5, 0.5, "No comparison data", ha="center")

    fig.suptitle("Figure 9. Identification and characterization of key prognostic microbiota", y=1.02)
    save_figure(fig, output_dir, "figure9_key_biomarkers")

    consensus.to_csv(output_dir / "figure9_consensus_genera.csv", index=False, encoding="utf-8-sig")
    corr_out.to_csv(output_dir / "figure9_genus_outcome_correlations.csv", index=False, encoding="utf-8-sig")
    return consensus
