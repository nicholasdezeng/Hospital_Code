"""Figure 8 & Table 2: 三组预测模型对比。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.microbiome import aggregate_taxonomy, relative_abundance
from src.utils.stats import bootstrap_auc_ci, delong_auc_test, format_p
from src.visualization.style import PALETTE, apply_style, save_figure


CLINICAL_FEATURES = ["age", "sex", "bmi", "asa", "surgery_duration_min", "anesthesia_duration_min", "opioid_morphine_mg"]
INFLAMMATION_FEATURES = ["wbc", "log_crp", "log_pct", "nlr"]
MICROBIOME_FEATURES = ["shannon", "chao1", "pielou_j"]


def _build_microbiome_features(clinical: pd.DataFrame) -> pd.DataFrame:
    asv = clinical.attrs["asv"]
    taxonomy = clinical.attrs["taxonomy"]
    rel_genus = relative_abundance(aggregate_taxonomy(asv, taxonomy, "Genus"))
    key_genera = ["Pseudomonas", "Klebsiella", "Prevotella", "Veillonella"]
    for g in key_genera:
        if g in rel_genus.columns:
            clinical[f"genus_{g}"] = rel_genus[g]
    if "pcoa" in clinical.attrs:
        clinical["pcoa_pc1"] = clinical.attrs["pcoa"]["PC1"]
        clinical["pcoa_pc2"] = clinical.attrs["pcoa"]["PC2"]
    micro_cols = [c for c in clinical.columns if c.startswith("genus_") or c in MICROBIOME_FEATURES + ["pcoa_pc1", "pcoa_pc2"]]
    return clinical, micro_cols


def _loo_predict(X: pd.DataFrame, y: pd.Series, seed: int = 42):
    loo = LeaveOneOut()
    probs = np.zeros(len(y))
    preds = np.zeros(len(y))
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)),
        ]
    )
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1]
        probs[test_idx[0]] = prob[0]
        preds[test_idx[0]] = int(prob[0] >= 0.5)
    return probs, preds


def _metrics(y_true, y_prob, y_pred):
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan
    acc = accuracy_score(y_true, y_pred)
    sens = np.mean(y_pred[y_true == 1] == 1) if (y_true == 1).any() else np.nan
    spec = np.mean(y_pred[y_true == 0] == 0) if (y_true == 0).any() else np.nan
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {"AUC": auc, "Accuracy": acc, "Sensitivity": sens, "Specificity": spec, "F1": f1}


def run_prediction(clinical: pd.DataFrame, output_dir: Path, seed: int = 42, n_boot: int = 1000) -> pd.DataFrame:
    apply_style()
    clinical, micro_cols = _build_microbiome_features(clinical)

    targets = {
        "delayed_extubation": "Extubation delay",
        "adverse_event": "Adverse events",
    }
    model_defs = {
        "Model A (Clinical)": CLINICAL_FEATURES,
        "Model B (+Inflammation)": CLINICAL_FEATURES + INFLAMMATION_FEATURES,
        "Model C (+Microbiome)": CLINICAL_FEATURES + INFLAMMATION_FEATURES + micro_cols,
    }

    all_metrics = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#9DA5B4", "#4C72B0", PALETTE["delayed"]]

    for t_idx, (target_col, target_label) in enumerate(targets.items()):
        ax = axes[t_idx]
        y = clinical[target_col].astype(int)
        valid = clinical[model_defs["Model C (+Microbiome)"] + [target_col]].dropna()
        y = valid[target_col].astype(int)

        prob_store = {}
        for (model_name, feats), color in zip(model_defs.items(), colors):
            X = valid[feats]
            probs, preds = _loo_predict(X, y, seed=seed)
            prob_store[model_name] = probs
            m = _metrics(y.values, probs, preds)
            _, auc_low, auc_high = bootstrap_auc_ci(y.values, probs, n_boot=n_boot)
            m.update(
                {
                    "Target": target_label,
                    "Model": model_name,
                    "AUC_CI_low": auc_low,
                    "AUC_CI_high": auc_high,
                }
            )
            all_metrics.append(m)
            fpr, tpr, _ = roc_curve(y, probs)
            ax.plot(fpr, tpr, label=f"{model_name.split('(')[1].strip(')')}: AUC={m['AUC']:.2f}", color=color)

        delta, p_ab, _ = delong_auc_test(y.values, prob_store["Model A (Clinical)"], prob_store["Model B (+Inflammation)"])
        _, p_bc, _ = delong_auc_test(y.values, prob_store["Model B (+Inflammation)"], prob_store["Model C (+Microbiome)"])
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        ax.set_title(f"{chr(65 + t_idx)}. ROC — {target_label}")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(fontsize=7, loc="lower right")
        ax.text(
            0.55,
            0.08,
            f"ΔAUC B-A={delta:.2f}, P={format_p(p_ab)}\nΔAUC C-B={delong_auc_test(y.values, prob_store['Model B (+Inflammation)'], prob_store['Model C (+Microbiome)'])[0]:.2f}, P={format_p(p_bc)}",
            transform=ax.transAxes,
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

    fig.suptitle("Figure 8. Predictive performance comparison of three models (LOO-CV)", y=1.02)
    save_figure(fig, output_dir, "figure8_prediction_roc")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(output_dir / "table2_model_performance.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_excel(output_dir / "table2_model_performance.xlsx", index=False)

    # SHAP for model C on delayed extubation
    try:
        target_col = "delayed_extubation"
        feats = model_defs["Model C (+Microbiome)"]
        valid = clinical[feats + [target_col]].dropna()
        X = valid[feats]
        y = valid[target_col].astype(int)
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)),
            ]
        )
        pipe.fit(X, y)
        X_imp = pipe.named_steps["imputer"].transform(X)
        X_scaled = pipe.named_steps["scaler"].transform(X_imp)
        explainer = shap.LinearExplainer(pipe.named_steps["clf"], X_scaled, feature_names=feats)
        shap_values = explainer.shap_values(X_scaled)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        mean_abs = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({"feature": feats, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
        shap_df.to_csv(output_dir / "figure8_shap_importance.csv", index=False, encoding="utf-8-sig")

        fig2, ax2 = plt.subplots(figsize=(8, 6))
        colors_map = []
        for f in shap_df["feature"]:
            if f in micro_cols or f.startswith("genus_") or f.startswith("pcoa"):
                colors_map.append("#59A14F")
            elif f in INFLAMMATION_FEATURES:
                colors_map.append("#F28E2B")
            else:
                colors_map.append("#9DA5B4")
        ax2.barh(shap_df["feature"], shap_df["mean_abs_shap"], color=colors_map)
        ax2.invert_yaxis()
        ax2.set_xlabel("Mean |SHAP|")
        ax2.set_title("Feature importance — Model C")
        save_figure(fig2, output_dir, "figure8c_shap_importance")
    except Exception as exc:
        pd.Series({"shap_error": str(exc)}).to_csv(output_dir / "figure8_shap_error.csv")

    return metrics_df
