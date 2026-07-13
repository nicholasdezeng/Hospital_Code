"""Figure 8 & Table 2: 预测模型（Logistic A/B/C + MLP 概念验证）。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import LeaveOneOut
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.microbiome import aggregate_taxonomy, relative_abundance
from src.utils.stats import bootstrap_auc_ci, delong_auc_test, format_p, permutation_auc_pvalue
from src.visualization.style import PALETTE, apply_style, save_figure


CLINICAL_FEATURES = ["age", "sex", "bmi", "asa", "surgery_duration_min", "anesthesia_duration_min", "opioid_morphine_mg"]
INFLAMMATION_FEATURES = ["wbc", "log_crp", "log_pct", "nlr"]
MICROBIOME_FEATURES = ["shannon", "chao1", "pielou_j"]


def _build_microbiome_features(clinical: pd.DataFrame) -> pd.DataFrame:
    asv = clinical.attrs["asv"]
    taxonomy = clinical.attrs["taxonomy"]
    rel_genus = relative_abundance(aggregate_taxonomy(asv, taxonomy, "Genus"))
    key_genera = ["Pseudomonas", "Klebsiella", "Prevotella", "Veillonella", "Streptococcus", "Haemophilus"]
    for g in key_genera:
        if g in rel_genus.columns:
            clinical[f"genus_{g}"] = rel_genus[g]
    if "pcoa" in clinical.attrs:
        clinical["pcoa_pc1"] = clinical.attrs["pcoa"]["PC1"]
        clinical["pcoa_pc2"] = clinical.attrs["pcoa"]["PC2"]
    micro_cols = [
        c for c in clinical.columns
        if c.startswith("genus_") or c in MICROBIOME_FEATURES + ["pcoa_pc1", "pcoa_pc2"]
    ]
    return clinical, micro_cols


def _loo_predict(X: pd.DataFrame, y: pd.Series, seed: int = 42, use_mlp: bool = False):
    loo = LeaveOneOut()
    probs = np.zeros(len(y))
    preds = np.zeros(len(y))
    if use_mlp:
        clf = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            max_iter=3000,
            random_state=seed,
            early_stopping=False,
        )
    else:
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])
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
    return {
        "AUC": auc,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Sensitivity": np.mean(y_pred[y_true == 1] == 1) if (y_true == 1).any() else np.nan,
        "Specificity": np.mean(y_pred[y_true == 0] == 0) if (y_true == 0).any() else np.nan,
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }


def run_prediction(
    clinical: pd.DataFrame,
    output_dir: Path,
    seed: int = 42,
    n_boot: int = 1000,
    n_perm: int = 500,
) -> pd.DataFrame:
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
    mlp_feats = model_defs["Model C (+Microbiome)"]

    all_metrics = []
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    colors = ["#9DA5B4", "#4C72B0", PALETTE["delayed"]]
    valid_store = {}

    for t_idx, (target_col, target_label) in enumerate(targets.items()):
        ax = axes[0, t_idx]
        valid = clinical[mlp_feats + [target_col]].dropna()
        y = valid[target_col].astype(int)
        valid_store[target_col] = valid

        prob_store = {}
        for (model_name, feats), color in zip(model_defs.items(), colors):
            X = valid[feats]
            probs, preds = _loo_predict(X, y, seed=seed)
            prob_store[model_name] = probs
            m = _metrics(y.values, probs, preds)
            _, auc_low, auc_high = bootstrap_auc_ci(y.values, probs, n_boot=n_boot)
            perm_p = permutation_auc_pvalue(y.values, probs, n_perm=n_perm, seed=seed)
            m.update({
                "Target": target_label,
                "Model": model_name,
                "AUC_CI_low": auc_low,
                "AUC_CI_high": auc_high,
                "Permutation_P": perm_p,
            })
            all_metrics.append(m)
            fpr, tpr, _ = roc_curve(y, probs)
            short = model_name.split("(")[1].strip(")")
            ax.plot(fpr, tpr, label=f"{short}: {m['AUC']:.2f}", color=color)

        # Model D: MLP 概念验证（方案要求）
        mlp_probs, mlp_preds = _loo_predict(valid[mlp_feats], y, seed=seed, use_mlp=True)
        m_mlp = _metrics(y.values, mlp_probs, mlp_preds)
        _, mlp_low, mlp_high = bootstrap_auc_ci(y.values, mlp_probs, n_boot=n_boot)
        mlp_perm = permutation_auc_pvalue(y.values, mlp_probs, n_perm=n_perm, seed=seed)
        m_mlp.update({
            "Target": target_label,
            "Model": "Model D (MLP)",
            "AUC_CI_low": mlp_low,
            "AUC_CI_high": mlp_high,
            "Permutation_P": mlp_perm,
        })
        all_metrics.append(m_mlp)
        fpr_m, tpr_m, _ = roc_curve(y, mlp_probs)
        ax.plot(fpr_m, tpr_m, "--", label=f"MLP: {m_mlp['AUC']:.2f}", color="#8172B3", linewidth=1.8)

        delta, p_ab, _ = delong_auc_test(
            y.values, prob_store["Model A (Clinical)"], prob_store["Model B (+Inflammation)"]
        )
        d_bc, p_bc, _ = delong_auc_test(
            y.values, prob_store["Model B (+Inflammation)"], prob_store["Model C (+Microbiome)"]
        )
        ax.plot([0, 1], [0, 1], ":", color="gray", linewidth=1)
        panel = "A" if t_idx == 0 else "B"
        ax.set_title(f"{panel}. ROC — {target_label}")
        ax.legend(fontsize=6, loc="lower right")
        ax.text(
            0.03, 0.08,
            f"ΔAUC B-A={delta:.2f}, P={format_p(p_ab)}\nΔAUC C-B={d_bc:.2f}, P={format_p(p_bc)}",
            transform=ax.transAxes, fontsize=7,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

    # 8C: SHAP
    ax_shap = axes[1, 0]
    try:
        target_col = "delayed_extubation"
        feats = model_defs["Model C (+Microbiome)"]
        valid = valid_store.get(target_col, clinical[feats + [target_col]].dropna())
        X = valid[feats]
        y = valid[target_col].astype(int)
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)),
        ])
        pipe.fit(X, y)
        X_scaled = pipe.named_steps["scaler"].transform(pipe.named_steps["imputer"].transform(X))
        explainer = shap.LinearExplainer(pipe.named_steps["clf"], X_scaled, feature_names=feats)
        shap_values = explainer.shap_values(X_scaled)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        shap_df = pd.DataFrame({
            "feature": feats,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)
        shap_df.to_csv(output_dir / "figure8_shap_importance.csv", index=False, encoding="utf-8-sig")
        top = shap_df.head(10)
        bar_colors = [
            PALETTE["microbiome"] if (f in micro_cols or f.startswith("genus_") or f.startswith("pcoa"))
            else PALETTE["inflammation"] if f in INFLAMMATION_FEATURES
            else PALETTE["clinical"]
            for f in top["feature"]
        ]
        ax_shap.barh(top["feature"], top["mean_abs_shap"], color=bar_colors)
        ax_shap.invert_yaxis()
        ax_shap.set_title("C. SHAP — Model C (extubation delay)")
    except Exception as exc:
        ax_shap.text(0.5, 0.5, f"SHAP unavailable:\n{exc}", ha="center", va="center")
        pd.Series({"shap_error": str(exc)}).to_csv(output_dir / "figure8_shap_error.csv")

    # 8D: MLP vs Logistic C 对比
    ax_mlp = axes[1, 1]
    cmp_rows = []
    for target_col, target_label in targets.items():
        valid = valid_store[target_col]
        y = valid[target_col].astype(int)
        lr_probs, lr_preds = _loo_predict(valid[mlp_feats], y, seed=seed)
        mlp_probs, mlp_preds = _loo_predict(valid[mlp_feats], y, seed=seed, use_mlp=True)
        cmp_rows.append({"Target": target_label, "Model": "Logistic C", "AUC": _metrics(y.values, lr_probs, lr_preds)["AUC"]})
        cmp_rows.append({"Target": target_label, "Model": "MLP D", "AUC": _metrics(y.values, mlp_probs, mlp_preds)["AUC"]})
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(output_dir / "figure8_mlp_comparison.csv", index=False, encoding="utf-8-sig")
    for i, target_label in enumerate(targets.values()):
        sub = cmp_df[cmp_df["Target"] == target_label]
        xpos = [i * 3, i * 3 + 1]
        ax_mlp.bar(xpos, sub["AUC"].values, color=[PALETTE["delayed"], "#8172B3"], width=0.7)
        ax_mlp.text(xpos[0], sub["AUC"].iloc[0] + 0.02, f"{sub['AUC'].iloc[0]:.2f}", ha="center", fontsize=8)
        ax_mlp.text(xpos[1], sub["AUC"].iloc[1] + 0.02, f"{sub['AUC'].iloc[1]:.2f}", ha="center", fontsize=8)
    ax_mlp.set_xticks([0.5, 3.5])
    ax_mlp.set_xticklabels(list(targets.values()))
    ax_mlp.set_ylim(0, 1.05)
    ax_mlp.set_ylabel("AUC (LOO-CV)")
    ax_mlp.set_title("D. MLP concept validation vs Logistic C")

    fig.suptitle("Figure 8. Predictive models: Logistic A/B/C + MLP validation (LOO-CV)", y=1.01)
    save_figure(fig, output_dir, "figure8_prediction_roc")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(output_dir / "table2_model_performance.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_excel(output_dir / "table2_model_performance.xlsx", index=False)
    return metrics_df
