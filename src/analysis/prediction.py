"""Figure 8 & Table 2: 预测模型（2×2 析因 Model A/B/C/E + MLP 补充）。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

from src.utils.stats import (
    bootstrap_auc_ci,
    bootstrap_delta_auc_ci,
    format_p,
    permutation_auc_pvalue,
)
from src.visualization.style import PALETTE, add_stat_box, apply_style, finalize_figure, save_figure

# 优化方案：与 Table 3 / Model E 对齐的精简特征
CLINICAL_BASE = ["asa", "anesthesia_duration_min"]
INFLAMMATION = ["log_crp"]
MICROBIOME = ["shannon"]

MODEL_A = CLINICAL_BASE
MODEL_B = CLINICAL_BASE + INFLAMMATION
MODEL_C = CLINICAL_BASE + MICROBIOME  # 不含炎症（2×2 析因）
MODEL_E = CLINICAL_BASE + INFLAMMATION + MICROBIOME

MAIN_MODELS = {
    "Model A (Clinical)": MODEL_A,
    "Model B (+Inflammation)": MODEL_B,
    "Model C (+Microbiome)": MODEL_C,
    "Model E (All)": MODEL_E,
}


def _loo_predict(X: pd.DataFrame, y: pd.Series, seed: int = 42, use_mlp: bool = False):
    loo = LeaveOneOut()
    probs = np.zeros(len(y))
    preds = np.zeros(len(y))
    pos_label = 1
    if use_mlp:
        clf = MLPClassifier(
            hidden_layer_sizes=(16, 8),
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
        proba = pipe.predict_proba(X_test)
        classes = list(pipe.named_steps["clf"].classes_)
        pos_idx = classes.index(pos_label) if pos_label in classes else 1
        prob = proba[:, pos_idx][0]
        probs[test_idx[0]] = prob
        preds[test_idx[0]] = int(prob >= 0.5)
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


def _eval_model(name: str, feats: list[str], valid: pd.DataFrame, y: pd.Series, *, n_boot: int, n_perm: int, seed: int):
    probs, preds = _loo_predict(valid[feats], y, seed=seed)
    m = _metrics(y.values, probs, preds)
    _, auc_low, auc_high = bootstrap_auc_ci(y.values, probs, n_boot=n_boot)
    perm_p = permutation_auc_pvalue(y.values, probs, n_perm=n_perm, seed=seed)
    m.update({
        "Target": "Extubation delay",
        "Model": name,
        "AUC_CI_low": auc_low,
        "AUC_CI_high": auc_high,
        "Permutation_P": perm_p,
    })
    return m, probs


def _factorial_table(prob_store: dict[str, np.ndarray], y: np.ndarray, n_boot: int, seed: int) -> pd.DataFrame:
    pairs = [
        ("B - A（炎症单独贡献）", "Model A (Clinical)", "Model B (+Inflammation)"),
        ("C - A（菌群单独贡献）", "Model A (Clinical)", "Model C (+Microbiome)"),
        ("E - B（菌群在炎症基础上增量）", "Model B (+Inflammation)", "Model E (All)"),
        ("E - C（炎症在菌群基础上增量）", "Model C (+Microbiome)", "Model E (All)"),
    ]
    rows = []
    deltas = {}
    for label, a_name, b_name in pairs:
        delta, ci_low, ci_high, p = bootstrap_delta_auc_ci(
            y, prob_store[a_name], prob_store[b_name], n_boot=n_boot, seed=seed
        )
        deltas[label] = delta
        rows.append({
            "比较": label,
            "ΔAUC": f"{delta:.3f}",
            "95%CI": f"{ci_low:.3f}-{ci_high:.3f}",
            "Bootstrap_P": format_p(p),
        })
    interaction = deltas.get("E - B（菌群在炎症基础上增量）", np.nan) - deltas.get("C - A（菌群单独贡献）", np.nan)
    rows.append({
        "比较": "交互效应 = (E-B) - (C-A)",
        "ΔAUC": f"{interaction:.3f}" if pd.notna(interaction) else "NA",
        "95%CI": "—",
        "Bootstrap_P": "—",
    })
    return pd.DataFrame(rows)


def run_prediction(
    clinical: pd.DataFrame,
    output_dir: Path,
    seed: int = 42,
    n_boot: int = 1000,
    n_perm: int = 500,
    tables_dir: Path | None = None,
) -> pd.DataFrame:
    apply_style()
    target_col = "delayed_extubation"
    all_feats = list(dict.fromkeys(MODEL_E))
    valid = clinical[all_feats + [target_col]].dropna()
    y = valid[target_col].astype(int)

    main_metrics = []
    prob_store: dict[str, np.ndarray] = {}
    for name, feats in MAIN_MODELS.items():
        m, probs = _eval_model(name, feats, valid, y, n_boot=n_boot, n_perm=n_perm, seed=seed)
        main_metrics.append(m)
        prob_store[name] = probs

    # 补充：不良反应 + MLP（移出主表）
    supp_metrics = []
    ae_col = "adverse_event"
    if ae_col in clinical.columns:
        ae_valid = clinical[MODEL_E + [ae_col]].dropna()
        if len(ae_valid) >= 10 and ae_valid[ae_col].nunique() == 2:
            y_ae = ae_valid[ae_col].astype(int)
            for name, feats in MAIN_MODELS.items():
                m, _ = _eval_model(
                    name.replace("Extubation delay", "Adverse events"),
                    feats,
                    ae_valid,
                    y_ae,
                    n_boot=n_boot,
                    n_perm=n_perm,
                    seed=seed,
                )
                m["Target"] = "Adverse events"
                m["Model"] = name
                supp_metrics.append(m)
            mlp_probs, mlp_preds = _loo_predict(ae_valid[MODEL_E], y_ae, seed=seed, use_mlp=True)
            m_mlp = _metrics(y_ae.values, mlp_probs, mlp_preds)
            _, low, high = bootstrap_auc_ci(y_ae.values, mlp_probs, n_boot=n_boot)
            m_mlp.update({
                "Target": "Adverse events",
                "Model": "Model D (MLP)",
                "AUC_CI_low": low,
                "AUC_CI_high": high,
                "Permutation_P": permutation_auc_pvalue(y_ae.values, mlp_probs, n_perm=n_perm, seed=seed),
            })
            supp_metrics.append(m_mlp)

    mlp_probs, mlp_preds = _loo_predict(valid[MODEL_E], y, seed=seed, use_mlp=True)
    m_mlp_main = _metrics(y.values, mlp_probs, mlp_preds)
    _, mlp_low, mlp_high = bootstrap_auc_ci(y.values, mlp_probs, n_boot=n_boot)
    m_mlp_main.update({
        "Target": "Extubation delay",
        "Model": "Model D (MLP)",
        "AUC_CI_low": mlp_low,
        "AUC_CI_high": mlp_high,
        "Permutation_P": permutation_auc_pvalue(y.values, mlp_probs, n_perm=n_perm, seed=seed),
    })
    supp_metrics.append(m_mlp_main)

    factorial_df = _factorial_table(prob_store, y.values, n_boot=n_boot, seed=seed)

    # ── Figure 8：主文 A/B/C/E ROC ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = ["#9DA5B4", "#4C72B0", PALETTE["microbiome"], PALETTE["delayed"]]
    ax = axes[0]
    for (name, _), color in zip(MAIN_MODELS.items(), colors):
        fpr, tpr, _ = roc_curve(y, prob_store[name])
        auc = roc_auc_score(y, prob_store[name])
        short = name.split("(")[1].strip(")")
        ax.plot(fpr, tpr, label=f"{short}: {auc:.2f}", color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], ":", color="gray", linewidth=1)
    ax.set_title("A. ROC — Extubation delay (A/B/C/E)", pad=8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(fontsize=8, loc="lower right")
    d_eb = factorial_df[factorial_df["比较"].str.startswith("E - B")]["ΔAUC"].iloc[0]
    d_ba = factorial_df[factorial_df["比较"].str.startswith("B - A")]["ΔAUC"].iloc[0]
    add_stat_box(ax, f"ΔAUC B−A={d_ba}\nΔAUC E−B={d_eb}", x=0.03, y=0.03, ha="left", va="bottom")

    # SHAP on Model E
    ax_shap = axes[1]
    try:
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)),
        ])
        X = valid[MODEL_E]
        pipe.fit(X, y)
        X_scaled = pipe.named_steps["scaler"].transform(pipe.named_steps["imputer"].transform(X))
        explainer = shap.LinearExplainer(pipe.named_steps["clf"], X_scaled, feature_names=MODEL_E)
        shap_values = explainer.shap_values(X_scaled)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        shap_df = pd.DataFrame({
            "feature": MODEL_E,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)
        shap_df.to_csv(output_dir / "figure8_shap_importance.csv", index=False, encoding="utf-8-sig")
        bar_colors = [
            PALETTE["microbiome"] if f == "shannon"
            else PALETTE["inflammation"] if f == "log_crp"
            else PALETTE["clinical"]
            for f in shap_df["feature"]
        ]
        ax_shap.barh(shap_df["feature"], shap_df["mean_abs_shap"], color=bar_colors, height=0.55)
        ax_shap.invert_yaxis()
        ax_shap.set_xlabel("Mean |SHAP|")
        ax_shap.set_title("B. SHAP — Model E", pad=8)
    except Exception as exc:
        ax_shap.text(0.5, 0.5, f"SHAP unavailable:\n{exc}", ha="center", va="center")
        pd.Series({"shap_error": str(exc)}).to_csv(output_dir / "figure8_shap_error.csv")

    finalize_figure(fig, "Figure 8. Factorial models A/B/C/E for extubation delay (LOO-CV)", wspace=0.32)
    save_figure(fig, output_dir, "figure8_prediction_roc")

    # ── 写表 ──
    main_df = pd.DataFrame(main_metrics)
    supp_df = pd.DataFrame(supp_metrics) if supp_metrics else pd.DataFrame()

    def _write(df: pd.DataFrame, stem: str, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        df.to_csv(directory / f"{stem}.csv", index=False, encoding="utf-8-sig")
        df.to_excel(directory / f"{stem}.xlsx", index=False)

    _write(main_df, "table2_model_performance", output_dir)
    _write(factorial_df, "table2_factorial_delta_auc", output_dir)
    if not supp_df.empty:
        _write(supp_df, "table2_model_performance_supplementary", output_dir)

    if tables_dir is not None:
        _write(main_df, "table2_model_performance", tables_dir)
        _write(factorial_df, "table2_factorial_delta_auc", tables_dir)
        if not supp_df.empty:
            _write(supp_df, "table2_model_performance_supplementary", tables_dir)

    # 向后兼容：figures/ 下旧文件名指向主表
    main_df.to_csv(output_dir / "table2_model_performance.csv", index=False, encoding="utf-8-sig")

    return main_df
