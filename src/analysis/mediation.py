"""Figure 7: 中介分析 (Shannon -> log(CRP) -> extubation time)。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.visualization.style import apply_style, save_figure


def _bootstrap_mediation(df: pd.DataFrame, n_boot: int = 1000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    cols = ["shannon", "log_crp", "extubation_time_min"]
    data = df[cols].dropna()
    n = len(data)
    if n < 10:
        return {"error": "insufficient samples"}

    def _fit(sub):
        X_a = sm.add_constant(sub["shannon"])
        model_a = sm.OLS(sub["log_crp"], X_a).fit()
        a = model_a.params["shannon"]

        X_bc = sm.add_constant(sub[["shannon", "log_crp"]])
        model_bc = sm.OLS(sub["extubation_time_min"], X_bc).fit()
        b = model_bc.params["log_crp"]
        c_prime = model_bc.params["shannon"]
        c = sm.OLS(sub["extubation_time_min"], sm.add_constant(sub["shannon"])).fit().params["shannon"]
        ab = a * b
        return a, b, c, c_prime, ab

    a, b, c, c_prime, ab = _fit(data)
    boot_ab = []
    idx = np.arange(n)
    for _ in range(n_boot):
        bidx = rng.choice(idx, size=n, replace=True)
        sub = data.iloc[bidx]
        try:
            boot_ab.append(_fit(sub)[4])
        except Exception:
            continue
    boot_ab = np.array(boot_ab)
    ci_low, ci_high = np.quantile(boot_ab, [0.025, 0.975]) if len(boot_ab) else (np.nan, np.nan)

    return {
        "a": a,
        "b": b,
        "c_total": c,
        "c_direct": c_prime,
        "ab_indirect": ab,
        "ab_ci_low": ci_low,
        "ab_ci_high": ci_high,
        "mediation_proportion": ab / c if c != 0 else np.nan,
        "indirect_significant": ci_low > 0 or ci_high < 0,
    }


def run_figure7(clinical: pd.DataFrame, output_dir: Path, n_boot: int = 1000) -> dict:
    apply_style()
    result = _bootstrap_mediation(clinical, n_boot=n_boot)
    if "error" in result:
        pd.Series(result).to_csv(output_dir / "figure7_mediation.csv", encoding="utf-8-sig")
        return result

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].axis("off")
    text = (
        f"Path a: Shannon → log(CRP): β={result['a']:.2f}\n"
        f"Path b: log(CRP) → Extubation time: β={result['b']:.2f}\n"
        f"Total effect (c): {result['c_total']:.1f}\n"
        f"Direct effect (c'): {result['c_direct']:.1f}\n"
        f"Indirect effect (ab): {result['ab_indirect']:.1f}\n"
        f"95% Bootstrap CI: [{result['ab_ci_low']:.1f}, {result['ab_ci_high']:.1f}]\n"
        f"Mediation proportion: {100 * result['mediation_proportion']:.1f}%"
    )
    axes[0].text(0.05, 0.5, text, fontsize=11, va="center")
    axes[0].set_title("A. Mediation path summary")

    rng = np.random.default_rng(42)
    cols = ["shannon", "log_crp", "extubation_time_min"]
    data = clinical[cols].dropna()
    boots = []
    idx = np.arange(len(data))
    for _ in range(n_boot):
        bidx = rng.choice(idx, size=len(idx), replace=True)
        sub = data.iloc[bidx]
        X_a = sm.add_constant(sub["shannon"])
        a = sm.OLS(sub["log_crp"], X_a).fit().params["shannon"]
        X_bc = sm.add_constant(sub[["shannon", "log_crp"]])
        b = sm.OLS(sub["extubation_time_min"], X_bc).fit().params["log_crp"]
        boots.append(a * b)
    axes[1].hist(boots, bins=30, color="#4C72B0", alpha=0.85)
    axes[1].axvline(result["ab_indirect"], color="black", linestyle="--", label="Indirect effect")
    axes[1].axvline(result["ab_ci_low"], color="red", linestyle=":")
    axes[1].axvline(result["ab_ci_high"], color="red", linestyle=":")
    axes[1].axvline(0, color="green", linestyle="-", linewidth=1)
    axes[1].set_title("B. Bootstrap distribution of indirect effect")
    axes[1].set_xlabel("Indirect effect (ab)")
    axes[1].legend()

    fig.suptitle("Figure 7. Mediation analysis: microbiome diversity, CRP, and extubation time", y=1.02)
    save_figure(fig, output_dir, "figure7_mediation")
    pd.Series(result).to_csv(output_dir / "figure7_mediation.csv", encoding="utf-8-sig")
    return result
