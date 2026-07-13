"""Table 3: 多因素 Logistic 回归（控制混杂因素）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.utils.stats import format_p


def _fit_logistic(df: pd.DataFrame, outcome: str, predictors: list[str]) -> pd.DataFrame:
    cols = [outcome] + predictors
    sub = df[cols].dropna()
    if len(sub) < len(predictors) + 5:
        return pd.DataFrame()

    y = sub[outcome].astype(int)
    X = sm.add_constant(sub[predictors])
    try:
        model = sm.Logit(y, X).fit(disp=0, maxiter=200)
    except Exception:
        return pd.DataFrame()

    rows = []
    for var in predictors:
        coef = model.params[var]
        or_val = float(np.exp(coef))
        ci = model.conf_int().loc[var]
        rows.append(
            {
                "outcome": outcome,
                "variable": var,
                "OR": or_val,
                "CI_low": float(np.exp(ci[0])),
                "CI_high": float(np.exp(ci[1])),
                "P_value": float(model.pvalues[var]),
                "N": len(sub),
                "AIC": float(model.aic),
            }
        )
    return pd.DataFrame(rows)


def run_multivariable(clinical: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """方案要求：控制 ASA、麻醉时长后，评估 Shannon 与 log(CRP) 的独立效应。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    model_specs = [
        ("adverse_event", "不良反应", ["shannon", "log_crp", "asa", "anesthesia_duration_min"]),
        ("adverse_event", "不良反应", ["log_crp", "asa", "anesthesia_duration_min"]),
        ("delayed_extubation", "拔管延迟", ["shannon", "log_crp", "asa", "anesthesia_duration_min"]),
        ("delayed_extubation", "拔管延迟", ["log_crp", "asa", "anesthesia_duration_min"]),
    ]

    frames = []
    for outcome, label, preds in model_specs:
        res = _fit_logistic(clinical, outcome, preds)
        if res.empty:
            continue
        res["outcome_label"] = label
        res["model"] = "Full" if "shannon" in preds else "Inflammation-adjusted"
        frames.append(res)

    if not frames:
        empty = pd.DataFrame(columns=["outcome", "outcome_label", "model", "variable", "OR", "CI_low", "CI_high", "P_value", "N"])
        empty.to_csv(output_dir / "table3_multivariable_or.csv", index=False, encoding="utf-8-sig")
        return empty

    out = pd.concat(frames, ignore_index=True)
    out["P_fmt"] = out["P_value"].map(format_p)
    out.to_csv(output_dir / "table3_multivariable_or.csv", index=False, encoding="utf-8-sig")
    out.to_excel(output_dir / "table3_multivariable_or.xlsx", index=False)
    return out
