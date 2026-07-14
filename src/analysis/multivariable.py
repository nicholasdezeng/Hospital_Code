"""Table 3 扩展：多因素 Logistic + 拔管时间连续变量回归。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.utils.stats import format_p

# 与 Model E 对齐的精简协变量（优化方案）
CORE_PREDICTORS = ["asa", "anesthesia_duration_min"]
INFLAMMATION_PREDICTOR = "log_crp"
MICROBIOME_PREDICTOR = "shannon"


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


def _fit_ols(df: pd.DataFrame, outcome: str, predictors: list[str]) -> pd.DataFrame:
    cols = [outcome] + predictors
    sub = df[cols].dropna()
    if len(sub) < len(predictors) + 5:
        return pd.DataFrame()

    y = sub[outcome]
    X = sm.add_constant(sub[predictors])
    try:
        model = sm.OLS(y, X).fit()
    except Exception:
        return pd.DataFrame()

    rows = []
    for var in predictors:
        coef = model.params[var]
        ci = model.conf_int().loc[var]
        rows.append(
            {
                "outcome": outcome,
                "variable": var,
                "beta": float(coef),
                "CI_low": float(ci[0]),
                "CI_high": float(ci[1]),
                "P_value": float(model.pvalues[var]),
                "N": len(sub),
                "R2": float(model.rsquared),
            }
        )
    return pd.DataFrame(rows)


def run_multivariable(clinical: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """控制 ASA、麻醉时长后，评估 Shannon 与 log(CRP) 的独立效应。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    base = CORE_PREDICTORS
    model_specs = [
        ("adverse_event", "不良反应", base + [MICROBIOME_PREDICTOR, INFLAMMATION_PREDICTOR]),
        ("adverse_event", "不良反应", base + [INFLAMMATION_PREDICTOR]),
        ("delayed_extubation", "拔管延迟", base + [MICROBIOME_PREDICTOR, INFLAMMATION_PREDICTOR]),
        ("delayed_extubation", "拔管延迟", base + [INFLAMMATION_PREDICTOR]),
    ]

    frames = []
    for outcome, label, preds in model_specs:
        res = _fit_logistic(clinical, outcome, preds)
        if res.empty:
            continue
        res["outcome_label"] = label
        res["model"] = "Full" if MICROBIOME_PREDICTOR in preds else "Inflammation-adjusted"
        frames.append(res)

    if not frames:
        empty = pd.DataFrame(
            columns=["outcome", "outcome_label", "model", "variable", "OR", "CI_low", "CI_high", "P_value", "N"]
        )
        empty.to_csv(output_dir / "table3_multivariable_or.csv", index=False, encoding="utf-8-sig")
        return empty

    out = pd.concat(frames, ignore_index=True)
    out["P_fmt"] = out["P_value"].map(format_p)
    out.to_csv(output_dir / "table3_multivariable_or.csv", index=False, encoding="utf-8-sig")
    out.to_excel(output_dir / "table3_multivariable_or.xlsx", index=False)
    return out


def run_continuous_extubation(clinical: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """拔管时间连续变量：log(拔管时间) ~ Shannon + log(CRP) + ASA + 麻醉时长。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = clinical.copy()
    df["log_extubation_time"] = np.log(pd.to_numeric(df["extubation_time_min"], errors="coerce"))
    predictors = CORE_PREDICTORS + [MICROBIOME_PREDICTOR, INFLAMMATION_PREDICTOR]
    res = _fit_ols(df, "log_extubation_time", predictors)
    if res.empty:
        res.to_csv(output_dir / "table2_continuous_extubation.csv", index=False, encoding="utf-8-sig")
        return res

    var_cn = {
        "shannon": "Shannon指数",
        "log_crp": "log(CRP)",
        "asa": "ASA分级",
        "anesthesia_duration_min": "麻醉时长(min)",
    }
    res["outcome_label"] = "log(拔管时间)"
    res["变量"] = res["variable"].map(lambda v: var_cn.get(v, v))
    res["β(95%CI)"] = res.apply(lambda r: f"{r['beta']:.3f} ({r['CI_low']:.3f}-{r['CI_high']:.3f})", axis=1)
    res["P_fmt"] = res["P_value"].map(format_p)
    out = res[["outcome_label", "变量", "β(95%CI)", "P_fmt", "R2", "N"]].rename(
        columns={"P_fmt": "P值", "R2": "R²"}
    )
    out.to_csv(output_dir / "table2_continuous_extubation.csv", index=False, encoding="utf-8-sig")
    out.to_excel(output_dir / "table2_continuous_extubation.xlsx", index=False)
    return out
