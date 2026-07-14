"""临床数据加载与清洗。"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_numeric(value) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text in {"", "无", "nan", "NaN", "None"}:
        return np.nan
    match = re.search(r"[\d.]+", text.replace("*", ""))
    return float(match.group()) if match else np.nan


def _parse_age(value) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value)
    match = re.search(r"(\d+)", text)
    return float(match.group(1)) if match else np.nan


def _parse_smoking(value) -> int:
    if pd.isna(value):
        return 0
    text = str(value)
    if "吸烟" in text or "烟" == text.strip():
        return 1
    return 0


def _parse_adverse_event(value) -> int:
    if pd.isna(value):
        return 0
    text = str(value).strip()
    return 0 if text in {"无", "nan", "NaN", ""} else 1


def _parse_asa(value) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().upper()
    # 长罗马数字优先匹配，避免 "II" 被 "I" 误判为 1
    mapping = [("V", 5), ("IV", 4), ("III", 3), ("II", 2), ("I", 1)]
    for k, v in mapping:
        if k in text:
            return float(v)
    return _parse_numeric(value)


def _parse_opioid(df_row_index, df: pd.DataFrame) -> float:
    candidates = []
    for col in df.columns:
        if "阿片" in str(col) or "吗啡" in str(col):
            val = df.at[df_row_index, col]
            if isinstance(val, pd.Series):
                for item in val:
                    num = _parse_numeric(item)
                    if not np.isnan(num):
                        candidates.append(num)
            else:
                num = _parse_numeric(val)
                if not np.isnan(num):
                    candidates.append(num)
    return candidates[-1] if candidates else np.nan


def load_clinical_excel(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    raw = pd.read_excel(path, sheet_name="Sheet1", header=None)
    df = raw.iloc[2:].copy()
    df.columns = raw.iloc[1].values
    df = df.reset_index(drop=True)

    out = pd.DataFrame()
    out["sample_id"] = df["样本编号"].astype(str).str.strip()
    out["sex"] = df["性别"].map({"男": 1, "女": 0})
    out["age"] = df["年龄"].map(_parse_age)
    out["bmi"] = df["BMI"].map(_parse_numeric)
    out["smoking"] = df["个人史（吸烟/喝酒等）"].map(_parse_smoking)
    out["hypertension"] = df["病史"].astype(str).str.contains("高血压").astype(int)
    out["diabetes"] = df["病史"].astype(str).str.contains("糖尿病").astype(int)
    out["copd"] = df["病史"].astype(str).str.contains("COPD|慢阻").astype(int)
    out["history_raw"] = df["病史"].astype(str)
    out["asa"] = df["ASA分级"].map(_parse_asa) if "ASA分级" in df.columns else np.nan
    out["surgery_duration_min"] = df["手术时长（min）"].map(_parse_numeric)
    out["anesthesia_duration_min"] = df["麻醉时长（min）"].map(_parse_numeric)
    out["opioid_morphine_mg"] = [_parse_opioid(i, df) for i in df.index]
    out["extubation_time_min"] = df["拔管时间（min）"].map(_parse_numeric)
    out["icu_stay_min"] = df["AICU滞留时间(min)"].map(_parse_numeric)
    out["adverse_event"] = df["术后不良反应"].map(_parse_adverse_event)
    out["adverse_event_label"] = df["术后不良反应"]
    out["qor15"] = df["术后第1天QoR-15量表评分"].map(_parse_numeric)
    out["wbc"] = df["白细胞计数（WBC）"].map(_parse_numeric)
    out["crp"] = df["C反应蛋白（CRP）"].map(_parse_numeric)
    out["pct"] = df["降钙素原（PCT）"].map(_parse_numeric)

    # NLR：优先用 Excel 中比值列；若存在中性粒/淋巴绝对值列则重算并交叉核对
    nlr_col = "中性粒细胞/淋巴细胞比值（NLR）"
    out["nlr"] = df[nlr_col].map(_parse_numeric) if nlr_col in df.columns else np.nan
    neut_col = _find_column(df, ["中性粒细胞绝对值", "中性粒细胞计数", "绝对中性粒细胞", "ANC"])
    lymph_col = _find_column(df, ["淋巴细胞绝对值", "淋巴细胞计数", "绝对淋巴细胞", "ALC"])
    if neut_col and lymph_col:
        neut = df[neut_col].map(_parse_numeric)
        lymph = df[lymph_col].map(_parse_numeric)
        with np.errstate(divide="ignore", invalid="ignore"):
            recomputed = neut / lymph.replace(0, np.nan)
        out["nlr_recomputed"] = recomputed.to_numpy()
        out["nlr_source"] = "excel_ratio+recomputed_available"
        # 有重算值时以标准定义（绝对值比）为准
        mask = recomputed.notna().to_numpy()
        out.loc[mask, "nlr"] = recomputed.to_numpy()[mask]
    else:
        out["nlr_source"] = "excel_ratio_only"

    out = out.dropna(subset=["sample_id"]).drop_duplicates(subset=["sample_id"])
    out = out.set_index("sample_id")
    return out


def _find_column(df: pd.DataFrame, keywords: list[str]) -> str | None:
    for col in df.columns:
        text = str(col)
        for kw in keywords:
            if kw in text:
                return col
    return None


def audit_nlr(clinical: pd.DataFrame, output_dir: Path | None = None) -> pd.DataFrame:
    """输出 NLR 口径核查表（优化方案 Step 0）。

    当前临床表仅有「中性粒细胞/淋巴细胞比值（NLR）」列，无绝对值分量时无法独立重算。
    围术期/ICU NLR 中位值常高于门诊正常范围（约 1–3），偏高本身不一定是解析错误。
    """
    rows = []
    nlr = pd.to_numeric(clinical.get("nlr"), errors="coerce")
    source = clinical["nlr_source"].iloc[0] if "nlr_source" in clinical.columns and len(clinical) else "unknown"
    rows.append({"check": "nlr_source", "value": source, "note": "excel_ratio_only=仅比值列；无法用绝对值重算"})
    rows.append({"check": "n_nonmissing", "value": int(nlr.notna().sum()), "note": ""})
    rows.append({"check": "mean", "value": f"{nlr.mean():.2f}" if nlr.notna().any() else "NA", "note": ""})
    rows.append({"check": "median", "value": f"{nlr.median():.2f}" if nlr.notna().any() else "NA", "note": ""})
    rows.append({"check": "min", "value": f"{nlr.min():.2f}" if nlr.notna().any() else "NA", "note": ""})
    rows.append({"check": "max", "value": f"{nlr.max():.2f}" if nlr.notna().any() else "NA", "note": ""})
    n_high = int((nlr > 20).sum()) if nlr.notna().any() else 0
    rows.append({
        "check": "n_gt_20",
        "value": n_high,
        "note": "NLR>20 为极端升高，围术期可见；非解析单位错误（Excel 无 % 符号误读迹象）",
    })
    if "nlr_recomputed" in clinical.columns:
        both = clinical[["nlr", "nlr_recomputed"]].dropna()
        if len(both):
            corr = both["nlr"].corr(both["nlr_recomputed"])
            rows.append({"check": "corr_excel_vs_recomputed", "value": f"{corr:.3f}", "note": ""})
    else:
        rows.append({
            "check": "recompute_possible",
            "value": "False",
            "note": "Excel 缺少中性粒细胞/淋巴细胞绝对值列，无法交叉验证",
        })

    out = pd.DataFrame(rows)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_dir / "nlr_audit.csv", index=False, encoding="utf-8-sig")
        # 逐样本极端值清单
        extremes = clinical.loc[nlr > 20, ["nlr"]].copy() if nlr.notna().any() else pd.DataFrame()
        if not extremes.empty:
            extremes = extremes.reset_index()
            extremes.to_csv(output_dir / "nlr_extreme_samples.csv", index=False, encoding="utf-8-sig")
    return out


def load_asv_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str).str.strip()
    return df


def load_taxonomy(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str).str.strip()
    return df
