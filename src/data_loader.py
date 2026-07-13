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
    out["nlr"] = df["中性粒细胞/淋巴细胞比值（NLR）"].map(_parse_numeric)

    out = out.dropna(subset=["sample_id"]).drop_duplicates(subset=["sample_id"])
    out = out.set_index("sample_id")
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
