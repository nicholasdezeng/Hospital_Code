"""Table 1: 基线特征表。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.stats import compare_categorical, compare_continuous, format_p


def _summarize_continuous(series: pd.Series, as_median: bool = False) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return "NA"
    if as_median:
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        return f"{s.median():.1f} ({q1:.1f}-{q3:.1f})"
    return f"{s.mean():.1f}±{s.std():.1f}"


def _summarize_binary(series: pd.Series) -> str:
    s = series.dropna()
    if len(s) == 0:
        return "NA"
    n = int(s.sum())
    return f"{n} ({100 * n / len(s):.1f}%)"


def build_table1(clinical: pd.DataFrame) -> pd.DataFrame:
    early = clinical[clinical["extubation_group"] == "Early"]
    delayed = clinical[clinical["extubation_group"] == "Delayed"]

    rows = []
    specs = [
        ("年龄（岁）", "age", "cont"),
        ("BMI（kg/m²）", "bmi", "cont"),
        ("吸烟史", "smoking", "bin"),
        ("高血压", "hypertension", "bin"),
        ("糖尿病", "diabetes", "bin"),
        ("COPD", "copd", "bin"),
        ("ASA分级", "asa", "cont"),
        ("手术时长（min）", "surgery_duration_min", "cont"),
        ("麻醉时长（min）", "anesthesia_duration_min", "cont"),
        ("阿片类药物（吗啡当量，mg）", "opioid_morphine_mg", "cont"),
        ("WBC（×10⁹/L）", "wbc", "cont"),
        ("CRP（mg/L）", "crp", "cont_med"),
        ("PCT（ng/mL）", "pct", "cont_med"),
        ("NLR", "nlr", "cont"),
        ("拔管时间（min）", "extubation_time_min", "cont_med"),
        ("ICU滞留时长（min）", "icu_stay_min", "cont_med"),
        ("不良反应发生", "adverse_event", "bin"),
        ("QoR-15评分", "qor15", "cont"),
    ]

    for label, col, kind in specs:
        all_val = delayed_val = early_val = "NA"
        p = np.nan
        if col in clinical.columns:
            if kind == "cont":
                all_val = _summarize_continuous(clinical[col])
                early_val = _summarize_continuous(early[col])
                delayed_val = _summarize_continuous(delayed[col])
                p, _ = compare_continuous(early[col], delayed[col])
            elif kind == "cont_med":
                all_val = _summarize_continuous(clinical[col], as_median=True)
                early_val = _summarize_continuous(early[col], as_median=True)
                delayed_val = _summarize_continuous(delayed[col], as_median=True)
                p, _ = compare_continuous(early[col], delayed[col])
            elif kind == "bin":
                all_val = _summarize_binary(clinical[col])
                early_val = _summarize_binary(early[col])
                delayed_val = _summarize_binary(delayed[col])
                table = pd.crosstab(clinical["extubation_group"], clinical[col])
                if table.shape[0] >= 2 and table.shape[1] >= 2:
                    p, _ = compare_categorical(table)
        rows.append(
            {
                "特征": label,
                f"全体（n={len(clinical)}）": all_val,
                f"早期拔管组（n={len(early)}）": early_val,
                f"延迟拔管组（n={len(delayed)}）": delayed_val,
                "P值": format_p(p),
            }
        )

    sex_table = pd.crosstab(clinical["extubation_group"], clinical["sex"])
    p_sex, _ = compare_categorical(sex_table) if sex_table.size else (np.nan, "")
    male_all = int((clinical["sex"] == 1).sum())
    female_all = int((clinical["sex"] == 0).sum())
    male_e = int((early["sex"] == 1).sum())
    female_e = int((early["sex"] == 0).sum())
    male_d = int((delayed["sex"] == 1).sum())
    female_d = int((delayed["sex"] == 0).sum())
    rows.insert(
        1,
        {
            "特征": "性别（男/女）",
            f"全体（n={len(clinical)}）": f"{male_all}/{female_all}",
            f"早期拔管组（n={len(early)}）": f"{male_e}/{female_e}",
            f"延迟拔管组（n={len(delayed)}）": f"{male_d}/{female_d}",
            "P值": format_p(p_sex),
        },
    )

    return pd.DataFrame(rows)


def _table1_footnotes(clinical: pd.DataFrame) -> pd.DataFrame:
    """Table 1 脚注（COPD 零方差、NLR 口径、WBC 方向等）。"""
    notes = [
        {
            "编号": "①",
            "内容": "COPD 两组均为 0 例（零方差），Table 1 保留描述行；后续回归/预测模型均不纳入该协变量。",
        },
        {
            "编号": "②",
            "内容": (
                "NLR 来自临床表「中性粒细胞/淋巴细胞比值（NLR）」列直接解析；"
                "当前表无中性粒/淋巴绝对值列，无法独立重算。详见 nlr_audit.csv。"
                "围术期 NLR 中位常高于门诊参考范围（约 1–3），偏高本身不一定表示解析错误。"
            ),
        },
        {
            "编号": "③",
            "内容": (
                "WBC 组间差异有统计学意义且 Early 组高于 Delayed 组，"
                "与「延迟组炎症更高」的先验假设方向相反；讨论中区分应激性升高与感染性升高。"
            ),
        },
        {
            "编号": "④",
            "内容": f"主分析样本量 n={len(clinical)}；拔管分组为时间中位数切分（Early/Delayed）。",
        },
    ]
    return pd.DataFrame(notes)


def run_baseline(clinical: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    table = build_table1(clinical)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "table1_baseline.csv", index=False, encoding="utf-8-sig")
    table.to_excel(output_dir / "table1_baseline.xlsx", index=False)
    notes = _table1_footnotes(clinical)
    notes.to_csv(output_dir / "table1_footnotes.csv", index=False, encoding="utf-8-sig")
    return table
