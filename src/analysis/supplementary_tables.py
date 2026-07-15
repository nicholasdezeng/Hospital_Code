"""优化方案补充表格：Table 1-补充、属水平 Wilcoxon、不良反应描述性等。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.microbiome import aggregate_taxonomy, relative_abundance
from src.utils.stats import cliffs_delta, fdr_correct, format_p, spearman_with_fdr


def _summarize_continuous(series: pd.Series, as_median: bool = False) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return "NA"
    if as_median:
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        return f"{s.median():.2f} ({q1:.2f}-{q3:.2f})"
    return f"{s.mean():.2f}±{s.std():.2f}"


def run_table1_supplement(clinical: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Table 1-补充：两组 α 多样性 + Mann-Whitney + Cliff's δ。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    early = clinical[clinical["extubation_group"] == "Early"]
    delayed = clinical[clinical["extubation_group"] == "Delayed"]
    metrics = [
        ("Shannon指数", "shannon"),
        ("Chao1指数", "chao1"),
        ("Pielou's J", "pielou_j"),
        ("Simpson指数", "simpson"),
    ]
    rows = []
    for label, col in metrics:
        if col not in clinical.columns:
            continue
        # 优化方案第3.2(A) 层次一：四项多样性指标统一采用 Mann-Whitney U（非参数）
        e = pd.to_numeric(early[col], errors="coerce").dropna()
        d = pd.to_numeric(delayed[col], errors="coerce").dropna()
        if len(e) >= 3 and len(d) >= 3:
            _, p = stats.mannwhitneyu(e, d, alternative="two-sided")
            method = "Mann-Whitney U"
        else:
            p, method = np.nan, "insufficient"
        delta = cliffs_delta(early[col], delayed[col])
        rows.append(
            {
                "多样性指标": label,
                f"早期拔管组（n={len(early)}）": _summarize_continuous(early[col]),
                f"延迟拔管组（n={len(delayed)}）": _summarize_continuous(delayed[col]),
                "检验方法": method,
                "P值": format_p(p),
                "Cliff's δ": f"{delta:.3f}" if pd.notna(delta) else "NA",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "table1_supplement_diversity.csv", index=False, encoding="utf-8-sig")
    out.to_excel(output_dir / "table1_supplement_diversity.xlsx", index=False)
    return out


def run_genus_wilcoxon_table(clinical: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """属水平 Wilcoxon + FDR（优化方案首选差异分析）。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    asv = clinical.attrs["asv"]
    taxonomy = clinical.attrs["taxonomy"]
    rel_genus = relative_abundance(aggregate_taxonomy(asv, taxonomy, "Genus"))
    groups = clinical["extubation_group"]
    records = []
    for genus in rel_genus.columns:
        early = rel_genus.loc[groups == "Early", genus]
        delayed = rel_genus.loc[groups == "Delayed", genus]
        if len(early) < 3 or len(delayed) < 3:
            continue
        stat, p = stats.mannwhitneyu(early, delayed, alternative="two-sided")
        enriched = "Delayed" if delayed.mean() > early.mean() else "Early"
        records.append(
            {
                "genus": genus,
                "early_mean_pct": float(early.mean() * 100),
                "delayed_mean_pct": float(delayed.mean() * 100),
                "U": float(stat),
                "p": float(p),
                "enriched_group": enriched,
            }
        )
    if not records:
        empty = pd.DataFrame(columns=["genus", "early_mean_pct", "delayed_mean_pct", "p", "q", "enriched_group"])
        empty.to_csv(output_dir / "table_genus_wilcoxon.csv", index=False, encoding="utf-8-sig")
        return empty

    df = pd.DataFrame(records).sort_values("p")
    _, q = fdr_correct(df["p"].values)
    df["q"] = q
    df["P值"] = df["p"].map(format_p)
    df["FDR-P"] = df["q"].map(format_p)
    df.to_csv(output_dir / "table_genus_wilcoxon.csv", index=False, encoding="utf-8-sig")
    df.to_excel(output_dir / "table_genus_wilcoxon.xlsx", index=False)
    return df


def run_ae_descriptive(clinical: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """不良反应描述性表格（不做假设检验）。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    ae = clinical[clinical["adverse_event"] == 1]
    no_ae = clinical[clinical["adverse_event"] == 0]
    rows = [
        {
            "特征": "样本量",
            f"不良反应组（n={len(ae)}）": str(len(ae)),
            f"无不良反应组（n={len(no_ae)}）": str(len(no_ae)),
        },
        {
            "特征": "Shannon指数",
            f"不良反应组（n={len(ae)}）": _summarize_continuous(ae["shannon"]) if len(ae) else "NA",
            f"无不良反应组（n={len(no_ae)}）": _summarize_continuous(no_ae["shannon"]),
        },
        {
            "特征": "log(CRP)",
            f"不良反应组（n={len(ae)}）": _summarize_continuous(ae["log_crp"]) if len(ae) else "NA",
            f"无不良反应组（n={len(no_ae)}）": _summarize_continuous(no_ae["log_crp"]),
        },
        {
            "特征": "WBC（×10⁹/L）",
            f"不良反应组（n={len(ae)}）": _summarize_continuous(ae["wbc"]) if len(ae) else "NA",
            f"无不良反应组（n={len(no_ae)}）": _summarize_continuous(no_ae["wbc"]),
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "table_ae_descriptive.csv", index=False, encoding="utf-8-sig")
    out.to_excel(output_dir / "table_ae_descriptive.xlsx", index=False)
    return out


def run_shannon_inflammation_matrix(clinical: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Shannon 与炎症指标 Spearman 相关（数据核查用）。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ["wbc", "crp", "pct", "nlr"] if c in clinical.columns]
    out = spearman_with_fdr(clinical, ["shannon"], cols)
    out.to_csv(output_dir / "table_shannon_inflammation_corr.csv", index=False, encoding="utf-8-sig")
    return out
