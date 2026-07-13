"""数据预处理与分组变量生成。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.microbiome import alpha_diversity_table, bray_curtis_matrix, relative_abundance

CORE_CLINICAL_FIELDS = [
    "age",
    "sex",
    "bmi",
    "asa",
    "surgery_duration_min",
    "anesthesia_duration_min",
    "wbc",
    "crp",
    "extubation_time_min",
]

HISTORY_EXCLUSION_KEYWORDS = [
    "术前感染",
    "术前肺炎",
    "明确感染",
    "呼吸道感染",
    "近4周抗生素",
    "近四周抗生素",
    "免疫抑制剂",
    "免疫缺陷",
    "气管切开",
    "长期机械通气",
]


def apply_inclusion_exclusion(clinical: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按研究方案过滤临床样本，返回 (过滤后数据, 排除日志)。"""
    inc = cfg.get("inclusion", {})
    if not inc.get("enabled", True):
        return clinical.copy(), pd.DataFrame(columns=["sample_id", "stage", "reason"])

    logs = []

    def _exclude(sample_id: str, reason: str):
        logs.append({"sample_id": sample_id, "stage": "clinical", "reason": reason})

    keep = pd.Series(True, index=clinical.index)
    for sid in clinical.index:
        row = clinical.loc[sid]

        if inc.get("require_extubation_time", True) and pd.isna(row.get("extubation_time_min")):
            _exclude(sid, "缺少拔管时间")
            keep[sid] = False
            continue

        if inc.get("exclude_history", True) and "history_raw" in clinical.columns:
            history = str(row.get("history_raw", ""))
            for kw in inc.get("history_keywords", HISTORY_EXCLUSION_KEYWORDS):
                if kw in history:
                    _exclude(sid, f"病史排除: {kw}")
                    keep[sid] = False
                    break
            if not keep[sid]:
                continue

        if inc.get("require_core_clinical", True):
            core = [f for f in inc.get("core_fields", CORE_CLINICAL_FIELDS) if f in clinical.columns]
            missing = [f for f in core if pd.isna(row.get(f))]
            if missing:
                _exclude(sid, f"核心临床字段缺失: {', '.join(missing)}")
                keep[sid] = False
                continue

        max_missing = inc.get("max_clinical_missing_frac", 0.20)
        check_cols = [c for c in clinical.columns if c not in {"history_raw", "adverse_event_label"}]
        miss_frac = row[check_cols].isna().mean()
        if miss_frac > max_missing:
            _exclude(sid, f"临床资料缺失率 {miss_frac:.0%} > {max_missing:.0%}")
            keep[sid] = False

    filtered = clinical.loc[keep].copy()
    log_df = pd.DataFrame(logs)
    return filtered, log_df


def filter_microbiome_qc(
    asv: pd.DataFrame,
    taxonomy: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """菌群 QC：样本过滤 + ASV 流行度过滤 + 未注释 ASV 过滤。"""
    qc = cfg.get("qc", {})
    logs = []
    asv = asv.copy()
    taxonomy = taxonomy.copy()

    # 1) 过滤未注释 ASV
    if qc.get("remove_unassigned_taxonomy", True):
        phylum_col = "Phylum" if "Phylum" in taxonomy.columns else None
        if phylum_col:
            assigned = taxonomy[phylum_col].fillna("Unknown").astype(str).str.lower() != "unknown"
            n_before = asv.shape[1]
            asv = asv.loc[:, asv.columns.isin(taxonomy.index[assigned])]
            if n_before > asv.shape[1]:
                logs.append(
                    {
                        "sample_id": "_global_",
                        "stage": "asv_filter",
                        "reason": f"移除未注释 ASV: {n_before - asv.shape[1]} 个",
                    }
                )

    # 2) ASV 流行度过滤
    min_prev = qc.get("asv_min_prevalence", 0.10)
    if min_prev > 0 and asv.shape[0] > 0:
        prevalence = (asv > 0).sum(axis=0) / asv.shape[0]
        keep_asvs = prevalence >= min_prev
        n_before = asv.shape[1]
        asv = asv.loc[:, keep_asvs]
        if n_before > asv.shape[1]:
            logs.append(
                {
                    "sample_id": "_global_",
                    "stage": "asv_filter",
                    "reason": f"移除低流行度 ASV (<{min_prev:.0%}): {n_before - asv.shape[1]} 个",
                }
            )

    # 3) 样本级 QC
    min_reads = qc.get("min_total_reads", 1000)
    min_obs = qc.get("min_observed_asv", 20)
    max_obs = qc.get("max_observed_asv", 500)
    max_shannon = qc.get("max_shannon", 6.0)

    alpha = alpha_diversity_table(asv)
    keep_samples = []
    for sid in asv.index:
        total_reads = float(asv.loc[sid].sum())
        obs = int(alpha.loc[sid, "observed_asv"]) if sid in alpha.index else 0
        sh = float(alpha.loc[sid, "shannon"]) if sid in alpha.index else 0.0

        if total_reads < min_reads:
            logs.append({"sample_id": sid, "stage": "microbiome_qc", "reason": f"总 reads {total_reads:.0f} < {min_reads}"})
            continue
        if obs < min_obs:
            logs.append({"sample_id": sid, "stage": "microbiome_qc", "reason": f"observed_asv {obs} < {min_obs}"})
            continue
        if obs > max_obs:
            logs.append({"sample_id": sid, "stage": "microbiome_qc", "reason": f"observed_asv {obs} > {max_obs} (疑似污染)"})
            continue
        if sh > max_shannon:
            logs.append({"sample_id": sid, "stage": "microbiome_qc", "reason": f"Shannon {sh:.2f} > {max_shannon} (疑似污染)"})
            continue
        keep_samples.append(sid)

    asv = asv.loc[keep_samples]
    log_df = pd.DataFrame(logs)
    return asv, taxonomy, log_df


def save_exclusion_log(logs: list[pd.DataFrame], output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    if logs:
        combined = pd.concat([df for df in logs if not df.empty], ignore_index=True)
    else:
        combined = pd.DataFrame(columns=["sample_id", "stage", "reason"])
    combined.to_csv(output_dir / "exclusion_log.csv", index=False, encoding="utf-8-sig")
    return combined


def add_outcome_groups(clinical: pd.DataFrame, split: str = "median", fixed_min: float = 68.5) -> pd.DataFrame:
    df = clinical.copy()
    ext = df["extubation_time_min"]
    if split == "median":
        cutoff = ext.median()
    else:
        cutoff = fixed_min
    df["extubation_cutoff"] = cutoff
    df["delayed_extubation"] = (ext > cutoff).astype(int)
    df["extubation_group"] = np.where(df["delayed_extubation"] == 1, "Delayed", "Early")
    df["log_crp"] = np.log1p(df["crp"])
    df["log_pct"] = np.log1p(df["pct"])
    return df


def add_quadrant_groups(clinical: pd.DataFrame, crp_threshold: float = 10.0) -> pd.DataFrame:
    df = clinical.copy()
    shannon_median = df["shannon"].median()
    df["high_diversity"] = (df["shannon"] >= shannon_median).astype(int)
    df["high_inflammation"] = (df["crp"] >= crp_threshold).astype(int)

    def _label(row):
        if row["high_diversity"] and not row["high_inflammation"]:
            return "HighDiv_LowCRP"
        if row["high_diversity"] and row["high_inflammation"]:
            return "HighDiv_HighCRP"
        if not row["high_diversity"] and not row["high_inflammation"]:
            return "LowDiv_LowCRP"
        return "LowDiv_HighCRP"

    df["quadrant_group"] = df.apply(_label, axis=1)
    return df


def merge_microbiome(clinical: pd.DataFrame, asv: pd.DataFrame, taxonomy: pd.DataFrame) -> pd.DataFrame:
    common = clinical.index.intersection(asv.index)
    if len(common) == 0:
        raise ValueError("临床样本与 ASV 表无交集，请检查 sample_id 是否一致。")
    clinical = clinical.loc[common].copy()
    asv = asv.loc[common].copy()

    alpha = alpha_diversity_table(asv)
    clinical = clinical.join(alpha, how="left")

    rel = relative_abundance(asv)
    dist = bray_curtis_matrix(rel)
    clinical.attrs["asv"] = asv
    clinical.attrs["rel_abund"] = rel
    clinical.attrs["taxonomy"] = taxonomy
    clinical.attrs["bray_curtis"] = dist
    clinical.attrs["sample_order"] = common.tolist()
    return clinical


def generate_demo_microbiome(clinical: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """基于临床结局生成演示用 ASV 数据，仅用于流程联调。"""
    rng = np.random.default_rng(seed)
    n = len(clinical)
    genera = [
        "Streptococcus",
        "Veillonella",
        "Prevotella",
        "Rothia",
        "Pseudomonas",
        "Klebsiella",
        "Acinetobacter",
        "Staphylococcus",
        "Haemophilus",
        "Neisseria",
        "Fusobacterium",
        "Enterobacter",
        "Moraxella",
        "Lactobacillus",
        "Gemella",
    ]
    phyla_map = {
        "Streptococcus": "Firmicutes",
        "Veillonella": "Firmicutes",
        "Prevotella": "Bacteroidetes",
        "Rothia": "Actinobacteria",
        "Pseudomonas": "Proteobacteria",
        "Klebsiella": "Proteobacteria",
        "Acinetobacter": "Proteobacteria",
        "Staphylococcus": "Firmicutes",
        "Haemophilus": "Proteobacteria",
        "Neisseria": "Proteobacteria",
        "Fusobacterium": "Fusobacteria",
        "Enterobacter": "Proteobacteria",
        "Moraxella": "Proteobacteria",
        "Lactobacillus": "Firmicutes",
        "Gemella": "Firmicutes",
    }

    asv_ids = [f"ASV_{i+1:03d}" for i in range(len(genera))]
    taxonomy = pd.DataFrame(
        {
            "Genus": genera,
            "Phylum": [phyla_map[g] for g in genera],
            "Family": [f"{g}_family" for g in genera],
            "Species": [f"{g}_sp" for g in genera],
        },
        index=asv_ids,
    )

    asv = pd.DataFrame(index=clinical.index, columns=asv_ids, dtype=float)
    for sid, row in clinical.iterrows():
        base_depth = rng.integers(25000, 45000)
        weights = rng.dirichlet(np.ones(len(genera)))
        delayed = row.get("delayed_extubation", 0)
        crp = row.get("crp", 5)
        ae = row.get("adverse_event", 0)

        # 延迟拔管/高炎症/不良反应 -> 条件致病菌富集
        for i, g in enumerate(genera):
            if g in {"Pseudomonas", "Klebsiella", "Acinetobacter", "Staphylococcus"}:
                weights[i] *= 1 + 0.8 * delayed + 0.4 * ae + 0.02 * crp
            if g in {"Prevotella", "Veillonella", "Rothia", "Fusobacterium"}:
                weights[i] *= max(0.1, 1 + 0.6 * (1 - delayed) - 0.015 * crp)
        weights = np.clip(weights, 1e-6, None)
        weights = weights / weights.sum()
        counts = rng.multinomial(base_depth, weights)
        asv.loc[sid] = counts

    return asv, taxonomy
