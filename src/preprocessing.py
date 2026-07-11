"""数据预处理与分组变量生成。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.microbiome import alpha_diversity_table, bray_curtis_matrix, relative_abundance


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
