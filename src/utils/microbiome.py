"""菌群多样性计算与距离矩阵。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def shannon(counts: np.ndarray) -> float:
    counts = counts[counts > 0]
    if len(counts) == 0:
        return 0.0
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)))


def chao1(counts: np.ndarray) -> float:
    counts = counts[counts > 0]
    s_obs = len(counts)
    singletons = np.sum(counts == 1)
    doubletons = np.sum(counts == 2)
    if doubletons == 0:
        return float(s_obs + singletons * (singletons - 1) / 2)
    return float(s_obs + singletons ** 2 / (2 * doubletons))


def pielou_j(counts: np.ndarray) -> float:
    s_obs = np.sum(counts > 0)
    if s_obs <= 1:
        return 0.0
    return shannon(counts) / np.log(s_obs)


def alpha_diversity_table(asv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sample_id, row in asv.iterrows():
        counts = row.values.astype(float)
        rows.append(
            {
                "sample_id": sample_id,
                "shannon": shannon(counts),
                "chao1": chao1(counts),
                "pielou_j": pielou_j(counts),
                "observed_asv": int(np.sum(counts > 0)),
            }
        )
    return pd.DataFrame(rows).set_index("sample_id")


def relative_abundance(asv: pd.DataFrame) -> pd.DataFrame:
    totals = asv.sum(axis=1).replace(0, np.nan)
    return asv.div(totals, axis=0).fillna(0)


def bray_curtis_matrix(rel_abund: pd.DataFrame) -> np.ndarray:
    return squareform(pdist(rel_abund.values, metric="braycurtis"))


def rarefaction_curve(counts: np.ndarray, steps: int = 20) -> tuple[np.ndarray, np.ndarray]:
    total = int(counts.sum())
    if total == 0:
        x = np.linspace(0, 1, steps)
        return x, np.zeros(steps)
    depths = np.linspace(max(1, total // steps), total, steps, dtype=int)
    observed = []
    rng = np.random.default_rng(42)
    expanded = np.repeat(np.arange(len(counts)), counts.astype(int))
    for depth in depths:
        if depth >= len(expanded):
            observed.append(np.sum(counts > 0))
            continue
        sampled = rng.choice(expanded, size=depth, replace=False)
        observed.append(len(np.unique(sampled)))
    return depths, np.array(observed)


PHYLUM_DISPLAY_NAMES = {
    "Bacillota": "Firmicutes",
    "Firmicutes": "Firmicutes",
    "Pseudomonadota": "Proteobacteria",
    "Proteobacteria": "Proteobacteria",
    "Bacteroidota": "Bacteroidetes",
    "Bacteroidetes": "Bacteroidetes",
    "Actinomycetota": "Actinobacteria",
    "Actinobacteria": "Actinobacteria",
    "Fusobacteriota": "Fusobacteria",
    "Fusobacteria": "Fusobacteria",
    "Unknown": "Unassigned",
    "Unassigned": "Unassigned",
}

PHYLUM_STACK_ORDER = [
    "Firmicutes",
    "Proteobacteria",
    "Bacteroidetes",
    "Actinobacteria",
    "Fusobacteria",
    "Other",
]


def normalize_phylum_name(name: str) -> str:
    text = str(name).strip()
    for prefix in ("k__", "p__"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    text = text.replace("_", " ")
    return PHYLUM_DISPLAY_NAMES.get(text, text)


def collapse_phyla_for_plot(rel_phylum: pd.DataFrame) -> pd.DataFrame:
    """合并同类门名，固定 SCI 图例顺序（含 Other）。"""
    rel = rel_phylum.copy()
    rel.columns = [normalize_phylum_name(c) for c in rel.columns]
    rel = rel.T.groupby(level=0).sum().T

    out = pd.DataFrame(index=rel.index)
    for phylum in PHYLUM_STACK_ORDER[:-1]:
        out[phylum] = rel[phylum] if phylum in rel.columns else 0.0
    other_cols = [c for c in rel.columns if c not in PHYLUM_STACK_ORDER[:-1]]
    out["Other"] = rel[other_cols].sum(axis=1) if other_cols else (1.0 - out.sum(axis=1)).clip(lower=0)
    row_sum = out.sum(axis=1).replace(0, np.nan)
    return out.div(row_sum, axis=0).fillna(0)


def aggregate_taxonomy(asv: pd.DataFrame, taxonomy: pd.DataFrame, level: str) -> pd.DataFrame:
    if level not in taxonomy.columns:
        raise ValueError(f"Taxonomy level {level} not found")
    tax = taxonomy.copy()
    tax[level] = tax[level].fillna("Unknown")
    grouped = {}
    for asv_id in asv.columns:
        if asv_id not in tax.index:
            label = "Unknown"
        else:
            label = tax.loc[asv_id, level]
        grouped.setdefault(label, []).append(asv_id)
    out = pd.DataFrame(index=asv.index)
    for label, cols in grouped.items():
        out[label] = asv[cols].sum(axis=1)
    return out
