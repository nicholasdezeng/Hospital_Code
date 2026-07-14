"""Model C' 触发评估：仅当属水平分析发现稳健显著差异菌属时重建。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.prediction import CLINICAL_BASE, _eval_model
from src.utils.microbiome import aggregate_taxonomy, relative_abundance


def assess_and_maybe_run_model_c_prime(
    clinical: pd.DataFrame,
    output_dir: Path,
    *,
    fdr_threshold: float = 0.1,
    n_boot: int = 1000,
    n_perm: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """根据 table_genus_wilcoxon / lefse 结果决定是否触发 Model C'。

    触发条件（需同时满足）：
    1. 至少 1 个具名属（非 Unknown/Unassigned）FDR < fdr_threshold，或
    2. LEfSe 具名属 LDA≥2 且 q<0.1
    """
    output_dir = Path(output_dir)
    tab_dir = output_dir if output_dir.name == "tables" else output_dir
    # 兼容传入 out_root 或 tables/
    wilt_path = tab_dir / "table_genus_wilcoxon.csv"
    if not wilt_path.exists() and (tab_dir / "tables" / "table_genus_wilcoxon.csv").exists():
        wilt_path = tab_dir / "tables" / "table_genus_wilcoxon.csv"
    fig_dir = tab_dir.parent / "figures" if tab_dir.name == "tables" else tab_dir / "figures"
    lefse_path = fig_dir / "figure4_lefse_results.csv"

    decision = {
        "triggered": False,
        "reason": "",
        "candidate_genera": "",
        "n_named_fdr_sig": 0,
        "n_lefse_named": 0,
    }

    named_sig: list[str] = []
    if wilt_path.exists():
        wilt = pd.read_csv(wilt_path)
        if "q" in wilt.columns and "genus" in wilt.columns:
            sub = wilt[(wilt["q"] < fdr_threshold) & (~wilt["genus"].astype(str).str.lower().isin({"unknown", "unassigned", "other"}))]
            named_sig = sub["genus"].astype(str).tolist()
            decision["n_named_fdr_sig"] = len(named_sig)

    lefse_named: list[str] = []
    if lefse_path.exists():
        lefse = pd.read_csv(lefse_path)
        if not lefse.empty and "genus" in lefse.columns:
            qcol = "q" if "q" in lefse.columns else None
            mask = ~lefse["genus"].astype(str).str.lower().isin({"unknown", "unassigned", "other"})
            if qcol:
                mask = mask & (lefse[qcol] < fdr_threshold)
            lefse_named = lefse.loc[mask, "genus"].astype(str).tolist()
            decision["n_lefse_named"] = len(lefse_named)

    candidates = list(dict.fromkeys(named_sig + lefse_named))
    decision["candidate_genera"] = ";".join(candidates) if candidates else ""

    if not candidates:
        decision["reason"] = (
            f"未触发：无具名属满足 FDR<{fdr_threshold}；"
            "当前属水平结果以 Unknown 为主，不重建 Model C'。"
        )
        out = pd.DataFrame([decision])
        out_path = tab_dir / "model_c_prime_decision.csv" if tab_dir.name == "tables" else tab_dir / "tables" / "model_c_prime_decision.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False, encoding="utf-8-sig")
        return out

    # 触发：用临床 + Top 具名属丰度构建 Model C'
    decision["triggered"] = True
    decision["reason"] = f"触发：发现具名显著差异属 {candidates}；以 Clinical + genus 丰度重建 Model C'"

    asv = clinical.attrs["asv"]
    taxonomy = clinical.attrs["taxonomy"]
    rel_genus = relative_abundance(aggregate_taxonomy(asv, taxonomy, "Genus"))
    feat_cols = list(CLINICAL_BASE)
    for g in candidates[:3]:  # 最多 3 个属，控制过拟合
        if g in rel_genus.columns:
            col = f"genus_{g}"
            clinical[col] = rel_genus[g]
            feat_cols.append(col)

    target_col = "delayed_extubation"
    valid = clinical[feat_cols + [target_col]].dropna()
    y = valid[target_col].astype(int)
    m, _ = _eval_model(
        "Model C' (Clinical+named genera)",
        feat_cols,
        valid,
        y,
        n_boot=n_boot,
        n_perm=n_perm,
        seed=seed,
    )
    metrics = pd.DataFrame([m])
    tables = tab_dir if tab_dir.name == "tables" else tab_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([decision]).to_csv(tables / "model_c_prime_decision.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(tables / "table2_model_c_prime.csv", index=False, encoding="utf-8-sig")
    return pd.DataFrame([decision])
