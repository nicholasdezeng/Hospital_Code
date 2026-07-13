"""敏感性分析：严格 QC 与放宽 QC 双队列对比。"""

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd

from src.preprocessing import (
    add_outcome_groups,
    apply_inclusion_exclusion,
    filter_microbiome_qc,
    merge_microbiome,
)
from src.utils.microbiome import aggregate_taxonomy, relative_abundance


def run_sensitivity_cohorts(
    clinical_raw: pd.DataFrame,
    asv: pd.DataFrame,
    taxonomy: pd.DataFrame,
    cfg: dict,
    output_dir: Path,
) -> pd.DataFrame:
    """在主分析之外，额外评估放宽 QC 后的样本量与核心指标。"""
    sens_cfg = cfg.get("sensitivity", {})
    if not sens_cfg.get("enabled", True):
        return pd.DataFrame()

    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = {
        "strict": {**cfg.get("qc", {}), **sens_cfg.get("strict_qc", {})},
        "main": cfg.get("qc", {}),
        "relaxed": sens_cfg.get("relaxed_qc", {}),
    }

    rows = []
    for name, qc_override in profiles.items():
        trial = copy.deepcopy(cfg)
        trial_qc = copy.deepcopy(cfg.get("qc", {}))
        trial_qc.update(qc_override)
        trial_qc["enabled"] = True
        trial["qc"] = trial_qc

        clin, _ = apply_inclusion_exclusion(clinical_raw, trial)
        clin = add_outcome_groups(
            clin,
            split=trial["grouping"]["extubation_split"],
            fixed_min=trial["grouping"]["extubation_fixed_min"],
        )
        asv_q, tax_q, qc_log = filter_microbiome_qc(asv.copy(), taxonomy.copy(), trial)
        common = clin.index.intersection(asv_q.index)
        if len(common) == 0:
            rows.append({"cohort": name, "n_samples": 0, "n_asv": 0, "n_genera": 0})
            continue

        merged = merge_microbiome(clin.loc[common], asv_q.loc[common], tax_q)
        rel_genus = relative_abundance(aggregate_taxonomy(merged.attrs["asv"], tax_q, "Genus"))
        n_genera = rel_genus.shape[1]
        from scipy import stats

        e = merged.loc[merged.extubation_group == "Early", "shannon"]
        d = merged.loc[merged.extubation_group == "Delayed", "shannon"]
        sh_p = stats.mannwhitneyu(e, d).pvalue if len(e) > 0 and len(d) > 0 else float("nan")
        rho, crp_p = stats.spearmanr(merged["shannon"], merged["crp"]) if len(merged) >= 5 else (float("nan"), float("nan"))

        rows.append({
            "cohort": name,
            "n_samples": len(merged),
            "n_early": int((merged.extubation_group == "Early").sum()),
            "n_delayed": int((merged.extubation_group == "Delayed").sum()),
            "n_asv": int(merged.attrs["asv"].shape[1]),
            "n_genera": int(n_genera),
            "shannon_early_mean": float(e.mean()) if len(e) else float("nan"),
            "shannon_delayed_mean": float(d.mean()) if len(d) else float("nan"),
            "shannon_group_p": float(sh_p),
            "shannon_crp_rho": float(rho),
            "shannon_crp_p": float(crp_p),
            "qc_excluded": int((qc_log["stage"] == "microbiome_qc").sum()) if not qc_log.empty else 0,
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "sensitivity_cohort_summary.csv", index=False, encoding="utf-8-sig")
    return summary
