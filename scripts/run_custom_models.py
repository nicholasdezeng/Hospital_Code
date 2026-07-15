"""自定义 5 模型 LOO-CV 评估（拔管延迟）。

在有真实 ASV 表的环境运行：
    python scripts/run_custom_models.py --config config.server.yaml
    python scripts/run_custom_models.py --config config.server.dada2_v2.yaml

模型（结局=拔管延迟 delayed_extubation）：
  Model 1: age, sex, BMI, ASA, 手术时长, 麻醉时长, 阿片当量, log(CRP)
  Model 2: Model1 + Bdellovibrio 丰度
  Model 3: Model1 + Lactobacillus 丰度
  Model 4: Model1 + Bdellovibrio + Lactobacillus
  Model 5: Model1 + 全部呼吸道菌群属丰度

说明：
  · "CRB" 按 CRP 处理，使用 log(CRP)（与 Table 2/3 一致）；改用原始 CRP 把下方 CRP_FEAT 换成 "crp"。
  · 属丰度为相对丰度（relative abundance）。
  · LOO-CV + 逻辑回归（中位数插补→标准化→class_weight=balanced），与主流水线 prediction.py 完全一致。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_analysis import _prepare_clinical, load_config
from src.analysis.prediction import _loo_predict, _metrics
from src.utils.microbiome import aggregate_taxonomy, relative_abundance
from src.utils.stats import bootstrap_auc_ci, format_p, permutation_auc_pvalue

CRP_FEAT = "log_crp"  # 改成 "crp" 可用原始 CRP
CLINICAL = ["age", "sex", "bmi", "asa", "surgery_duration_min", "anesthesia_duration_min", "opioid_morphine_mg", CRP_FEAT]
TARGET = "delayed_extubation"


def _genus_col(clinical: pd.DataFrame, rel_genus: pd.DataFrame, name: str) -> str:
    """把某个属的相对丰度写入 clinical 并返回列名；缺失则返回 None。"""
    if name not in rel_genus.columns:
        print(f"  [警告] 属 {name} 不在属丰度表中，跳过该特征")
        return None
    col = f"genus_{name}"
    clinical[col] = rel_genus[name].reindex(clinical.index)
    return col


def _eval(clinical: pd.DataFrame, feats: list[str], *, n_boot: int, n_perm: int, seed: int, name: str) -> dict:
    sub = clinical[feats + [TARGET]].dropna()
    yy = sub[TARGET].astype(int)
    probs, preds = _loo_predict(sub[feats], yy, seed=seed)
    m = _metrics(yy.values, probs, preds)
    _, lo, hi = bootstrap_auc_ci(yy.values, probs, n_boot=n_boot)
    p = permutation_auc_pvalue(yy.values, probs, n_perm=n_perm, seed=seed)
    return {
        "模型": name,
        "特征数": len(feats),
        "N": len(sub),
        "AUC": round(float(m["AUC"]), 3),
        "AUC_CI": f"{lo:.3f}-{hi:.3f}",
        "灵敏度": round(float(m["Sensitivity"]), 3),
        "特异度": round(float(m["Specificity"]), 3),
        "F1": round(float(m["F1"]), 3),
        "准确率": round(float(m["Accuracy"]), 3),
        "置换检验P": format_p(p),
    }


def main(config_path: str, cutoff: float | None) -> None:
    cfg = load_config(Path(config_path))
    cfg["_config_path"] = str(Path(config_path).resolve())
    clinical, *_ = _prepare_clinical(cfg, ROOT)

    if cutoff is not None:
        clinical["extubation_cutoff"] = cutoff
        clinical[TARGET] = (clinical["extubation_time_min"] > cutoff).astype(int)
        clinical["extubation_group"] = np.where(clinical[TARGET] == 1, "Delayed", "Early")

    n_early = int((clinical[TARGET] == 0).sum())
    n_delayed = int((clinical[TARGET] == 1).sum())
    used_cut = clinical["extubation_cutoff"].iloc[0]
    print(f"\n拔管分组阈值 = {used_cut} min → Early {n_early} / Delayed {n_delayed}（n={len(clinical)}）\n")

    rel_genus = relative_abundance(
        aggregate_taxonomy(clinical.attrs["asv"], clinical.attrs["taxonomy"], "Genus")
    )
    bdello = _genus_col(clinical, rel_genus, "Bdellovibrio")
    lacto = _genus_col(clinical, rel_genus, "Lactobacillus")

    all_genus_cols = []
    for g in rel_genus.columns:
        clinical[f"genus_{g}"] = rel_genus[g].reindex(clinical.index)
        all_genus_cols.append(f"genus_{g}")

    models = {
        "Model 1（临床+CRP）": CLINICAL,
        "Model 2（+Bdellovibrio）": CLINICAL + [c for c in [bdello] if c],
        "Model 3（+Lactobacillus）": CLINICAL + [c for c in [lacto] if c],
        "Model 4（+Bdello+Lacto）": CLINICAL + [c for c in [bdello, lacto] if c],
        f"Model 5（+全部属 n={len(all_genus_cols)}）": CLINICAL + all_genus_cols,
    }

    seed = cfg["analysis"]["random_seed"]
    n_boot = cfg["analysis"]["bootstrap_n"]
    n_perm = cfg["analysis"].get("permutation_n", 500)

    rows = [_eval(clinical, feats, n_boot=n_boot, n_perm=n_perm, seed=seed, name=name)
            for name, feats in models.items()]
    out = pd.DataFrame(rows)

    pd.set_option("display.unicode.east_asian_width", True)
    print(out.to_string(index=False))

    out_dir = ROOT / cfg["project"]["output_dir"] / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "custom_models_performance.csv"
    out.to_csv(dst, index=False, encoding="utf-8-sig")
    print(f"\n已保存: {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="自定义 5 模型 LOO-CV 评估（拔管延迟）")
    ap.add_argument("--config", default="config.server.yaml")
    ap.add_argument("--cutoff", type=float, default=None, help="拔管分组固定阈值(min)，默认用配置的中位数分组")
    args = ap.parse_args()
    main(args.config, args.cutoff)
