"""AICU 呼吸道菌群预后研究 — 主分析入口。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.analysis.alpha_beta import run_figure2, run_figure3
from src.analysis.baseline import run_baseline
from src.analysis.biomarkers import run_figure9
from src.analysis.differential import run_figure4
from src.analysis.inflammation_axis import run_figure5, run_figure6
from src.analysis.mediation import run_figure7
from src.analysis.microbiome_desc import run_figure1
from src.analysis.multivariable import run_multivariable
from src.analysis.prediction import run_prediction
from src.analysis.sensitivity import run_sensitivity_cohorts
from src.data_loader import load_asv_table, load_clinical_excel, load_taxonomy
from src.preprocessing import (
    add_outcome_groups,
    apply_inclusion_exclusion,
    filter_microbiome_qc,
    generate_demo_microbiome,
    merge_microbiome,
    save_exclusion_log,
)


def load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_microbiome(cfg: dict, root: Path, clinical: pd.DataFrame):
    asv_path = Path(cfg["paths"]["microbiome_asv"])
    tax_path = Path(cfg["paths"]["taxonomy"])
    if not asv_path.is_absolute():
        asv_path = root / asv_path
    if not tax_path.is_absolute():
        tax_path = root / tax_path
    use_demo = cfg["analysis"]["use_demo_microbiome"]

    if asv_path.exists() and tax_path.exists():
        print(f"[2/11] 加载真实菌群数据")
        asv = load_asv_table(asv_path)
        taxonomy = load_taxonomy(tax_path)
        print(f"      ASV 表: {asv.shape[0]} 样本 × {asv.shape[1]} ASV")
        qc_log = pd.DataFrame()
        if cfg.get("qc", {}).get("enabled", True):
            asv, taxonomy, qc_log = filter_microbiome_qc(asv, taxonomy, cfg)
            n_qc_excl = int((qc_log["stage"] == "microbiome_qc").sum()) if not qc_log.empty else 0
            print(f"      QC 后: {asv.shape[0]} 样本 × {asv.shape[1]} ASV（排除 {n_qc_excl} 例）")
        return asv, taxonomy, qc_log, False
    if use_demo:
        print(f"[2/11] 未找到 ASV 数据，生成演示用菌群数据")
        asv, taxonomy = generate_demo_microbiome(clinical, seed=cfg["analysis"]["random_seed"])
        demo_dir = root / "data" / "microbiome"
        demo_dir.mkdir(parents=True, exist_ok=True)
        asv.to_csv(demo_dir / "asv_table_demo.csv")
        taxonomy.to_csv(demo_dir / "taxonomy_demo.csv")
        return asv, taxonomy, pd.DataFrame(), True
    raise FileNotFoundError("缺少菌群数据，请将 ASV 表放入 data/microbiome/ 或开启 use_demo_microbiome")


def main(config_path: str = "config.yaml"):
    cfg = load_config(Path(config_path))
    root = Path(__file__).resolve().parent
    out_root = root / cfg["project"]["output_dir"]
    fig_dir = out_root / "figures"
    tab_dir = out_root / "tables"
    sens_dir = out_root / "sensitivity"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    clinical_path = (root / cfg["paths"]["clinical_data"]).resolve()
    print(f"[1/11] 加载临床数据: {clinical_path}")
    clinical_raw = load_clinical_excel(clinical_path)
    print(f"      原始样本: {len(clinical_raw)} 例")

    clinical, clinical_log = apply_inclusion_exclusion(clinical_raw, cfg)
    print(f"      临床纳入: {len(clinical)} 例（排除 {len(clinical_raw) - len(clinical)} 例）")

    clinical = add_outcome_groups(
        clinical,
        split=cfg["grouping"]["extubation_split"],
        fixed_min=cfg["grouping"]["extubation_fixed_min"],
    )
    exclusion_logs = [clinical_log]

    asv_path = Path(cfg["paths"]["microbiome_asv"])
    if not asv_path.is_absolute():
        asv_path = root / asv_path
    tax_path = Path(cfg["paths"]["taxonomy"])
    if not tax_path.is_absolute():
        tax_path = root / tax_path

    asv, taxonomy, qc_log, use_demo = _load_microbiome(cfg, root, clinical)
    if not qc_log.empty:
        exclusion_logs.append(qc_log)

    # 保存原始 ASV 副本供敏感性分析
    asv_for_sensitivity = load_asv_table(asv_path) if asv_path.exists() else asv.copy()
    tax_for_sensitivity = load_taxonomy(tax_path) if tax_path.exists() else taxonomy.copy()

    pre_merge_ids = set(clinical.index)
    clinical = merge_microbiome(clinical, asv, taxonomy)
    lost_ids = pre_merge_ids - set(clinical.index)
    if lost_ids:
        merge_log = pd.DataFrame(
            [{"sample_id": sid, "stage": "merge", "reason": "临床样本与 QC 后菌群表无交集"} for sid in sorted(lost_ids)]
        )
        exclusion_logs.append(merge_log)
    print(f"      最终纳入分析: {len(clinical)} 例")

    exclusion_df = save_exclusion_log(exclusion_logs, out_root)

    print("[3/11] Table 1 — 基线特征")
    run_baseline(clinical, tab_dir)

    print("[4/11] Figure 1 — 菌群全景")
    run_figure1(clinical, fig_dir)

    print("[5/11] Figure 2-3 — α/β 多样性")
    run_figure2(clinical, fig_dir)
    run_figure3(clinical, fig_dir, permutations=cfg["analysis"]["permutations"])

    print("[6/11] Figure 4 — 差异菌群")
    run_figure4(clinical, fig_dir, lda_threshold=cfg["analysis"]["lda_threshold"])

    print("[7/11] Figure 5-7 — 菌群-炎症轴 & 中介分析")
    run_figure5(clinical, fig_dir, fdr_alpha=cfg["analysis"]["fdr_alpha"])
    run_figure6(clinical, fig_dir, crp_threshold=cfg["grouping"]["crp_threshold"])
    run_figure7(clinical, fig_dir, n_boot=cfg["analysis"]["bootstrap_n"])

    print("[8/11] Figure 8 & Table 2 — 预测模型 + MLP 验证")
    run_prediction(
        clinical,
        fig_dir,
        n_boot=cfg["analysis"]["bootstrap_n"],
        n_perm=cfg["analysis"].get("permutation_n", 500),
    )

    print("[9/11] Figure 9 — 关键预后菌群")
    run_figure9(clinical, fig_dir)

    print("[10/11] Table 3 — 多因素 Logistic 回归")
    run_multivariable(clinical, tab_dir)

    print("[11/11] 敏感性分析 — 严格 vs 放宽 QC")
    run_sensitivity_cohorts(clinical_raw, asv_for_sensitivity, tax_for_sensitivity, cfg, sens_dir)

    n_qc_excl = int((exclusion_df["stage"] == "microbiome_qc").sum()) if not exclusion_df.empty else 0
    n_clinical_excl = len(clinical_raw) - len(apply_inclusion_exclusion(clinical_raw, cfg)[0])
    summary = {
        "project": cfg["project"]["name"],
        "run_time": datetime.now().isoformat(),
        "n_samples_raw": len(clinical_raw),
        "n_samples": len(clinical),
        "n_clinical_excluded": n_clinical_excl,
        "n_qc_excluded": n_qc_excl,
        "n_early": int((clinical["extubation_group"] == "Early").sum()),
        "n_delayed": int((clinical["extubation_group"] == "Delayed").sum()),
        "demo_microbiome": use_demo,
        "output_dir": str(out_root),
        "modules_completed": [
            "table1_baseline", "figure1-9", "table2_model_performance",
            "table3_multivariable_or", "sensitivity_cohort_summary",
            "exclusion_log", "mlp_validation", "permutation_test",
        ],
    }
    with open(out_root / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    clinical.reset_index().to_csv(out_root / "processed_clinical_data.csv", index=False, encoding="utf-8-sig")
    print("\n✅ 完整分析流程已完成！")
    print(f"   图表: {fig_dir}")
    print(f"   表格: {tab_dir}")
    print(f"   敏感性: {sens_dir}")
    print(f"   摘要: {out_root / 'run_summary.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AICU 呼吸道菌群预后研究分析流水线")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    args = parser.parse_args()
    main(args.config)
