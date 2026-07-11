"""AICU 呼吸道菌群预后研究 — 主分析入口。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

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
from src.analysis.prediction import run_prediction
from src.data_loader import load_asv_table, load_clinical_excel, load_taxonomy
from src.preprocessing import add_outcome_groups, generate_demo_microbiome, merge_microbiome


def load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str = "config.yaml"):
    cfg = load_config(Path(config_path))
    root = Path(__file__).resolve().parent
    out_root = root / cfg["project"]["output_dir"]
    fig_dir = out_root / "figures"
    tab_dir = out_root / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    clinical_path = (root / cfg["paths"]["clinical_data"]).resolve()
    print(f"[1/9] 加载临床数据: {clinical_path}")
    clinical = load_clinical_excel(clinical_path)
    clinical = add_outcome_groups(
        clinical,
        split=cfg["grouping"]["extubation_split"],
        fixed_min=cfg["grouping"]["extubation_fixed_min"],
    )
    print(f"      纳入样本: {len(clinical)} 例")

    asv_path = root / cfg["paths"]["microbiome_asv"]
    tax_path = root / cfg["paths"]["taxonomy"]
    use_demo = cfg["analysis"]["use_demo_microbiome"]

    if asv_path.exists() and tax_path.exists():
        print(f"[2/9] 加载真实菌群数据")
        asv = load_asv_table(asv_path)
        taxonomy = load_taxonomy(tax_path)
    elif use_demo:
        print(f"[2/9] 未找到 ASV 数据，生成演示用菌群数据（请替换为真实测序结果）")
        asv, taxonomy = generate_demo_microbiome(clinical, seed=cfg["analysis"]["random_seed"])
        demo_dir = root / "data" / "microbiome"
        demo_dir.mkdir(parents=True, exist_ok=True)
        asv.to_csv(demo_dir / "asv_table_demo.csv")
        taxonomy.to_csv(demo_dir / "taxonomy_demo.csv")
    else:
        raise FileNotFoundError("缺少菌群数据，请将 ASV 表放入 data/microbiome/ 或开启 use_demo_microbiome")

    clinical = merge_microbiome(clinical, asv, taxonomy)

    print("[3/9] Table 1 — 基线特征")
    run_baseline(clinical, tab_dir)

    print("[4/9] Figure 1 — 菌群全景")
    run_figure1(clinical, fig_dir)

    print("[5/9] Figure 2-3 — α/β 多样性")
    run_figure2(clinical, fig_dir)
    run_figure3(clinical, fig_dir, permutations=cfg["analysis"]["permutations"])

    print("[6/9] Figure 4 — 差异菌群")
    run_figure4(clinical, fig_dir, lda_threshold=cfg["analysis"]["lda_threshold"])

    print("[7/9] Figure 5-7 — 菌群-炎症轴 & 中介分析")
    run_figure5(clinical, fig_dir, fdr_alpha=cfg["analysis"]["fdr_alpha"])
    run_figure6(clinical, fig_dir, crp_threshold=cfg["grouping"]["crp_threshold"])
    run_figure7(clinical, fig_dir, n_boot=cfg["analysis"]["bootstrap_n"])

    print("[8/9] Figure 8 & Table 2 — 预测模型")
    run_prediction(clinical, fig_dir)

    print("[9/9] Figure 9 — 关键预后菌群")
    run_figure9(clinical, fig_dir)

    summary = {
        "project": cfg["project"]["name"],
        "run_time": datetime.now().isoformat(),
        "n_samples": len(clinical),
        "n_early": int((clinical["extubation_group"] == "Early").sum()),
        "n_delayed": int((clinical["extubation_group"] == "Delayed").sum()),
        "demo_microbiome": use_demo and not (asv_path.exists() and tax_path.exists()),
        "output_dir": str(out_root),
    }
    with open(out_root / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    clinical.reset_index().to_csv(out_root / "processed_clinical_data.csv", index=False, encoding="utf-8-sig")
    print("\n✅ 分析完成！")
    print(f"   图表: {fig_dir}")
    print(f"   表格: {tab_dir}")
    print(f"   摘要: {out_root / 'run_summary.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AICU 呼吸道菌群预后研究分析流水线")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    args = parser.parse_args()
    main(args.config)
