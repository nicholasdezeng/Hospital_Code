"""AICU 呼吸道菌群预后研究 — 主分析入口。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

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
from src.analysis.multivariable import run_continuous_extubation, run_multivariable
from src.analysis.prediction import run_prediction
from src.analysis.sensitivity import run_sensitivity_cohorts
from src.analysis.supplementary_tables import (
    run_ae_descriptive,
    run_genus_wilcoxon_table,
    run_shannon_inflammation_matrix,
    run_table1_supplement,
)
from src.data_loader import load_asv_table, load_clinical_excel, load_taxonomy
from src.preprocessing import (
    add_outcome_groups,
    apply_inclusion_exclusion,
    filter_microbiome_qc,
    generate_demo_microbiome,
    merge_microbiome,
    save_exclusion_log,
)
from src.report.export_workbook import export_results_workbook

FIGURE_CATALOG = {
    1: ("Figure 1 — 菌群全景", "figure1_microbiome_overview.png"),
    2: ("Figure 2 — α 多样性", "figure2_alpha_diversity.png"),
    3: ("Figure 3 — β 多样性", "figure3_beta_diversity.png"),
    4: ("Figure 4 — 差异菌群", "figure4_differential_microbiota.png"),
    5: ("Figure 5 — 多样性-炎症相关", "figure5_inflammation_correlation.png"),
    6: ("Figure 6 — 四象限分层", "figure6_quadrant_analysis.png"),
    7: ("Figure 7 — 中介分析", "figure7_mediation.png"),
    8: ("Figure 8 — 预测模型 ROC", "figure8_prediction_roc.png"),
    9: ("Figure 9 — 关键预后菌群", "figure9_key_biomarkers.png"),
}

TABLE_CATALOG = {
    "table1": "Table 1 — 基线特征",
    "table1_supplement": "Table 1-补充 — 两组 α 多样性",
    "table2": "Table 2 — 模型性能 A/B/C/E（随 Figure 8）",
    "table2_continuous": "Table 2 敏感性 — 拔管时间连续变量",
    "table3": "Table 3 — 多因素 Logistic",
    "genus_wilcoxon": "属水平 Wilcoxon + FDR",
    "ae_descriptive": "不良反应描述性",
    "sensitivity": "敏感性分析",
}


def load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_selection(spec: str | None, *, valid: range | set | None = None) -> set:
    """解析 'all' / '1' / '1,3,8' / '2-5' 形式的选择。"""
    if spec is None or str(spec).strip().lower() in ("all", "*", ""):
        if isinstance(valid, range):
            return set(valid)
        if valid is not None:
            return set(valid)
        return set()

    selected: set = set()
    for part in str(spec).split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part in ("all", "*"):
            if isinstance(valid, range):
                selected.update(valid)
            elif valid is not None:
                selected.update(valid)
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            selected.update(range(int(start), int(end) + 1))
        else:
            token = part.replace("figure", "").replace("fig", "")
            selected.add(int(token) if token.isdigit() else part)
    if valid is not None:
        allowed = set(valid) if not isinstance(valid, range) else set(valid)
        unknown = selected - allowed
        if unknown:
            raise ValueError(f"未知选项: {sorted(unknown)}；可选: {sorted(allowed)}")
    return selected


def _load_microbiome(cfg: dict, root: Path, clinical: pd.DataFrame):
    asv_path = Path(cfg["paths"]["microbiome_asv"])
    tax_path = Path(cfg["paths"]["taxonomy"])
    if not asv_path.is_absolute():
        asv_path = root / asv_path
    if not tax_path.is_absolute():
        tax_path = root / tax_path
    use_demo = cfg["analysis"]["use_demo_microbiome"]

    if asv_path.exists() and tax_path.exists():
        print("[数据] 加载真实菌群数据")
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
        print("[数据] 未找到 ASV 数据，生成演示用菌群数据")
        asv, taxonomy = generate_demo_microbiome(clinical, seed=cfg["analysis"]["random_seed"])
        demo_dir = root / "data" / "microbiome"
        demo_dir.mkdir(parents=True, exist_ok=True)
        asv.to_csv(demo_dir / "asv_table_demo.csv")
        taxonomy.to_csv(demo_dir / "taxonomy_demo.csv")
        return asv, taxonomy, pd.DataFrame(), True
    raise FileNotFoundError("缺少菌群数据，请将 ASV 表放入 data/microbiome/ 或开启 use_demo_microbiome")


def _resolve_dependencies(selected_figures: set[int]) -> list[int]:
    """按依赖顺序返回要运行的图编号（含自动补跑的依赖图）。"""
    deps = {8: [3], 9: [4]}
    ordered = sorted(selected_figures)
    resolved: list[int] = []
    for fig_id in ordered:
        for dep in deps.get(fig_id, []):
            if dep not in resolved:
                resolved.append(dep)
        if fig_id not in resolved:
            resolved.append(fig_id)
    return resolved


def _print_catalog():
    print("\n可选图表 (--figures):")
    for num, (label, filename) in FIGURE_CATALOG.items():
        print(f"  {num:>2}  {label:<28}  →  output/figures/{filename}")
    print("\n可选表格/模块 (--tables):")
    for key, label in TABLE_CATALOG.items():
        print(f"  {key:<12} {label}")
    print("\n示例:")
    print("  python run_analysis.py --config config.server.yaml --figures 1")
    print("  python run_analysis.py --config config.server.yaml --figures 1,3,8")
    print("  python run_analysis.py --config config.server.yaml --figures 2-4 --tables table1")
    print("  python run_analysis.py --config config.server.yaml   # 默认跑全部\n")


def _prepare_clinical(cfg: dict, root: Path):
    """加载并合并临床 + 菌群数据，返回分析用 DataFrame。"""
    clinical_path = Path(cfg["paths"]["clinical_data"])
    if not clinical_path.is_absolute():
        clinical_path = root / clinical_path
    clinical_path = clinical_path.resolve()

    print(f"[数据] 加载临床数据: {clinical_path}")
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

    exclusion_df = save_exclusion_log(exclusion_logs, root / cfg["project"]["output_dir"])
    return clinical, clinical_raw, exclusion_df, asv_for_sensitivity, tax_for_sensitivity, use_demo


def _run_figure(fig_id: int, clinical: pd.DataFrame, fig_dir: Path, cfg: dict, tab_dir: Path) -> str:
    runners: dict[int, Callable] = {
        1: lambda: run_figure1(clinical, fig_dir),
        2: lambda: run_figure2(clinical, fig_dir),
        3: lambda: run_figure3(clinical, fig_dir, permutations=cfg["analysis"]["permutations"]),
        4: lambda: run_figure4(clinical, fig_dir, lda_threshold=cfg["analysis"]["lda_threshold"]),
        5: lambda: run_figure5(clinical, fig_dir, fdr_alpha=cfg["analysis"]["fdr_alpha"]),
        6: lambda: run_figure6(clinical, fig_dir, crp_threshold=cfg["grouping"]["crp_threshold"]),
        7: lambda: run_figure7(clinical, fig_dir, n_boot=cfg["analysis"]["bootstrap_n"]),
        8: lambda: run_prediction(
            clinical,
            fig_dir,
            n_boot=cfg["analysis"]["bootstrap_n"],
            n_perm=cfg["analysis"].get("permutation_n", 500),
            tables_dir=tab_dir,
        ),
        9: lambda: run_figure9(clinical, fig_dir),
    }
    label, filename = FIGURE_CATALOG[fig_id]
    print(f"\n▶ {label}")
    runners[fig_id]()
    out_path = fig_dir / filename
    print(f"  ✓ 已保存: {out_path}")
    return str(out_path)


def main(
    config_path: str = "config.yaml",
    *,
    figures: str | None = None,
    tables: str | None = None,
    list_catalog: bool = False,
):
    if list_catalog:
        _print_catalog()
        return

    cfg = load_config(Path(config_path))
    cfg["_config_path"] = str(Path(config_path).resolve())
    root = Path(__file__).resolve().parent
    out_root = root / cfg["project"]["output_dir"]
    fig_dir = out_root / "figures"
    tab_dir = out_root / "tables"
    sens_dir = out_root / "sensitivity"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    run_all = figures is None and tables is None
    if run_all:
        selected_figures = set(range(1, 10))
        selected_tables = set(TABLE_CATALOG.keys())
    else:
        selected_figures = parse_selection(figures, valid=range(1, 10)) if figures is not None else set()
        selected_tables = (
            parse_selection(tables, valid=set(TABLE_CATALOG.keys()))
            if tables is not None
            else set()
        )

    if not selected_figures and not selected_tables:
        print("未选择任何输出。使用 --list 查看选项，或 --figures 1 指定单张图。")
        _print_catalog()
        return

    clinical, clinical_raw, exclusion_df, asv_for_sensitivity, tax_for_sensitivity, use_demo = _prepare_clinical(cfg, root)

    modules_completed: list[str] = []
    generated_files: list[str] = []

    if "table1" in selected_tables:
        print("\n▶ Table 1 — 基线特征")
        run_baseline(clinical, tab_dir)
        modules_completed.append("table1_baseline")
        generated_files.append(str(tab_dir / "table1_baseline.csv"))

    if "table1_supplement" in selected_tables:
        print("\n▶ Table 1-补充 — 两组 α 多样性")
        run_table1_supplement(clinical, tab_dir)
        modules_completed.append("table1_supplement_diversity")
        generated_files.append(str(tab_dir / "table1_supplement_diversity.csv"))

    figure_run_list = _resolve_dependencies(selected_figures)
    auto_deps = set(figure_run_list) - selected_figures
    if auto_deps:
        print(f"[依赖] 自动补跑: Figure {', '.join(str(x) for x in sorted(auto_deps))}")

    for fig_id in figure_run_list:
        generated_files.append(_run_figure(fig_id, clinical, fig_dir, cfg, tab_dir))
        modules_completed.append(f"figure{fig_id}")
        if fig_id == 8:
            modules_completed.extend(["table2_model_performance", "table2_factorial", "mlp_validation", "permutation_test"])
            generated_files.extend([
                str(tab_dir / "table2_model_performance.csv"),
                str(tab_dir / "table2_factorial_delta_auc.csv"),
            ])

    if "table2" in selected_tables and 8 not in selected_figures and 8 not in figure_run_list:
        print("\n▶ Table 2 — 模型性能（需 Figure 8 逻辑）")
        run_prediction(
            clinical,
            fig_dir,
            n_boot=cfg["analysis"]["bootstrap_n"],
            n_perm=cfg["analysis"].get("permutation_n", 500),
            tables_dir=tab_dir,
        )
        modules_completed.append("table2_model_performance")
        generated_files.append(str(fig_dir / "table2_model_performance.csv"))

    if "table3" in selected_tables:
        print("\n▶ Table 3 — 多因素 Logistic 回归")
        run_multivariable(clinical, tab_dir)
        modules_completed.append("table3_multivariable_or")
        generated_files.append(str(tab_dir / "table3_multivariable_or.csv"))

    if "table2_continuous" in selected_tables:
        print("\n▶ Table 2 敏感性 — 拔管时间连续变量")
        run_continuous_extubation(clinical, tab_dir)
        modules_completed.append("table2_continuous_extubation")
        generated_files.append(str(tab_dir / "table2_continuous_extubation.csv"))

    if "genus_wilcoxon" in selected_tables:
        print("\n▶ 属水平 Wilcoxon + FDR")
        run_genus_wilcoxon_table(clinical, tab_dir)
        modules_completed.append("table_genus_wilcoxon")
        generated_files.append(str(tab_dir / "table_genus_wilcoxon.csv"))

    if "ae_descriptive" in selected_tables:
        print("\n▶ 不良反应描述性分析")
        run_ae_descriptive(clinical, tab_dir)
        modules_completed.append("table_ae_descriptive")
        generated_files.append(str(tab_dir / "table_ae_descriptive.csv"))

    # 数据核查：Shannon-炎症相关（优化方案 Step 0）
    if run_all or "table1_supplement" in selected_tables:
        run_shannon_inflammation_matrix(clinical, tab_dir)
        generated_files.append(str(tab_dir / "table_shannon_inflammation_corr.csv"))

    if "sensitivity" in selected_tables:
        print("\n▶ 敏感性分析 — 严格 vs 放宽 QC")
        run_sensitivity_cohorts(clinical_raw, asv_for_sensitivity, tax_for_sensitivity, cfg, sens_dir)
        modules_completed.append("sensitivity_cohort_summary")
        generated_files.append(str(sens_dir / "sensitivity_cohort_summary.csv"))

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
        "figures_requested": sorted(selected_figures),
        "figures_run": figure_run_list,
        "tables_requested": sorted(selected_tables),
        "modules_completed": modules_completed,
        "generated_files": generated_files,
    }
    with open(out_root / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    clinical.reset_index().to_csv(out_root / "processed_clinical_data.csv", index=False, encoding="utf-8-sig")

    print("\n▶ 汇总 Excel — 分析结果工作簿")
    workbook_path = export_results_workbook(
        out_root, cfg, summary, clinical=clinical, exclusion_df=exclusion_df
    )
    generated_files.append(str(workbook_path))
    modules_completed.append("export_workbook")
    summary["modules_completed"] = modules_completed
    summary["generated_files"] = generated_files
    with open(out_root / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  ✓ 已保存: {workbook_path}")

    print("\n✅ 完成！")
    if selected_figures:
        print(f"   本次图表: Figure {', '.join(str(x) for x in sorted(selected_figures))}")
    if generated_files:
        print("   输出文件:")
        for path in generated_files:
            print(f"     - {path}")
    print(f"   摘要: {out_root / 'run_summary.json'}")
    print(f"   Excel汇总: {out_root / 'AICU_分析结果汇总.xlsx'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AICU 呼吸道菌群预后研究分析流水线（支持按图选择性输出）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_analysis.py --config config.server.yaml --figures 1
  python run_analysis.py --config config.server.yaml --figures 1,3,8
  python run_analysis.py --config config.server.yaml --figures 2-4
  python run_analysis.py --list
        """,
    )
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument(
        "--figures", "-f",
        metavar="SPEC",
        help="要生成的图: 1 / 1,3,8 / 2-5 / all（默认 all）",
    )
    parser.add_argument(
        "--tables", "-t",
        metavar="SPEC",
        help="要生成的表格: table1,table1_supplement,table2,table2_continuous,table3,genus_wilcoxon,ae_descriptive,sensitivity / all",
    )
    parser.add_argument("--list", action="store_true", help="列出所有可选图表与用法")
    args = parser.parse_args()

    if args.list:
        _print_catalog()
    else:
        main(args.config, figures=args.figures, tables=args.tables)
