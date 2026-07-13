"""将分散的 CSV/JSON 结果汇总为规范化 Excel 工作簿。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.utils.stats import format_p

WORKBOOK_NAME = "AICU_分析结果汇总.xlsx"

# 工作表顺序与中文名
SHEET_FLOW = "01_分析流程"
SHEET_SUMMARY = "00_分析摘要"
SHEET_EXCLUSION = "02_排除日志"
SHEET_COHORT = "03_分析样本清单"
SHEET_TABLE1 = "04_基线特征(Table1)"
SHEET_TABLE2 = "05_模型性能(Table2)"
SHEET_TABLE3 = "06_多因素回归(Table3)"
SHEET_FIG1 = "07_菌群门水平"
SHEET_FIG3 = "08_β多样性(PERMANOVA)"
SHEET_FIG4 = "09_差异菌群(LEfSe)"
SHEET_FIG5 = "10_多样性-炎症相关"
SHEET_FIG6 = "11_四象限分层"
SHEET_FIG7 = "12_中介分析"
SHEET_FIG8 = "13_预测模型详情"
SHEET_FIG9 = "14_关键预后菌属"
SHEET_SENS = "15_敏感性分析"
SHEET_INDEX = "16_图表文件索引"

VAR_LABELS = {
    "shannon": "Shannon指数",
    "log_crp": "log(CRP)",
    "asa": "ASA分级",
    "anesthesia_duration_min": "麻醉时长(min)",
    "const": "截距",
}

COHORT_COLS = [
    ("sample_id", "样本编号"),
    ("extubation_group", "拔管分组"),
    ("extubation_time_min", "拔管时间(min)"),
    ("icu_stay_min", "ICU滞留(min)"),
    ("adverse_event", "不良反应"),
    ("age", "年龄"),
    ("sex", "性别(1男0女)"),
    ("asa", "ASA"),
    ("wbc", "WBC"),
    ("crp", "CRP"),
    ("pct", "PCT"),
    ("nlr", "NLR"),
    ("shannon", "Shannon"),
    ("observed_asv", "观测ASV数"),
    ("assigned_reads", "有效读段"),
]

HEADER_FMT = {
    "bold": True,
    "font_color": "white",
    "bg_color": "#2F5496",
    "border": 1,
    "valign": "vcenter",
    "text_wrap": True,
}
TITLE_FMT = {"bold": True, "font_size": 14, "font_color": "#1F3864"}
SECTION_FMT = {"bold": True, "font_size": 11, "font_color": "#2F5496"}
NOTE_FMT = {"font_size": 9, "font_color": "#666666", "text_wrap": True}


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path)


def _first_csv(*paths: Path) -> pd.DataFrame | None:
    for path in paths:
        df = _read_csv(path)
        if df is not None and not df.empty:
            return df
    for path in paths:
        df = _read_csv(path)
        if df is not None:
            return df
    return None


def _format_table3(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["结局", "模型", "变量", "OR(95%CI)", "P值", "N"])
    out = df.copy()
    out["变量"] = out["variable"].map(lambda v: VAR_LABELS.get(v, v))
    out["OR(95%CI)"] = out.apply(
        lambda r: f"{r['OR']:.2f} ({r['CI_low']:.2f}-{r['CI_high']:.2f})", axis=1
    )
    out["P值"] = out["P_value"].map(format_p)
    return out[["outcome_label", "model", "变量", "OR(95%CI)", "P值", "N"]].rename(
        columns={"outcome_label": "结局", "model": "模型"}
    )


def _format_table2(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    rename = {
        "Target": "预测结局",
        "Model": "模型",
        "AUC": "AUC",
        "AUC_CI_low": "AUC CI下限",
        "AUC_CI_high": "AUC CI上限",
        "Permutation_P": "置换检验P",
        "Accuracy": "准确率",
        "Sensitivity": "灵敏度",
        "Specificity": "特异度",
        "F1": "F1",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    if "置换检验P" in out.columns:
        out["置换检验P"] = out["置换检验P"].map(format_p)
    for col in ("AUC", "AUC CI下限", "AUC CI上限", "准确率", "灵敏度", "特异度", "F1"):
        if col in out.columns:
            out[col] = out[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    return out


def _format_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    labels = {"strict": "严格QC", "main": "主分析QC", "relaxed": "放宽QC"}
    out = df.copy()
    out["cohort"] = out["cohort"].map(lambda c: labels.get(c, c))
    if "shannon_group_p" in out.columns:
        out["shannon_group_p"] = out["shannon_group_p"].map(format_p)
    if "shannon_crp_p" in out.columns:
        out["shannon_crp_p"] = out["shannon_crp_p"].map(format_p)
    rename = {
        "cohort": "队列",
        "n_samples": "样本量",
        "n_early": "早期拔管n",
        "n_delayed": "延迟拔管n",
        "n_asv": "ASV数",
        "n_genera": "属数",
        "shannon_early_mean": "Early组Shannon均值",
        "shannon_delayed_mean": "Delayed组Shannon均值",
        "shannon_group_p": "Shannon组间P",
        "shannon_crp_rho": "Shannon-CRP ρ",
        "shannon_crp_p": "Shannon-CRP P",
        "qc_excluded": "QC排除数",
    }
    return out.rename(columns={k: v for k, v in rename.items() if k in out.columns})


def _build_flow_table(summary: dict) -> pd.DataFrame:
    n_raw = summary.get("n_samples_raw", "—")
    n_clin_ex = summary.get("n_clinical_excluded", "—")
    n_qc_ex = summary.get("n_qc_excluded", "—")
    n_final = summary.get("n_samples", "—")
    n_early = summary.get("n_early", "—")
    n_delayed = summary.get("n_delayed", "—")
    after_clin = int(n_raw) - int(n_clin_ex) if str(n_raw).isdigit() else "—"
    after_qc = after_clin if int(n_qc_ex or 0) == 0 else "—"

    rows = [
        ("1", "临床数据加载", "临床样本收集表", f"原始 {n_raw} 例", "—"),
        ("2", "临床纳入排除", "require_extubation_time 等", f"纳入 {after_clin} 例", f"排除 {n_clin_ex} 例"),
        ("3", "菌群 QC", "ASV 表 + taxonomy", f"通过 QC {after_qc} 例" if after_qc != "—" else f"QC排除 {n_qc_ex} 例", f"排除 {n_qc_ex} 例"),
        ("4", "临床-菌群合并", "样本编号交集", f"最终分析 {n_final} 例", "—"),
        ("5", "结局分组", "拔管时间中位数切分", f"Early {n_early} / Delayed {n_delayed}", "—"),
        ("6", "Figure 1-9", "α/β多样性、差异、预测等", "见「图表文件索引」", "—"),
        ("7", "Table 1-3", "基线、模型、多因素回归", "见对应工作表", "—"),
    ]
    return pd.DataFrame(rows, columns=["步骤", "环节", "输入/规则", "结果", "备注"])


def _build_key_findings(
    summary: dict,
    permanova: pd.DataFrame | None,
    table2: pd.DataFrame | None,
    table3: pd.DataFrame | None,
    sensitivity: pd.DataFrame | None,
    lefse: pd.DataFrame | None,
) -> list[tuple[str, str]]:
    findings: list[tuple[str, str]] = []

    n = summary.get("n_samples", "?")
    findings.append(("样本量", f"主分析纳入 {n} 例（Early {summary.get('n_early', '?')} / Delayed {summary.get('n_delayed', '?')}）"))

    if permanova is not None and not permanova.empty:
        p = permanova.iloc[0].get("p", float("nan"))
        r2 = permanova.iloc[0].get("R2", float("nan"))
        findings.append(("β多样性", f"PERMANOVA R²={r2:.3f}, P={format_p(p)}（组间菌群组成差异）"))

    if sensitivity is not None and not sensitivity.empty:
        main = sensitivity[sensitivity["cohort"] == "main"]
        if not main.empty:
            sp = main.iloc[0].get("shannon_group_p", float("nan"))
            findings.append(("α多样性", f"Shannon 组间比较 P={format_p(sp)}"))

    if lefse is not None and not lefse.empty:
        pcol = "p" if "p" in lefse.columns else "p_value" if "p_value" in lefse.columns else None
        n_sig = int((lefse[pcol] < 0.05).sum()) if pcol else len(lefse)
        findings.append(("差异菌群", f"LEfSe 分析完成；P<0.05 的分类单元 {n_sig} 个（详见工作表）"))

    if table2 is not None and not table2.empty:
        for target in table2["Target"].unique() if "Target" in table2.columns else []:
            sub = table2[table2["Target"] == target]
            best = sub.loc[sub["AUC"].idxmax()]
            findings.append(
                (
                    f"预测—{target}",
                    f"最佳模型 {best['Model']}：AUC={best['AUC']:.3f}，置换P={format_p(best.get('Permutation_P', float('nan')))}",
                )
            )

    if table3 is not None and not table3.empty:
        shannon_rows = table3[table3["variable"] == "shannon"]
        for _, r in shannon_rows.iterrows():
            findings.append(
                (
                    f"多因素—{r.get('outcome_label', r.get('outcome', ''))}",
                    f"Shannon OR={r['OR']:.2f}，P={format_p(r['P_value'])}（{r.get('model', '')}）",
                )
            )

    demo = summary.get("demo_microbiome", False)
    if demo:
        findings.append(("数据说明", "当前使用演示菌群数据；服务器真实 FASTQ/DADA2 结果请以 config.server.yaml 重跑为准"))

    if len(findings) <= 2:
        findings.append(("提示", "部分模块未运行；完整结论请执行全流程: python run_analysis.py --config <config>"))

    return findings


def _build_figure_index(fig_dir: Path) -> pd.DataFrame:
    catalog = [
        (1, "figure1_microbiome_overview.png", "菌群全景：门水平组成、Top属、稀释曲线"),
        (2, "figure2_alpha_diversity.png", "α多样性：Shannon/Observed/Evenness 组间比较"),
        (3, "figure3_beta_diversity.png", "β多样性：PCoA + PERMANOVA"),
        (4, "figure4_differential_microbiota.png", "差异菌群：LEfSe 柱状图"),
        (5, "figure5_inflammation_correlation.png", "Shannon 与炎症指标 Spearman 相关"),
        (6, "figure6_quadrant_analysis.png", "高/低多样性 × 高/低炎症 四象限"),
        (7, "figure7_mediation.png", "菌群→炎症→结局 中介效应"),
        (8, "figure8_prediction_roc.png", "Logistic A/B/C + MLP 预测 ROC"),
        (9, "figure9_key_biomarkers.png", "关键预后属与结局相关热图"),
    ]
    rows = []
    for num, fname, desc in catalog:
        path = fig_dir / fname
        rows.append(
            {
                "图号": f"Figure {num}",
                "文件名": fname,
                "说明": desc,
                "状态": "已生成" if path.exists() else "未生成",
                "路径": str(path) if path.exists() else "",
            }
        )
    return pd.DataFrame(rows)


def _write_df_sheet(
    writer: pd.ExcelWriter,
    sheet_name: str,
    df: pd.DataFrame,
    *,
    col_widths: dict[int, int] | None = None,
) -> None:
    if df is None or df.empty:
        pd.DataFrame({"说明": ["（本次分析未生成此项数据）"]}).to_excel(
            writer, sheet_name=sheet_name, index=False
        )
        return
    df.to_excel(writer, sheet_name=sheet_name, index=False)
    ws = writer.sheets[sheet_name]
    workbook = writer.book
    header = workbook.add_format(HEADER_FMT)
    for col_num, value in enumerate(df.columns.values):
        ws.write(0, col_num, value, header)
    ws.freeze_panes(1, 0)
    ws.autofilter(0, 0, len(df), len(df.columns) - 1)
    if col_widths:
        for col, width in col_widths.items():
            ws.set_column(col, col, width)
    else:
        for i, col in enumerate(df.columns):
            max_len = max(len(str(col)), df[col].astype(str).str.len().max() if len(df) else 0)
            ws.set_column(i, i, min(max_len + 2, 48))


def _write_summary_sheet(
    writer: pd.ExcelWriter,
    summary: dict,
    findings: list[tuple[str, str]],
    cfg: dict,
) -> None:
    sheet = "00_分析摘要"
    wb = writer.book
    ws = wb.add_worksheet(sheet)
    writer.sheets[sheet] = ws

    title = wb.add_format(TITLE_FMT)
    section = wb.add_format(SECTION_FMT)
    note = wb.add_format(NOTE_FMT)
    cell = wb.add_format({"text_wrap": True, "valign": "top"})

    row = 0
    ws.write(row, 0, summary.get("project", "AICU呼吸道菌群预后研究"), title)
    row += 2

    ws.write(row, 0, "运行信息", section)
    row += 1
    info = [
        ("运行时间", summary.get("run_time", datetime.now().isoformat())),
        ("配置文件", cfg.get("_config_path", "")),
        ("输出目录", summary.get("output_dir", "")),
        ("菌群数据", "演示数据" if summary.get("demo_microbiome") else "真实 DADA2 数据"),
        ("已完成模块", ", ".join(summary.get("modules_completed", []))),
    ]
    for label, val in info:
        ws.write(row, 0, label, section)
        ws.write(row, 1, str(val), cell)
        row += 1
    row += 1

    ws.write(row, 0, "主要结论（自动摘要）", section)
    row += 1
    ws.write(row, 0, "维度", header := wb.add_format(HEADER_FMT))
    ws.write(row, 1, "结论", header)
    row += 1
    for dim, text in findings:
        ws.write(row, 0, dim, cell)
        ws.write(row, 1, text, cell)
        row += 1

    row += 1
    ws.write(
        row,
        0,
        "说明：本工作簿整合 Table 1–3、各 Figure 统计表、排除日志与敏感性分析；"
        "论文用表以 Table 1–3 为准，图表见 figures/ 目录 PNG 文件。",
        note,
    )
    ws.set_column(0, 0, 22)
    ws.set_column(1, 1, 72)


def export_results_workbook(
    out_root: Path,
    cfg: dict,
    summary: dict,
    clinical: pd.DataFrame | None = None,
    exclusion_df: pd.DataFrame | None = None,
) -> Path:
    """汇总 output/ 下已有 CSV，写出规范化 Excel。"""
    fig_dir = out_root / "figures"
    tab_dir = out_root / "tables"
    sens_dir = out_root / "sensitivity"
    out_path = out_root / WORKBOOK_NAME

    table1 = _read_csv(tab_dir / "table1_baseline.csv")
    table2_raw = _first_csv(tab_dir / "table2_model_performance.csv", fig_dir / "table2_model_performance.csv")
    table3_raw = _read_csv(tab_dir / "table3_multivariable_or.csv")
    exclusion = exclusion_df if exclusion_df is not None else _read_csv(out_root / "exclusion_log.csv")
    sensitivity_raw = _read_csv(sens_dir / "sensitivity_cohort_summary.csv")
    permanova = _read_csv(fig_dir / "figure3_permanova.csv")
    lefse = _read_csv(fig_dir / "figure4_lefse_results.csv")
    fig1_phylum = _read_csv(fig_dir / "figure1_phylum_summary.csv")
    fig5 = _read_csv(fig_dir / "figure5_correlations.csv")
    fig6 = _read_csv(fig_dir / "figure6_quadrant_summary.csv")
    fig7 = _read_csv(fig_dir / "figure7_mediation.csv")
    fig8_mlp = _read_csv(fig_dir / "figure8_mlp_comparison.csv")
    fig8_shap = _read_csv(fig_dir / "figure8_shap_importance.csv")
    fig9_cons = _read_csv(fig_dir / "figure9_consensus_genera.csv")
    fig9_corr = _read_csv(fig_dir / "figure9_genus_outcome_correlations.csv")

    table2_fmt = _format_table2(table2_raw) if table2_raw is not None else pd.DataFrame()
    table3_fmt = _format_table3(table3_raw) if table3_raw is not None else pd.DataFrame()
    sensitivity_fmt = _format_sensitivity(sensitivity_raw) if sensitivity_raw is not None else pd.DataFrame()
    flow = _build_flow_table(summary)
    findings = _build_key_findings(summary, permanova, table2_raw, table3_raw, sensitivity_raw, lefse)
    fig_index = _build_figure_index(fig_dir)

    if clinical is not None and not clinical.empty:
        cols = [c for c, _ in COHORT_COLS if c in clinical.columns or c == "sample_id"]
        cohort = clinical.reset_index()
        if "sample_id" not in cohort.columns and cohort.columns[0] != "sample_id":
            cohort = cohort.rename(columns={cohort.columns[0]: "sample_id"})
        pick = [c for c, _ in COHORT_COLS if c in cohort.columns]
        cohort = cohort[pick].rename(columns=dict(COHORT_COLS))
    else:
        cohort = _read_csv(out_root / "processed_clinical_data.csv")
        if cohort is not None and "sample_id" not in cohort.columns:
            id_col = cohort.columns[0]
            cohort = cohort.rename(columns={id_col: "样本编号"})

    if exclusion is not None and not exclusion.empty:
        exclusion = exclusion.rename(columns={"sample_id": "样本编号", "stage": "阶段", "reason": "原因"})

    if fig8_mlp is not None and fig8_shap is not None:
        fig8 = pd.concat(
            [fig8_mlp.assign(类型="MLP对比"), fig8_shap.assign(类型="SHAP重要性")],
            ignore_index=True,
            sort=False,
        )
    elif fig8_mlp is not None:
        fig8 = fig8_mlp
    elif fig8_shap is not None:
        fig8 = fig8_shap
    else:
        fig8 = None

    if fig9_cons is not None and fig9_corr is not None:
        fig9 = pd.concat(
            [fig9_cons.assign(表="共识属"), fig9_corr.assign(表="属-结局相关")],
            ignore_index=True,
            sort=False,
        )
    elif fig9_cons is not None:
        fig9 = fig9_cons
    elif fig9_corr is not None:
        fig9 = fig9_corr
    else:
        fig9 = None

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        _write_summary_sheet(writer, summary, findings, cfg)
        _write_df_sheet(writer, SHEET_FLOW, flow, col_widths={0: 6, 1: 14, 2: 28, 3: 22, 4: 16})
        _write_df_sheet(writer, SHEET_EXCLUSION, exclusion if exclusion is not None else pd.DataFrame())
        _write_df_sheet(writer, SHEET_COHORT, cohort if cohort is not None else pd.DataFrame())
        _write_df_sheet(writer, SHEET_TABLE1, table1 if table1 is not None else pd.DataFrame())
        _write_df_sheet(writer, SHEET_TABLE2, table2_fmt)
        _write_df_sheet(writer, SHEET_TABLE3, table3_fmt)
        _write_df_sheet(writer, SHEET_FIG1, fig1_phylum if fig1_phylum is not None else pd.DataFrame())
        _write_df_sheet(writer, SHEET_FIG3, permanova if permanova is not None else pd.DataFrame())
        if lefse is not None and not lefse.empty:
            lefse_out = lefse.head(50).copy()
            if "lda" in lefse_out.columns:
                lefse_out = lefse_out.sort_values("lda", key=abs, ascending=False)
        else:
            lefse_out = pd.DataFrame()
        _write_df_sheet(writer, SHEET_FIG4, lefse_out)
        _write_df_sheet(writer, SHEET_FIG5, fig5 if fig5 is not None else pd.DataFrame())
        _write_df_sheet(writer, SHEET_FIG6, fig6 if fig6 is not None else pd.DataFrame())
        _write_df_sheet(writer, SHEET_FIG7, fig7 if fig7 is not None else pd.DataFrame())
        _write_df_sheet(writer, SHEET_FIG8, fig8 if fig8 is not None else pd.DataFrame())
        _write_df_sheet(writer, SHEET_FIG9, fig9 if fig9 is not None else pd.DataFrame())
        _write_df_sheet(writer, SHEET_SENS, sensitivity_fmt)
        _write_df_sheet(writer, SHEET_INDEX, fig_index)

    # 同步 table2 到 tables/（规范化目录）
    if table2_raw is not None and not table2_raw.empty:
        tab_dir.mkdir(parents=True, exist_ok=True)
        table2_fmt.to_csv(tab_dir / "table2_model_performance.csv", index=False, encoding="utf-8-sig")

    return out_path
