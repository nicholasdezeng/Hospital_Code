#!/usr/bin/env python3
"""测序质量诊断：逐样本原始 reads、QC 通过率、A/a 块对比。

用法（服务器）:
  python scripts/diagnose_sequencing_qc.py --config config.server.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_asv_table, load_clinical_excel, load_taxonomy
from src.preprocessing import filter_microbiome_qc


def main(config_path: str = "config.server.yaml"):
    cfg = yaml.safe_load(open(ROOT / config_path, encoding="utf-8"))
    asv_path = Path(cfg["paths"]["microbiome_asv"])
    tax_path = Path(cfg["paths"]["taxonomy"])
    clinical_path = Path(cfg["paths"]["clinical_data"])

    asv = load_asv_table(asv_path)
    taxonomy = load_taxonomy(tax_path)
    clinical = load_clinical_excel(clinical_path)

    raw_reads = asv.sum(axis=1)
    obs_asv = (asv > 0).sum(axis=1)
    diag = pd.DataFrame({
        "sample_id": raw_reads.index,
        "raw_reads": raw_reads.values,
        "observed_asv_raw": obs_asv.values,
        "block": ["upper_A" if s[0].isupper() else "lower_a" for s in raw_reads.index],
    })
    diag = diag.merge(
        clinical.reset_index()[["sample_id", "extubation_time_min", "crp", "adverse_event"]],
        on="sample_id",
        how="left",
    )

    qc = cfg.get("qc", {})
    min_raw = qc.get("min_raw_reads", 3000)
    diag["pass_raw_reads"] = diag["raw_reads"] >= min_raw

    _, _, qc_log = filter_microbiome_qc(asv.copy(), taxonomy.copy(), cfg)
    excluded = set(qc_log.loc[qc_log["stage"] == "microbiome_qc", "sample_id"])
    diag["pass_full_qc"] = ~diag["sample_id"].isin(excluded)

    out_dir = ROOT / cfg["project"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sequencing_qc_diagnosis.csv"
    diag.sort_values("raw_reads").to_csv(out_path, index=False, encoding="utf-8-sig")

    print("=" * 60)
    print("测序质量诊断报告")
    print("=" * 60)
    print(f"ASV 表样本数: {len(diag)}")
    print(f"原始 reads 中位数: {diag['raw_reads'].median():.0f}")
    print(f"原始 reads >= {min_raw}: {diag['pass_raw_reads'].sum()} / {len(diag)}")
    print(f"通过完整 QC: {diag['pass_full_qc'].sum()} / {len(diag)}")
    print()
    print("按块统计 (upper_A vs lower_a):")
    grp = diag.groupby("block").agg(
        n=("sample_id", "count"),
        median_reads=("raw_reads", "median"),
        pass_qc=("pass_full_qc", "sum"),
    )
    print(grp.to_string())
    print()
    print("reads 最低的 10 个样本:")
    print(diag.nsmallest(10, "raw_reads")[["sample_id", "block", "raw_reads", "pass_full_qc"]].to_string(index=False))
    print()
    print(f"详细结果: {out_path}")
    print()
    print("建议:")
    low_a = diag[(diag["block"] == "upper_A") & (~diag["pass_raw_reads"])]
    if len(low_a) >= 5:
        print("  [!] 大写 A 样本大量低 reads → 请核对 FASTQ 对照表与 DADA2 样本 ID")
    if diag["pass_full_qc"].sum() < 35:
        print("  [!] 通过 QC 样本 < 35 → 考虑放宽 min_raw_reads 或排查测序失败样本")
    if diag["raw_reads"].median() < 5000:
        print("  [!] 整体测序深度偏低 → 检查 DADA2 filterAndTrim 参数是否过严")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.server.yaml")
    args = parser.parse_args()
    main(args.config)
