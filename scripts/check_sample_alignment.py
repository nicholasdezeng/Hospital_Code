#!/usr/bin/env python3
"""检查临床表、对照表、FASTQ 三者的样本对齐情况。

用法:
  python scripts/check_sample_alignment.py --config config.server.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_clinical_excel


def _split_case(ids: set[str]) -> tuple[set[str], set[str]]:
    lower = {i for i in ids if i and i[0].islower()}
    upper = {i for i in ids if i and i[0].isupper()}
    other = ids - lower - upper
    return lower, upper | other


def main():
    parser = argparse.ArgumentParser(description="检查样本对齐")
    parser.add_argument("--config", default="config.server.yaml")
    args = parser.parse_args()

    with open(ROOT / args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    clinical_path = Path(cfg["paths"]["clinical_data"])
    manifest_path = Path(cfg["paths"].get("sample_manifest_output", ROOT / "data/microbiome/sample_manifest.csv"))
    fastq_dir = Path(cfg["paths"]["raw_fastq_dir"])

    print("=== 样本对齐检查 ===\n")

    clinical = load_clinical_excel(clinical_path)
    clinical_ids = set(clinical.index.astype(str))
    clin_lower, clin_upper = _split_case(clinical_ids)

    print(f"临床表样本数: {len(clinical_ids)}")
    print(f"  小写 (a*): {len(clin_lower)}")
    print(f"  大写 (A*): {len(clin_upper)}")

    if not manifest_path.exists():
        print(f"\n对照表不存在: {manifest_path}")
        print("请先运行: python scripts/build_sample_manifest.py --config config.server.yaml")
        return

    manifest = pd.read_csv(manifest_path)
    manifest_clinical = set(manifest["clinical_sample_id"].astype(str))
    man_lower, man_upper = _split_case(manifest_clinical)

    print(f"\n对照表 FASTQ 数: {len(manifest)}")
    print(f"对照表临床 ID 数: {len(manifest_clinical)}")
    print(f"  小写 (a*): {len(man_lower)}")
    print(f"  大写 (A*): {len(man_upper)}")

    print("\n--- 小写编号对齐 (a1, a2, ...) ---")
    print(f"  交集: {len(clin_lower & man_lower)}")
    print(f"  仅在临床表: {len(clin_lower - man_lower)}", sorted(clin_lower - man_lower)[:8])
    print(f"  仅在对照表: {len(man_lower - clin_lower)}", sorted(man_lower - clin_lower)[:8])

    print("\n--- 大写编号对齐 (A1, A2, ...) ---")
    print(f"  交集: {len(clin_upper & man_upper)}")
    print(f"  仅在临床表: {len(clin_upper - man_upper)}", sorted(clin_upper - man_upper)[:8])
    print(f"  仅在对照表: {len(man_upper - clin_upper)}", sorted(man_upper - clin_upper)[:8])

    multi = manifest.groupby("clinical_sample_id").filter(lambda g: len(g) > 1)
    if len(multi):
        print(f"\n--- ⚠ 同一临床编号多个 FASTQ ({multi['clinical_sample_id'].nunique()} 个) ---")
        print(multi[["clinical_sample_id", "fastq_prefix", "block"]].to_string(index=False))

    if fastq_dir.exists():
        fastq_prefixes = {p.name.rsplit("_", 1)[0] for p in fastq_dir.glob("*_1.fq")}
        mapped = set(manifest["fastq_prefix"])
        print(f"\n--- FASTQ 文件 ---")
        print(f"  磁盘 FASTQ 数: {len(fastq_prefixes)}")
        print(f"  已对照: {len(fastq_prefixes & mapped)}")
        unmapped = sorted(fastq_prefixes - mapped)
        if unmapped:
            print(f"  未对照: {unmapped[:10]}")

    print("\n--- 关键示例 ---")
    for cid, fq in [("a1", "A4"), ("A1", "A1"), ("a4", "75-59"), ("A4", "55-63")]:
        hit = manifest[(manifest["clinical_sample_id"] == cid) & (manifest["fastq_prefix"] == fq)]
        status = "✓" if len(hit) else "✗"
        print(f"  {status} 临床 {cid} ← FASTQ {fq}")


if __name__ == "__main__":
    main()
