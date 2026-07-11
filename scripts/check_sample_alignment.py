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


def main():
    parser = argparse.ArgumentParser(description="检查样本对齐")
    parser.add_argument("--config", default="config.server.yaml")
    args = parser.parse_args()

    with open(ROOT / args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    clinical_path = Path(cfg["paths"]["clinical_data"])
    manifest_path = Path(cfg["paths"].get("sample_manifest_output", ROOT / "data/microbiome/sample_manifest.csv"))
    if not manifest_path.exists():
        manifest_path = ROOT / "data/microbiome/sample_manifest.csv"
    fastq_dir = Path(cfg["paths"]["raw_fastq_dir"])

    print("=== 样本对齐检查 ===\n")

    clinical = load_clinical_excel(clinical_path)
    clinical_ids = set(clinical.index.astype(str))
    print(f"临床表样本数: {len(clinical_ids)}")

    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        fastq_map = dict(zip(manifest["fastq_prefix"], manifest["clinical_sample_id"]))
        manifest_clinical = set(manifest["clinical_sample_id"].astype(str))
        print(f"对照表 FASTQ 数: {len(manifest)}")
        print(f"对照表临床 ID 数: {len(manifest_clinical)}")
    else:
        print(f"对照表不存在: {manifest_path}")
        print("请先运行: python scripts/build_sample_manifest.py --config config.server.yaml")
        fastq_map = {}
        manifest_clinical = set()

    if fastq_dir.exists():
        fastq_prefixes = sorted({p.name.rsplit("_", 1)[0] for p in fastq_dir.glob("*_1.fq")})
        print(f"FASTQ 文件数 (_1): {len(fastq_prefixes)}")
    else:
        fastq_prefixes = []
        print(f"FASTQ 目录不存在: {fastq_dir}")

    print("\n--- 临床表 vs 对照表 ---")
    only_clinical = sorted(clinical_ids - manifest_clinical)
    only_manifest = sorted(manifest_clinical - clinical_ids)
    both = sorted(clinical_ids & manifest_clinical)
    print(f"  交集: {len(both)}")
    print(f"  仅在临床表: {len(only_clinical)}", only_clinical[:10] if only_clinical else "")
    print(f"  仅在对照表: {len(only_manifest)}", only_manifest[:10] if only_manifest else "")

    if fastq_prefixes:
        print("\n--- FASTQ vs 对照表 ---")
        unmapped = sorted(set(fastq_prefixes) - set(fastq_map.keys()))
        no_file = sorted(set(fastq_map.keys()) - set(fastq_prefixes))
        print(f"  已对照 FASTQ: {len(set(fastq_prefixes) & set(fastq_map.keys()))}")
        print(f"  磁盘有、对照表无: {len(unmapped)}", unmapped[:10] if unmapped else "")
        print(f"  对照表有、磁盘无: {len(no_file)}", no_file[:10] if no_file else "")

    print("\n--- 右块示例（临床 A4 → FASTQ 55-63）---")
    if manifest_path.exists():
        sub = manifest[manifest["clinical_sample_id"] == "A4"]
        if not sub.empty:
            print(sub.to_string(index=False))


if __name__ == "__main__":
    main()
