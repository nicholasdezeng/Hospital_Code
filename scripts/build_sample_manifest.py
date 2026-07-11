#!/usr/bin/env python3
"""从 data_number_sorted.xlsx 生成 FASTQ → 临床样本编号 对照表。

表格结构（每行左右两块，是两个不同样本）：
  左块：样本原始名称(a1…) → 数据编号(FASTQ前缀)   对应临床表小写 a1/a2/...
  右块：样本原始名称.1(A1…) → 数据编号.1(FASTQ前缀) 对应临床表大写 A1/A2/...

临床表 59 例 = 小写 a1–a43（左块）+ 大写 A1–A16（右块），编号大小写不可混用。

用法:
  python scripts/build_sample_manifest.py --config config.server.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _clean(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text if text and text.lower() != "nan" else None


def parse_data_number_sorted(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path)
    records: list[dict] = []

    for idx, row in raw.iterrows():
        # 右块 → 临床大写 A1, A2, ...
        fastq_r = _clean(row.get("数据编号.1"))
        clinical_r = _clean(row.get("样本原始名称.1"))
        lab_r = _clean(row.get("样本编号.1"))
        if fastq_r and clinical_r:
            records.append(
                {
                    "fastq_prefix": fastq_r,
                    "clinical_sample_id": clinical_r,  # 保留大写 A1
                    "lab_id": lab_r,
                    "original_name": clinical_r,
                    "block": "right",
                    "excel_row": idx,
                }
            )

        # 左块 → 临床小写 a1, a2, ...
        fastq_l = _clean(row.get("数据编号"))
        clinical_l = _clean(row.get("样本原始名称"))
        lab_l = _clean(row.get("样本编号"))
        if fastq_l and clinical_l:
            records.append(
                {
                    "fastq_prefix": fastq_l,
                    "clinical_sample_id": clinical_l,  # 保留小写 a1
                    "lab_id": lab_l,
                    "original_name": clinical_l,
                    "block": "left",
                    "excel_row": idx,
                }
            )

    manifest = pd.DataFrame(records)
    if manifest.empty:
        raise ValueError(f"未能从 {path} 解析出任何样本记录")

    # 每个 FASTQ 只保留一条；同一 FASTQ 在左右块都出现时右块优先
    manifest["priority"] = manifest["block"].map({"right": 0, "left": 1})
    manifest = manifest.sort_values(["fastq_prefix", "priority", "excel_row"])
    duplicates = manifest[manifest.duplicated(subset=["fastq_prefix"], keep=False)].copy()
    deduped = manifest.drop_duplicates(subset=["fastq_prefix"], keep="first").copy()

    # 检查：一个临床编号是否对应多个 FASTQ
    multi = deduped.groupby("clinical_sample_id").filter(lambda g: len(g) > 1)
    deduped.attrs["duplicates"] = duplicates
    deduped.attrs["multi_clinical"] = multi
    deduped.attrs["raw_count"] = len(records)
    return deduped.sort_values(["block", "clinical_sample_id"]).reset_index(drop=True)


def check_fastq_files(manifest: pd.DataFrame, fastq_dir: Path) -> pd.DataFrame:
    if not fastq_dir.exists():
        return manifest.assign(fastq_r1_exists=False, fastq_r2_exists=False, fastq_paired=False)

    r1_files = {p.name.rsplit("_", 1)[0] for p in fastq_dir.glob("*_1.fq")}
    r2_files = {p.name.rsplit("_", 1)[0] for p in fastq_dir.glob("*_2.fq")}

    out = manifest.copy()
    out["fastq_r1_exists"] = out["fastq_prefix"].isin(r1_files)
    out["fastq_r2_exists"] = out["fastq_prefix"].isin(r2_files)
    out["fastq_paired"] = out["fastq_r1_exists"] & out["fastq_r2_exists"]
    return out


def load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="构建 FASTQ → 临床样本编号 对照表")
    parser.add_argument("--config", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--fastq-dir", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.config:
        cfg = load_config(Path(args.config))
        manifest_path = Path(cfg["paths"]["sample_manifest"])
        fastq_dir = Path(cfg["paths"]["raw_fastq_dir"])
        out_cfg = cfg["paths"].get("sample_manifest_output", "data/microbiome/sample_manifest.csv")
        output = Path(out_cfg)
        if not output.is_absolute():
            output = ROOT / output
    else:
        if not args.manifest:
            parser.error("请指定 --config 或 --manifest")
        manifest_path = Path(args.manifest)
        fastq_dir = Path(args.fastq_dir) if args.fastq_dir else None
        output = Path(args.output or "sample_manifest.csv")

    print(f"读取对照表: {manifest_path}")
    manifest = parse_data_number_sorted(manifest_path)

    if fastq_dir:
        print(f"检查 FASTQ: {fastq_dir}")
        manifest = check_fastq_files(manifest, fastq_dir)

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False, encoding="utf-8-sig")

    n_lower = manifest["clinical_sample_id"].str[0].str.islower().sum()
    n_upper = manifest["clinical_sample_id"].str[0].str.isupper().sum()

    print(f"\n解析完成，写入: {output}")
    print(f"  解析记录（去重前）: {manifest.attrs.get('raw_count', '?')}")
    print(f"  唯一 FASTQ 前缀: {len(manifest)}")
    print(f"  小写临床编号 (a*): {n_lower}")
    print(f"  大写临床编号 (A*): {n_upper}")

    if "fastq_paired" in manifest.columns:
        print(f"  成对 FASTQ 存在: {manifest['fastq_paired'].sum()}/{len(manifest)}")

    multi = manifest.attrs.get("multi_clinical")
    if multi is not None and len(multi):
        print(f"\n  ⚠ 同一临床编号对应多个 FASTQ ({multi['clinical_sample_id'].nunique()} 个编号):")
        print(multi[["clinical_sample_id", "fastq_prefix", "block"]].to_string(index=False))

    print("\n前 10 行预览:")
    cols = ["fastq_prefix", "clinical_sample_id", "block", "lab_id"]
    if "fastq_paired" in manifest.columns:
        cols.append("fastq_paired")
    print(manifest[cols].head(10).to_string(index=False))

    print("\n右块示例 A4 → 55-63:")
    sub = manifest[(manifest["clinical_sample_id"] == "A4") & (manifest["block"] == "right")]
    if not sub.empty:
        print(sub[cols].to_string(index=False))
    else:
        print("  (未找到)")


if __name__ == "__main__":
    main()
