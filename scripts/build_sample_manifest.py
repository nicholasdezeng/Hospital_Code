#!/usr/bin/env python3
"""从 data_number_sorted.xlsx 生成 FASTQ → 临床样本编号 对照表。

表格结构说明（data_number_sorted.xlsx）：
  每行左右两列样本信息：
  - 左块：样本原始名称(a1…)、样本编号(D2604-xxxx)、数据编号(FASTQ前缀)
  - 右块：样本原始名称.1(A1…)、样本编号.1(D2604-xxxx)、数据编号.1(FASTQ前缀)

  临床表使用 A1/A2/... 编号，对应右块「样本原始名称.1」。
  FASTQ 文件名前缀对应「数据编号」或「数据编号.1」（如 55-63_1.fq → 55-63）。

用法:
  python scripts/build_sample_manifest.py --config config.server.yaml
  python scripts/build_sample_manifest.py --manifest /path/to/data_number_sorted.xlsx --output data/microbiome/sample_manifest.csv
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
        # 右块：临床分析主用（A1, A2, ...）
        fastq_r = _clean(row.get("数据编号.1"))
        clinical_r = _clean(row.get("样本原始名称.1")) or _clean(row.get("样本编号.1"))
        lab_r = _clean(row.get("样本编号.1"))
        if fastq_r and clinical_r:
            records.append(
                {
                    "fastq_prefix": fastq_r,
                    "clinical_sample_id": clinical_r.upper() if clinical_r.startswith(("a", "A")) and len(clinical_r) <= 4 else clinical_r,
                    "lab_id": lab_r,
                    "original_name": _clean(row.get("样本原始名称.1")),
                    "block": "right",
                    "excel_row": idx,
                }
            )

        # 左块：补充样本（部分 FASTQ 仅出现在左块）
        fastq_l = _clean(row.get("数据编号"))
        original_l = _clean(row.get("样本原始名称"))
        lab_l = _clean(row.get("样本编号"))
        if fastq_l:
            # 左块原始名 a1/a2 不等于临床 A1；优先用 FASTQ 名本身若为 A 编号
            if fastq_l.upper().startswith("A") and fastq_l[1:].isdigit():
                clinical_l = fastq_l.upper()
            elif original_l:
                clinical_l = original_l.upper()
            else:
                clinical_l = fastq_l
            records.append(
                {
                    "fastq_prefix": fastq_l,
                    "clinical_sample_id": clinical_l,
                    "lab_id": lab_l,
                    "original_name": original_l,
                    "block": "left",
                    "excel_row": idx,
                }
            )

    manifest = pd.DataFrame(records)
    if manifest.empty:
        raise ValueError(f"未能从 {path} 解析出任何样本记录")

    # 同一 FASTQ 前缀出现多次时，右块优先（与临床表一致）
    manifest["priority"] = manifest["block"].map({"right": 0, "left": 1})
    manifest = manifest.sort_values(["fastq_prefix", "priority", "excel_row"])
    deduped = manifest.drop_duplicates(subset=["fastq_prefix"], keep="first").copy()
    duplicates = manifest[manifest.duplicated(subset=["fastq_prefix"], keep=False)]

    deduped = deduped.sort_values("clinical_sample_id").reset_index(drop=True)
    deduped.attrs["duplicates"] = duplicates
    deduped.attrs["raw_count"] = len(manifest)
    return deduped


def check_fastq_files(manifest: pd.DataFrame, fastq_dir: Path) -> pd.DataFrame:
    if not fastq_dir.exists():
        manifest["fastq_r1_exists"] = False
        manifest["fastq_r2_exists"] = False
        return manifest

    r1_files = {p.name.rsplit("_", 1)[0] for p in fastq_dir.glob("*_1.fq")}
    r2_files = {p.name.rsplit("_", 1)[0] for p in fastq_dir.glob("*_2.fq")}

    manifest = manifest.copy()
    manifest["fastq_r1_exists"] = manifest["fastq_prefix"].isin(r1_files)
    manifest["fastq_r2_exists"] = manifest["fastq_prefix"].isin(r2_files)
    manifest["fastq_paired"] = manifest["fastq_r1_exists"] & manifest["fastq_r2_exists"]
    return manifest


def load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="构建 FASTQ → 临床样本编号 对照表")
    parser.add_argument("--config", default=None, help="config.server.yaml 路径")
    parser.add_argument("--manifest", default=None, help="data_number_sorted.xlsx 路径")
    parser.add_argument("--fastq-dir", default=None, help="FASTQ 目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    if args.config:
        cfg = load_config(Path(args.config))
        manifest_path = Path(cfg["paths"]["sample_manifest"])
        fastq_dir = Path(cfg["paths"]["raw_fastq_dir"])
        output = Path(args.output or ROOT / "data/microbiome/sample_manifest.csv")
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

    print(f"\n解析完成，写入: {output}")
    print(f"  总记录（去重前）: {manifest.attrs.get('raw_count', '?')}")
    print(f"  唯一 FASTQ 前缀: {len(manifest)}")

    if "fastq_paired" in manifest.columns:
        paired = manifest["fastq_paired"].sum()
        print(f"  成对 FASTQ 存在: {paired}/{len(manifest)}")
        missing = manifest[~manifest["fastq_paired"]]
        if len(missing):
            print(f"\n  ⚠ 缺少成对 FASTQ 的样本 ({len(missing)}):")
            print(missing[["fastq_prefix", "clinical_sample_id", "block"]].to_string(index=False))

        fastq_on_disk = set()
        if fastq_dir and fastq_dir.exists():
            fastq_on_disk = {p.name.rsplit("_", 1)[0] for p in fastq_dir.glob("*_1.fq")}
        mapped = set(manifest["fastq_prefix"])
        unmapped_fastq = sorted(fastq_on_disk - mapped)
        if unmapped_fastq:
            print(f"\n  ⚠ 磁盘上有但未写入对照表的 FASTQ ({len(unmapped_fastq)}):")
            print(", ".join(unmapped_fastq[:20]), "..." if len(unmapped_fastq) > 20 else "")

    dup = manifest.attrs.get("duplicates")
    if dup is not None and len(dup):
        print(f"\n  ℹ 重复 FASTQ 前缀（已按右块优先去重）: {dup['fastq_prefix'].nunique()} 个")

    print("\n前 10 行预览:")
    cols = ["fastq_prefix", "clinical_sample_id", "block", "lab_id"]
    if "fastq_paired" in manifest.columns:
        cols.append("fastq_paired")
    print(manifest[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
