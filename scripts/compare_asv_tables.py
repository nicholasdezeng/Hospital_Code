#!/usr/bin/env python3
"""对比两版 ASV 表 reads 与 QC 通过率。

用法:
  python scripts/compare_asv_tables.py
  python scripts/compare_asv_tables.py --config config.server.dada2_v2.yaml
  python scripts/compare_asv_tables.py --old .../microbiome/asv_table.csv --new .../microbiome_v2/asv_table.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(path: Path) -> pd.Series:
    df = pd.read_csv(path, index_col=0)
    return df.sum(axis=1).astype(int)


def main():
    parser = argparse.ArgumentParser(description="对比 old vs new ASV 表")
    parser.add_argument("--config", default=None, help="含 v2 路径的 yaml；old 取同目录 microbiome/")
    parser.add_argument("--old", default=None)
    parser.add_argument("--new", default=None)
    parser.add_argument("--min-reads", type=int, default=1500)
    args = parser.parse_args()

    if args.config:
        cfg = yaml.safe_load(open(ROOT / args.config, encoding="utf-8"))
        new_path = Path(cfg["paths"]["microbiome_asv"])
        old_path = Path(str(new_path).replace("microbiome_v2", "microbiome").replace("_v2", ""))
        if not old_path.exists():
            old_path = Path("/media/cxhlab/backup/Hospital_Data_Analysis/data/microbiome/asv_table.csv")
    else:
        old_path = Path(args.old or "/media/cxhlab/backup/Hospital_Data_Analysis/data/microbiome/asv_table.csv")
        new_path = Path(args.new or "/media/cxhlab/backup/Hospital_Data_Analysis/data/microbiome_v2/asv_table.csv")

    if not old_path.exists():
        sys.exit(f"找不到 old: {old_path}")
    if not new_path.exists():
        sys.exit(f"找不到 new: {new_path}")

    old = _load(old_path)
    new = _load(new_path)
    ids = sorted(set(old.index) | set(new.index))

    print("=" * 70)
    print(f"OLD: {old_path}")
    print(f"NEW: {new_path}")
    print("=" * 70)
    print(f"{'sample':<8} {'old':>10} {'new':>10} {'delta':>10}")
    print("-" * 42)
    for sid in ids:
        o = int(old.get(sid, 0))
        n = int(new.get(sid, 0))
        if o < 500 and n < 500:
            continue
        print(f"{sid:<8} {o:>10,} {n:>10,} {n - o:>+10,}")

    print("-" * 42)
    print(f"median reads:  {old.median():,.0f}  ->  {new.median():,.0f}")
    print(f">= {args.min_reads}: {(old >= args.min_reads).sum()} / {len(old)}  ->  {(new >= args.min_reads).sum()} / {len(new)}")

    low_old = old[old < 200].sort_values().head(10)
    print("\nOLD 最低 10 例:")
    for sid, val in low_old.items():
        n = int(new.get(sid, 0))
        print(f"  {sid}: {int(val):>6} -> {n:>6}")


if __name__ == "__main__":
    main()
