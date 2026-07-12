# DADA2 生信处理

## SILVA 数据库下载（重要：旧 .rds 链接已失效）

SILVA 138.2 官方 DADA2 格式已改为 **`.fa.gz`**，来源：

- 官方页面：https://benjjneb.github.io/dada2/training.html
- Zenodo：https://zenodo.org/records/14169026

```bash
mkdir -p /media/cxhlab/backup/databases/silva
cd /media/cxhlab/backup/databases/silva

# 属级注释（必须，约 140 MB）
wget -O silva_nr99_v138.2_toGenus_trainset.fa.gz \
  "https://zenodo.org/records/14169026/files/silva_nr99_v138.2_toGenus_trainset.fa.gz?download=1"

# 种级注释（可选，约 70 MB）
wget -O silva_v138.2_assignSpecies.fa.gz \
  "https://zenodo.org/records/14169026/files/silva_v138.2_assignSpecies.fa.gz?download=1"
```

`config.server.yaml` 中对应路径：

```yaml
silva_ref: "/media/cxhlab/backup/databases/silva/silva_nr99_v138.2_toGenus_trainset.fa.gz"
silva_species: "/media/cxhlab/backup/databases/silva/silva_v138.2_assignSpecies.fa.gz"
```

## 运行流程

```bash
cd /media/cxhlab/backup/Hospital_Code

# 1. 样本对照（已完成可跳过）
python scripts/build_sample_manifest.py --config config.server.yaml

# 2. DADA2（若 asv_table.csv 已有可跳过）
Rscript scripts/dada2/run_dada2.R --config config.server.yaml

# 3. 仅物种注释（推荐：已有 ASV 表时只跑这步）
Rscript scripts/dada2/assign_taxonomy.R --config config.server.yaml

# 4. 统计分析
python run_analysis.py --config config.server.yaml
```

## 输出文件

```
.../data/microbiome/
├── sample_manifest.csv
├── asv_table.csv
├── taxonomy.csv
├── seqtab.rds          # 中间文件，供 assign_taxonomy.R 使用
└── filtered/           # 过滤后 FASTQ
```
