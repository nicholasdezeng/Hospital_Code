# DADA2 生信处理

## 作用

将 `data/data/*.fq` 原始测序数据处理为统计分析所需的：

- `asv_table.csv` — 行 = 临床样本 ID（a1/A1/...），列 = ASV，值 = counts
- `taxonomy.csv` — 行 = ASV ID，列 = Genus/Phylum/...

`sample_manifest.csv` 已在构建时把 FASTQ 前缀映射为临床编号，DADA2 输出直接使用临床 ID，**无需再重命名**。

## 前置条件

```bash
# R 包
R -e "if (!require('BiocManager')) install.packages('BiocManager'); BiocManager::install('dada2')"
R -e "install.packages('yaml')"

# SILVA V3-V4 训练集（示例路径，按服务器实际位置配置）
# 下载: https://benjjneb.github.io/dada2/training.html
```

在 `config.server.yaml` 中配置（可选，无则只出 ASV 不注释）：

```yaml
paths:
  silva_ref: "/path/to/silva138.2_v3v4_train_set.rds"
  silva_species: "/path/to/silva138.2_v3v4_species_assignment.rds"
```

## 运行

```bash
cd /media/cxhlab/backup/Hospital_Code

# 1. 确认样本对照表已生成
python scripts/build_sample_manifest.py --config config.server.yaml
python scripts/check_sample_alignment.py --config config.server.yaml

# 2. 跑 DADA2（耗时较长，建议 screen/tmux）
Rscript scripts/dada2/run_dada2.R --config config.server.yaml

# 3. 统计分析
python run_analysis.py --config config.server.yaml
```

## 参数调整

`run_dada2.R` 中 `truncLen`、`maxEE` 为 V3-V4 常用默认值。若过滤后 reads 过少，可先跑：

```bash
Rscript scripts/dada2/check_read_quality.R --config config.server.yaml
```

（质量预检脚本可按需添加。）

## 输出位置

```
/media/cxhlab/backup/Hospital_Data_Analysis/data/microbiome/
├── sample_manifest.csv    ← 样本对照（已有）
├── asv_table.csv          ← DADA2 输出
├── taxonomy.csv           ← SILVA 注释
└── filtered/              ← 过滤后 FASTQ（中间文件）
```
