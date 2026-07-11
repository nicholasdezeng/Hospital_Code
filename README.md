# AICU 呼吸道菌群预后研究 — 分析代码

基于 `ALL/AICU呼吸道菌群预后研究——完整研究方案与预期结果.docx` 实现的完整分析流水线。

> **服务器部署与后续开发**：请参阅 [`分析开发指南.md`](./分析开发指南.md)

## 快速开始

```bash
cd Hospital_Code
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_analysis.py
```

## 项目结构

```
Hospital_Code/
├── config.yaml              # 分析参数配置
├── run_analysis.py          # 主入口
├── requirements.txt
├── data/microbiome/         # 16S ASV 数据（真实数据放这里）
├── output/
│   ├── figures/             # Figure 1-9
│   └── tables/              # Table 1-2
└── src/
    ├── data_loader.py       # 临床/菌群数据加载
    ├── preprocessing.py     # 分组与演示数据生成
    └── analysis/            # 各层分析模块
```

## 三层分析架构

| 层级 | 内容 | 输出 |
|------|------|------|
| 第一层 | 基线特征 + 菌群全景 + α/β 多样性 + LEfSe | Table 1, Figure 1-4 |
| 第二层 | 菌群-炎症相关 + 四象限分层 + 中介分析 | Figure 5-7 |
| 第三层 | 三组预测模型 + 关键菌群鉴定 | Figure 8-9, Table 2 |

## 数据说明

### 临床数据
默认读取 `../ALL/临床样本收集表3.xlsx`（含 ASA 分级，最完整）。

### 菌群数据
将 DADA2 输出的 ASV 丰度表和分类注释放入：

- `data/microbiome/asv_table.csv` — 行=样本编号，列=ASV，值=counts
- `data/microbiome/taxonomy.csv` — 行=ASV ID，列=Genus/Phylum 等

若尚无真实测序数据，`config.yaml` 中 `use_demo_microbiome: true` 会自动生成与临床结局相关的演示数据用于流程联调。**正式分析请务必替换为真实 ASV 数据并设置 `use_demo_microbiome: false`。**

## 配置项

在 `config.yaml` 中可调整：

- `grouping.extubation_split`: 拔管分组方式（median / fixed）
- `grouping.crp_threshold`: 四象限分析 CRP 阈值（默认 10 mg/L）
- `analysis.permutations`: PERMANOVA 置换次数
- `analysis.bootstrap_n`: 中介分析 Bootstrap 次数

## 输出清单

- `table1_baseline.csv/xlsx` — 基线特征表
- `figure1_microbiome_overview.png` — 菌群组成全景
- `figure2_alpha_diversity.png` — α 多样性
- `figure3_beta_diversity.png` — β 多样性 PCoA
- `figure4_differential_microbiota.png` — LEfSe 差异分析
- `figure5_inflammation_correlation.png` — 菌群-炎症相关
- `figure6_quadrant_analysis.png` — 四象限分层
- `figure7_mediation.png` — 中介分析
- `figure8_prediction_roc.png` — ROC 对比
- `table2_model_performance.csv/xlsx` — 模型性能
- `figure9_key_biomarkers.png` — 关键预后菌群
