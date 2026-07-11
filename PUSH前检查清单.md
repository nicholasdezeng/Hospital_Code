# Push 前检查清单

> 服务器路径基准：`/media/cxhlab/backup/`

---

## 一、服务器当前目录（你已确认的）

```
/media/cxhlab/backup/
├── Hospital_Code/                              ← 代码仓（待 push）
└── Hospital_Data_Analysis/
    └── data/                                   ← 数据根目录
        ├── 临床样本收集表1.xlsx
        ├── 临床样本收集表2.xlsx
        ├── 临床样本收集表3.xlsx                ← 分析用这个（含 ASA）
        ├── AICU呼吸道菌群预后研究——完整研究方案与预期结果.docx
        ├── 梁星宝59个原始数据 -260604汇总.zip
        ├── data/                               ← 原始 FASTQ
        │   ├── A1_1.fq, A1_2.fq, ...
        │   ├── 1-65_1.fq, 1-65_2.fq, ...
        │   ├── data_number.xlsx                ← FASTQ名 → 样本编号 对照
        │   └── data_number_sorted.xlsx
        └── microbiome/                         ← DADA2 输出目录（待创建）
            ├── asv_table.csv
            └── taxonomy.csv
```

**代码与数据是分开的两个文件夹**，push 只更新 `Hospital_Code/`，不动 `Hospital_Data_Analysis/`。

---

## 二、Push 前要放进仓库的文件

本地 `Hospital_Code/` 应包含（**不要**提交 `.venv/`、`output/`）：

```
Hospital_Code/
├── .gitignore
├── config.yaml                 # 本地开发用
├── config.server.yaml          # 服务器用（已写好路径）
├── requirements.txt
├── run_analysis.py
├── README.md
├── 分析开发指南.md
├── PUSH前检查清单.md           # 本文件
├── data/microbiome/README.md
└── src/                        # 全部源码
```

---

## 三、Push 前本地自检

在本地 Mac 上执行：

```bash
cd Hospital_Code

# 1. 确认 .gitignore 生效，不会误提交大文件
git status
# 不应出现：.venv/、output/、*.csv（microbiome 演示数据）

# 2. 本地流程能跑通（演示菌群）
source .venv/bin/activate
python run_analysis.py
ls output/run_summary.json

# 3. 确认要提交的文件
git add .
git commit -m "feat: AICU 呼吸道菌群预后分析流水线"
git push
```

---

## 四、Push 后服务器上要做的事

### 4.1 拉代码

```bash
cd /media/cxhlab/backup/Hospital_Code
git pull    # 或首次 clone 到该目录
ls          # 应看到 run_analysis.py、src/、config.server.yaml 等
```

### 4.2 准备数据（代码不管的事，需你手动确认）

| 数据 | 服务器路径 | 状态 |
|------|------------|------|
| 临床 Excel | `.../data/临床样本收集表3.xlsx` | ✅ 已有 |
| 原始 FASTQ | `.../data/data/*.fq` | ✅ 已有 |
| 样本对照表 | `.../data/data/data_number.xlsx` | ✅ 已有 |
| 研究方案 docx | `.../data/AICU呼吸道菌群预后研究——完整研究方案与预期结果.docx` | ✅ 已有 |
| 原始数据 zip | `.../data/梁星宝59个原始数据 -260604汇总.zip` | ✅ 已有（备份） |
| ASV 丰度表 | `.../data/microbiome/asv_table.csv` | ⬜ DADA2 后生成 |
| 物种注释 | `.../data/microbiome/taxonomy.csv` | ⬜ DADA2 后生成 |

```bash
# 创建 DADA2 输出目录
mkdir -p /media/cxhlab/backup/Hospital_Data_Analysis/data/microbiome

# 确认临床表
ls /media/cxhlab/backup/Hospital_Data_Analysis/data/临床样本收集表3.xlsx
```

### 4.3 安装 Python 环境

```bash
cd /media/cxhlab/backup/Hospital_Code

# 方式 A：venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 方式 B：用已有 mamba 环境（你当前是 mambair）
# conda activate zyh  # 或你的环境名
# pip install -r requirements.txt
```

### 4.4 运行分析

```bash
export MPLCONFIGDIR=/media/cxhlab/backup/Hospital_Code/.mplconfig

# ASV 表还没好时，可先测临床部分（会报错或需临时开 demo）
# python run_analysis.py --config config.server.yaml

# ASV 表准备好后正式跑
python run_analysis.py --config config.server.yaml

# 检查结果
cat output/run_summary.json
# 确认 "demo_microbiome": false
```

---

## 五、完整数据分析顺序（只做分析）

```
步骤 1  [已有]  FASTQ 在 data/data/
步骤 2  [待做]  读 data_number.xlsx，弄清 FASTQ 名 → A1/A2 对应关系
步骤 3  [待做]  DADA2：FASTQ → asv_table.csv + taxonomy.csv
步骤 4  [待做]  把 ASV 表样本名改成临床编号（A1、A2…）
步骤 5  [push后] python run_analysis.py --config config.server.yaml
步骤 6  [输出]  output/figures/ 和 output/tables/
```

**当前卡在第 3 步**：有 FASTQ，还没有 ASV 表，`Hospital_Code` 暂时跑不了真实菌群分析。

---

## 六、config 用哪个？

| 环境 | 配置文件 | 说明 |
|------|----------|------|
| 本地 Mac | `config.yaml` | `use_demo_microbiome: true`，无 ASV 也能联调 |
| 服务器 | `config.server.yaml` | 绝对路径，`use_demo_microbiome: false` |

---

## 七、Push 前你还Optional 可做的调整

- [ ] 删除服务器上无用的 `a.md`（或保留无妨）
- [ ] 确认 `Hospital_Code` 是独立 git 仓，remote 已配置
- [ ] 上传临床表到服务器 `Hospital_Data_Analysis/ALL/`
- [ ] 确认服务器是否已有 DADA2 结果（有则跳过生信步骤）

---

## 八、需要你下一步提供的信息

把下面命令在服务器跑完，结果发我，可继续写样本对照 + DADA2 脚本：

```bash
# 1. FASTQ 样本数
ls /media/cxhlab/backup/Hospital_Data_Analysis/data/data/*_1.fq | wc -l

# 2. 对照表结构
python3 -c "
import pandas as pd
df = pd.read_excel('/media/cxhlab/backup/Hospital_Data_Analysis/data/data/data_number.xlsx')
print(df.columns.tolist())
print(df.head())
"

# 3. 是否已有 ASV 结果
find /media/cxhlab/backup/Hospital_Data_Analysis -name '*.tsv' -o -name '*asv*' -o -name '*feature*' 2>/dev/null | head -20

# 4. 临床表
ls -la /media/cxhlab/backup/Hospital_Data_Analysis/data/临床样本收集表3.xlsx
```
