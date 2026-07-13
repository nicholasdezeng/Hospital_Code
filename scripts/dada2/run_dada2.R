#!/usr/bin/env Rscript
# AICU 16S (V3-V4) DADA2 流程
# 输入: sample_manifest.csv + FASTQ 目录
# 输出: asv_table.csv (行=临床样本ID, 列=ASV) + taxonomy.csv
#
# 用法:
#   Rscript scripts/dada2/run_dada2.R --config config.server.yaml
#
# 依赖: R packages dada2, Biostrings
# SILVA: 下载 Zenodo 14169026 的 .fa.gz 文件（见 scripts/dada2/README.md）

suppressPackageStartupMessages({
  if (!requireNamespace("dada2", quietly = TRUE)) {
    stop("请先安装 dada2: BiocManager::install('dada2')")
  }
  if (!requireNamespace("yaml", quietly = TRUE)) {
    stop("请先安装 yaml: install.packages('yaml')")
  }
  library(dada2)
  library(yaml)
})

args <- commandArgs(trailingOnly = TRUE)
config_path <- if (length(args) >= 2 && args[1] == "--config") args[2] else "config.server.yaml"

cfg <- yaml::read_yaml(config_path)
fastq_dir <- cfg$paths$raw_fastq_dir
manifest_csv <- if (!is.null(cfg$paths$sample_manifest_output)) {
  cfg$paths$sample_manifest_output
} else {
  file.path(dirname(cfg$paths$microbiome_asv), "sample_manifest.csv")
}
out_asv <- cfg$paths$microbiome_asv
out_tax <- cfg$paths$taxonomy
out_dir <- dirname(out_asv)

silva_ref <- if (!is.null(cfg$paths$silva_ref)) cfg$paths$silva_ref else "/media/cxhlab/backup/databases/silva/silva_nr99_v138.2_toGenus_trainset.fa.gz"
silva_species <- if (!is.null(cfg$paths$silva_species)) cfg$paths$silva_species else "/media/cxhlab/backup/databases/silva/silva_v138.2_assignSpecies.fa.gz"

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

cat("=== DADA2 Pipeline ===\n")
cat("FASTQ dir:", fastq_dir, "\n")
cat("Manifest:", manifest_csv, "\n")

manifest <- read.csv(manifest_csv, stringsAsFactors = FALSE)
manifest <- manifest[order(manifest$clinical_sample_id), ]

fnFs <- file.path(fastq_dir, paste0(manifest$fastq_prefix, "_1.fq"))
fnRs <- file.path(fastq_dir, paste0(manifest$fastq_prefix, "_2.fq"))
sample_ids <- manifest$clinical_sample_id

if (!all(file.exists(fnFs)) || !all(file.exists(fnRs))) {
  missing <- manifest$fastq_prefix[!file.exists(fnFs) | !file.exists(fnRs)]
  stop("缺少 FASTQ: ", paste(missing, collapse = ", "))
}

cat("样本数:", nrow(manifest), "\n")

`%||%` <- function(x, y) if (is.null(x)) y else x

dada2_cfg <- cfg$dada2 %||% list()
truncLen <- unlist(dada2_cfg$truncLen %||% c(240, 200))
maxEE <- unlist(dada2_cfg$maxEE %||% c(2, 2))
minOverlap <- as.integer(dada2_cfg$minOverlap %||% 12L)
maxMismatch <- as.integer(dada2_cfg$maxMismatch %||% 0L)
minBoot <- as.integer(dada2_cfg$minBoot %||% 50L)

cat("DADA2 参数: truncLen=", paste(truncLen, collapse = ","),
    " maxEE=", paste(maxEE, collapse = ","),
    " minOverlap=", minOverlap, " maxMismatch=", maxMismatch, "\n", sep = "")

cat("\n[1/6] 质量过滤...\n")
filt_dir <- file.path(out_dir, "filtered")
dir.create(filt_dir, showWarnings = FALSE)
filtFs <- file.path(filt_dir, paste0(manifest$fastq_prefix, "_F_filt.fastq.gz"))
filtRs <- file.path(filt_dir, paste0(manifest$fastq_prefix, "_R_filt.fastq.gz"))

out <- filterAndTrim(
  fnFs, filtFs, fnRs, filtRs,
  truncLen = truncLen, maxN = 0, maxEE = maxEE,
  truncQ = 2, rm.phix = TRUE, compress = TRUE, multithread = TRUE
)
cat("过滤后 reads:\n")
print(out)

cat("\n[2/6] 学习测序错误...\n")
errF <- learnErrors(filtFs, multithread = TRUE)
errR <- learnErrors(filtRs, multithread = TRUE)

cat("\n[3/6] 去重...\n")
derepFs <- derepFastq(filtFs)
derepRs <- derepFastq(filtRs)
names(derepFs) <- sample_ids
names(derepRs) <- sample_ids

cat("\n[4/6] DADA2 推断 ASV...\n")
dadaFs <- dada(derepFs, err = errF, multithread = TRUE)
dadaRs <- dada(derepRs, err = errR, multithread = TRUE)

cat("\n[5/6] 合并双端...\n")
mergers <- mergePairs(
  dadaFs, derepFs, dadaRs, derepRs,
  minOverlap = minOverlap,
  maxMismatch = maxMismatch,
  verbose = TRUE
)
seqtab <- makeSequenceTable(mergers)
rownames(seqtab) <- sample_ids
cat("ASV 表维度:", dim(seqtab), "\n")
cat("各样本 reads 摘要 (min/median/max):\n")
rs <- rowSums(seqtab)
print(summary(rs))
read_track <- data.frame(
  sample_id = sample_ids,
  raw_reads = out[, 1],
  filt_reads = out[, 2],
  merged_reads = as.integer(rs[sample_ids])
)
write.csv(read_track, file.path(out_dir, "dada2_read_tracking.csv"), row.names = FALSE)
cat("  已保存:", file.path(out_dir, "dada2_read_tracking.csv"), "\n")

cat("\n[6/6] SILVA 物种注释...\n")
seqtab_rds <- file.path(out_dir, "seqtab.rds")
saveRDS(seqtab, seqtab_rds)
cat("  已保存中间文件:", seqtab_rds, "\n")

if (!file.exists(silva_ref)) {
  cat("⚠ 未找到 SILVA 训练集:", silva_ref, "\n")
  cat("  跳过注释，仅输出 ASV 表。\n")
  cat("  下载 SILVA 后运行: Rscript scripts/dada2/assign_taxonomy.R --config config.server.yaml\n")
  asv_df <- as.data.frame(seqtab)
  write.csv(asv_df, out_asv, row.names = TRUE)
  cat("已写入:", out_asv, "\n")
  quit(save = "no", status = 0)
}

taxa <- assignTaxonomy(seqtab, silva_ref, multithread = TRUE, minBoot = minBoot)
if (file.exists(silva_species)) {
  taxa <- addSpecies(taxa, silva_species)
}
asv_ids <- paste0("ASV_", seq_len(ncol(seqtab)))
colnames(seqtab) <- asv_ids
asv_df <- as.data.frame(seqtab)
write.csv(asv_df, out_asv, row.names = TRUE)

# 输出 taxonomy（兼容下游 Python 分析）
tax_df <- as.data.frame(taxa, stringsAsFactors = FALSE)
tax_df$Genus <- tax_df$Genus
tax_df$Phylum <- tax_df$Phylum
tax_df$Family <- tax_df$Family
tax_df$Species <- tax_df$Species
rownames(tax_df) <- asv_ids
write.csv(tax_df, out_tax, row.names = TRUE)

cat("\n✅ 完成\n")
cat("  ASV:", out_asv, "\n")
cat("  Taxonomy:", out_tax, "\n")
cat("  样本 ID 已使用临床编号 (a1/A1/...)\n")
