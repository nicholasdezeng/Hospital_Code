#!/usr/bin/env Rscript
# 仅运行 SILVA 物种注释（无需重跑 DADA2 全流程）
# 需要: seqtab.rds（由 run_dada2.R 生成）或已有 asv_table.csv（列名为 DNA 序列）
#
# 用法:
#   Rscript scripts/dada2/assign_taxonomy.R --config config.server.yaml

suppressPackageStartupMessages({
  library(dada2)
  library(yaml)
})

args <- commandArgs(trailingOnly = TRUE)
config_path <- if (length(args) >= 2 && args[1] == "--config") args[2] else "config.server.yaml"
cfg <- yaml::read_yaml(config_path)

out_dir <- dirname(cfg$paths$microbiome_asv)
out_asv <- cfg$paths$microbiome_asv
out_tax <- cfg$paths$taxonomy
seqtab_rds <- file.path(out_dir, "seqtab.rds")
silva_ref <- cfg$paths$silva_ref
silva_species <- if (!is.null(cfg$paths$silva_species)) cfg$paths$silva_species else sub("train_set", "species_assignment", silva_ref)

if (!file.exists(silva_ref)) {
  stop("SILVA 训练集不存在: ", silva_ref, "\n请先下载并配置 paths.silva_ref")
}

cat("=== SILVA 物种注释 ===\n")

if (file.exists(seqtab_rds)) {
  cat("读取 seqtab.rds:", seqtab_rds, "\n")
  seqtab <- readRDS(seqtab_rds)
} else if (file.exists(out_asv)) {
  cat("读取 asv_table.csv:", out_asv, "\n")
  asv_df <- read.csv(out_asv, row.names = 1, check.names = FALSE)
  seqtab <- as.matrix(asv_df)
  storage.mode(seqtab) <- "integer"
} else {
  stop("未找到 seqtab.rds 或 asv_table.csv，请先运行 run_dada2.R")
}

cat("ASV 表维度:", dim(seqtab), "\n")
cat("SILVA:", silva_ref, "\n")

taxa <- assignTaxonomy(seqtab, silva_ref, multithread = TRUE, minBoot = 50)
if (file.exists(silva_species)) {
  taxa <- addSpecies(taxa, silva_species)
}

asv_ids <- paste0("ASV_", seq_len(ncol(seqtab)))
colnames(seqtab) <- asv_ids

write.csv(as.data.frame(seqtab), out_asv, row.names = TRUE)

tax_df <- as.data.frame(taxa, stringsAsFactors = FALSE)
rownames(tax_df) <- asv_ids
write.csv(tax_df, out_tax, row.names = TRUE)

cat("\n✅ 注释完成\n")
cat("  ASV:", out_asv, "\n")
cat("  Taxonomy:", out_tax, "\n")
