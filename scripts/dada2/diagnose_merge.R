#!/usr/bin/env Rscript
# 从已有 filtered FASTQ 诊断 dada + mergePairs（无需 seqtab.rds）
# 用法:
#   Rscript scripts/dada2/diagnose_merge.R --config config.server.yaml
#   Rscript scripts/dada2/diagnose_merge.R --config config.server.yaml --samples A1,a1,A14,a43

suppressPackageStartupMessages({
  library(dada2)
  library(yaml)
})

args <- commandArgs(trailingOnly = TRUE)
config_path <- "config.server.yaml"
sample_arg <- NULL
i <- 1
while (i <= length(args)) {
  if (args[i] == "--config" && i < length(args)) {
    config_path <- args[i + 1]
    i <- i + 2
  } else if (args[i] == "--samples" && i < length(args)) {
    sample_arg <- strsplit(args[i + 1], ",", fixed = TRUE)[[1]]
    i <- i + 2
  } else {
    i <- i + 1
  }
}

cfg <- yaml::read_yaml(config_path)
out_dir <- dirname(cfg$paths$microbiome_asv)
manifest_csv <- cfg$paths$sample_manifest_output
filt_dir <- file.path(out_dir, "filtered")

manifest <- read.csv(manifest_csv, stringsAsFactors = FALSE)
if (is.null(sample_arg)) {
  sample_arg <- c("A1", "A14", "a1", "a43")
}

cat("=== DADA2 merge 诊断（基于 filtered FASTQ）===\n")
cat("filtered 目录:", filt_dir, "\n\n")

rows <- manifest[manifest$clinical_sample_id %in% sample_arg, ]
if (nrow(rows) == 0) stop("manifest 中未找到指定样本")

filtFs <- file.path(filt_dir, paste0(rows$fastq_prefix, "_F_filt.fastq.gz"))
filtRs <- file.path(filt_dir, paste0(rows$fastq_prefix, "_R_filt.fastq.gz"))
names(filtFs) <- rows$clinical_sample_id
names(filtRs) <- rows$clinical_sample_id

for (sid in names(filtFs)) {
  if (!file.exists(filtFs[[sid]]) || !file.exists(filtRs[[sid]])) {
    stop("缺少 filtered 文件: ", sid, " (", rows$fastq_prefix[rows$clinical_sample_id == sid], ")")
  }
}

cat("[1] 学习错误率（仅诊断样本，约 1–3 分钟）...\n")
errF <- learnErrors(filtFs, multithread = TRUE, verbose = FALSE)
errR <- learnErrors(filtRs, multithread = TRUE, verbose = FALSE)

cat("\n[2] 逐样本 dada + merge（默认参数 vs minOverlap=8）\n")
cat(sprintf("%-6s %-10s %10s %10s %10s %10s\n",
            "sample", "fastq", "filt_R1", "merged_def", "merged_lo", "ASV_def"))
cat(paste(rep("-", 72), collapse = ""), "\n")

for (idx in seq_len(nrow(rows))) {
  sid <- rows$clinical_sample_id[idx]
  prefix <- rows$fastq_prefix[idx]
  f1 <- filtFs[[sid]]
  f2 <- filtRs[[sid]]

  derepF <- derepFastq(f1)
  derepR <- derepFastq(f2)
  n_filt <- sum(derepF$uniques)
  ddF <- dada(derepF, err = errF, multithread = FALSE, verbose = FALSE)
  ddR <- dada(derepR, err = errR, multithread = FALSE, verbose = FALSE)

  m_def <- mergePairs(ddF, derepF, ddR, derepR, verbose = FALSE)
  m_lo <- mergePairs(ddF, derepF, ddR, derepR, minOverlap = 8, maxMismatch = 2, verbose = FALSE)

  sum_def <- if (!is.null(m_def)) sum(m_def$abundance) else 0
  sum_lo <- if (!is.null(m_lo)) sum(m_lo$abundance) else 0

  asv_old <- tryCatch({
    old <- read.csv(cfg$paths$microbiome_asv, row.names = 1, check.names = FALSE)
    if (sid %in% rownames(old)) as.integer(sum(old[sid, ])) else NA_integer_
  }, error = function(e) NA_integer_)

  cat(sprintf("%-6s %-10s %10s %10s %10s %10s\n",
              sid, prefix,
              ifelse(is.na(n_filt), "?", format(n_filt, big.mark = ",")),
              format(sum_def, big.mark = ","),
              format(sum_lo, big.mark = ","),
              ifelse(is.na(asv_old), "?", format(asv_old, big.mark = ","))))
}

cat("\n说明:\n")
cat("  filt_R1     = filterAndTrim 后 R1 reads（与 seqkit stats filtered 一致）\n")
cat("  merged_def  = 当前 run_dada2.R 默认 mergePairs 后 reads\n")
cat("  merged_lo   = minOverlap=8, maxMismatch=2 后 reads\n")
cat("  ASV_def     = 现有 asv_table.csv 中该样本总 reads\n")
cat("\n若 filt 很大但 merged 接近 0 → 问题在 merge/dada，应重跑 DADA2 v2（见 config.server.dada2_v2.yaml）\n")
