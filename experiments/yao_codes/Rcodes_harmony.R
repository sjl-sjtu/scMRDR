library(Seurat)
library(MuDataSeurat)
library(tidyverse)
library(patchwork)
library(harmony)
library(EnsDb.Hsapiens.v86)
library(Signac)

# harmony
rna.so <- ReadH5AD("/ailab/user/sunjianle-hdd/integration27/mop/Yao/RNA_counts_qc.h5ad")
atac.so <- ReadH5AD("/ailab/user/sunjianle-hdd/integration27/mop/Yao/ATAC_counts_qc.h5ad")
atac.so <- RenameAssays(atac.so, RNA = "ATAC")
counts <- GetAssayData(atac.so, assay = "ATAC", slot = "counts")
chrom_assay <- CreateChromatinAssay(counts = counts,sep = c(":", "-"),)
atac.so[["ATAC"]] <- chrom_assay
DefaultAssay(atac.so) <- "ATAC"

library(future)
library(future.apply)
library(parallel)
library(pbapply)
options(future.globals.maxSize = 10 * 1024^3) 
CreateGeneActivityMatrix <- function(
  peak.matrix,
  annotation.file,
  seq.levels = c(1:22, "X", "Y"),
  include.body = TRUE,
  upstream = 2000,
  downstream = 0,
  verbose = TRUE
) {
  if (!PackageCheck('GenomicRanges', error = FALSE)) {
    stop("Please install GenomicRanges from Bioconductor.")
  }
  if (!PackageCheck('rtracklayer', error = FALSE)) {
    stop("Please install rtracklayer from Bioconductor.")
  }

  # convert peak matrix to GRanges object
  peak.df <- rownames(x = peak.matrix)
  peak.df <- do.call(what = rbind, args = strsplit(x = gsub(peak.df, pattern = ":", replacement = "-"), split = "-"))
  peak.df <- as.data.frame(x = peak.df)
  colnames(x = peak.df) <- c("chromosome", 'start', 'end')
  peaks.gr <- GenomicRanges::makeGRangesFromDataFrame(df = peak.df)

  # get annotation file, select genes
  gtf <- rtracklayer::import(con = annotation.file)
  gtf <- GenomeInfoDb::keepSeqlevels(x = gtf, value = seq.levels, pruning.mode = 'coarse')
  GenomeInfoDb::seqlevelsStyle(gtf) <- "UCSC"
  gtf.genes <- gtf[gtf$type == 'gene']

  # Extend definition up/downstream
  if (include.body) {
    gtf.body_prom <- Extend(x = gtf.genes, upstream = upstream, downstream = downstream)
  } else {
    gtf.body_prom <- SummarizedExperiment::promoters(x = gtf.genes, upstream = upstream, downstream = downstream)
  }
  gene.distances <- GenomicRanges::distanceToNearest(x = peaks.gr, subject = gtf.body_prom)
  keep.overlaps <- gene.distances[rtracklayer::mcols(x = gene.distances)$distance == 0]
  peak.ids <- peaks.gr[S4Vectors::queryHits(x = keep.overlaps)]
  gene.ids <- gtf.genes[S4Vectors::subjectHits(x = keep.overlaps)]
  peak.ids$gene.name <- gene.ids$gene_name
  peak.ids <- as.data.frame(x = peak.ids)
  peak.ids$peak <- paste0(peak.ids$seqnames, ":", peak.ids$start, "-", peak.ids$end)
  annotations <- peak.ids[, c('peak', 'gene.name')]
  colnames(x = annotations) <- c('feature', 'new_feature')

  # collapse into expression matrix
  peak.matrix <- as(object = peak.matrix, Class = 'matrix')
  all.features <- unique(x = annotations$new_feature)

  if (nbrOfWorkers() > 1) {
    mysapply <- future_sapply
  } else {
    mysapply <- ifelse(test = verbose, yes = pbsapply, no = sapply)
  }
  newmat <- mysapply(1:length(x = all.features), FUN = function(x){
    features.use <- annotations[annotations$new_feature == all.features[[x]], ]$feature
    # features.use <- intersect(features.use, rownames(peak.matrix))
    submat <- peak.matrix[features.use, ]
    if (length(x = features.use) > 1) {
      return(Matrix::colSums(x = submat))
    } else {
      return(submat)
    }
  })
  newmat <- t(x = newmat)
  rownames(x = newmat) <- all.features
  colnames(x = newmat) <- colnames(x = peak.matrix)
  return(as(object = newmat, Class = 'dgCMatrix'))
}

mt <- atac.so$ATAC@counts
if(length(strsplit(rownames(mt)[1],"-")[[1]])==3){
    rownames(mt) <- sub("-", ":", rownames(mt))
}
gene.activities <- CreateGeneActivityMatrix(peak.matrix = mt, 
    annotation.file = "/ailab/user/sunjianle-hdd/integration27/BMMC/gencode.vM10.chr_patch_hapl_scaff.annotation.gtf", 
    seq.levels = paste0("chr",c(1:19, "X", "Y")),upstream = 2000, verbose = TRUE)

atac.so[["RNA"]] <- CreateAssayObject(counts = gene.activities)
DefaultAssay(atac.so) <- "RNA"

rna.so$domain <- c(rep("scRNA-seq"))
atac.so$domain <- c(rep("scATAC-seq"))

combined.so <- merge(rna.so, atac.so)
combined.so <- NormalizeData(combined.so)
# VariableFeatures(combined.so) <- hvg
combined.so <- FindVariableFeatures(combined.so, selection.method = "vst", verbose = FALSE)
combined.so <- ScaleData(combined.so)
# combined.so <- SCTransform(combined.so)
combined.so <- RunPCA(combined.so, npcs = 50, seed.use = 0, verbose = FALSE)
combined.so

combined.so_harmony <- RunHarmony(combined.so, plot_convergence = TRUE, 
                  group.by.vars = "domain", max.iter.harmony = 20, lambda = 0.1, epsilon.harmony=-Inf)
Embeddings(combined.so_harmony, reduction='harmony') |> as.data.frame() |> write.csv("/ailab/user/sunjianle-hdd/integration27/mop/Yao/harmony.csv")
saveRDS(combined.so_harmony,"/ailab/user/sunjianle-hdd/integration27/mop/Yao/harmony_obj.rds")