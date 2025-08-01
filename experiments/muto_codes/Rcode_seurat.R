library(Seurat)
library(MuDataSeurat)
library(tidyverse)
library(patchwork)
library(harmony)
library(EnsDb.Hsapiens.v86)
library(Signac)

seurat.rna <- ReadH5AD("/ailab/user/sunjianle-hdd/integration27/mop/muto/RNA_counts_qc.h5ad")
seurat.atac <- ReadH5AD("/ailab/user/sunjianle-hdd/integration27/mop/muto/ATAC_counts_qc.h5ad")

seurat.atac <- RenameAssays(seurat.atac, RNA = "ATAC")
counts <- GetAssayData(seurat.atac, assay = "ATAC", slot = "counts")
chrom_assay <- CreateChromatinAssay(counts = counts,sep = c(":", "-"),)
seurat.atac[["ATAC"]] <- chrom_assay
DefaultAssay(seurat.atac) <- "ATAC"

seurat.rna <- NormalizeData(seurat.rna)
seurat.rna <- FindVariableFeatures(seurat.rna, selection.method = "vst", nfeatures = 3000)
seurat.rna <- ScaleData(seurat.rna)
seurat.rna <- RunPCA(seurat.rna)
# seurat.rna <- IntegrateLayers(object = seurat.rna, method = CCAIntegration, orig.reduction = "pca", 
#           new.reduction = "integrated.cca", features = "batch", verbose = FALSE)
seurat.rna <- RunUMAP(seurat.rna, dims = 1:30, reduction = "pca")

annotations <- GetGRangesFromEnsDb(ensdb = EnsDb.Hsapiens.v86)
seqlevelsStyle(annotations) <- "NCBI" #"UCSC" #
genome(annotations) <- "hg38"
Annotation(seurat.atac) <- annotations

seurat.atac <- RunTFIDF(seurat.atac)
seurat.atac <- FindTopFeatures(seurat.atac, min.cutoff = "q0")
seurat.atac <- RunSVD(seurat.atac)
# seurat.atac <- IntegrateLayers(object = seurat.atac, method = CCAIntegration, orig.reduction = "lsi", 
#           new.reduction = "integrated.cca", features = "batch", verbose = FALSE)
seurat.atac <- RunUMAP(seurat.atac, reduction = "lsi", dims = 2:30, 
              reduction.name = "umap.atac", reduction.key = "atacUMAP_")

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

mt <- seurat.atac$ATAC@counts
if(length(strsplit(rownames(mt)[1],"-")[[1]])==3){
    rownames(mt) <- sub("-", ":", rownames(mt))
}
gene.activities <- CreateGeneActivityMatrix(peak.matrix = mt, 
    annotation.file = "/ailab/user/sunjianle-hdd/integration27/BMMC/gencode.v38.primary_assembly.annotation.gtf", 
    seq.levels = paste0("chr",c(1:22, "X", "Y")),upstream = 2000, verbose = TRUE)

seurat.atac[["ACTIVITY"]] <- CreateAssayObject(counts = gene.activities)

# normalize gene activities
DefaultAssay(seurat.atac) <- "ACTIVITY"
seurat.atac <- NormalizeData(seurat.atac)
seurat.atac <- ScaleData(seurat.atac, features = rownames(seurat.atac))

# Identify anchors
genes.use <- intersect(VariableFeatures(seurat.rna),rownames(seurat.atac[["ACTIVITY"]]))
transfer.anchors <- FindTransferAnchors(reference = seurat.rna, query = seurat.atac, features = genes.use,
                                        reference.assay = "RNA", query.assay = "ACTIVITY", reduction = "cca")

# note that we restrict the imputation to variable genes from scRNA-seq, but could impute the
# full transcriptome if we wanted to
refdata <- GetAssayData(seurat.rna, assay = "RNA", slot = "data")[genes.use, ]

# refdata (input) contains a scRNA-seq expression matrix for the scRNA-seq cells.  imputation
# (output) will contain an imputed scRNA-seq matrix for each of the ATAC cells
imputation <- TransferData(anchorset = transfer.anchors, refdata = refdata, weight.reduction = seurat.atac[["lsi"]],
                           dims = 2:30)
seurat.atac[["RNA"]] <- imputation

coembed <- merge(x = seurat.rna, y = seurat.atac)

# Finally, we run PCA and UMAP on this combined object, to visualize the co-embedding of both
# datasets
coembed <- FindVariableFeatures(coembed, selection.method = "vst", nfeatures = 3000)
coembed <- ScaleData(coembed, features = genes.use, do.scale = FALSE)
coembed <- RunPCA(coembed, features = genes.use, verbose = FALSE)
# print(coembed)
# print(coembed[['RNA']])
# coembed <- IntegrateLayers(object = coembed, method = CCAIntegration, orig.reduction = "pca", 
#           assay = "RNA", new.reduction = "integrated.cca", features = "batch", verbose = FALSE)
# Embeddings(coembed, reduction = "integrated.cca") |> as.data.frame() |> write.csv("/ailab/user/sunjianle-hdd/integration27/mop/muto/seurat_pca.csv")


# coembed <- RunUMAP(coembed, dims = 1:30)

saveRDS(coembed,"/ailab/user/sunjianle-hdd/integration27/mop/muto/seurat_obj.rds")
Embeddings(coembed, reduction = "pca") |> as.data.frame() |> write.csv("/ailab/user/sunjianle-hdd/integration27/mop/muto/seurat_pca.csv")



