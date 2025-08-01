import anndata as ad
import networkx as nx
import scanpy as sc
import scglue
from matplotlib import rcParams
import numpy as np
import pandas as pd
# from muon import atac as ac

rna = sc.read("/ailab/user/sunjianle-hdd/integration27/mop/BMMC/RNA_counts_qc_sampled.h5ad")
atac = sc.read("/ailab/user/sunjianle-hdd/integration27/mop/BMMC/ATAC_counts_qc_sampled.h5ad")
prot = sc.read("/ailab/user/sunjianle-hdd/integration27/mop/BMMC/protein_counts_qc_sampled.h5ad")

atac = atac[:, ~atac.var_names.duplicated(keep="first")]
rna = rna[:, ~rna.var_names.duplicated(keep="first")]
prot = prot[:, ~prot.var_names.duplicated(keep="first")]

# atac.X = atac.layers['counts'].copy()
# rna.X = rna.layers['counts'].copy()
# prot.X = prot.layers['counts'].copy()
# sc.pp.normalize_total(rna)
# sc.pp.log1p(rna)
# sc.pp.highly_variable_genes(rna, flavor="seurat_v3", batch_key="batch")
# sc.pp.highly_variable_genes(atac, min_mean=0.05, max_mean=1.5, min_disp=.5, batch_key="batch")
# sc.pp.scale(rna)
sc.tl.pca(rna, n_comps=100, svd_solver="auto")
sc.tl.pca(prot, n_comps=100, svd_solver="auto")
scglue.data.lsi(atac, n_components=100, n_iter=15)

genelist = rna.var.index[rna.var['highly_variable']==True]
peaklist = atac.var.index[atac.var['highly_variable']==True]
# protlist = prot.var.index[prot.var['highly_variable']==True]
atac = atac[:,peaklist]
rna = rna[:,genelist]
# prot = prot[:,protlist]

scglue.data.get_gene_annotation(
    rna, gtf="/ailab/user/sunjianle-hdd/integration27/BMMC/gencode.v38.chr_patch_hapl_scaff.annotation.gtf",
    gtf_by="gene_name"
)

scglue.data.get_gene_annotation(
    prot, gtf="/ailab/user/sunjianle-hdd/integration27/BMMC/gencode.v38.chr_patch_hapl_scaff.annotation.gtf",
    gtf_by="gene_name"
)


atac.var_names = [name.replace("-", ":", 1) for name in atac.var_names]
split = atac.var_names.str.split(r"[:-]")
atac.var["chrom"] = split.map(lambda x: x[0])
atac.var["chromStart"] = split.map(lambda x: x[1]).astype(int)
atac.var["chromEnd"] = split.map(lambda x: x[2]).astype(int)

rna.var.loc[:, ["chrom", "chromStart", "chromEnd"]].head()
atac.var.loc[:, ["chrom", "chromStart", "chromEnd"]].head()
prot.var.loc[:, ["chrom", "chromStart", "chromEnd"]].head()

rna.var = rna.var.loc[:,~rna.var.columns.duplicated(keep="first")]
atac.var = atac.var.loc[:,~atac.var.columns.duplicated(keep="first")]
prot.var = prot.var.loc[:,~prot.var.columns.duplicated(keep="first")]

rna = rna[:,~rna.var[["chrom", 'chromStart', 'chromEnd']].isna().any(axis=1)]
atac = atac[:,~atac.var[["chrom", 'chromStart', 'chromEnd']].isna().any(axis=1)]
prot = prot[:,~prot.var[["chrom", 'chromStart', 'chromEnd']].isna().any(axis=1)]

import os
os.environ['PATH'] += "/ailab/user/sunjianle/.conda/envs/jl_bio/bin"
scglue.config.BEDTOOLS_PATH = "/ailab/user/sunjianle/.conda/envs/jl_bio/bin"
guidance = scglue.genomics.rna_anchored_guidance_graph(rna, atac, prot)
guidance

scglue.graph.check_graph(guidance, [rna, atac, prot])

rna.write("/ailab/user/sunjianle-hdd/integration27/mop/BMMC/rna-pp_sampled.h5ad", compression="gzip")
atac.write("/ailab/user/sunjianle-hdd/integration27/mop/BMMC/atac-pp_sampled.h5ad", compression="gzip")
prot.write("/ailab/user/sunjianle-hdd/integration27/mop/BMMC/prot-pp_sampled.h5ad", compression="gzip")

nx.write_graphml(guidance, "/ailab/user/sunjianle-hdd/integration27/mop/BMMC/guidance_sampled.graphml.gz")

from itertools import chain

import anndata as ad
import itertools
import networkx as nx
import pandas as pd
import scanpy as sc
import scglue
import seaborn as sns
from matplotlib import rcParams

scglue.plot.set_publication_params()
rcParams["figure.figsize"] = (4, 4)

rna = ad.read_h5ad("/ailab/user/sunjianle-hdd/integration27/mop/BMMC/rna-pp_sampled.h5ad")
atac = ad.read_h5ad("/ailab/user/sunjianle-hdd/integration27/mop/BMMC/atac-pp_sampled.h5ad")
prot = ad.read_h5ad("/ailab/user/sunjianle-hdd/integration27/mop/BMMC/prot-pp_sampled.h5ad")
guidance = nx.read_graphml("/ailab/user/sunjianle-hdd/integration27/mop/BMMC/guidance_sampled.graphml.gz")

scglue.models.configure_dataset(
    rna, "NB", use_highly_variable=True,
    use_layer="counts", 
    use_rep="X_pca"#,
    # use_batch="batch"
)
scglue.models.configure_dataset(
    atac, "NB", use_highly_variable=True,
    use_layer="counts", 
    use_rep="X_lsi"#,
    # use_batch="batch"
)
scglue.models.configure_dataset(
    prot, "NB", use_highly_variable=False,
    use_layer="counts", 
    use_rep="X_pca"#,
    # use_batch="batch"
)
guidance_hvf = guidance.subgraph(chain(
    rna.var.query("highly_variable").index,
    atac.var.query("highly_variable").index,
    prot.var.index #.query("highly_variable")
)).copy()

glue = scglue.models.fit_SCGLUE(
    {"rna": rna, "atac": atac, "prot": prot}, guidance_hvf,
    fit_kws={"directory": "glue"}
)

glue.save("/ailab/user/sunjianle-hdd/integration27/mop/BMMC/glue_sampled.dill")

rna.obsm["X_glue"] = glue.encode_data("rna", rna)
atac.obsm["X_glue"] = glue.encode_data("atac", atac)
prot.obsm["X_glue"] = glue.encode_data("prot", prot)
combined = ad.concat([rna, atac, prot])
np.save("/ailab/user/sunjianle-hdd/integration27/mop/BMMC/glue_sampled.npy",combined.obsm["X_glue"])

