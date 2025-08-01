import anndata as ad
import networkx as nx
import scanpy as sc
import scglue
from matplotlib import rcParams
import numpy as np
import pandas as pd
# from muon import atac as ac

atac = sc.read("/ailab/user/sunjianle-hdd/integration27/mop/muto/ATAC_counts_qc.h5ad")
rna = sc.read("/ailab/user/sunjianle-hdd/integration27/mop/muto/RNA_counts_qc.h5ad")
atac = atac[:, ~atac.var_names.duplicated(keep="first")]
rna = rna[:, ~rna.var_names.duplicated(keep="first")]


sc.tl.pca(rna, n_comps=100, svd_solver="auto")
scglue.data.lsi(atac, n_components=100, n_iter=15)

genelist = rna.var.index[rna.var['highly_variable']==True]
peaklist = atac.var.index[atac.var['highly_variable']==True]
atac = atac[:,peaklist]
rna = rna[:,genelist]

scglue.data.get_gene_annotation(
    rna, gtf="/ailab/user/sunjianle-hdd/integration27/BMMC/gencode.v38.primary_assembly.annotation.gtf",
    gtf_by="gene_name"
)
rna.var.loc[:, ["chrom", "chromStart", "chromEnd"]].head()

rna.var = rna.var.loc[:,~rna.var.columns.duplicated(keep="first")]

atac.var_names = [name.replace("-", ":", 1) for name in atac.var_names]
split = atac.var_names.str.split(r"[:-]")
atac.var["chrom"] = split.map(lambda x: x[0])
atac.var["chromStart"] = split.map(lambda x: x[1]).astype(int)
atac.var["chromEnd"] = split.map(lambda x: x[2]).astype(int)
atac.var.loc[:, ["chrom", "chromStart", "chromEnd"]].head()

rna = rna[:,~rna.var["chrom"].isnull()]
atac = atac[:,~atac.var["chrom"].isnull()]

import os
os.environ['PATH'] += "/ailab/user/sunjianle/.conda/envs/jl_bio/bin"
scglue.config.BEDTOOLS_PATH = "/ailab/user/sunjianle/.conda/envs/jl_bio/bin"
guidance = scglue.genomics.rna_anchored_guidance_graph(rna, atac)
guidance

scglue.graph.check_graph(guidance, [rna, atac])

rna.write("/ailab/user/sunjianle-hdd/integration27/mop/muto/rna-pp.h5ad", compression="gzip")
atac.write("/ailab/user/sunjianle-hdd/integration27/mop/muto/atac-pp.h5ad", compression="gzip")
nx.write_graphml(guidance, "/ailab/user/sunjianle-hdd/integration27/mop/muto/guidance.graphml.gz")

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

rna = ad.read_h5ad("/ailab/user/sunjianle-hdd/integration27/mop/muto/rna-pp.h5ad")
atac = ad.read_h5ad("/ailab/user/sunjianle-hdd/integration27/mop/muto/atac-pp.h5ad")
guidance = nx.read_graphml("/ailab/user/sunjianle-hdd/integration27/mop/muto/guidance.graphml.gz")

scglue.models.configure_dataset(
    rna, "NB", use_highly_variable=True,
    use_layer="counts", 
    use_rep="X_pca",
    use_batch="batch"
)
scglue.models.configure_dataset(
    atac, "NB", use_highly_variable=True,
    use_layer="counts", 
    use_rep="X_lsi",
    use_batch="batch"
)
guidance_hvf = guidance.subgraph(chain(
    rna.var.query("highly_variable").index,
    atac.var.query("highly_variable").index
)).copy()

glue = scglue.models.fit_SCGLUE(
    {"rna": rna, "atac": atac}, guidance_hvf,
    fit_kws={"directory": "glue"}
)

glue.save("/ailab/user/sunjianle-hdd/integration27/mop/muto/glue.dill")

rna.obsm["X_glue"] = glue.encode_data("rna", rna)
atac.obsm["X_glue"] = glue.encode_data("atac", atac)
combined = ad.concat([rna, atac])
np.save("/ailab/user/sunjianle-hdd/integration27/mop/muto/glue.npy",combined.obsm["X_glue"])


