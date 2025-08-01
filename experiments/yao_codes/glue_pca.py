import anndata as ad
import networkx as nx
import scanpy as sc
import scglue
from matplotlib import rcParams
import numpy as np
import pandas as pd
# from muon import atac as ac

atac = sc.read("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao/ATAC_counts_qc.h5ad")
rna = sc.read("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao/RNA_counts_qc.h5ad")
atac = atac[:, ~atac.var_names.duplicated(keep="first")]
rna = rna[:, ~rna.var_names.duplicated(keep="first")]
print(rna)
print(atac)

sc.tl.pca(rna, n_comps=100, svd_solver="auto")
sc.tl.pca(atac, n_comps=100,svd_solver="auto") # 
# scglue.data.lsi(atac, n_components=100, n_iter=15) #
from sklearn.decomposition import TruncatedSVD
lsi = TruncatedSVD(n_components=100)
atac.obsm['X_lsi'] = lsi.fit_transform(atac.X.toarray())

genelist = rna.var.index[rna.var['highly_variable']==True]
peaklist = atac.var.index[atac.var['highly_variable']==True]
print(len(genelist))
print(len(peaklist))
atac = atac[:,peaklist]
rna = rna[:,genelist]

# scglue.data.get_gene_annotation(
#     rna, gtf="/ailab/user/sunjianle-hdd/integration27/BMMC/gencode.vM10.chr_patch_hapl_scaff.annotation.gtf",
#     gtf_by="gene_name"
# )
rna.var.loc[:, ["chrom", "chromStart", "chromEnd"]].head()

rna.var = rna.var.loc[:,~rna.var.columns.duplicated(keep="first")]

# atac.var_names = [name.replace("-", ":", 1) for name in atac.var_names]
# split = atac.var_names.str.split(r"[:-]")
# atac.var["chrom"] = split.map(lambda x: x[0])
# atac.var["chromStart"] = split.map(lambda x: x[1]).astype(int)
# atac.var["chromEnd"] = split.map(lambda x: x[2]).astype(int)
atac.var.loc[:, ["chrom", "chromStart", "chromEnd"]].head()

rna = rna[:,~rna.var["chrom"].isnull()]
atac = atac[:,~atac.var["chrom"].isnull()]

import os
os.environ['PATH'] += "/home/bingxing2/ailab/scxlab0179/miniconda3/envs/scglue/bin"
scglue.config.BEDTOOLS_PATH = "/home/bingxing2/ailab/scxlab0179/miniconda3/envs/scglue/bin"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

guidance = scglue.genomics.rna_anchored_guidance_graph(rna, atac)
guidance

scglue.graph.check_graph(guidance, [rna, atac])

rna.write("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao/rna-pp.h5ad", compression="gzip")
atac.write("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao/atac-pp.h5ad", compression="gzip")
nx.write_graphml(guidance, "/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao/guidance.graphml.gz")

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

rna = ad.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao/rna-pp.h5ad")
atac = ad.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao/atac-pp.h5ad")
guidance = nx.read_graphml("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao/guidance.graphml.gz")

scglue.models.configure_dataset(
    rna, "NB", use_highly_variable=True,
    use_layer="counts", 
    use_rep="X_pca",
    use_batch="batch"
)
scglue.models.configure_dataset(
    atac, "NB", use_highly_variable=True,
    use_layer="counts", 
    use_rep= "X_pca", #"X_lsi", #
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

glue.save("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao/glue.dill")

rna.obsm["X_glue"] = glue.encode_data("rna", rna)
atac.obsm["X_glue"] = glue.encode_data("atac", atac)
combined = ad.concat([rna, atac])
np.save("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao/glue_nb_hvg_pca.npy",combined.obsm["X_glue"])

##################
scglue.plot.set_publication_params()
rcParams["figure.figsize"] = (4, 4)

rna = ad.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao/rna-pp.h5ad")
atac = ad.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao/atac-pp.h5ad")
guidance = nx.read_graphml("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao/guidance.graphml.gz")

scglue.models.configure_dataset(
    rna, "NB", # use_highly_variable=True,
    use_layer="counts", 
    use_rep="X_pca",
    use_batch="batch"
)
scglue.models.configure_dataset(
    atac, "NB", # use_highly_variable=True,
    use_layer="counts", 
    use_rep= "X_pca", #"X_lsi", #
    use_batch="batch"
)
guidance_hvf = guidance.subgraph(chain(
    rna.var.index,
    atac.var.index
)).copy()

glue = scglue.models.fit_SCGLUE(
    {"rna": rna, "atac": atac}, guidance_hvf,
    fit_kws={"directory": "glue"}
)

glue.save("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao/glue.dill")

rna.obsm["X_glue"] = glue.encode_data("rna", rna)
atac.obsm["X_glue"] = glue.encode_data("atac", atac)
combined = ad.concat([rna, atac])
np.save("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao/glue_nb_pca.npy",combined.obsm["X_glue"])

###############
scglue.plot.set_publication_params()
rcParams["figure.figsize"] = (4, 4)

rna = ad.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao/rna-pp.h5ad")
atac = ad.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao/atac-pp.h5ad")
guidance = nx.read_graphml("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao/guidance.graphml.gz")

scglue.models.configure_dataset(
    rna, "Normal", use_highly_variable=True,
    # use_layer="counts", 
    use_rep="X_pca",
    use_batch="batch"
)
scglue.models.configure_dataset(
    atac, "Normal", use_highly_variable=True,
    # use_layer="counts", 
    use_rep= "X_pca", #"X_lsi", #
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

glue.save("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao/glue.dill")

rna.obsm["X_glue"] = glue.encode_data("rna", rna)
atac.obsm["X_glue"] = glue.encode_data("atac", atac)
combined = ad.concat([rna, atac])
np.save("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao/glue_norm_hvg_pca.npy",combined.obsm["X_glue"])

##########
scglue.plot.set_publication_params()
rcParams["figure.figsize"] = (4, 4)

rna = ad.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao/rna-pp.h5ad")
atac = ad.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao/atac-pp.h5ad")
guidance = nx.read_graphml("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao/guidance.graphml.gz")

scglue.models.configure_dataset(
    rna, "Normal", # use_highly_variable=True,
    # use_layer="counts", 
    use_rep="X_pca",
    use_batch="batch"
)
scglue.models.configure_dataset(
    atac, "Normal", # use_highly_variable=True,
    # use_layer="counts", 
    use_rep= "X_pca", #"X_lsi", #
    use_batch="batch"
)
guidance_hvf = guidance.subgraph(chain(
    rna.var.index,
    atac.var.index
)).copy()

glue = scglue.models.fit_SCGLUE(
    {"rna": rna, "atac": atac}, guidance_hvf,
    fit_kws={"directory": "glue"}
)

glue.save("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao/glue.dill")

rna.obsm["X_glue"] = glue.encode_data("rna", rna)
atac.obsm["X_glue"] = glue.encode_data("atac", atac)
combined = ad.concat([rna, atac])
np.save("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao/glue_norm_pca.npy",combined.obsm["X_glue"])
