import numpy as np 
import pandas as pd 
import scanpy as sc 
import torch
import scvi 

# adata = sc.read_h5ad("/ailab/user/sunjianle-hdd/integration27/BMMC/data2/feature_aligned_unpaired.h5ad")
# adata = adata[adata.obs['modality'].cat.codes==0,:].copy()
adata = sc.read_h5ad("/ailab/user/sunjianle-hdd/integration27/mop/BMMC/feature_aligned_sampled.h5ad")
print(adata)
scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="batch")
model = scvi.model.SCVI(adata, n_layers=2, n_latent=30, gene_likelihood="zinb")
model.train()
adata.obsm["X_scVI"] = model.get_latent_representation()
np.save("/ailab/user/sunjianle-hdd/integration27/mop/BMMC/scvi.npy",adata.obsm["X_scVI"])
adata.obsm["X_normalized_scVI"] = model.get_normalized_expression(n_samples=1)
# adata.write("/ailab/user/sunjianle-hdd/integration27/BMMC/data2/scvi_rna_unpaired_modalmix.h5ad")
adata.write("/ailab/user/sunjianle-hdd/integration27/mop/BMMC/scvi_unpaired_modalmix.h5ad")
