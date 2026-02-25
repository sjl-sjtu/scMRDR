import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import optim
import torch.utils.data as Data
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
import anndata as ad
from torch.utils.tensorboard import SummaryWriter
from scipy.sparse import lil_matrix
import scanpy as sc
# from data import IntegrateDataset,CombinedDataset
# from model import embeddingNet,integrateNet
# from train import train_model, inference_model, train_integrate, inference_integrate
# from module import Integration

import sys
sys.path.insert(1, '/home/bingxing2/ailab/group/ai4bio/sunjianle/integration/models2/scMRDR/src/')
from scMRDR.module import Integration


import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42  
mpl.rcParams['ps.fonttype'] = 42


################
adata = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mouse_brain_simo/merged.h5ad")
# adata.obs['modality2'] = (adata.obs['modality']=="methy").astype(str) # 0 for RNA, 1 for methy
print(adata)

adata[adata.obs.modality=="methy_mCG",:].X = -adata[adata.obs.modality=="methy_mCG",:].X
adata[adata.obs.modality=="methy_mCH",:].X = -adata[adata.obs.modality=="methy_mCH",:].X

label_mapping = {
    # data 1
    "mL2/3": "Layer2/3",
    "mL4": "Layer4/5",
    "mL5-1": "Layer4/5",
    "mL5-2": "Layer4/5",
    "mL6-1": "Layer6",
    "mL6-2": "Layer6",
    "mDL-1": "MGE",
    "mDL-2": "MGE",
    "mDL-3": "MGE",
    "mIn-1": "MGE",
    "mSst-1": "MGE",
    "mSst-2": "MGE",
    "mPv": "MGE",
    "mNdnf-1": "CGE",
    "mNdnf-2": "CGE",
    "mVip": "CGE",

    # data 2
    "Layer2/3": "Layer2/3",
    "Layer5": "Layer4/5",
    "Layer5a": "Layer4/5",
    "Layer5b": "Layer4/5",
    "Layer6": "Layer6",
    "CGE": "CGE",
    "MGE": "MGE",
    "Claustrum": "Claustrum"
}

adata.obs['harmonic_celltype'] = adata.obs['cell_type'].map(label_mapping)
order = ["Layer2/3", "Layer4/5", "Layer6", "MGE","CGE","Claustrum"] 
cat_type = pd.CategoricalDtype(categories=order, ordered=True)
adata.obs['harmonic_celltype'] = adata.obs['harmonic_celltype'].astype(cat_type)

from matplotlib.colors import ListedColormap
cmap = mpl.colormaps["Dark2"]
colors = [cmap(i) for i in range(len(order))]
# palette = dict(zip(order, colors))
palette = ListedColormap(colors)


rna_hvg = adata.uns['rna_feat']
methy_hvg = adata.uns['methy_mcg_feat']
methy_hvg2 = adata.uns['methy_mch_feat']
st_hvg = adata.uns['st_feat']
shared_feat = list(set(rna_hvg)&set(methy_hvg)&set(methy_hvg2)&set(st_hvg))
adata = adata[:,shared_feat].copy()
print(adata)
# adata.X = adata.X
model = Integration(data=adata, modality_key="modality", #layer="norm",# batch_key="donor_id",#mask_key="modality", #batch_key="batch", 
                    # celltype_key="harmonic_celltype", 
                    feature_list=None, distribution="Normal") #feature_list)
# model = Integration(data=adata, modality_key="modality", batch_key="batch", 
#                     distribution="Normal_positive") #, feature_list=feature_list)
model.setup(hidden_layers = [512,512], latent_dim_shared = 20, latent_dim_specific=20, 
            beta = 10, gamma = 20, lambda_adv = 20, dropout_rate=0.2)
model.train(epoch_num = 200, batch_size = 128, lr = 1e-3, adaptlr = False, num_warmup = 0,
            early_stopping = True, valid_prop = 0.1, weighted=True, patience=20)
model.inference(n_samples=1,update=True,returns=False)
# model.integrate(num_heads = 5, paired = False, epoch_num = 200, batch_size = 128, lr = 5e-4,update=True,returns=False)
adata = model.get_adata()
adata.write("/home/bingxing2/ailab/group/ai4bio/sunjianle/mouse_brain_simo/adata_aligned_trained.h5ad")

sc.pp.neighbors(adata,use_rep="latent_shared")
sc.tl.umap(adata)
sc.pl.umap(
    adata,
    color=["modality","harmonic_celltype"],
    # Setting a smaller point size to get prevent overlap
    size=2,
    save="_integrate.png"
)
sc.pl.umap(
    adata[adata.obs.modality=="st",:].copy(),
    color=["seurat_clusters"],
    # Setting a smaller point size to get prevent overlap
    size=10,
    save="_st.png"
)
sc.pl.umap(
    adata[adata.obs.modality=="rna",:].copy(),
    color=["harmonic_celltype"],
    # Setting a smaller point size to get prevent overlap
    size=2,
    save="_rna.png"
)
sc.pl.umap(
    adata[adata.obs.modality=="methy_mCG",:].copy(),
    color=["harmonic_celltype"],
    # Setting a smaller point size to get prevent overlap
    size=10,
    save="_methy_mCG.png"
)
sc.pl.umap(
    adata[adata.obs.modality=="methy_mCH",:].copy(),
    color=["harmonic_celltype"],
    # Setting a smaller point size to get prevent overlap
    size=10,
    save="_methy_mCH.png"
)


adata.obs['has_loc'] = adata.obs.modality=='st'
latent = adata.obsm['latent_shared']
z0 = latent[adata.obs.modality=="rna",:]
z1 = latent[adata.obs.modality=="methy_mCG",:]
z3 = latent[adata.obs.modality=="methy_mCH",:]
z2 = latent[adata.obs.has_loc==1,:]
r2 = adata[adata.obs.has_loc==1,:].obsm['spatial'].copy()
#
import ot
p = ot.unif(z0.shape[0])
q = ot.unif(z2.shape[0])
cost_matrix = ot.dist(z0, z2, metric='sqeuclidean')
cost_matrix = cost_matrix/np.max(cost_matrix)
W = ot.emd(p, q, cost_matrix)
# W = ot.sinkhorn(p, q, cost_matrix, reg=0.01) # ot.sinkhorn
# W = W / np.median(W)
W = W / W.sum(axis=1, keepdims=True)
r0 = W @ r2
#
p = ot.unif(z1.shape[0])
q = ot.unif(z2.shape[0])
cost_matrix = ot.dist(z1, z2, metric='sqeuclidean')
cost_matrix = cost_matrix/np.max(cost_matrix)
W = ot.emd(p, q, cost_matrix)
# W = ot.sinkhorn(p, q, cost_matrix, reg=0.01) # ot.sinkhorn
W = W / W.sum(axis=1, keepdims=True)
r1 = W @ r2

p = ot.unif(z3.shape[0])
q = ot.unif(z2.shape[0])
cost_matrix = ot.dist(z3, z2, metric='sqeuclidean')
cost_matrix = cost_matrix/np.max(cost_matrix)
W = ot.emd(p, q, cost_matrix)
# W = ot.sinkhorn(p, q, cost_matrix, reg=0.01) # ot.sinkhorn
W = W / W.sum(axis=1, keepdims=True)
r3 = W @ r2

adata.obsm['X_spatial_imputed'] = np.concatenate([r0,r1,r3,r2],axis=0)

import squidpy as sq
sq.pl.spatial_scatter(adata[adata.obs.modality=="st",:], library_id=None,spatial_key = 'X_spatial_imputed', 
                      color=["seurat_clusters"],shape=None,size=10,save="_st_spatial.pdf",palette="Dark2")
sq.pl.spatial_scatter(adata[adata.obs.modality=="rna",:], library_id=None,spatial_key = 'X_spatial_imputed', 
                      color=["harmonic_celltype"],shape=None,size=5,save="_rna_spatial.pdf",palette=palette)
sq.pl.spatial_scatter(adata[adata.obs.modality=="methy_mCG",:], library_id=None,spatial_key = 'X_spatial_imputed', 
                      color=["harmonic_celltype"],shape=None,size=10,save="_methy_mCG_spatial.pdf",palette=palette)
sq.pl.spatial_scatter(adata[adata.obs.modality=="methy_mCH",:], library_id=None,spatial_key = 'X_spatial_imputed', 
                      color=["harmonic_celltype"],shape=None,size=10,save="_methy_mCH_spatial.pdf",palette=palette)

sq.pl.spatial_scatter(adata[adata.obs.modality=="st",:], library_id=None,spatial_key = 'X_spatial_imputed', 
                      color=["seurat_clusters"],shape=None,size=10,save="_st_spatial.png",palette="Dark2")
sq.pl.spatial_scatter(adata[adata.obs.modality=="rna",:], library_id=None,spatial_key = 'X_spatial_imputed', 
                      color=["harmonic_celltype"],shape=None,size=5,save="_rna_spatial.png",palette=palette)
sq.pl.spatial_scatter(adata[adata.obs.modality=="methy_mCG",:], library_id=None,spatial_key = 'X_spatial_imputed', 
                      color=["harmonic_celltype"],shape=None,size=10,save="_methy_mCG_spatial.png",palette=palette)
sq.pl.spatial_scatter(adata[adata.obs.modality=="methy_mCH",:], library_id=None,spatial_key = 'X_spatial_imputed', 
                      color=["harmonic_celltype"],shape=None,size=10,save="_methy_mCH_spatial.png",palette=palette)

adata[adata.obs.modality=="methy_mCG",:].write_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mouse_brain_simo/methy_mCG_imputed.h5ad")
adata[adata.obs.modality=="rna",:].write_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mouse_brain_simo/rna_imputed.h5ad")
adata[adata.obs.modality=="methy_mCH",:].write_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mouse_brain_simo/methy_mCH_imputed.h5ad")

# sq.pl.spatial_scatter(adata[adata.obs.modality=="st",:], library_id=None,spatial_key = 'spatial', color="seurat_clusters",
#                       shape=None,size=10,save="st_plot.pdf")
# sq.pl.spatial_scatter(adata[adata.obs.modality=="rna",:], library_id=None,spatial_key = 'X_spatial_imputed', 
#                       color=["cell_type"],shape=None,size=10,save="rna_imputated_plot.pdf")
# sq.pl.spatial_scatter(adata[adata.obs.modality=="methy_mCG",:], library_id=None,spatial_key = 'X_spatial_imputed', 
#                       color=["cell_type"],shape=None,size=10,save="methy_mCG_imputated_plot.pdf")
# sq.pl.spatial_scatter(adata[adata.obs.modality=="methy_mCH",:], library_id=None,spatial_key = 'X_spatial_imputed', 
#                       color=["cell_type"],shape=None,size=10,save="methy_mCH_imputated_plot.pdf")
