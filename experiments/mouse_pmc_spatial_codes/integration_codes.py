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

import sys
sys.path.insert(1, '/home/bingxing2/ailab/scxlab0179/integration/models2')
from module import Integration

# using hvg only
adata = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mouse_pmc/merged.h5ad")
# adata.obs['modality2'] = (adata.obs['modality']=="atac").astype(str) # 0 for RNA, 1 for ATAC
print(adata)
# rna_hvg = np.where(adata.var_names.isin(adata.uns['rna_feat']))[0].tolist()
# atac_hvg = np.where(adata.var_names.isin(adata.uns['atac_feat']))[0].tolist()
# st_hvg = np.where(adata.var_names.isin(adata.uns['st_feat']))[0].tolist()
rna_hvg = np.where(adata.var_names.isin(adata.uns['rna_hvg']))[0].tolist()
atac_hvg = np.where(adata.var_names.isin(adata.uns['atac_hvg']))[0].tolist()
st_hvg = np.where(adata.var_names.isin(adata.uns['st_hvg']))[0].tolist()
feature_list = {"rna":rna_hvg,"atac":atac_hvg,"st":st_hvg}
model = Integration(data=adata, layer="counts", modality_key="modality", batch_key="batch",#mask_key="modality", #batch_key="batch", 
                    feature_list=feature_list, distribution="ZINB") #feature_list)
# model = Integration(data=adata, modality_key="modality", batch_key="batch", 
#                     count_data=False) #, feature_list=feature_list)
model.setup(hidden_layers = [500,500], latent_dim_shared = 30, latent_dim_specific=30, 
            beta = 5, gamma = 10, lambda_adv = 25, dropout_rate=0.2)
model.train(epoch_num = 200, batch_size = 64, lr = 1e-3, adaptlr = True, num_warmup = 0,
            early_stopping = True, valid_prop = 0.1)
model.inference(n_samples=1,update=True,returns=False)
adata = model.get_adata()
adata.write("/home/bingxing2/ailab/group/ai4bio/sunjianle/mouse_pmc/adata_aligned_trained_hvg.h5ad")

# using all features
adata = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mouse_pmc/merged.h5ad")
# adata.obs['modality2'] = (adata.obs['modality']=="atac").astype(str) # 0 for RNA, 1 for ATAC
print(adata)
rna_hvg = np.where(adata.var_names.isin(adata.uns['rna_feat']))[0].tolist()
atac_hvg = np.where(adata.var_names.isin(adata.uns['atac_feat']))[0].tolist()
st_hvg = np.where(adata.var_names.isin(adata.uns['st_feat']))[0].tolist()
# rna_hvg = np.where(adata.var_names.isin(adata.uns['rna_hvg']))[0].tolist()
# atac_hvg = np.where(adata.var_names.isin(adata.uns['atac_hvg']))[0].tolist()
# st_hvg = np.where(adata.var_names.isin(adata.uns['st_hvg']))[0].tolist()
feature_list = {"rna":rna_hvg,"atac":atac_hvg,"st":st_hvg}
model = Integration(data=adata, layer="counts", modality_key="modality", batch_key="batch",#mask_key="modality", #batch_key="batch", 
                    feature_list=feature_list, distribution="ZINB") #feature_list)
# model = Integration(data=adata, modality_key="modality", batch_key="batch", 
#                     count_data=False) #, feature_list=feature_list
model.setup(hidden_layers = [500,500], latent_dim_shared = 30, latent_dim_specific=30, 
            beta = 5, gamma = 10, lambda_adv = 30, dropout_rate=0.2)
model.train(epoch_num = 200, batch_size = 64, lr = 1e-3, adaptlr = True, num_warmup = 0,
            early_stopping = True, valid_prop = 0.1)
model.inference(n_samples=1,update=True,returns=False)
adata = model.get_adata()
adata.write("/home/bingxing2/ailab/group/ai4bio/sunjianle/mouse_pmc/adata_aligned_trained.h5ad")
