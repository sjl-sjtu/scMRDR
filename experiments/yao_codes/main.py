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
sys.path.insert(1, '/home/bingxing2/ailab/group/ai4bio/sunjianle/integration/models2')
from module import Integration

# adata = sc.read_h5ad("/ailab/user/sunjianle-hdd/integration27/BMMC/data2/feature_aligned_unpaired.h5ad")
# atac = sc.read("/ailab/user/sunjianle-hdd/integration27/BMMC/data2/ATAC_counts_qc_slt.h5ad")
# rna = sc.read("/ailab/user/sunjianle-hdd/integration27/BMMC/data2/RNA_counts_qc_slt.h5ad")
# rna = adata[adata.obs.modality=="0",:][rna.obs_names,:]
# atac = adata[adata.obs.modality=="1",:][atac.obs_names,:]
# adata = ad.concat([rna,atac], join='inner')
# adata.write("/ailab/user/sunjianle-hdd/integration27/BMMC/data2/feature_aligned_unpaired.h5ad")
adata = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao/feature_aligned.h5ad")
adata
rna_hvg = np.where(adata.var_names.isin(adata.uns['rna_hvg']))[0].tolist()
atac_hvg = np.where(adata.var_names.isin(adata.uns['atac_hvg']))[0].tolist()
feature_list = {"0":rna_hvg,"1":atac_hvg}
model = Integration(data=adata, layer="counts", modality_key="modality", batch_key="batch", 
                    feature_list=None, count_data=True) #feature_list)
# model = Integration(data=adata, modality_key="modality", batch_key="batch", 
#                     count_data=False) #, feature_list=feature_list)
model.setup(hidden_layers = [128,128], latent_dim_shared = 25, latent_dim_specific=25, 
            beta = 2, gamma = 5, lambda_adv = 5, dropout_rate=0.5)
model.train(epoch_num = 200, batch_size = 128, lr = 1e-3, adaptlr = False, num_warmup = 0,
            early_stopping = True, valid_prop = 0.1, patience = 25)
model.inference(n_samples=1,update=True,returns=False)
# model.integrate(num_heads = 5, paired = False, epoch_num = 200, batch_size = 128, lr = 5e-4,update=True,returns=False)
adata = model.get_adata()
adata.write("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao/feature_aligned_trained.h5ad")
# # adata.write("/ailab/user/sunjianle-hdd/integration27/BMMC/data2/feature_aligned_unpaired_trained_mse.h5ad")

