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
sys.path.insert(1, '/home/bingxing2/ailab/group/ai4bio/sunjianle/integration/models2')
from module import Integration


adata = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/BMMC/feature_aligned_sampled.h5ad")

adata
rna_hvg = np.where(adata.var_names.isin(adata.uns['rna_hvg']))[0].tolist()
atac_hvg = np.where(adata.var_names.isin(adata.uns['atac_hvg']))[0].tolist()
prot_hvg = np.where(adata.var_names.isin(adata.uns['prot_hvg']))[0].tolist()
feature_list = {"0":rna_hvg,"1":atac_hvg,"2":prot_hvg}

model = Integration(data=adata, layer = 'counts', modality_key="modality",count_data=True, 
                    batch_key="batch", feature_list=feature_list)
model.setup(hidden_layers = [128,128], latent_dim_shared = 30, latent_dim_specific = 30, 
            beta = 5, gamma = 10, lambda_adv = 10, dropout_rate=0.2)
model.train(epoch_num = 200, batch_size = 64, lr = 1e-3, adaptlr = False, num_warmup = 0,
            early_stopping = True, valid_prop = 0.1, patience = 25)
model.inference(n_samples=1,update=True,returns=False)
# model.integrate(num_heads = 5, paired = False, epoch_num = 200, batch_size = 128, lr = 5e-4,update=True,returns=False)
adata = model.get_adata()
adata.write("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/BMMC/feature_aligned_trained_sampled.h5ad")
# adata.write("/home/bingxing2/ailab/group/ai4bio/sunjianle/BMMC/data2/feature_aligned_unpaired_trained_mse.h5ad")

