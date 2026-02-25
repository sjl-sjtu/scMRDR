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

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300   # 显示时分辨率
mpl.rcParams['savefig.dpi'] = 300 # 保存时分辨率

# 保存pdf时字体而非锚点
mpl.rcParams['pdf.fonttype'] = 42  

adata = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/sc-fm/cross_species_scgpt/cross_species.h5ad")
adata

sc.pp.highly_variable_genes(adata, n_top_genes=3000)
adata = adata[:,adata.var.highly_variable].copy()
adata.obs.species = adata.obs.species.astype(str).map({"0":"human","1":"macaque","2":"marmoset","3":"chimpanzee"})
print(adata)
print(adata.obs.species.value_counts())

model = Integration(data=adata, modality_key="species", batch_key="batch", 
                    feature_list=None, distribution="Normal_positive") #feature_list)
# model = Integration(data=adata, modality_key="modality", batch_key="batch", 
#                     count_data=False) #, feature_list=feature_list)
model.setup(hidden_layers = [512,512], latent_dim_shared = 25, latent_dim_specific=25, 
            beta = 1.5, gamma = 5, lambda_adv = 5, dropout_rate=0.5)
model.train(epoch_num = 200, batch_size = 128, lr = 1e-3, adaptlr = False, num_warmup = 0,
            early_stopping = True, valid_prop = 0.1, patience = 25)
model.inference(n_samples=1,update=True,returns=False)
# model.integrate(num_heads = 5, paired = False, epoch_num = 200, batch_size = 128, lr = 5e-4,update=True,returns=False)
adata = model.get_adata()
# adata.write("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/cross_species/ours_trained.h5ad")
# # adata.write("/ailab/user/sunjianle-hdd/integration27/BMMC/data2/feature_aligned_unpaired_trained_mse.h5ad")

sc.pp.neighbors(adata, use_rep='latent_shared')
sc.tl.umap(adata)
sc.pl.umap(adata, color=['Class','species','batch'], wspace=0.5, save='_cross.png')
sc.pl.umap(adata, color=['Subclass_ori','species','batch'], wspace=0.5, save='_cross_sub.png')
sc.pl.umap(adata, color=['Supertype_ori','species','batch'], wspace=0.5, save='_cross_super.png')

sc.pl.umap(adata, color=['Class','species','batch'], wspace=0.5, save='_cross.pdf')
sc.pl.umap(adata, color=['Subclass_ori','species','batch'], wspace=0.5, save='_cross_sub.pdf')
sc.pl.umap(adata, color=['Supertype_ori','species','batch'], wspace=0.5, save='_cross_super.pdf')

adata.write("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/cross_species/ours_trained.h5ad")

sc.pp.neighbors(adata, use_rep='latent_shared')
sc.tl.umap(adata)
sc.pl.umap(adata, color=['Class','species','batch'], wspace=0.5, save='_cross.pdf')
sc.pl.umap(adata, color=['Subclass_ori','species','batch'], wspace=0.5, save='_cross_sub.pdf')
sc.pl.umap(adata, color=['Supertype_ori','species','batch'], wspace=0.5, save='_cross_super.pdf')


