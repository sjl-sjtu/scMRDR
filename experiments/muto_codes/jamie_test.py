import numpy as np
import pandas as pd
import scanpy as sc
from jamie import JAMIE
import torch
# from muon import atac as ac

atac = sc.read("/ailab/user/sunjianle-hdd/integration27/mop/muto/ATAC_counts_qc.h5ad")
rna = sc.read("/ailab/user/sunjianle-hdd/integration27/mop/muto/RNA_counts_qc.h5ad")
atac.X = atac.layers['counts'].copy()
rna.X = rna.layers['counts'].copy()
atac = atac[:, ~atac.var_names.duplicated(keep="first")]
rna = rna[:, ~rna.var_names.duplicated(keep="first")]

genelist = rna.var.index[rna.var['highly_variable']==True]
peaklist = atac.var.index[atac.var['highly_variable']==True]
atac = atac[:,peaklist]
rna = rna[:,genelist]

from sklearn import preprocessing
data1 = rna.X.toarray() #.todense()
data2 = atac.X.toarray() #.todense()
data1 = preprocessing.scale(np.asarray(data1), axis=0)
data2 = preprocessing.scale(np.asarray(data2), axis=0)
data1[np.isnan(data1)] = 0  # Replace NaN with average
data2[np.isnan(data2)] = 0

reduced_dim = 32
kwargs = {
    'output_dim': reduced_dim,
    'epoch_DNN': 10000,
    'min_epochs': 2500,
    'log_DNN': 500,
    'use_early_stop': True,
    'batch_size': 512,
    'pca_dim': 2*[512],
    'dist_method': 'euclidean',
    'loss_weights': [1,1,1,1],
    # 'loss_weights': [1,1,1,0],
    # 'use_f_tilde': False,
    'dropout': 0,
    'enable_memory_logging': True,
    'device': "cuda"
}

corr = np.eye(data1.shape[0], data2.shape[0])
jm = JAMIE(**kwargs)
print(jm.device)
integrated_data = jm.fit_transform(dataset=[data1, data2],P=corr)
np.save("/ailab/user/sunjianle-hdd/integration27/mop/muto/jamie.npy",integrated_data)
