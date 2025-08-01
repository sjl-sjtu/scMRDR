
import sys
sys.path.insert(1, '/ailab/user/sunjianle/integration26/UnionCom2')

from UnionCom2 import unioncom
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import scanpy as sc
import pandas as pd
import random
import anndata
seed = 1
np.random.seed(seed)
random.seed(seed)

import scanpy as sc 
import anndata as ad

atac = sc.read("/ailab/user/sunjianle-hdd/integration27/mop/muto/ATAC_counts_qc.h5ad")
rna = sc.read("/ailab/user/sunjianle-hdd/integration27/mop/muto/RNA_counts_qc.h5ad")

atac = atac[:, ~atac.var_names.duplicated(keep="first")]
rna = rna[:, ~rna.var_names.duplicated(keep="first")]

genelist = rna.var.index[rna.var['highly_variable']==True]
peaklist = atac.var.index[atac.var['highly_variable']==True]
atac = atac[:,peaklist]
rna = rna[:,genelist]

RNA = rna.X.toarray().copy()
ATAC = atac.X.toarray().copy()

uc = unioncom(distance_mode='geodesic',manual_seed=1234, kmin=20)

F = uc.find_correspondece(
    RNA, 
    ATAC, 
    epoch=1000, 
    integration_mode='v',
    device='cuda:0', 
)
data1_aligned, data2_aligned = uc.align(ATAC,F,device='cuda:0')
aligned_domain = np.concatenate([data1_aligned, data2_aligned],axis=0)
np.save("/ailab/user/sunjianle-hdd/integration27/mop/muto/unioncom.npy",aligned_domain)
