import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
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

print(rna)
print(atac)

import sys
sys.path.insert(1, '/ailab/user/sunjianle/integration26/SCOT/src')

RNA = rna.X.toarray().copy()
ATAC = atac.X.toarray().copy()

# import scotv1 
# scot = scotv1.SCOT(RNA,ATAC)
# aligned_domain1, aligned_domain2 = scot.align(e=0.001,normalize=True,XontoY=True,k=50)
# aligned_domain = np.concatenate([aligned_domain1,aligned_domain2],axis=0)
# np.save("/ailab/user/sunjianle-hdd/integration27/mop/muto/SCOT.npy",aligned_domain)

# import scotv2
# scot = scotv2.SCOTv2([RNA, ATAC])
# aligned_domain1, aligned_domain2 = scot.align(normalize=True, eps=0.005, rho=0.1, projMethod="barycentric",k=50)
# aligned_domain = np.concatenate([aligned_domain1,aligned_domain2],axis=0)
# np.save("/ailab/user/sunjianle-hdd/integration27/mop/muto/SCOTv2.npy",aligned_domain)

from pamona import Pamona
Pa = Pamona.Pamona(n_shared=[0], Lambda=1, epsilon=0.01,max_iter=1000)
integrated_data, T = Pa.run_Pamona([RNA, ATAC])
integrated_data = np.concatenate(integrated_data,axis=0)
np.save("/ailab/user/sunjianle-hdd/integration27/mop/muto/pamona.npy",integrated_data)


