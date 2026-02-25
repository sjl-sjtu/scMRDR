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
from module import Integration,to_dense_array


adata = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/10x/feature_aligned.h5ad")
adata
print(adata)
rna_hvg = np.where(adata.var_names.isin(adata.uns['rna_hvg']))[0].tolist()
atac_hvg = np.where(adata.var_names.isin(adata.uns['atac_hvg']))[0].tolist()
feat = list(set(rna_hvg)|set(atac_hvg))
adata = adata[:,feat].copy()
print(adata)
# feature_list = {"0":rna_hvg,"1":atac_hvg}

celltypes = adata.obs.cell_type.unique()


model = Integration(data=adata, modality_key="modality", distribution="Normal_positive",
                    ) #, batch_key="batch") , feature_list=feature_list 
model.setup(hidden_layers = [512,512], latent_dim_shared = 20, latent_dim_specific=20, 
            beta = 2, gamma = 10, lambda_adv = 20, dropout_rate=0.5)
model.train(epoch_num = 200, batch_size = 128, lr = 1e-3, adaptlr = False, num_warmup = 1,
            early_stopping = True, valid_prop = 0.1)
model.inference(n_samples=1,update=True,returns=False)
# model.integrate(num_heads = 5, paired = False, epoch_num = 200, batch_size = 128, lr = 5e-4,update=True,returns=False)
adata = model.get_adata()
adata.write("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/10x/feature_aligned_trained_mse.h5ad")
# adata.write("/home/bingxing2/ailab/group/ai4bio/sunjianle//BMMC/data2/feature_aligned_unpaired_trained_mse.h5ad")

# prediction
adata_urna_all = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/10x/RNA_counts_unseen.h5ad")
adata_uatac_all = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/10x/ATAC_gene_unseen.h5ad")
featlist = adata.uns['feat']
adata_urna_all = adata_urna_all[:,featlist].copy()
adata_uatac_all = adata_uatac_all[:,featlist].copy()

##
print("ot")
unseen_rna_all = model.predict(predict_modality="0", method="ot")
unseen_atac_all = model.predict(predict_modality="1", method="ot")

np.save("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/10x/predict_rna_mse_ot.npy", unseen_rna_all)
np.save("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/10x/predict_atac_mse_ot.npy", unseen_atac_all)

for cell_type in celltypes:
    idx_rna = adata_urna_all[adata_urna_all.obs['cell_type'] == cell_type].obs_names
    idx_atac = adata_uatac_all[adata_uatac_all.obs['cell_type'] == cell_type].obs_names

    adata_urna = adata_urna_all[idx_rna,adata.uns['rna_hvg']].copy()
    adata_uatac = adata_uatac_all[idx_atac,adata.uns['atac_hvg']].copy()

    unseen_rna = unseen_rna_all[adata_urna_all.obs['cell_type'] == cell_type,:][:,rna_hvg].copy()
    unseen_atac = unseen_atac_all[adata_uatac_all.obs['cell_type'] == cell_type,:][:,atac_hvg].copy()

    m_original_rna = np.nanmean(true_rna, axis=0)
    m_original_atac = np.nanmean(true_atac, axis=0)
    m_unseen_rna = np.nanmean(unseen_rna, axis=0)
    m_unseen_atac = np.nanmean(unseen_atac, axis=0)
    rmse = np.sqrt(np.mean((m_unseen_rna - m_original_rna)**2))
    print("RMSE for mean expression of unseen RNA: ", rmse)
    rmse = np.sqrt(np.mean((m_unseen_atac - m_original_atac)**2))
    print("RMSE for mean expression of unseen ATAC: ", rmse)

    true_rna = to_dense_array(adata_urna.X)
    true_atac = to_dense_array(adata_uatac.X)

    true_rna = np.where(true_rna!=0,true_rna,np.nan)
    true_atac = np.where(true_atac!=0,true_atac,np.nan)
    unseen_atac = np.where(unseen_atac != 0, unseen_atac, np.nan)
    unseen_rna = np.where(unseen_rna != 0, unseen_rna, np.nan)

    ## median relative error for mean expression
    m_original_rna = np.nanmean(to_dense_array(adata_urna.layers['counts']), axis=0)
    m_original_atac = np.nanmean(to_dense_array(adata_uatac.layers['counts']), axis=0)
    m_unseen_rna = np.nanmedian(unseen_rna, axis=0)
    m_unseen_atac = np.nanmedian(unseen_atac, axis=0)
    mre_rna = np.nanmedian(np.abs(m_unseen_rna - m_original_rna) / (m_original_rna + 1e-10))
    print("Median relative error for mean expression of unseen RNA: ", mre_rna)
    mre_atac = np.nanmedian(np.abs(m_unseen_atac - m_original_atac) / (m_original_atac + 1e-10))
    print("Median relative error for mean expression of unseen ATAC: ", mre_atac) 

    ## mean
    m_original_rna = np.nanmean(true_rna, axis=0)
    m_original_atac = np.nanmean(true_atac, axis=0)
    m_unseen_rna = np.nanmean(unseen_rna, axis=0)
    m_unseen_atac = np.nanmean(unseen_atac, axis=0)
    rmse = np.sqrt(np.nanmean((m_unseen_rna - m_original_rna)**2))
    print("RMSE for mean expression of unseen RNA: ", rmse)
    rmse = np.sqrt(np.nanmean((m_unseen_atac - m_original_atac)**2))
    print("RMSE for mean expression of unseen ATAC: ", rmse)

    mre_rna = np.nanmean(np.abs(m_unseen_rna - m_original_rna) / (m_original_rna + 1e-10))
    print("Mean relative error for mean expression of unseen RNA: ", mre_rna)
    mre_atac = np.nanmean(np.abs(m_unseen_atac - m_original_atac) / (m_original_atac + 1e-10))
    print("Mean relative error for mean expression of unseen ATAC: ", mre_atac)  

    # pearson r between mean expression of unseen and original RNA
    from scipy.stats import pearsonr
    valid = np.isfinite(m_unseen_rna) & np.isfinite(m_original_rna)
    r_rna = pearsonr(m_unseen_rna[valid], m_original_rna[valid])[0]
    print("Pearson r for mean expression of unseen RNA: ", r_rna)
    valid = np.isfinite(m_unseen_atac) & np.isfinite(m_original_atac)
    r_atac = pearsonr(m_unseen_atac[valid], m_original_atac[valid])[0]
    print("Pearson r for mean expression of unseen ATAC: ", r_atac)

    # spearman r between mean expression of unseen and original RNA
    from scipy.stats import spearmanr
    r_rna = spearmanr(m_unseen_rna, m_original_rna, nan_policy='omit')[0]
    print("Spearman r for mean expression of unseen RNA: ", r_rna)  
    r_atac = spearmanr(m_unseen_atac, m_original_atac, nan_policy='omit')[0]
    print("Spearman r for mean expression of unseen ATAC: ", r_atac)


    print("true rna",m_unseen_rna)
    print("predicted rna",m_original_rna)
    print("true atac",m_unseen_atac)
    print("predicted atac",m_original_atac)

####
print("knn")
unseen_rna_all = model.predict(predict_modality="0", method="knn")
unseen_atac_all = model.predict(predict_modality="1", method="knn")

np.save("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/10x/predict_rna_mse_knn.npy", unseen_rna_all)
np.save("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/10x/predict_atac_mse_knn.npy", unseen_atac_all)

for cell_type in celltypes:
    idx_rna = adata_urna_all[adata_urna_all.obs['cell_type'] == cell_type].obs_names
    idx_atac = adata_uatac_all[adata_uatac_all.obs['cell_type'] == cell_type].obs_names

    adata_urna = adata_urna_all[idx_rna,adata.uns['rna_hvg']].copy()
    adata_uatac = adata_uatac_all[idx_atac,adata.uns['atac_hvg']].copy()

    unseen_rna = unseen_rna_all[adata_urna_all.obs['cell_type'] == cell_type,:][:,rna_hvg].copy()
    unseen_atac = unseen_atac_all[adata_uatac_all.obs['cell_type'] == cell_type,:][:,atac_hvg].copy()

    rmse = np.sqrt(np.mean((unseen_rna - to_dense_array(adata_urna.X))**2, axis=1))
    print("RMSE for unseen RNA: ", np.mean(rmse))
    rmse = np.sqrt(np.mean((unseen_atac - to_dense_array(adata_uatac.X))**2, axis=1))
    print("RMSE for unseen ATAC: ", np.mean(rmse))

    ## rmse for mean expression
    m_original_rna = np.mean(to_dense_array(adata_urna.X), axis=0)
    m_original_atac = np.mean(to_dense_array(adata_uatac.X), axis=0)
    m_unseen_rna = np.mean(unseen_rna, axis=0)
    m_unseen_atac = np.mean(unseen_atac, axis=0)
    rmse = np.sqrt(np.mean((m_unseen_rna - m_original_rna)**2))
    print("RMSE for mean expression of unseen RNA: ", rmse)
    rmse = np.sqrt(np.mean((m_unseen_atac - m_original_atac)**2))
    print("RMSE for mean expression of unseen ATAC: ", rmse)

    true_rna = to_dense_array(adata_urna.X)
    true_atac = to_dense_array(adata_uatac.X)

    true_rna = np.where(true_rna!=0,true_rna,np.nan)
    true_atac = np.where(true_atac!=0,true_atac,np.nan)
    unseen_atac = np.where(unseen_atac != 0, unseen_atac, np.nan)
    unseen_rna = np.where(unseen_rna != 0, unseen_rna, np.nan)

    ## median relative error for mean expression
    m_original_rna = np.nanmean(to_dense_array(adata_urna.layers['counts']), axis=0)
    m_original_atac = np.nanmean(to_dense_array(adata_uatac.layers['counts']), axis=0)
    m_unseen_rna = np.nanmedian(unseen_rna, axis=0)
    m_unseen_atac = np.nanmedian(unseen_atac, axis=0)
    mre_rna = np.nanmedian(np.abs(m_unseen_rna - m_original_rna) / (m_original_rna + 1e-10))
    print("Median relative error for mean expression of unseen RNA: ", mre_rna)
    mre_atac = np.nanmedian(np.abs(m_unseen_atac - m_original_atac) / (m_original_atac + 1e-10))
    print("Median relative error for mean expression of unseen ATAC: ", mre_atac) 


    ## mean
    m_original_rna = np.nanmean(true_rna, axis=0)
    m_original_atac = np.nanmean(true_atac, axis=0)
    m_unseen_rna = np.nanmean(unseen_rna, axis=0)
    m_unseen_atac = np.nanmean(unseen_atac, axis=0)
    rmse = np.sqrt(np.nanmean((m_unseen_rna - m_original_rna)**2))
    print("RMSE for mean expression of unseen RNA: ", rmse)
    rmse = np.sqrt(np.nanmean((m_unseen_atac - m_original_atac)**2))
    print("RMSE for mean expression of unseen ATAC: ", rmse)

    mre_rna = np.nanmean(np.abs(m_unseen_rna - m_original_rna) / (m_original_rna + 1e-10))
    print("Mean relative error for mean expression of unseen RNA: ", mre_rna)
    mre_atac = np.nanmean(np.abs(m_unseen_atac - m_original_atac) / (m_original_atac + 1e-10))
    print("Mean relative error for mean expression of unseen ATAC: ", mre_atac)  

    # pearson r between mean expression of unseen and original RNA
    from scipy.stats import pearsonr
    valid = np.isfinite(m_unseen_rna) & np.isfinite(m_original_rna)
    r_rna = pearsonr(m_unseen_rna[valid], m_original_rna[valid])[0]
    print("Pearson r for mean expression of unseen RNA: ", r_rna)
    valid = np.isfinite(m_unseen_atac) & np.isfinite(m_original_atac)
    r_atac = pearsonr(m_unseen_atac[valid], m_original_atac[valid])[0]
    print("Pearson r for mean expression of unseen ATAC: ", r_atac)

    # spearman r between mean expression of unseen and original RNA
    from scipy.stats import spearmanr
    r_rna = spearmanr(m_unseen_rna, m_original_rna, nan_policy='omit')[0]
    print("Spearman r for mean expression of unseen RNA: ", r_rna)  
    r_atac = spearmanr(m_unseen_atac, m_original_atac, nan_policy='omit')[0]
    print("Spearman r for mean expression of unseen ATAC: ", r_atac)


    print("true rna",m_unseen_rna)
    print("predicted rna",m_original_rna)
    print("true atac",m_unseen_atac)
    print("predicted atac",m_original_atac)
