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
from scipy.stats import pearsonr,spearmanr

import sys
sys.path.insert(1, '/home/bingxing2/ailab/group/ai4bio/sunjianle/integration/models2')
from module import Integration,to_dense_array

# set matplotlib save dpi
SAVE_DPI = 300
plt.rcParams['savefig.dpi'] = SAVE_DPI
# set figures save path
sc.settings.figdir = './figures/'

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

def plot_corrslation(true, pred, title, savepath=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import pearsonr

    plt.figure(figsize=(4,4))
    plt.scatter(true, pred, s=1)
    
    # fit line and plotting
    m, b = np.polyfit(true, pred, 1)
    plt.plot(true, m*true + b, color='red')
    
    # calculate pearsonr
    r, p = pearsonr(true, pred)
    
    # title 只显示标题
    plt.title(title)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    
    # Use limits based on the data, or a common range for true/pred for a square plot
    # The original was:
    plt.xlim([np.min(true), np.max(true)])
    plt.ylim([np.min(true), np.max(true)])
    # For a more robust square plot based on the *combined* range:
    # min_val = min(np.min(true), np.min(pred))
    # max_val = max(np.max(true), np.max(pred))
    # plt.xlim([min_val, max_val])
    # plt.ylim([min_val, max_val])
    
    plt.text(
        0.95, 0.05, 
        f"r={r:.2f}\np={p:.2e}", 
        ha="right", va="bottom", 
        transform=plt.gca().transAxes, fontsize=10,
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")
    )
    
    plt.tight_layout()
    if savepath:
        # Note: The original had "figures/" prepended. Keep or remove as needed.
        if "figures/" not in savepath:
            savepath = "figures/" + savepath
        plt.savefig(savepath)
    plt.show()



adata = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/10x/feature_aligned.h5ad")
adata
print(adata)
rna_hvg = np.where(adata.var_names.isin(adata.uns['rna_hvg']))[0].tolist()
atac_hvg = np.where(adata.var_names.isin(adata.uns['atac_hvg']))[0].tolist()
feat = list(set(rna_hvg)|set(atac_hvg))
adata = adata[:,feat].copy()
print(adata)
adata.obs.modality = adata.obs.modality.map({"0":"RNA","1":"ATAC"})
adata.obs.cell_type = adata.obs.cell_type.str.replace(" ", "_")
# feature_list = {"0":rna_hvg,"1":atac_hvg}

celltypes = adata.obs.cell_type.unique()
print(celltypes)


print("scaled")

# prediction
adata_urna_all = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/10x/RNA_counts_unseen.h5ad")
adata_uatac_all = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/10x/ATAC_gene_unseen.h5ad")
adata_urna_all.obs.cell_type = adata_urna_all.obs.cell_type.str.replace(" ", "_")
adata_uatac_all.obs.cell_type = adata_uatac_all.obs.cell_type.str.replace(" ", "_")
featlist = adata.uns['feat']
adata_urna_all = adata_urna_all[:,featlist].copy()
adata_uatac_all = adata_uatac_all[:,featlist].copy()

df_res = pd.DataFrame(columns = ['cell_type','method','metrics','modality','value'])

##
print("ot")
method = "ot"

unseen_rna_all = np.load("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/10x/predict_rna_mse_ot.npy")
unseen_atac_all = np.load("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/10x/predict_atac_mse_ot.npy")

cell_type = "all"
adata_urna = adata_urna_all[:,:].copy()
adata_uatac = adata_uatac_all[:,:].copy()
true_rna = to_dense_array(adata_urna.X)
true_atac = to_dense_array(adata_uatac.X)
unseen_rna = unseen_rna_all[:,:].copy()
unseen_atac = unseen_atac_all[:,:].copy()
# rmse
rmse = np.sqrt(np.mean((unseen_rna - true_rna)**2, axis=1))
df_res = df_res._append({'cell_type':cell_type,'method':method,'metrics':'rmse','modality':'rna','value':np.mean(rmse)},ignore_index=True)
rmse = np.sqrt(np.mean((unseen_atac - true_atac)**2, axis=1))
df_res = df_res._append({'cell_type':cell_type,'method':method,'metrics':'rmse','modality':'atac','value':np.mean(rmse)},ignore_index=True)
# mean expression
true_mean_rna = np.mean(true_rna, axis=0)   
true_mean_atac = np.mean(true_atac, axis=0)
pred_mean_rna = np.mean(unseen_rna, axis=0)
pred_mean_atac = np.mean(unseen_atac, axis=0)
# rmse for mean expression
rmse = np.sqrt(np.mean((pred_mean_rna - true_mean_rna)**2))
df_res = df_res._append({'cell_type':cell_type,'method':method,'metrics':'rmse_mean','modality':'rna','value':rmse},ignore_index=True)
rmse = np.sqrt(np.mean((pred_mean_atac - true_mean_atac)**2))
df_res = df_res._append({'cell_type':cell_type,'method':method,'metrics':'rmse_mean','modality':'atac','value':rmse},ignore_index=True)
# pearson r between mean expression of unseen and original RNA
r_rna = pearsonr(pred_mean_rna, true_mean_rna)[0]
df_res = df_res._append({'cell_type':cell_type,'method':method,'metrics':'pearsonr_mean','modality':'rna','value':r_rna},ignore_index=True)
r_atac = pearsonr(pred_mean_atac, true_mean_atac)[0]
df_res = df_res._append({'cell_type':cell_type,'method':method,'metrics':'pearsonr_mean','modality':'atac','value':r_atac},ignore_index=True)
# spearman r between mean expression of unseen and original RNA
r_rna = spearmanr(pred_mean_rna, true_mean_rna)[0]
df_res = df_res._append({'cell_type':cell_type,'method':method,'metrics':'spearmanr_mean','modality':'rna','value':r_rna},ignore_index=True)
r_atac = spearmanr(pred_mean_atac, true_mean_atac)[0]
df_res = df_res._append({'cell_type':cell_type,'method':method,'metrics':'spearmanr_mean','modality':'atac','value':r_atac},ignore_index=True)

plot_corrslation(true_mean_rna, pred_mean_rna, title=f"RNA_{cell_type}_{method}", savepath=f"RNA_{cell_type}_{method}.png")
plot_corrslation(true_mean_atac, pred_mean_atac, title=f"ATAC_{cell_type}_{method}", savepath=f"ATAC_{cell_type}_{method}.png")

for cell_type in celltypes:
    idx_rna = adata_urna_all[adata_urna_all.obs['cell_type'] == cell_type].obs_names
    idx_atac = adata_uatac_all[adata_uatac_all.obs['cell_type'] == cell_type].obs_names

    adata_urna = adata_urna_all[idx_rna,:].copy()
    adata_uatac = adata_uatac_all[idx_atac,:].copy()

    true_rna = to_dense_array(adata_urna.X)
    true_atac = to_dense_array(adata_uatac.X)

    unseen_rna = unseen_rna_all[adata_urna_all.obs['cell_type'] == cell_type,:][:,:].copy()
    unseen_atac = unseen_atac_all[adata_uatac_all.obs['cell_type'] == cell_type,:][:,:].copy()

    # rmse
    rmse = np.sqrt(np.mean((unseen_rna - true_rna)**2, axis=1))
    df_res = df_res._append({'cell_type':cell_type,'method':method,'metrics':'rmse','modality':'rna','value':np.mean(rmse)},ignore_index=True)
    rmse = np.sqrt(np.mean((unseen_atac - true_atac)**2, axis=1))
    df_res = df_res._append({'cell_type':cell_type,'method':method,'metrics':'rmse','modality':'atac','value':np.mean(rmse)},ignore_index=True)

    # mean expression
    true_mean_rna = np.mean(true_rna, axis=0)
    true_mean_atac = np.mean(true_atac, axis=0)
    pred_mean_rna = np.mean(unseen_rna, axis=0)
    pred_mean_atac = np.mean(unseen_atac, axis=0)

    # rmse for mean expression
    rmse = np.sqrt(np.mean((pred_mean_rna - true_mean_rna)**2))
    df_res = df_res._append({'cell_type':cell_type,'method':method,'metrics':'rmse_mean','modality':'rna','value':rmse},ignore_index=True)
    rmse = np.sqrt(np.mean((pred_mean_atac - true_mean_atac)**2))
    df_res = df_res._append({'cell_type':cell_type,'method':method,'metrics':'rmse_mean','modality':'atac','value':rmse},ignore_index=True)

    # pearson r between mean expression of unseen and original RNA
    r_rna = pearsonr(pred_mean_rna, true_mean_rna)[0]
    df_res = df_res._append({'cell_type':cell_type,'method':method,'metrics':'pearsonr_mean','modality':'rna','value':r_rna},ignore_index=True)
    r_atac = pearsonr(pred_mean_atac, true_mean_atac)[0]
    df_res = df_res._append({'cell_type':cell_type,'method':method,'metrics':'pearsonr_mean','modality':'atac','value':r_atac},ignore_index=True)

    # spearman r between mean expression of unseen and original RNA
    r_rna = spearmanr(pred_mean_rna, true_mean_rna)[0]
    df_res = df_res._append({'cell_type':cell_type,'method':method,'metrics':'spearmanr_mean','modality':'rna','value':r_rna},ignore_index=True)
    r_atac = spearmanr(pred_mean_atac, true_mean_atac)[0]
    df_res = df_res._append({'cell_type':cell_type,'method':method,'metrics':'spearmanr_mean','modality':'atac','value':r_atac},ignore_index=True)

    plot_corrslation(true_mean_rna, pred_mean_rna, title=f"RNA_{cell_type}_{method}", savepath=f"RNA_{cell_type}_{method}.png")
    plot_corrslation(true_mean_atac, pred_mean_atac, title=f"ATAC_{cell_type}_{method}", savepath=f"ATAC_{cell_type}_{method}.png")


####
print("knn")
method = "knn"

unseen_rna_all = np.load("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/10x/predict_rna_mse_knn.npy")
unseen_atac_all = np.load("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/10x/predict_atac_mse_knn.npy")

cell_type = "all"
adata_urna = adata_urna_all[:,:].copy()
adata_uatac = adata_uatac_all[:,:].copy()
true_rna = to_dense_array(adata_urna.X)
true_atac = to_dense_array(adata_uatac.X)
unseen_rna = unseen_rna_all[:,:].copy()
unseen_atac = unseen_atac_all[:,:].copy()
# rmse
rmse = np.sqrt(np.mean((unseen_rna - true_rna)**2, axis=1))
df_res = df_res._append({'cell_type':cell_type,'method':method,'metrics':'rmse','modality':'rna','value':np.mean(rmse)},ignore_index=True)
rmse = np.sqrt(np.mean((unseen_atac - true_atac)**2, axis=1))
df_res = df_res._append({'cell_type':cell_type,'method':method,'metrics':'rmse','modality':'atac','value':np.mean(rmse)},ignore_index=True)
# mean expression
true_mean_rna = np.mean(true_rna, axis=0)   
true_mean_atac = np.mean(true_atac, axis=0)
pred_mean_rna = np.mean(unseen_rna, axis=0)
pred_mean_atac = np.mean(unseen_atac, axis=0)
# rmse for mean expression
rmse = np.sqrt(np.mean((pred_mean_rna - true_mean_rna)**2))
df_res = df_res._append({'cell_type':cell_type,'method': method,'metrics':'rmse_mean','modality':'rna','value':rmse},ignore_index=True)
rmse = np.sqrt(np.mean((pred_mean_atac - true_mean_atac)**2))
df_res = df_res._append({'cell_type':cell_type,'method': method,'metrics':'rmse_mean','modality':'atac','value':rmse},ignore_index=True)
# pearson r between mean expression of unseen and original RNA
r_rna = pearsonr(pred_mean_rna, true_mean_rna)[0]
df_res = df_res._append({'cell_type':cell_type,'method': method,'metrics':'pearsonr_mean','modality':'rna','value':r_rna},ignore_index=True)
r_atac = pearsonr(pred_mean_atac, true_mean_atac)[0]
df_res = df_res._append({'cell_type':cell_type,'method': method,'metrics':'pearsonr_mean','modality':'atac','value':r_atac},ignore_index=True)
# spearman r between mean expression of unseen and original RNA
r_rna = spearmanr(pred_mean_rna, true_mean_rna)[0]
df_res = df_res._append({'cell_type':cell_type,'method': method,'metrics':'spearmanr_mean','modality':'rna','value':r_rna},ignore_index=True)
r_atac = spearmanr(pred_mean_atac, true_mean_atac)[0]
df_res = df_res._append({'cell_type':cell_type,'method': method,'metrics':'spearmanr_mean','modality':'atac','value':r_atac},ignore_index=True)

plot_corrslation(true_mean_rna, pred_mean_rna, title=f"RNA_{cell_type}_{method}", savepath=f"RNA_{cell_type}_{method}.png")
plot_corrslation(true_mean_atac, pred_mean_atac, title=f"ATAC_{cell_type}_{method}", savepath=f"ATAC_{cell_type}_{method}.png")

for cell_type in celltypes:
    idx_rna = adata_urna_all[adata_urna_all.obs['cell_type'] == cell_type].obs_names
    idx_atac = adata_uatac_all[adata_uatac_all.obs['cell_type'] == cell_type].obs_names

    adata_urna = adata_urna_all[idx_rna,:].copy()
    adata_uatac = adata_uatac_all[idx_atac,:].copy()

    unseen_rna = unseen_rna_all[adata_urna_all.obs['cell_type'] == cell_type,:][:,:].copy()
    unseen_atac = unseen_atac_all[adata_uatac_all.obs['cell_type'] == cell_type,:][:,:].copy()

    true_rna = to_dense_array(adata_urna.X)
    true_atac = to_dense_array(adata_uatac.X)

    # rmse
    rmse = np.sqrt(np.mean((unseen_rna - true_rna)**2, axis=1))
    df_res = df_res._append({'cell_type':cell_type,'method':method,'metrics':'rmse','modality':'rna','value':np.mean(rmse)},ignore_index=True)
    rmse = np.sqrt(np.mean((unseen_atac - true_atac)**2, axis=1))
    df_res = df_res._append({'cell_type':cell_type,'method':method,'metrics':'rmse','modality':'atac','value':np.mean(rmse)},ignore_index=True)

    # mean expression
    true_mean_rna = np.mean(true_rna, axis=0)   
    true_mean_atac = np.mean(true_atac, axis=0)
    pred_mean_rna = np.mean(unseen_rna, axis=0)
    pred_mean_atac = np.mean(unseen_atac, axis=0)

    # rmse for mean expression
    rmse = np.sqrt(np.mean((pred_mean_rna - true_mean_rna)**2))
    df_res = df_res._append({'cell_type':cell_type,'method':method,'metrics':'rmse_mean','modality':'rna','value':rmse},ignore_index=True)
    rmse = np.sqrt(np.mean((pred_mean_atac - true_mean_atac)**2))
    df_res = df_res._append({'cell_type':cell_type,'method':method,'metrics':'rmse_mean','modality':'atac','value':rmse},ignore_index=True)

    # pearson r between mean expression of unseen and original RNA
    r_rna = pearsonr(pred_mean_rna, true_mean_rna)[0]
    df_res = df_res._append({'cell_type':cell_type,'method':method,'metrics':'pearsonr_mean','modality':'rna','value':r_rna},ignore_index=True)
    r_atac = pearsonr(pred_mean_atac, true_mean_atac)[0]
    df_res = df_res._append({'cell_type':cell_type,'method':method,'metrics':'pearsonr_mean','modality':'atac','value':r_atac},ignore_index=True)

    # spearman r between mean expression of unseen and original RNA
    r_rna = spearmanr(pred_mean_rna, true_mean_rna)[0]
    df_res = df_res._append({'cell_type':cell_type,'method':method,'metrics':'spearmanr_mean','modality':'rna','value':r_rna},ignore_index=True)
    r_atac = spearmanr(pred_mean_atac, true_mean_atac)[0]
    df_res = df_res._append({'cell_type':cell_type,'method':method,'metrics':'spearmanr_mean','modality':'atac','value':r_atac},ignore_index=True)

    plot_corrslation(true_mean_rna, pred_mean_rna, title=f"RNA_{cell_type}_{method}", savepath=f"RNA_{cell_type}_{method}.png")
    plot_corrslation(true_mean_atac, pred_mean_atac, title=f"ATAC_{cell_type}_{method}", savepath=f"ATAC_{cell_type}_{method}.png")

df_res.to_csv("./prediction_metrics_mse.csv", index=False)


top_celltypes = adata.obs.cell_type.value_counts().sort_values(ascending=False).head(5).index.tolist()
from PIL import Image
import math
def concat_grid(image_files, n_rows, n_cols, save_path="concat.png"):
    # 打开所有图片
    images = [Image.open(f) for f in image_files]
    w, h = images[0].size
    
    # 创建大画布
    new_img = Image.new("RGB", (n_cols * w, n_rows * h), (255, 255, 255))
    
    for i, img in enumerate(images):
        row = i // n_cols
        col = i % n_cols
        new_img.paste(img, (col * w, row * h))
    
    new_img.save(save_path)
    print(f"拼接完成，保存到 {save_path}")

methods = ["ot","knn"]
configs = ["all"] + top_celltypes
modalities = ["RNA","ATAC"]

image_files = [f"figures/{mod}_{cfg}_{method}.png"  for mod in modalities for method in methods for cfg in configs]
concat_grid(image_files, n_rows=len(methods)*len(modalities), n_cols=len(configs), save_path="results_grid_scatter.pdf")


############ UMAP ############

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import sys

### --- NEW: Import matplotlib and set vector fonts --- ###
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
### -------------------------------------------------- ###


adata = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/10x/feature_aligned.h5ad")
adata
print(adata)
rna_hvg = np.where(adata.var_names.isin(adata.uns['rna_hvg']))[0].tolist()
atac_hvg = np.where(adata.var_names.isin(adata.uns['atac_hvg']))[0].tolist()
feat = list(set(rna_hvg)|set(atac_hvg))
adata = adata[:,feat].copy()
print(adata)
adata.obs.modality = adata.obs.modality.map({"0":"RNA","1":"ATAC"})
adata.obs.cell_type = adata.obs.cell_type.str.replace(" ", "_")

adata_urna_all = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/10x/RNA_counts_unseen.h5ad")
adata_uatac_all = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/10x/ATAC_gene_unseen.h5ad")


featlist = adata.uns['feat'] 
adata_urna_all = adata_urna_all[:,featlist].copy()
adata_uatac_all = adata_uatac_all[:,featlist].copy()

adata_urna_pred = adata_urna_all.copy()
adata_uatac_pred = adata_uatac_all.copy()

### --- NEW: Create the 2x3 plot grid --- ###
n_rows = 2
n_cols = 3
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 8.5, n_rows * 5))
### ---------------------------------------- ###


# --- 1. KNN and True Data Processing ---
unseen_rna_all = np.load("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/10x/predict_rna_mse_knn.npy")
unseen_atac_all = np.load("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/10x/predict_atac_mse_knn.npy")
adata_urna_pred.X = unseen_rna_all.copy()
adata_uatac_pred.X = unseen_atac_all.copy()

# PCA, Neighbors, UMAP for True data
sc.pp.pca(adata_urna_all, n_comps=50)
sc.pp.pca(adata_uatac_all, n_comps=50)
sc.pp.neighbors(adata_urna_all, n_neighbors=15)
sc.pp.neighbors(adata_uatac_all, n_neighbors=15)
sc.tl.umap(adata_urna_all)
sc.tl.umap(adata_uatac_all)

# PCA, Neighbors, UMAP for KNN pred data
sc.pp.pca(adata_urna_pred, n_comps=50)
sc.pp.pca(adata_uatac_pred, n_comps=50)
sc.pp.neighbors(adata_urna_pred, n_neighbors=15)
sc.pp.neighbors(adata_uatac_pred, n_neighbors=15)
sc.tl.umap(adata_urna_pred)
sc.tl.umap(adata_uatac_pred)

print("Plotting True and KNN UMAPs...")

# Col 0: True
sc.pl.umap(adata_urna_all, color="cell_type", ax=axes[0, 0], show=False, title="RNA (true)")
sc.pl.umap(adata_uatac_all, color="cell_type", ax=axes[1, 0], show=False, title="ATAC (true)")

# Col 2: KNN
sc.pl.umap(adata_urna_pred, color="cell_type", ax=axes[0, 2], show=False, title="RNA (KNN prediction)")
sc.pl.umap(adata_uatac_pred, color="cell_type", ax=axes[1, 2], show=False, title="ATAC (KNN prediction)")
### ------------------------------------------------- ###


# --- 2. OT Data Processing ---
unseen_rna_all = np.load("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/10x/predict_rna_mse_ot.npy")
unseen_atac_all = np.load("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/10x/predict_atac_mse_ot.npy")
adata_urna_pred.X = unseen_rna_all.copy()
adata_uatac_pred.X = unseen_atac_all.copy()

# PCA, Neighbors, UMAP for OT pred data
sc.pp.pca(adata_urna_pred, n_comps=50)
sc.pp.pca(adata_uatac_pred, n_comps=50)
sc.pp.neighbors(adata_urna_pred, n_neighbors=15)
sc.pp.neighbors(adata_uatac_pred, n_neighbors=15)
sc.tl.umap(adata_urna_pred)
sc.tl.umap(adata_uatac_pred)

print("Plotting OT UMAPs...")
### --- MODIFIED: Plot on the grid (OT) --- ###
# Col 1: OT
sc.pl.umap(adata_urna_pred, color="cell_type", ax=axes[0, 1], show=False, title="RNA (OT prediction)")
sc.pl.umap(adata_uatac_pred, color="cell_type", ax=axes[1, 1], show=False, title="ATAC (OT prediction)")
### ------------------------------------------- ###


print("All UMAPs drawn. Saving the combined vector PDF...")


fig.tight_layout()

fig.savefig("umap_for_all.pdf")
plt.close(fig) 

print("Combined UMAP PDF saved successfully.")
### -------------------------------------------- ###
