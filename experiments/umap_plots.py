import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import scanpy as sc 
import anndata as ad
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection


import os
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
from functools import partial
from typing import Any
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap
from plottable.plots import bar
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

import scib_metrics
from scib_metrics.nearest_neighbors import NeighborsResults, pynndescent

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42 
mpl.rcParams['ps.fonttype'] = 42 
mpl.rcParams['font.family'] = 'Arial'

sc.set_figure_params(dpi=300,dpi_save=300)

import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from PIL import Image

import os
os.chdir("/home/bingxing2/ailab/group/ai4bio/sunjianle/integration/plots/")


### BMMC
adata = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/BMMC/feature_aligned_trained_sampled.h5ad")
adata.obs.modality = adata.obs.modality.astype('str')
adata.obs.loc[adata.obs.modality=="0",'modality'] = "RNA"
adata.obs.loc[adata.obs.modality=="1",'modality'] = "ATAC"
adata.obs.loc[adata.obs.modality=="2",'modality'] = "Protein"
adata.obsm['Ours'] = adata.obsm['latent_shared'].copy()

glue = np.load("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/BMMC/glue_sampled.npy")
adata.obsm['GLUE'] = glue.copy()
scvi = np.load("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/BMMC/scvi.npy")
adata.obsm['scVI'] = scvi.copy()
harmony = pd.read_csv("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/BMMC/harmony.csv")
adata.obsm['Harmony'] = harmony.iloc[:,1:].to_numpy()
adata.obs['cell_type'] = adata.obs.celltype.copy()

methods = ['Ours', 'GLUE', 'scVI', 'Harmony']
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

fig, axs = plt.subplots(len(methods), 1, figsize=(25, 30)) 

for i, method in enumerate(methods):
    sc.pp.neighbors(adata, use_rep=method)
    sc.tl.umap(adata, min_dist=0.3)
    sc.pl.umap(adata, color=["cell_type","modality","batch"],size=3, wspace=0.5,show=False, save=f'_{method}.pdf')
    sc.pl.umap(adata, color=["cell_type","modality","batch"],size=3, wspace=0.5,show=False, save=f'_{method}.png')
    img = Image.open(f'figures/umap_{method}.png')

    axs[i].imshow(img)
    axs[i].axis('off')  # 关掉坐标轴

    axs[i].text(-0.01, 0.5, method, transform=axs[i].transAxes,
                fontsize=12, fontweight='bold', va='center', ha='right', rotation=0)
    axs[i].text(-0.05, 0.95, f"({letters[i]})", transform=axs[i].transAxes,
                fontsize=12, fontweight='bold', va='top', ha='left', color='black')

plt.tight_layout()
fig.savefig('BMMC_merged.pdf', bbox_inches='tight')
plt.show()


##### yao
adata = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao/sensitivity/feature_aligned_1_5_5.h5ad")
adata.obs.modality = adata.obs.modality.astype('str')
adata.obs.loc[adata.obs.modality=="0",'modality'] = "RNA"
adata.obs.loc[adata.obs.modality=="1",'modality'] = "ATAC"
adata.obsm['Ours'] = adata.obsm['latent_shared'].copy()
glue = np.load("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao/glue_nb_hvg.npy")
adata.obsm['GLUE_lsi'] = glue.copy()
glue = np.load("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao/glue_nb_hvg_pca.npy")
adata.obsm['GLUE_pca'] = glue.copy()
scvi = np.load("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao/scvi.npy")
adata.obsm['scVI'] = scvi.copy()
seurat = pd.read_csv("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao/seurat_pca.csv")
adata.obsm['Seurat'] = seurat.iloc[:,1:].to_numpy()
harmony = pd.read_csv("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao/harmony.csv")
adata.obsm['Harmony'] = harmony.iloc[:,1:].to_numpy()
simba1 = pd.read_csv("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao/Yao-SIMBA-RNA-latent.csv")
simba2 = pd.read_csv("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao/Yao-SIMBA-ATAC-latent.csv")
adata.obsm['SIMBA'] = pd.concat([simba1,simba2],axis=0).iloc[:,1:].to_numpy()
maxfuse1 = pd.read_csv("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao/Yao-MaxFuse-RNA-latent.csv")
maxfuse2 = pd.read_csv("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao/Yao-MaxFuse-ATAC-latent.csv")
adata.obsm['MaxFuse'] = pd.concat([maxfuse1, maxfuse2],axis=0).iloc[:,1:].to_numpy()


methods = ['Ours','Seurat','GLUE_lsi','GLUE_pca','scVI','Harmony','MaxFuse','SIMBA']
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

fig, axs = plt.subplots(len(methods), 1, figsize=(25, 30)) 

for i, method in enumerate(methods):
    sc.pp.neighbors(adata, use_rep=method)
    sc.tl.umap(adata, min_dist=0.3)
    sc.pl.umap(adata, color=["cell_type","modality","batch"],size=3, wspace=0.5,show=False, save=f'_{method}.pdf')
    sc.pl.umap(adata, color=["cell_type","modality","batch"],size=3, wspace=0.5,show=False, save=f'_{method}.png')
    img = Image.open(f'figures/umap_{method}.png')

    axs[i].imshow(img)
    axs[i].axis('off')  # 关掉坐标轴

    axs[i].text(-0.01, 0.5, method, transform=axs[i].transAxes,
                fontsize=12, fontweight='bold', va='center', ha='right', rotation=0)
    axs[i].text(-0.05, 0.95, f"({letters[i]})", transform=axs[i].transAxes,
                fontsize=12, fontweight='bold', va='top', ha='left', color='black')

plt.tight_layout()
fig.savefig('yao_merged.pdf', bbox_inches='tight')
plt.show()

#### muto
adata = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/sensitivity/feature_aligned_5_5_5.h5ad")
adata.obs.modality = adata.obs.modality.astype('str')
adata.obs.loc[adata.obs.modality=="0",'modality'] = "RNA"
adata.obs.loc[adata.obs.modality=="1",'modality'] = "ATAC"
adata.obsm['Ours'] = adata.obsm['latent_shared'].copy()
glue = np.load("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/glue.npy")
adata.obsm['GLUE'] = glue.copy()
scvi = np.load("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/scvi.npy")
adata.obsm['scVI'] = scvi.copy()
seurat = pd.read_csv("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/seurat_pca.csv")
adata.obsm['Seurat'] = seurat.iloc[:,1:].to_numpy()
harmony = pd.read_csv("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/harmony.csv")
adata.obsm['Harmony'] = harmony.iloc[:,1:].to_numpy()
jamie = np.load("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/jamie.npy")
adata.obsm['JAMIE'] = jamie.copy() #.reshape(-1, jamie.shape[-1])
pamona = np.load("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/pamona.npy")
adata.obsm['Pamona'] = pamona.copy()
union = np.load("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/unioncom.npy")
adata.obsm['UnionCom'] = union.copy()
simba1 = pd.read_csv("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/muto-SIMBA-RNA-latent.csv")
simba2 = pd.read_csv("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/muto-SIMBA-ATAC-latent.csv")
adata.obsm['SIMBA'] = pd.concat([simba1,simba2],axis=0).iloc[:,1:].to_numpy()
maxfuse1 = pd.read_csv("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/muto-MaxFuse-RNA-latent.csv")
maxfuse2 = pd.read_csv("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/muto-MaxFuse-ATAC-latent.csv")
adata.obsm['MaxFuse'] = pd.concat([maxfuse1, maxfuse2],axis=0).iloc[:,1:].to_numpy()

original_labels = adata.obs['batch'].unique()
label_mapping = {label: f'batch{i+1}' for i, label in enumerate(original_labels)}
adata.obs['batch'] = adata.obs['batch'].map(label_mapping)

methods = ['Ours','GLUE','scVI','MaxFuse','Seurat','Pamona','JAMIE','SIMBA','Harmony','UnionCom']
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h','i','j','k','l','m']

fig, axs = plt.subplots(len(methods), 1, figsize=(25, 30)) 

for i, method in enumerate(methods):
    sc.pp.neighbors(adata, use_rep=method)
    sc.tl.umap(adata, min_dist=0.3)
    sc.pl.umap(adata, color=["cell_type","modality","batch"],size=3, wspace=0.5,show=False, save=f'_{method}.pdf')
    sc.pl.umap(adata, color=["cell_type","modality","batch"],size=3, wspace=0.5,show=False, save=f'_{method}.png')
    img = Image.open(f'figures/umap_{method}.png')

    axs[i].imshow(img)
    axs[i].axis('off')  # 关掉坐标轴

    axs[i].text(-0.01, 0.5, method, transform=axs[i].transAxes,
                fontsize=12, fontweight='bold', va='center', ha='right', rotation=0)
    axs[i].text(-0.05, 0.95, f"({letters[i]})", transform=axs[i].transAxes,
                fontsize=12, fontweight='bold', va='top', ha='left', color='black')

plt.tight_layout()
fig.savefig('muto_merged.pdf', bbox_inches='tight')
plt.show()
