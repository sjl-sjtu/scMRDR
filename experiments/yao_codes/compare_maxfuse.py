import os
import sys
import json
import glob

# import metrics
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.io import mmread
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import maxfuse as mf

# https://github.com/shuxiaoc/maxfuse/tree/main/Archive/strong-link/10xpbmc/method_running
# https://github.com/shuxiaoc/maxfuse/blob/main/Archive/strong-link/10xpbmc/method_running/mf_pbmc.ipynb

# %%
# def MaxFuse_module(input_path,
#                    output_path,
#                    config
#                   ):

input_path = "/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao"
output_path = "/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao"
config = {'rna_h5ad_filename':"RNA_counts_qc.h5ad",
                       'atac_h5ad_filename':"ATAC_counts_qc.h5ad",
                       'gam_h5ad_filename':'ATAC_gas.h5ad',
                       'output_prefix':'Yao'}

# Make Dir
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Load Metadata
# metadata = pd.read_csv(os.path.join(input_path, config['metadata']), header=0)
# metadata.index = metadata[config['barcode_key']].values

# Load scRNA_SCTransform
# rna = sc.read(os.path.join(output_path, 'scRNA_SCTransform.csv.gz'))
# rna.obs = metadata
rna = sc.read_h5ad(os.path.join(input_path, config['rna_h5ad_filename']))

# Load Gene_Activity_Matrix
gam = sc.read_h5ad(os.path.join(input_path, config['gam_h5ad_filename']))

# Load ATAC_LSI
# atac = sc.read(os.path.join(output_path, 'scATAC_LSI.csv.gz'))
# atac.obs = metadata
atac = sc.read_h5ad(os.path.join(input_path, config['atac_h5ad_filename']))
from sklearn.decomposition import TruncatedSVD
lsi = TruncatedSVD(n_components=100)
atac.obsm['X_lsi'] = lsi.fit_transform(atac.X.toarray())

# # convert format to sparse matrix
# rna.X=scipy.sparse.csr_matrix(rna.X)
# gam.X=scipy.sparse.csr_matrix(gam.X)
# atac.X=scipy.sparse.csr_matrix(atac.X)

# subset shared features
shared_features = np.intersect1d(rna.var_names, gam.var_names)
rna_shared = rna[:, shared_features].copy()
gam_shared = gam[:, shared_features].copy()

# process rna_shared
## skip normalization: input is SCTransformed
sc.pp.normalize_total(rna_shared)
sc.pp.log1p(rna_shared)
sc.pp.highly_variable_genes(rna_shared, n_top_genes = 3000)
sc.pp.scale(rna_shared)

# process atac_shared
sc.pp.normalize_total(gam_shared)
sc.pp.log1p(gam_shared)
sc.pp.scale(gam_shared)

## only retain highly variable genes
HVGs = rna_shared.var.highly_variable

# all features
rna_active = rna_shared[:,HVGs].X
atac_active = atac.obsm['X_lsi'].copy() #atac.X

# shared features
rna_shared = rna_shared[:,HVGs].X
gam_shared = gam_shared[:,HVGs].X


spm = mf.model.Fusor(
        shared_arr1=rna_shared,
        shared_arr2=gam_shared,
        active_arr1=rna_active,
        active_arr2=atac_active,
        # method='centroid_shrinkage',
        labels1=None, # if None, then use scanpy clustering pipeline
        labels2=None
    )

spm.split_into_batches(
        max_outward_size=5000,
        matching_ratio=5,
        metacell_size=2,
        method='binning',
        verbose=True,
        seed=42
    )

#     spm.plot_singular_values(
#         target='active_arr1',
#         batch=0,
#         n_components=None,
#         randomized_svd=False,  # @Shuxiao: Had to change this from True to False.  Doesn't work when true, why?
#         svd_runs=1
#     )

#     spm.plot_singular_values(
#         target='active_arr2',
#         batch=0,
#         n_components=None,
#         randomized_svd=False,  # @Shuxiao: Had to change this from True to False.  Doesn't work when true, why?
#         svd_runs=1
#     )

spm.construct_graphs(
    n_neighbors1=15,
    n_neighbors2=15,
    svd_components1=30,
    svd_components2=15,
    resolution1=2,
    resolution2=2,
    randomized_svd=False, 
    svd_runs=1,
    resolution_tol=0.1,
    leiden_runs=1,
    leiden_seed=None,
    verbose=True
)

spm.find_initial_pivots(
    wt1=0.7, wt2=0.7,
    svd_components1=20, svd_components2=20,
    randomized_svd=False, svd_runs=1,
    verbose=True
)

spm.refine_pivots(
    wt1=0.7, wt2=0.7,
#         svd_components1=200, svd_components2=None,
    svd_components1=100, svd_components2=None, # c500
    cca_components=20,
    filter_prop=0.,
    n_iters=8,
    randomized_svd=False, 
    svd_runs=1,
    verbose=True
)

spm.filter_bad_matches(target='pivot', filter_prop=0.4, verbose=True)

spm.propagate(
    wt1=0.7,
    wt2=0.7,
    svd_components1=30, 
    svd_components2=None, 
    randomized_svd=False, 
    svd_runs=1, 
    verbose=True
)

# filters away unreliable pivots
spm.filter_bad_matches(
        target='propagated',
        filter_prop=0.,
        verbose=True
    )

# get joint embedding
rna_cca, atac_cca = spm.get_embedding(
        active_arr1 = spm.active_arr1,
        active_arr2 = spm.active_arr2,
        refit=False,
        matching=None,
        order=None,
        cca_components=20,
        cca_max_iter=None
    )

dim_use = 15 # dimensions of the CCA embedding to be used for UMAP etc

# mf.metrics.get_foscttm(
#     dist=mf.utils.cdist_correlation(rna_cca[:,:dim_use], atac_cca[:,:dim_use]),
#     true_matching='identity'
# )

#     # UMAP
#     cca_adata = ad.AnnData(
#         np.concatenate((rna_cca[:,:dim_use], atac_cca[:,:dim_use]), axis=0), 
#         dtype=np.float32
#     )
#     cca_adata.obs['data_type'] = ['rna'] * rna_cca.shape[0] + ['atac'] * atac_cca.shape[0]
#     cell_type = rna.obs['cell_type'].to_numpy()
#     cca_adata.obs['cell_type'] = list(cell_type) * 2
#     sc.pp.neighbors(cca_adata, n_neighbors=15)
#     sc.tl.umap(cca_adata)
#     # sc.pl.umap(cca_adata, color='data_type')

#     # save UMAP
#     ## RNA
#     rna_adata = cca_adata[cca_adata.obs['data_type'] == 'rna']
#     rna_umap = pd.DataFrame(data=rna_adata.obsm['X_umap'], columns=["UMAP1","UMAP2"] , index=rna.obs.index)
#     rna_umap.to_csv(os.path.join(output_path, config["output_prefix"] + "-MaxFuse-RNA-umap.csv"))

#     ## ATAC
#     atac_adata = cca_adata[cca_adata.obs['data_type'] == 'atac']
#     atac_umap = pd.DataFrame(data=atac_adata.obsm['X_umap'], columns=["UMAP1","UMAP2"] , index=atac.obs.index)
#     atac_umap.to_csv(os.path.join(output_path, config["output_prefix"] + "-MaxFuse-ATAC-umap.csv"))

# save latent
## RNA
rna_latent = pd.DataFrame(data=rna_cca, index=rna.obs.index)
rna_latent.to_csv(os.path.join(output_path, config["output_prefix"] + "-MaxFuse-RNA-latent.csv"))

## ATAC
atac_latent = pd.DataFrame(data=atac_cca, index=atac.obs.index)
atac_latent.to_csv(os.path.join(output_path, config["output_prefix"] + "-MaxFuse-ATAC-latent.csv"))

    # # multi
    # multi_latent = pd.merge(rna_latent, atac_latent, left_index=True, right_index=True, how='outer')
    # multi_latent.to_csv(os.path.join(output_path, config["output_prefix"] + "-MaxFuse-multi-latent.csv"))


# %%
# input_path = sys.argv[1]
# output_path = sys.argv[2]
# config = json.load(open(sys.argv[3]))

# %%
# sys.path.insert(1, os.path.join(config['utils_path'], 'MaxFuse_devo_09302022V'))
# import match

# r_env = '/home/wsg/software/miniconda3/envs/r4/bin/Rscript'
# r_script = os.path.join(config['utils_path'], 'Get_SCTransform_and_LSI.R')

# r_command = f"{r_env} {r_script} {input_path} {output_path} {sys.argv[3]}"
# os.system(r_command)

# %%
# MaxFuse_module(input_path = "/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao", 
#                output_path = "/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao",
#                config = {'rna_h5ad_filename':"RNA_counts_qc.h5ad",
#                        'atac_h5ad_filename':"ATAC_counts_qc.h5ad",
#                        'gam_h5ad_filename':'ATAC_gas_qc.h5ad',
#                        'output_prefix':'Yao'}
#               )
