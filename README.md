# scMRDR

We implement a scalable and flexible generative framework called single-cell Multi-omics Regularized Disentangled Representations (scMRDR) for unpaired multi-omics integration. The manuscript has been accepted as a **spotlight** paper on The Thirty-ninth Annual Conference on Neural Information Processing Systems (NeurIPS 2025).

* Free software: GPL-3.0 License
* Documentation: https://sjl-sjtu.github.io/scMRDR/

## Tutorials

#### Installation
```
git clone https://github.com/sjl-sjtu/scMRDR.git
cd scMRDR
pip install -e .
```

#### Examples
```
import scanpy as sc
import anndata as ad
from scMRDR.module import Integration

rna = sc.read_h5ad("rna_processed.h5ad") # h5ad file of scRNA (after preprocessing)
atac_gas = sc.read_h5ad("atac_gas_processed.h5ad") # h5ad file of gene activity score from scATAC (after preprocessing)
rna.obs.modality == "rna"; atac_gas.obs.modality="atac"
rna_hvg = rna.var_names[rna.var['highly_variable']]; atac_hvg = atac.var_names[atac.var['highly_variable']]
adata = ad.concat([rna[:,rna_hvg].copy(),atatc_gas[:,atac_hvg].copy()], axis='obs', join='inner', label="modality") # an adata concated with different omics (as different observations). If you want to use masked features, you can specify join='outer' here and specify feature_list for each modality then
model = Integration(data=adata, modality_key="modality", layer="count", batch_key="batch",
                    feature_list=None, distribution="ZINB") # If we model the count data (stored in adata.layers['count']) with ZINB model, with omics information stored in adata.obs.modality, batch information stored in adata.obs.batch
# If you want to use masked features to handle mismatched features in different omics
# adata = ad.concat([rna[:,rna_hvg].copy(),atatc_gas[:,atac_hvg].copy()], axis='obs', join='outer', label="modality", fill_value=0)
# feature_list = {"rna":rna_hvg,"atac":atac_hvg}
# model = Integration(data=adata, modality_key="modality", layer="count", batch_key="batch", feature_list=feature_list, mask_key="modality", distribution="ZINB")
model.setup(hidden_layers = [512,512], latent_dim_shared = 20, latent_dim_specific=20, 
            beta = 2, gamma = 5, lambda_adv = 5, dropout_rate=0.2)
model.train(epoch_num = 200, batch_size = 128, lr = 1e-3, adaptlr = False, num_warmup = 0,
            early_stopping = True, valid_prop = 0.1, weighted=False, patience=10)
model.inference(n_samples=1,update=True,returns=False)
adata = model.get_adata() # The integrated embeddings are stored in adata.obsm["latent_shared"]

# visualization
sc.pp.neighbors(adata,use_rep="latent_shared")
sc.tl.umap(adata)
sc.pl.umap(
    adata,
    color=["modality","cell_type","batch"],
    size=2, wspace=0.5
)
```

## Citation
Jianle Sun, Chaoqi Liang, Ran Wei, Peng Zheng, LEI BAI, Wanli Ouyang, Hongliang Yan, Peng Ye. scMRDR: A scalable and flexible framework for unpaired single-cell multi-omics data integration. The Thirty-ninth Annual Conference on Neural Information Processing Systems (NeurIPS), 2025.

## Contact
Feel free to contact me via jianles@andrew.cmu.edu or sjl-2017@alumni.sjtu.edu.cn if you have any questions.
