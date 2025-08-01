import os
import sys
import json
import glob
import shutil

import simba as si
import scanpy as sc
import pandas as pd
import numpy as np

# def SIMBA_module(input_path,
#                  output_path,
#                  config
#                 ):

if __name__ == '__main__':
    input_path = "/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao"
    output_path = "/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao"
    config = {'rna_h5ad_filename':"RNA_counts_qc.h5ad",
                        'atac_h5ad_filename':"ATAC_counts_qc.h5ad",
                        'specie':'mouse',
                        'output_prefix':'Yao'}
        # Make Dir
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Load data
    adata_CG = sc.read_h5ad(os.path.join(input_path, config['rna_h5ad_filename']))
    adata_CP = sc.read_h5ad(os.path.join(input_path, config['atac_h5ad_filename']))

    # gam = sc.read_h5ad(os.path.join(input_path, config['gam_h5ad_filename']))

    si.settings.set_workdir(output_path)

    # rename index
    adata_CP.obs.index = adata_CP.obs.index + '_atac'
    adata_CG.obs.index = adata_CG.obs.index + '_rna'

    # add [chr, start, end] to var
    chr_start_end = pd.Series(adata_CP.var.index).str.split('[:-]', expand=True)
    chr_start_end.columns = ['chr','start','end']
    chr_start_end.index = adata_CP.var.index.copy()
    adata_CP.var = pd.concat([adata_CP.var, chr_start_end], axis=1)
    adata_CP.obs.head()

    # preprocessing: ATAC
    si.pp.filter_peaks(adata_CP, min_n_cells=3)
    si.pp.cal_qc_atac(adata_CP)
    # # visulization
    # si.pl.violin(adata_CP,list_obs=['n_counts','n_peaks','pct_peaks'], list_var=['n_cells'])
    # si.pl.hist(adata_CP,list_obs=['n_counts','n_peaks','pct_peaks'], log=True, list_var=['n_cells'])
    # # Filter out cells if needed:
    # si.pp.filter_cells_atac(adata_CP, min_n_peaks=100)
    # select peaks (optional)
    si.pp.pca(adata_CP, n_components=50)
    # si.pl.pca_variance_ratio(adata_CP)
    # The number of selected PCs can be further reduced based on the elbow plot
    si.pp.select_pcs(adata_CP, n_pcs=40)
    si.pp.select_pcs_features(adata_CP)
    # si.pl.pcs_features(adata_CP, fig_size=(3,3))

    # preprocessing: RNA
    si.pp.filter_genes(adata_CG, min_n_cells=3)
    si.pp.cal_qc_rna(adata_CG)
    si.pp.normalize(adata_CG, method='lib_size')
    si.pp.log_transform(adata_CG)
    # Select HVGs
    # si.pp.select_variable_genes(adata_CG, n_top_genes=4000) 
    # si.pp.select_variable_genes(adata_CG, n_top_genes=6000) # R10/A10 需要提高HVG数量，否则报错
    # # visulization
    # si.pl.variable_genes(adata_CG,show_texts=True)
    # # discretize RNA expression
    #     si.tl.discretize(adata_CG, n_bins=5)
    # si.pl.discretize(adata_CG, kde=False)

    if config["specie"] == "human":
        genome = "hg38"
    elif config["specie"] == "mouse":
        genome = "mm10"

    # Infer edges between cells of different modalities
    #     adata_CG_atac = si.tl.gene_scores(adata_CP, genome=genome, use_gene_weigt=True, use_top_pcs=True)
    adata_CG_atac = si.tl.gene_scores(adata_CP, genome=genome, use_gene_weigt=True, use_top_pcs=True)


    si.pp.filter_genes(adata_CG_atac, min_n_cells=3)
    si.pp.cal_qc_rna(adata_CG_atac)
    si.pp.normalize(adata_CG_atac, method='lib_size')
    si.pp.log_transform(adata_CG_atac)

    # RNA: Select HVGs
    # for n_genes in [4000,6000,8000, 10000, 15000, 20000, len(adata_CG.var_names)]: # 4000,6000,
    #     try:
    #         si.pp.select_variable_genes(adata_CG, n_top_genes=n_genes) 
    #         shared_feature1 = adata_CG.var_names[adata_CG.var["highly_variable"]].intersection(adata_CG_atac.var_names)
    #         if len(shared_feature1) >100:
    #             if any(adata_CG_atac[:,shared_feature1].X.sum(axis=1)==0):
    #                 pass
    #             else:
    #                 break
    #     except:
    #         continue

    n_genes = 4000
    si.pp.select_variable_genes(adata_CG, n_top_genes=n_genes) 
    shared_feature = adata_CG.var_names[adata_CG.var["highly_variable"]].intersection(adata_CG_atac.var_names)
    si.tl.discretize(adata_CG, n_bins=5)

    adata_CrnaCatac = si.tl.infer_edges(adata_CG, adata_CG_atac, n_components=15, k=15)

    # adata_CrnaCatac
    # si.pl.node_similarity(adata_CrnaCatac, cutoff=0.5)
    # si.pl.svd_nodes(adata_CrnaCatac,
    #                 color=['cell_type'],
    #                 size=3,
    #                 cutoff=0.5,
    #                 fig_legend_ncol=2)

    # edges can be futhere trimmed if needed. Here we keep all of them
    si.tl.trim_edges(adata_CrnaCatac, cutoff=0.5)

    # generate Graph
    si.tl.gen_graph(list_CP=[adata_CP],
                    list_CG=[adata_CG],
                    list_CC=[adata_CrnaCatac],
                    copy=False,
                    use_highly_variable=True,
                    use_top_pcs=True,
                    dirname='graph0')

    # PBG training
    si.settings.pbg_params

    dict_config = si.settings.pbg_params.copy()
    # dict_config['num_gpus'] = 1 # gpu
    dict_config['workers'] = 10 # multi-threading

    # start training
    si.tl.pbg_train(pbg_params = dict_config, auto_wd=True, save_wd=True, output='model')

    # load in graph ('graph0') info
    si.load_graph_stats()
    # load in model info for ('graph0')
    si.load_pbg_config()

    # Post-training Analysis
    dict_adata = si.read_embedding()
    # dict_adata

    # dict_adata
    adata_C = dict_adata['C']  # embeddings for ATACseq cells
    adata_C2 = dict_adata['C2']  # embeddings for RNAseq cells
    adata_G = dict_adata['G']  # embeddings for genes
    adata_P = dict_adata['P']  # embeddings for peaks

    # save latent
    rna_latent = pd.DataFrame(data=adata_C2.X, index=adata_C2.obs.index)
    atac_latent = pd.DataFrame(data=adata_C.X, index=adata_C.obs.index)

    ## rename latent index
    def replace_last(string, old, new):
        return new.join(string.rsplit(old, 1))
    rna_latent.index = [replace_last(barcode, '_rna', '') for barcode in rna_latent.index]
    atac_latent.index = [replace_last(barcode, '_atac', '') for barcode in atac_latent.index]
    # multi_latent = pd.merge(rna_latent, atac_latent, left_index=True, right_index=True, how='outer')

    # Save Results
    rna_latent.to_csv(os.path.join(output_path, config["output_prefix"] + "-SIMBA-RNA-latent.csv"))
    atac_latent.to_csv(os.path.join(output_path, config["output_prefix"] + "-SIMBA-ATAC-latent.csv"))

    shutil.rmtree("{}/pbg".format(output_path))

    # # %%
    # SIMBA_module(input_path = "/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao", 
    #              output_path = "/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/Yao",
    #              config = {'rna_h5ad_filename':"RNA_counts_qc.h5ad",
    #                        'atac_h5ad_filename':"ATAC_counts_qc.h5ad",
    #                        'specie':'human',
    #                        'output_prefix':'Yao'} # json.load(open(sys.argv[3]))
    #             )
