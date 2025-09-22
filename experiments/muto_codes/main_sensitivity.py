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

import sys
sys.path.insert(1, '/home/bingxing2/ailab/group/ai4bio/sunjianle/integration/models2')
from module import Integration

# adata = sc.read_h5ad("/ailab/user/sunjianle-hdd/integration27/BMMC/data2/feature_aligned_unpaired.h5ad")
# atac = sc.read("/ailab/user/sunjianle-hdd/integration27/BMMC/data2/ATAC_counts_qc_slt.h5ad")
# rna = sc.read("/ailab/user/sunjianle-hdd/integration27/BMMC/data2/RNA_counts_qc_slt.h5ad")
# rna = adata[adata.obs.modality=="0",:][rna.obs_names,:]
# atac = adata[adata.obs.modality=="1",:][atac.obs_names,:]
# adata = ad.concat([rna,atac], join='inner')
# adata.write("/ailab/user/sunjianle-hdd/integration27/BMMC/data2/feature_aligned_unpaired.h5ad")

beta, gamma, lambda_adv = 1, 0, 0
adata = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/feature_aligned.h5ad")
adata
rna_hvg = np.where(adata.var_names.isin(adata.uns['rna_hvg']))[0].tolist()
atac_hvg = np.where(adata.var_names.isin(adata.uns['atac_hvg']))[0].tolist()
feature_list = {"0":rna_hvg,"1":atac_hvg}
model = Integration(data=adata, layer="counts", modality_key="modality", batch_key="batch", 
                    feature_list=None, distribution="ZINB") #feature_list)
model.setup(hidden_layers = [500,500], latent_dim_shared = 20, latent_dim_specific=20, 
            beta = beta, gamma = gamma, lambda_adv = lambda_adv, dropout_rate=0.2)
model.train(epoch_num = 100, batch_size = 128, lr = 1e-3, adaptlr = False, num_warmup = 0,
            early_stopping = True, valid_prop = 0.1, patience = 25)
model.inference(n_samples=1,update=True,returns=False)
adata = model.get_adata()
adata.write(f"/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/sensitivity/feature_aligned_{beta}_{gamma}_{lambda_adv}.h5ad")

beta, gamma, lambda_adv = 1, 5, 5
adata = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/feature_aligned.h5ad")
adata
rna_hvg = np.where(adata.var_names.isin(adata.uns['rna_hvg']))[0].tolist()
atac_hvg = np.where(adata.var_names.isin(adata.uns['atac_hvg']))[0].tolist()
feature_list = {"0":rna_hvg,"1":atac_hvg}
model = Integration(data=adata, layer="counts", modality_key="modality", batch_key="batch", 
                    feature_list=None, distribution="ZINB") #feature_list)
model.setup(hidden_layers = [500,500], latent_dim_shared = 20, latent_dim_specific=20, 
            beta = beta, gamma = gamma, lambda_adv = lambda_adv, dropout_rate=0.2)
model.train(epoch_num = 100, batch_size = 128, lr = 1e-3, adaptlr = False, num_warmup = 0,
            early_stopping = True, valid_prop = 0.1, patience = 25)
model.inference(n_samples=1,update=True,returns=False)
adata = model.get_adata()
adata.write(f"/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/sensitivity/feature_aligned_{beta}_{gamma}_{lambda_adv}.h5ad")

beta, gamma, lambda_adv = 2, 5, 5
adata = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/feature_aligned.h5ad")
adata
rna_hvg = np.where(adata.var_names.isin(adata.uns['rna_hvg']))[0].tolist()
atac_hvg = np.where(adata.var_names.isin(adata.uns['atac_hvg']))[0].tolist()
feature_list = {"0":rna_hvg,"1":atac_hvg}
model = Integration(data=adata, layer="counts", modality_key="modality", batch_key="batch", 
                    feature_list=None, distribution="ZINB") #feature_list)
model.setup(hidden_layers = [500,500], latent_dim_shared = 20, latent_dim_specific=20, 
            beta = beta, gamma = gamma, lambda_adv = lambda_adv, dropout_rate=0.2)
model.train(epoch_num = 100, batch_size = 128, lr = 1e-3, adaptlr = False, num_warmup = 0,
            early_stopping = True, valid_prop = 0.1, patience = 25)
model.inference(n_samples=1,update=True,returns=False)
adata = model.get_adata()
adata.write(f"/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/sensitivity/feature_aligned_{beta}_{gamma}_{lambda_adv}.h5ad")

beta, gamma, lambda_adv = 3, 5, 5
adata = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/feature_aligned.h5ad")
adata
rna_hvg = np.where(adata.var_names.isin(adata.uns['rna_hvg']))[0].tolist()
atac_hvg = np.where(adata.var_names.isin(adata.uns['atac_hvg']))[0].tolist()
feature_list = {"0":rna_hvg,"1":atac_hvg}
model = Integration(data=adata, layer="counts", modality_key="modality", batch_key="batch", 
                    feature_list=None, distribution="ZINB") #feature_list)
model.setup(hidden_layers = [500,500], latent_dim_shared = 20, latent_dim_specific=20, 
            beta = beta, gamma = gamma, lambda_adv = lambda_adv, dropout_rate=0.2)
model.train(epoch_num = 100, batch_size = 128, lr = 1e-3, adaptlr = False, num_warmup = 0,
            early_stopping = True, valid_prop = 0.1, patience = 25)
model.inference(n_samples=1,update=True,returns=False)
adata = model.get_adata()
adata.write(f"/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/sensitivity/feature_aligned_{beta}_{gamma}_{lambda_adv}.h5ad")

beta, gamma, lambda_adv = 4, 5, 5
adata = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/feature_aligned.h5ad")
adata
rna_hvg = np.where(adata.var_names.isin(adata.uns['rna_hvg']))[0].tolist()
atac_hvg = np.where(adata.var_names.isin(adata.uns['atac_hvg']))[0].tolist()
feature_list = {"0":rna_hvg,"1":atac_hvg}
model = Integration(data=adata, layer="counts", modality_key="modality", batch_key="batch", 
                    feature_list=None, distribution="ZINB") #feature_list)
model.setup(hidden_layers = [500,500], latent_dim_shared = 20, latent_dim_specific=20, 
            beta = beta, gamma = gamma, lambda_adv = lambda_adv, dropout_rate=0.2)
model.train(epoch_num = 100, batch_size = 128, lr = 1e-3, adaptlr = False, num_warmup = 0,
            early_stopping = True, valid_prop = 0.1, patience = 25)
model.inference(n_samples=1,update=True,returns=False)
adata = model.get_adata()
adata.write(f"/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/sensitivity/feature_aligned_{beta}_{gamma}_{lambda_adv}.h5ad")

beta, gamma, lambda_adv = 5, 5, 5
adata = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/feature_aligned.h5ad")
adata
rna_hvg = np.where(adata.var_names.isin(adata.uns['rna_hvg']))[0].tolist()
atac_hvg = np.where(adata.var_names.isin(adata.uns['atac_hvg']))[0].tolist()
feature_list = {"0":rna_hvg,"1":atac_hvg}
model = Integration(data=adata, layer="counts", modality_key="modality", batch_key="batch", 
                    feature_list=None, distribution="ZINB") #feature_list)
model.setup(hidden_layers = [500,500], latent_dim_shared = 20, latent_dim_specific=20, 
            beta = beta, gamma = gamma, lambda_adv = lambda_adv, dropout_rate=0.2)
model.train(epoch_num = 100, batch_size = 128, lr = 1e-3, adaptlr = False, num_warmup = 0,
            early_stopping = True, valid_prop = 0.1, patience = 25)
model.inference(n_samples=1,update=True,returns=False)
adata = model.get_adata()
adata.write(f"/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/sensitivity/feature_aligned_{beta}_{gamma}_{lambda_adv}.h5ad")

beta, gamma, lambda_adv = 2, 0, 5
adata = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/feature_aligned.h5ad")
adata
rna_hvg = np.where(adata.var_names.isin(adata.uns['rna_hvg']))[0].tolist()
atac_hvg = np.where(adata.var_names.isin(adata.uns['atac_hvg']))[0].tolist()
feature_list = {"0":rna_hvg,"1":atac_hvg}
model = Integration(data=adata, layer="counts", modality_key="modality", batch_key="batch", 
                    feature_list=None, distribution="ZINB") #feature_list)
model.setup(hidden_layers = [500,500], latent_dim_shared = 20, latent_dim_specific=20, 
            beta = beta, gamma = gamma, lambda_adv = lambda_adv, dropout_rate=0.2)
model.train(epoch_num = 100, batch_size = 128, lr = 1e-3, adaptlr = False, num_warmup = 0,
            early_stopping = True, valid_prop = 0.1, patience = 25)
model.inference(n_samples=1,update=True,returns=False)
adata = model.get_adata()
adata.write(f"/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/sensitivity/feature_aligned_{beta}_{gamma}_{lambda_adv}.h5ad")

beta, gamma, lambda_adv = 2, 3, 5
adata = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/feature_aligned.h5ad")
adata
rna_hvg = np.where(adata.var_names.isin(adata.uns['rna_hvg']))[0].tolist()
atac_hvg = np.where(adata.var_names.isin(adata.uns['atac_hvg']))[0].tolist()
feature_list = {"0":rna_hvg,"1":atac_hvg}
model = Integration(data=adata, layer="counts", modality_key="modality", batch_key="batch", 
                    feature_list=None, distribution="ZINB") #feature_list)
model.setup(hidden_layers = [500,500], latent_dim_shared = 20, latent_dim_specific=20, 
            beta = beta, gamma = gamma, lambda_adv = lambda_adv, dropout_rate=0.2)
model.train(epoch_num = 100, batch_size = 128, lr = 1e-3, adaptlr = False, num_warmup = 0,
            early_stopping = True, valid_prop = 0.1, patience = 25)
model.inference(n_samples=1,update=True,returns=False)
adata = model.get_adata()
adata.write(f"/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/sensitivity/feature_aligned_{beta}_{gamma}_{lambda_adv}.h5ad")

beta, gamma, lambda_adv = 2, 5, 5
adata = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/feature_aligned.h5ad")
adata
rna_hvg = np.where(adata.var_names.isin(adata.uns['rna_hvg']))[0].tolist()
atac_hvg = np.where(adata.var_names.isin(adata.uns['atac_hvg']))[0].tolist()
feature_list = {"0":rna_hvg,"1":atac_hvg}
model = Integration(data=adata, layer="counts", modality_key="modality", batch_key="batch", 
                    feature_list=None, distribution="ZINB") #feature_list)
model.setup(hidden_layers = [500,500], latent_dim_shared = 20, latent_dim_specific=20, 
            beta = beta, gamma = gamma, lambda_adv = lambda_adv, dropout_rate=0.2)
model.train(epoch_num = 100, batch_size = 128, lr = 1e-3, adaptlr = False, num_warmup = 0,
            early_stopping = True, valid_prop = 0.1, patience = 25)
model.inference(n_samples=1,update=True,returns=False)
adata = model.get_adata()
adata.write(f"/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/sensitivity/feature_aligned_{beta}_{gamma}_{lambda_adv}.h5ad")

beta, gamma, lambda_adv = 2, 7, 5
adata = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/feature_aligned.h5ad")
adata
rna_hvg = np.where(adata.var_names.isin(adata.uns['rna_hvg']))[0].tolist()
atac_hvg = np.where(adata.var_names.isin(adata.uns['atac_hvg']))[0].tolist()
feature_list = {"0":rna_hvg,"1":atac_hvg}
model = Integration(data=adata, layer="counts", modality_key="modality", batch_key="batch", 
                    feature_list=None, distribution="ZINB") #feature_list)
model.setup(hidden_layers = [500,500], latent_dim_shared = 20, latent_dim_specific=20, 
            beta = beta, gamma = gamma, lambda_adv = lambda_adv, dropout_rate=0.2)
model.train(epoch_num = 100, batch_size = 128, lr = 1e-3, adaptlr = False, num_warmup = 0,
            early_stopping = True, valid_prop = 0.1, patience = 25)
model.inference(n_samples=1,update=True,returns=False)
adata = model.get_adata()
adata.write(f"/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/sensitivity/feature_aligned_{beta}_{gamma}_{lambda_adv}.h5ad")

beta, gamma, lambda_adv = 2, 10, 5
adata = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/feature_aligned.h5ad")
adata
rna_hvg = np.where(adata.var_names.isin(adata.uns['rna_hvg']))[0].tolist()
atac_hvg = np.where(adata.var_names.isin(adata.uns['atac_hvg']))[0].tolist()
feature_list = {"0":rna_hvg,"1":atac_hvg}
model = Integration(data=adata, layer="counts", modality_key="modality", batch_key="batch", 
                    feature_list=None, distribution="ZINB") #feature_list)
model.setup(hidden_layers = [500,500], latent_dim_shared = 20, latent_dim_specific=20, 
            beta = beta, gamma = gamma, lambda_adv = lambda_adv, dropout_rate=0.2)
model.train(epoch_num = 100, batch_size = 128, lr = 1e-3, adaptlr = False, num_warmup = 0,
            early_stopping = True, valid_prop = 0.1, patience = 25)
model.inference(n_samples=1,update=True,returns=False)
adata = model.get_adata()
adata.write(f"/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/sensitivity/feature_aligned_{beta}_{gamma}_{lambda_adv}.h5ad")

beta, gamma, lambda_adv = 2, 5, 0
adata = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/feature_aligned.h5ad")
adata
rna_hvg = np.where(adata.var_names.isin(adata.uns['rna_hvg']))[0].tolist()
atac_hvg = np.where(adata.var_names.isin(adata.uns['atac_hvg']))[0].tolist()
feature_list = {"0":rna_hvg,"1":atac_hvg}
model = Integration(data=adata, layer="counts", modality_key="modality", batch_key="batch", 
                    feature_list=None, distribution="ZINB") #feature_list)
model.setup(hidden_layers = [500,500], latent_dim_shared = 20, latent_dim_specific=20, 
            beta = beta, gamma = gamma, lambda_adv = lambda_adv, dropout_rate=0.2)
model.train(epoch_num = 100, batch_size = 128, lr = 1e-3, adaptlr = False, num_warmup = 0,
            early_stopping = True, valid_prop = 0.1, patience = 25)
model.inference(n_samples=1,update=True,returns=False)
adata = model.get_adata()
adata.write(f"/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/sensitivity/feature_aligned_{beta}_{gamma}_{lambda_adv}.h5ad")

beta, gamma, lambda_adv = 2, 5, 3
adata = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/feature_aligned.h5ad")
adata
rna_hvg = np.where(adata.var_names.isin(adata.uns['rna_hvg']))[0].tolist()
atac_hvg = np.where(adata.var_names.isin(adata.uns['atac_hvg']))[0].tolist()
feature_list = {"0":rna_hvg,"1":atac_hvg}
model = Integration(data=adata, layer="counts", modality_key="modality", batch_key="batch", 
                    feature_list=None, distribution="ZINB") #feature_list)
model.setup(hidden_layers = [500,500], latent_dim_shared = 20, latent_dim_specific=20, 
            beta = beta, gamma = gamma, lambda_adv = lambda_adv, dropout_rate=0.2)
model.train(epoch_num = 100, batch_size = 128, lr = 1e-3, adaptlr = False, num_warmup = 0,
            early_stopping = True, valid_prop = 0.1, patience = 25)
model.inference(n_samples=1,update=True,returns=False)
adata = model.get_adata()
adata.write(f"/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/sensitivity/feature_aligned_{beta}_{gamma}_{lambda_adv}.h5ad")

beta, gamma, lambda_adv = 2, 5, 7
adata = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/feature_aligned.h5ad")
adata
rna_hvg = np.where(adata.var_names.isin(adata.uns['rna_hvg']))[0].tolist()
atac_hvg = np.where(adata.var_names.isin(adata.uns['atac_hvg']))[0].tolist()
feature_list = {"0":rna_hvg,"1":atac_hvg}
model = Integration(data=adata, layer="counts", modality_key="modality", batch_key="batch", 
                    feature_list=None, distribution="ZINB") #feature_list)
model.setup(hidden_layers = [500,500], latent_dim_shared = 20, latent_dim_specific=20, 
            beta = beta, gamma = gamma, lambda_adv = lambda_adv, dropout_rate=0.2)
model.train(epoch_num = 100, batch_size = 128, lr = 1e-3, adaptlr = False, num_warmup = 0,
            early_stopping = True, valid_prop = 0.1, patience = 25)
model.inference(n_samples=1,update=True,returns=False)
adata = model.get_adata()
adata.write(f"/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/sensitivity/feature_aligned_{beta}_{gamma}_{lambda_adv}.h5ad")

beta, gamma, lambda_adv = 2, 5, 10
adata = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/feature_aligned.h5ad")
adata
rna_hvg = np.where(adata.var_names.isin(adata.uns['rna_hvg']))[0].tolist()
atac_hvg = np.where(adata.var_names.isin(adata.uns['atac_hvg']))[0].tolist()
feature_list = {"0":rna_hvg,"1":atac_hvg}
model = Integration(data=adata, layer="counts", modality_key="modality", batch_key="batch", 
                    feature_list=None, distribution="ZINB") #feature_list)
model.setup(hidden_layers = [500,500], latent_dim_shared = 20, latent_dim_specific=20, 
            beta = beta, gamma = gamma, lambda_adv = lambda_adv, dropout_rate=0.2)
model.train(epoch_num = 100, batch_size = 128, lr = 1e-3, adaptlr = False, num_warmup = 0,
            early_stopping = True, valid_prop = 0.1, patience = 25)
model.inference(n_samples=1,update=True,returns=False)
adata = model.get_adata()
adata.write(f"/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/sensitivity/feature_aligned_{beta}_{gamma}_{lambda_adv}.h5ad")

beta, gamma, lambda_adv = 2, 2, 2
adata = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/feature_aligned.h5ad")
adata
rna_hvg = np.where(adata.var_names.isin(adata.uns['rna_hvg']))[0].tolist()
atac_hvg = np.where(adata.var_names.isin(adata.uns['atac_hvg']))[0].tolist()
feature_list = {"0":rna_hvg,"1":atac_hvg}
model = Integration(data=adata, layer="counts", modality_key="modality", batch_key="batch", 
                    feature_list=None, distribution="ZINB") #feature_list)
model.setup(hidden_layers = [500,500], latent_dim_shared = 20, latent_dim_specific=20, 
            beta = beta, gamma = gamma, lambda_adv = lambda_adv, dropout_rate=0.2)
model.train(epoch_num = 100, batch_size = 128, lr = 1e-3, adaptlr = False, num_warmup = 0,
            early_stopping = True, valid_prop = 0.1, patience = 25)
model.inference(n_samples=1,update=True,returns=False)
adata = model.get_adata()
adata.write(f"/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/sensitivity/feature_aligned_{beta}_{gamma}_{lambda_adv}.h5ad")

beta, gamma, lambda_adv = 2, 5, 5
adata = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/feature_aligned.h5ad")
adata
rna_hvg = np.where(adata.var_names.isin(adata.uns['rna_hvg']))[0].tolist()
atac_hvg = np.where(adata.var_names.isin(adata.uns['atac_hvg']))[0].tolist()
feature_list = {"0":rna_hvg,"1":atac_hvg}
model = Integration(data=adata, layer="counts", modality_key="modality", batch_key="batch", 
                    feature_list=None, distribution="ZINB") #feature_list)
model.setup(hidden_layers = [500,500], latent_dim_shared = 20, latent_dim_specific=20, 
            beta = beta, gamma = gamma, lambda_adv = lambda_adv, dropout_rate=0.2)
model.train(epoch_num = 100, batch_size = 128, lr = 1e-3, adaptlr = False, num_warmup = 0,
            early_stopping = True, valid_prop = 0.1, patience = 25)
model.inference(n_samples=1,update=True,returns=False)
adata = model.get_adata()
adata.write(f"/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/sensitivity/feature_aligned_{beta}_{gamma}_{lambda_adv}.h5ad")

beta, gamma, lambda_adv = 5, 5, 5
adata = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/feature_aligned.h5ad")
adata
model = Integration(data=adata, layer="counts", modality_key="modality", batch_key="batch", 
                    feature_list=None, distribution="ZINB")
model.setup(hidden_layers = [500,500], latent_dim_shared = 20, latent_dim_specific=20, 
            beta = beta, gamma = gamma, lambda_adv = lambda_adv, dropout_rate=0.2)
model.train(epoch_num = 100, batch_size = 128, lr = 1e-3, adaptlr = False, num_warmup = 0,
            early_stopping = True, valid_prop = 0.1, patience = 25)
model.inference(n_samples=1,update=True,returns=False)
adata = model.get_adata()
adata.write(f"/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/sensitivity/feature_aligned_{beta}_{gamma}_{lambda_adv}_noadv.h5ad")

beta, gamma, lambda_adv = 2, 10, 5
adata = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/feature_aligned.h5ad")
adata
model = Integration(data=adata, layer="counts", modality_key="modality", batch_key="batch", 
                    feature_list=None, distribution="ZINB")
model.setup(hidden_layers = [500,500], latent_dim_shared = 20, latent_dim_specific=20, 
            beta = beta, gamma = gamma, lambda_adv = lambda_adv, dropout_rate=0.2)
model.train(epoch_num = 100, batch_size = 128, lr = 1e-3, adaptlr = False, num_warmup = 0,
            early_stopping = True, valid_prop = 0.1, patience = 25)
model.inference(n_samples=1,update=True,returns=False)
adata = model.get_adata()
adata.write(f"/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/sensitivity/feature_aligned_{beta}_{gamma}_{lambda_adv}.h5ad")

beta, gamma, lambda_adv = 2, 5, 10
adata = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/feature_aligned.h5ad")
adata
model = Integration(data=adata, layer="counts", modality_key="modality", batch_key="batch", 
                    feature_list=None, distribution="ZINB")
model.setup(hidden_layers = [500,500], latent_dim_shared = 20, latent_dim_specific=20, 
            beta = beta, gamma = gamma, lambda_adv = lambda_adv, dropout_rate=0.2)
model.train(epoch_num = 100, batch_size = 128, lr = 1e-3, adaptlr = False, num_warmup = 0,
            early_stopping = True, valid_prop = 0.1, patience = 25)
model.inference(n_samples=1,update=True,returns=False)
adata = model.get_adata()
adata.write(f"/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/sensitivity/feature_aligned_{beta}_{gamma}_{lambda_adv}.h5ad")

beta, gamma, lambda_adv = 2, 1, 5
adata = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/feature_aligned.h5ad")
adata
rna_hvg = np.where(adata.var_names.isin(adata.uns['rna_hvg']))[0].tolist()
atac_hvg = np.where(adata.var_names.isin(adata.uns['atac_hvg']))[0].tolist()
feature_list = {"0":rna_hvg,"1":atac_hvg}
model = Integration(data=adata, layer="counts", modality_key="modality", batch_key="batch", 
                    feature_list=None, distribution="ZINB") #feature_list)
model.setup(hidden_layers = [500,500], latent_dim_shared = 20, latent_dim_specific=20, 
            beta = beta, gamma = gamma, lambda_adv = lambda_adv, dropout_rate=0.2)
model.train(epoch_num = 100, batch_size = 128, lr = 1e-3, adaptlr = False, num_warmup = 0,
            early_stopping = True, valid_prop = 0.1, patience = 25)
model.inference(n_samples=1,update=True,returns=False)
adata = model.get_adata()
adata.write(f"/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/sensitivity/feature_aligned_{beta}_{gamma}_{lambda_adv}.h5ad")

beta, gamma, lambda_adv = 2, 5, 1
adata = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/feature_aligned.h5ad")
adata
rna_hvg = np.where(adata.var_names.isin(adata.uns['rna_hvg']))[0].tolist()
atac_hvg = np.where(adata.var_names.isin(adata.uns['atac_hvg']))[0].tolist()
feature_list = {"0":rna_hvg,"1":atac_hvg}
model = Integration(data=adata, layer="counts", modality_key="modality", batch_key="batch", 
                    feature_list=None, distribution="ZINB") #feature_list)
model.setup(hidden_layers = [500,500], latent_dim_shared = 20, latent_dim_specific=20, 
            beta = beta, gamma = gamma, lambda_adv = lambda_adv, dropout_rate=0.2)
model.train(epoch_num = 100, batch_size = 128, lr = 1e-3, adaptlr = False, num_warmup = 0,
            early_stopping = True, valid_prop = 0.1, patience = 25)
model.inference(n_samples=1,update=True,returns=False)
adata = model.get_adata()
adata.write(f"/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/sensitivity/feature_aligned_{beta}_{gamma}_{lambda_adv}.h5ad")

beta, gamma, lambda_adv = 5, 10, 10
adata = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/feature_aligned.h5ad")
adata
model = Integration(data=adata, layer="counts", modality_key="modality", batch_key="batch", 
                    feature_list=None, distribution="ZINB")
model.setup(hidden_layers = [500,500], latent_dim_shared = 20, latent_dim_specific=20, 
            beta = beta, gamma = gamma, lambda_adv = lambda_adv, dropout_rate=0.2)
model.train(epoch_num = 100, batch_size = 128, lr = 1e-3, adaptlr = False, num_warmup = 0,
            early_stopping = True, valid_prop = 0.1, patience = 25)
model.inference(n_samples=1,update=True,returns=False)
adata = model.get_adata()
adata.write(f"/home/bingxing2/ailab/group/ai4bio/sunjianle/mop/muto/sensitivity/feature_aligned_{beta}_{gamma}_{lambda_adv}.h5ad")
