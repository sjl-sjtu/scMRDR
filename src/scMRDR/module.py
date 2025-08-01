import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import optim
import torch.utils.data as Data
import numpy as np
import pandas as pd
from .data import CombinedDataset #IntegrateDataset,
from .model import EmbeddingNet
from .train import train_model, inference_model#, train_integrate, inference_integrate
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
import anndata as ad
from torch.utils.tensorboard import SummaryWriter
from scipy.sparse import lil_matrix,csr_matrix,issparse
import scanpy as sc
from sklearn.model_selection import train_test_split

def to_dense_array(x):
    """
    Convert input to a dense numpy array.
    Args:
        x: Input data, can be a sparse matrix, numpy array, or other types.
    Returns:
        Dense numpy array.
    """
    if issparse(x):
        return x.toarray()
    elif isinstance(x, np.ndarray):
        return x.copy()
    else:
        raise TypeError(f"Unsupported type: {type(x)}")

class Integration:
    '''
    Integration class.
    Args:
        data: AnnData object
        layer: str, layer name in adata.layers containing the data to be integrated
        modality_key: str, key in adata.obs for modality information
        batch_key: str, key in adata.obs for batch information
        count_data: bool, whether the data is count data or normalized data
        positive_outputs: bool, whether the outputs are all positive
        feature_list: distionary, containing unmasked feature indices for each mask group (by default, modality). Default is None, indicating all features are unmasked.
        mask_key: str, key in adata.obs to indicate mask information, corresponding to feature_list. Default is None, indicating modality_key will be used.
    '''
    def __init__(self, data, layer=None, modality_key="modality", batch_key=None, #celltype_key=None, 
                 count_data=True, positive_outputs=True, mask_key=None, feature_list=None):
        super(Integration,self).__init__()
        if isinstance(data, list) & isinstance(data[0], ad.AnnData):
            self.adata = ad.concat(data, axis='obs', join='inner', label="modality")
        elif isinstance(data, ad.AnnData):
            self.adata = data
        else:
            raise ValueError("Wrong type of data!")
        if layer is None:
            self.data = to_dense_array(self.adata.X)
        else:
            self.data = to_dense_array(self.adata.layers[layer])
        label_encoder = LabelEncoder()
        onehot_encoder = OneHotEncoder(sparse_output=False)
        self.modality_label = self.adata.obs[modality_key].to_numpy()
        self.modality = label_encoder.fit_transform(self.modality_label)
        self.modality_ordered = [label for label in label_encoder.classes_]
        # if shared_modality_list is not None:
        #     first_modality = shared_modality_list[0]
        #     first_encoding = self.modality_ordered.index(first_modality)
        #     to_replace = np.isin(self.modality, [self.modality_ordered.index(mod) for mod in shared_modality_list])
        #     self.modality[to_replace] = first_encoding
        self.modality = onehot_encoder.fit_transform(self.modality.reshape(-1, 1))
        # if celltype_key is None:
        #     self.celltype = None
        #     self.celltype_ordered = None
        # else:
        #     self.celltype_label = self.adata.obs[celltype_key].to_numpy()
        #     self.celltype = label_encoder.fit_transform(self.celltype_label)
        #     self.celltype_ordered = [label for label in label_encoder.classes_]
        #     self.celltype = onehot_encoder.fit_transform(self.celltype.reshape(-1, 1))     
        if batch_key is None:
            self.covariates = None
            self.covariates_ordered = None
        else:
            self.covariates_label = self.adata.obs[batch_key].to_numpy()
            self.covariates = label_encoder.fit_transform(self.covariates_label)
            self.covariates_ordered = [label for label in label_encoder.classes_]
            self.covariates = onehot_encoder.fit_transform(self.covariates.reshape(-1, 1))
        
        self.modality_num = self.modality.shape[1]
        # if self.celltype is not None:
        #     self.celltype_num = self.celltype.shape[1]
        # else:
        #     self.celltype_num = 0
        
        if self.covariates is not None:
            self.covariates_dim = self.covariates.shape[1]
        else:
            self.covariates_dim = 0
        
        if mask_key is None:
            self.mask = self.modality
            self.mask_num = self.modality_num
            self.mask_ordered = self.modality_ordered
        else:
            self.mask_label = self.adata.obs[mask_key].to_numpy()
            self.mask = label_encoder.fit_transform(self.mask_label)
            self.mask_ordered = [label for label in label_encoder.classes_]
            self.mask = onehot_encoder.fit_transform(self.mask.reshape(-1, 1))
            self.mask_num = self.mask.shape[1]

        if feature_list is not None:
            self.feat_mask = 0 * torch.ones(self.mask_num, self.data.shape[1])
            feature_list_ordered = [feature_list[label] for label in self.mask_ordered]
            for i, feat_idx in enumerate(feature_list_ordered):
                self.feat_mask[i, feat_idx] = 1
        else:
            self.feat_mask = torch.ones(self.mask_num, self.data.shape[1])
        
        self.count_data = count_data
        self.positive_outputs = positive_outputs
    
    def setup(self, hidden_layers = [100,50], latent_dim_shared = 15, latent_dim_specific = 15, dropout_rate=0.5, 
              beta = 2, gamma = 1, lambda_adv = 0.01, device=None):
        '''
        Setup the model.
        Args:
            hidden_layers: list, hidden layers dimensions of the model
            latent_dim_shared: int, latent dimension of the shared latent space
            latent_dim_specific: int, latent dimension of the specific latent space
            dropout_rate: float, dropout rate in neural network
            beta: float, beta parameter for the beta distribution
            gamma: float, gamma parameter for the gamma distribution
            lambda_adv: float, lambda parameter for the adversarial loss
            device: device to train the model. Default is None, indicating GPU will be used if available.
        '''
        self.input_dim = self.data.shape[1]
        self.hidden_layers = hidden_layers
        self.latent_dim_shared = latent_dim_shared
        self.latent_dim_specific = latent_dim_specific
        self.dropout_rate = dropout_rate
        self.beta = beta
        self.gamma = gamma
        self.lambda_adv = lambda_adv
        
        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
        else:
            self.device = device
        print("using "+str(self.device))
        self.model = EmbeddingNet(self.device, self.input_dim, self.modality_num, self.covariates_dim, layer_dims=self.hidden_layers, 
                    latent_dim_shared=self.latent_dim_shared, latent_dim_specific=self.latent_dim_specific,dropout_rate = self.dropout_rate, 
                    beta=self.beta, gamma = self.gamma, lambda_adv = self.lambda_adv,
                    feat_mask = self.feat_mask, count_data=self.count_data,
                    positive_outputs=self.positive_outputs).to(self.device)
        self.train_dataset = CombinedDataset(self.data,self.covariates,self.modality,self.mask)
    
    def train(self,epoch_num = 200, batch_size = 64, lr = 1e-5, accumulation_steps = 1, 
              adaptlr = False, valid_prop = 0.1, num_warmup = 0, early_stopping = True, patience = 10,
              tensorboard = False, savepath = "./", random_state=42):
        '''
        Train the model.
        Args:
            epoch_num: int, number of epochs
            batch_size: int, batch size
            lr: float, learning rate    
            accumulation_steps: int, number of steps to accumulate gradients
            adaptlr: bool, whether to adapt learning rate
            valid_prop: float, proportion of data to use for validation
            num_warmup: int, number of warmup epochs
            early_stopping: bool, whether to use early stopping
            patience: int, patience for early stopping
            tensorboard: bool, whether to use tensorboard
            savepath: str, path to save the tensorboard logs
            random_state: int, random seed
        '''
        if tensorboard:
            print("Using tensorboard!")
            self.writer = SummaryWriter(savepath)
        else:
            self.writer = None
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.lr = lr
        self.accumulation_steps = accumulation_steps
        self.adaptlr = adaptlr
        if valid_prop > 0:
            train_indices, valid_indices = train_test_split(
                np.arange(len(self.train_dataset)),
                test_size=valid_prop,
                stratify=self.modality.argmax(-1),
                random_state=random_state
            )
            train_dataset = Data.Subset(self.train_dataset, train_indices)
            valid_dataset = Data.Subset(self.train_dataset, valid_indices)
        else:
            train_dataset, valid_dataset = self.train_dataset, self.train_dataset
        self.num_batch = len(train_dataset)//self.batch_size
        
        print("Training start!")
        train_model(self.device, self.writer, train_dataset, valid_dataset,
                    self.model, self.epoch_num, self.batch_size, 
                    self.num_batch, self.lr, accumulation_steps = self.accumulation_steps, 
                    adaptlr = self.adaptlr, num_warmup = num_warmup, early_stopping = early_stopping,
                    patience = patience)
        if tensorboard:
            self.writer.close()
        print("Training finished!")
    
    def inference(self, n_samples=1, dataset=None, batch_size=None, update=True, returns=False):
        '''
        Inference the model.
        Args:
            n_samples: int, number of samples to average in reparametrization trick
            dataset: dataset to use for inference
            batch_size: int, batch size
            update: bool, whether to update the latent embeddings in the adata
            returns: bool, whether to return the results, including latent shared, latent specific, rho, dispersion, pi, library size
        '''
        if dataset is None:
            dataset = self.train_dataset
        if batch_size is None:
            batch_size = self.batch_size
        if n_samples > 1:
            z_shared,z_specific,rho,dispersion,pi,library_size = \
                zip(*[inference_model(self.device, dataset, self.model, batch_size) for _ in range(n_samples)])
            self.z_shared = np.mean(np.stack(z_shared, axis=0), axis=0)
            self.z_specific = np.mean(np.stack(z_specific, axis=0), axis=0) 
            self.rho = np.mean(np.stack(rho, axis=0), axis=0) 
            self.dispersion = np.mean(np.stack(dispersion, axis=0), axis=0) 
            self.pi = np.mean(np.stack(pi, axis=0), axis=0) 
            self.library_size = np.mean(np.stack(library_size, axis=0), axis=0) 
        else:
            self.z_shared,self.z_specific = \
                inference_model(self.device, dataset, self.model, batch_size)
        if update:
            self.adata.obsm['latent_shared'] = self.z_shared
            self.adata.obsm['latent_specific'] = self.z_specific
            # self.adata.layers['estimated_mean_expression'] = self.rho
            # self.adata.layers['estimated_dropout_rate'] = self.pi
            # self.adata.var['estimated_dispersion_factor'] = self.dispersion
            # self.adata.obs['estimated_library_size'] = self.library_size
            print('All results recorded in adata.')
        if returns:
            return self.z_shared,self.z_specific,self.rho,self.dispersion,self.pi,self.library_size  
    
    # def integrate(self, data_shared=None, data_specific=None, num_heads = 5, paired = False,
    #               latent_dim_out=20, epoch_num = 200, batch_size = 64, lr = 1e-5, accumulation_steps = 1, 
    #               adaptlr = True, update=True, returns=True):
    #     if data_shared is None:
    #         data_shared = self.z_shared
    #     if data_specific is None:
    #         data_specific = self.z_specific
    #     self.latent_dim_out = latent_dim_out
    #     modality_names = np.unique(self.modality_label) 
    #     aligned_z = lil_matrix((len(self.train_dataset), self.latent_dim_out))
        
    #     anchor_name = modality_names[0]
    #     anchor_idx = np.where(self.modality_label==anchor_name)[0]
    #     anchor_modal_shared = data_shared[anchor_idx,:].copy()  
    #     anchor_modal_specific = data_specific[anchor_idx,:].copy()            
    #     for i in range(1,len(modality_names)):
    #         modal_name = modality_names[i]
    #         modal_idx = np.where(self.modality_label==modal_name)[0]
    #         modal_shared = data_shared[modal_idx,:].copy()
    #         modal_specific = data_shared[modal_idx,:].copy()
            
    #         dataset = IntegrateDataset(anchor_modal_shared, anchor_modal_specific, modal_shared, modal_specific)
    #         inte_model = imputeNet(self.device, self.latent_dim_shared, self.latent_dim_out,
    #                                   num_heads = num_heads, paired = paired).to(self.device)
    #         num_batch = len(dataset)//batch_size
    #         print("start alignment for anchor modal " + str(anchor_name) + " and modal "+str(modal_name))
    #         train_integrate(self.device, self.writer, dataset, 
    #                 inte_model, epoch_num, batch_size, 
    #                 num_batch, lr, accumulation_steps, 
    #                 adaptlr = adaptlr)
    #         aligned_z1, aligned_z2 = inference_integrate(self.device, dataset, inte_model, batch_size)
    #         aligned_z[anchor_idx,:] = aligned_z1
    #         aligned_z[modal_idx,:] = aligned_z2
    #     print("All alignment finished!")
    #     aligned_z = aligned_z.toarray()
    #     if update:
    #         self.adata.obsm['aligned_latent'] = aligned_z
    #     if returns:
    #         return aligned_z
    
    # def get_adata(self):
    #     return self.adata
            
        
        