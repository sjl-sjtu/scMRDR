import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
# import torchvision.transforms as transforms
from .loss import * 
#ZINBLoss,ZIBLoss,klLoss,klLoss_prior,catLoss,mseLoss,orthogonalityLoss,wasserstein_loss,local_structure_loss,frobenius_alignment_loss,laplacian_loss,HSICloss,MMD,isometric_loss
from torch.nn import functional as F
import scipy as sp
# import ot

# class AttentionIntegrate(nn.Module):
#     def __init__(self, latent_dim_hidden, num_heads, latent_dim_out, paired=True):
#         super(AttentionIntegrate, self).__init__()
#         self.paired = paired
#         self.multihead_modal_to_anchor = nn.MultiheadAttention(embed_dim=latent_dim_hidden, 
#                                                                 num_heads=num_heads, batch_first=True)
#         self.multihead_anchor_to_modal = nn.MultiheadAttention(embed_dim=latent_dim_hidden, 
#                                                                 num_heads=num_heads, batch_first=True)
#         self.fc_fusion_anchor = nn.Sequential(nn.Linear(latent_dim_hidden, latent_dim_out),
#                                            nn.LeakyReLU(),
#                                            nn.BatchNorm1d(latent_dim_out),
#                                            nn.Linear(latent_dim_out, latent_dim_out),
#                                            nn.BatchNorm1d(latent_dim_out)) 
#         self.fc_fusion_modal = nn.Sequential(nn.Linear(latent_dim_hidden, latent_dim_out),
#                                            nn.LeakyReLU(),
#                                            nn.BatchNorm1d(latent_dim_out),
#                                            nn.Linear(latent_dim_out, latent_dim_out),
#                                            nn.BatchNorm1d(latent_dim_out)) 
#         self.transform_anchor = nn.Linear(latent_dim_hidden, latent_dim_out)
#         self.transform_modal = nn.Linear(latent_dim_hidden, latent_dim_out)

#     def forward(self, z1, z2):
#         attn_output_modal_to_anchor, _ = self.multihead_modal_to_anchor(
#             query=z2.unsqueeze(0),  # Shape: (1, n, latent_dim_hidden)
#             key=z1.unsqueeze(0),     # Shape: (1, n, latent_dim_hidden)
#             value=z1.unsqueeze(0)    # Shape: (1, n, latent_dim_hidden)
#         )
#         attn_output_anchor_to_modal, _ = self.multihead_modal_to_anchor(
#             query=z1.unsqueeze(0),  # Shape: (1, n, latent_dim_hidden)
#             key=z2.unsqueeze(0),     # Shape: (1, n, latent_dim_hidden)
#             value=z2.unsqueeze(0)    # Shape: (1, n, latent_dim_hidden)
#         )

#         refined_z1 = self.transform_anchor(z1)+self.fc_fusion_anchor(attn_output_anchor_to_modal.squeeze(0))
#         refined_z2 = self.transform_modal(z2)+self.fc_fusion_modal(attn_output_modal_to_anchor.squeeze(0))  # Shape: (n, latent_dim_out)
#         return refined_z1, refined_z2

# class AttentionFusionLayer(nn.Module):
#     def __init__(self, latent_dim_rna, latent_dim_atac, num_heads, latent_dim_hidden, latent_dim_out, paired=True):
#         super(AttentionFusionLayer, self).__init__()
#         self.paired = paired

#         # 将 RNA 和 ATAC 的潜在表示映射到共同的隐藏空间
#         self.fc_rna = nn.Sequential(nn.Linear(latent_dim_rna, latent_dim_hidden),
#                                     nn.BatchNorm1d(latent_dim_hidden))
#         self.fc_atac = nn.Sequential(nn.Linear(latent_dim_atac, latent_dim_hidden),
#                                      nn.BatchNorm1d(latent_dim_hidden))
        
#         # 多头注意力机制，输入来自 RNA 和 ATAC 两个模态
#         self.multihead_attn_rna_to_atac = nn.MultiheadAttention(embed_dim=latent_dim_hidden, num_heads=num_heads, batch_first=True)
#         self.multihead_attn_atac_to_rna = nn.MultiheadAttention(embed_dim=latent_dim_hidden, num_heads=num_heads, batch_first=True)

#         # 融合层：将注意力结果整合成 latent_dim_out
#         self.fc_fusion = nn.Sequential(nn.Linear(latent_dim_out, latent_dim_out),
#                                        nn.BatchNorm1d(latent_dim_out))
#         self.fc_fusion_rna = nn.Sequential(nn.Linear(latent_dim_hidden, latent_dim_out),
#                                            nn.LeakyReLU(),
#                                            self.fc_fusion)
#         self.fc_fusion_atac = nn.Sequential(nn.Linear(latent_dim_hidden, latent_dim_out),
#                                            nn.LeakyReLU(),
#                                            self.fc_fusion) 

#     def forward(self, z1, z2):
#         # 将 RNA 和 ATAC 模态分别映射到共同的隐藏空间
#         z1_transformed = self.fc_rna(z1)  # Shape: (n, latent_dim_hidden)
#         z2_transformed = self.fc_atac(z2)  # Shape: (n, latent_dim_hidden)

#         # 多头注意力：RNA 模态作为 query，ATAC 模态作为 key 和 value
#         attn_output_rna_to_atac, _ = self.multihead_attn_rna_to_atac(
#             query=z1_transformed.unsqueeze(0),  # Shape: (1, n, latent_dim_hidden)
#             key=z2_transformed.unsqueeze(0),   # Shape: (1, n, latent_dim_hidden)
#             value=z2_transformed.unsqueeze(0)  # Shape: (1, n, latent_dim_hidden)
#         )

#         # 多头注意力：ATAC 模态作为 query，RNA 模态作为 key 和 value
#         attn_output_atac_to_rna, _ = self.multihead_attn_atac_to_rna(
#             query=z2_transformed.unsqueeze(0),  # Shape: (1, n, latent_dim_hidden)
#             key=z1_transformed.unsqueeze(0),     # Shape: (1, n, latent_dim_hidden)
#             value=z1_transformed.unsqueeze(0)    # Shape: (1, n, latent_dim_hidden)
#         )

#         refined_z1 = self.fc_fusion_rna(attn_output_rna_to_atac.squeeze(0))  # Shape: (n, latent_dim_out)
#         refined_z2 = self.fc_fusion_atac(attn_output_atac_to_rna.squeeze(0))  # Shape: (n, latent_dim_out)
        
#         # refined_mu_1, refined_logvar1 = torch.chunk(refined_z1, 2, dim=1)
#         # refined_mu_2, refined_logvar2 = torch.chunk(refined_z2, 2, dim=1)
#         return refined_z1, refined_z2 #refined_mu_1, refined_logvar1, refined_mu_2, refined_logvar2 #

class ModalityDiscriminator(nn.Module):
    '''
    Discriminator for modality classification.
    Args:
        z_dim (int): Dimension of the input latent space.
        num_modalities (int): Number of modalities to classify.
        layer_dims (list): List of hidden layer dimensions.
        dropout_rate (float): Dropout rate for regularization.
    '''
    def __init__(self, z_dim, num_modalities, layer_dims=[128,128], dropout_rate=0.2):
        super(ModalityDiscriminator, self).__init__()
        layers = []
        current_dim = z_dim
        for dim in layer_dims:
            layers.append(nn.Linear(current_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = dim
        layers.append(nn.Linear(layer_dims[-1], num_modalities))
        self.model = nn.Sequential(*layers) 
        # self.model = nn.Sequential(
        #     nn.Linear(z_dim, hidden_dim),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.LeakyReLU(0.1),
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(hidden_dim, num_modalities)
        # )    
    def forward(self, z):
        '''
        Forward pass through the discriminator.
        Args:
            z (torch.Tensor): Input tensor of shape (batch_size, z_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_modalities).
        '''
        z = self.model(z)
        return z


class Encoder(nn.Module):
    '''
    Encoder for the VAE model.
    Args:
        device (torch.device): Device to run the model on.
        input_dim (int): Dimension of the input data.
        layer_dims (list): List of hidden layer dimensions.
        latent_dim (int): Dimension of the latent space.
        dropout_rate (float): Dropout rate for regularization.
    '''
    def __init__(self, device, input_dim = 3000, layer_dims = [500,100], latent_dim = 20,
                 dropout_rate = 0.5):
        super(Encoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # q(z|x)
        layers_zxy = []
        current_dim = input_dim
        for dim in layer_dims:
            layers_zxy.append(nn.Linear(current_dim, dim))
            layers_zxy.append(nn.BatchNorm1d(dim))
            layers_zxy.append(nn.LeakyReLU())
            layers_zxy.append(nn.Dropout(dropout_rate))
            current_dim = dim
        self.zxy_encoder = nn.Sequential(*layers_zxy) 
        
        self.mu_layer = nn.Linear(layer_dims[-1], latent_dim)
        self.logvar_layer = nn.Linear(layer_dims[-1], latent_dim)
        
    def forward(self,x):
        '''
        Forward pass through the encoder.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        Returns:
            z (torch.Tensor): Latent variable tensor of shape (batch_size, latent_dim).
            mu (torch.Tensor): Mean of the latent variable distribution.
            logvar (torch.Tensor): Log variance of the latent variable distribution.
        '''
        h = self.zxy_encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar #
    
    def reparameterize(self, mu, logvar):
        '''
        Reparameterization trick to sample from the latent variable distribution.
        Args:
            mu (torch.Tensor): Mean of the latent variable distribution.
            logvar (torch.Tensor): Log variance of the latent variable distribution.
        Returns:
            z (torch.Tensor): Sampled latent variable tensor.
        '''
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

class Decoder(nn.Module):
    '''
    ZINB Decoder for the VAE model.
    Args:
        device (torch.device): Device to run the model on.
        input_dim (int): Dimension of the input data.
        batch_dim (int): Dimension of the batch size.
        modality_num (int): Number of modalities.
        layer_dims (list): List of hidden layer dimensions.
        latent_dim (int): Dimension of the latent space.
        dropout_rate (float): Dropout rate for regularization.
    '''
    def __init__(self, device, input_dim = 3000, batch_dim = 1, modality_num=2, layer_dims = [500,100], latent_dim = 20,
                 dropout_rate = 0.5):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.batch_dim = batch_dim
        self.modality_num = modality_num
        
        # p(x|z,c)
        layers_xz = []
        current_dim =  latent_dim + batch_dim
        for dim in reversed(layer_dims):
            layers_xz.append(nn.Linear(current_dim, dim))
            layers_xz.append(nn.BatchNorm1d(dim))
            layers_xz.append(nn.LeakyReLU(0.1))
            layers_xz.append(nn.Dropout(dropout_rate))
            current_dim = dim
        
        self.decoder = nn.Sequential(*layers_xz)    
        self.mean_layer = nn.Sequential(nn.Linear(layer_dims[0], input_dim),
                                        nn.Softmax(dim=-1))
        self.dispersion_layer = nn.Sequential(nn.Linear(layer_dims[0], input_dim),
                                              nn.Softplus()) # gene-cell-wise dispersion
        self.dispersion = nn.Parameter(torch.randn(input_dim)) # gene-wise dispersion
        # self.modality_flag = nn.Sequential(nn.Linear(modality_num,1),nn.Tanh())
        self.dispersion_modality = nn.Parameter(torch.randn(modality_num, input_dim)) 
        self.dropout_layer = nn.Sequential(
            nn.Linear(layer_dims[0], input_dim),
            nn.Sigmoid())
        
    def forward(self,z,b,m,dispersion_strategy="gene-modality"):
        '''
        Forward pass through the decoder.
        Args:  
            z (torch.Tensor): Latent variable tensor of shape (batch_size, latent_dim).
            b (torch.Tensor): Batch information tensor of shape (batch_size, batch_dim).
            m (torch.Tensor): Modality information tensor of shape (batch_size, modality_num).
        Returns:
            rho (torch.Tensor): Mean of the output distribution.
            dispersion (torch.Tensor): Dispersion parameter of the output distribution.
            pi (torch.Tensor): Dropout probabilities for the output distribution.
        '''
        if self.batch_dim > 0:
            z = torch.cat([z, b],dim=1)
        h = self.decoder(z)
        rho = self.mean_layer(h)  # Ensure positive outputs
        if dispersion_strategy == "gene":
            dispersion = torch.exp(self.dispersion)
        elif dispersion_strategy == "gene-modality":
            # dispersion = torch.outer(torch.squeeze(self.modality_flag(m)), self.dispersion) # N * G
            dispersion = m @ self.dispersion_modality    # N * G
            dispersion = torch.exp(dispersion) # Ensure positive outputs # gene-wise
        elif dispersion_strategy == "gene-cell":
            dispersion = self.dispersion_layer(h) # gene-cell wise
        pi = self.dropout_layer(h) 
        return rho, dispersion, pi
    
class MSEDecoder(nn.Module):
    '''
    MSE Decoder for the VAE model.
    Args:
        device (torch.device): Device to run the model on.
        input_dim (int): Dimension of the input data.
        batch_dim (int): Dimension of the batch size.
        layer_dims (list): List of hidden layer dimensions.
        latent_dim (int): Dimension of the latent space.
        dropout_rate (float): Dropout rate for regularization.
    '''
    def __init__(self, device, input_dim = 3000, batch_dim = 1, layer_dims = [500,100], latent_dim = 20,
                 dropout_rate = 0.5, positive_outputs=True):
        super(MSEDecoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.batch_dim = batch_dim
        
        # p(x|z,c)
        layers_xz = []
        current_dim =  latent_dim + batch_dim
        for dim in reversed(layer_dims):
            layers_xz.append(nn.Linear(current_dim, dim))
            layers_xz.append(nn.LeakyReLU(0.1))
            layers_xz.append(nn.BatchNorm1d(dim))
            layers_xz.append(nn.Dropout(dropout_rate))
            current_dim = dim
        
        self.decoder = nn.Sequential(*layers_xz)   
        if positive_outputs: 
            self.mean_layer = nn.Sequential(nn.Linear(layer_dims[0], input_dim),
                                            nn.Softplus())
            # self.zero_inflation_rates = nn.Parameter(torch.ones(input_dim) * 0.5) 
        else:
            self.mean_layer = nn.Sequential(nn.Linear(layer_dims[0], input_dim)
                                            # nn.Softplus()
                                            )
        
    def forward(self,z,b):
        '''
        Forward pass through the decoder.
        Args:
            z (torch.Tensor): Latent variable tensor of shape (batch_size, latent_dim).
            b (torch.Tensor): Batch information tensor of shape (batch_size, batch_dim).
        Returns:
            rho (torch.Tensor): Mean of the output distribution.
        '''
        if self.batch_dim > 0:
            z = torch.cat([z, b],dim=1)
        h = self.decoder(z)
        rho = self.mean_layer(h) 
        return rho

class EmbeddingNet(nn.Module):
    '''
    Models to get the unified latent embeddings.
    Args:
        device (torch.device): Device to run the model on.
        input_dim (int): Dimension of the input data.
        modality_num (int): Number of modalities.
        batch_dim (int): Dimension of the covariates (like sequencing batches).
        layer_dims (list): List of hidden layer dimensions.
        latent_dim_shared (int): Dimension of the shared latent space.
        latent_dim_specific (int): Dimension of the modality-specific latent space.
        dropout_rate (float): Dropout rate for regularization.
        beta (float): Weight for the KL divergence term.
        gamma (float): Weight for the isometric loss term.
        lambda_adv (float): Weight for the adversarial loss term.
        feat_mask (torch.Tensor): Feature mask for the input data.
        count_data (bool): Whether the input data is count data.
        positive_outputs (bool): Whether to ensure positive outputs in the decoder.
        encoder_covariates (bool): Whether to include covariates in the encoder.
        eps (float): Small value to avoid division by zero in loss calculations.
    '''

    def __init__(self, device, input_dim, modality_num, batch_dim = 1, layer_dims=[500,100], latent_dim_shared=20,
                latent_dim_specific=20, dropout_rate = 0.5, beta = 2, gamma = 1, lambda_adv = 0.01,
                feat_mask = None, count_data = True, positive_outputs = True,
                encoder_covariates=False, eps=1e-10):
        super(EmbeddingNet, self).__init__()
        
        self.beta = beta
        self.device = device
        self.eps = eps
        # self.paired= paired
        self.input_dim = input_dim
        self.latent_dim_shared = latent_dim_shared
        self.latent_dim_specific = latent_dim_specific
        self.batch_dim = batch_dim
        self.modality_num = modality_num
        self.encoder_covariates = encoder_covariates
        self.gamma = gamma
        self.lambda_adv = lambda_adv
        self.feat_mask = feat_mask.to(self.device)
        self.count_data = count_data
        self.positive_outputs = positive_outputs
        
        self.encoder_shared = Encoder(device, input_dim+batch_dim*encoder_covariates, layer_dims, latent_dim_shared, dropout_rate)
        self.encoder_specific = Encoder(device, input_dim+modality_num+batch_dim*encoder_covariates, layer_dims, latent_dim_specific, dropout_rate)
        if self.count_data:
            self.decoder = Decoder(device, input_dim, batch_dim, modality_num, layer_dims, 
                                   latent_dim_shared+latent_dim_specific, dropout_rate)
        else:
            self.decoder = MSEDecoder(device, input_dim, batch_dim, layer_dims, latent_dim_shared+latent_dim_specific, 
                                      dropout_rate, positive_outputs=positive_outputs)
        self.prior_net_specific = nn.Sequential(nn.Linear(modality_num, 10),
                                       nn.LeakyReLU(0.1),
                                       nn.Dropout(dropout_rate),
                                       nn.Linear(10, 2))  
        self.discriminator = ModalityDiscriminator(latent_dim_shared, modality_num, layer_dims=layer_dims, dropout_rate=dropout_rate)   
    
    def forward(self,x,b,m,i,stage="vae"):
        '''
        Forward pass through the embedding network.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            b (torch.Tensor): Batch information tensor of shape (batch_size, batch_dim).
            m (torch.Tensor): Modality information tensor of shape (batch_size, modality_num).
            i (torch.Tensor): Mask indicator tensor of shape (batch_size, input_dim).
            stage (str): Stage of the model, can be "vae", "discriminator", or "warmup".
        Returns:
            mu_shared (torch.Tensor): Mean of the shared latent variable distribution.
            mu_specific (torch.Tensor): Mean of the specific latent variable distribution.
            total_loss (torch.Tensor): Total loss for the VAE model.
            loss_dict (dict): Dictionary containing individual loss components.
        '''
        if stage=="vae":
            x_original = x
            if self.count_data:
                x = torch.log1p(x)
            
            prior_mu, prior_logvar = torch.chunk(self.prior_net_specific(m), 2, dim=-1)
            if self.encoder_covariates:
                z_shared, mu_shared, logvar_shared = self.encoder_shared(torch.cat([x,b],dim=-1))
                z_specific, mu_specific, logvar_specific = self.encoder_specific(torch.cat([x,m,b],dim=-1))
            else:
                z_shared, mu_shared, logvar_shared = self.encoder_shared(x)
                z_specific, mu_specific, logvar_specific = self.encoder_specific(torch.cat([x,m],dim=-1))
            z = torch.cat([z_shared,z_specific],dim=-1)
            if self.count_data:
                rho,dispersion,pi = self.decoder(z, b, m)
                s = self.sample_sequencing_depth(x_original)
            else:
                rho = self.decoder(z, b)
    
            # rho2,dispersion2,pi2 = self.decoder2(z_shared, b, m)
            
            #loss
            if self.feat_mask is not None:
                mask = i @ self.feat_mask
            else:
                mask = None
            
            zinb_loss = ZINBLoss()
            if self.count_data:
                recon_loss = zinb_loss(x_original, rho, dispersion, pi, s, mask, eps = self.eps)
            else:
                if self.positive_outputs:
                    # recon_loss = ZeroInflatedMSELoss()(x_original, rho, self.decoder.zero_inflation_rates)
                    recon_loss = mseLoss(x_original, rho)
                else:
                    recon_loss = mseLoss(x_original, rho)
            kl_z = klLoss_prior(mu_specific, logvar_specific, prior_mu, prior_logvar)+\
                klLoss(mu_shared, logvar_shared)
            # preserve_loss = zinb_loss(x_original, rho2, dispersion2, pi2, s, eps = self.eps)
            preserve_loss = isometric_loss(torch.cat([mu_shared, mu_specific],dim=-1),mu_shared,m)
            # preserve_loss = isometric_loss(torch.cat([z_shared, z_specific],dim=-1),z_shared,m)
            # preserve_loss = sammon_loss(torch.cat([z_shared, z_specific],dim=-1),z_shared,m)
            # preserve_loss = laplacian_loss(torch.cat([z_shared, z_specific],dim=-1),z_shared,m)
            # preserve_loss = frobenius_isometric_loss(torch.cat([z_shared, z_specific],dim=-1),z_shared,m)
            # preserve_loss = knn_structure_loss(torch.cat([mu_shared, mu_specific],dim=-1),mu_shared,m)
            #z_per_class = [z_shared[m[:, i].bool()] for i in range(m.shape[1])]
            #align_loss = torch.stack([MMD(z_per_class[0], z_per_class[i]) for i in range(1, len(z_per_class))]).sum()

            modality_labels = torch.argmax(m, dim=1)
            modality_logits_adv = self.discriminator(z_shared)  # gradients allowed here
            adv_loss = -F.cross_entropy(modality_logits_adv, modality_labels, reduction='sum')/m.shape[0]

            # z_shared_detached = z_shared.clone().detach()
            # modality_logits = self.discriminator(z_shared_detached) #
            # discri_loss = F.cross_entropy(modality_logits, modality_labels, reduction='sum')/m.shape[0]

                
            total_loss = recon_loss + self.beta*kl_z + self.gamma * preserve_loss + self.lambda_adv * adv_loss # + align_loss
            loss_dict = {'total_loss':total_loss.item(), 
                        'recon_loss':recon_loss.item(),'kl_z':kl_z.item(),
                        'preserve_loss': preserve_loss.item(), #,'align_loss':align_loss.item()
                        'adv_loss': adv_loss.item()
                        } 
            return mu_shared, mu_specific, total_loss, loss_dict
        elif stage=="discriminator":
            x_original = x
            if self.count_data:
                x = torch.log1p(x)
            
            # prior_mu, prior_logvar = torch.chunk(self.prior_net_specific(m), 2, dim=-1)
            if self.encoder_covariates:
                z_shared, mu_shared, logvar_shared = self.encoder_shared(torch.cat([x,b],dim=-1))
                # z_specific, mu_specific, logvar_specific = self.encoder_specific(torch.cat([x,m,b],dim=-1))
            else:
                z_shared, mu_shared, logvar_shared = self.encoder_shared(x)
                # z_specific, mu_specific, logvar_specific = self.encoder_specific(torch.cat([x,m],dim=-1))
    
            # rho2,dispersion2,pi2 = self.decoder2(z_shared, b, m)
            
            #loss
            modality_labels = torch.argmax(m, dim=1)
            z_shared_detached = z_shared.clone().detach()
            modality_logits = self.discriminator(z_shared_detached) #
            discri_loss = F.cross_entropy(modality_logits, modality_labels, reduction='sum')/m.shape[0]
            return discri_loss
        
        elif stage=="warmup":
            x_original = x
            if self.count_data:
                x = torch.log1p(x)
            
            prior_mu, prior_logvar = torch.chunk(self.prior_net_specific(m), 2, dim=-1)
            if self.encoder_covariates:
                z_shared, mu_shared, logvar_shared = self.encoder_shared(torch.cat([x,b],dim=-1))
                z_specific, mu_specific, logvar_specific = self.encoder_specific(torch.cat([x,m,b],dim=-1))
            else:
                z_shared, mu_shared, logvar_shared = self.encoder_shared(x)
                z_specific, mu_specific, logvar_specific = self.encoder_specific(torch.cat([x,m],dim=-1))
            z = torch.cat([z_shared,z_specific],dim=-1)
            if self.count_data:
                rho,dispersion,pi = self.decoder(z, b, m)
                s = self.sample_sequencing_depth(x_original)
            else:
                rho = self.decoder(z, b)
            
            # loss
            if self.feat_mask is not None:
                mask = m @ self.feat_mask
            else:
                mask = None
            
            zinb_loss = ZINBLoss()
            if self.count_data:
                recon_loss = zinb_loss(x_original, rho, dispersion, pi, s, mask, eps = self.eps)
            else:
                recon_loss = mseLoss(x_original, rho, mask)
            kl_z = klLoss_prior(mu_specific, logvar_specific, prior_mu, prior_logvar)+\
                klLoss(mu_shared, logvar_shared)
            preserve_loss = isometric_loss(torch.cat([mu_shared, mu_specific],dim=-1),mu_shared,m)  
            # hsic = 1000 * HSICloss(z_shared,m)
            total_loss = recon_loss + self.beta*kl_z + self.gamma * preserve_loss #+ hsic
            loss_dict = {'recon_loss':recon_loss.item(),'kl_z':kl_z.item(),
                        'preserve_loss': preserve_loss.item(),
                        # 'hsic': hsic.item(),
                        'total_loss':total_loss.item()
                        } 
            return mu_shared, mu_specific, total_loss, loss_dict
    
    def sample_sequencing_depth(self, x, strategy="observed"):
        '''
        Sample sequencing depth based on the strategy.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            strategy (str): Strategy for sampling sequencing depth, can be "batch_sample" or "observed".
        Returns:
            s (torch.Tensor): Sampled sequencing depth tensor of shape (batch_size, 1).
        '''
        if strategy=="batch_sample": # batch-wise empirically sample
            mu_s = torch.log(x.sum(dim=1) + 1.0).mean()
            sigma_s = torch.log(x.sum(dim=1) + 1.0).std()
            log_s = mu_s + sigma_s * torch.randn_like(sigma_s)
            s = torch.exp(log_s)
            # s = s.detach()
        elif strategy == "observed": # directly observed
            log_s = torch.log(x.sum(dim=1)).unsqueeze(1)
            s = torch.exp(log_s)
            # s = s.detach()
        return s
    
    def reparameterize(self, mu, logvar):
        '''
        Reparameterization trick to sample from the latent variable distribution.
        Args:
            mu (torch.Tensor): Mean of the latent variable distribution.
            logvar (torch.Tensor): Log variance of the latent variable distribution.
        Returns:
            z (torch.Tensor): Sampled latent variable tensor.
        '''
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    
# class imputeNet(nn.Module):
#     def __init__(self, device, latent_dim_in, latent_dim_out, num_heads = 5, paired = False, eps=1e-10):
#         super(imputeNet,self).__init__()
#         self.device = device
#         self.latent_dim_in = latent_dim_in
#         self.latent_dim_out = latent_dim_out
#         self.attention_fusion_layer = AttentionIntegrate(latent_dim_in, num_heads, latent_dim_out, paired=paired)
#         self.paired = paired
    
#     def forward(self, z1_shared, z1_specific, z2_shared, z2_specific):
#         aligned_z1, aligned_z2 = self.attention_fusion_layer(z1_shared, z2_shared)
        
#         preserve_loss = isometric_loss(torch.cat([z1_shared,z1_specific],dim=-1),aligned_z1)+\
#             isometric_loss(torch.cat([z2_shared,z2_specific],dim=-1),aligned_z2)
            
#         # if self.paired == True:
#         #     cost_matrix = ot.dist(aligned_z1, aligned_z2, metric='sqeuclidean', p=2)
#         #     D = torch.eye(z1.shape[0]).to(self.device)
#         #     align_loss = wasserstein_loss(cost_matrix, D) #+ frobenius_alignment_loss(aligned_z1, aligned_z2, D)
#         #     # align_loss = nn.MSELoss()(aligned_z1, aligned_z2)
#         # else:
#         #     p = torch.tensor(ot.unif(z1.shape[0])).to(self.device)
#         #     q = torch.tensor(ot.unif(z2.shape[0])).to(self.device)
#         #     cost_matrix = ot.dist(aligned_z1, aligned_z2, metric='sqeuclidean', p=2)
#         #     D = ot.emd(p, q, cost_matrix) # ot.sinkhorn
#         #     # D = torch.tensor(D, dtype=torch.float32).to(self.device) #.detach()
#         #     align_loss = wasserstein_loss(cost_matrix, D) #+ frobenius_alignment_loss(aligned_z1, aligned_z2, D)
#         #     # align_loss = nn.MSELoss()(aligned_z1, D@aligned_z2)
#         align_loss = MMD(aligned_z1,aligned_z2)
        
#         total_loss = align_loss + preserve_loss
#         loss_dict = {'total_loss':total_loss.item(), 
#                      'align_loss': align_loss.item(), 'preserve_loss': preserve_loss.item()}
#         return aligned_z1, aligned_z2, total_loss, loss_dict
    

# class Net(nn.Module):
#     def __init__(self, device, input_dim1=784, input_dim2=784, batch_dim = 1, layer_dims=[500,100], latent_dim=20, #cluster_dim = 10, 
#                 fusion_hidden_dim = 20, num_heads = 4,dropout_rate = 0.5, beta = 1000,
#                 eta = 1000, paired = True, eps=1e-10):
#         super(Net, self).__init__()
        
#         self.eta = eta
#         self.beta = beta
#         self.device = device
#         self.eps = eps
#         self.paired= paired
#         self.input_dim1 = input_dim1
#         self.input_dim2 = input_dim2
#         self.latent_dim = latent_dim
#         self.batch_dim = batch_dim
        
#         # self.mu_layer_shared = nn.Linear(latent_dim, latent_dim)
#         # self.logvar_layer_shared = nn.Linear(latent_dim, latent_dim)
#         # self.mu_layer_specific_rna = nn.Linear(latent_dim, latent_dim)
#         # self.logvar_layer_specific_rna = nn.Linear(latent_dim, latent_dim)
#         # self.mu_layer_specific_atac = nn.Linear(latent_dim, latent_dim)
#         # self.logvar_layer_specific_atac = nn.Linear(latent_dim, latent_dim)
        
#         # self.encoder1 = Encoder(device, input_dim1, batch_dim, layer_dims, latent_dim, dropout_rate)
#         # self.encoder2 = Encoder(device, input_dim2, batch_dim, layer_dims, latent_dim, dropout_rate)
#         self.encoder1_shared = Encoder(device, input_dim1, batch_dim, layer_dims, latent_dim, dropout_rate)
#         self.encoder2_shared = Encoder(device, input_dim2, batch_dim, layer_dims, latent_dim, dropout_rate)
#         self.encoder1_specific = Encoder(device, input_dim1, batch_dim, layer_dims, latent_dim, dropout_rate)
#         self.encoder2_specific = Encoder(device, input_dim2, batch_dim, layer_dims, latent_dim, dropout_rate)
#         self.decoder1 = Decoder(device, input_dim1, batch_dim, layer_dims, 2*latent_dim, dropout_rate)
#         self.decoder2 = Decoder(device, input_dim2, batch_dim, layer_dims, 2*latent_dim, dropout_rate)
#         self.decoder1_ = Decoder(device, input_dim1, batch_dim, layer_dims, latent_dim, dropout_rate)
#         self.decoder2_ = Decoder(device, input_dim2, batch_dim, layer_dims, latent_dim, dropout_rate)
#         self.prior_net = nn.Sequential(nn.Linear(1, 5),
#                                        nn.LeakyReLU(0.1),
#                                        nn.Linear(5, 2))
#         # self.decoder1 = MSEDecoder(device, input_dim1, batch_dim, layer_dims, latent_dim, dropout_rate)
#         # self.decoder2 = MSEDecoder(device, input_dim2, batch_dim, layer_dims, latent_dim, dropout_rate)
#         self.attention_fusion_layer = AttentionFusionLayer(2*latent_dim, 2*latent_dim, num_heads, 
#                                                            fusion_hidden_dim, 2*latent_dim, paired=paired)
        
#     def forward(self,x1,b1,x2,b2):
#         x1_original = x1
#         x2_original = x2
#         # x1 = x1 / torch.sum(x1, dim=1, keepdim=True)
#         # x2 = x2 / torch.sum(x2, dim=1, keepdim=True)
#         x1 = torch.log1p(x1)
#         x2 = torch.log1p(x2)
        
#         prior_mu1, prior_logvar1 = torch.chunk(self.prior_net(torch.zeros(x1.shape[0],device=self.device).reshape(-1,1)), 2, dim=1)
#         prior_mu2, prior_logvar2 = torch.chunk(self.prior_net(torch.ones(x2.shape[0],device=self.device).reshape(-1,1)), 2, dim=1)
        
#         # z1, mu1, logvar1 = self.encoder1(x1,b1)
#         # z2, mu2, logvar2 = self.encoder2(x2,b2)
#         z1_shared, mu1_shared, logvar1_shared = self.encoder1_shared(x1,b1)
#         z2_shared, mu2_shared, logvar2_shared = self.encoder2_shared(x2,b2)
#         z1_specific, mu1_specific, logvar1_specific = self.encoder1_specific(x1,b1)
#         z2_specific, mu2_specific, logvar2_specific = self.encoder2_specific(x2,b2)
#         z1 = torch.cat([z1_shared,z1_specific],dim=1)
#         z2 = torch.cat([z2_shared,z2_specific],dim=1)
#         # mu1_specific = self.mu_layer_specific_rna(mu1)
#         # logvar1_specific = self.logvar_layer_specific_rna(logvar1)
#         # mu2_specific = self.mu_layer_specific_atac(mu2)
#         # logvar2_specific = self.logvar_layer_specific_atac(logvar2)
#         # mu1_shared = self.mu_layer_shared(mu1)
#         # logvar1_shared = self.logvar_layer_shared(logvar1)
#         # mu2_shared = self.mu_layer_shared(mu2)
#         # logvar2_shared = self.logvar_layer_shared(logvar2)
#         # z1_specific = self.reparameterize(mu1_specific, logvar1_specific)
#         # z1_shared = self.reparameterize(mu1_shared, logvar1_shared)
#         # z2_specific = self.reparameterize(mu2_specific, logvar2_specific)
#         # z2_shared = self.reparameterize(mu2_shared, logvar2_shared)       
        
#         refined_mu_1, refined_logvar1, refined_mu_2, refined_logvar2 = \
#             self.attention_fusion_layer(torch.cat([mu1_shared, logvar1_shared], dim=1), 
#                                         torch.cat([mu2_shared, logvar2_shared], dim=1))
#         # refined_mu_1, refined_logvar1, refined_mu_2, refined_logvar2 = \
#         #     self.attention_fusion_layer(torch.cat([mu1, logvar1], dim=1), torch.cat([mu2, logvar2], dim=1))
#         aligned_z1 = self.reparameterize(refined_mu_1, refined_logvar1)
#         aligned_z2 = self.reparameterize(refined_mu_2, refined_logvar2)
        
#         # aligned_z1,aligned_z2 = self.attention_fusion_layer(torch.cat([mu1, logvar1], dim=1), torch.cat([mu2, logvar2], dim=1))
#         # z1_new = torch.cat([z1_shared,aligned_z1],dim=1)
#         # z2_new = torch.cat([z2_shared,aligned_z2],dim=1)
#         # z1_new = torch.cat([z1_shared,z1_specific],dim=1)
#         # z2_new = torch.cat([z2_shared,z2_specific],dim=1)
        
#         # rho1,dispersion1,pi1 = self.decoder1(z1_new, b1)
#         # rho2,_,pi2, = self.decoder2(z2_new, b2)
        
#         rho1,dispersion1,pi1 = self.decoder1(z1, b1) #(aligned_z1, b1)
#         rho2,dispersion2,pi2 = self.decoder2(z1, b1)  #(aligned_z2, b2)
#         rho1_,dispersion1_,pi1_ = self.decoder1_(aligned_z1, b1)
#         rho2_,dispersion2_,pi2_ = self.decoder2_(aligned_z2, b2)
        
#         # rho1 = self.decoder1(aligned_z1, b1)
#         # rho2 = self.decoder1(aligned_z2, b2)
#         s1 = self.sample_sequencing_depth(x1_original)
#         s2 = self.sample_sequencing_depth(x2_original)
        
#         #loss
#         zinb_loss = ZINBLoss()
#         zib_loss = ZIBLoss()
        
#         # # Structural preserve
#         # preserve_loss = laplacian_loss(z1_specific,aligned_z1)+laplacian_loss(z2_specific,aligned_z2)
#         # preserve_loss = laplacian_loss(x1,z1_new)+laplacian_loss(x2,z2_new)
#         # preserve_loss = laplacian_loss(x1,aligned_z1)+laplacian_loss(x2,aligned_z2)
#         # preserve_loss = local_structure_loss(torch.cat([mu1,logvar1],dim=1),torch.cat([refined_mu_1,refined_logvar1],dim=1))+\
#         #     local_structure_loss(torch.cat([mu2,logvar2],dim=1),torch.cat([refined_mu_2,refined_logvar2],dim=1))
#         preserve_loss = local_structure_loss(z1,aligned_z1)+\
#             local_structure_loss(z2,aligned_z2)
        
#         # MSE between RNA_z and ATAC_z for paired data  
#         if self.paired == True:
#             cost_matrix = ot.dist(aligned_z1, aligned_z2, metric='sqeuclidean', p=2)
#             D = torch.eye(x1.shape[0]).to(self.device)
#             align_loss = wasserstein_loss(cost_matrix, D) #+ frobenius_alignment_loss(aligned_z1, aligned_z2, D)
#             # align_loss = nn.MSELoss()(aligned_z1, aligned_z2)
#         else:
#             p = ot.unif(x1.shape[0])
#             q = ot.unif(x2.shape[0])
#             # cost_matrix = ot.dist(x1, x2, metric='sqeuclidean', p=2)
#             cost_matrix = ot.dist(aligned_z1, aligned_z2, metric='sqeuclidean', p=2)
#             D = ot.sinkhorn(p, q, cost_matrix, reg=0.1)
#             D = torch.tensor(D, dtype=torch.float32).to(self.device).detach()
#             align_loss = wasserstein_loss(cost_matrix, D) #+ frobenius_alignment_loss(aligned_z1, aligned_z2, D)
#             # align_loss = nn.MSELoss()(aligned_z1, D@aligned_z2)
        
#         # reconstruct loss
#         recon_loss_rna = zinb_loss(x1_original, rho1, dispersion1, pi1, s1, eps = self.eps)
#         recon_loss_atac = zinb_loss(x2_original, rho2, dispersion2, pi2, s2, eps = self.eps)
#         recon_loss_rna_ = zinb_loss(x1_original, rho1_, dispersion1_, pi1_, s1, eps = self.eps)
#         recon_loss_atac_ = zinb_loss(x2_original, rho2_, dispersion2_, pi2_, s2, eps = self.eps)
        
#         # recon_loss_rna = nn.MSELoss(reduction='mean')(x1, rho1)
#         # recon_loss_atac = nn.MSELoss(reduction='mean')(x2, rho2)
        
#         # KL divergence of z
#         # kl_z1 = klLoss(torch.cat([mu1_specific,mu1_shared],dim=1), torch.cat([logvar1_specific, logvar1_shared],dim=1))
#         # kl_z2 = klLoss(torch.cat([mu2_specific,mu2_shared],dim=1), torch.cat([logvar2_specific, logvar2_shared],dim=1))  
        
#         # kl_z1 = klLoss(mu1_specific, logvar1_specific) + klLoss(mu1_shared, logvar1_shared)
#         # kl_z2 = klLoss(mu2_specific, logvar2_specific) + klLoss(mu2_shared, logvar2_shared)
        
#         kl_z1 = klLoss_prior(mu1_specific, logvar1_specific, prior_mu1, prior_logvar1)+\
#             klLoss(mu1_shared, logvar1_shared)
#         kl_z2 = klLoss_prior(mu2_specific, logvar2_specific, prior_mu2, prior_logvar2)+\
#             klLoss(mu2_shared, logvar2_shared)
#         # kl_z1 = klLoss(mu1, logvar1)
#         # kl_z2 = klLoss(mu2, logvar2)
#         kl_z1_ = klLoss(refined_mu_1, refined_logvar1)
#         kl_z2_ = klLoss(refined_mu_2, refined_logvar2)
        
#         # # orthogonality_loss
#         # or_loss1 = HSICloss(z1_specific,z1_shared)
#         # or_loss2 = HSICloss(z2_specific,z2_shared)
        
#         loss_rna = recon_loss_rna + 5*kl_z1
#         loss_atac = recon_loss_atac + 5*kl_z2
#         loss_rna_ = recon_loss_rna_ + kl_z1_
#         loss_atac_ = recon_loss_atac_ + kl_z2_
#         # lambda_hsic1 = loss_rna.detach().mean() / or_loss1.detach().mean()
#         # lambda_hsic2 = loss_atac.detach().mean() / or_loss2.detach().mean()
#         # lambda_hsic1,lambda_hsic2 = 1,1
#         # or_loss = lambda_hsic1*or_loss1 + lambda_hsic2*or_loss2
        
#         total_loss = loss_rna + loss_atac + loss_rna_ + loss_atac_ + self.beta * align_loss + self.eta * preserve_loss #+ or_loss
#         loss_dict = {'total_loss':total_loss.item(), 
#                      'loss_rna':loss_rna.item(), 'loss_atac':loss_atac.item(), 
#                      'loss_rna_':loss_rna_.item(), 'loss_atac_':loss_atac_.item(), 
#                      'preserve_loss':preserve_loss.item(), 
#                      'align_loss':align_loss.item(), 
#                      'recon_loss_rna':recon_loss_rna.item(),'kl_z_rna':kl_z1.item(),
#                      'recon_loss_atac':recon_loss_atac.item(),'kl_z_atac':kl_z2.item(),
#                      'recon_loss_rna_':recon_loss_rna_.item(),'kl_z_rna_':kl_z1_.item(),
#                      'recon_loss_atac_':recon_loss_atac_.item(),'kl_z_atac_':kl_z2_.item()
#                      } 
#         #,'or_loss_rna':or_loss1.item(),'or_loss_atac':or_loss2.item()}
        
#         return aligned_z1, aligned_z2, total_loss, loss_dict
    
#     def sample_sequencing_depth(self, x, strategy="observed"):
#         if strategy=="batch_sample":
#             # batch empirically sample
#             mu_s = torch.log(x.sum(dim=1) + 1.0).mean()
#             sigma_s = torch.log(x.sum(dim=1) + 1.0).std()
#             log_s = mu_s + sigma_s * torch.randn_like(sigma_s)
#             s = torch.exp(log_s)
#             s = s.detach()
#         elif strategy == "observed":
#             log_s = torch.log(x.sum(dim=1)).unsqueeze(1)
#             s = torch.exp(log_s)
#             s = s.detach()
#         return s
    
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std
    
    
    
# class VAE(nn.Module):
#     def __init__(self, device, input_dim1=784, input_dim2=784, batch_dim = 1, cluster_dim = 10, layer_dims=[500,100], latent_dim=20, 
#                 fusion_hidden_dim = 20, num_heads = 4,dropout_rate = 0.5, 
#                 eta = [0.25,0.25,0.25,0.25], paired = True, eps=1e-10):
#         super(VAE, self).__init__()
        
#         self.eta = eta
#         self.device = device
#         self.eps = eps
#         self.paired= paired
#         self.input_dim1 = input_dim1
#         self.input_dim2 = input_dim2
#         self.latent_dim = latent_dim
#         self.batch_dim = batch_dim
#         self.cluster_dim = cluster_dim
        
#         # q(y|x,b)
#         layers_yx1 = []
#         current_dim = input_dim1 + batch_dim
#         for dim in layer_dims:
#             layers_yx1.append(nn.Linear(current_dim, dim))
#             layers_yx1.append(nn.LeakyReLU())
#             layers_yx1.append(nn.BatchNorm1d(dim))
#             layers_yx1.append(nn.Dropout(dropout_rate))
#             current_dim = dim
#         self.yx_encoder1 = nn.Sequential(*layers_yx1)
#         self.get_y1 = GumbelSoftmax(layer_dims[-1], cluster_dim)
        
#         layers_yx2 = []
#         current_dim = input_dim2 + batch_dim
#         for dim in layer_dims:
#             layers_yx2.append(nn.Linear(current_dim, dim))
#             layers_yx2.append(nn.LeakyReLU())
#             layers_yx2.append(nn.BatchNorm1d(dim))
#             layers_yx2.append(nn.Dropout(dropout_rate))
#             current_dim = dim
#         self.yx_encoder2 = nn.Sequential(*layers_yx2)
#         self.get_y2 = GumbelSoftmax(layer_dims[-1], cluster_dim)
        
#         # q(z|x,y,b)
#         layers_zxy1 = []
#         current_dim = input_dim1 + batch_dim + cluster_dim
#         for dim in layer_dims:
#             layers_zxy1.append(nn.Linear(current_dim, dim))
#             layers_zxy1.append(nn.LeakyReLU())
#             layers_zxy1.append(nn.BatchNorm1d(dim))
#             layers_zxy1.append(nn.Dropout(dropout_rate))
#             current_dim = dim
#         self.zxy_encoder1 = nn.Sequential(*layers_zxy1)
        
#         layers_zxy2 = []
#         current_dim = input_dim2 + batch_dim + cluster_dim
#         for dim in layer_dims:
#             layers_zxy2.append(nn.Linear(current_dim, dim))
#             layers_zxy2.append(nn.LeakyReLU(0.1))
#             layers_zxy2.append(nn.BatchNorm1d(dim))
#             layers_zxy2.append(nn.Dropout(dropout_rate))
#             current_dim = dim
#         self.zxy_encoder2 = nn.Sequential(*layers_zxy2)
        
#         # latent perameters 
#         self.mu_layer1 = nn.Linear(layer_dims[-1], latent_dim)
#         self.logvar_layer1 = nn.Linear(layer_dims[-1], latent_dim)
#         self.mu_layer2 = nn.Linear(layer_dims[-1], latent_dim)
#         self.logvar_layer2 = nn.Linear(layer_dims[-1], latent_dim)
        
#         # # Sequencing depth layer
#         # self.fc_mu_s1 = nn.Linear(input_dim1+batch_dim, 1)
#         # self.fc_logvar_s1 = nn.Linear(input_dim1+batch_dim, 1)
#         # self.fc_mu_s2 = nn.Linear(input_dim2+batch_dim, 1)
#         # self.fc_logvar_s2 = nn.Linear(input_dim2+batch_dim, 1)
        
#         # decoder     
#         r_layers1 = []
#         current_dim =  latent_dim + batch_dim
        
#         for dim in reversed(layer_dims):
#             r_layers1.append(nn.Linear(current_dim, dim))
#             r_layers1.append(nn.LeakyReLU(0.1))
#             r_layers1.append(nn.BatchNorm1d(dim))
#             r_layers1.append(nn.Dropout(dropout_rate))
#             current_dim = dim
        
#         # r_layers2 = r_layers1.copy()
#         r_layers2 = []
#         current_dim =  latent_dim + batch_dim
        
#         for dim in reversed(layer_dims):
#             r_layers2.append(nn.Linear(current_dim, dim))
#             r_layers2.append(nn.LeakyReLU())
#             r_layers2.append(nn.BatchNorm1d(dim))
#             r_layers2.append(nn.Dropout(dropout_rate))
#             current_dim = dim

#         self.decoder1 = nn.Sequential(*r_layers1)
#         self.decoder2 = nn.Sequential(*r_layers2)
        
#         self.mean_layer1 = nn.Linear(layer_dims[0], input_dim1)
#         self.dispersion_layer1 = nn.Linear(layer_dims[0], input_dim1)
#         self.dropout_layer1 = nn.Sequential(
#             nn.Linear(layer_dims[0], input_dim1),
#             nn.Sigmoid())
        
#         self.mean_layer2 = nn.Linear(layer_dims[0], input_dim2)
#         self.dispersion_layer2 = nn.Linear(layer_dims[0], input_dim2)
#         self.dropout_layer2 = nn.Sequential(
#             nn.Linear(layer_dims[0], input_dim2),
#             nn.Sigmoid())
        
#         # self.rnn = nn.GRU(2*latent_dim, 2*latent_dim, batch_first=True)
#         self.attention_fusion_layer = AttentionFusionLayer(latent_dim, latent_dim, num_heads, 
#                                                            fusion_hidden_dim, latent_dim, paired=paired)
        
#     def forward(self,x1,b1,x2,b2,m1,std1,m2,std2):
#         x1_original = x1
#         x2_original = x2
        
#         x1 = torch.log(x1+1)
#         x2 = torch.log(x2+1)
#         # m1 = x1.mean(dim=0)
#         # std1 = x1.std(dim=0)
#         # m2 = x2.mean(dim=0)
#         # std2 = x2.std(dim=0)
#         x1 = (x1-m1)/(std1+1e-5)
#         x2 = (x2-m2)/(std2+1e-5)
        
#         # mu1, logvar1, s1, mu_s1, logvar_s1 = self.encode(x1, b1, modal="RNA")
#         # mu2, logvar2, s2, mu_s2, logvar_s2 = self.encode(x2, b2, modal="ATAC")   
#         mu1, logvar1, y1, ylogits1, yprob1 = self.encode(x1, b1, modal="RNA")
#         s1 = self.sample_sequencing_depth(x1_original)
#         mu2, logvar2, y2, ylogits2, yprob2 = self.encode(x2, b2, modal="ATAC")
#         s2 = self.sample_sequencing_depth(x2_original)
#         z1 = self.reparameterize(mu1, logvar1)
#         z2 = self.reparameterize(mu2, logvar2)
        
#         aligned_z1, aligned_z2 = self.attention_fusion_layer(z1, z2)
#         # if self.paired == True:
#         #     refined_z1 = fused_z
#         #     refined_z2 = fused_z
#         # else:
#         #     refined_z1 = fused_z[:x1.size(0), :]  # First half for z1: Shape (batch_size, z_dim)
#         #     refined_z2 = fused_z[x1.size(0):, :]  # Second half for z2: Shape (batch_size, z_dim)
        
#         # z = torch.cat([z1, z2, fused_z],dim=1)
#         # z1 = (z1+refined_z1)/2 #torch.cat([z1, refined_z1],dim=1)
#         # z2 = (z2+refined_z2)/2 #torch.cat([z2, refined_z2],dim=1)
#         # z = torch.cat([z1, z2],dim=1)
#         rho1,dispersion1,pi1 = self.decode1(aligned_z1, b1)
#         rho2,pi2 = self.decode2(aligned_z2, b2)
        
#         # loss: refined_z和z之间；
        
#         # rho1,dispersion1,pi1 = torch.exp(rho1*std1+m1)-1,torch.exp(dispersion1*std1+m1)-1,torch.sigmoid(torch.exp(pi1*std1+m1)-1)
#         # rho2,pi2 = torch.exp(rho2*std2+m2)-1,torch.sigmoid(torch.exp(pi2*std2+m2)-1)
        
#         # rho1,dispersion1 = torch.exp(rho1*std1+m1)-1,torch.exp(dispersion1*std1+m1)-1
#         # rho2 = torch.exp(rho2*std2+m2)-1
        
#         # loss_rna = rna_loss(x1, rho1, dispersion1, pi1, mu1, logvar1, s1, mu_s1, logvar_s1,eps=self.eps)
#         # loss_atac = atac_loss(x2, rho2, pi2, mu2, logvar2, s2, mu_s2, logvar_s2,eps=self.eps)
#         mse_loss = nn.MSELoss()
#         preserve_loss = mse_loss(z1,aligned_z1)+mse_loss(z2,aligned_z2)
#         if self.paired == True:
#             align_loss = mse_loss(aligned_z1,aligned_z2)
#         else:
#             align_loss = 0
#         loss_rna = rna_loss(x1_original, rho1, dispersion1, pi1, mu1, logvar1, s1, eps=self.eps)
#         loss_atac = atac_loss(x2_original, rho2, pi2, mu2, logvar2, s2, eps=self.eps)
#         # total_loss = self.eta*loss_rna/loss_rna.detach()+(1-self.eta)*loss_atac/loss_atac.detach()
#         total_loss = self.eta[0]*loss_rna+self.eta[1]*loss_atac+self.eta[2]*preserve_loss+self.eta[3]*align_loss
        
#         return aligned_z1, aligned_z2, total_loss, loss_rna, loss_atac, preserve_loss, align_loss
            
#     def encode(self,x,b,modal):
#         if self.batch_dim > 0:
#             x = torch.cat([x,b],dim=1)
#         if modal=='RNA':
#             logits, prob, y = self.yx_encoder1(x)
#             x = torch.cat([x,y],dim=1)
#             h = self.zxy_encoder1(x)
#             mu = self.mu_layer1(h)
#             logvar = self.logvar_layer1(h)
#             # s = torch.exp(self.depth_layer1(h))  # Ensure positive depth
#         elif modal == 'ATAC':
#             logits, prob, y = self.yx_encoder2(x)
#             x = torch.cat([x,y],dim=1)
#             h = self.zxy_encoder2(x)
#             mu = self.mu_layer2(h)
#             logvar = self.logvar_layer2(h)
#             # s = torch.exp(self.depth_layer2(h))  # Ensure positive depth
#         # s, mu_s, sigma_s = self.sample_sequencing_depth(x, modal=modal)
#         return mu, logvar, y, logits, prob #, mu_s, sigma_s
    
#     def decode1(self, z, b):
#         # zs = torch.cat([z, s.unsqueeze(1)], dim=1)  # Concatenate latent variable and sequencing depth
#         z = torch.cat([z, b],dim=1)
#         h = self.decoder1(z)
#         rho = torch.exp(self.mean_layer1(h))  # Ensure positive outputs
#         dispersion = torch.exp(self.dispersion_layer1(h))
#         pi = self.dropout_layer1(h)        
#         return rho, dispersion, pi
    
#     def decode2(self, z, b):
#         z = torch.cat([z, b],dim=1)
#         h = self.decoder2(z)
#         rho = self.mean_layer2(h)
#         pi = self.dropout_layer2(h) 
#         return rho,pi
        
#     def sample_sequencing_depth(self, x): #, modal):
#         mu_s = torch.log(x.sum(dim=1) + 1).mean()
#         sigma_s = torch.log(x.sum(dim=1) + 1).std()
#         # if modal == "RNA":
#         #     mu_s = self.fc_mu_s1(x)
#         #     logvar_s = torch.clamp(0.5*self.fc_logvar_s1(x),max=10)
#         #     sigma_s = torch.exp(logvar_s)
#         # elif modal == "ATAC":
#         #     mu_s = self.fc_mu_s2(x)
#         #     logvar_s = torch.clamp(0.5*self.fc_logvar_s2(x),max=10)
#         #     sigma_s = torch.exp(logvar_s)
#         logs = mu_s + sigma_s * torch.randn_like(sigma_s)
#         s = torch.exp(logs)
#         return s #, mu_s, logvar_s
    
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std
    
    