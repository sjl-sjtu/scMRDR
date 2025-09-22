import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
from sklearn.neighbors import kneighbors_graph

class ZINBLoss(nn.Module):
    """
    Zero-Inflated Negative Binomial Loss
    This loss function is used for modeling count data with excess zeros.
    It combines a zero-inflated component with a negative binomial distribution.
    Args:
        x: observed count data (batch_size, num_features)
        rho: mean parameter of the negative binomial distribution (batch_size, num_features)
        dispersion: dispersion parameter of the negative binomial distribution (batch_size, num_features)
        pi: zero-inflation probability (batch_size, num_features)
        s: scaling factor (batch_size, num_features)
        mask: optional mask to ignore certain elements in the loss computation (batch_size, num_features)
        eps: small value to avoid log(0) (default: 1e-8)
    Returns:
        mean_loss: mean loss value across the batch
    """
    def __init__(self):
        super(ZINBLoss, self).__init__()

    def forward(self, x, rho, dispersion, pi, s, mask=None, eps=1e-8):
        # P_NB(x; mu,r) = Gamma(x+r)/[Gamma(r)Gamma(x+1)] * [r/(r+mu)]^r * [mu/(r+mu)]^x
        # logP_NB(x) = logGamma(x+r) - logGamma(r) - logGamma(x+1) + rlog(r) - rlog(r+mu) + xlog(mu) - xlog(r+mu)
        # -logP_NB(x) = [-logGamma(x+r) + logGamma(r) + logGamma(x+1)] + [- rlog(r) - xlog(mu) + (r+x)log(r+mu)]
        
        mean = torch.clamp(rho * s, min=eps)
        dispersion = torch.clamp(dispersion, min=eps)

        # negative likelihood of NB
        # t1 = -logGamma(x+r) + logGamma(r) + logGamma(x+1)
        t1 = torch.lgamma(dispersion) + torch.lgamma(x + 1.0) - torch.lgamma(x + dispersion) 
        # t2 = - rlog(r) - xlog(mu) + (r+x)log(r+mu)
        t2 = -dispersion * torch.log(dispersion) - x * torch.log(mean) + (dispersion + x) * torch.log(dispersion + mean)
        nb_final = t1 + t2
        
        # zero-inflation
        zero_nb = torch.exp(dispersion * (torch.log(dispersion) - torch.log(dispersion + mean)))  # P_{NB}(x=0) = [r/(r+mu)]^r = exp{r[log(r)-log(r+mu)]}
        # zero_nb = torch.pow(dispersion / (dispersion + mean), dispersion)  # P_{NB}(x=0)
        zero_case = -torch.log(pi + (1.0 - pi) * zero_nb + eps)   # loss when x=0: -log[pi + (1-pi)*P_{NB}(x=0)]
        nb_case = nb_final - torch.log(1.0 - pi + eps)      # loss when x>0: -log[1-pi]-log[P_{NB}(x)]

        loss = torch.where(x <= eps, zero_case, nb_case)
        if mask is not None:
            mean_loss = torch.mean(torch.sum(loss * mask,dim=1)*(x.shape[1]/torch.sum(mask, dim=1)),dim=0) #
        else:
            mean_loss = torch.mean(torch.sum(loss,dim=1),dim=0)
        return mean_loss


def mseLoss(x,y,mask=None):
    """
    Mean Squared Error Loss
    Args:
        x: predicted values (batch_size, num_features)
        y: target values (batch_size, num_features)
        mask: optional mask to ignore certain elements in the loss computation (batch_size, num_features)
    Returns:
        mean_loss: mean squared error loss across the batch
    """
    loss = (x-y).pow(2)
    if mask is not None:
        mean_loss = torch.mean(torch.sum(loss * mask,dim=1)*(x.shape[1]/torch.sum(mask, dim=1)),dim=0) #
    else:
        mean_loss = torch.mean(torch.sum(loss,dim=1),dim=0)
    return mean_loss # torch.mean(torch.sum(loss,dim=1),dim=0)
    

def klLoss(mu, logvar):
    """
    Compute KL divergence between q(z|x) ~ N(mu, exp(logvar)) and p(z) ~ N(0, 1).
    Args:
        mu: Mean of q(z|x) (batch_size, latent_dim)
        logvar: Log variance of q(z|x) (batch_size, latent_dim)
    Returns:
        - KL divergence for each sample in the batch (scalar).
    """
    kl = torch.mean(
        -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar), dim=1),dim=0)
    return kl

def klLoss_prior(mu_q, logvar_q, mu_p, logvar_p):
    """ 
    Compute KL(q || p) for two Gaussians q(z|x) ~ N(mu_p, exp(logvar_p)) and p(z) ~ N(mu_q, exp(logvar_q))
    
    Args:
        mu_q: mean of q
        logvar_q: log variance of q
        mu_p: mean of p
        logvar_p: log variance of p
    
    Returns:
        kl: KL divergence
    """
    # Compute KL divergence
    kl = -0.5 * torch.sum(
        - torch.exp(logvar_q - logvar_p)  # sigma_q^2 / sigma_p^2
        - ((mu_q - mu_p).pow(2)) * torch.exp(-logvar_p)  # (mu_q - mu_p)^2 / sigma_p^2
        + 1
        - logvar_p  # log(sigma_p^2)
        + logvar_q,  # - log(sigma_q^2)
        dim=1  # Sum over latent dimensions
    )
    return torch.mean(kl,dim=0)  # mean over batch 
    # return -0.5 * torch.sum(1 + logvar_q - logvar_p - 
    #                         (mu_q - mu_p).pow(2) / torch.exp(logvar_p) - 
    #                         torch.exp(logvar_q) / torch.exp(logvar_p))


# structure preserve
def isometric_loss(X, X_prime, m, p=2):
    """
    Compute Isometric Loss while preserving the structure within each class separately.

    Args:
        X: Feature matrix in the original space (batch_size, feature_dim)
        X_prime: Feature matrix in the latent space (batch_size, latent_dim)
        m: One-hot encoded class labels (batch_size, num_classes)
        p: Norm type for distance computation (default: Euclidean distance, p=2)

    Returns:
        loss: Isometric Loss (Mean Squared Error between pairwise distances within each class)
    """
    # X = X.detach()
    
    # Compute pairwise distance matrices using PyTorch's cdist
    D_X = torch.cdist(X, X, p=p)  # Distance matrix in original space
    D_X_prime = torch.cdist(X_prime, X_prime, p=p)  # Distance matrix in latent space

    # Convert one-hot class labels to class similarity mask
    mask = (m @ m.T).float()  # (batch_size, batch_size), 1 for same class, 0 otherwise
    
    D_X = D_X * mask
    D_X_prime = D_X_prime * mask

    # Compute MSE loss only for distances within the same class
    loss = F.mse_loss(D_X_prime, D_X, reduction='sum') / X.shape[0] # Element-wise masking
    return loss

