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
    
# class ZIBLoss(nn.Module):
#     def __init__(self):
#         super(ZIBLoss, self).__init__()
        
#     def forward(self, x, rho, pi, s, eps=1e-10):
#         """
#         Args:
#             x: 观测数据 (batch_size, num_features)，每个元素为 0 或 1
#             pi: 零膨胀概率 (batch_size, num_features)
#             theta: 伯努利分布中的成功概率 (batch_size, num_features)
#             eps: 防止 log(0) 的小数值
#         Returns:
#             loss: 零膨胀伯努利分布的负对数似然损失
#         """
#         theta = torch.sigmoid(rho*s)
#         # 对于 x = 0 的情况
#         log_prob_zero = torch.log(torch.clamp(pi + (1 - pi) * (1 - theta), min=eps))
#         # 对于 x = 1 的情况
#         log_prob_one = torch.log(torch.clamp((1 - pi) * theta, min=eps))
#         # 计算负对数似然损失
#         loss = -torch.where(x <= eps, log_prob_zero, log_prob_one)
#         # 返回 batch 中所有数据的平均损失
#         return torch.mean(torch.sum(loss,dim=1),dim=0)

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
    
# def GMMklLoss(mu, logvar, a, b):
#     """
#     Compute KL divergence between q(z|x) ~ N(mu, exp(logvar)) and p(z) ~ N(a, exp(b)).
    
#     Args:
#         mu: Mean of q(z|x) (batch_size, latent_dim)
#         logvar: Log variance of q(z|x) (batch_size, latent_dim)
#         a: Mean of p(z) (scalar or tensor with shape (latent_dim,))
#         b: Log variance of p(z) (scalar or tensor with shape (latent_dim,))
    
#     Returns:
#         - KL divergence for each sample in the batch (scalar).
#     """
#     # Compute KL divergence
#     kl = -0.5 * torch.sum(
#         - torch.exp(logvar - b)  # sigma_q^2 / sigma_p^2
#         - ((mu - a).pow(2)) / torch.exp(b)  # (mu_q - mu_p)^2 / sigma_p^2
#         + 1
#         - b  # log(sigma_p^2)
#         + logvar,  # - log(sigma_q^2)
#         dim=1  # Sum over latent dimensions
#     )
#     return torch.mean(kl,dim=0)  # mean over batch 

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

# def info_nce_loss(z, y, temperature=0.1):
#     """
#     InfoNCE loss to minimize the mutual information between z and y
#     Args:
#         z (torch.Tensor): Embedding of size (batch_size, embedding_dim)
#         y (torch.Tensor): Labels or another embedding of size (batch_size, embedding_dim)
#         temperature (float): Temperature parameter for scaling similarity
#     """
#     batch_size = z.size(0)
#     similarity_matrix = torch.matmul(z, y.T) / temperature  # Compute the similarity matrix
#     labels = torch.arange(batch_size).to(z.device)  # Diagonal (positive pair) indices

#     loss = F.cross_entropy(similarity_matrix, labels, reduction="sum")/z.shape[0]
#     return loss


# def orthogonalityLoss(z1, z2):
#     # z1 = z1 / torch.norm(z1, dim=1, keepdim=True)
#     # z2 = z2 / torch.norm(z2, dim=1, keepdim=True)
#     inner_products = torch.sum(z1 * z2, dim=1)
#     losses = inner_products ** 2 
#     return torch.sum(losses)

# def orthogonalityLoss(z1, z2):
#     '''
#     Orthogonality Loss to enforce orthogonality between z1 and z2
#     Args:
#         z1: (batch_size, latent_dim)
#         z2: (batch_size, latent_dim)
#     Returns:
#         loss: (scalar)
#     '''
#     return torch.norm(torch.mm(z1.T, z2), p='fro') ** 2

# def rbf_kernel(X, sigma=1.0):
#     '''
#     RBF Kernel to compute the pairwise similarity matrix
#     Args:
#         X: (batch_size, latent_dim)
#         sigma: (float)
#     Returns:
#         K: (batch_size, batch_size)
#     '''
#     pairwise_sq_dists = torch.cdist(X, X, p=2) ** 2  # 计算欧式距离的平方
#     K = torch.exp(-pairwise_sq_dists / (2 * sigma ** 2))  # 计算 RBF 核
#     return K



# def catLoss(logits, prob_cat, num_cluster):
#     '''
#     Entropy loss
#     Args:
#         logits: (batch_size, num_classes)
#         prob_cat: (batch_size, num_classes)
#         num_cluster: (int)
#     Returns:
#         kl_y: (scalar)
#     '''
#     log_q = F.log_softmax(logits, dim=-1) # log(prob_cat)
#     kl_y = torch.mean(torch.sum(prob_cat * log_q, dim=1),dim=0) - np.log(1/num_cluster)
#     return kl_y

# Frobenius 范数损失
# def frobenius_preserve_loss(X, X_prime):
#     """
#     计算 Frobenius 范数损失，目标是计算原始数据 X 和映射后的数据 X_prime 之间的全局结构差异。
#     X: 原始数据 (n_samples, n_features)
#     X_prime: 映射后的数据 (n_samples, n_features)
#     """
#     # 计算原始数据和映射数据的距离矩阵
#     D_X = torch.cdist(X, X, p=2)  # 原始数据的距离矩阵
#     D_X_prime = torch.cdist(X_prime, X_prime, p=2)  # 映射后数据的距离矩阵
    
#     # 计算 Frobenius 范数损失
#     loss = torch.norm(D_X - D_X_prime, p='fro') / X.shape[0]  # 计算两个矩阵的 Frobenius 范数
#     return loss

# # Frobenius损失：||X - MY||_F
# def frobenius_alignment_loss(X, Y, M):
#     """
#     计算Frobenius范数损失：||X - MY||_F
#     X, Y: 形状为(n, d)的张量，表示两个模态的数据
#     M: 形状为(n, n)的OT配对矩阵
#     """
#     # 计算映射后的X
#     X_prime = torch.matmul(M, Y)  # 计算M * Y，得到映射后的数据
    
#     # 计算X和X_prime之间的差异
#     diff = X - X_prime  # 形状为(n, d)
    
#     # 计算Frobenius范数
#     # frobenius_loss = torch.norm(diff, p='fro', dim=1)  # 对每个样本计算Frobenius范数
#     frobenius_loss = torch.norm(diff, p='fro') / X.shape[0]
    
#     # # 对batch样本求平均
#     # frobenius_loss = frobenius_loss.mean()  # 对所有样本的损失取平均
#     return frobenius_loss

# # 基于加权样本对差异的 Local Structure Loss

# # def knn_adjacency_matrix(X, k=5, sigma=1.0):
# #     """
# #     计算高斯核的邻接矩阵 (W)。
    
# #     Args:
# #         X: 输入数据，形状为 (n_samples, n_features) 的 NumPy 数组。
# #         k: 每个点的 k 个最近邻。
# #         sigma: 高斯核参数。
    
# #     Returns:
# #         W: 邻接矩阵，形状为 (n_samples, n_samples)。
# #     """
# #     # 使用 kNN 计算邻接矩阵（稀疏格式）
# #     knn_graph = kneighbors_graph(X.detach().cpu().numpy(), n_neighbors=k, mode='distance', include_self=False)
# #     distances = knn_graph.toarray() # 转为稠密矩阵

# #     # 构造高斯核权重
# #     W = np.exp(-distances**2 / (2 * sigma**2))
# #     W[distances == 0] = 0  # 非邻居的点权重为 0
# #     return W

# # def local_structure_loss(X, X_prime, k=5, sigma=1.0):
# #     """
# #     计算基于加权样本对差异的 Local Structure Loss
# #     X: 原始数据 (n_samples, n_features)
# #     X_prime: 映射后的数据 (n_samples, n_features)
# #     sigma: 高斯核的超参数，用于计算样本对的权重
# #     """
    
# #     W = knn_adjacency_matrix(X,k,sigma)
# #     W = torch.tensor(W, device = X.device).detach()
    
# #     # 计算映射后的样本对差异
# #     dist_prime_matrix = torch.cdist(X_prime, X_prime, p=2)  # 计算映射后数据样本对的欧几里得距离
    
# #     # 计算加权的损失，sum 权重后的样本对差异
# #     weighted_loss = torch.sum(W * dist_prime_matrix)  / X.shape[0]
    
# #     return weighted_loss

# from sklearn.neighbors import NearestNeighbors

# def knn_sigma(x, k=5):
#     """
#     Compute per-sample sigma using KNN distances.

#     Args:
#         x (torch.Tensor): Input tensor of shape (n, d)
#         k (int): Number of nearest neighbors

#     Returns:
#         torch.Tensor: Estimated per-sample sigma values of shape (n,)
#     """
#     x_np = x.cpu().detach().numpy()
#     nbrs = NearestNeighbors(n_neighbors=k).fit(x_np)
#     distances, _ = nbrs.kneighbors(x_np)
#     sigmas = torch.tensor(distances[:, -1], dtype=torch.float32, device=x.device)  # kth neighbor distance
#     return sigmas


# def gaussian_adjacency_matrix(x, sigma=1.0, eps=1e-8):
#     """
#     Construct an adjacency matrix using a Gaussian kernel.

#     Args:
#         x (torch.Tensor): Input tensor of shape (n, d)
#         sigma (float): Gaussian kernel width

#     Returns:
#         torch.Tensor: Adjacency matrix of shape (n, n)
#     """
#     dist_matrix = torch.cdist(x, x, p=2)  # Compute pairwise distances
#     W = torch.exp(-dist_matrix**2 / (2 * sigma**2))  # Apply Gaussian kernel
#     # adj = adj * (1 - torch.eye(adj.size(0), device=adj.device)) # Remove self-loops
#     # adj.fill_diagonal_(0)  # Remove self-loops
#     W = W + torch.eye(W.shape[0], device = x.device) * eps
#     return W

# def laplacian_loss(X,X_prime,eps=1e-8):
#     """
#     Laplacian Loss to preserve local structure.
#     """
#     X_detached = X.detach()
#     W = gaussian_adjacency_matrix(X_detached) # adjacency_matrix
#     D = torch.diag(W.sum(dim=1))+eps  # Degree matrix
#     L = D - W  # Laplacian matrix L = D - W    
#     # d_inv_sqrt = torch.diag(1.0 / torch.sqrt(D))  # Avoid division by zero
#     # L = torch.eye(W.shape[0], device=W.device) - d_inv_sqrt @ W @ d_inv_sqrt #L_norm = I - D^{-1/2} W D^{-1/2}
    
#     loss = torch.trace(X_prime.T @ L @ X_prime) / X_prime.shape[0]
#     return loss

# def wasserstein_loss(cost, M):  #X_prime, Y_prime
#     # 计算Wasserstein距离损失
#     wasserstein_distance = torch.sum(M * cost)  / cost.shape[0] 
    
#     return wasserstein_distance


# def MMD(x, y, kernel='rbf'):
#     """
#     Emprical maximum mean discrepancy. The lower the result the more evidence that distributions are the same.

#     Args:
#         x: first sample, distribution P
#         y: second sample, distribution Q
#         kernel: kernel type such as "multiscale" or "rbf"
#     """
#     xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
#     rx = (xx.diag().unsqueeze(0).expand_as(xx))
#     ry = (yy.diag().unsqueeze(0).expand_as(yy))

#     dxx = rx.t() + rx - 2. * xx # Used for A in (1)
#     dyy = ry.t() + ry - 2. * yy # Used for B in (1)
#     dxy = rx.t() + ry - 2. * zz # Used for C in (1)

#     XX, YY, XY = (torch.zeros(xx.shape,device=xx.device),
#                   torch.zeros(xx.shape,device=xx.device),
#                   torch.zeros(xx.shape,device=xx.device))

#     if kernel == "multiscale":

#         bandwidth_range = [0.2, 0.5, 0.9, 1.3]
#         for a in bandwidth_range:
#             XX += a**2 * (a**2 + dxx)**-1
#             YY += a**2 * (a**2 + dyy)**-1
#             XY += a**2 * (a**2 + dxy)**-1

#     if kernel == "rbf":

#         bandwidth_range = [10, 15, 20, 50]
#         for a in bandwidth_range:
#             XX += torch.exp(-0.5*dxx/a)
#             YY += torch.exp(-0.5*dyy/a)
#             XY += torch.exp(-0.5*dxy/a)

#     return torch.mean(XX + YY - 2. * XY)

# def MMD(x, y, kernel='rbf', sigma=1.0):
#     def gaussian_kernel(a, b, sigma):
#         dist = torch.cdist(a, b) ** 2
#         return torch.exp(-dist / (2 * sigma ** 2))

#     K_xx = gaussian_kernel(x, x, sigma)
#     K_yy = gaussian_kernel(y, y, sigma)
#     K_xy = gaussian_kernel(x, y, sigma)

#     mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
#     return mmd

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

# def sammon_loss(X, X_prime, m, p=2, sigma=1.0, eps=1e-10):
#     # with torch.no_grad():
#     # X = X.detach()
#     D_X = torch.cdist(X, X, p=p)  # Distance matrix in original space
#     # Convert one-hot class labels to class similarity mask
#     mask = (m @ m.T).float()  # (batch_size, batch_size), 1 for same class, 0 otherwise
#     D_X = D_X * mask
#     W = 1-torch.exp(-D_X ** 2 / (2*sigma**2)) 
#     # W = W/W.sum(dim=1, keepdim=True) * W.shape[1]
#     D_X_prime = torch.cdist(X_prime, X_prime, p=p)  # Distance matrix in latent space
#     D_X_prime = D_X_prime * mask
    
#     # weights = (1/(D_X+eps)) / ((1/(D_X+eps)).sum(dim=1,keepdim=True)) * X.shape[0]
    
#     loss = ((D_X - D_X_prime)**2 * W).sum() / X.shape[0]
#     return loss

# def gaussian_adjacency_matrix(x, m, sigma=1.0, eps=1e-8):
#     """
#     Construct an adjacency matrix using a Gaussian kernel, conditioned on a one-hot label m.

#     Args:
#         x (torch.Tensor): Input tensor of shape (n, d)
#         m (torch.Tensor): One-hot label tensor of shape (n, c)
#         sigma (float): Gaussian kernel width

#     Returns:
#         torch.Tensor: Conditioned adjacency matrix of shape (n, n)
#     """
#     dist_matrix = torch.cdist(x, x, p=2)  # Compute pairwise distances
#     W = torch.exp(-dist_matrix**2 / (2 * sigma**2))  # Apply Gaussian kernel

#     # Compute mask: only connect points with the same label
#     mask = (m @ m.T)  # (n, n), 1 if same class, 0 otherwise
#     W = W * mask  # Apply the mask

#     W = W + torch.eye(W.shape[0], device=x.device) * eps  # Avoid zero rows
#     return W

# def laplacian_loss(X, X_prime, m, eps=1e-8):
#     """
#     Laplacian Loss to preserve local structure, conditioned on one-hot labels.

#     Args:
#         X (torch.Tensor): Original data of shape (n, d)
#         X_prime (torch.Tensor): Reconstructed data of shape (n, d)
#         m (torch.Tensor): One-hot labels of shape (n, c)

#     Returns:
#         torch.Tensor: Loss value
#     """
#     # X_detached = X.detach()
#     W = gaussian_adjacency_matrix(X, m)  # Conditioned adjacency matrix
#     D = torch.diag(W.sum(dim=1)) + eps  # Degree matrix
#     L = D - W  # Laplacian matrix

#     loss = torch.trace(X_prime.T @ L @ X_prime) / X_prime.shape[0]
#     return loss

# def knn_adjacency_matrix(X, m, k=5, sigma=1.0):
#     """
#     Compute the adjacency matrix (W) using a Gaussian kernel, conditioned on a one-hot label m.
    
#     Args:
#         X (torch.Tensor): Input data of shape (n_samples, n_features).
#         m (torch.Tensor): One-hot label tensor of shape (n_samples, num_classes).
#         k (int): Number of nearest neighbors.
#         sigma (float): Gaussian kernel parameter.
    
#     Returns:
#         np.ndarray: Conditioned adjacency matrix of shape (n_samples, n_samples).
#     """
#     # Compute kNN adjacency matrix (sparse format)
#     knn_graph = kneighbors_graph(X.detach().cpu().numpy(), n_neighbors=k, mode='distance', include_self=False)
#     distances = knn_graph.toarray()  # Convert to dense matrix

#     # Compute Gaussian kernel weights
#     W = np.exp(-distances**2 / (2 * sigma**2))
#     W[distances == 0] = 0  # Set non-neighboring points to 0

#     # Compute class-wise mask (same class = 1, different class = 0)
#     mask = (m @ m.T).detach().cpu().numpy()  # Convert to NumPy for element-wise multiplication
#     W = W * mask  # Apply the mask

#     return W

# def knn_structure_loss(X, X_prime, m, k=5, sigma=1.0):
#     """
#     Compute the Local Structure Loss, conditioned on the one-hot label m.

#     Args:
#         X (torch.Tensor): Original data of shape (n_samples, n_features).
#         X_prime (torch.Tensor): Mapped data of shape (n_samples, n_features).
#         m (torch.Tensor): One-hot label tensor of shape (n_samples, num_classes).
#         k (int): Number of nearest neighbors.
#         sigma (float): Gaussian kernel parameter.

#     Returns:
#         torch.Tensor: Loss value.
#     """
#     W = knn_adjacency_matrix(X, m, k, sigma)
#     W = torch.tensor(W, device=X.device).detach()

#     # Compute pairwise distances in the mapped space
#     dist_prime_matrix = torch.cdist(X_prime, X_prime, p=2)

#     # Compute weighted loss: sum of distance differences weighted by W
#     weighted_loss = torch.sum(W * dist_prime_matrix) / X.shape[0]

#     return weighted_loss

# def frobenius_isometric_loss(X, X_prime, m):
#     """
#     Compute the Frobenius norm loss, conditioned on the one-hot label m.
#     This loss measures the global structure difference between the original data X
#     and the mapped data X_prime, considering only pairwise distances within the same class.

#     Args:
#         X (torch.Tensor): Original data of shape (n_samples, n_features).
#         X_prime (torch.Tensor): Mapped data of shape (n_samples, n_features).
#         m (torch.Tensor): One-hot label tensor of shape (n_samples, num_classes).

#     Returns:
#         torch.Tensor: Loss value.
#     """
#     # Compute pairwise distance matrices for original and mapped data
#     D_X = torch.cdist(X, X, p=2)  # Distance matrix of original data
#     D_X_prime = torch.cdist(X_prime, X_prime, p=2)  # Distance matrix of mapped data

#     # Compute class-wise mask (1 for same class, 0 for different classes)
#     mask = m @ m.T  # (n_samples, n_samples), ensuring only intra-class distances are considered

#     # Apply the mask to the distance matrices
#     D_X = D_X * mask
#     D_X_prime = D_X_prime * mask

#     # Compute Frobenius norm loss, normalized by the number of samples
#     loss = torch.norm(D_X - D_X_prime, p='fro') / X.shape[0]
    
#     return loss


# def rbf_kernel(X, Y, sigma=1.0):
#     """
#     Compute the RBF (Gaussian) kernel matrix
    
#     Args:
#         X: input samples 2 (batch_size, dim)
#         Y: input samples 2 (batch_size, dim)
#         sigma: RBF kernel width
    
#     Returns:
#         K: RBF kernel matrix
#     """
#     pairwise_sq_dists = torch.cdist(X, Y, p=2) ** 2 
#     K = torch.exp(-pairwise_sq_dists / (2 * sigma ** 2))
#     return K

# def HSICloss(X, Y, sigma=1.0):
#     """
#     Compute Hilbert-Schmidt Independence Criterion (HSIC) between two sets of samples
    
#     Args:
#         X: input samples 1 (batch_size, dim)
#         Y: input samples 2 (batch_size, dim)
#         sigma: RBF kernel width
    
#     Returns:
#         HSIC: Hilbert-Schmidt Independence Criterion (smaller HSIC means more independent)

#     """
#     n = X.shape[0]
#     K = rbf_kernel(X, X, sigma) #rbf_kernel(X, X, gamma=1 / (2 * sigma**2))
#     L = rbf_kernel(Y, Y, sigma) #rbf_kernel(Y, Y, gamma=1 / (2 * sigma**2))
#     H = torch.eye(n).to(X.device) - (1.0 / n) * torch.ones((n, n)).to(X.device)
#     # HSIC: Tr(KHLH) / (n-1)^2
#     # HSIC = torch.trace(K @ H @ L @ H) / ((n - 1) ** 2)
#     HSIC = torch.trace(K @ H @ L @ H)  / ((n - 1) ** 2)
#     return HSIC

# class ZeroInflatedMSELoss(nn.Module):
#     def __init__(self):
#         super(ZeroInflatedMSELoss, self).__init__()
#         # Initialize a learnable parameter for zero-inflation rates
#         # self.zero_inflation_rates = nn.Parameter(torch.ones(num_features) * 0.5)  # Start with 0.5 for all features

#     def forward(self, predictions, targets, zero_inflation_rates):
#         # Ensure predictions and targets are the same shape
#         assert predictions.shape == targets.shape, "Predictions and targets must be the same shape."

#         # Number of samples and features
#         num_samples, num_features = targets.shape

#         # Calculate MSE for non-zero targets
#         non_zero_mask = targets != 0
#         # print(targets.shape)
#         # print(non_zero_mask.shape)
#         if non_zero_mask.any():
#             mse_loss = (predictions[non_zero_mask] - targets[non_zero_mask]) ** 2
#             # print(predictions[non_zero_mask].shape)
#             # print(mse_loss.shape)
#             mse_loss_sum = mse_loss.sum()  # Sum across features
#         else:
#             mse_loss_sum = torch.zeros(num_samples, device=predictions.device).sum()

#         # Calculate loss for zero targets
#         zero_mask = targets == 0
#         if zero_mask.any():
#             zero_loss = (zero_inflation_rates.repeat(num_samples, 1)[zero_mask] * (predictions[zero_mask] ** 2)).sum()
#         else:
#             zero_loss = torch.zeros(num_samples, device=predictions.device).sum()

#         # Combine losses
#         total_loss = mse_loss_sum + zero_loss

#         # Mean across batch samples
#         mean_loss = total_loss/num_samples
#         return mean_loss
