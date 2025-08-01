import torch
import numpy as np
import pandas as pd
import torch.utils.data as Data
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset

class CombinedDataset(Dataset):
    '''
    Dataset for combined data.
    Args:
        X: (n, d) feature matrix
        b: (n, ) covariates like batches
        m: (n, ) one-hot encoded modality index
        i: (n, ) index to indicate which masked-feature group the sample belongs to
    '''
    def __init__(self, X, b, m, i):
        super(CombinedDataset,self).__init__()
        # self.device = device
        self.X = torch.tensor(X).float()#.to(device)
        self.len = len(X)
        if b is not None:
            self.b = torch.tensor(b).float()#.to(device)
        else:
            self.b = torch.zeros(self.len).float()#.to(device)
        self.m = torch.tensor(m).float()#.to(device)
        self.i = torch.tensor(i).float()
        # self.X.requires_grad = True
        # self.b.requires_grad = True
        # self.m.requires_grad = True
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        x_sample = self.X[index]
        b_sample = self.b[index]
        m_sample = self.m[index]
        i_sample = self.i[index]
        
        return x_sample, b_sample, m_sample, i_sample

# class IntegrateDataset(Dataset):
#     def __init__(self, Z1_shared, Z1_specific, Z2_shared, Z2_specific):
#         super(IntegrateDataset,self).__init__()
#         # self.device = device
#         self.Z1_shared = torch.tensor(Z1_shared).float()#.to(device)
#         self.Z2_shared = torch.tensor(Z2_shared).float()#.to(device)
#         self.Z1_specific = torch.tensor(Z1_specific).float()#.to(device)
#         self.Z2_specific = torch.tensor(Z2_specific).float()#.to(device)
#         # self.Z1_shared.requires_grad = True
#         # self.Z2_shared.requires_grad = True
#         self.len_Z1 = len(Z1_shared)
#         self.len_Z2 = len(Z2_shared)
#         self.len = max(self.len_Z1, self.len_Z2)
#         self.lenmin = min(self.len_Z1, self.len_Z2)
    
#     def Z1_len(self):
#         return self.len_Z1
    
#     def Z2_len(self):
#         return self.len_Z2
    
#     def __len__(self):
#         return self.len
    
#     def __getitem__(self, index):
#         Z2_shared_sample = self.Z2_shared[index % self.lenmin]
#         Z2_specific_sample = self.Z2_specific[index % self.lenmin]
#         Z1_shared_sample = self.Z1_shared[index % self.lenmin]  # 当 index 超过 Z1_shared 的长度时，循环加载
#         Z1_specific_sample = self.Z1_specific[index % self.lenmin]
#         return Z1_shared_sample, Z1_specific_sample, Z2_shared_sample, Z2_specific_sample


# class IntegrateDataset(Dataset):
#     def __init__(self, X1, X2):
#         super(IntegrateDataset,self).__init__()
#         # self.device = device
#         self.X1 = torch.tensor(X1).float()#.to(device)
#         self.X2 = torch.tensor(X2).float()#.to(device)
#         # self.X1.requires_grad = True
#         # self.X2.requires_grad = True
#         self.len_X1 = len(X1)
#         self.len_X2 = len(X2)
#         self.len = max(self.len_X1, self.len_X2)
#         self.lenmin = min(self.len_X1, self.len_X2)
    
#     def X1_len(self):
#         return self.len_X1
    
#     def X2_len(self):
#         return self.len_X2
    
#     def __len__(self):
#         return self.len
    
#     def __getitem__(self, index):
#         x2_sample = self.X2[index % self.lenmin]

#         x1_sample = self.X1[index % self.lenmin]  # 当 index 超过 X1 的长度时，循环加载
#         return x1_sample, x2_sample

# def data_loader(device, X1, b1, X2, b2, batch_size, paired=True,shuffle=True,drop_last=True):
#     # X1 = torch.tensor(X1,requires_grad=True).float().to(device)
#     # X2 = torch.tensor(X2,requires_grad=True).float().to(device)
#     # b1 = torch.tensor(b1,requires_grad=True).float().to(device)
#     # b2 = torch.tensor(b2,requires_grad=True).float().to(device)
#     if paired == True:
#         dataset = Data.TensorDataset(X1,b1,X2,b2)  
#     else:
#         dataset = CombinedDataset(X1,b1,X2,b2) 
#     loader = Data.DataLoader(dataset,batch_size,shuffle=shuffle,drop_last=drop_last)
#     return loader