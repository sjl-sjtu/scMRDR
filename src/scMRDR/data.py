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
