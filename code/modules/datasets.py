''' IMPORTS '''

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os


#ONE-HOT DATASET

class OneHotTdLstmDataset(Dataset):
    def __init__(self, x_right_seqs, x_left_seqs, y, vocab_length, transform=None):
        self.x_right_seqs = x_right_seqs
        self.x_left_seqs = x_left_seqs
        self.y = y
        self.vocab_length = vocab_length
        self.transform = transform
    
    def __len__(self):
        return len(self.x_right_seqs)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        x_r, x_l = self.x_right_seqs[idx], self.x_left_seqs[idx]
        x_r_one_hot, x_l_one_hot = np.zeros((len(x_r), self.vocab_length)), np.zeros((len(x_l), self.vocab_length))
        
        rows_r, cols_r = zip(*[(i, x_r[i]) for i in range(len(x_r))])
        rows_l, cols_l = zip(*[(i, x_l[i]) for i in range(len(x_l))])
 
        x_r_one_hot[rows_r, cols_r] = np.ones(len(x_r))
        x_l_one_hot[rows_l, cols_l] = np.ones(len(x_l))
        
        x_r_one_hot = torch.tensor(x_r_one_hot)
        x_l_one_hot = torch.tensor(x_l_one_hot)
        y_ = torch.tensor(self.y[idx])
        
        return x_r_one_hot, x_l_one_hot, y_
    
        
#FASTTEXT DATASET

class FastTextTdLstmDataset(Dataset):
    def __init__(self, x_right_seqs, x_left_seqs, y, transform=None):
        self.x_right_seqs = x_right_seqs
        self.x_left_seqs = x_left_seqs
        self.y = y
        self.transform = transform
    
    def __len__(self):
        return len(self.x_right_seqs)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        x_r, x_l = self.x_right_seqs[idx], self.x_left_seqs[idx]
        
        x_r_ftxt, x_l_ftxt = torch.cat(x_r, dim=0), torch.cat(x_l, dim=0)
        y_ = torch.tensor(self.y[idx])
        
        return x_r_ftxt, x_l_ftxt, y_