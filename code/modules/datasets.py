''' IMPORTS '''

from torch.utils.data import Dataset
import os

class CharDataset(Dataset):
    '''
    Gets left and right sequences for every character in the dataset
    '''
    
    def __init__(self, article_directory, label_directory):
        self.article_directory = article_directory
        self.label_directory = label_directory
        
        corpus = ''
        