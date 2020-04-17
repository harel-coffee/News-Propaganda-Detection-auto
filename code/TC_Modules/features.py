from transformers import *
import time
import os
import numpy as np
import pandas as pd
import re
import itertools
from tqdm import tqdm
from tqdm import  tqdm_notebook
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder as LE
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import glob
import os.path
import sys
import codecs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import datetime
import warnings
warnings.filterwarnings('ignore')
from scipy.sparse import hstack
import random
import tensorflow as tf

tfidf_c_config={
    'min' : 5,
    'ng_l' : 1,
    'ng_h' :6,
    'max_features' : 1500
}

tfidf_w_config={
    'min' : 3,
    'ng_l' : 1,
    'ng_h' :3,
    'max_features' : 2000
}

class FeatureExtraction:
    def __init__(self, train_data, dev_data, tfidf_c_config=tfidf_c_config, tfidf_w_config=tfidf_w_config):
        self.train_data, self.dev_data = train_data, dev_data
        self.tfidf_c=TfidfVectorizer(sublinear_tf=True, min_df=tfidf_c_config['min'],ngram_range=(tfidf_c_config['ng_l'],tfidf_c_config['ng_h']),stop_words='english',analyzer='char',max_features=tfidf_c_config['max_features'],lowercase=train_data.lower)
        self.tfidf_w=TfidfVectorizer(sublinear_tf=True, min_df=tfidf_w_config['min'],ngram_range=(tfidf_w_config['ng_l'],tfidf_w_config['ng_h']),stop_words='english',analyzer='word',max_features=tfidf_w_config['max_features'],lowercase=dev_data.lower)
        
    def get_features(self):
        sentences=self.train_data.sentences+self.dev_data.sentences
        spans=np.asarray(self.train_data.spans+self.dev_data.spans).reshape(-1,1)
        self.tfidf_c.fit(sentences)
        self.tfidf_w.fit(sentences)
        train_sentences_c=self.tfidf_c.transform(sentences[:self.train_data.size])
        train_sentences_w=self.tfidf_w.transform(sentences[:self.train_data.size])
        dev_sentences_c=self.tfidf_c.transform(sentences[self.train_data.size:])
        dev_sentences_w=self.tfidf_w.transform(sentences[self.train_data.size:])
        self.train_features=hstack([train_sentences_c, train_sentences_w, spans[:self.train_data.size]])
        self.dev_features=hstack([dev_sentences_c, dev_sentences_w, spans[self.train_data.size:]])