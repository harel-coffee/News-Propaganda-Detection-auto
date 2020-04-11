from transformers import *
import time
import os
import numpy as np
import pandas as pd
import re
import itertools
from tqdm import tqdm
from tqdm import  tqdm_notebook
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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
from .utils_tc import *
class Dataset:
    def __init__(self, articles_folder, labels_file):
        self.articles_folder = articles_folder
        self.labels_file = labels_file
        self.articles = read_articles_from_file_list(articles_folder)
        self.read()

    def read(self):
    	articles_id, span_starts, span_ends, self.gold_labels = read_predictions_from_file(self.labels_file)
    	self.spans = [int(end)-int(start) for start, end in zip(span_starts, span_ends)]
    	print("Read %d annotations from %d articles" % (len(span_starts), len(set(articles_id))))
    	self.sentences=[self.articles[id][int(start):int(end)] for id, start, end in zip(articles_id, span_starts, span_ends)]
    	self.size=len(self.sentences)



class SLDataset(Dataset):
    def __init__(self,  articles_folder=None, labels_file=None, lower=True):
        super().__init__(articles_folder, labels_file)
        self.lower=lower

    def clean(self):
        def text_clean(text):
            if self.lower:
                text=text.lower()
            text=re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
            text=re.sub('[“"”]',' " ',text)
            if self.lower:
                retain='[^abcdefghijklmnopqrstuvwxyz!#?". ]'
            else:
                retain='[^abcdefghijklmnopqrstuvwxyzQWERTYUIOPASDFGHJKLZXCVBNM!#?". ]'
            text=re.sub('[()–-]',' ',text)
            text=re.sub(retain,'',text)
            text=re.sub('[.]',' . ',text)
            text=text.replace('?',' ? ')
            text=text.replace('#',' # ')
            text=text.replace('!',' ! ')
            return ' '.join(text.split())
        
        print("Cleaning Sentences")
        self.sentences=[text_clean(sentence) for sentence in self.sentences]

class TransformerDataset(Dataset):
    def __init__(self, articles_folder=None, labels_file=None):
        super().__init__(articles_folder, labels_file)
        self.clean()
        self.sentences = ["[CLS] " + sentence + " [SEP]" for sentence in self.sentences]
        self.le=LE()
        self.labels=self.le.fit_transform(self.gold_labels)

    def clean(self):
        def text_clean(text):
            text=re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
            text=re.sub('[“"”]',' " ',text)
            retain='[^abcdefghijklmnopqrstuvwxyzQWERTYUIOPASDFGHJKLZXCVBNM!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~.0123456789 ]'
            return ' '.join(text.split())
        
        print("Cleaning Sentences")
        self.sentences=[text_clean(sentence) for sentence in self.sentences]
        
    def tokenize(self, tokenizer=BertTokenizer.from_pretrained('bert-base-uncased'), verbosity=True):
        self.tokenizer=tokenizer
        print("Tokenizing")
        self.tokenized_texts = [self.tokenizer.tokenize(sent) for sent in self.sentences]
        if verbosity:
          print("Tokenized \n", self.tokenized_texts[0])
    
    def encode(self, MAX_LEN=90):
      input_ids=[]
      for i in tqdm_notebook(range(len(self.tokenized_texts))):
        input_ids.append(self.tokenizer.convert_tokens_to_ids(self.tokenized_texts[i]))
      
      input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
      attention_masks = []
      # Create a mask of 1s for each token followed by 0s for padding
      for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)
      self.inputs, self.masks, self.labels = torch.tensor(input_ids), torch.tensor(attention_masks), torch.tensor(self.labels)

    def load(self, batch_size=32):
      self.data = TensorDataset(self.inputs, self.masks, self.labels)
      self.dataloader = DataLoader(self.data, shuffle=False, batch_size=batch_size)