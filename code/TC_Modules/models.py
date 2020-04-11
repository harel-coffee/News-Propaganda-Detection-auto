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
import random

class TransformerModel:
  def __init__(self, device=None, transformer=None, seed=1234):
    self.device=device
    self.train_loss_set = []
    self.predictions=[]

    if device is None:
      self.device='cuda' if torch.cuda.is_available() else 'cpu'

    if transformer is None:
      self.transformer=BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=14).to(self.device)
    else:
      self.transformer=transformer.to(self.device)
      
    self.__seed=seed
    self.seed()

  def seed(self):
    np.random.seed(self.__seed)
    random.seed(self.__seed)
    torch.manual_seed(self.__seed)
    if self.device == 'cuda':
      torch.cuda.manual_seed(self.__seed)
      torch.cuda.manual_seed_all(self.__seed)
      torch.backends.cudnn.enabled = False 
      torch.backends.cudnn.benchmark = False
      torch.backends.cudnn.deterministic = True

  def freeze(self, condition=None):
    if condition is None:
      condition = lambda name : True if 'classifier' in name or 'pooler' in name or '11' in name else False
    for name, param in self.transformer.named_parameters():
      param.requires_grad=condition(name)

  def updater(self,optimizer=None, lr=1e-4, scheduler=None):
    self.optimizer =optimizer
    if self.optimizer is None:
      self.optimizer = AdamW(self.transformer.parameters(), lr=lr, correct_bias=False)
    if scheduler is None:
      max_grad_norm = 1.0
      num_training_steps = 1000
      num_warmup_steps = 100
      self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
      self.clip=max_grad_norm
    elif scheduler != False:
      self.scheduler=scheduler
      

  def train(self,train_dataset, valid_dataset, epochs=1,verbosity=4):
    total_step = len(train_dataset.dataloader)
    verbosity=total_step/verbosity

    for epoch in tqdm_notebook(range(epochs)):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for i, batch in enumerate(train_dataset.dataloader):
          batch = tuple(t.to(self.device) for t in batch)
          b_input_ids, b_input_mask, b_labels = batch
          outputs = self.transformer(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
          loss = outputs[0]
          tr_loss+=loss.item() 
          loss.backward()
          if self.scheduler != False:
          	torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), self.clip)
          self.optimizer.step()
          if self.scheduler != False:
          	self.scheduler.step()
          self.optimizer.zero_grad()
          
          if i % verbosity == verbosity-1:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, i+1, total_step, tr_loss/i))

        train_epoch_accuracy = self.evaluate(train_dataset, mode='train')
        valid_epoch_accuracy = self.evaluate(valid_dataset, mode='valid')
        print ('\033[1m'+'Epoch [{}/{}], Train_micro_avg: {:.4f}, Valid_micro_avg: {:.4f}'.format(epoch+1, epochs,train_epoch_accuracy, valid_epoch_accuracy)+'\033[0m')

  def evaluate(self, dataset, mode = 'train'):
    with torch.no_grad():
      correct, total = 0, 0
      true=[]
      for i, batch in enumerate(dataset.dataloader):
        batch = tuple(t.to(self.device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        outputs = self.transformer(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        prediction = torch.argmax(outputs[0],dim=1)
        total += b_labels.size(0)
        true += [b_labels.cpu()]
        correct+=(prediction==b_labels).sum().item()
        if mode == 'test':
          self.predictions.extend(list(np.asarray(prediction.cpu())))
      
      if mode == 'train' or mode == 'valid':
        return (100*correct/total)
      else:
        self.predictions = dataset.le.inverse_transform(self.predictions)
        return None
  
  def predict(self, test_dataset):
    self.evaluate(test_dataset, mode='test')
    return self.predictions

  def logits(self, dataset):
    logits=[]
    with torch.no_grad():
      for i, batch in enumerate(dataset.dataloader):
        batch = tuple(t.to(self.device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        outputs = self.transformer(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        logits.extend(list(np.asarray(outputs[0].cpu())))
    return logits
