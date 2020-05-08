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

def read_articles_from_file_list(folder_name, file_pattern="*.txt"):
    """
    Read articles from files matching patterns <file_pattern> from  
    the directory <folder_name>. 
    The content of the article is saved in the dictionary whose key
    is the id of the article (extracted from the file name).
    Each element of <sentence_list> is one line of the article.
    """
    file_list = glob.glob(os.path.join(folder_name, file_pattern))
    articles = {}
    article_id_list, sentence_id_list, sentence_list = ([], [], [])
    for filename in sorted(file_list):
        article_id = os.path.basename(filename).split(".")[0][7:]
        with codecs.open(filename, "r", encoding="utf8") as f:
            articles[article_id] = f.read()
    return articles


def read_predictions_from_file(filename):
    """
    Reader for the gold file and the template output file. 
    Return values are four arrays with article ids, labels 
    (or ? in the case of a template file), begin of a fragment, 
    end of a fragment. 
    """
    articles_id, span_starts, span_ends, gold_labels = ([], [], [], [])
    with open(filename, "r") as f:
        for row in f.readlines():
            article_id, gold_label, span_start, span_end = row.rstrip().split("\t")
            articles_id.append(article_id)
            gold_labels.append(gold_label)
            span_starts.append(span_start)
            span_ends.append(span_end)
    return articles_id, span_starts, span_ends, gold_labels

def report(true, pred):
    cm=confusion_matrix(true, pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize = (10,8))
    sns.heatmap(cm,annot=True)
    cf_rep=classification_report(true,pred)
    print(cf_rep)

def seed(seed_val=1234):
	random.seed(seed_val)
	np.random.seed(seed_val)
	torch.manual_seed(seed_val)
	torch.cuda.manual_seed_all(seed_val)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def check_cuda():
	device_name = tf.test.gpu_device_name()
	if device_name == '/device:GPU:0':
	    print('Found GPU at: {}'.format(device_name))
	else:
	    raise SystemError('GPU device not found')
	if torch.cuda.is_available():     
	    device = torch.device("cuda")
	    print('There are %d GPU(s) available.' % torch.cuda.device_count())
	    print('We will use the GPU:', torch.cuda.get_device_name(0))
	else:
	    print('No GPU available, using the CPU instead.')
	    device = torch.device("cpu")

def check(_seed=1234):
	seed(seed_val=_seed)
	check_cuda()