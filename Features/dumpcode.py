#@title
!pip install python-Levenshtein

from google.colab import drive
drive.mount('/content/drive')

#@title
import glob
import os.path
import numpy as np
import sys
import codecs
import json
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
import string
import Levenshtein as lev

#@title
train_folder = "./drive/My Drive/datasets/train-articles" # check that the path to the datasets folder is correct, 
dev_folder = "./drive/My Drive/datasets/dev-articles"     # if not adjust these variables accordingly
train_labels_file = "./drive/My Drive/datasets/train-task2-TC.labels"
dev_template_labels_file = "./drive/My Drive/datasets/dev-task-TC-template.out"
task_TC_output_file = "TFIDF_LR.txt"

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

# loading articles' content from *.txt files in the train folder
articles = read_articles_from_file_list(train_folder)
dev_articles = read_articles_from_file_list(dev_folder)

def read_span(id,span,dev=False):
    if dev:
        return dev_articles[id][span[0]:span[1]]
    else:
        return articles[id][span[0]:span[1]]

#@title
def clean(text):
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text=re.sub('[“"”]',' " ',text)
    retain='[^{}". ]'.format(string.ascii_letters+string.punctuation)
    text=re.sub('[()–-]',' ',text)
    text=re.sub(retain,'',text)
    text=re.sub('[.]',' . ',text)
    for punc in string.punctuation:
      text=text.replace(punc,' '+punc+' ')
    return ' '.join(text.split())

#@title
# loading gold labels, articles ids and sentence ids from files *.task-TC.labels in the train labels folder 
ref_articles_id, ref_span_starts, ref_span_ends, train_gold_labels = read_predictions_from_file(train_labels_file)
print("Loaded %d annotations from %d articles" % (len(ref_span_starts), len(set(ref_articles_id))))

# reading data from the development set
dev_article_ids, dev_span_starts, dev_span_ends, dev_labels = read_predictions_from_file(dev_template_labels_file)

#@title
in_data=[(id, [int(sps),int(spe)])for id, sps, spe in zip(ref_articles_id, ref_span_starts, ref_span_ends)]
df=pd.DataFrame(in_data,columns=['ID','Span'])
df['Sentence']=[read_span(id,span) for id,span in zip(df['ID'].tolist(),df['Span'].tolist())]
df['Sentence']=df['Sentence'].apply(lambda x : clean(x))
df['Target']=train_gold_labels
df=df.drop_duplicates()
df['Span']=df['Span'].apply(lambda x : x[1]-x[0])

dev_data=[(id, [int(sps),int(spe)])for id, sps, spe in zip(dev_article_ids, dev_span_starts, dev_span_ends)]
df_dev=pd.DataFrame(dev_data,columns=['ID','Span'])
df_dev['text']=[read_span(id,span,dev=True) for id,span in zip(df_dev['ID'].tolist(),df_dev['Span'].tolist())]
df_dev['text']=df_dev['text'].apply(lambda x : clean(x))
df_dev['Span']=df_dev['Span'].apply(lambda x : x[1]-x[0])

#@title
label_df=dict()
for label in set(df['Target']):
  label_df[label]=df[df['Target']==label]
labels=list(label_df.keys())

#POS Tagging
NLP=df['Sentence'].apply(lambda x : nlp(x))
def p_pos(doc):
  doc=nlp(doc)
  for token in doc:
      if not token.is_stop:
        print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,token.shape_, token.is_alpha)
  print('\n\n')
  for chunk in doc.noun_chunks:
      print(chunk.text, chunk.root.text, chunk.root.dep_,chunk.root.head.text)
  print('\n\n')
  for ent in doc.ents:
      print(ent.text, ent.start_char, ent.end_char, ent.label_)

def capital_score(s):
  a=0
  for word in s.split():
    if word.isupper():
      a+=1
  return a/(len(s.split())+1)
cap=lambda x : capital_score(x)
df['cap']=df['Sentence'].apply(cap)

def rep_c(df,thresh=75):
  df['rep']=0
  x=pd.Series([0*len(df)])
  for text in df['Sentence'].tolist():
    x=pd.Series([0*len(df)])
    x=df['Sentence'].apply(lambda x : lev.ratio(x,text))
    x=x.apply(lambda x : 1 if x>thresh/100 else 0)
    df['rep']+=x
  df['rep']-=1

%%time
rep_c(df,thresh=70)

plt.figure(figsize=(18,9))
sns.boxplot(x='Target',y='rep',data=df)
plt.xticks(rotation=90)
plt.savefig("Repetition.eps",dpi=1200)

plt.figure(figsize=(18,9))
sns.boxplot(x='Target',y='cap',data=df[df['cap']>0])
plt.xticks(rotation=90)
plt.savefig("CAPITALS.eps",dpi=1200)


#LDA
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(max_df=0.8, min_df=2, stop_words='english')
doc_term_matrix = count_vect.fit_transform(df['Sentence'].values.astype('U'))
from sklearn.decomposition import LatentDirichletAllocation
LDA = LatentDirichletAllocation(n_components=14, random_state=1234)
LDA.fit(doc_term_matrix)

#TFIDF