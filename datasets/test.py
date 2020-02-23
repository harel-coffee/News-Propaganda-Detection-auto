import numpy as np
import pandas as pd
import re
import string
from fuzzywuzzy import fuzz

train=pd.read_csv('train.csv')
Sentences=train['Sentence'].tolist()

fuz=pd.DataFrame()

def rep_c(df):
  ret_df=pd.DataFrame()
  for i,text in enumerate(Sentences):
    ret_df[i]=df['Sentence'].apply(lambda x : fuzz.token_set_ratio(x.lower(),text.lower()))
    if i%50==0:
    	print(i)
  return ret_df

fuz=rep_c(train)
fuz.to_csv('Fuzz_table.csv')
