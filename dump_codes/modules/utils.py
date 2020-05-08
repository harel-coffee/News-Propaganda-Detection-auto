''' IMPORTS '''

import os
import csv
import re
import sys 
sys.setrecursionlimit(10**6) 


def read_article(article_id):
    '''
    returns article raw text and span offsets given the article id
    '''
    
    article_fname = '../datasets/train-articles/article' + str(article_id) + '.txt'
    label_fname = '../datasets/train-labels-task1-span-identification/article' + str(article_id) + '.task1-SI.labels' 
    with open(article_fname, newline = '\n') as article:
        raw = article.read()
    with open(label_fname, newline = '\n') as label_file:
        labels = []
        labels_list = csv.reader(label_file, delimiter='\t')
        for x in labels_list:
            labels.append(x)
        spans = [[int(span_loc[1]), int(span_loc[2])] for span_loc in labels]
    return raw, spans


def get_spans(article_id):
    '''
    returns span texts
    '''
    
    raw, spans = read_article(article_id)
    span_texts = []
    for span in spans:
        span_texts.append(raw[span[0]:span[1]])
    return span_texts


def label_article_tokens(article_id):
    '''
    returns a binary 0/1 token-level lablled data propaganada span for train-set
    '''
    
    raw, spans = read_article(article_id)    #reading raw data and span locations
    token_list = raw.split()                 #splitting into space separated tokens
    
    first_letter_index = []                  #creating a list of charecter index of the first charecter of every token
    index = 0
    for token in token_list:
        success = False
        while not success:
            if raw[index] != '\n':
                first_letter_index.append(index)
                index += len(token) + 1
                success = True
            else:
                index += 1
         
    binary_labels = [0] * len(token_list)    #Assigning label of 1 (is propanagada) if first index of token lies in spans
    for i in range(len(first_letter_index)):
        for span in spans:
            if (span[0] <= first_letter_index[i] < span[1]):
                binary_labels[i] = 1
                
    labelled_tokens_list = []                #Creating common list of [token,label] entries
    for i in range(len(binary_labels)):
        labelled_tokens_list.append([token_list[i], binary_labels[i]])
        
    for i in range(len(labelled_tokens_list)-1):    
        if (labelled_tokens_list[i][0][0] == '\"') & (labelled_tokens_list[i+1][1] == 1):
            labelled_tokens_list[i][1] = 1   #Handling corner cases of quotation
    raw_new = ""
    for token in labelled_tokens_list:
        raw_new += token[0]+ " "
        
    assert (len(raw)==len(raw_new)), str("Cannot regenerate the raw text from these tokens " + str(len(raw)) + " != " + str(len(raw_new)))
    
    return labelled_tokens_list

def label_article_chars(article_id):
    '''
    returns a binary 0/1 char-level lablled data propaganada span for train-set (1: is_propangada)
    '''
    
    raw, spans = read_article(article_id)    #reading raw data and span locations
    label_list = []
    labels = [0] * len(raw)
    chars = list(raw)
    assert (len(labels)==len(chars)), "length mismatch in character and label sequence"
    
    for i in range(len(chars)):
        for span in spans:
            if span[0] <= i < span[1]:
                labels[i] = 1
                break
        label_list.append([chars[i], labels[i]])
        
    raw_new = ""
    for char in label_list:
        raw_new += char[0]
        
    assert (raw == raw_new), "unable to reporduce raw text from split characters"
        
    return label_list


#Word Level Data Processing
def getWordSpans(text):
    text=re.sub("\w[\'‘’“”]\w",'aaa',text)
    wordlist=[]
    def trans(text,pointer=0):
        if pointer==len(text)-1:
            return True
        else:
            while(text[pointer]==' ' and pointer<len(text)-1):
                pointer=pointer+1
            s=pointer
            while(not text[pointer]==' ' and pointer<len(text)-1):
                pointer=pointer+1
            wordlist.append([s,pointer])
            return trans(text,pointer)
    try:
        trans(text)
    except :
        return -1
    if(wordlist[-1][1]==wordlist[-1][0]):
        wordlist=wordlist[:-1]
    if(text[-1].isalpha()):
        wordlist[-1][1]+=1
    return wordlist

def getCharSpans(prediction,wordlist):
    charSpans=[]
    def getSpan(prediction,wordlist):
        for i in range(len(prediction)):
            if(i==0):
                if(prediction[i]==1):
                    charSpans.append(wordlist[0][0])
            elif(prediction[i]==0 and prediction[i-1]==1):
                charSpans.append(wordlist[i-1][1])
            elif(prediction[i]==1 and prediction[i-1]==0):
                charSpans.append(wordlist[i][0])
            if(i==len(prediction)-1 and prediction[i]==1):
                charSpans.append(wordlist[-1][1])
    getSpan(prediction,wordlist)
    return [[charSpans[i],charSpans[i+1]] for i in range(0,len(charSpans),2)]

def pred_span(text,prediction):
    wordlist=getWordSpans(text)
    return getCharSpans(prediction,wordlist)   

def getLabeledWords(article_id):
    text, spans = read_article(article_id)
    wordlist = getWordSpans(text)
    
    words = []
    labels = [0]*len(wordlist)
    for i in range(len(wordlist)):
        for span in spans:
            if (wordlist[i][0] in range(span[0], span[1]+1)) & (wordlist[i][1] in range(span[0], span[1]+1)):
                labels[i] = 1
                break
    for word in wordlist:
        words.append(text[word[0]:word[1]])
        
    assert (len(words)==len(labels))
    
    return words, labels

def get_BIO(article_id,pair=True,cleaner=None):
    text, p_spans = read_article(article_id)
    spans = getWordSpans(text)
    span_starts,span_stops = [span[0] for span in spans],[span[1] for span in spans]
    pre_BIO=[0]*len(spans)
    words = [text[span[0]:span[1]] for span in spans]
    for span in p_spans:
        x,y=span_starts.index(span[0]),span_stops.index(span[1])
        pre_BIO[x]+=1
        pre_BIO[y]-=1
        if x==y:
            pre_BIO[x]='X'
    BIO=[]
    pointer=0
    for val in pre_BIO:
        if type(val)==int:
            pointer+=val
        if val!=0:
            BIO.append('B')
        else:
            if pointer > 0:
                BIO.append('I')
            else:
                BIO.append('O')
    if cleaner!=None:
        BIO_zip=[(cleaner(word),bio) for word,bio in zip(words,BIO)]
        words=[cleaner(word) for word in words]
    else:
        BIO_zip=[(word,bio) for word,bio in zip(words,BIO)]
    if pair:
        return BIO_zip
    else:
        return words,BIO

def decode_BIO(BIO):
    bio=BIO
    BIO='O'+''.join(BIO)+'O'
    spans=[]
    pointer=0
    while 'OB' in BIO[pointer:]:
        try :
            spans.append((BIO.index('OB',pointer),BIO.index('BO',pointer)-1))
            pointer=BIO.index('BO',pointer)+1
        except:
            break
    BIO=bio
    output=[]
    for span in spans:
        start,end=span
        if start==end:
            output.append([start,end])
        else:
            for pointer in range(start,end):
                if BIO[pointer] == 'B':
                    output.append([pointer,end])
    return output

def Char_pred(BIO,article_id):
    text=read_article(article_id)[0]
    decoded=decode_BIO(BIO)
    wordlist=getWordSpans(text)
    charSpans=[[wordlist[span[0]][0],wordlist[span[1]][1]]for span in decoded]
    return charSpans

def word_list(text,span_list):
    return [text[span[0]:span[1]] for span in span_list]