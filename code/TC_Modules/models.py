from .utils_tc import *

class Dataset:
    def __init__(self, articles_folder=None, labels_file=None, df=pd.DataFrame()):
        assert not ((articles_folder==None or labels_file==None) and df.empty), 	"Inputs Invalid"

        if df.empty:	        	
	        self.articles_folder = articles_folder
	        self.labels_file = labels_file
	        self.articles = read_articles_from_file_list(articles_folder)
	        self.read()
	        self.df=pd.DataFrame()
	        self.df['Sentences']=self.sentences
	        self.df['Labels']=self.gold_labels
        else:
          self.df=df
    
    def read(self):
    	articles_id, span_starts, span_ends, self.gold_labels = read_predictions_from_file(self.labels_file)
    	self.spans = [int(end)-int(start) for start, end in zip(span_starts, span_ends)]
    	print("Read %d annotations from %d articles" % (len(span_starts), len(set(articles_id))))
    	self.sentences=[self.articles[id][int(start):int(end)] for id, start, end in zip(articles_id, span_starts, span_ends)]
    	self.size=len(self.sentences)

    def split(self, test_size=0.1, seed=1234):
      a,b= train_test_split(self.df, stratify=self.df['Labels'], test_size=test_size, random_state=seed)
      a,b = Dataset(df=a), Dataset(df=b)
      a.size, b.size=a.df.shape[0], b.df.shape[0]
      a.gold_labels, b.gold_labels=a.df['Labels'], b.df['Labels'] 
      return a,b

class SLDataset:
    def __init__(self,  df, lower=True):
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
        self.sentences=[text_clean(sentence) for sentence in self.df.Sentences]

class TransformerDataset:
    def __init__(self, df):
        self.df=df
        self.clean()
        self.sentences = df.df['Sentences'].apply(lambda x : "[CLS] {} [SEP]".format(x))
        self.le=LE()
        self.labels=self.le.fit_transform(self.df.gold_labels)

    def clean(self):
        def text_clean(text):
            text=re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
            text=re.sub('[“"”]',' " ',text)
            retain='[^abcdefghijklmnopqrstuvwxyzQWERTYUIOPASDFGHJKLZXCVBNM!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~.0123456789 ]'
            return ' '.join(text.split())
        
        print("Cleaning Sentences")
        self.sentences=[text_clean(sentence) for sentence in self.df.df.Sentences]
        
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

      for i in range(len(input_ids)):
      	if not input_ids[i][-1] == 0:
      		input_ids[i][-1]=102

      attention_masks = []
      # Create a mask of 1s for each token followed by 0s for padding
      for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)
      self.inputs, self.masks, self.labels = torch.tensor(input_ids), torch.tensor(attention_masks), torch.tensor(self.labels)

    def load(self, batch_size=32):
      self.data = TensorDataset(self.inputs, self.masks, self.labels)
      self.dataloader = DataLoader(self.data, shuffle=False, batch_size=batch_size)