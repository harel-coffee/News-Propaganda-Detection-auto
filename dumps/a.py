import re
import string
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

train=open("Text_stitch.txt","r").read()
dev=open("Text_stitch_dev.txt","r").read()

final=train+' '+dev
with open("cleaned","w+") as f:
	f.write(clean(final))
	