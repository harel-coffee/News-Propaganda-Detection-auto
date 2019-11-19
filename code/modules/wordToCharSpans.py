def getWordSpans(text):
    wordlist=[]
    def trans(text,pointer=0):
        if pointer==len(text)-1:
            return True
        else:
            while(not text[pointer].isalpha() and pointer<len(text)-1):
                pointer=pointer+1
            s=pointer
            while(text[pointer].isalpha() and pointer<len(text)-1):
                pointer=pointer+1
            wordlist.append([s,pointer])
            return trans(text,pointer)
    try:
        trans(text)
    except :
        return -1
    if(wordlist[-1][1]==wordlist[-1][0]):
        wordlist=wordlist[-1]
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