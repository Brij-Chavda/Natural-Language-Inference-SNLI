#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer()


# In[2]:


data_set = np.array(pd.read_csv("snli_1.0/snli_1.0_train.txt",error_bad_lines=False, delimiter = '\t'))


# In[4]:


col0 = [i[0] for i in data_set]
col5 = [i[5] for i in data_set]
col6 = [i[6] for i in data_set]


# In[1]:


import json
data = [json.loads(line) for line in open('snli_1.0/snli_1.0_train.jsonl', 'r')]


# In[ ]:


labels = []
sentence1 = []
sentence2 = []
count = 0
for i in data:   
    if i['gold_label'] != '-':
        sentence1.append(i['sentence1'])
        sentence2.append(i['sentence2'])
        if i['gold_label'] == 'neutral':
            labels.append(0)
        elif i['gold_label'] == 'contradiction':
            labels.append(1)
        elif i['gold_label'] == 'entailment':
            labels.append(2)


# In[ ]:


from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
lemmatizer = WordNetLemmatizer() 
porter = PorterStemmer()
def lemma(sentence): 
    lemma_sen = []
    word_list = nltk.word_tokenize(sentence)
    lemma_sen = ' '.join([lemmatizer.lemmatize(w.lower(),pos='v') for w in word_list])
    return lemma_sen
def stemSentence(sentence):
    stem_sentence=[]
    token_words=word_tokenize(str(sentence))
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return stem_sentence
def rmpunctuation(sentence):
    senapp = []
    for i in sentence:
        tokenizer = RegexpTokenizer(r'\w+')
        rmsentence = tokenizer.tokenize(str(i))
        senapp.append(" ".join(rmsentence))
    return senapp
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
def rmstwords(sentence):   
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(sentence.lower())
    fil_sen = [w for w in word_tokens if not w in stop_words]
    return fil_sen


# In[ ]:


s1_rmst = [rmstwords(i) for i in sentence1]


# In[ ]:


s1_stem = [lemma(str(i)) for i in s1_rmst]


# In[ ]:


s1_final = rmpunctuation(s1_stem) 


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
vector = TfidfVectorizer()
x1 = vector.fit_transform(s1_final)


# In[ ]:


s2_rmst = [rmstwords(str(i)) for i in sentence2]
s2_stem = [lemma(str(i)) for i in s2_rmst]
s2_final = rmpunctuation(s2_stem) 


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
vector1 = TfidfVectorizer()
x2 = vector1.fit_transform(s2_final)


# In[12]:


test_dataset = np.array(pd.read_csv("snli_1.0/snli_1.0_test.txt",error_bad_lines=False, delimiter = '\t'))


# In[13]:


import json
testdata = [json.loads(line) for line in open('snli_1.0/snli_1.0_test.jsonl', 'r')]


# In[14]:


test_labels = []
test_sentence1 = []
test_sentence2 = []
count = 0
for i in testdata:
    if i['gold_label'] != '-':
        test_sentence1.append(i['sentence1'])
        test_sentence2.append(i['sentence2'])
    if i['gold_label'] == 'neutral':
        test_labels.append(0)
    elif i['gold_label'] == 'contradiction':
        test_labels.append(1)
    elif i['gold_label'] == 'entailment':
        test_labels.append(2)
test_sentence1 = np.array(test_sentence1)
test_sentence2 = np.array(test_sentence2)
test_labels = np.array(test_labels)
#print(labels)


# In[ ]:


test_col0 = [i[0] for i in test_dataset]
test_col5 = [i[5] for i in test_sentence1]
test_col6 = [i[6] for i in test_dataset]


# In[16]:


ts1_rmst = [rmstwords(str(i)) for i in test_sentence1]
ts1_stem = [lemma(str(i)) for i in ts1_rmst]
ts1_final = rmpunctuation(ts1_stem)
ts2_rmst = [rmstwords(str(i)) for i in test_sentence2]
ts2_stem = [lemma(str(i)) for i in ts2_rmst]
ts2_final = rmpunctuation(ts2_stem) 


# In[17]:


test_x1 = vector.transform(ts1_final)
test_x2 = vector1.transform(ts2_final)


# In[ ]:


labels = np.array(labels)
import numpy as np
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
x = hstack((x1,x2))
clf = LogisticRegression(max_iter = 5000)
clf.fit(x,labels)


# In[40]:


from sklearn import metrics
x_test = hstack((test_x1,test_x2))
predict_data = clf.predict(x_test)
#print(predict_data)
#print("model accuracy {}".format(metrics.accuracy_score(test_labels,predict_data)))


# In[46]:


filename = 'logistic_model.sav'
pickle.dump(clf,open(filename,'wb'))

