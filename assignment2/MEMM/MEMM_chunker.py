#!/usr/bin/env python
# coding: utf-8

# In[28]:


from tqdm import tqdm
import pandas as pd
tqdm.pandas()
import nltk
from nltk.corpus import brown
from nltk.tokenize import word_tokenize, sent_tokenize 
import numpy as np
from sklearn.model_selection import KFold
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
import torch
from torch import LongTensor
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch .optim as optim
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision


# In[86]:


class MEMM(nn.Module):
    def __init__(self, feature_vector_size,num_classes=3):
        super(MEMM, self).__init__()
        self.classes=num_classes
        self.fc=nn.Sequential(
            nn.Linear(feature_vector_size,num_classes))
    def forward(self, x):
        out=self.fc(x)
#         print(out.shape)
        out=out.reshape(-1,self.classes)
        out = torch.softmax(out,dim=1)
        return out


# In[4]:


def OneHotEncoder(tag, total_tag):
    return np.eye(total_tag)[tag]


# In[5]:


def featureExtractor(word_id, sample, word2vec,tags):
    chunk=['B','I','O']
    feature_vector = []
    vocab = word2vec.wv.vocab.keys()
    #Add word2vec embedding of current and previous word and previous to previous
    no_of_prev_words = 2
    for idx in reversed(range(0,no_of_prev_words+1)):
        if sample[word_id-idx][0] not in vocab:
            feature_vector.append(np.zeros(word2vec.vector_size))
        else:
            feature_vector.append(word2vec[sample[word_id-idx][0]])

    #Add word2vec embedding of next word and next to next word 
    no_of_next_words = 2
    for idx in range(1,no_of_next_words+1):
        if sample[word_id+idx][0] not in vocab:
            feature_vector.append(np.zeros(word2vec.vector_size))
        else:
            feature_vector.append(word2vec[sample[word_id+idx][0]])
            
    #Add one-hot embedding of previous 2 tags and current tag
    no_of_prev_tags = 2
    for idx in reversed(range(0,no_of_prev_tags+1)):
        tag = sample[word_id-idx][1]
        feature_vector.append(OneHotEncoder(tags.index(tag),len(tags)))
    
    #Add one-hot embedding of next 2 tags
    no_of_prev_tags = 2
    for idx in reversed(range(1,no_of_prev_tags+1)):
        tag = sample[word_id+idx][1]
        feature_vector.append(OneHotEncoder(tags.index(tag),len(tags)))
        
    #Add one-hot embedding of previous 2 chunking label
    no_of_prev_tags = 2
    for idx in reversed(range(1,no_of_prev_tags+1)):
        chunk_label = sample[word_id-idx][2]
        feature_vector.append(OneHotEncoder(chunk.index(chunk_label),len(chunk)))

    #Set the bit if word has verb prefix
    if len(sample[word_id][0]) > 4:
        if sample[word_id][0][:5].lower() in ["trans"] or sample[word_id][0][:4].lower() in ["over","fore"] or sample[word_id][0][:3].lower() in ["mis","out","pre","sub"] or sample[word_id][0][:2].lower() in ["un","be","de"]:
            feature_vector.append([1])
        else: 
            feature_vector.append([0])
    else:
        feature_vector.append([0])
    
    #Set the bit if word has verb suffix
    if len(sample[word_id][0]) > 4:
        if sample[word_id][0][-3:].lower() in ["ise","ate"] or sample[word_id][0][-2:].lower() in ["fy","en"]:
            feature_vector.append([1])
        else: 
            feature_vector.append([0])
    else:
        feature_vector.append([0])

    #Set the bit if word has noun prefix
    if len(sample[word_id][0]) > 4:
        if sample[word_id][0][:5].lower() in ["hyper","super","ultra"] or sample[word_id][0][:4].lower() in ["anti","semi","auto","kilo","mega","mini","mono","poly"] or sample[word_id][0][:3].lower() in ["mis","out","sub"] or sample[word_id][0][:2].lower() in ["bi","in"]:
            feature_vector.append([1])
        else: 
            feature_vector.append([0])
    else:
        feature_vector.append([0])
    
    #Set the bit if word has noun suffix
    if len(sample[word_id][0]) > 4:
        if sample[word_id][0][-4:].lower() in ["tion","sion","ment","ance","ence","ship","ness"] or sample[word_id][0][-3:].lower() in ["ism","ity","ant","ent","age","ery"] or sample[word_id][0][-2:].lower() in ["ry","al","er","cy"]:
            feature_vector.append([1])
        else: 
            feature_vector.append([0])
    else:
        feature_vector.append([0])

    #Set the bit if word has adjective prefix
    if len(sample[word_id][0]) > 4:
        if sample[word_id][0][:3].lower() in ["non"] or sample[word_id][0][:2].lower() in ["im","in","ir",'il']:
            feature_vector.append([1])
        else: 
            feature_vector.append([0])
    else:
        feature_vector.append([0])
    
    #Set the bit if next word has adjective suffix
    if len(sample[word_id][0]) > 4:
        if sample[word_id][0][-4:].lower() in ["able","less"] or sample[word_id][0][-3:].lower() in ["ive","ous","ful"] or sample[word_id][0][-2:].lower() in ["al"]:
            feature_vector.append([1])
        else: 
            feature_vector.append([0])
    else:
        feature_vector.append([0])

    #Set the bit if next word has verb prefix
    if len(sample[word_id+1][0]) > 4:
        if sample[word_id+1][0][:5].lower() in ["trans"] or sample[word_id+1][0][:4].lower() in ["over","fore"] or sample[word_id+1][0][:3].lower() in ["mis","out","pre","sub"] or sample[word_id+1][0][:2].lower() in ["un","be","de"]:
            feature_vector.append([1])
        else: 
            feature_vector.append([0])
    else:
        feature_vector.append([0])
    
    #Set the bit if next word has verb suffix
    if len(sample[word_id+1][0]) > 4:
        if sample[word_id+1][0][-3:].lower() in ["ise","ate"] or sample[word_id+1][0][-2:].lower() in ["fy","en"]:
            feature_vector.append([1])
        else: 
            feature_vector.append([0])
    else:
        feature_vector.append([0])

    #Set the bit if next word has noun prefix
    if len(sample[word_id+1][0]) > 4:
        if sample[word_id+1][0][:5].lower() in ["hyper","super","ultra"] or sample[word_id+1][0][:4].lower() in ["anti","semi","auto","kilo","mega","mini","mono","poly"] or sample[word_id+1][0][:3].lower() in ["mis","out","sub"] or sample[word_id+1][0][:2].lower() in ["bi","in"]:
            feature_vector.append([1])
        else: 
            feature_vector.append([0])
    else:
        feature_vector.append([0])
    
    #Set the bit if next word has noun suffix
    if len(sample[word_id+1][0]) > 4:
        if sample[word_id+1][0][-4:].lower() in ["tion","sion","ment","ance","ence","ship","ness"] or sample[word_id+1][0][-3:].lower() in ["ism","ity","ant","ent","age","ery"] or sample[word_id+1][0][-2:].lower() in ["ry","al","er","cy"]:
            feature_vector.append([1])
        else: 
            feature_vector.append([0])
    else:
        feature_vector.append([0])

    #Set the bit if next word has adjective prefix
    if len(sample[word_id+1][0]) > 4:
        if sample[word_id+1][0][:3].lower() in ["non"] or sample[word_id+1][0][:2].lower() in ["im","in","ir",'il']:
            feature_vector.append([1])
        else: 
            feature_vector.append([0])
    else:
        feature_vector.append([0])
    
    #Set the bit if next word has adjective suffix
    if len(sample[word_id+1][0]) > 4:
        if sample[word_id+1][0][-4:].lower() in ["able","less"] or sample[word_id+1][0][-3:].lower() in ["ive","ous","ful"] or sample[word_id+1][0][-2:].lower() in ["al"]:
            feature_vector.append([1])
        else: 
            feature_vector.append([0])
    else:
        feature_vector.append([0])


    #Set the bit if it is starting of sample
    if word_id ==0:
          feature_vector.append([1])
    else:
        feature_vector.append([0])

    #Set the bit if it is ending of sample
    if word_id == len(sample)-1:
          feature_vector.append([1])
    else:
        feature_vector.append([0])
      
    #Set the bit if  all letter of word are capital
    if sample[word_id][0].upper() == sample[word_id][0]:
          feature_vector.append([1])
    else:
        feature_vector.append([0])

    #Set the bit if all letter of word are small
    if sample[word_id][0].lower() == sample[word_id][0]:
          feature_vector.append([1])
    else:
        feature_vector.append([0])
      
    #Set the bit if first letter of word is capital   
    if sample[word_id][0][0].isupper():
        feature_vector.append([1])
    else:
        feature_vector.append([0])

    #Set the bit if letters other than 1st letter are capital
    if sample[word_id][0][1:].lower() != sample[word_id][0][1:]:
          feature_vector.append([1])
    else:
        feature_vector.append([0])
      
    #Set the bit if word is numeric
    if sample[word_id][0].isdigit():
          feature_vector.append([1])
    else:
        feature_vector.append([0])
    
    #Set the bit if word contains a "-"
    if '-' in sample[word_id][0]:
          feature_vector.append([1])
    else:
        feature_vector.append([0])

    flat_list = [item for sublist in feature_vector for item in sublist]
    return flat_list


# In[7]:


def get_features(train):    
    chunks={'B':0,'I':1,'O':2}
    X_dataset=[]
    Y_dataset=[]
    for j in tqdm(range(len(train))):
        if train[j][1] not in ['<^>','<$>']:
            X_dataset.append(featureExtractor(j, train, w2v_mapping, tags_list))
            Y_dataset.append(chunks[train[j][2]])
    X_dataset=np.array(X_dataset).astype(np.float)
    Y_dataset=np.array(Y_dataset)
    return X_dataset,Y_dataset


# In[8]:


def preprocessing(file):
    file=file[:]
    for i in tqdm(range(len(file))):
        if len(file[i]) > 1:
            file[i][2]=file[i][2][0]
#             if file[i][2] == 'O':
#                 file[i][2]='B'
    return file
   


# ### Loading and Editing Data

# In[9]:


train=[]
test=[]
sentences=[]
with open("train.txt", "r") as f:
    sentence=[]
    train.append(["","<^>","B"])
    train.append(["","<^>","B"])
    for line in f:
        if len(line) > 3:
            train.append(line.split())
            sentence.append(line.split()[0])
        else:
            train.append(["","<$>","B"])
            sentences.append(sentence)
            sentence=[]
            train.append(["","<^>","B"])
            train.append(["","<^>","B"])
            
with open("test.txt", "r") as f:
    sentence=[]
    test.append(["","<^>","B"])
    for line in f:
        if len(line) > 3:
            test.append(line.split())
            sentence.append(line.split()[0])
        else:
            test.append(["","<$>","B"])
            sentences.append(sentence)
            sentence=[]
            test.append(["","<^>","B"])
            train.append(["","<^>","B"])
        


# In[10]:


train=preprocessing(train)
test=preprocessing(test)
df_train = pd.DataFrame(train, columns = ['word', 'POS','C'])
df_test = pd.DataFrame(test, columns = ['word', 'POS','C'])
tags_list=sorted(df_train.POS.unique())
tags={}
for i,tag in enumerate(tags_list):
    tags[tag]=i
w2v_mapping = Word2Vec(sentences, size=50)


# In[9]:


### Extracting Features


# In[11]:


X_train,Y_train = get_features(train)
X_test,Y_test = get_features(test)


# In[87]:


model=MEMM(feature_vector_size=506)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(params=model.parameters(),lr=0.001,weight_decay=0.001)


# In[89]:


total_loss=[]
epoch_accuracy=[]
epoch=20
batch_size=4096
for e in range(epoch):
    for i in (range(0,X_train.shape[0], batch_size)):
        output=model(torch.tensor(X_train[i:i+batch_size]).float())
        loss=criterion(output,torch.tensor(Y_train[i:i+batch_size]))
        optimizer.zero_grad()
        total_loss.append(loss)
        loss.backward()
        optimizer.step()
    with torch.no_grad() :
        output=model(torch.tensor(X_test).float())
        y_pred=np.argmax(output.numpy(), axis=1)
        print(accuracy_score(Y_test,y_pred))
#     print("############################################ ",e," #############################################")
    


# In[272]:


print(classification_report(Y_test,y_pred))


# In[283]:



def tag_pos( text,tokenize=False,POS=False):
    
        if tokenize :
            text = nltk.word_tokenize(str(text))

        if POS :   
            text=nltk.pos_tag(text)

        sentence=[]
        for i in range(len(text)):
            sentence.append(list(text[i])+['B'])
            
        chunk_list=['B','I','O']
        sentence = [["","<^>",'B'],["","<^>",'B']] + sentence + [["","<$>",'B'],["","<$>",'B']]
        chunks={0:'B',1:'I',2:'O'}
#         print(sentence)
        matrix = [{tag: (0, ) for tag in chunk_list} for _ in sentence]
        matrix[0]['B'] = (1, 'B')
        matrix[1]['B'] = (1,'B' )
        matrix[len(sentence)-1]['B'] = (1,'B' )
        matrix[len(sentence)-2]['B'] = (1,'O')

        for i, word in enumerate(sentence):
            if i in [0,1,len(sentence)-1,len(sentence)-2]:
                continue
            inp=[]
            for j,curr in enumerate(chunk_list):
                sentence[i-1][2]=curr
                X=featureExtractor(i, sentence, w2v_mapping, tags_list)
                inp.append(X)
            inp=torch.tensor(np.array(inp).astype(np.float)).float()
            out=model(inp)
            out=(out.detach().numpy().T*np.array([matrix[i-1]['B'][0],matrix[i-1]['I'][0],matrix[i-1]['O'][0]])).T

            for j,curr in enumerate(chunk_list):
                matrix[i][curr] = (out.max(axis=0)[j],chunks[out.argmax(axis=0)[j]])
        kk=len(sentence)-1
        test_pos = [(sentence[kk][0],[max(matrix[-1], key=lambda key: matrix[-1][key])])]
        while matrix:
            row = max(matrix.pop().values())
            kk=kk-1
            test_pos.insert(0, (sentence[kk][0],row[-1]))
        return(test_pos[3:-2])


# In[285]:


# tag_pos("Shubham is a good boy. ",True,True)
text=[('Frank', 'NNP'),
 ('Carlucci', 'NNP'),
 ('III', 'NNP'),
 ('was', 'VBD'),
 ('named', 'VBN'),
 ('to', 'TO'),
 ('this', 'DT'),
 ('telecommunications','NNS'),
 ('company', 'NN'),
 ("'s", 'POS'),
 ('board', 'NN'),
 (',', ','),
 ('filling','VBG'),
 ('the', 'DT'),
 ('vacancy', 'NN'),
 ('created', 'VBN'),
 ('by', 'IN'),
 ('the', 'DT'),
 ('death', 'NN'),
 ('of', 'IN'),
 ('William', 'NNP'),
 ('Sobey', 'NNP'),
 ('last', 'JJ'),
 ('May', 'NNP'),
 ('.', '.')]
tag_pos(text,False,False)

# tag_pos("Frank Carlucci III was named to this telecommunications company\'s board filling the vacancy created by the death of William Sobey last May. ",True,True)


# Original chunk labels


#[['Frank', 'NNP', 'B'],
 #['Carlucci', 'NNP', 'I'],
 #['III', 'NNP', 'I'],
 #['was', 'VBD', 'B'],
 #['named', 'VBN', 'I'],
 #['to', 'TO', 'B'],
 #['this', 'DT', 'B'],
 #['telecommunications', 'NNS', 'I'],
 #['company', 'NN', 'I'],
 #["'s", 'POS', 'B'],
 #['board', 'NN', 'I'],
 #[',', ',', 'O'],
 #['filling', 'VBG', 'B'],
 #['the', 'DT', 'B'],
 #['vacancy', 'NN', 'I'],
 #['created', 'VBN', 'B'],
 #['by', 'IN', 'B'],
 #['the', 'DT', 'B'],
 #['death', 'NN', 'I'],
 #['of', 'IN', 'B'],
 #['William', 'NNP', 'B'],
 #['Sobey', 'NNP', 'I'],
 #['last', 'JJ', 'B'],
 #['May', 'NNP', 'I'],
 #['.', '.', 'O']]


# In[ ]:





# In[ ]:





# In[ ]:




