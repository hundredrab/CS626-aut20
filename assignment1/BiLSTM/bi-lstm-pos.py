# -*- coding: utf-8 -*-
"""BiLSTM POS TAGGER

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FoFllAdFrinLTBh8_3gKhe9NB7v5dqTT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data import Field
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from sklearn.model_selection import KFold
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

import warnings
warnings.filterwarnings('ignore')
 
from tqdm import tqdm
import numpy as np 
import nltk
from nltk.corpus import brown

nltk.download('brown')
nltk.download('universal_tagset')
nltk.download('punkt')

# Glove dictionary 200 dimentions
# !wget https://www.cse.iitb.ac.in/~kartavya/glove_dict_tensor.pkl

import pickle
pickle_in = open("glove_dict_tensor.pkl","rb")
glove = pickle.load(pickle_in)
pickle_in = None

if torch.cuda.is_available():    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

tagged_sentences = brown.tagged_sents(tagset='universal')
sentences = brown.sents()
sentence_tags = brown.tagged_words(tagset='universal')

print(tagged_sentences[0])
print(sentences[0])

tag_set = list(set([tag for _,tag in sentence_tags]))
tag_set.extend(['PAD'])
tag_set.sort()
tag2id = {}
for i,t in enumerate(tag_set):
  tag2id[t] = i
sentence_tags = None
id2tag = {v: k for k, v in tag2id.items()}
tag2id

def word2glove(word):
  try:
    embed = glove[word.lower()]
  except:
    embed = torch.zeros(200)
  return embed

# oneLoop = True
MAX_LEN = 40

sentences = []
sentence_tags = []
epoch = 0
for data in tagged_sentences:
  # epoch+=1
  # if epoch%1000 == 0:
  #   print(epoch)
  sentence = []
  tags = []
  for i,(word,tag) in enumerate(data):
    if i == MAX_LEN-1:
      break
    sentence.append(word2glove(word))
    tags.append(tag2id[tag])

  # if oneLoop:
  #   oneLoop = False
  # else:
  #   break
  # print(len(sentence[0]))
  sentence = torch.cat(sentence).view(len(sentence),-1)
  # print(sentence.size())

  if(len(sentence)<MAX_LEN):
    # print(len(sentence))
    sentence = torch.cat((sentence,torch.zeros((MAX_LEN-len(sentence),200))))
    tags = tags+[tag2id['PAD']]*(MAX_LEN-len(tags))
    tags = torch.tensor(tags)
    # print(tags)

  # print(sentence[0]) 

  sentences.append(sentence)
  # sentences.append(sentence)
  sentence_tags.append(tags)
glove = None
sentences = torch.stack(sentences)
sentence_tags = torch.stack(sentence_tags)
# sentences = torch.cat(sentences).view(-1,MAX_LEN,200)

print(sentences.size())
print(sentence_tags.size())

# dataset = TensorDataset(sentences, sentence_tags)

# batchSize = 32

# tot = len(dataset)
# trainSize = int(round(0.7*tot))
# valSize = int(round(0.1*tot))
# testSize = tot - trainSize - valSize

# print(trainSize, valSize, testSize)
# trainSet, valSet, testSet = random_split(dataset, [trainSize, valSize, testSize])

# print("tvt lengths",len(trainSet), len(valSet), len(testSet))
# print("Batch size", batchSize)

# train_dataloader = DataLoader(
#     trainSet,  # The training samples.
#     sampler = RandomSampler(trainSet), # Select batches randomly #TODO RandomSampler
#     batch_size = batchSize, # Trains with this batch size.
# )
# val_dataloader = DataLoader(
#     valSet,
#     sampler = SequentialSampler(valSet), #Select batches sequentially
#     batch_size = batchSize
# )
# test_dataloader = DataLoader(
#     testSet,
#     sampler = SequentialSampler(testSet),
#     batch_size = batchSize
# )
# print(len(train_dataloader), len(val_dataloader), len(test_dataloader))

class NET(nn.Module):
  def __init__(self, input_dim, hidden_dim, layer_dim, batchSize, seqLength, numClasses):
    super(NET,self).__init__()
    self.batchSize = batchSize
    self.hidden_dim = hidden_dim
    self.layer_dim = layer_dim

    self.bilstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional = True)
    self.fc = nn.Linear(hidden_dim*2,numClasses)

  def forward(self, input_ids):
    # Initialize hidden state
    # h0 = torch.zeros((self.layer_dim*2, input_ids.size(0), self.hidden_dim)).requires_grad_()
    # # Initialize cell state
    # c0 = torch.zeros((self.layer_dim*2, input_ids.size(0), self.hidden_dim)).requires_grad_()
    # BiLSTM
    # print(input_ids.size())
    out , (hn,cn) = self.bilstm(input_ids)
    # print('^^^^^^^^^^^',out.size(0))
    out = self.fc(out)
    # out = F.softmax(out,dim=1)

    return out

def train(model,data,optimizer,loss_criterion, batch_size):
  model.train()
  
  epoch_loss = 0
  epoch_acc = 0
  
  for sentence_batch,tags_batch in data:
    if tags_batch.shape[0]!=batch_size:
      # print("label mismatch")
      continue

    if torch.cuda.is_available():
      sentence_batch = sentence_batch.cuda()
      tags_batch = tags_batch.cuda()
      # label = label.cuda()
    
    optimizer.zero_grad()

    out = model(sentence_batch)
    out = out.view(-1,out.shape[-1])
    tags_batch = tags_batch.view(-1)

    loss=loss_criterion(out, tags_batch)
    acc = categorical_accuracy(out, tags_batch, tag2id['PAD'])

    loss.backward()
    optimizer.step()

    epoch_loss += loss.item()
    epoch_acc += acc.item()
  
  return epoch_loss / len(data), epoch_acc / len(data)

def evaluate(model,data,loss_criterion,batch_size):
  epoch_loss = 0
  epoch_acc = 0

  model.eval()

  with torch.no_grad():
    for sentence_batch,tags_batch in data:
      if tags_batch.shape[0]!=batch_size:
        # print("label mismatch")
        continue 

      if torch.cuda.is_available():
        sentence_batch = sentence_batch.cuda()
        tags_batch = tags_batch.cuda()

      out = model(sentence_batch)
      out = out.view(-1,out.shape[-1])
      tags_batch = tags_batch.view(-1)

      loss=loss_criterion(out, tags_batch)
      acc = categorical_accuracy(out, tags_batch, tag2id['PAD'])

      # loss.backward()
      # optimizer.step()

      epoch_loss += loss.item()
      epoch_acc += acc.item()
  return epoch_loss / len(data), epoch_acc / len(data)

def categorical_accuracy(preds, y, tag_pad_idx):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
  
    max_preds = preds.argmax(dim = 1, keepdim = True).squeeze(1) # get the index of the max probability
    non_pad_elements = (y != tag_pad_idx).nonzero()
    return accuracy_score(max_preds[non_pad_elements].cpu(),y[non_pad_elements].cpu())

# input_dim = 200
# hidden_dim = 50
# layer_dim = 2

# batch_size = 32
# num_epochs = 10
# learning_rate = 1e-3

# # import random

# model=NET(input_dim, hidden_dim, layer_dim,32,40,13)
# model.double()

# trainSet, valSet, testSet = train_dataloader,val_dataloader,test_dataloader
# optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)
# loss_criterion = nn.CrossEntropyLoss(ignore_index=tag2id['PAD'])

# for epochs in tqdm(range(num_epochs)):
#   train_loss, train_acc = train(model, trainSet, optimizer, loss_criterion, batch_size)
#   valid_loss, valid_acc = evaluate(model, valSet, loss_criterion, batch_size)
  
#   print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
#   print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

# test_loss, test_acc = evaluate(model, testSet, loss_criterion, batch_size)
# print(f'\tTest Loss: {test_loss:.3f} | Train Acc: {test_acc*100:.2f}%')

sentences.size()

input_dim = 200
hidden_dim = 50
layer_dim = 2

batch_size = 32
num_epochs = 5
learning_rate = 1e-3

# import random

model=NET(input_dim, hidden_dim, layer_dim,32,40,13)
model.double()
if torch.cuda.is_available():
  model.cuda()

optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_criterion = nn.CrossEntropyLoss(ignore_index=tag2id['PAD'])

crosss_validator = KFold(n_splits=5, random_state=42, shuffle=False)

for i in tqdm(range(num_epochs)):
  epoch_accuracy=[]
  epoch_loss=[]

  for train_index, test_index in crosss_validator.split(sentences):
    X_train, X_test, y_train, y_test = sentences[train_index], sentences[test_index], sentence_tags[train_index], sentence_tags[test_index]
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    X_train, X_test, y_train, y_test = None,None,None,None
    train_index, test_index = None,None
    
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        # sampler = RandomSampler(train_dataset), # Select batches randomly #TODO RandomSampler
        batch_size = batch_size, # Trains with this batch size.
    )
    train_dataset = None

    test_dataloader = DataLoader(
        test_dataset,
        # sampler = SequentialSampler(test_dataset),
        batch_size = batch_size
    )
    test_dataset = None

    # if cuda.is_available():

    train_loss, train_acc = train(model, train_dataloader, optimizer, loss_criterion, batch_size)
    train_dataloader = None
    test_loss, test_acc = evaluate(model, test_dataloader, loss_criterion, batch_size)
    test_dataloader = None

    epoch_accuracy.append(test_acc)
    epoch_loss.append(test_loss)

  print ('Accuracy: %f  loss: %f' % (np.mean(epoch_accuracy)*100,np.mean(epoch_loss)) )

# torch.save(model, 'latest_model.pth')
model = torch.load('latest_model.pth')
model.cpu()
model.double()

# model.double()
# predicted_tags = []
# true_tags = []

"""Per POS accuracy"""

predicted_tags=[]
true_tags=[]

model.eval()

with torch.no_grad():
  for sentence_batch,tags_batch in DataLoader(TensorDataset(sentences,sentence_tags),32):
    if tags_batch.shape[0]!=batch_size:
      # print("label mismatch")
      continue 

    if torch.cuda.is_available():
      sentence_batch = sentence_batch.cuda()
      tags_batch = tags_batch.cuda()

    out = model(sentence_batch)

    out = out.view(-1,out.shape[-1])
    tags_batch = tags_batch.view(-1)
  
    max_preds = out.argmax(dim = 1, keepdim = True).squeeze(1) # get the index of the max probability
    non_pad_elements = (tags_batch != tag2id['PAD']).nonzero()
    predicted_tags.extend(max_preds[non_pad_elements].cpu())
    true_tags.extend(tags_batch[non_pad_elements].cpu())

# print(accuracy_score(predicted_tags,true_tags))
cr = classification_report(true_tags, predicted_tags)
print(cr)

"""Confusion matrix"""

print("Confusion Matrix")
POS_list = list(tag2id.keys())
POS_list.remove('PAD')
cm_df = pd.DataFrame(confusion_matrix(true_tags, predicted_tags,labels=list(map(lambda x: tag2id[x],POS_list))),index = POS_list, columns =POS_list)
# cm_df.drop(columns=['PAD'],inplace=True)
# cm_df.drop(["<$>","<^>"],inplace=True)
cm_df=cm_df.div(cm_df.sum(axis=1)*0.01, axis=0).fillna(0)
plt.figure(figsize=(10,8))
ax=sns.heatmap(cm_df, annot=True ,fmt=".2f", cmap="Reds")
ax.set_ylim(12, 0)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

"""Get editable classification report"""

# cr_tab = []
# for row in cr.split('\n'):
#   if row=='\n' or row=='':
#     continue
#   line = []
#   for cell in row.split(' '):
#     if cell == ' ' or cell == '':
#       continue
#     line.append(cell)
#   cr_tab.append(line)

# f1score = {}
# for row in cr_tab[1:-3]:
#   f1score[id2tag[int(row[0])]] = float(row[3])


# f1score

"""POS demo uitlity"""

import pickle
pickle_in = open("glove_dict_tensor.pkl","rb")
glove = pickle.load(pickle_in)
pickle_in = None

def predict(text,model):
  sentence = []
  tags = []
  for word in text:
    sentence.append(word2glove(word))
  sentence = torch.cat(sentence).view(len(sentence),-1)
  sentence = torch.stack([sentence])
  ds = TensorDataset(sentence)
  dl = DataLoader(ds,1)

  for i in dl:
    # print(i[0])
    out = model(i[0])

  # print(F.softmax(out[0]).argmax())

  return list(map(lambda x: id2tag[x],F.softmax(out[0]).argmax(dim=1).numpy()))

# text = "I am going north to join my husband in Santa Fe, and you might prefer the southern route"
# text = text.split(' ')
# model.cpu()
# model.double()
# pos = predict(text,model)
# tagged_text = [(w,t) for w,t in zip(text,pos)]
# tagged_text

# sentence_bad=[]
# predicted_tag_bad = []
# actual_tag_bad = []
# accuracy_bad = []
# model.cpu()
# model.double()
# tagged_sentences = brown.tagged_sents(tagset='universal')
# # len(sentences)
# # sentences[0]
# i = 0
# for i in tqdm(range(len(tagged_sentences))):
#   # if i == 5:
#   #   break
#   # i+=1
#   sentence,tags = zip(*tagged_sentences[i])
#   tags = list(tags)
#   try:
#     predict_tags = predict(sentence,model)
#     acc = accuracy_score(tags,predict_tags)
#     # print(accuracy_score(tags,predict_tags))
#   except:
#     continue
#   if acc<0.6 :
#     sentence_bad.append(sentence)
#     actual_tag_bad.append(tags)
#     predicted_tag_bad.append(predict_tags)
#     accuracy_bad.append(acc)
# # model.eval()

# # with torch.no_grad():
# #   for sentence_batch,tags_batch in DataLoader(TensorDataset(sentences,sentence_tags),32):
# #     if tags_batch.shape[0]!=batch_size:
# #       # print("label mismatch")
# #       continue 

# #     if torch.cuda.is_available():
# #       sentence_batch = sentence_batch.cuda()
# #       tags_batch = tags_batch.cuda()

# #     out = model(sentence_batch)

# #     out = out.view(-1,out.shape[-1])
# #     tags_batch = tags_batch.view(-1)
  
# #     max_preds = out.argmax(dim = 1, keepdim = True).squeeze(1) # get the index of the max probability
# #     non_pad_elements = (tags_batch != tag2id['PAD']).nonzero()
# #     predicted_tags.extend(max_preds[non_pad_elements].cpu())
# #     true_tags.extend(tags_batch[non_pad_elements].cpu())

# # # print(accuracy_score(predicted_tags,true_tags))
# # cr = classification_report(true_tags, predicted_tags)
# # print(cr)

# print(len(sentence_bad))

# for i in range(len(sentence_bad)):
#   if len(sentence_bad[i]) > 2:
#     print(sentence_bad[i])
#     print(actual_tag_bad[i])
#     print(predicted_tag_bad[i])
#     print("==========")