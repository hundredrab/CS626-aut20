# -*- coding: utf-8 -*-

# !wget www.cse.iitb.ac.in/~kartavya/assignment2dataset.zip
# !unzip assignment2dataset.zip
# !pip install sklearn-crfsuite

# %matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from itertools import chain

from tqdm import tqdm

import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

def data_cleaning(content):
  sentences = []
  sentence = []
  for line in content:
    line = line.split(' ')
    if len(line)==1:
      sentences.append([(tok,pos,tag[0]) for tok,pos,tag in sentence])
      sentence = []
    else:
      sentence.append(line)
  return sentences

with open("train.txt", "r") as f:
  content = f.read().split('\n')
  train = data_cleaning(content)
with open("test.txt", "r") as f:
  content = f.read().split('\n')
  test = data_cleaning(content)

# print(len(train),len(test))

# train[0]

def generate_features(sent, i):
  word = sent[i][0]
  postag = sent[i][1]
  
  features = {
    'bias': 1.0,
    'word.lower()': word.lower(),
    'word.isupper()': word.isupper(),
    'word.istitle()': word.istitle(),
    'word.isdigit()': word.isdigit(),
    'postag': postag,  
  }

  if i == 0:
    features['BOS'] = True

  if i >= 1:
    word1 = sent[i-1][0]
    postag1 = sent[i-1][1]

    features.update({
        '-1:word.lower()': word1.lower(),
        '-1:postag': postag1,
    })

  if i >= 2:
    word2 = sent[i-2][0]
    postag2 = sent[i-2][1]

    features.update({
        '-2:word.lower()': word2.lower(),
        '-2:postag': postag2,
    })

  if i < len(sent)-2:
    word2 = sent[i+2][0]
    postag2 = sent[i+2][1]
    features.update({
      '+2:word.lower()': word2.lower(),
      '+2:postag': postag2,
    })

  if i < len(sent)-1:
    word1 = sent[i+1][0]
    postag1 = sent[i+1][1]
    features.update({
      '+1:word.lower()': word1.lower(),
      '+1:postag': postag1,
    })

  if i == len(sent) -1:
    features['EOS'] = True
              
  if len(word) > 4:
    if word[-4:].lower() in ["tion","sion","ment","ance","ence","ship","ness"] or word[-3:].lower() in ["ism","ity","ant","ent","age","ery"] or word[-2:].lower() in ["ry","al","er","cy"]:
      features['NOUN_SUFFIX'] = True

  if len(word) > 4:
      if word[:3].lower() in ["non"] or word[:2].lower() in ["im","in","ir",'il']:
        features['ADJ_PREFIX'] = True

  return features


def sent2features(sent):
  return [generate_features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
  return [label for token, postag, label in sent]

def sent2tokens(sent):
  return [token for token, postag, label in sent]

# sent2features(train[0])[3]

X_train = [sent2features(s) for s in train]
y_train = [sent2labels(s) for s in train]

X_test = [sent2features(s) for s in test]
y_test = [sent2labels(s) for s in test]

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs', 
    c1=0.1, 
    c2=0.1, 
    max_iterations=150, 
    all_possible_transitions=True
)
crf.fit(X_train, y_train)

y_pred = crf.predict(X_test)
# metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)

labels = list(crf.classes_)

sorted_labels = sorted(
    labels, 
    key=lambda name: (name[1:], name[0])
)
print(metrics.flat_classification_report(
    y_test, y_pred, labels=sorted_labels, digits=3
))

# y_test_flat = []
# for sent in y_test:
#   for i in sent:
#     y_test_flat.append(i)
# y_pred_flat = []
# for sent in y_pred:
#   for i in sent:
#     y_pred_flat.append(i)

# import pandas as pd
# import seaborn as sns
# print("Confusion Matrix")
# POS_list = ['B','I','O']
# cm_df = pd.DataFrame(confusion_matrix(y_pred_flat, y_test_flat,labels=POS_list),index = POS_list, columns =POS_list)
# cm_df=cm_df.div(cm_df.sum(axis=1)*0.01, axis=0).fillna(0)
# ax=sns.heatmap(cm_df, annot=True ,fmt=".2f", cmap="Reds")
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.show()

# import random

# def demo_sent():
#   r_sent = test[random.randint(0,len(test)-1)]
#   print(r_sent)
#   for i in range(len(r_sent)):
#     tok,pos,chunk = r_sent[i]
#     r_sent[i] = (tok,pos)
#   return r_sent

# text = [sent2features(demo_sent())]
# print(crf.predict(text))