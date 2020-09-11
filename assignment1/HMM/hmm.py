## ------------------------------------
import numpy as np
import nltk
nltk.download('brown')
nltk.download('universal_tagset')
from nltk.corpus import brown
from collections import defaultdict

## ------------------------------------

tags = set(w[1] for w in brown.tagged_words(tagset='universal')).union({'^'})

## ------------------------------------ Load dataset and create transition and emission counts' dicts

dataset = brown.tagged_sents(tagset='universal')
print(len(dataset))
emissions = defaultdict(dict)
transitions = defaultdict(dict)
for sent in dataset:
    sent.insert(0, ('^', '^'))
    for word, tag in sent:
        word = word.lower()
        if word in emissions[tag]:
            emissions[tag][word] += 1
        else:
            emissions[tag][word] = 1
    for i, (word, tag) in enumerate(sent):
        if i == 0:
            continue # skip '^'
        if tag in transitions[sent[i-1][1]]:
            transitions[sent[i-1][1]][tag] += 1
        else:
            transitions[sent[i-1][1]][tag] = 1

## ------------------------------------

emissions['DET']
transitions['DET']
for tag in transitions:
    print(transitions[tag])

## ------------------------------------ Normalize to probabilities

for tag, data in transitions.items():
    total = sum(data.values())
    transitions[tag] = {
        k: v/total for k,v in data.items()}

for tag, data in emissions.items():
    total = sum(data.values())
    emissions[tag] = {
        k: v/total for k, v in data.items()}

## ------------------------------------ Make sure there are transitions between all states
for tag1 in tags:
    for tag2 in tags:
        if tag2 not in transitions[tag1]:
            print(tag1, '->', tag2, 'is 0')
            transitions[tag1][tag2] = 0

## ------------------------------------ Testing 1 (ONE) sentence

test_sent = nltk.word_tokenize("This girl is called Hilarious!")
test_sent.insert(0, '^')
test_sent
## ------------------------------------ Viterbi matrix (actually a list of dicts)

matrix = [{tag: (0, ) for tag in tags} for _ in test_sent]
matrix[0]['^'] = (1, )
for i, word in enumerate(test_sent):
    word = word.lower()
    if i == 0:
        continue
    for curr in tags:
        if word not in emissions[curr]:
            emissions[curr][word] = 0
        all_p = [(emissions[curr][word] * transitions[prev][curr] * matrix[i-1][prev][0], prev) for prev in tags]
        matrix[i][curr] = max(all_p)

## ------------------------------------ Finally, print it out
import operator

test_pos = [max(matrix[i].items(), key = operator.itemgetter(1))[0]]
while matrix:
    row = max(matrix.pop().values())
    test_pos.insert(0, row[-1])
print(test_pos[1:])
print(test_sent)
