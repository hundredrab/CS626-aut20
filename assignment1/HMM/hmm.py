## -------------------------------------
from collections import defaultdict

import nltk
import numpy as np
import seaborn as sns
import  matplotlib.pyplot as plt
from nltk.corpus import brown
from sklearn.model_selection import KFold

nltk.download('brown')
nltk.download('universal_tagset')

## -------------------------------------

tags = set(w[1] for w in brown.tagged_words(tagset='universal'))

##
known_words = set(brown.words())
## ------

class HMM:
    def __init__(self):
        self.emissions = defaultdict(dict)
        self.transitions = defaultdict(dict)
        self.tags = set(tags).union({'^'})

    def train(self, dataset):
        for sent in dataset:
            sent = sent[:]
            sent.insert(0, ('^', '^'))
            for word, tag in sent:
                word = word.lower()
                if word in self.emissions[tag]:
                    self.emissions[tag][word] += 1
                else:
                    self.emissions[tag][word] = 1
            for i, (word, tag) in enumerate(sent):
                if i == 0:
                    continue  # skip '^'
                if tag in self.transitions[sent[i-1][1]]:
                    self.transitions[sent[i-1][1]][tag] += 1
                else:
                    self.transitions[sent[i-1][1]][tag] = 1

        for tag in self.emissions:
            if tag == '^':
                continue
            self.emissions[tag]['<UNK>'] = 1
            for word in known_words:
                word = word.lower()
                if word not in self.emissions[tag].keys():
                    self.emissions[tag][word] = 1
                else:
                    self.emissions[tag][word] += 0

        ## Normalize to probabilities

        for tag, data in self.transitions.items():
            total = sum(data.values())
            self.transitions[tag] = {
                k: v/total for k, v in data.items()}

        for tag, data in self.emissions.items():
            total = sum(data.values())
            self.emissions[tag] = {
                k: v/total for k, v in data.items()}

        ## Make sure there are self.transitions between all states
        for tag1 in self.tags:
            for tag2 in self.tags:
                self.transitions[tag1]['^'] = 0  # Can't go back to ^
                if tag2 not in self.transitions[tag1]:
                    self.transitions[tag1][tag2] = 0.00001

    def tag_pos(self, sentence):
        """On given a tokenized sentence (list) returns a list of POS tokens."""
        sentence = [x if x in known_words else '<UNK>' for x in sentence]
        sentence.insert(0, '^')
        matrix = [{tag: (0, ) for tag in self.tags} for _ in sentence]
        matrix[0]['^'] = (1, )
        for i, word in enumerate(sentence):
            word = word.lower()
            if i == 0:
                continue
            for curr in self.tags:
                if word not in self.emissions[curr]:
                    self.emissions[curr][word] = 0
                all_p = [(self.emissions[curr][word] * self.transitions[prev][curr]
                          * matrix[i-1][prev][0], prev) for prev in self.tags]
                matrix[i][curr] = max(all_p)

        test_pos = [max(matrix[-1], key=lambda key: matrix[-1][key])]
        while matrix:
            row = max(matrix.pop().values())
            test_pos.insert(0, row[-1])
        return(test_pos[2:])

##
# sentence = nltk.word_tokenize("He said that he will go soon")
# tag_pos(sentence)
# hmm = HMM()
# hmm.train(dataset)
# hmm.tag_pos(sentence)

## ---------------------Load dataset and create transition and emission counts' dicts

full_dataset = np.asarray(brown.tagged_sents(tagset='universal'))
print(f"Total sents = {len(full_dataset)}")

##

tags = sorted(list(tags))

if '^' in tags:
    tags.remove('^')
tag_to_i = dict([(j,i) for (i, j) in enumerate(tags)])

confusion = np.zeros((len(tags), len(tags)))
predictions = []
truths = []
def calculate_accuracy(row):
    # print(row.shape)
    sent, ground = zip(*row[0])
    pred = model.tag_pos(sent)
    predictions.extend(pred)
    truths.extend(ground)
    for i,j in zip(ground, pred):
        if i == '^' or j == '^':
            continue
        confusion[tag_to_i[i]][tag_to_i[j]] += 1
    ret = sum(x==y for x,y in zip(ground, pred))
    # if ret < len(sent)/2:
        # print("\t".join(sent))
        # print("\t".join(ground))
        # print("\t".join(pred))
    return ret


kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in kf.split(full_dataset):
    train_set = full_dataset[train_index]
    test_set = full_dataset[test_index]
    model = HMM()
    model.train(train_set)
    # print("trained", end="")
    ts = test_set.reshape(-1, 1)
    tr = train_set.reshape(-1, 1)

    # correct_train = sum(np.apply_along_axis(calculate_accuracy, 1, tr))
    # total_train = sum(np.apply_along_axis(lambda x: len(x[0]), 1, tr))
    # print(f"Train acc: {100* correct_train/total_train}", end="\t")

    correct = sum(np.apply_along_axis(calculate_accuracy, 1, ts))
    total = sum(np.apply_along_axis(lambda x: len(x[0]), 1, ts))
    print(f"Test accuracy: {100 * correct/total}")
##
conf = (confusion.T/(confusion.sum(axis=1))).T
conf = np.nan_to_num(conf)

plt.figure(figsize=(10, 8))
ax = sns.heatmap(conf, annot=True, fmt=".1f", cmap="Reds",
                 xticklabels=tags, yticklabels=tags)
ax.set_ylim(12, 0)
plt.show()

##
