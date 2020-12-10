import argparse
import json
import pickle
from http.server import BaseHTTPRequestHandler, HTTPServer

import nltk
import numpy as np
import pandas as pd
import requests
import torch
from nltk.corpus import cmudict
from sklearn.metrics import balanced_accuracy_score
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

nltk.download('cmudict')


MAX_PRONUNS = 10
TOP_K = 10


def get_data():
  c = set()
  word2pronun = cmudict.dict()
  for k,v in word2pronun.items():
    for p in v:
      c.update(p)
  pronun_set = sorted(list(set(map(lambda x: x[:-1] if x[-1].isdigit() else x, c))))

  return pronun_set, word2pronun


def encode_pronuns(pronun_set):
  pronun2vec = {}
  for i, pronun in enumerate(pronun_set):
    pronun2vec[pronun] = np.eye(len(pronun_set))[i]
  return pronun2vec


def get_feature_vector(word,word2pronun,pronun_set,pronun2vec):
  # Take the first pronunciation for the word
  pronunciation = word2pronun.get(word, [['UNK']])[0]
  feature_vector = []
  zero_vec = np.zeros(len(pronun_set))

  for i in range(MAX_PRONUNS):
    # Pad if no pronun
    if i >= len(pronunciation):
      feature_vector.extend(zero_vec)
    # Add the pronun vec
    else:
      p = pronunciation[len(pronunciation)-i-1]
      p = p[:-1] if p[-1].isdigit() else p
      feature_vector.extend(list(map(lambda x: x*(len(pronun_set)-i),pronun2vec[p])))
  return np.asarray(feature_vector)


pronun_set, word2pronun = get_data()
cmudictwords = list(word2pronun.keys())
pronun2vec = encode_pronuns(pronun_set)


nbrs = pickle.load(open('nnk.model', 'rb'))

def getTopKRhymes(word):
    distances, indices = nbrs.kneighbors(
        [get_feature_vector(word, word2pronun, pronun_set, pronun2vec)]
    )
    return list(
        filter(lambda x: x != word, map(
            lambda x: cmudictwords[x], indices[0])))

print(len(word2pronun), len(pronun_set), len(pronun2vec))
print('dog', getTopKRhymes('dog'))
print('hello', getTopKRhymes('hello'))

##
class RhymeDataset(Dataset):

    def __init__(self, X_a,X_b,label):

        self.X_a = X_a
        self.X_b=X_b
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        a = get_feature_vector(self.X_a[idx], word2pronun, pronun_set, pronun2vec)
        b = get_feature_vector(self.X_b[idx], word2pronun, pronun_set, pronun2vec)

        return a, b, self.label[idx]



class RhymingClassifier(nn.Module):
    def __init__(self):
        super(RhymingClassifier, self).__init__()

        self.layer1_1 = nn.Linear(390, 256)
        # self.layer1_2 = nn.Linear(390, 256)
        self.layer2 = nn.Linear(512, 128)
        self.layer3 = nn.Linear(128, 64)
        self.lout = nn.Linear(64, 1)

        self.dropout = nn.Dropout(p=.2)
        # self.batchnorm1 = nn.BatchNorm1d(128)
        # self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, a, b):
        v_a=nn.functional.relu(self.layer1_1(a))
        v_b=nn.functional.relu(self.layer1_1(b))
        x = torch.cat([v_a.reshape(-1,256),v_b.reshape(-1,256)],dim=1)

        x = nn.functional.relu(self.layer2(x))
        # x = self.batchnorm1(x)
        x = self.dropout(x)
        x = nn.functional.relu(self.layer3(x))
        # x = self.batchnorm2(x)
        x = self.dropout(x)
        x = torch.sigmoid(self.lout(x))
        return x


def demo(model, word1, word2):
    if word1 not in word2pronun or word2 not in word2pronun:
        return None
    k = [word1]
    kk = [word2]
    l = [0]
    demo_dataset = RhymeDataset(k, kk, l)
    demo_loader = DataLoader(demo_dataset,2,num_workers=2,pin_memory=False)

    with torch.no_grad():
        preds = []
        for batch in (demo_loader):
            # y_batch = batch[2].to(torch.device('cuda'))
            a=batch[0].float().to(torch.device('cpu'))
            b=batch[1].float().to(torch.device('cpu'))
            y_pred = model(b, a)#*X_batch)
            del a,b

            preds.extend(y_pred.cpu().detach().numpy())
        print(f"{word1} and {word2} {preds[0][0]} rhyme")
        return preds[0][0]



model = RhymingClassifier()
model.load_state_dict(torch.load('isRhyme.model.state', map_location="cpu"))
model.eval()


with torch.no_grad():
  a = demo(model, 'clan', 'fayanne')
  aa = demo(model, 'fayanne','clan')
  bb = demo(model, 'hello', 'fayanne')
  b = demo(model, 'hello', 'hello')






HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:76.0) '
                  'Gecko/20100101 Firefox/76.0',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.5',
    'Content-Type': 'application/x-www-form-urlencoded',
    'DNT': '1',
    'Connection': 'keep-alive',
}


class S(BaseHTTPRequestHandler):
    def _set_headers(self, status):
        self.send_response(status)
        self.send_header("Content-type", "application/json")

        self.send_header('Access-Control-Allow-Credentials', 'true')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With, Content-type")

        self.end_headers()

    def do_GET(self):
        data = self.path[1:]
        status, details = get_details(data)
        self._set_headers(status)
        self.wfile.write(details.encode())


    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.send_header('Access-Control-Allow-Credentials', 'true')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With, Content-type")


def run(server_class=HTTPServer, handler_class=S, addr="localhost", port=8000):
    server_address = (addr, port)
    httpd = server_class(server_address, handler_class)

    print(f"Starting httpd server on {addr}:{port}")
    httpd.serve_forever()

def algorithm(a, b=None):
    """Machine Learning go brrrrrrrrrr"""
    if not b:
        return getTopKRhymes(a)
    with torch.no_grad():
        return demo(model, a, b)


def get_details(data):
    """Returns the json extracted from username page."""

    # STUFF HERE
    print(data)
    data = list(filter(lambda x: x, data.strip().split('%20')))
    print(data)
    output, nearRhymes = None, None
    if len(data) > 1:
        output = (algorithm(*data))
        if output == 0.5:
            output = "These words may rhymes; I'm really not sure."
        elif output > 0.5:
            output = f"These words rhyme. I'm {output*100}% confident!"
        else:
            output = f"These words DON'T rhyme. I'm {100-output*100}% confident!"
    else:
        nearRhymes = algorithm(*data)
    return 200, json.dumps({
        "message": f"You asked me for {data}",
        "nearRhymes": nearRhymes,
        "output": output
    })



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run a simple HTTP server")
    parser.add_argument(
        "-l",
        "--listen",
        default="localhost",
        help="Specify the IP address on which the server listens",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8000,
        help="Specify the port on which the server listens",
    )
    args = parser.parse_args()
    run(addr=args.listen, port=args.port)
