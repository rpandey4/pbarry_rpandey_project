import argparse
import os
import pandas as pd
import gensim
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import datetime
from subprocess import check_output
import struct
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

np.set_printoptions(precision=30)
np.random.seed(1)  # random seeding for reproducability

WORD2VEC_PATH = "data/GoogleNews-vectors-negative300.bin"
GLOVE_PATH = {"42b": "",
            "840b": "/Users/rahulpandey/Downloads/mason/data/input/word_embeddings/glove.840B.300d.txt"}


def load_data(data_path):
    """
    Load the STSBenchmark data
    """
    train_path = os.path.join(data_path, "sts-train.csv")
    dev_path = os.path.join(data_path, "sts-dev.csv")
    test_path = os.path.join(data_path, "sts-test.csv")
    train = pd.read_csv(train_path, delimiter='\t', header=None, names=['genre', 'a', 'as', 'asd','score', 'category', 'text1', 'text2'])
    train['category'] = train['score'].apply(lambda x: round(float(x)))
    dev = pd.read_csv(dev_path, delimiter='\t', header=None, names=['genre', 'a', 'as', 'asd', 'score', 'category', 'text1', 'text2'])
    dev['category'] = dev['score'].apply(lambda x: round(float(x)))
    test = pd.DataFrame()
    # test = pd.read_csv(test_path, delimiter='\t', header=None, names=['genre', 'a', 'as', 'asd', 'score', 'category', 'text1', 'text2'])
    # test['category'] = test['score'].apply(lambda x: round(float(x)))
    return train, dev, test


def get_word_embedding(text, model, dims, is_binary=False):
    """
    Get average of the word embeddings as representations for the given text
    """
    feat = np.zeros(dims)
    cnt = 0
    for word in word_tokenize(text):
        if is_binary:
            if word in model.wv.vocab:
                feat += model.wv.word_vec(word)
                cnt += 1
        else:
            if word in model.keys():
                feat += model[word]
                cnt += 1
    feat /= min(cnt, 1)
    return feat


def get_glove_data(train_size):
    """
    Get pretrained Glove model
    """
    model = {}
    assert train_size in GLOVE_PATH.keys()
    dims = 300  # right now only working for 300 dimensions
    file_path = GLOVE_PATH[train_size]
    for lines in open(file_path):
        lines = lines.strip().split(" ")
        model[lines[0]] = [float(l) for l in lines[1:]]
    return model, dims


def get_baseline_results_embeddings(data_dict, file_type, word_embedding, distance):
    """

    """
    print("Loading Embeddings Model\n")
    if word_embedding == "word2vec":
        model, dims = (gensim.models.KeyedVectors.load_word2vec_format(WORD2VEC_PATH, binary=True), 300)
    else:
        model, dims = get_glove_data(train_size=word_embedding.split("_")[-1])
    print("Computing %s distance for similarity" % (distance))
    df = data_dict[file_type]
    original = df["score"].tolist()
    predicted = []
    is_binary = True if word_embedding == "word2vec" else False
    X = []
    Y = []
    for i in range(len(df)):
        text1 = df.loc[i, "text1"]
        text2 = df.loc[i, "text2"]
        embed_1 = get_word_embedding(text1, model, dims, is_binary)
        embed_2 = get_word_embedding(text2, model, dims, is_binary)
        predicted.append(np.dot(embed_1, embed_2) / (np.linalg.norm(embed_1) * np.linalg.norm(embed_2)))
    predicted = [x*5 for x in predicted]   # scale to 5

    now = datetime.datetime.now()
    out_file_name = "output/%s_%s_%s.txt" % (file_type, word_embedding, now.strftime("%Y_%m_%d_%H_%M"))
    with open(out_file_name, "w") as f:
        for p in predicted:
            f.write("%.3f\n" % (p))
    print("Saved the predicted scores in %s " % (out_file_name))
    return out_file_name


class LogReg(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.linear = nn.Linear(dims, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.linear(x))

class FFNN(nn.Module):
    def __init__(self, dims, hidden_nodes):
        super().__init__()
        self.linear = nn.Linear(dims, hidden_nodes)
        self.linear2 = nn.Linear(dims, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z1 = self.linear(x)
        a1 = self.sigmoid(z1)
        z2 = self.linear2(a1)
        a2 = self.sigmoid(z2)
        return a2

def evaluate_result(out_file_name, file_type):
    original_file = "data/sts-%s.csv" % (file_type)
    predicted_file = out_file_name
    out = check_output(["perl", "data/correlation.pl", original_file,
                        predicted_file]).decode("utf-8")
    pearson_score = float(out[:-1].split(" ")[1])
    return pearson_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='./data/')
    parser.add_argument("--eval_data_type", type=str, default='test')
    parser.add_argument("--word_embedding", type=str, default='word2vec')
    parser.add_argument("--distance", type=str, default='cosine')
    args = parser.parse_args()
    print("%s\nSemantic Textual Similarity\nPaul Barry and Rahul Pandey\n%s\n\nLoading Dataset" % ("*"*100, "*"*100))
    train, dev, test = load_data(data_path=args.path)
    data_dict = {"train": train, "dev": dev, "test": test}
    print("Train #%d | Dev #%d | Test #%d\n%s" % (len(train), len(dev), len(test), "*"*100))
    out_file_name = get_baseline_results_embeddings(data_dict,
                                                    file_type=args.eval_data_type,
                                                    word_embedding=args.word_embedding,
                                                    distance=args.distance)
    score = evaluate_result(out_file_name, file_type=args.eval_data_type)
    print("Evaluated %s data by comparing %s distance of its %s word embeddings.\nPearson score = %.4f\n%s"
            % (args.eval_data_type, args.distance, args.word_embedding, score, "*"*100))
    return


if __name__ == '__main__':
    main()
