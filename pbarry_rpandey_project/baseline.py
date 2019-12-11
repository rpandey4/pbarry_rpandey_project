"""
Author: Paul Barry and Rahul Pandey
Description: AIT726 Term Project

Baseline approach to evaluate similarity based on the pretrained word embeddings.

NOTE: Before executing this script, please update the pretrained word embeddings file path in `config_baseline.yml`
This script requires Py YAML package. To install with pip:
> pip install pyyaml

Usage: python baseline.py --path <path_of_sts_data> --eval_data_type <test/train/dev> --word_embedding <word2vec/glove_42b/glove_840b>

e.g.
To evaluate test data:
> python baseline.py --path ./data/ --eval_data_type test --word_embedding word2vec
> python baseline.py --path ./data/ --eval_data_type test --word_embedding glove_42b
> python baseline.py --path ./data/ --eval_data_type test --word_embedding glove_840b

To evaluate train data:
> python baseline.py --path ./data/ --eval_data_type train --word_embedding word2vec
> python baseline.py --path ./data/ --eval_data_type train --word_embedding glove_42b
> python baseline.py --path ./data/ --eval_data_type train --word_embedding glove_840b

To evaluate dev data:
> python baseline.py --path ./data/ --eval_data_type dev --word_embedding word2vec
> python baseline.py --path ./data/ --eval_data_type dev --word_embedding glove_42b
> python baseline.py --path ./data/ --eval_data_type dev --word_embedding glove_840b

Best pearson correlation coefficient got was 62.35% with word2vec on test set.

Flow:
i. main
ii. Parse arguments
iii. Load dataset  (load_data)
    1. Open and preprocess train, validation, and test datasets.
iv. Get baseline results (get_baseline_results_embeddings)
    1. Load the pretrained word embeddings
    2. for each sentences, get the average of their word embeddings vectors
    3. Take cosine similarity and scale to 5
v. store and evaluate the results (evaluate_result)

"""
import argparse
import os
import pandas as pd
import gensim
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import datetime
from nltk.corpus import stopwords
from subprocess import check_output
import struct
import yaml
import string

STOPWORDS = set(stopwords.words("english"))
pretrained_model_path = yaml.safe_load(open("config_baseline.yml", "r"))
WORD2VEC_PATH = pretrained_model_path["word2vec"]
GLOVE_PATH = pretrained_model_path["glove"]


def pre_process_text(text):
    text_split = word_tokenize(text)
    text_split = [x for x in text_split if x not in STOPWORDS and x not in string.punctuation]  # removing stop words
    return " ".join(text_split)

np.set_printoptions(precision=30)
np.random.seed(1)  # random seeding for reproducability

def load_data(data_path):
    """
    Load the STSBenchmark data
    """
    train_path = os.path.join(data_path, "sts-train.csv")
    dev_path = os.path.join(data_path, "sts-dev.csv")
    test_path = os.path.join(data_path, "sts-test.csv")
    train = pd.DataFrame()
    dev = pd.DataFrame()
    test = pd.DataFrame()
    i = 0
    for lines in open(train_path):
        lines = lines.strip().split("\t")
        train.loc[i, "genre"] = lines[0]
        train.loc[i, "score"] = float(lines[4])
        train.loc[i, "category"] = round(float(lines[4]))
        train.loc[i, "text1"] = pre_process_text(lines[5])
        train.loc[i, "text2"] = pre_process_text(lines[6])
        i += 1
    i = 0
    for lines in open(dev_path):
        lines = lines.strip().split("\t")
        dev.loc[i, "genre"] = lines[0]
        dev.loc[i, "score"] = float(lines[4])
        dev.loc[i, "category"] = round(float(lines[4]))
        dev.loc[i, "text1"] = pre_process_text(lines[5])
        dev.loc[i, "text2"] = pre_process_text(lines[6])
        i += 1
    i = 0
    for lines in open(test_path):
        lines = lines.strip().split("\t")
        test.loc[i, "genre"] = lines[0]
        test.loc[i, "score"] = float(lines[4])
        test.loc[i, "category"] = round(float(lines[4]))
        test.loc[i, "text1"] = pre_process_text(lines[5])
        test.loc[i, "text2"] = pre_process_text(lines[6])
        i += 1
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


def get_baseline_results_embeddings(data_dict, file_type, word_embedding):
    """
    Load embedding and compute sentence vector and their cosine similarity scaled to 5
    """
    print("Loading Embeddings Model\n")
    if word_embedding == "word2vec":
        model, dims = (gensim.models.KeyedVectors.load_word2vec_format(WORD2VEC_PATH, binary=True), 300)
    else:
        model, dims = get_glove_data(train_size=word_embedding.split("_")[-1])
    print("Computing %s distance for similarity" % ("cosine"))
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


def evaluate_result(out_file_name, file_type):
    '''
    Evaluate the results through official perl script
    '''
    original_file = "data/sts-%s.csv" % (file_type)
    predicted_file = out_file_name
    out = check_output(["perl", "data/correlation.pl", original_file,
                        predicted_file]).decode("utf-8")
    pearson_score = float(out[:-1].split(" ")[1])
    return pearson_score


def main():
    '''
    Main function
    Takes 3 arguments
    1. path to the data
    2. eval data type should be test or train or dev
    3. word embedding can be word2vec, glove_42b, or glove_840b
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='./data/')
    parser.add_argument("--eval_data_type", type=str, default='test')
    parser.add_argument("--word_embedding", type=str, default='word2vec')
    args = parser.parse_args()
    print("%s\nSemantic Textual Similarity\nPaul Barry and Rahul Pandey\n%s\n\nLoading Dataset" % ("*"*100, "*"*100))
    train, dev, test = load_data(data_path=args.path)
    data_dict = {"train": train, "dev": dev, "test": test}
    print("Train #%d | Dev #%d | Test #%d\n%s" % (len(train), len(dev), len(test), "*"*100))
    out_file_name = get_baseline_results_embeddings(data_dict,
                                                    file_type=args.eval_data_type,
                                                    word_embedding=args.word_embedding)
    score = evaluate_result(out_file_name, file_type=args.eval_data_type)
    print("Evaluated %s data by comparing %s distance of its %s word embeddings.\nPearson score = %.4f\n%s"
            % (args.eval_data_type, "cosine", args.word_embedding, score, "*"*100))
    return


if __name__ == '__main__':
    main()
