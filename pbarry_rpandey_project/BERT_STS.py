"""
Author: Paul Barry and Rahul Pandey
Description: AIT726 Term Project
Usage: python BERT_STS.py
Arguments:
    --train_set for training data file
    --valid_set for validation data file
    --test_set for test data file
    --learning_rate to set the learning rate of the file
    --output_file to set the path and filename for the model's predictions
    --early_stopping to set the early stopping threshold
    --batch_size default 32 use low value for small gpus
Best pearson correlation coefficient uses default parameters to achieve 84.86.

Flow:
i. main
ii. Parse arguments
iii. Generate dataset
    1. Open and preprocess train, validation, and test datasets.
    2. Stored in StsDataset objects
v. Train the model
    1. Load pretrained BERT weights and initialize FFNN weights
    2. Create Loss and Optimization objects
    3. for each epoch
        a. For each batch
            I. calculate y_hat from forward pass
            II. Calculate loss from y_hat to y
            III. Update the weigths with gradient times learning rate
        b. Do forward pass with validation set.
        c. Calculate loss on validation set.
        d. If new minimum loss on validation set, then save weights to RAM.
        e. If early stopping threshold hit without improvement on validation set, then end training.
vi. Test the model
    1. Forward pass through test dataset.
    2. Write prediction to output file.

"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils import data
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-large-cased')

class StsDataset(data.Dataset):
    def __init__(self, fpath):
        """
        When given a file path, this loads the dataset for use in training.
        """
        entries = open(fpath, 'r')  # open file
        sents1, sents2, tags = [], [], []
        for line in entries:
            if line != '':
                line = line.split('\t')
                sents1.append(line[5])
                sents2.append(line[6])
                tags.append(float(line[4]))
        self.sents1, self.sents2, self.tags = sents1, sents2, tags

    def __len__(self):
        return len(self.sents1)

    def __getitem__(self, idx):
        """
        During batching this processes a portion of the dataset and returns it for use in training.
        """
        words1, words2, tags = self.sents1[idx], self.sents2[idx], self.tags[idx]
        x1 = tokenizer.tokenize(words1)
        x2 = tokenizer.tokenize(words2)
        x1_x2 = ['[CLS]'] + x1 + ['[SEP]'] + x2  # create the string to feed into BERT

        x1 = tokenizer.convert_tokens_to_ids(x1)
        x2 = tokenizer.convert_tokens_to_ids(x2)
        x1_x2 = tokenizer.convert_tokens_to_ids(x1_x2)
        y = tags

        seqlen1 = len(x1)
        seqlen2 = len(x2)
        seqlen3 = len(x1_x2)
        return words1, words2, x1, x2, y, seqlen1, seqlen2, seqlen3, x1_x2

def pad(batch):
    '''Zero pads each batch to the longest sample'''
    f = lambda x: [sample[x] for sample in batch]
    words1 = f(0)
    words2 = f(1)
    y = f(4)
    seqlens1 = f(5)
    seqlens2 = f(6)
    seqlens3 = f(7)
    maxlen1 = np.array(seqlens1).max()
    maxlen2 = np.array(seqlens2).max()
    maxlen3 = np.array(seqlens3).max()

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]  # pads each sequence with 0 values
    x1 = f(2, maxlen1)
    x2 = f(3, maxlen2)

    f = lambda x, seqlen: [sample[x] + ([0] * (seqlen - len(sample[x]))) for sample in batch]
    x1_x2 = f(8, maxlen3)

    f = torch.LongTensor
    return words1, words2, f(x1), f(x2), torch.FloatTensor(y), seqlens1, seqlens2, seqlens3, f(x1_x2)


class StsClassifier(nn.Module):
    def __init__(self):
        super(StsClassifier, self).__init__()
        self.xlnet = BertModel.from_pretrained('bert-large-cased')
        self.linear1 = nn.Linear(1024, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 1)
        self.activation = nn.ReLU()

    def forward(self, x1_x2):
        x1_x2 = self.xlnet(x1_x2.cuda())[0]
        z1 = self.activation(self.linear1(x1_x2[:, 0, :]))  # Feed first token in sequence from BERT ('[CLS]') to FFNN
        z2 = self.activation(self.linear2(z1))
        z3 = self.linear3(z2)
        return z3

    def predict(self, x1_x2):
        '''
        If doing predictions, call this so that argmax returns the integer of the max index
        '''
        return self.forward(x1_x2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default='./data/sts-train.csv')
    parser.add_argument("--valid_path", type=str, default='data/sts-dev.csv')
    parser.add_argument("--test_path", type=str, default='data/sts-test.csv')
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--output_file", type=str, default="./output/test_bert.txt")
    parser.add_argument("--early_stopping", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    #  Load the training, validation, and test datasets
    trainset = StsDataset(args.train_path)
    train_iter = data.DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, collate_fn=pad)
    validset = StsDataset(args.valid_path)
    valid_iter = data.DataLoader(dataset=validset, batch_size=args.batch_size, shuffle=False, collate_fn=pad)
    testset = StsDataset(args.test_path)
    test_iter = data.DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=False, collate_fn=pad)

    model = StsClassifier()  # Initialize the STS model
    model.cuda()  # send model to the GPU
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)  # Optimizing with Adam
    loss_func = nn.MSELoss()

    history = []  # list for storing historical F1 scores. Used for early stopping.
    best_weights = None
    run = True
    while run:
        model.train()
        for i, batch in enumerate(train_iter):  # batches training set
            words1, words2, x1, x2, y, seqlen1, seqlen2, seqlen3, x1_x2 = batch
            optimizer.zero_grad()  # zero all gradients
            yhat = model(x1_x2)  # Forward pass through Neural Net

            yhat = yhat.reshape(-1)  # reshapes to (num_sequences * sequence_length, num_classes)
            y = y.reshape(-1)  # reshapes to (num_sequences * sequence_length)

            loss = loss_func(yhat, y.cuda())  # Computes loss
            loss.backward()  # Back propagates loss
            optimizer.step()  # Updates weights

            if i % 10 == 0:  # monitoring loss
                print(f"step: {i}, loss: {loss.item()}")

        model.eval()  # set model to eval mode
        current_valiation_loss = 0.0
        ys, yhs = [], []
        with torch.no_grad():  # Don't store gradients.
            for i, batch in enumerate(valid_iter):
                words1, words2, x1, x2, y, seqlen1, seqlen2, seqlen3, x1_x2 = batch
                yhat = model.predict(x1_x2).to(torch.float32)
                yhs.extend(yhat.tolist())
                ys.extend(y.tolist())

        mse_loss = nn.MSELoss()
        mse_loss = mse_loss(torch.FloatTensor(yhs).flatten(), torch.FloatTensor(ys).flatten())
        print("Validation Mean Squared Error:", mse_loss)
        history.append(mse_loss)

        history.reverse()
        if history.index(
                max(history)) > args.early_stopping:  # Early stopping mechanism. If the best F1 score was more than 5 epochs ago, stop.
            run = False
        elif history.index(max(history)) == 0:  # Save weights if they improved performance on validation set.
            best_weights = model.state_dict()
        history.reverse()

    model.load_state_dict(best_weights)  # Load the best weights

    model.eval()
    yhats = []
    with torch.no_grad():
        for i, batch in enumerate(test_iter):
            words1, words2, x1, x2, y, seqlen1, seqlen2, seqlen3, x1_x2 = batch

            y_hat = model(x1_x2)
            yhats.extend(y_hat.tolist())

    # save test results to file to compute Pearson Correlation Coefficient
    with open(args.output_file, 'w') as fout:
        for yhat in yhats:
            fout.write("%f\n" % yhat[0])


if __name__ == '__main__':
    main()
