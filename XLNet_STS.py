import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils import data
from pytorch_transformers import XLNetTokenizer, XLNetModel

tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')

class NerDataset(data.Dataset):
    def __init__(self, fpath):
        """
        When given a file path, this loads the dataset for use in training.
        """
        entries = open(fpath, 'r')  #.read()  #.strip().split("\n")
        # entries = [entry for entry in entries if entry != '-DOCSTART- -X- -X- O']
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
        # print(words)
        x1 = tokenizer.tokenize(words1)
        x2 = tokenizer.tokenize(words2)
        x1 = tokenizer.convert_tokens_to_ids(x1)
        x2 = tokenizer.convert_tokens_to_ids(x2)
        y = tags

        # assert len(x)==len(y)==len(is_heads), f"len(x)={len(x)}, len(y)={len(y)}"
        seqlen1 = len(x1)
        seqlen2 = len(x2)
        # if seqlen >= 190:  # Because 32 batch size and the longest sample will overrun my VRAM
        #     return [], [], [], [], [], 0
        return words1, words2, x1, x2, y, seqlen1, seqlen2

def pad(batch):
    '''Zero pads each batch to the longest sample'''
    f = lambda x: [sample[x] for sample in batch]
    words1 = f(0)
    words2 = f(1)
    x1 = f(2)
    x2 = f(3)
    y = f(4)
    # is_heads = f(4)
    seqlens1 = f(-2)
    seqlens2 = f(-1)
    maxlen1 = np.array(seqlens1).max()
    maxlen2 = np.array(seqlens2).max()

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]  # pads each sequence with 0 values
    x1 = f(2, maxlen1)
    x2 = f(3, maxlen2)

    f = torch.LongTensor

    return words1, words2, f(x1), f(x2), torch.FloatTensor(y), seqlens1, seqlens2

#  Load the data set for training and validating
trainset = NerDataset('./data/sts-train.csv')
train_iter = data.DataLoader(dataset=trainset, batch_size=32, shuffle=True, collate_fn=pad)
validset = NerDataset('data/sts-dev.csv')
valid_iter = data.DataLoader(dataset=validset, batch_size=32, shuffle=False, collate_fn=pad)
testset = NerDataset('data/sts-test.csv')
test_iter = data.DataLoader(dataset=testset, batch_size=32, shuffle=False, collate_fn=pad)

class RecurrentNeuralNetwork(nn.Module):
    def __init__(self):
        super(RecurrentNeuralNetwork, self).__init__()
        self.xlnet = XLNetModel.from_pretrained('xlnet-large-cased')
        # self.rnn = nn.RNN(input_size=1024, bidirectional=True, hidden_size=1024//2, num_layers=3, batch_first=True)
        self.linear1 = nn.Linear(2048, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 1)
        self.activation = nn.Tanh()

    def forward(self, x1, x2):
        x1, _ = self.xlnet(x1.cuda())
        x1 = torch.sum(x1, dim=1)
        x2, _ = self.xlnet(x2.cuda())
        x2 = torch.sum(x2, dim=1)
        x = torch.cat((x1, x2), dim=1)
        # out, _ = self.rnn(x)  # Passes x matrix through RNN
        z1 = self.activation(self.linear1(x))  # Linear transformation from RNN's output to the number of classes
        z2 = self.activation(self.linear2(z1))
        z3 = self.linear3(z2)
        return z3

    def predict(self, x1, x2):
        '''
        If doing predictions, call this so that argmax returns the integer of the max index
        '''
        return self.forward(x1, x2)

model = RecurrentNeuralNetwork()  # RNN with embedding object
model.cuda()  #send model to the GPU
optimizer = optim.Adam(model.parameters(), lr=1e-5)  # Optimizing with Adam
loss_func = nn.MSELoss()

history = []  # list for storing historical F1 scores. Used for early stopping.
best_weights = None
run = True
while run:
    model.train()
    for i, batch in enumerate(train_iter):  # Runs batches on training set
        words1, words2, x1, x2, y, seqlens1, seqlens2 = batch
        optimizer.zero_grad()  # zero all gradients
        yhat = model(x1, x2)  # Forward pass through Neural Net

        yhat = yhat.reshape(-1) # reshapes to (num_sequences * sequence_length, num_classes)
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
            words1, words2, x1, x2, y, seqlens1, seqlens2 = batch
            yhat = model.predict(x1, x2).to(torch.float32)
            yhs.extend(yhat.tolist())
            ys.extend(y.tolist())
            # print(y.dtype)
            # print(yhat)
            # yhat = y_hat.reshape(-1)  # reshapes to (num_sequences * sequence_length, num_classes)
            # y = y.reshape(-1)  # reshapes to (num_sequences * sequence_length)

            # lv = y.cuda()-yhat
            # lv = torch.pow(2, lv).sum()
            # current_valiation_loss += lv
    # print(current_valiation_loss)
    mse_loss = nn.MSELoss()
    mse_loss = mse_loss(torch.FloatTensor(yhs), torch.FloatTensor(ys))
    print("Validation Mean Squared Error:", mse_loss)
    history.append(mse_loss)

    history.reverse()
    if history.index(max(history)) > 5:  # Early stopping mechanism. If the best F1 score was more than 5 epochs ago, stop.
        run = False
    elif history.index(max(history)) == 0:  #Save weights if they improved performance on validation set.
        best_weights = model.state_dict()
    history.reverse()

model.load_state_dict(best_weights)  # Load the best weights

model.eval()
yhats = []
with torch.no_grad():
    for i, batch in enumerate(test_iter):
        words1, words2, x1, x2, y, seqlens1, seqlens2 = batch

        y_hat = model.predict(x1, x2)
        yhats.extend(y_hat.tolist())

## save test results to file and compute metrics
with open("./output/test_xlnet.txt", 'w') as fout:
    for yhat in yhats:
        fout.write("%f\n"%yhat[0])