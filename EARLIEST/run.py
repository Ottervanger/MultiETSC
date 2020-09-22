#!/usr/bin/env python
import numpy as np
import torch
from model import EARLIEST
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import time


# --- methods ---
def exponentialDecay(N):
    tau = 1
    tmax = 4
    t = np.linspace(0, tmax, N)
    y = np.exp(-t/tau)
    y = torch.FloatTensor(y)
    return y/10.


def makeLoader(filename, cMap=None):
    dat = np.genfromtxt(filename, delimiter='\t')
    if not cMap:
        cMap = {v: i for i, v in enumerate(np.unique(dat[:, 0]))}
    dat[:, 0] = np.fromiter((cMap[v] for v in dat[:, 0]), dat.dtype)
    lst = [(torch.tensor(np.expand_dims(x[1:], 1).astype(np.float32), dtype=torch.float),
            torch.tensor(x[0].astype(np.int32), dtype=torch.long)) for x in dat]
    loader = torch.utils.data.DataLoader(dataset=lst)
    return loader, cMap


def loadData(trainFile, testFile):
    train_loader, cMap = makeLoader(trainFile)
    test_loader, _ = makeLoader(testFile, cMap)

    tsNc = len(cMap)
    tsL = len(train_loader.dataset[0][0])
    return train_loader, test_loader, tsNc, tsL


def trainModel(data, tsL, tsNf, tsNc, hiddenDim, cellType, nLayers, LAMBDA,
               lr, lrf=1., epochs=10, df=1.):
    exponentials = exponentialDecay(epochs)
    # --- initialize the model and the optimizer ---
    model = EARLIEST(N_FEATURES=tsNf, N_CLASSES=tsNc, HIDDEN_DIM=hiddenDim,
                     CELL_TYPE=cellType, N_LAYERS=nLayers, DF=df, LAMBDA=LAMBDA)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lrf)

    # --- training ---
    training_loss = []
    training_locations = []
    training_predictions = []
    for epoch in range(epochs):
        model._REWARDS = 0
        model._r_sums = np.zeros(tsL).reshape(1, -1)
        model._r_counts = np.zeros(tsL).reshape(1, -1)
        model._epsilon = exponentials[epoch]
        loss_sum = 0
        for i, (X, y) in enumerate(data):
            X = torch.transpose(X, 0, 1)
            # --- Forward pass ---
            predictions = model(X)

            # --- Compute gradients and update weights ---
            optimizer.zero_grad()
            loss = model.applyLoss(predictions, y)
            loss.backward()
            loss_sum += loss.item()
            optimizer.step()
            scheduler.step()

            # --- Collect prediction locations ---
            for j in range(len(y)):
                training_locations.append(model.locations[j])
                training_predictions.append(predictions[j])
            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1:3d}/{epochs:3d}], Step [{i+1:4d}/{len(data):4d}], Loss: {loss.item():10.4f}', end='\r')
        training_loss.append(np.round(loss_sum/len(data), 3))
    training_locations = torch.stack(training_locations).numpy()
    print()
    return model


def test(model, data, tsL):
    testing_loss = []
    testing_predictions = []
    testing_labels = []
    testing_locations = []
    loss_sum = 0
    for i, (X, y) in enumerate(data):
        X = torch.transpose(X, 0, 1)
        predictions = model(X)
        for j in range(len(y)):
            testing_locations.append(model.locations[j])
            testing_predictions.append(predictions[j])
            testing_labels.append(y[j])
        loss = model.applyLoss(predictions, y)
        loss.backward()
        loss_sum += loss.item()
        testing_loss.append(np.round(loss_sum/len(data), 3))
    _, testing_predictions = torch.max(torch.stack(testing_predictions).detach(), 1)
    testing_predictions = np.array(testing_predictions)

    earliness = np.mean(testing_locations)/tsL
    errorRate = 1 - accuracy_score(testing_labels, testing_predictions)
    return earliness, errorRate


def main():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    start = time.time()

    # --- hyperparameters ---
    hiddenDim = 10
    cellType = 'LSTM'  # in ['RNN', 'LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU']
    nLayers = 1
    LAMBDA = 1e-10
    df = 1.            # discount factor for optimizing the Controller

    lr = 1e-3
    lrf = 1.
    epochs = 100

    # load data
    train_loader, test_loader, tsNc, tsL = loadData(
        '/scratch/ottervanger/UCR/CBF/CBF_TRAIN.tsv',
        '/scratch/ottervanger/UCR/CBF/CBF_TEST.tsv')
    model = trainModel(data=train_loader, tsL=tsL, tsNf=1, tsNc=tsNc,
                       hiddenDim=hiddenDim, cellType=cellType, nLayers=nLayers,
                       LAMBDA=LAMBDA, lr=lr, lrf=lrf, epochs=epochs, df=df)
    earliness, errorRate = test(model=model, data=test_loader, tsL=tsL)
    print(f'earliness:  {earliness:10.4f}')
    print(f'error rate: {errorRate:10.4f}')
    print(f'elapsed:    {time.time()-start:10.4f}')


if __name__ == '__main__':
    main()
