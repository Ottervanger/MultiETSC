import numpy as np
import torch
from model import EARLIEST
from dataset import SyntheticTimeSeries
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


def loadData(BATCH_SIZE=1):
    tsL = 10
    data = SyntheticTimeSeries(T=tsL)
    tsNc = data.N_CLASSES  # Number of classes for classification
    train_sampler = SubsetRandomSampler(data.train_ix)
    test_sampler = SubsetRandomSampler(data.test_ix)
    train_loader = torch.utils.data.DataLoader(dataset=data,
                                               batch_size=BATCH_SIZE,
                                               sampler=train_sampler,
                                               drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=data,
                                              batch_size=BATCH_SIZE,
                                              sampler=test_sampler,
                                              drop_last=True)
    return train_loader, test_loader, tsNc, tsL


def trainModel(train_loader, tsL, tsNf, tsNc, hiddenDim,
               cellType, nLayers, LAMBDA, lr, epochs):
    exponentials = exponentialDecay(epochs)
    # --- initialize the model and the optimizer ---
    model = EARLIEST(N_FEATURES=tsNf, N_CLASSES=tsNc, HIDDEN_DIM=hiddenDim,
                     CELL_TYPE=cellType, N_LAYERS=nLayers, DF=1., LAMBDA=LAMBDA)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

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
        for i, (X, y) in enumerate(train_loader):
            X = torch.transpose(X, 0, 1)
            # --- Forward pass ---
            predictions = model(X)

            # --- Compute gradients and update weights ---
            optimizer.zero_grad()
            loss = model.applyLoss(predictions, y)
            loss.backward()
            loss_sum += loss.item()
            optimizer.step()
            # scheduler.step()

            # --- Collect prediction locations ---
            for j in range(len(y)):
                training_locations.append(model.locations[j])
                training_predictions.append(predictions[j])
            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1:3d}/{epochs:3d}], Step [{i+1:4d}/{len(train_loader):4d}], Loss: {loss.item():10.4f}', end='\r')
        training_loss.append(np.round(loss_sum/len(train_loader), 3))
    training_locations = torch.stack(training_locations).numpy()
    print()
    return model


def test(model, test_loader, tsL):
    testing_loss = []
    testing_predictions = []
    testing_labels = []
    testing_locations = []
    loss_sum = 0
    for i, (X, y) in enumerate(test_loader):
        X = torch.transpose(X, 0, 1)
        predictions = model(X)
        for j in range(len(y)):
            testing_locations.append(model.locations[j])
            testing_predictions.append(predictions[j])
            testing_labels.append(y[j])
        loss = model.applyLoss(predictions, y)
        loss.backward()
        loss_sum += loss.item()
        testing_loss.append(np.round(loss_sum/len(test_loader), 3))
    _, testing_predictions = torch.max(torch.stack(testing_predictions).detach(), 1)
    testing_predictions = np.array(testing_predictions)

    earliness = np.mean(testing_locations)/tsL
    errorRate = 1 - accuracy_score(testing_labels, testing_predictions)
    return earliness, errorRate


def main():
    torch.set_num_threads(1)
    start = time.time()

    # --- hyperparameters ---
    tsNf = 1
    hiddenDim = 10
    cellType = 'LSTM'  # in ['RNN', 'LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU']
    nLayers = 1
    LAMBDA = 1e-02

    lr = 1e-3
    epochs = 20

    # load data
    train_loader, test_loader, tsNc, tsL = loadData()
    model = trainModel(train_loader, tsL, tsNf, tsNc,
                       hiddenDim, cellType, nLayers, LAMBDA,
                       lr, epochs)
    earliness, errorRate = test(model, test_loader, tsL)
    print(f'earliness:  {earliness:10.4f}')
    print(f'error rate: {errorRate:10.4f}')
    print(f'elapsed:    {time.time()-start:10.4f}')


if __name__ == '__main__':
    main()
