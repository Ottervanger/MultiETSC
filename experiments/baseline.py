#!/usr/bin/env python
import numpy as np
import os
import sys
import glob

from sklearn.neighbors import KNeighborsClassifier


def rmDominated(Y):
    inp = np.ones(len(Y), dtype="bool")
    for i in range(len(Y)):
        if inp[i]:
            inp = inp & np.any(Y < Y[i], 1)
            inp[i] = True
    return Y[inp]


def computeBaseline(XTrain, yTrain, XTest, yTest):
    p = np.empty((0, 2))
    sLen = XTrain.shape[1]

    # zeroR classifier
    u, c = np.unique(yTrain, return_counts=True)
    zeroR = u[np.argmax(c)]
    p = np.append(p, [[0, np.mean(yTest != zeroR)]], axis=0)

    # 1-NN Euclidean classifiers for each time step
    for t in range(1, 1+sLen):
        kNN = KNeighborsClassifier(n_neighbors=1, n_jobs=4)
        kNN.fit(XTrain[:, :t], yTrain)
        erRate = 1 - kNN.score(XTest[:, :t], yTest)
        p = np.append(p, [[t/sLen, erRate]], axis=0)
    return p


def getData(dataset):
    UCR_ROOT = os.environ.get('UCR_ROOT', '/scratch/ottervanger/UCR/')
    train = np.genfromtxt(f'{UCR_ROOT}{dataset}/{dataset}_TRAIN.tsv', delimiter='\t')
    test =  np.genfromtxt(f'{UCR_ROOT}{dataset}/{dataset}_TEST.tsv',  delimiter='\t')
    XTest,  yTest  = test[:, 1:],  test[:, 0]
    XTrain, yTrain = train[:, 1:], train[:, 0]
    return XTrain, yTrain, XTest, yTest


def main():
    os.chdir(os.path.dirname(sys.argv[0]))
    datasets = [os.path.basename(p) for p in glob.glob('output/test/*')]
    os.makedirs(f'output/baseline/', exist_ok=True)
    done = [os.path.basename(p).split('.')[0] for p in glob.glob('output/baseline/*')]
    for dataset in [d for d in datasets if d not in done]:
        print(f'START: {dataset}')
        bline = rmDominated(computeBaseline(*getData(dataset)))
        np.savetxt(f'output/baseline/{dataset}.csv', bline, delimiter=",")
        print(f'DONE:  {dataset}\n')


if __name__ == '__main__':
    main()
