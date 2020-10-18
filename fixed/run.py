#!/usr/bin/env python
import numpy as np
import sys
import time

from sklearn.neighbors import KNeighborsClassifier


def computeBaseline(XTrain, yTrain, XTest, yTest, percLen):
    p = np.empty((0, 2))
    t = int(np.round(XTrain.shape[1] * percLen))
    if (t == 0):
        # zeroR classifier
        u, c = np.unique(yTrain, return_counts=True)
        zeroR = u[np.argmax(c)]
        return 0, np.mean(yTest != zeroR)

    # 1-NN Euclidean classifier
    kNN = KNeighborsClassifier(n_neighbors=1, n_jobs=4)
    kNN.fit(XTrain[:, :t], yTrain)
    erRate = 1 - kNN.score(XTest[:, :t], yTest)
    return t/XTrain.shape[1], erRate


def getData(trainFile, testFile):
    train = np.genfromtxt(trainFile, delimiter='\t')
    test =  np.genfromtxt(testFile,  delimiter='\t')
    XTest,  yTest  = test[:, 1:],  test[:, 0]
    XTrain, yTrain = train[:, 1:], train[:, 0]
    return XTrain, yTrain, XTest, yTest


def getArgs(arg):
    argIter = iter(sys.argv)
    for argi in argIter:
        if argi == '-data':
            arg['trainFile'] = next(argIter)
            arg['testFile'] = next(argIter)
        elif argi[0] == '-':
            arg[argi.strip('-')] = type(arg.get(argi.strip('-'), 0))(next(argIter))
    return arg


def main():
    start = time.time()

    # --- hyperparameters ---
    arg = dict(percLen=0.0)

    try:
        arg = getArgs(arg)
    except StopIteration:
        sys.exit('Missing commandline arguments')

    if not (arg.get('trainFile', '') and arg.get('testFile', '')):
        print(arg)
        sys.exit('No data provided')

    earliness, errorRate = computeBaseline(*getData(arg['trainFile'], arg['testFile']), arg['percLen'])
    print(f'Result: SUCCESS, {time.time()-start:g}, [{earliness:g}, {errorRate:g}], 0')


if __name__ == '__main__':
    main()
