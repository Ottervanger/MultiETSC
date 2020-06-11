from matplotlib import pyplot as plt
import numpy as np
import os
import sys
import re

def colors():
    while True:
        for c in ['#1f77b4', # blue
                  '#ff7f0e', # orange
                  '#2ca02c', # green
                  '#d62728', # red
                  '#9467bd', # purple
                  '#8c564b', # brown
                  '#e377c2', # pink
                  '#7f7f7f', # grey
                  '#bcbd22', # olive
                  '#17becf'  # cyan
                 ]:
            yield c

OUTPATH_BASE = 'output/results/detailed-traj-run-{}.csv'

def pareto(Y):
    '''
    returns a boolean array indicating which items in Y are 
    in the pareto front
    '''
    p = np.ones(len(Y), dtype="bool")
    for i in range(len(Y)):
        if p[i]:
            p = p & np.any(Y < Y[i],1)
            p[i] = True
    return p

def paretoPlot(Y, ax, color='#17becf'):
    Y = Y[pareto(Y)]
    ax.plot(Y[:,0],Y[:,1], '-o', c=color)

def getData(csvPath):
    # read data from csv
    pat = re.compile('[",\s\[\]]*[",][",\s\[\]]*')
    with open(csvPath) as f:
        data = np.array([re.split(pat, s)[1:-1] for s in f.readlines()[2:]])
    # sort by the first objective
    data = data[np.argsort(data[:,1])]
    Y = np.array(data[:,1:3], dtype=float)
    metadata = np.hstack((data[:,[0]],data[:,3:]))
    return Y, metadata

def getTargets():
    if len(sys.argv) < 2:
        return ['0']
    else:
        return sys.argv[1:]

def main():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    color = colors()
    for seed in getTargets():
        try:
            Y, metadata = getData(OUTPATH_BASE.format(seed))
            paretoPlot(Y, ax, color=next(color)) 
        except FileNotFoundError:
            continue
    ax.grid()
    plt.title("Earliness-Accuracy tradeoff")
    plt.xlabel("Earliness")
    plt.ylabel("Error rate")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.savefig('pareto.pdf')

if __name__ == '__main__':
    main()
