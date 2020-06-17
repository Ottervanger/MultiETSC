#!/usr/bin/env python
from matplotlib import pyplot as plt
import numpy as np
import os
import sys
import re


OUTPATH_BASE = 'output/{}/validation-{}.csv'

def colors(opaque=False):
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
            yield c if not opaque else c + '99'

class Pareto:
    def  __init__(self, Y, metadata, label):
        inp = np.ones(len(Y), dtype="bool")
        for i in range(len(Y)):
            if inp[i]:
                inp = inp & np.any(Y < Y[i],1)
                inp[i] = True
        self.P = Y[inp]
        self.metadata = metadata[inp]
        self.size = len(self.P)
        self.label = label

    def plot(self, ax, c):
        ax.plot(self.P[:,0],self.P[:,1], '-o', c=c, label=self.label, linewidth=1, markersize=3)

    def hmean(self):
        return np.min(1-self.P.shape[1]/np.sum((1-self.P)**-1,1))

    def hypervolume(self):
        '''
        compute dominated hypervolume given pareto set Y
        pre: Y consists of two dimensional non-dominated
             points sorted by the first objective.
        '''
        ref = np.array([1.,1.])
        s = 0
        for y in self.P:
            s += np.prod(ref-y)
            ref[1] = y[1]
        return s

    def M3(self):
        return np.linalg.norm(self.P[0]-self.P[-1])

    def delta(self):
        di = np.zeros(self.size-1)
        for i in range(len(di)):
            di[i] = np.linalg.norm(self.P[i]-self.P[i+1])
        df = np.linalg.norm(self.P[0]-np.array([0,1]))
        dl = np.linalg.norm(self.P[-1]-np.array([1,0]))
        return (df+dl+np.sum(np.abs(di-np.mean(di))))/(df+dl+np.sum(di))

def getData(csvPath):
    # read data from csv
    pat = re.compile('\[(.*), (.*)\], 0, (.*)')
    with open(csvPath) as f:
        data = np.array([re.findall(pat, s)[0] for s in f.readlines()])
    # sort by the first objective
    data = data[np.argsort(data[:,0])]
    Y = np.array(data[:,:2], dtype=float)
    metadata = data[:,2]
    return Y, metadata

def getTargets():
    if len(sys.argv) < 2:
        return [['mo','0']]
    else:
        return [[sys.argv[i], sys.argv[i+1]] for i in range(len(sys.argv)-1) if i%2]

def main():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    color = colors()
    for method, seed in getTargets():
        try:
            Y, metadata = getData(OUTPATH_BASE.format(method, seed))
            pareto = Pareto(Y, metadata, label='{} {}'.format(method, seed))
            pareto.plot(ax, c=next(color))
            sf = '|{:>12s}: {:8.4f}'
            sd = '|{:>12s}: {:3d}'
            print('| method: {}, seed: {}'.format(method, seed))
            print(sd.format('size', pareto.size))
            print(sf.format('delta', pareto.delta()))
            print(sf.format('M3', pareto.M3()))
            print(sf.format('min hmean', pareto.hmean()))
            print(sf.format('hypervolume', pareto.hypervolume()))
            print()
            
        except FileNotFoundError:
            continue
    ax.grid()
    ax.legend()
    plt.title("Earliness-Accuracy tradeoff")
    plt.xlabel("Earliness")
    plt.ylabel("Error rate")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.savefig('pareto.pdf')

if __name__ == '__main__':
    main()
