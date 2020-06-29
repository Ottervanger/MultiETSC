#!/usr/bin/env python
import numpy as np
import os
import sys
import re
import glob


os.chdir(os.path.dirname(sys.argv[0]))

def usage():
    print('Please provide glob for input files:\n  {} ECG200*')
    sys.exit()

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

    def plot(self, ax, c, textoffset, angle):
        ax.plot(self.P[:,0],self.P[:,1], '-o', c=c, label=self.label, linewidth=1, markersize=3)
        # adding labels
        labels = [re.sub('^-algorithm ([^/]*)[^\s]* (.*)$', '\\1 \\2', i.replace("'", '')) for i in self.metadata]
        for x, y, label in zip(self.P[:,0], self.P[:,1], labels):
            # label
            txt = ax.annotate(label, (x, y), textcoords="axes fraction", xytext=textoffset['xy'],
                        ha='left', va='center', fontsize=8, clip_on=False,
                        bbox=dict(boxstyle="round", fc=(1,1,1,.8), ec='.8'),
                        arrowprops=dict(arrowstyle='-|>', linestyle='--', color=c+'aa',
                        connectionstyle="angle,angleA=180,angleB={}".format(angle),
                        relpos=(0., .5), shrinkA=10, shrinkB=10))
            textoffset['xy'][1] -= 0.06

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

def printTable(d):
    formats = dict(int='{:3d}'+' '*5, float='{:8.4f}', float64='{:8.4f}', str='{:>8s}')
    for k, v in d.items():
         print(('{:>12s}'+formats[v[0].__class__.__name__]*len(v)).format(k, *v))

def best(metric, val, l):
    if metric == 'method':
        return False
    if metric in ['delta', 'hmean']:
        return val == min(l)
    return val == max(l)

def hlfmt(metric, val, l):
    formats = dict(int='{:3d}'+' '*5, float='{:8.4f}', float64='{:8.4f}', str='{:>8s}')
    if best(metric, val, l):
        return '\\textbf{'+formats[val.__class__.__name__].format(val)+'}'
    return formats[val.__class__.__name__].format(val)

def latexTable(d):
    tex = '\\begin{tabular}{l r r}\n'
    for k, v in d.items():
        tex += ' &'.join([hlfmt(k, i, v) for i in [k]+v]) + '\\\\'
        if k == 'method':
            tex += '\\hline'
        tex += '\n'
    tex += '\\end{tabular}\n'
    return tex

def main():
    color = colors()
    ruler = 1.02
    if (len(sys.argv) < 2):
        usage()
    files = glob.glob('output/results/'+sys.argv[1])
    if not len(files):
        usage()
    print('Files found:')
    for f in files:
        print('  '+f)

    # prepare for plotting
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(8,4.8))
    metricTable = {}
    textoffset = dict(xy=[ruler,.98])
    angle = 75
    for f in files:
        try:
            Y, metadata = getData(f)
            label = f.split('/')[-1].split('.')[0]
            method = label.split('-')[1].upper()
            pareto = Pareto(Y, metadata, label=method)
            pareto.plot(ax, c=next(color), textoffset=textoffset, angle=angle)
            textoffset['xy'][1] -= 0.05
            angle -= 15
            metrics = dict(
                method=method,
                size=pareto.size,
                delta=pareto.delta(),
                M3=pareto.M3(),
                hmean=pareto.hmean(),
                hypervolume=pareto.hypervolume()
            )
            metricTable = {k: metricTable.get(k, []) + [metrics[k]] for k in metrics}
        except FileNotFoundError:
            continue
    
    # print metrics to terminal
    printTable(metricTable)
    dataname = files[0].split('/')[-1].split('-')[0]
    seed = re.sub('.*-([0-9]*)\\..*', '\\1', files[0])
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid()
    ax.legend(loc=(ruler,0))
    plt.title("Earliness-Accuracy tradeoff\n{}".format(dataname))
    plt.xlabel("Earliness")
    plt.ylabel("Error rate")

    fig.tight_layout()
    # save plot
    plt.savefig('output/plots/{}-{}.pdf'.format(dataname, seed))
    # save tex table
    with open('output/tex/{}-{}.tex'.format(dataname, seed), 'w') as f:
        f.write(latexTable(metricTable))

if __name__ == '__main__':
    main()
