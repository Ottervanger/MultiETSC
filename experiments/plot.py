#!/usr/bin/env python
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import re
import glob
import itertools


def colors(opaque=False):
    while True:
        for c in ['#1f77b4',  # blue
                  '#ff7f0e',  # orange
                  '#2ca02c',  # green
                  '#d62728',  # red
                  '#9467bd',  # purple
                  '#8c564b',  # brown
                  '#e377c2',  # pink
                  '#7f7f7f',  # grey
                  '#bcbd22',  # olive
                  '#17becf'   # cyan
                  ]:
            yield c if not opaque else c + '99'


def roundIfFloat(s):
    try:
        return '{:.2g}'.format(float(s))
    except ValueError:
        return s


def roundFloatsInString(s):
    return " ".join(roundIfFloat(su) for su in s.split())


class Pareto:
    def __init__(self, Y, metadata, label):
        inp = np.ones(len(Y), dtype="bool")
        for i in range(len(Y)):
            if inp[i]:
                inp = inp & np.any(Y < Y[i], 1)
                inp[i] = True
        self.P = Y[inp]
        self.metadata = metadata[inp]
        self.size = len(self.P)
        self.label = label

    def plot(self, ax, c, textoffset, angle):
        # stepped line
        ax.step(np.hstack((0, self.P[:, 0], 1)),
                np.hstack((1, self.P[:, 1], 0)),
                where='post', label=self.label, linewidth=1)
        # points
        ax.plot(self.P[:, 0], self.P[:, 1],
                'o', c=c, markersize=3)
        # adding labels
        labels = [re.sub('^-algorithm ([^/]*)[^\s]* (.*)$', '\\1 \\2', i.replace("'", '')) for i in self.metadata]
        for x, y, label in zip(self.P[:, 0], self.P[:, 1], labels):
            # label
            txt = ax.annotate(
                label, (x, y), textcoords="axes fraction", xytext=textoffset['xy'],
                ha='left', va='center', fontsize=5, clip_on=False,
                arrowprops=dict(
                    arrowstyle='-|>', linestyle='--', color=c+'aa',
                    connectionstyle="angle,angleA=180,angleB={}".format(angle),
                    relpos=(0., .5), shrinkA=10, shrinkB=10))
            textoffset['xy'][1] -= 0.03

    def hmean(self):
        return np.min(1-self.P.shape[1]/np.sum((1-self.P)**-1, 1))

    def hypervolume(self):
        '''
        compute dominated hypervolume given pareto set Y
        pre: Y consists of two dimensional non-dominated
             points sorted by the first objective.
        '''
        ref = np.array([1., 1.])
        s = 0
        for y in self.P:
            s += np.prod(ref-y)
            ref[1] = y[1]
        return s

    def M3(self):
        return np.linalg.norm(self.P[0]-self.P[-1])

    def delta(self):
        if (self.size < 3):
            return np.inf
        di = np.zeros(self.size-1)
        for i in range(len(di)):
            di[i] = np.linalg.norm(self.P[i]-self.P[i+1])
        df = np.linalg.norm(self.P[0]-np.array([0, 1]))
        dl = np.linalg.norm(self.P[-1]-np.array([1, 0]))
        return (df+dl+np.sum(np.abs(di-np.mean(di))))/(df+dl+np.sum(di))


def getData(csvPath):
    # read data from csv
    pat = re.compile('\[(.*), (.*)\], 0, (.*)')
    with open(csvPath) as f:
        data = np.array([re.findall(pat, s)[0] for s in f.readlines()])

    df = pd.DataFrame(data, columns=['earliness', 'accuracy', 'configuration'])
    df[['earliness', 'accuracy']] = df[['earliness', 'accuracy']].astype(float)
    # compute evaluation means by configuration
    df = df.groupby('configuration', as_index=False).mean()
    df = df.sort_values(by=['earliness'])
    Y = np.array(df[['earliness', 'accuracy']])
    metadata = np.array([roundFloatsInString(re.sub('(.*)(-algorithm [^\s]* )(.*)', '\\2 \\1 \\3', d)) for d in df['configuration']])
    return Y, metadata


def printTable(d):
    formats = dict(int='{:3d}'+' '*5, float='{:8.4f}', float64='{:8.4f}', str='{:>8s}')
    for k, v in d.items():
        print(('{:>12s}'+''.join(formats[vi.__class__.__name__] for vi in v)).format(k, *v))


def best(metric, val, l):
    if metric == 'method':
        return False
    if metric in ['delta', 'hmean']:
        return val == min(l)
    return val == max(l)


def hlfmt(metric, val, l):
    formats = dict(int='{:3d}'+' '*5, float='{:8.4f}', float64='{:8.4f}', str='{:>8s}')
    # if best(metric, val, l):
    #     return '\\textbf{'+formats[val.__class__.__name__].format(val)+'}'
    return formats[val.__class__.__name__].format(val)


def latexTable(d):
    tex = '\\begin{tabular}{l'+' r'*(len(d['method']))+'}\n'
    for k, v in d.items():
        tex += ' &'.join([hlfmt(k, i, v) for i in [k]+v]) + '\\\\'
        if k == 'method':
            tex += '\\hline'
        tex += '\n'
    tex += '\\end{tabular}\n'
    return tex


def namesFromFiles(files):
    # get files and extract the varying part as labels
    r = re.compile('.*/(.*)-(.*)-(.*)-(.*).csv')
    conds = [list(sorted(set([r.match(f).group(i) for f in files]))) for i in range(1, 5)]
    flabels = [('output/validation/'+'-'.join(i)+'.csv', '-'.join([v for j, v in enumerate(i) if len(conds[j]) > 1])) for i in itertools.product(*conds)]
    title = '-'.join(c[0] for c in conds if len(c) == 1)
    return flabels, title


def processData(title, labels):
    print('Processing {} ({} files)'.format(title, len(labels)))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    color = colors()
    ruler = 1.02
    metricTable = {}
    textoffset = dict(xy=[ruler, .98])
    angle = 80
    for label in labels:
        try:
            Y, metadata = getData('output/validation/'+title+'-'+label+'.csv')
            pareto = Pareto(Y, metadata, label=label)
            pareto.plot(ax, c=next(color), textoffset=textoffset, angle=angle)
            textoffset['xy'][1] -= 0.01
            angle -= 5
            metrics = dict(
                method=label,
                size=pareto.size,
                delta=pareto.delta(),
                M3=pareto.M3(),
                hmean=pareto.hmean(),
                hypervolume=pareto.hypervolume()
            )
            metricTable = {k: metricTable.get(k, []) + [metrics[k]] for k in metrics}
        except FileNotFoundError:
            continue

    # add column with means
    for k in metricTable:
        if k != 'method':
            metricTable[k] += [np.mean(metricTable[k])]
        else:
            metricTable[k] += ['mean']

    # print metrics to terminal
    printTable(metricTable)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ticks = np.arange(0, 1.1, .1)
    plt.xticks(ticks)
    plt.yticks(ticks)
    ax.grid()
    ax.legend(loc=(ruler, 0))
    plt.title("Earliness-Accuracy tradeoff\n{}".format(title))
    plt.xlabel("Earliness")
    plt.ylabel("Error rate")

    fig.tight_layout()
    # save plot
    plt.savefig('output/plot/{}.pdf'.format(title))
    # save tex table
    with open('output/tex/{}.tex'.format(title), 'w') as f:
        f.write(latexTable(metricTable))


def main():
    os.chdir(os.path.dirname(sys.argv[0]))
    files = glob.glob('output/validation/*')
    if not len(files):
        sys.exit('No files found in output/validation/. Nothing to be done.')

    # get files and extract the varying part as labels
    r = re.compile('.*/(.*)-(.*)-(.*)-(.*).csv')
    conds = [list(sorted(set([r.match(f).group(i) for f in files]))) for i in range(1, 4)]

    for title in ['-'.join(i) for i in itertools.product(*conds)]:
        fls = glob.glob('output/validation/{}*'.format(title))
        labels = list(sorted(set([r.match(f).group(4) for f in fls])))
        processData(title, labels)


if __name__ == '__main__':
    main()
