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
                where='post', label=self.label, c=c, linewidth=1)
        # points
        ax.plot(self.P[:, 0], self.P[:, 1],
                'o', c=c, markersize=3)
        # for x, y, label in zip(self.P[:, 0], self.P[:, 1], self.metadata):
        #     # label
        #     txt = ax.annotate(
        #         label, (x, y), textcoords="axes fraction", xytext=textoffset['xy'],
        #         ha='left', va='center', fontsize=5, clip_on=False,
        #         arrowprops=dict(
        #             arrowstyle='-|>', linestyle='--', color=c+'aa',
        #             connectionstyle="angle,angleA=180,angleB={}".format(angle),
        #             relpos=(0., .5), shrinkA=10, shrinkB=10))
        #     textoffset['xy'][1] -= 0.03

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


def roundIfFloat(s):
    try:
        return '{:.2g}'.format(float(s))
    except ValueError:
        return s


def roundFloats(s):
    return " ".join(roundIfFloat(su) for su in s.split())


def cleanConfStr(s):
    s = s.replace("'", '')
    s = re.sub(r'(.*)(-algorithm [^\s]* )(.*)', r'\2 \1 \3', s)
    return roundFloats(s)


def getData(csvPath):
    # read data from csv
    pat = re.compile(r'\[(.*), (.*)\], 0, (.*)')
    with open(csvPath) as f:
        data = np.array([re.findall(pat, s)[0] for s in f.readlines()])

    df = pd.DataFrame(data, columns=['earliness', 'accuracy', 'configuration'])
    df[['earliness', 'accuracy']] = df[['earliness', 'accuracy']].astype(float)
    # compute evaluation means by configuration
    df = df.groupby('configuration', as_index=False).mean()
    df = df.sort_values(by=['earliness'])
    Y = np.array(df[['earliness', 'accuracy']])
    metadata = np.array([cleanConfStr(d) for d in df['configuration']])
    return Y, metadata


def best(metric, val, l):
    if metric == 'method':
        return False
    if metric in ['delta', 'hmean']:
        return val == min(l)
    return val == max(l)


def formatCell(x, spec=None):
    formats = dict(int=     '{:8d}'+' '*4,
                   float=   '{:12.3f}',
                   float64= '{:12.3f}',
                   str=     ' {:>11s}')
    if spec and spec in ['b', 'bf', 'bold']:
        s = formats[x.__class__.__name__].format(x).strip()
        return formats['str'].format(f'\\bft{{{s}}}')
    return formats[x.__class__.__name__].format(x)


def printTable(d):
    for k, v in d.items():
        print('{:>10s}'.format(k)+''.join(formatCell(vi) for vi in v))


def hlfmt(val, metric=None, l=None):
    if metric and l and best(metric, val, l):
        return formatCell(val, 'bf')
    return formatCell(val)


def twoColumn(s):
    return r'\multicolumn{2}{c}{' + f'{s}' + r'}'


def latexMetricTable(d):
    tex = '\\begin{tabular}{l'+'r'*(len(d['method']))+'}\n'
    for k, v in d.items():
        tex += ' &'.join([hlfmt(i, k, v) for i in reversed(v+[k])]) + r'\\'
        if k == 'method':
            tex += '\\hline'
        tex += '\n'
    tex += '\\end{tabular}\n'
    return tex


def latexDatasetTable(datasetTabel, labels):
    tex = r'\begin{tabular}{@{\extracolsep{4pt}}l'+'rr'*(len(labels))+'@{}}\n'
    tex += '&'+' & '.join([twoColumn(l) for l in labels]) + '\\\\\n'
    tex += ''.join([f'\\cline{{{2*i+2}-{2*i+3}}}' for i in range(len(labels))]) + '\n'
    tex += '&'+' & '.join(['HV', r'$\Delta$'] * len(labels)) + '\\\\\\hline\n'
    for row in datasetTabel:
        vals = sum(row[1:], ())
        hvs = [hlfmt(hv, 'hv', vals[::2]) for hv in vals[::2]]
        dlts = [hlfmt(dlt, 'delta', vals[1::2]) for dlt in vals[1::2]]
        cells = [hlfmt(row[0])]+[j for i in zip(hvs, dlts) for j in i]
        tex += ' &'.join(cells) + '\\\\\n'
    tex += '\\end{tabular}\n'
    return tex


def processData(dataset, labels):
    files = [os.path.basename(p) for p in sorted(glob.glob(f'output/pareto/{dataset}/*'), reverse=True)]
    print(f'{dataset}')
    if (set(files) != set(labels)):
        print('some files are missing')
    fig, ax = plt.subplots(figsize=(8, 4.8))
    color = colors()
    ruler = 1.02
    metricTable = {}
    textoffset = dict(xy=[ruler, .98])
    angle = 80
    datasetTableRow = []
    for label in reversed(labels):
        try:
            Y, metadata = getData(f'output/test/{dataset}/{label}/test.csv')
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
                HV=pareto.hypervolume()
            )
            datasetTableRow = [(metrics['HV'], metrics['delta'])] + datasetTableRow
            metricTable = {k: metricTable.get(k, []) + [metrics[k]] for k in metrics}
        except FileNotFoundError:
            datasetTableRow += [('?', '?')]
            continue

    # add column with means
    # for k in metricTable:
    #     if k != 'method':
    #         metricTable[k] += [np.mean(metricTable[k])]
    #     else:
    #         metricTable[k] += ['mean']

    # print metrics to terminal
    printTable(metricTable)
    print()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ticks = np.arange(0, 1.1, .1)
    plt.xticks(ticks)
    plt.yticks(ticks)
    ax.grid()
    ax.legend(loc=(ruler, 0))
    plt.title(f'Earliness-Accuracy tradeoff\n{dataset}')
    plt.xlabel("Earliness")
    plt.ylabel("Error rate")

    fig.tight_layout()
    # save plot
    plt.savefig(f'output/plot/{dataset}.pdf')
    # save tex table
    with open(f'output/tex/{dataset}.tex', 'w') as f:
        f.write(latexMetricTable(metricTable))
    return [dataset] + datasetTableRow


def main():
    os.chdir(os.path.dirname(sys.argv[0]))
    datasetDirs = glob.glob('output/pareto/*')
    if not len(datasetDirs):
        sys.exit('No files found in output/pareto/. Nothing to be done.')
    os.makedirs('output/plot/', exist_ok=True)
    os.makedirs('output/tex/', exist_ok=True)
    labels = ['mo-ects',
              'mo-ecdire',
              'mo-srcf',
              'mo-relclass',
              'so-all',
              'mo-all']
    datasetTable = []
    for dataset in sorted([os.path.basename(p) for p in datasetDirs]):
        datasetTable += [processData(dataset, labels)]
    with open(f'output/tex/dataset.tex', 'w') as f:
        f.write(latexDatasetTable(datasetTable, labels))


if __name__ == '__main__':
    main()
