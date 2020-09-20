#!/usr/bin/env python
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import re
import glob
import itertools
import json
import seaborn as sns
from scipy.stats import wilcoxon, binom_test


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
    def __init__(self, Y, metadata=None, label=None):
        inp = np.ones(len(Y), dtype="bool")
        for i in range(len(Y)):
            if inp[i]:
                inp = inp & np.any(Y < Y[i], 1)
                inp[i] = True
        self.P = Y[inp]
        order = np.argsort(self.P[:, 0])
        self.P = self.P[order]
        self._size = len(self.P)
        if metadata is not None:
            self.metadata = np.array([cleanConfStr(d) for d in metadata[inp]])[order]
        if label is not None:
            self.label = label

    def plot(self, ax, c, annotate=None):
        # stepped line
        ax.step(np.hstack((0, self.P[:, 0], 1)),
                np.hstack((1, self.P[:, 1], 0)),
                where='post', label=self.label, c=c, linewidth=1)
        # points
        ax.plot(self.P[:, 0], self.P[:, 1],
                'o', c=c, markersize=3)
        if annotate:
            for x, y, label in zip(self.P[:, 0], self.P[:, 1], self.metadata):
                # label
                txt = ax.annotate(
                    label, (x, y), textcoords="axes fraction", xytext=annotate['xyoffset'],
                    ha='left', va='center', fontsize=5, clip_on=False,
                    arrowprops=dict(
                        arrowstyle='-|>', linestyle='--', color=c+'aa',
                        connectionstyle="angle,angleA=180,angleB={}".format(annotate['angle']),
                        relpos=(0., .5), shrinkA=10, shrinkB=10))
                annotate['xyoffset'][1] -= 0.03
            annotate['xyoffset'][1] += annotate['ystep']
            annotate['angle'] += annotate['anglestep']

    def hmean(self):
        return np.min(1-2/np.sum((1-self.P[np.all(self.P != 1., 1)])**-1, 1))

    def HV(self):
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
        if (self._size < 2):
            return np.inf
        di = np.zeros(self._size-1)
        for i in range(len(di)):
            di[i] = np.linalg.norm(self.P[i]-self.P[i+1])
        df = np.linalg.norm(self.P[0]-np.array([0, 1]))
        dl = np.linalg.norm(self.P[-1]-np.array([1, 0]))
        return (df+dl+np.sum(np.abs(di-np.mean(di))))/(df+dl+np.sum(di))

    def size(self):
        return self._size


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
    metadata = np.array(df['configuration'])
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


def processData(dataset, methods):
    files = [os.path.basename(p) for p in glob.glob(f'output/pareto/{dataset}/*')]
    print(f'{dataset}')
    if (set(files) != set(methods)):
        print('some files are missing')
    fig, ax = plt.subplots(figsize=(8, 4))
    color = colors()
    metricTable = {}
    ruler = 1.02
    annotate = dict(xyoffset=[ruler, .98], angle=80, ystep=-0.01, anglestep=-5)
    for method in reversed(methods):
        try:
            Y, metadata = getData(f'output/test/{dataset}/{method}/test.csv')
        except FileNotFoundError:
            continue
        pareto = Pareto(Y, metadata, label=method)
        pareto.plot(ax, c=next(color))
        metrics = dict(
            method=method,
            size=pareto.size(),
            delta=pareto.delta(),
            M3=pareto.M3(),
            hmean=pareto.hmean(),
            HV=pareto.HV()
        )
        metricTable = {k: metricTable.get(k, []) + [metrics[k]] for k in metrics}

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
    plt.savefig(f'output/plot/{dataset}/pareto.pdf')
    # save tex table
    with open(f'output/tex/{dataset}/table.tex', 'w') as f:
        f.write(latexMetricTable(metricTable))


def bootstrap(datasets, methods, metrics):
    df = pd.DataFrame()
    for dataset in datasets:
        for method in methods:
            Y, metadata = getData(f'output/test/{dataset}/{method}/test.csv')
            with open(f'output/pareto/{dataset}/{method}/fronts.json', 'r') as f:
                effSets = json.load(f)
            # match config strings from json with test performance from test.csv
            fronts = [[Y[np.where(metadata == c)[0][0]] for c in s] for s in effSets]
            # Extracting test Pareto fronts
            paretos = [Pareto(np.array(f)) for f in fronts]
            # compute metrics
            dl = pd.DataFrame([[getattr(p, m)()for m in metrics] for p in paretos],
                              columns=metrics)
            dl['method'] = method
            dl['dataset'] = dataset
            df = df.append(dl)
    df = df.replace([np.inf, -np.inf], np.nan)

    # plot distributions
    for dataset in datasets:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax = sns.scatterplot(data=df.loc[df['dataset'] == dataset].iloc[::-1],
                             x='delta', y='HV', style='method', hue='method',
                             alpha=.5, s=3, linewidth=0, ax=ax)
        ax.grid()
        fig.tight_layout()
        plt.savefig(f'output/plot/{dataset}/scatter.pdf')
        print(f'DONE: {dataset}')
    return df


def fmt(vp):
    v = vp[0]
    p = vp[1]
    s = f'{v:.3f}'
    if ((vp.name[1] == 'HV' and v > 0.) or
        (vp.name[1] == 'delta' and v < 0.)):
        s = f'\\bft{{{s}}}'
    if p < 0.001:
        s += '\\sstr'
    elif p < 0.05:
        s += '\\str'
    return s


def to_latex(df):
    rename = {'delta': r'$\Delta$'}
    pre = r'\begin{tabular}{'+'l'*len(df.index.names)+'rr'*(len(df.columns))+'@{}}\n\\toprule\n'
    head = [[]]*2
    cells = [[] for _ in range(len(df))]
    post = '\\bottomrule\n\\end{tabular}\n'

    # header
    head[0] = ['']*(df.index.nlevels-1) + [df.columns.name] + list(df.columns)
    head[1] = list(df.index.names)

    # row indices
    for lvl in range(df.index.nlevels):
        i = 0
        while i < len(df.index):
            nrow = 1
            rname = df.index[i][lvl]
            rlen = len(df.index.names) + len(df.columns) - lvl
            cline = f'\\cline{{1-{rlen}}}\n'
            if i == 0:
                cline = ''
            for r in df.index[i+1:]:
                if rname == r[lvl]:
                    nrow += 1
                else:
                    break
            rname = rename.get(rname, rname)
            if nrow == 1:
                cells[i] += [rname]
            else:
                cells[i] += [f'{cline}\\multirow{{{nrow:d}}}{{*}}{{{rname}}}']
                for j in range(i+1, i+nrow):
                    cells[j] += ['']
            i += nrow

    # cells
    for i, (idx, row) in enumerate(df.iterrows()):
        cells[i] += [hlfmt(v, row.name[1], list(row)) for v in row]

    hlines = ['&'.join([f'{c:^14}' for c in rc]) for rc in head]
    clines = ['&'.join([f'{c:^14}' for c in rc]) for rc in cells]
    body = '\\\\\n'.join(hlines) + '\\\\\n\\midrule\n' + '\\\\\n'.join(clines) + '\\\\\n'
    return pre + body + post


def wilcoxon_test(x):
    return wilcoxon(x)[1]


def sign_test(x):
    # the sign test is the binominal test on the sign of the differences
    return binom_test(sum(x > 0.), n=len(x != 0.))


def statTest(df, methods, testFn=sign_test):
    # compute pairwise differences
    d = df.pivot(columns=['method', 'dataset'])
    d.columns.rename('metric', level=0, inplace=True)
    d = d.reorder_levels(['method', 'dataset', 'metric'], 'columns')
    diff = d.sub(d['mo-all'], axis='index').drop(columns=['mo-all'], level=2)
    diffa = diff.stack('dataset')
    # compute medians and p values of Wilcoxon rank-sum test
    # (test symmetric distribution of differences about 0)
    res = pd.concat({'median': diff.median(),
                     'p': diff.apply(testFn)}, 1)
    resa = pd.concat({'median': diffa.median(),
                     'p': diffa.apply(testFn)}, 1)
    res = res.append(pd.concat({'All': resa}, names=['dataset']))
    res = res.unstack('method').stack(0)
    dfPrint = res.stack('method').unstack(2).apply(fmt, 1).unstack('method')
    dfPrint.rename(index={'delta': r'$\Delta$'}, inplace=True)
    dfPrint = dfPrint[methods[:-1]]
    with open(f'output/tex/stats.tex', 'w') as f:
        f.write(dfPrint.to_latex(escape=False, multirow=True, column_format='ll'+'r'*len(dfPrint.columns)))
    # just the median metrics
    meds = d.median().unstack('method')[methods]
    tex = to_latex(meds)
    with open(f'output/tex/dataset.tex', 'w') as f:
        f.write(tex)
    return res


def main():
    os.chdir(os.path.dirname(sys.argv[0]))
    datasetDirs = glob.glob('output/pareto/*')
    if not len(datasetDirs):
        sys.exit('No files found in output/pareto/. Nothing to be done.')
    datasets = [os.path.basename(p) for p in sorted(datasetDirs)]
    for dataset in datasets:
        os.makedirs(f'output/plot/{dataset}/', exist_ok=True)
        os.makedirs(f'output/tex/{dataset}/', exist_ok=True)
    methods = ['mo-ects',
               'mo-ecdire',
               'mo-srcf',
               'mo-relclass',
               'so-all',
               'mo-all']
    metrics = ['HV', 'delta']
    df = bootstrap(datasets, methods, metrics)
    statTest(df, methods)
    for dataset in datasets:
        processData(dataset, methods)


if __name__ == '__main__':
    main()
