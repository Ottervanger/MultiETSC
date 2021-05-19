#!/usr/bin/env python
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import re
import glob
import itertools
import subprocess
import json
import seaborn as sns
import random
import pickle
from scipy.stats import wilcoxon, binom_test, friedmanchisquare
from cdgraph import cdGraph


def colors(alpha=None):
    while True:
        for c in ['#1f77b4',  # blue
                  '#ff7f0e',  # orange
                  '#17becf',  # cyan
                  '#d62728',  # red
                  '#9467bd',  # purple
                  '#8c564b',  # brown
                  '#e377c2',  # pink
                  '#7f7f7f',  # grey
                  '#2ca02c',  # green
                  '#bcbd22',  # olive
                  ]:
            yield c if not alpha else c + alpha


UCR_TYPE_MAP = {
    'ACSF1': 'DEVICE',
    'Adiac': 'IMAGE',
    'AllGestureWiimoteX': 'MOTION',
    'AllGestureWiimoteY': 'MOTION',
    'AllGestureWiimoteZ': 'MOTION',
    'ArrowHead': 'IMAGE',
    'Beef': 'SPECTRO',
    'BeetleFly': 'IMAGE',
    'BirdChicken': 'IMAGE',
    'BME': 'SIMULATED',
    'Car': 'SENSOR',
    'CBF': 'SIMULATED',
    'Chinatown': 'TRAFFIC',
    'ChlorineConcentration': 'SIMULATED',
    'CinCECGTorso': 'ECG',
    'Coffee': 'SPECTRO',
    'Computers': 'DEVICE',
    'Cricket': 'HAR',
    'CricketX': 'MOTION',
    'CricketY': 'MOTION',
    'CricketZ': 'MOTION',
    'Crop': 'IMAGE',
    'DiatomSizeReduction': 'IMAGE',
    'DistalPhalanxOutlineAgeGroup': 'IMAGE',
    'DistalPhalanxOutlineCorrect': 'IMAGE',
    'DistalPhalanxTW': 'IMAGE',
    'DodgerLoopDay': 'SENSOR',
    'DodgerLoopGame': 'SENSOR',
    'DodgerLoopWeekend': 'SENSOR',
    'DuckDuckGeese': 'AUDIO',
    'DucksAndGeese': 'AUDIO',
    'Earthquakes': 'SENSOR',
    'ECG200': 'ECG',
    'ECG5000': 'ECG',
    'ECGFiveDays': 'ECG',
    'EigenWorms': 'MOTION',
    'ElectricDeviceDetection': 'DEVICE',
    'ElectricDevices': 'DEVICE',
    'EOGHorizontalSignal': 'EOG',
    'EOGVerticalSignal': 'EOG',
    'Epilepsy': 'HAR',
    'ERing': 'HAR',
    'EthanolConcentration': 'OTHER',
    'EthanolLevel': 'SPECTRO',
    'EyesOpenShut': 'EEG',
    'FaceAll': 'IMAGE',
    'FaceDetection': 'EEG',
    'FaceFour': 'IMAGE',
    'FacesUCR': 'IMAGE',
    'FiftyWords': 'IMAGE',
    'FingerMovements': 'EEG',
    'Fish': 'IMAGE',
    'FordA': 'SENSOR',
    'FordB': 'SENSOR',
    'FreezerRegularTrain': 'SENSOR',
    'FreezerSmallTrain': 'SENSOR',
    'FruitFlies': 'AUDIO',
    'Fungi': 'OTHER',
    'GestureMidAirD1': 'MOTION',
    'GestureMidAirD2': 'MOTION',
    'GestureMidAirD3': 'MOTION',
    'GesturePebbleZ1': 'MOTION',
    'GesturePebbleZ2': 'MOTION',
    'GunPoint': 'MOTION',
    'GunPointAgeSpan': 'MOTION',
    'GunPointMaleVersusFemale': 'MOTION',
    'GunPointOldVersusYoung': 'MOTION',
    'Ham': 'SPECTRO',
    'HandMovementDirection': 'EEG',
    'HandOutlines': 'IMAGE',
    'Handwriting': 'HAR',
    'Haptics': 'MOTION',
    'Heartbeat': 'AUDIO',
    'Herring': 'IMAGE',
    'HouseTwenty': 'DEVICE',
    'InlineSkate': 'MOTION',
    'InsectEPGRegularTrain': 'EPG',
    'InsectEPGSmallTrain': 'EPG',
    'InsectSound': 'AUDIO',
    'InsectWingbeat': 'AUDIO',
    'ItalyPowerDemand': 'SENSOR',
    'JapaneseVowels': 'AUDIO',
    'LargeKitchenAppliances': 'DEVICE',
    'Libras': 'HAR',
    'Lightning2': 'SENSOR',
    'Lightning7': 'SENSOR',
    'LSST': 'OTHER',
    'Mallat': 'SIMULATED',
    'Meat': 'SPECTRO',
    'MedicalImages': 'IMAGE',
    'MelbournePedestrian': 'TRAFFIC',
    'MiddlePhalanxOutlineAgeGroup': 'IMAGE',
    'MiddlePhalanxOutlineCorrect': 'IMAGE',
    'MiddlePhalanxTW': 'IMAGE',
    'MixedShapes': 'IMAGE',
    'MixedShapesRegularTrain': 'IMAGE',
    'MixedShapesSmallTrain': 'IMAGE',
    'MosquitoSound': 'AUDIO',
    'MoteStrain': 'SENSOR',
    'MotorImagery': 'EEG',
    'NATOPS': 'HAR',
    'NonInvasiveFetalECGThorax1': 'ECG',
    'NonInvasiveFetalECGThorax2': 'ECG',
    'OliveOil': 'SPECTRO',
    'OSULeaf': 'IMAGE',
    'PenDigits': 'MOTION',
    'PhalangesOutlinesCorrect': 'IMAGE',
    'PickupGestureWiimoteZ': 'SENSOR',
    'PigAirwayPressure': 'HEMODYNAMICS',
    'PigArtPressure': 'HEMODYNAMICS',
    'PigCVP': 'HEMODYNAMICS',
    'PLAID': 'DEVICE',
    'Plane': 'SENSOR',
    'PowerCons': 'DEVICE',
    'ProximalPhalanxOutlineAgeGroup': 'IMAGE',
    'ProximalPhalanxOutlineCorrect': 'IMAGE',
    'ProximalPhalanxTW': 'IMAGE',
    'RacketSports': 'HAR',
    'RefrigerationDevices': 'DEVICE',
    'RightWhaleCalls': 'AUDIO',
    'Rock': 'SPECTRO',
    'ScreenType': 'DEVICE',
    'SelfRegulationSCP1': 'EEG',
    'SelfRegulationSCP2': 'EEG',
    'SemgHandGenderCh2': 'SPECTRO',
    'SemgHandMovementCh2': 'SPECTRO',
    'SemgHandSubjectCh2': 'SPECTRO',
    'ShakeGestureWiimoteZ': 'SENSOR',
    'ShapeletSim': 'SIMULATED',
    'ShapesAll': 'IMAGE',
    'SmallKitchenAppliances': 'DEVICE',
    'SmoothSubspace': 'SIMULATED',
    'SonyAIBORobotSurface1': 'SENSOR',
    'SonyAIBORobotSurface2': 'SENSOR',
    'SpokenArabicDigits': 'SPEECH',
    'StandWalkJump': 'ECG',
    'StarLightCurves': 'SENSOR',
    'Strawberry': 'SPECTRO',
    'SwedishLeaf': 'IMAGE',
    'Symbols': 'IMAGE',
    'SyntheticControl': 'SIMULATED',
    'Tiselac': 'IMAGE',
    'ToeSegmentation1': 'MOTION',
    'ToeSegmentation2': 'MOTION',
    'Trace': 'SENSOR',
    'TwoLeadECG': 'ECG',
    'TwoPatterns': 'SIMULATED',
    'UMD': 'SIMULATED',
    'UrbanSound': 'AUDIO',
    'UWaveGestureLibrary': 'HAR',
    'UWaveGestureLibraryAll': 'MOTION',
    'UWaveGestureLibraryX': 'MOTION',
    'UWaveGestureLibraryY': 'MOTION',
    'UWaveGestureLibraryZ': 'MOTION',
    'Wafer': 'SENSOR',
    'Wine': 'SPECTRO',
    'WordSynonyms': 'IMAGE',
    'Worms': 'MOTION',
    'WormsTwoClass': 'MOTION',
    'Yoga': 'IMAGE'
}

UCR_TYPE_NAMES = {
    'DEVICE': 'Electric Devices',
    'AUDIO': 'Audio',
    'ECG': 'Electro Cardiogram',
    'EEG': 'Electro Encephalogram',
    'HAR': 'Motion Capture',
    'HEMODYNAMICS': 'HEMODYNAMICS',
    'IMAGE': 'Image Outlines',
}

ALG_NAMES = {
    "EDSC/bin/edsc": "EDSC",
    "ECDIRE/run": "ECDIRE",
    "SR-CF/run": "SR-CF",
    "ECTS/bin/ects": "ECTS",
    "TEASER/ECECrun": "ECEC",
    "RelClass/run": "RelClass",
    "EARLIEST/run.py": "EARLIEST",
    "TEASER/TEASERrun": "TEASER",
    "fixed/run.py": "Fixed"
}


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

    def plot(self, ax, c, annotate=None, points=True, **args):
        # stepped line
        ax.step(np.hstack((0, self.P[:, 0], 1)),
                np.hstack((1, self.P[:, 1], 0)),
                where='post', label=self.label, c=c, **args)
        # points
        if points:
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
        if not len(self.P[np.all(self.P != 1., 1)]):
            return 1.
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
        if not self._size:
            return np.nan
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
    pat = re.compile('^.*\[(.*), (.*)\], 0, (.*)$', re.MULTILINE)
    with open(csvPath) as f:
        data = np.array(re.findall(pat, f.read()))
    if not len(data):
        return np.empty((0, 2)), np.empty((0,))
    df = pd.DataFrame(data, columns=['earliness', 'accuracy', 'configuration'])
    df[['earliness', 'accuracy']] = df[['earliness', 'accuracy']].astype(float)
    # compute evaluation means by configuration
    df = df.groupby('configuration', as_index=False).mean()
    Y = np.array(df[['earliness', 'accuracy']])
    conf = np.array(df['configuration'])
    return Y, conf


def best(metric, val, l):
    if metric == 'method':
        return False
    if metric in ['delta', 'hmean']:
        return val == min(l)
    return val == max(l)


def formatCell(x, metric=None, spec=None):
    formats = dict(int=     '{:8d}'+' '*4,
                   float=   '{:12.5f}',
                   float64= '{:12.5f}',
                   str=     ' {:>11s}')
    if 'str' not in x.__class__.__name__:
        if metric == 'HV' and x == 0:
            return formats['str'].format('')
        if metric == 'hmean' and x > (1 - 1e-14):
            return formats['str'].format('')
        if np.isnan(x):
            return formats['str'].format('')
    if spec and spec in ['b', 'bf', 'bold']:
        s = formats[x.__class__.__name__].format(x).strip()
        return formats['str'].format(f'\\bft{{{s}}}')
    return formats[x.__class__.__name__].format(x)


def printTable(d):
    for k, v in d.items():
        print('{:>10s}'.format(k)+''.join(formatCell(vi) for vi in v))


def hlfmt(val, metric=None, l=None):
    if metric and l and best(metric, val, l):
        return formatCell(val, metric=metric, spec='bf')
    return formatCell(val, metric=metric)


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


def metName(m):
    return {
     'mo-fixed': 'Fixed',
     'mo-srcf': 'SR-CF',
     'mo-relclass': 'RelClass',
     'mo-earliest-long': 'EARLIEST 10m',
     'so-all': 'SO-All',
     'mo-all': 'MultiETSC'
    }.get(m, m[3:].upper())


def combinedRunsPlot(dataset, methods):
    files = [os.path.basename(p) for p in glob.glob(f'output/test/{dataset}/*')]
    if not (set(methods) <= set(files)):
        print('some files are missing')
    fig, ax = plt.subplots(figsize=(8, 4))
    color = colors('aa')
    metricTable = {}
    ruler = 1.02
    annotate = dict(xyoffset=[ruler, .98], angle=80, ystep=-0.01, anglestep=-5)

    for method in reversed(methods):
        try:
            Y, metadata = getData(f'output/test/{dataset}/{method}/test.csv')
        except FileNotFoundError:
            continue
        pareto = Pareto(Y, metadata, label=metName(method))
        pareto.plot(ax, c=next(color))
        metrics = dict(
            method=metName(method),
            size=pareto.size(),
            delta=pareto.delta(),
            M3=pareto.M3(),
            hmean=pareto.hmean(),
            HV=pareto.HV()
        )
        metricTable = {k: metricTable.get(k, []) + [metrics[k]] for k in metrics}

    # print metrics to terminal
    # print(f'{dataset}')
    # printTable(metricTable)
    # print()
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
    fig.clear()
    plt.close(fig)
    # save tex table
    with open(f'output/tex/{dataset}/table.tex', 'w') as f:
        f.write(latexMetricTable(metricTable))


def randomSample(dataset, methods):
    fig, ax = plt.subplots(figsize=(8, 4))
    color = colors('aa')
    metricTable = {}
    ruler = 1.02
    annotate = dict(xyoffset=[ruler, .98], angle=80, ystep=-0.01, anglestep=-5)

    for method in reversed(methods):
        try:
            Y, metadata = getData(f'output/test/{dataset}/{method}/test.csv')
            with open(f'output/test/{dataset}/{method}/fronts.json', 'r') as f:
                effSets = json.load(f)
        except FileNotFoundError:
            continue
        try:
            # match config strings from json with test performance from test.csv
            s = random.choice(effSets)
            f = [Y[np.where(metadata == c)[0][0]] for c in s]
        except IndexError:
            continue
        pareto = Pareto(np.array(f+[np.ones(2)]), np.array(s+['default']), label=metName(method))
        ls = '-' if method[-3:] == 'all' else '--'
        lw = 2 if method[-3:] == 'all' else 1.5
        pareto.plot(ax, c=next(color), ls=ls, linewidth=lw)
        metrics = dict(
            method=metName(method),
            size=pareto.size(),
            delta=pareto.delta(),
            M3=pareto.M3(),
            hmean=pareto.hmean(),
            HV=pareto.HV()
        )
        metricTable = {k: metricTable.get(k, []) + [metrics[k]] for k in metrics}

    # print metrics to terminal
    # print(f'{dataset}')
    # printTable(metricTable)
    # print()
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
    fig.clear()
    plt.close(fig)
    # save tex table
    with open(f'output/tex/{dataset}/table.tex', 'w') as f:
        f.write(latexMetricTable(metricTable))


def rmDominated(Y):
    inp = np.ones(len(Y), dtype="bool")
    for i in range(len(Y)):
        if inp[i]:
            inp = inp & np.any(Y < Y[i], 1)
            inp[i] = True
    return Y[inp]


def plotFront(ax, f, **args):
    Y = rmDominated(f[np.argsort(f[:,0])])
    ax.step(np.hstack((0, Y[:, 0], 1)),
            np.hstack((1, Y[:, 1], 0)),
            where='post', **args)
    ref = np.array([1., 1.])
    s = 0
    for y in Y:
        s += np.prod(ref-y)
        ref[1] = y[1]
    return s, len(Y)


def methodPlot(dataset, methods, idx=0, combinded=False):
    fig, axs = plt.subplots(ncols=len(methods),figsize=(10, 3.5), sharex='row', sharey='row')
    color = colors()
    colMap = {
        "EDSC/bin/edsc": next(color),
        "ECDIRE/run": next(color),
        "SR-CF/run": next(color),
        "ECTS/bin/ects": next(color),
        "TEASER/ECECrun": next(color),
        "RelClass/run": next(color),
        "EARLIEST/run.py": next(color),
        "TEASER/TEASERrun": next(color),
        "fixed/run.py": next(color)
    }
    limitMap = {
        'CBF': (1., .8),
        'ECG200': (1., .4),
        'GunPoint': (.8, .6),
        'OliveOil': (1., .8),
        'Wafer': (.6, .125),
    }
    by_label = {}

    for i, (method, ax) in enumerate(zip(methods, axs)):
        if method == 'mo-man-sep':
            method = 'mo-man'
            separate = True
        else:
            separate = False
        Y, metadata = getData(f'output/test/{dataset}/{method}/test.csv')
        if combinded:
            s = np.array(list(map(lambda x: cleanConfStr(x).split()[1], metadata)))
            f = Y
        else:
            # match config strings from json with test performance from test.csv
            with open(f'output/test/{dataset}/{method}/fronts.json', 'r') as f:
                effSets = json.load(f)
            idx = idx % len(effSets)
            s = np.array(list(map(lambda x: cleanConfStr(x).split()[1], effSets[idx])))
            f = np.array([Y[np.where(metadata == c)[0][0]] for c in effSets[idx]])

        y = 1 if not dataset in limitMap else limitMap[dataset][1]
        y += .01

        if separate:
            # plot separate Pareto fronts per algorithm
            hv, size = 0, 0
            for alg in np.unique(s):
                # breakpoint()
                tmphv, tmpsize = plotFront(ax, f[s == alg], c=colMap[alg])
                if tmphv > hv:
                    hv, size = tmphv, tmpsize

        else:
            # plot the Pareto front
            hv, size = plotFront(ax, f, c='#000000aa')
            ax.text(0, y, f"{size} solutions\nHV = {hv:.2f}", size=11)

        # plot the individual confs
        for (x, y), alg  in zip(f, s):
            (marker, s) = ('o', 5) if alg != "fixed/run.py" else ('+', 100)
            ax.scatter(x, y, marker=marker, c=colMap[alg], label=ALG_NAMES[alg], s=s)

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        if dataset in limitMap:
            ax.set_xlim([0, limitMap[dataset][0]])
            ax.set_ylim([0, limitMap[dataset][1]])
        ax.grid()
        ax.title.set_text(chr(ord('A') + i)+"\n\n")

        handles, labels = ax.get_legend_handles_labels()
        by_label.update(dict(zip(labels, handles)))

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    
    fig.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, 0.1), ncol=9, handlelength=1)
    # plt.title(f'Earliness-Accuracy tradeoff\n{dataset}\n\n\n')
    plt.xlabel("Earliness")
    plt.ylabel("Error rate")

    fig.tight_layout(rect=(-.03,0,1.02,1.0))
    # save plot
    plt.savefig(f'output/plot/{dataset}/method-plot.pdf')
    fig.clear()
    plt.close(fig)


def bootstrap(datasets, methods, metrics, df, algCounts):
    print('Bootstrapping')
    if df is None:
        df = pd.DataFrame()
    algCountsValid = algCounts[0]
    algCountsTest = algCounts[1]
    for dataset in datasets:
        # if dataset in ['Crop', 'ElectricDevices', 'FordB', 'FordA', 'InsectWingbeatSound']:
        #     continue
        if 'dataset' in df and dataset in df['dataset'].values:
            continue
        for method in methods:
            try:
                Y, metadata = getData(f'output/test/{dataset}/{method}/test.csv')
                with open(f'output/test/{dataset}/{method}/fronts.json', 'r') as f:
                    effSets = json.load(f)
            except FileNotFoundError:
                continue
            try:
                # match config strings from json with test performance from test.csv
                fronts = [[Y[np.where(metadata == c)[0][0]] for c in s] for s in effSets]
            except IndexError:
                continue
            # Extracting test Pareto fronts
            paretos = [Pareto(np.array(f+[np.ones(2)]), np.array(s+['alg ref'])) for f, s in zip(fronts, effSets)]

            if method == 'mo-all':
                for pareto in paretos:
                    for conf in pareto.metadata:
                        alg = conf.split()[1]
                        algCountsTest[alg] = algCountsTest.get(alg, 0) + 1
                for effSet in effSets:
                    for conf in effSet:
                        alg = cleanConfStr(conf).split()[1]
                        algCountsValid[alg] = algCountsValid.get(alg, 0) + 1
            # compute metrics
            dl = pd.DataFrame([[getattr(p, m)()for m in metrics] for p in paretos],
                              columns=metrics)
            dl['method'] = metName(method)
            dl['dataset'] = dataset
            df = df.append(dl)
        print(f'DONE: {dataset}')
    print('All Done')
    df = df.replace([np.inf, -np.inf], np.nan)
    return df, (algCountsValid, algCountsTest)


def plotAlgCounts(algCountsValid, algCountsTest):
    barWidth = 1./3
    xlabels = np.array(list(algCountsValid.keys()))

    bars1 = np.array([algCountsValid[k] for k in xlabels])
    bars1 = bars1 / np.sum(bars1)
    bars2 = np.array([algCountsTest[k] for k in xlabels])
    bars2 = bars2 / np.sum(bars2)
    order = np.argsort(bars1)[::-1]

    xlabels = np.array([ALG_NAMES[k] for k in xlabels])

    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(r1, bars1[order], width=barWidth, edgecolor='white', label='Validation')
    ax.bar(r2, bars2[order], width=barWidth, edgecolor='white', label='Test')
    plt.xticks([r + barWidth/2 for r in range(len(bars1))], xlabels[order])
    plt.xticks(rotation=45)
    plt.xlabel('Algorithm')
    plt.ylabel('Proportion')
    ax.set_title('Selected algorithms\nas proportion of non-dominated solutions')
    ax.legend()
    fig.tight_layout()
    plt.savefig('output/tex/counts.pdf')
    fig.clear()
    plt.close(fig)


def plotDistributions(df, datasets, blines=None):
    for dataset in datasets:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax = sns.scatterplot(data=df.loc[df['dataset'] == dataset].iloc[::-1],
                             x='delta', y='HV', hue='method',
                             alpha=.5, s=3, linewidth=0, ax=ax)
        if blines is not None:
            ax.scatter([blines[dataset].delta()], [blines[dataset].HV()],
                       c='k', marker='*', label=blines[dataset].label)
            ax.axhline(blines[dataset].HV(), c='#00000099', lw=1)
        ax.legend()
        ax.grid()
        fig.tight_layout()
        plt.savefig(f'output/plot/{dataset}/scatter.pdf')
        fig.clear()
        plt.close(fig)


def violins(df, metric):
    plt.clf()
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.violinplot(x="method", y=metric, data=df, bw=.1,
                   cut=0, width=1, linewidth=1, ax=ax)
    sns.despine(ax=ax, left=True, bottom=True)
    plt.xticks(rotation=-30)
    fig.tight_layout()
    plt.savefig(f'output/tex/dist-{metric}.pdf')
    fig.clear()
    plt.close(fig)


def fmt(vp, metric=None):
    v = vp[0]
    p = vp[1]
    s = f'{v:.5f}'
    if ((metric == 'HV' and v > 0.) or
            (metric == 'delta' and v < 0.)):
        s = f'\\bft{{{s}}}'
    if p < 0.001:
        s += '\\ssignif'
    elif p < 0.05:
        s += '\\signif'
    return s


def to_latex(df, metric):
    rename = {'delta': r'$\Delta$'}
    pre = r'\begin{tabular}{'+'l'*len(df.index.names)+'r'*(len(df.columns))+'@{}}\n\\toprule\n'
    head = [[]]*2
    cells = [[] for _ in range(len(df))]
    post = '\\bottomrule\n\\end{tabular}\n'

    # header
    head[0] = ['']*(df.index.nlevels-1) + [df.columns.name] + list(df.columns)
    head[1] = list(df.index.names)

    # row indices
    if df.index.nlevels > 1:
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
    else:
        for i, rname in enumerate(df.index):
            cells[i] += [rname]


    # cells
    for i, (idx, row) in enumerate(df.iterrows()):
        cells[i] += [hlfmt(v, metric, list(row)) for v in row]

    hlines = ['&'.join([f'{c:^14}' for c in rc]) for rc in head]
    clines = ['&'.join([f'{c:^14}' for c in rc]) for rc in cells]
    body = '\\\\\n'.join(hlines) + '\\\\\n\\midrule\n' + '\\\\\n'.join(clines) + '\\\\\n'
    return pre + body + post


def wilcoxon_test(x):
    return wilcoxon(x)[1]


def sign_test(x):
    # the sign test is the binominal test on the sign of the differences
    return binom_test(sum(x > 0.), n=len(x != 0.))


def nemenyiCD(k, n):
    """
    Returns critical difference for Nemenyi test for alpha = 0.05
    for k groups with n paired observations.
    """
    q = [0, 0, 1.959964, 2.343701, 2.569032, 2.727774, 2.849705, 2.94832,
         3.030879, 3.101730, 3.163684, 3.218654, 3.268004, 3.312739, 3.353618,
         3.39123, 3.426041, 3.458425, 3.488685, 3.517073, 3.543799]
    cd = q[k] * (k * (k + 1) / (6.0 * n)) ** 0.5
    return cd


def cdTestAndGraph(df, methods, metric):
    d = df.pivot(columns=['method', 'dataset'])
    obs = d[metric].stack('dataset')
    ranks = obs.rank(axis=1, na_option='bottom', ascending=(metric != 'HV'))
    _, p = friedmanchisquare(*(obs[c] for c in methods))
    print(f'friedmanchisquare: p = {p:f}')

    # Nemenyi test critical difference plot
    avRanks = ranks.mean()
    cd = nemenyiCD(len(avRanks), len(ranks))
    cdGraph(avRanks, names=avRanks.index, cd=cd,
            filename=f'output/tex/difference-{metric}.pdf')


def percentWins(df, methods, metric):
    df['type'] = df['dataset'].map(UCR_TYPE_MAP)
    d = df.pivot_table(columns=['method'], index=[df.index, df['dataset'], df['type']])[metric]
    dbest = (d.rank(axis=1, method='min', na_option='bottom', ascending=(metric != 'HV')) == 1)
    pbest = dbest.groupby('type').mean() * 100
    pbest['Counts'] = dbest.groupby('type').size()
    ptotal = dbest.mean() * 100
    ptotal.name = 'Overall'
    pcounts = dbest.sum()
    pcounts['Counts'] = len(dbest)
    pcounts.name = 'Counts'
    pbest = pbest.append([ptotal, pcounts])
    with open(f'output/tex/perc-best-{metric}.tex', 'w') as f:
        f.write(pbest.to_latex(
            escape=False,
            column_format='l'+'r'*len(pbest.columns),
            float_format="{:0.2f}".format))
    pBetterSO = (d[metName('mo-all')] > d[metName('so-all')]).mean() * 100
    print(f'MO-All     > SO-All:     {pBetterSO:.3f}')
    df.drop('type', axis=1, inplace=True)


def makeBigTables(df, methods, metric, testFn=sign_test):
    d = df.pivot(columns=['method', 'dataset'])[metric]
    # just the median metrics
    meds = d.median().unstack('method')[methods]
    with open(f'output/tex/dataset-{metric}.tex', 'w') as f:
        f.write(to_latex(meds, metric))
     
    # compute pairwise differences
    diff = d.sub(d[metName('mo-all')], axis='index').drop(columns=[metName('mo-all')], level=0)
    diffa = diff.stack('dataset')
    # compute medians and p values of Wilcoxon rank-sum test
    # (test symmetric distribution of differences about 0)
    res = pd.concat({'median': diff.median(),
                     'p': diff.apply(testFn)}, 1)
    resa = pd.concat({'median': diffa.median(),
                     'p': diffa.apply(testFn)}, 1)

    res = res.append(pd.concat({'All': resa}, names=['dataset']).reorder_levels([1,0]))
    res = res.unstack('method').stack(0)

    # dfPrint stores formated strings
    dfPrint = res.stack('method').unstack(1).apply(fmt, 1, metric=metric).unstack('method')
    dfPrint.rename(index={'delta': r'$\Delta$'}, inplace=True)
    # move 'All' to top
    dfPrint['order'] = range(1, len(dfPrint)+1)
    dfPrint.loc['All', 'order'] = 0
    dfPrint = dfPrint.sort_values("order").drop('order', axis=1)
    dfPrint = dfPrint[methods[:-1]]
    with open(f'output/tex/stats-{metric}.tex', 'w') as f:
        f.write(dfPrint.to_latex(escape=False, column_format='l'+'r'*len(dfPrint.columns)))



    return res


def nrSearchedConfs(datasets, methods):
    """
    Result:
    {
      "mo-fixed": 99.98846828943869,
      "mo-ects": 198.14141613250422,
      "mo-edsc": 706.7960821265945,
      "mo-ecdire": 45.42471811586604,
      "mo-srcf": 50.35429728137417,
      "mo-relclass": 79.69192723789588,
      "mo-teaser": 61.274130083423444,
      "mo-ecec": 41.297369409535655,
      "mo-earliest": 55.78036489339299,
      "so-all": 63.830689655172414,
      "mo-all": 109.63705318710898
    }
    """

    print('Calculating average number of evaluated configurations')
    pat = re.compile('^(?:\"(.*?)\",\"(.*?)\",){2}.*$', re.MULTILINE)
    nconfs = {
        'mo-ects': 200,
        'mo-relclass': 606,
        'mo-ecdire': 16000,
        'mo-srcf': 1184000,
        'mo-earliest': 16800000,
        'mo-teaser': 90720,
        'mo-ecec': 252000,
        'mo-edsc': 102400,
        'mo-fixed': 100,
        'so-all': 18446026,
        'mo-all': 18446026
    }
    avgNConfs = {}
    for method in methods:
        print(f'Method: {method}')
        avgNConfs[method] = 0
        count = 0
        for dataset in datasets:
            trajFileGlob = f'output/configurator/{dataset}/{method}/*/detailed-traj-run-*.csv'
            if method == 'so-all':
                trajFileGlob = f'output/configurator/{dataset}/{method}/*/runhistory.json'
            trajFiles = glob.glob(trajFileGlob)
            for traj in trajFiles:
                count += 1
                if method == 'so-all':
                    with open(traj) as f:
                        avgNConfs[method] += len(json.load(f)['configs'])
                    continue
                with open(traj) as f:
                    data = np.array(re.findall(pat, f.read()))[1:].astype(float)
                samp = data[np.argmax(data[:,1])]
                if samp[0] == 0:
                    # all confs timed out
                    avgNConfs[method] += min(7200 / 180, nconfs[method])
                    continue
                avgNConfs[method] += min(7200 * samp[1] / samp[0], nconfs[method])
        avgNConfs[method] /= count
        print(avgNConfs[method])
    print(json.dumps(avgNConfs, indent=2))
                    


def main():
    datacache = 'output/datacache.pkl'
    os.chdir(os.path.dirname(sys.argv[0]))
    datasetDirs = glob.glob('output/test/*')
    if not len(datasetDirs):
        sys.exit('No files found in output/test/. Nothing to be done.')
    datasets = [os.path.basename(p) for p in sorted(datasetDirs)]
    datasets = [
        "CBF",
        "ECG200",
        "GunPoint",
        "OliveOil",
        "SyntheticControl",
        "TwoPatterns",
        "Wafer",
    ]
    methods = ['mo-fixed',
               'mo-ects',
               'mo-edsc',
               'mo-ecdire',
               'mo-srcf',
               'mo-relclass',
               'mo-teaser',
               'mo-ecec',
               'mo-earliest',
               'so-all',
               'mo-all',
               'mo-lit']
    filterDatasets = []
    for dataset in datasets:
        for method in methods:
            if os.path.isdir(f'output/plot/{dataset}/'):
                continue
            cFile = f'output/test/{dataset}/{method}/confs.txt'
            tFile = f'output/test/{dataset}/{method}/test.csv'
            try:
                nConf = int(subprocess.check_output(['wc', '-l', f'{cFile}'], stderr=subprocess.DEVNULL).split()[0])
                nTest = int(subprocess.check_output(['wc', '-l', f'{tFile}'], stderr=subprocess.DEVNULL).split()[0])
            except (ValueError, subprocess.CalledProcessError) as e:
                print(f'Skipping {dataset}: test not complete.')
                break
            if nConf != nTest:
                print(f'Skipping {dataset}: test not complete.')
                break
        else:
            filterDatasets += [dataset]
            os.makedirs(f'output/plot/{dataset}/', exist_ok=True)
            os.makedirs(f'output/tex/{dataset}/', exist_ok=True)
    datasets = filterDatasets
    metrics = ['HV', 'delta', 'hmean', 'size']
    try:
        with open(datacache, 'rb') as f:
            df, algCounts = pickle.load(f)
    except FileNotFoundError:
        df = pd.DataFrame()
        algCounts = (dict(), dict())
    df, algCounts = bootstrap(datasets, methods, metrics, df, algCounts)
    plotAlgCounts(*algCounts)
    with open(datacache, 'wb') as f:
        pickle.dump((df, algCounts), f)
    methodnames = [metName(m) for m in methods]

    print(methodnames)
    methodPlot('GunPoint', ['mo-man-sep','mo-man','mo-lit','mo-all'], idx=1, combinded=True)
    makeBigTables(df, ['LIT','MultiETSC'], 'HV')
    return
    # Scatter plots
    # plotDistributions(df, datasets)
    random.seed(3)
    # nrSearchedConfs(datasets, methods)
    for dataset in datasets:
        # combinedRunsPlot(dataset, methods)
        # randomSample(dataset, methods)
        pass

    for metric in metrics:
        # CD graphs for desired metrics
        cdTestAndGraph(df, methodnames, metric)
        # median table; difference table
        # makeBigTables(df, methodnames, metric)
        # violin plots
        # violins(df, metric)
        # table with percentages where each method performed best
        percentWins(df, methodnames, metric)


if __name__ == '__main__':
    main()
