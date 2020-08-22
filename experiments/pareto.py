#!/usr/bin/env python
import numpy as np
import pandas as pd
import os
import sys
import re
import glob
import json


def rmDominated(Y, conf):
    inp = np.ones(len(Y), dtype="bool")
    for i in range(len(Y)):
        if inp[i]:
            inp = inp & np.any(Y < Y[i], 1)
            inp[i] = True
    return Y[inp], conf[inp]


# reads data from validation files and averages runs of same configurations
def getData(csvPath):
    # read data from csv
    pat = re.compile('\[(.*), (.*)\], 0, (.*)')
    with open(csvPath) as f:
        data = np.array([re.findall(pat, s)[0] for s in f.readlines()])
    df = pd.DataFrame(data, columns=['earliness', 'accuracy', 'configuration'])
    df[['earliness', 'accuracy']] = df[['earliness', 'accuracy']].astype(float)
    # compute evaluation means by configuration
    df = df.groupby('configuration', as_index=False).mean()
    Y = np.array(df[['earliness', 'accuracy']])
    conf = np.array(df['configuration'])
    return Y, conf


def mergeFronts(fronts):
    Y = np.vstack([a[0] for a in fronts])
    conf = np.concatenate([a[1] for a in fronts])
    return rmDominated(Y, conf)


def processData(dirname, resample=None):
    condition = dirname.split('/')[-1]
    files = glob.glob(dirname+'/*')
    fronts = [rmDominated(*getData(f)) for f in files]
    pFronts = []
    if resample:
        confs = np.array([], dtype=str)
        for _ in range(resample['nSamples']):
            pSample = fronts[np.random.choice(len(fronts), resample['size'], replace=False)]
            Y, conf = mergeFronts(pSample)
            pFronts += [list(conf)]
            confs.concatenate([confs, conf])
    else:
        Y, confs = mergeFronts(fronts)
        pFronts += [list(confs)]
    with open('output/pareto/{}-confs.txt'.format(condition), 'w') as f:
        for line in confs:
            f.write('{:s}\n'.format(line))
    with open('output/pareto/{}-fronts.json'.format(condition), 'w') as f:
        json.dump(pFronts, f, indent=2)
    print('Found {:3d} nondominated configurations for {}'.format(len(Y), condition))


def main():
    np.random.seed(sys.argv[1] if len(sys.argv) > 1 else 0)
    dirs = glob.glob('output/validation/*')
    if not len(dirs):
        sys.exit('No files found in output/validation/')
    for dirname in dirs:
        processData(dirname)


if __name__ == '__main__':
    os.chdir(os.path.dirname(sys.argv[0]))
    main()
