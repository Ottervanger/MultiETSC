#!/usr/bin/env python3
import sys
import os
import shutil
import numpy as np


def usage():
    print("""
    Usage: python validationsplitter.py --folds=5 <path/to/UCR/data.tsv> [output/path]

    This script generates train-validation splits for stratified cross
    validation using the scikit-learn StratifiedKFold method.
    Additionally it produces the correct test and train lists compatible
    with paramILS/SMAC scenarios.
    """)


def argparse():
    folds = None
    datafile = None
    outputpath = None
    for arg in sys.argv[1:]:
        argname, argval = (arg+'=').split('=')[0:2]
        if 'folds' in argname:
            folds = int(argval)
            continue
        if 'seed' in argname:
            np.random.seed(int(argval))
            continue
        if not datafile:
            datafile = arg
            continue
        if not outputpath:
            outputpath = arg
            continue
        break
    if not folds:
        folds = 5
    if not datafile:
        usage()
        sys.exit('Error: no data path specified')
    return folds, datafile, outputpath


if __name__ == '__main__':
    np.random.seed(0)
    folds, datafile, outputpath = argparse()

    # read data
    y = np.genfromtxt(datafile)[:, 0]
    with open(datafile, 'r') as f:
        lines = np.array(f.readlines())
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=folds, shuffle=True)

    # prepare output dir
    if not outputpath:
        dataname = datafile.split('/')[datafile.split('/').index('UCR')+1]
        outputpath = '/tmp/paramilsdata/UCR/{}/'.format(dataname)
    if (outputpath[-1] != '/'):
        outputpath += '/'
    if os.path.isdir(outputpath):
        shutil.rmtree(outputpath)
    os.makedirs(outputpath)

    splits = []
    for i, (idx_train, idx_valid) in enumerate(skf.split(lines, y)):
        train, valid = lines[idx_train], lines[idx_valid]
        trainFileName = 'TRAIN-{:03d}.tsv'.format(i)
        validFileName = 'VALID-{:03d}.tsv'.format(i)
        with open(outputpath+validFileName, 'w') as f:
            for line in valid:
                f.write(line)
        with open(outputpath+trainFileName, 'w') as f:
            for line in train:
                f.write(line)
        splits += ['{path}{train}:{path}{valid}\n'.format(path=outputpath, train=trainFileName, valid=validFileName)]

    with open(outputpath+'train_list.txt', 'w') as f:
        for line in splits:
            f.write(line)

    print('{}train_list.txt'.format(outputpath))
