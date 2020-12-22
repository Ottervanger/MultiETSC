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
    seed = 0
    reps = 1
    for arg in sys.argv[1:]:
        argname, argval = (arg+'=').split('=')[0:2]
        if 'folds' in argname:
            folds = int(argval)
            continue
        if 'reps' in argname:
            reps = int(argval)
            continue
        if 'seed' in argname:
            seed = int(argval)
            continue
        if 'outpath' in argname:
            outputpath = argval
            continue
        if not datafile:
            datafile = arg
            continue
        break
    if not folds:
        folds = 5
    if not datafile:
        usage()
        sys.exit('Error: no data path specified')
    return folds, datafile, seed, reps, outputpath


def makeSplit(outputpath, lines, y, skf):
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


def main():
    folds, datafile, initSeed, reps, outputpath = argparse()

    # read data
    y = np.genfromtxt(datafile)[:, 0]
    with open(datafile, 'r') as f:
        lines = np.array(f.readlines())
    from sklearn.model_selection import StratifiedShuffleSplit
    skf = StratifiedShuffleSplit(n_splits=folds, test_size=1/folds)

    if outputpath is None:
        tmp = os.environ['TMP']
        if not tmp:
            tmp = '/tmp'
        dataname = datafile.split('/')[datafile.split('/').index('UCR')+1]
        outputpath = f'{tmp}/UCRsplits/{dataname}/seed-{{}}/'

    for seed in range(initSeed, initSeed+reps):
        np.random.seed(seed)
        # for simplicity we assume if the dir exists it is correctly populated
        try:
            os.makedirs(outputpath.format(seed))
        except FileExistsError:
            continue
        try:
            makeSplit(outputpath.format(seed), lines, y, skf)
        except ValueError:
            shutil.rmtree(outputpath.format(seed))
            sys.exit(1)


if __name__ == '__main__':
    main()
