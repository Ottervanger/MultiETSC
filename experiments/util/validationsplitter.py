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
    return folds, datafile, outputpath, seed, reps


def main():
    folds, datafile, outputpath, initSeed, reps = argparse()

    # read data
    y = np.genfromtxt(datafile)[:, 0]
    with open(datafile, 'r') as f:
        lines = np.array(f.readlines())
    from sklearn.model_selection import StratifiedShuffleSplit
    skf = StratifiedShuffleSplit(n_splits=folds, test_size=1/folds)

    for seed in range(initSeed, initSeed+reps):
        np.random.seed(seed)
        # prepare output dir
        if not outputpath:
            tmp = os.environ['TMP']
            if not tmp:
                tmp = '/tmp'
            dataname = datafile.split('/')[datafile.split('/').index('UCR')+1]
            outputpath = tmp+'/UCRsplits/{}/seed-{}'.format(dataname, seed)
        if (outputpath[-1] != '/'):
            outputpath += '/'
        # for simplicity we assume if the dir exists it is correctly populated
        try:
            os.makedirs(outputpath)
        except FileExistsError:
            pass

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


if __name__ == '__main__':
    main()
