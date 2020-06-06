#!/usr/bin/python3
import sys
import os
import numpy as np

def usage():
    print("""
    Usage: python validationsplitter.py --folds=5 --holdout=.2 <path/to/UCR/data.tsv> [output/path]

    This script generates train-validation splits either for cross validation
    or for holdout validation. Additionally it produces the correct test and
    train lists compatible with paramILS scenarios.
    """)

def argparse():
    folds = None
    holdout = None
    datafile = None
    outputpath = None
    for arg in sys.argv[1:]:
        if arg[:7] == '--folds':
            folds = int(arg[8:])
            continue
        if arg[:9] == '--holdout':
            folds = float(arg[10:])
            continue
        if not datafile:
            datafile = arg
            continue
        if not outputpath:
            outputpath = arg
            continue
        break

    if folds and holdout:
        sys.exit('Error: either folds or holdout can be defined, not both.')
    if not datafile:
        usage()
        sys.exit('Error: no data path specified')
    return folds, holdout, datafile, outputpath
        
if __name__ == '__main__':
    folds, holdout, datafile, outputpath = argparse()

    with open(datafile, 'r') as f:
        data = np.array(f.readlines())
    
    if not outputpath:
        dataname = datafile.split('/')[datafile.split('/').index('UCR')+1]
        outputpath = '/tmp/paramilsdata/UCR/{}/'.format(dataname)
    if (outputpath[-1] != '/'):
        outputpath += '/'
    try:
        os.makedirs(outputpath)
    except:
        pass
    np.random.seed(0)
    idx = list(range(len(data)))
    np.random.shuffle(idx)
    if not folds:
        folds = 1
    if not holdout:
        holdout = 1/folds
    splits = []
    for i in range(folds):
        validIdx = np.array([j%folds == i for j in idx], dtype=bool)
        valid, train = data[validIdx], data[~validIdx]
        validFileName = 'VALID-{:03d}.tsv'.format(i)
        trainFileName = 'TRAIN-{:03d}.tsv'.format(i)
        with open(outputpath+validFileName, 'w') as f:
            for line in valid:
                f.write(line)
        with open(outputpath+trainFileName, 'w') as f:
            for line in train:
                f.write(line)
        splits += ['{path}{train} {path}{valid}\n'.format(path=outputpath, train=trainFileName, valid=validFileName)]

    with open(outputpath+'train_list.txt', 'w') as f:
        for line in splits:
            f.write(line)

    print('instance_file = {}train_list.txt'.format(outputpath))
