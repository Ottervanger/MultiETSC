#!/usr/bin/env python3
import sys
import json

if len(sys.argv) < 2:
    print('usage: {} traj.json'.format(sys.argv[0]))
    exit(1)

with open(sys.argv[1]) as f:
    for l in f.readlines():
        print(' '.join(['-{} {}'.format(*p) for p in json.loads(l)['incumbent'].items()]))

