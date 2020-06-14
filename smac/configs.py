#!/usr/bin/env python3
import sys
import json

if len(sys.argv) < 2:
    print('usage: {} traj.json'.format(sys.argv[0]))
    exit(1)

with open(sys.argv[1]) as f:
    for line in f.readlines():
        for k, v in json.loads(line)['incumbent'].items():
            print('-{} {} '.format(k, v), end='')
        print()
    
