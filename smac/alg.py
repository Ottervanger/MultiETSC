#!/usr/bin/env python3
import sys

a = 1
b = 1

for i in range(len(sys.argv)-1):
    if sys.argv[i] == '-a':
        a = float(sys.argv[i+1])
    if sys.argv[i] == '-b':
        b = float(sys.argv[i+1])

def f(a,b):
    return 100*(a-b**2)**2 + (1-b)**2

print('Result of this algorithm run: SUCCESS, 0.01, 0.01, {:.5f}, 0'.format(f(a,b)))
