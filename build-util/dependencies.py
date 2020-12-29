#!/usr/bin/env python
import pkgutil
import sys

modules = [{
            'name': 'numpy',
            'reqby': 'Fixed',
           }, {
            'name': 'sklearn',
            'reqby': 'Fixed',
           }, {
            'name': 'torch',
            'reqby': 'EARLIEST',
           }, {
            'name': 'smac',
            'reqby': 'SMAC',
           }]

missing = 0
for mod in modules:
    if not pkgutil.find_loader(mod['name']):
        print(f'No module named \'{mod["name"]}\' required by {mod["reqby"]}')
        missing += 1

if missing:
    sys.exit(f'Error: Missing {missing} required python module(s)')
