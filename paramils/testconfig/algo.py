import numpy as np
import sys
import time

def argparse():
    args={}
    for i, arg in enumerate(sys.argv):
        if i+1 < len(sys.argv):
            args[arg] = sys.argv[i+1]
    return args

if __name__ == '__main__':
    start = time.time()
    args = argparse()
    y0 = float(args.get('-y0',1))
    y1 = float(args.get('-y1',0))
    x0 = y0
    x1 = 2 - (y0 + y1)
    dt = time.time()-start
    print('Result: SUCCESS, {:f}, [{:f}, {:f}], 0'.format(dt, x0, x1))
