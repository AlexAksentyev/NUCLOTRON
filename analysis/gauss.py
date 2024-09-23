#!/Users/alexaksentyev/opt/anaconda3/bin/python3

import numpy as np
import sys

gauss = lambda mu, sig, n: np.random.normal(mu, sig, n)

if __name__ == '__main__':
    mu = float(sys.argv[1])
    sig = float(sys.argv[2])
    n = int(sys.argv[3])
    print('gauss.py with', mu, sig, n)
    if len(sys.argv)>4: fname = sys.argv[4]
    else: fname = 'GAUSS.in'
    gg = gauss(mu, sig, n)
    np.savetxt(fname, np.transpose(gg), delimiter=' ', newline='\n')
