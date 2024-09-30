#!/Users/alexaksentyev/opt/anaconda3/bin/python3

import numpy as np
import sys

gauss = lambda mu, sig, n: np.random.normal(mu, sig, n)

def dist_tilts(sigma, number):
    gg = gauss(0, sigma, number) # generate a gaussian distribution
     # correct the vector to have a strict mean = 0
    mu = gg.mean()
    gg -= mu
    return gg

if __name__ == '__main__':
    sig = float(sys.argv[1])
    n = int(sys.argv[2])
    print('tilting.py with', sig, n)
    if len(sys.argv)>3: fname = sys.argv[3]
    else: fname = 'TILTING.in'
    gg = dist_tilts(sig, n)
    # writing to file
    np.savetxt(fname, np.transpose(gg), delimiter=' ', newline='\n')
