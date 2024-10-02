import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from analysis import load_data, DAVEC, NBAR

DIR = '../DATA/16PER/optimize-EB-sweep/fine-mesh/'
TAG = 'CASE_0'
NPNT = 5

def load_parameters(path, tag):
    return np.loadtxt(path+'LATTICE-PARAMETERS:'+tag+'.txt',
                          dtype = list(zip(['EBE','SGX1','SGY1','SGX2','SGY2'],[float]*5)))
def load_nu(path, tag):
    return DAVEC(path+'NU:'+tag+'.da')
def load_nbar(path, tag):
    return NBAR(path, tag)

if __name__ == '__main__':
    EBE = np.zeros(NPNT+1)
    NU0 = np.zeros(NPNT+1)
    N0  = np.zeros(NPNT+1, dtype = list(zip(['X','Y','Z'],[float]*3)))
    for i in range(1,NPNT+2):
        print(i)
        parsi = load_parameters(DIR, TAG+str(i))
        nui = load_nu(DIR, TAG+str(i))
        nbari = load_nbar(DIR, TAG+str(i))
        EBE[i-1] = parsi['EBE']
        NU0[i-1] = nui.const
        N0[i-1]  = nbari.mean[0], *nbari.mean[1:]



