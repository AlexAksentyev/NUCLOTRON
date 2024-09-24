import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt; plt.ion()
from analysis import DAVEC, NBAR
from glob import glob
import re

mpl.rcParams['font.size']=14

LATTICE = 'MOD2'

DATDIR = '../data/'+LATTICE+'/TSS/'

Fcyc = .5822942764643650e6 # cyclotron frequency [Hz = rev/sec]
TAU = 1/Fcyc

mrkr_form = lambda n: 'CASE_{:d}'.format(n)
case_sign = '*'

def load_tss(dir):
    cases = [int(re.findall(r'\d+',e)[1]) for e in glob(DATDIR+'ABERRATIONS:CW_'+case_sign)]
    cases.sort()
    cases = np.arange(20)
    ncases = len(cases)
    nbar = {}; nu = {}
    n0 = np.zeros(ncases, dtype=list(zip(['X_CW','Y_CW','Z_CW','X_CCW','Y_CCW','Z_CCW','tilt'],[float]*7)));
    nu0 = np.zeros(ncases, dtype=list(zip(['CW','CCW'],[float]*2)))
    tilts = np.zeros((ncases, 48))
    for i, case in enumerate(cases):
        print(case)
        tmp = []
        for dn in ['CW','CCW']:
            nbar.update({str(case)+dn: NBAR(DATDIR, dn+'_'+mrkr_form(case))})
            nu.update({str(case)+dn: DAVEC(DATDIR+'MU:'+dn+'_'+mrkr_form(case)+'.da')})
            tmp += [nbar[str(case)+dn].mean[e] for e in range(3)]
            
        tilts[i] = np.loadtxt(DATDIR+'TILTS:'+mrkr_form(case)+'.in')
        n0[i] = tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tilts[i].mean()
        nu0[i] = nu[str(case)+'CW'].const, nu[str(case)+'CCW'].const
    return nu, nbar, nu0, n0, tilts


if __name__ == '__main__':
    
    nu, nbar, nu0, n0, tilts = load_tss(DATDIR)
    
   
