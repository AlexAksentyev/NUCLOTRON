import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt; plt.ion()
from analysis import DAVEC, NBAR
from glob import glob
import re

mpl.rcParams['font.size']=14

mrkr_form = lambda n: 'CASE_{:d}'.format(n)
case_sign = '*'


DIR_8 = '../data/8PER/TSS-strict0/'
LEN_8 = 250.047
Fcyc8 = .5822942764643650e6 # cyclotron frequency [Hz = rev/sec]
Wcyc8 = 2*np.pi*Fcyc8

DIR_16 = '../data/16PER/TSS-strict0/'
LEN_16 = 261.760
Fcyc16 = 0.5577531383758286e6
Wcyc16 = 2*np.pi*Fcyc16


def load_tss(dir):
    cases = [int(re.findall(r'\d+',e)[1]) for e in glob(dir+'ABERRATIONS:CW_'+case_sign)]
    cases.sort()
    cases=np.arange(1)
    ncases = len(cases)
    nbar = {}; nu = {}
    n0 = np.zeros(ncases, dtype=list(zip(['X_CW','Y_CW','Z_CW','X_CCW','Y_CCW','Z_CCW','tilt'],[float]*7)));
    nu0 = np.zeros(ncases, dtype=list(zip(['CW','CCW'],[float]*2)))
    tilts = np.zeros((ncases, 48))
    for i, case in enumerate(cases):
        print(case)
        tmp = []
        for dn in ['CW','CCW']:
            nbar.update({str(case)+dn: NBAR(dir, dn+'_'+mrkr_form(case))})
            nu.update({str(case)+dn: DAVEC(dir+'MU:'+dn+'_'+mrkr_form(case)+'.da')})
            tmp += [nbar[str(case)+dn].mean[e] for e in range(3)]
            
        tilts[i] = np.loadtxt(dir+'TILTS:'+mrkr_form(case)+'.in')
        n0[i] = tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tilts[i].mean()
        nu0[i] = nu[str(case)+'CW'].const, nu[str(case)+'CCW'].const
    return nu, nbar, nu0, n0, tilts

def pick_nvec(nbar, label='CW'):
    vec = np.array([nbar[coo+'_'+label] for coo in ['X','Y','Z']])
    return vec

def compare(nu08,nu016,n08,n016):
    nu_stack = pd.DataFrame(np.concatenate((nu08,nu016)),index=['8','16'])
    n_stack  = pd.DataFrame(np.concatenate((n08, n016)), index=['8','16'])
    nvec8cw = pick_nvec(n08, 'CW')
    nvec16cw = pick_nvec(n016,'CW')
    return nu_stack, n_stack


if __name__ == '__main__':
    
    nu8, nbar8, nu08, n08, tilts8 = load_tss(DIR_8)
    nu16, nbar16, nu016, n016, tilts16 = load_tss(DIR_16)
    nu_stack, n_stack = compare(nu08,nu016,n08,n016)
    
   
