import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt; plt.ion()
from analysis import DAVEC, NBAR
from glob import glob
import re

mpl.rcParams['font.size']=14

LATTICE = '8PER'

DATDIR = '../data/'+LATTICE+'/TSS/'

Fcyc = .5822942764643650e6 # cyclotron frequency [Hz = rev/sec]
TAU = 1/Fcyc
Wcyc = 2*np.pi*Fcyc

mrkr_form = lambda n: 'CASE_{:d}'.format(n)
case_sign = '*'

def load_tss(dir):
    cases = [int(re.findall(r'\d+',e)[1]) for e in glob(dir+'ABERRATIONS:CW_'+case_sign)]
    cases.sort()
    cases=np.arange(51)
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


if __name__ == '__main__':
    
    nu, nbar, nu0, n0, tilts = load_tss(DATDIR)
    Wx = np.zeros(len(nu0), dtype=list(zip(['CW','CCW'],[float]*2)))
    Wy = np.zeros(len(nu0), dtype=list(zip(['CW','CCW'],[float]*2)))
    Wz = np.zeros(len(nu0), dtype=list(zip(['CW','CCW'],[float]*2)))
    Wx['CW'], Wx['CCW'] = [Wcyc*nu0[lab]*n0['X_'+lab] for lab in ('CW','CCW')]
    Wy['CW'], Wy['CCW'] = [Wcyc*nu0[lab]*n0['Y_'+lab] for lab in ('CW','CCW')]
    Wz['CW'], Wz['CCW'] = [Wcyc*nu0[lab]*n0['Z_'+lab] for lab in ('CW','CCW')]
    n00_CW, n00_CCW = [np.array((n0[0]['X_'+lab], n0[0]['Y_'+lab], n0[0]['Z_'+lab])) for lab in ('CW','CCW')]
    W0_CW, W0_CCW = Wcyc*nu0[0]['CW']*n00_CW, Wcyc*nu0[0]['CCW']*n00_CCW
    mean_tilt = tilts.mean(axis=1)
    DeltaCW = Wx['CW']-Wx['CW'][0]
    DeltaCCW = Wx['CCW']-Wx['CCW'][0]

    fig,ax = plt.subplots(3,1,sharex=True)
    ax[0].set_title(LATTICE)
    for lab in ('CW','CCW'):
        ax[0].plot(mean_tilt, Wx[lab],'.', label=lab)
        ax[1].plot(mean_tilt, Wy[lab],'.', label=lab)
        ax[2].plot(mean_tilt, Wz[lab],'.', label=lab)
    ax[2].set_xlabel(r'$\langle\theta_{tilt}\rangle$')
    ax[0].set_ylabel(r'$\Omega_x$ [rad/s]')
    ax[1].set_ylabel(r'$\Omega_y$ [rad/s]')
    ax[2].set_ylabel(r'$\Omega_z$ [rad/s]')
    for i in range(3):
        ax[i].grid()
        ax[i].legend()
        ax[i].ticklabel_format(style='sci',scilimits=(0,0),useMathText=True,axis='both')
    
   
