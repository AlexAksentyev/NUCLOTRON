import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt; plt.ion()
from analysis import DAVEC, NBAR
from glob import glob
import re

mpl.rcParams['font.size']=14

LATTICE = '16PER'

DATDIR = '../data/'+LATTICE+'/tilted-unzeroed-comparison/'

LEN_8 = 250.047
Fcyc8 = .5822942764643650e6 # cyclotron frequency [Hz = rev/sec]

LEN_16 = 261.760
Fcyc16 = 0.5577531383758286e6

Wcyc_dict = {8: 2*np.pi*Fcyc8, 16: 2*np.pi*Fcyc16}

def load_tss(dir):
    cases = [int(re.findall(r'\d+',e)[1]) for e in glob(dir+'MU:CW_*')]
    cases.sort()
    ncases = len(cases)
    nbar = {}; nu = {}
    n0 = np.zeros(ncases, dtype=list(zip(['X_CW','Y_CW','Z_CW','X_CCW','Y_CCW','Z_CCW','tilt'],[float]*7)));
    nu0 = np.zeros(ncases, dtype=list(zip(['CW','CCW'],[float]*2)))
    tilts = np.zeros((ncases, 48))
    for i, case in enumerate(cases):
        print(case)
        tmp = []
        for dn in ['CW','CCW']:
            nbar.update({dn+str(case): NBAR(dir, '{}_CASE_{}'.format(dn, case))})
            nu.update({dn+str(case): DAVEC(dir+'MU:{}_CASE_{}'.format(dn, case)+'.da')})
            tmp += [nbar[dn+str(case)].mean[e] for e in range(3)]
            
        tilts[i] = np.loadtxt(dir+'TILTS:CASE_{}'.format(case)+'.in')
        n0[i] = tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tilts[i].mean()
        nu0[i] = nu['CW'+str(case)].const, nu['CCW'+str(case)].const
    return nu, nbar, nu0, n0, tilts


if __name__ == '__main__':
    Wcyc = Wcyc_dict[int(LATTICE.replace('PER',''))]
    
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
    
   
