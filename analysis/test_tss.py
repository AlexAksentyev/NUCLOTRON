import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt; plt.ion()
from analysis import DAVEC, NBAR
from glob import glob
import re

mpl.rcParams['font.size']=14

LATNAME = '8PER'
TESTNAME = 'tilted-zeroed-comparison'

DATDIR = '../data/{}/{}/'.format(LATNAME, TESTNAME)

LEN_8 = 250.047
Fcyc8 = .5822942764643650e6 # cyclotron frequency [Hz = rev/sec]

LEN_16 = 261.760
Fcyc16 = 0.5577531383758286e6

Wcyc_dict = {8: 2*np.pi*Fcyc8, 16: 2*np.pi*Fcyc16}
wf_num_dict = {'8PER':16, '16PER':32}

def load_tss(datdir, tltdir=None):
    latname, testname = datdir.split('/')[2:4]
    cases = [int(re.findall(r'\d+',e)[1]) for e in glob(datdir+'MU:CW_*')]
    cases.sort()
    ncases = len(cases)
    nbar = {}; nu = {}
    n0 = np.zeros(ncases, dtype=list(zip(['X_CW','Y_CW','Z_CW','X_CCW','Y_CCW','Z_CCW','tilt_wf'],[float]*7)));
    nu0 = np.zeros(ncases, dtype=list(zip(['CW','CCW'],[float]*2)))
    tilts_b = np.zeros((ncases, 48))
    tilts_wf = np.zeros((ncases, wf_num_dict[latname]))
    if tltdir!=None:
        testname = tltdir
    tltdir = '../data/{}/{}/'.format(latname, testname)
    for i, case in enumerate(cases):
        print(case)
        tmp = []
        for dn in ['CW','CCW']:
            nbar.update({dn+str(case): NBAR(datdir, '{}_CASE_{}'.format(dn, case))})
            nu.update({dn+str(case): DAVEC(datdir+'MU:{}_CASE_{}'.format(dn, case)+'.da')})
            tmp += [nbar[dn+str(case)].mean[e] for e in range(3)]
            
        tilts_b[i] = np.zeros(48) #np.loadtxt(tltdir+'TILTSb:CASE_{}'.format(case)+'.in')
        tilts_wf[i] = np.loadtxt(tltdir+'TILTSwf:CASE_{}'.format(case)+'.in')
        n0[i] = tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tilts_wf[i].mean()
        nu0[i] = nu['CW'+str(case)].const, nu['CCW'+str(case)].const
    return nu, nbar, nu0, n0, (tilts_b, tilts_wf)

def process(datdir, tltdir='TILTS-source'):
    Wcyc = Wcyc_dict[int(datdir.split('/')[2].replace('PER',''))]
    nu, nbar, nu0, n0, tilts = load_tss(datdir, tltdir)
    Wx = np.zeros(len(nu0), dtype=list(zip(['CW','CCW'],[float]*2)))
    Wy = np.zeros(len(nu0), dtype=list(zip(['CW','CCW'],[float]*2)))
    Wz = np.zeros(len(nu0), dtype=list(zip(['CW','CCW'],[float]*2)))
    Wx['CW'], Wx['CCW'] = [Wcyc*nu0[lab]*n0['X_'+lab] for lab in ('CW','CCW')]
    Wy['CW'], Wy['CCW'] = [Wcyc*nu0[lab]*n0['Y_'+lab] for lab in ('CW','CCW')]
    Wz['CW'], Wz['CCW'] = [Wcyc*nu0[lab]*n0['Z_'+lab] for lab in ('CW','CCW')]
    W_dict = {'X': Wx, 'Y': Wy, 'Z': Wz}
    return W_dict, tilts

def compare8vs16(place):
    W8, tilts8 = process('../data/8PER/{}/'.format(place))
    std8 = tilts8.std(axis=1)
    W16, tilts16 = process('../data/16PER/{}/'.format(place))
    std16 = tilts16.std(axis=1)
    fig, ax = plt.subplots(1,1)
    DWY8 = W8['Y']['CW'] - W8['Y']['CCW']
    SWX8 = W8['X']['CW'] + W8['X']['CCW']
    DWY16 = W16['Y']['CW'] - W16['Y']['CCW']
    SWX16 = W16['X']['CW'] + W16['X']['CCW']
    ax.plot(DWY8, SWX8, '.', label='8')
    ax.plot(DWY16, SWX16, '.', label='16')
    ax.set_xlabel(r'$\Delta W_y$'); ax.set_ylabel(r'$\Sigma W_x$')
    ax.legend(); ax.grid(); ax.ticklabel_format(style='sci',scilimits=(0,0),useMathText=True,axis='both')
    fig2, ax2 = plt.subplots(2,1,sharex=True)
    ax2[0].set_title('8 vs 16 comparison')
    ax2[0].plot(W8['X']['CW'], W8['Z']['CW'], '.', label='8')
    ax2[0].plot(W16['X']['CW'], W16['Z']['CW'], '.', label='16')
    ax2[0].set_ylabel(r'$W_z$')
    ax2[1].set_title('CW vs CCW comparison (for 8-periodic)')
    ax2[1].plot(W8['X']['CW'], W8['Z']['CW'], '.', label='CW')
    ax2[1].plot(W8['X']['CCW'], W8['Z']['CCW'], '.', label='CCW')
    ax2[1].set_xlabel(r'$W_x$'); ax2[1].set_ylabel(r'$W_z$')
    for i in range(2):
        ax2[i].legend(); ax2[i].grid(); ax2[i].ticklabel_format(style='sci',scilimits=(0,0),useMathText=True,axis='both')
    fig3, ax3 = plt.subplots(1,1)
    ax3.plot(W8['Y']['CW'], W8['X']['CW'],'.')


if __name__ == '__main__':
    
    
    W, tilts = process(DATDIR)
    mean_tilt = tilts[1].mean(axis=1) # tlits[0] = tilts_bend, tlits[1] = tilits_wf

    fig,ax = plt.subplots(3,2)
    ax[0,0].set_title(LATNAME+' (only WF spin-kicks)')
    # col 0
    for lab in ('CW','CCW'):
        ax[0,0].plot(mean_tilt, W['X'][lab],'.', label=lab)
        ax[1,0].plot(mean_tilt, W['Y'][lab],'.', label=lab)
        ax[2,0].plot(mean_tilt, W['Z'][lab],'.', label=lab)
    ax[2,0].set_xlabel(r'$\langle\theta_{kick}\rangle$')
    ax[0,0].set_ylabel(r'$\Omega_x$ [rad/s]')
    ax[1,0].set_ylabel(r'$\Omega_y$ [rad/s]')
    ax[2,0].set_ylabel(r'$\Omega_z$ [rad/s]')
    # col 1
    ax[0,1].plot(mean_tilt, np.abs(W['X']['CW'])-np.abs(W['X']['CCW']),  '.')
    ax[1,1].plot(mean_tilt, W['Y']['CW']-W['Y']['CCW'],                  '.')
    ax[2,1].plot(W['Y']['CW']-W['Y']['CCW'], W['X']['CW']+W['X']['CCW'], '.')
    ax[0,1].set_ylabel(r'$\Delta |W_x|$')
    ax[1,1].set_xlabel(r'$\langle\theta_{kick}\rangle$'); ax[1,1].set_ylabel(r'$\Delta W_y$')
    ax[2,1].set_xlabel(r'$\Delta W_y$ [rad/s]');          ax[2,1].set_ylabel(r'$\Sigma W_x$')
    # niceties
    for i in range(3):
        for j in range(2):
            ax[i,j].grid()
            ax[i,j].ticklabel_format(style='sci',scilimits=(0,0),useMathText=True,axis='both')
        ax[i,0].legend()
    
   
