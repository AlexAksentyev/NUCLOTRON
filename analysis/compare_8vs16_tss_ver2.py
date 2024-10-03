import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; plt.ion()
import matplotlib as mpl
from analysis import DAVEC, NBAR
from glob import glob
import re

mpl.rcParams['font.size']=14

LEN_8 = 250.047
Fcyc8 = .5822942764643650e6 # cyclotron frequency [Hz = rev/sec]

LEN_16 = 261.760
Fcyc16 = 0.5577531383758286e6

Wcyc = {8: 2*np.pi*Fcyc8, 16: 2*np.pi*Fcyc16}

def load_tss(path):
    cases = glob(path+'ABERRATIONS:*')
    tags = [string.replace(path+'ABERRATIONS:','') for string in cases]
    ncases = len(cases)
    nbar = {}; nu = {}
    tss0 = np.zeros(ncases,
                        dtype=list(zip(['nu','X','Y','Z','tilt_sd','periodicity','clockwiseness'],[float]*6+[object])));
    tilts = np.zeros((ncases, 48))
    for i, tag in enumerate(tags):
        print(tag)
        nbar = NBAR(path, tag)
        nu   = DAVEC(path+'MU:'+tag+'.da')
        meta, case = tag.split('-')
        clockwiseness, periodicity = meta.split('_')
        periodicity = periodicity[:-3] # remove the 'PER'
        tilts[i] = np.loadtxt(path+'TILTS:'+case+'.in')
        tilt_sd = tilts[i].std()
        tss0[i] = nu.const, nbar.mean[0], nbar.mean[1], nbar.mean[2], tilt_sd, periodicity, clockwiseness
    return tss0, tilts

if __name__ == '__main__':
    tss0, tilts = load_tss('../data/COMPARISON/')
    df = pd.DataFrame(tss0)
    df = df.sort_values(['tilt_sd','periodicity','clockwiseness'])
    tilts = tilts[df.index]
    
    Wcyc_list = np.tile(np.repeat(list(Wcyc.values()),2),51)
    df.insert(5, 'Wcyc', Wcyc_list, False)
    
    Wx = np.array(df['nu']*df['X']*df['Wcyc'])
    Wy = np.array(df['nu']*df['Y']*df['Wcyc'])
    Wz = np.array(df['nu']*df['Z']*df['Wcyc'])
    W_df = pd.DataFrame([Wx,Wy,Wz]).T
    W_df.columns = ['X','Y','Z']
    W_df.index = df.index
    W_df.insert(3, 'tilt_sd', df['tilt_sd'], False)
    W_df.insert(4,'periodicity', df['periodicity'], False)
    W_df.insert(5, 'clockwiseness', df['clockwiseness'], False)
    
    icw = np.arange(1,204,2)
    iccw = np.arange(0,204,2)

    tilt_sd = tilts.std(axis=1)[iccw]; tilt_sd.shape = (-1,2)
    DWx = Wx[icw]+Wx[iccw]; DWx.shape = (-1,2)
    DWy = Wy[icw]-Wy[iccw]; DWy.shape = (-1,2)
    DWz = Wz[icw]-Wz[iccw]; DWz.shape = (-1,2)

    fig, ax = plt.subplots(3,1,sharex=True)
    ax[0].plot(tilt_sd[:,0], DWx[:,0], '.', label='8')
    ax[0].plot(tilt_sd[:,1], DWx[:,1], '.', label='16')
    ax[0].set_ylabel(r'$\Omega_x^{CW} + \Omega_x^{CCW}$')
    ax[0].set_title('systematic error CW + CCW [rad/sec]')
    
    ax[1].plot(tilt_sd[:,0], DWy[:,0], '.', label='8')
    ax[1].plot(tilt_sd[:,1], DWy[:,1], '.', label='16')
    ax[1].set_ylabel(r'$\Omega_y^{CW} - \Omega_y^{CCW}$')
    
    ax[2].plot(tilt_sd[:,0], DWz[:,0], '.', label='8')
    ax[2].plot(tilt_sd[:,1], DWz[:,1], '.', label='16')
    ax[2].set_ylabel(r'$\Omega_z^{CW} - \Omega_z^{CCW}$')
    ax[2].set_xlabel(r'$\sigma[\theta_{tilt}]$')
    for i in range(3):
        ax[i].grid()
        ax[i].legend()
        ax[i].ticklabel_format(style='sci',scilimits=(0,0),useMathText=True,axis='both')

    Wx.shape = (-1,2,2)
    Wy.shape = (-1,2,2)
    Wz.shape = (-1,2,2)
    fig2, ax2 = plt.subplots(3,1,sharex=True)
    ax2[0].plot(tilt_sd[:,0], Wx[:,0,1], '.', label='8')
    ax2[0].plot(tilt_sd[:,1], Wx[:,1,1], '.', label='16')
    ax2[0].set_ylabel(r'$\Omega_x$ [rad/s]')
    ax2[0].set_title('CW')
    
    ax2[1].plot(tilt_sd[:,0], Wy[:,0,1], '.', label='8')
    ax2[1].plot(tilt_sd[:,1], Wy[:,1,1], '.', label='16')
    ax2[1].set_ylabel(r'$\Omega_y$ [rad/s]')
    
    ax2[2].plot(tilt_sd[:,0], Wx[:,0,1]/Wy[:,0,1], '.', label='8')
    ax2[2].plot(tilt_sd[:,1], Wx[:,1,1]/Wy[:,1,1], '.', label='16')
    ax2[2].set_ylabel(r'$\Omega_x/\Omega_y$')
    ax2[2].set_xlabel(r'$\sigma[\theta_{tilt}]$')
    for i in range(3):
        ax2[i].grid()
        ax2[i].legend()
        ax2[i].ticklabel_format(style='sci',scilimits=(0,0),useMathText=True,axis='both')

    df_nusort = df.sort_values('nu')
    j8cw = (df_nusort['periodicity']==8)&(df_nusort['clockwiseness']=='CW')
    j16cw = (df_nusort['periodicity']==16)&(df_nusort['clockwiseness']=='CW')
    fig3, ax3 = plt.subplots(2,1,sharex=True)
    ax3[0].set_title(r'$\bar n$ : CW')
    ax3[0].set_ylabel('8')
    ax3[0].plot(df_nusort['nu'][j8cw], df_nusort['X'][j8cw], '-.',label='X')
    ax3[0].plot(df_nusort['nu'][j8cw], df_nusort['Y'][j8cw], '-.',label='Y')
    ax3[0].plot(df_nusort['nu'][j8cw], df_nusort['Z'][j8cw], '-.',label='Z')
    ax3[1].set_ylabel('16')
    ax3[1].plot(df_nusort['nu'][j16cw], df_nusort['X'][j16cw], '-.',label='X')
    ax3[1].plot(df_nusort['nu'][j16cw], df_nusort['Y'][j16cw], '-.',label='Y')
    ax3[1].plot(df_nusort['nu'][j16cw], df_nusort['Z'][j16cw], '-.',label='Z')
    ax3[1].set_xlabel(r'$\nu$')
    for i in range(2):
        ax3[i].grid()
        ax3[i].legend()
        ax3[i].ticklabel_format(style='sci',scilimits=(0,0),useMathText=True,axis='both')
