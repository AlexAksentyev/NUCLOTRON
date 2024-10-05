import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt; plt.ion()
from analysis import load_data, guess_freq, guess_phase
from scipy.optimize import curve_fit
from analysis import DAVEC, NBAR

mpl.rcParams['font.size']=14

LATTICE = '8PER'

DATDIR = '../data/'+LATTICE+'/calibration-phase0/'

Wcyc = 2*np.pi*.5822942764643650e6 # cyclotron frequency [Hz = rev/sec]
TAU = 1/Wcyc

rate_per_sec = lambda rate_per_turn: rate_per_turn*Wcyc
gauss = lambda x,s,m: 1/np.sqrt(2*np.pi*s**2)*np.exp(-(x-m)**2/(2*s**2))

def plot_PS(dat):
    fig, ax = plt.subplots(2,1)
    ax[0].plot(dat['X'],dat['A'],'.')
    ax[1].plot(dat['Y'],dat['B'],'.')
    ax[0].set_ylabel('A'); ax[0].set_xlabel('X')
    ax[1].set_ylabel('B'); ax[1].set_xlabel('Y')
    return fig, ax

def plot_SP(spdat):
    fig, ax = plt.subplots(3,1,sharex=True)
    t = spdat['iteration']*TAU
    ax[0].plot(t, spdat['S_X'],'.'); ax[0].set_ylabel('S_X')
    ax[1].plot(t, spdat['S_Y'],'.'); ax[1].set_ylabel('S_Y')
    ax[2].plot(t, spdat['S_Z'],'.'); ax[2].set_ylabel('S_Z')
    ax[2].set_xlabel('t [sec]')
    return fig, ax

def plot_gauss(mean,sigma):
    x = np.linspace(mean-5*sigma,mean+5*sigma,1000)
    y = gauss(x,sigma,mean)
    fig, ax = plt.subplots(1,1)
    ax.plot(x, y, color='blue')
    ax.vlines(x=mean, ymin=0, ymax=y.max(),linestyles='dashed',colors='blue')
    ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0),useMathText=True)
    return fig, ax
    

def load_tracking(path, mrkr, dn='CW'):
    datCW = load_data(path, 'TRPRAY:{}_{}.dat'.format(dn, mrkr))
    spdatCW = load_data(path, 'TRPSPI:{}_{}.dat'.format(dn, mrkr))
    return datCW, spdatCW

def load_tss(path, mrkr, dn='CW'):
    nu = DAVEC(path+'MU:{}_{}'.format(dn, mrkr)+'.da')
    nbar = NBAR(path, '{}_{}'.format(dn, mrkr))
    return nu, nbar

def fit_sine(x,y, plot=False):
    f0 = guess_freq(x,y)
    sine = lambda x,a,f: a*np.sin(2*np.pi*f*x) #longitudinal beam injection hence phase 0 (^)
    popt, pcov = curve_fit(sine,x,y,p0=(1,f0))
    perr = np.sqrt(np.diag(pcov))
    if plot:
        yfit = sine(x,popt[0],popt[1])
        fig3, ax3 = plt.subplots(1,1)
        ax3.plot(x,y,'-')
        ax3.plot(x,yfit,'--r')
    return popt, perr

def fit_line(x,y, plot=False):
    f0 = (y[-1]-y[0])/(x[-1]-x[0])
    line = lambda x,a,f: a + f*x                 #no intercept for same (^) reason
    popt, pcov = curve_fit(line,x,y,p0=(0,f0))
    perr = np.sqrt(np.diag(pcov))
    if plot:
        yfit = line(x,popt[0],popt[1])
        fig3, ax3 = plt.subplots(1,1)
        ax3.plot(x,y,'-',label = r'a = {:4.2e} $\pm$ {:4.2e}'.format(popt[0],perr[0]))
        ax3.plot(x,yfit,'--r',label=r'b = {:4.2e} $\pm$ {:4.2e}'.format(popt[1],perr[1]))
        ax3.legend()
    return popt, perr

def compute(trpray):
    # make computable phase space holder
    z = np.zeros(trpray.shape, dtype = list(zip(['X','A','Y','B','T','D'],[float]*6)))
    for lab in ['X','A','Y','B','T','D']:
        z[lab] = trpray[lab]

    # compute spin-tune and n-bar at this order expansion and this tracking data (orbit)
    nu_v = np.zeros(trpray.shape)
    nbar_v = np.zeros(trpray.shape, dtype=list(zip(['X','Y','Z'],[float]*3)))
    for p in range(trpray.shape[1]):
        nu_v[:,p] = nu(z[:,p])
        for lab in ('X','Y','Z'):
            nbar_v[lab][:,p] = nbar[lab](z[:,p])
    return nu_v, nbar_v
                
if __name__ == '__main__':

    tiltcase = 'CASE_1'
    offset_rng = '2_positive'
    cosy_expansion_order = 3
    skip_compute = True
    
    # load cosy-infinity data
    tilts = np.loadtxt(DATDIR+'TILTS:{}.in'.format(tiltcase))
    trpray, trpspi = load_tracking(DATDIR+'tracking/range-{}/'.format(offset_rng), tiltcase)
    nu, nbar = load_tss(DATDIR+'tss/{}-order-expansion/'.format(cosy_expansion_order), tiltcase)
    n_pcl = trpray.shape[1]

    # computing data for analysis
    if not skip_compute:
        nu_v, nbar_v = compute(trpray)
        # save it to avoid recomputing
        np.save('../data/{}/calibration-phase0/tracking/range-{}/analysis/nu_v'.format(LATTICE, offset_rng), nu_v)
        np.save('../data/{}/calibration-phase0/tracking/range-{}/analysis/nbar_v'.format(LATTICE, offset_rng), nbar_v)
    else:
        nu_v = np.load(DATDIR+'tracking/range-{}/analysis/nu_v.npy'.format(offset_rng))
        nbar_v = np.load(DATDIR+'tracking/range-{}/analysis/nbar_v.npy'.format(offset_rng))

    ######################################      PLOTTING    ################################

    # graphic output for communication with colleagues
    # plot of incoherent spin field gamma-effective[-correctible] effect
    n = trpray[:,0]['iteration']
    omega_x = nu_v*nbar_v['X']*Wcyc # ! these are "momentary" omegas that we'll never actually see
    omega_y = nu_v*nbar_v['X']*Wcyc # ! (=> mean omega-values computed not from these)
    fig0, ax0 = plt.subplots(1,1)
    ax0.plot(n, omega_x[:,1], label='-1mm x-shift', ls='-', color='thistle')
    ax0.axhline(omega_x[:,1].mean(), label='-1mm x-shift mean', ls='--', color='red')
    ax0.plot(n, omega_x[:,0], label='co', ls='-', color='royalblue')
    ax0.legend()
    ax0.set_xlabel('turn')
    ax0.set_ylabel('omega_x')
    ax0.set_title('delta mean - co = {:4.2e}'.format(omega_x[:,1].mean()-omega_x[0,0]))
    
    
    # plot of gamma-effective[-...] effects
    whole_omega_x = nu_v.mean(axis=0)*nbar_v['X'].mean(axis=0)*Wcyc # ! these are supposedly /like/ what we'll truly see
    whole_omega_y = nu_v.mean(axis=0)*nbar_v['Y'].mean(axis=0)*Wcyc # ! we'll never see "momentary" omegas, => wouldn't "average" /them/
    whole_omega_z = nu_v.mean(axis=0)*nbar_v['Z'].mean(axis=0)*Wcyc
    n = np.arange( 1, int( (n_pcl-1)/3 )+1 )
    y1 = whole_omega_x
    y2 = whole_omega_y
    fig, ax = plt.subplots(4,1)
    ax[0].set_title('omega_x')
    for i, lab in enumerate(['X','Y','D']):
        ax[i].plot(n, y1[range(i+1,n_pcl,3)], '-.')
        ax[i].set_ylabel('inc {}'.format(lab))
        ax[i].axhline(y=y1[0], ls='--', color='r')
        ax[i].ticklabel_format(style='sci', scilimits=(0,0), useMathText=True, axis='y')
        ax[3].plot(y2[range(i+1,n_pcl,3)], y1[range(i+1,n_pcl,3)], '.', label=lab)
    ax[3].ticklabel_format(style='sci', scilimits=(0,0), useMathText=True, axis='both')
    ax[3].set_xlabel('omega_y'); ax[3].set_ylabel('omega_x')
    ax[3].legend()
