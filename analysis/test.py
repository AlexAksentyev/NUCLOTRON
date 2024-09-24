import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib as mpl
import matplotlib.pyplot as plt; plt.ion()
from analysis import load_data, guess_freq, guess_phase
import numpy.lib.recfunctions as rfn
from scipy.optimize import curve_fit

mpl.rcParams['font.size']=14

LATTICE = 'MOD2'

DATDIR = '../data/'+LATTICE+'/TRACKING/'

Wcyc = .5822942764643650e6 # cyclotron frequency [Hz = rev/sec]
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
    

def load(case):
    datCW = load_data(DATDIR, 'TRPRAY:CW_CASE_{}.dat'.format(case))
    spdatCW = load_data(DATDIR, 'TRPSPI:CW_CASE_{}.dat'.format(case))
    # datCCW = load_data(DATDIR, 'TRPRAY:CCW_CASE_{}.dat'.format(case))
    # spdatCCW = load_data(DATDIR, 'TRPSPI:CCW_CASE_{}.dat'.format(case))
    tilts = np.loadtxt(DATDIR+'TILTS:CASE_{}.in'.format(case))/np.pi * 180  # rad -> deg
    return datCW, spdatCW, tilts #, datCCW, spdatCCW

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

def process(case):
    datCW, spdatCW, tilts = load(case)
    tilt_mu, tilt_std = tilts.mean(), tilts.std()
    tilt_dist = lambda x: gauss(x,tilt_std,tilt_mu) # 1/np.sqrt(2*np.pi*tilt_std**2)*np.exp(-(x-tilt_mu)**2/(2*tilt_std**2))

    # fig, ax = plt.subplots(2,2,sharey='row')
    # ax[0,0].set_title('CW')                ; ax[0,1].set_title('CCW')
    # ax[0,0].plot(datCW['X'],datCW['A'],'.'); ax[0,1].plot(datCCW['X'],datCCW['A'],'.')
    # ax[1,0].plot(datCW['Y'],datCW['B'],'.'); ax[1,1].plot(datCCW['Y'],datCCW['B'],'.')
    # ax[0,0].set_ylabel('A'); ax[0,0].set_xlabel('X'); ax[0,1].set_xlabel('X')
    # ax[1,0].set_ylabel('B'); ax[1,0].set_xlabel('Y'); ax[1,1].set_xlabel('Y')
    # fig1,ax1 = plt.subplots(1,1)
    # ax1.hist(tilts, histtype='step', density=True, label=r'$\bar\theta = {:4.2e}$'.format(tilt_mu))
    # ax1.set_xlabel(r'$\theta_{tilt}$'); ax1.set_ylabel('density')
    # tilts_sorted = np.sort(tilts)
    # ax1.plot(tilts_sorted, tilt_dist(tilts_sorted),label=r'$\sigma[\theta] = {:4.2e}$'.format(tilt_std))
    # ax1.legend()

    t = spdatCW['iteration'][:,0]*TAU
    PyCW = spdatCW['S_Y'].mean(axis=1)
    noise = np.random.normal(0,1e-3,len(PyCW)) # add "polarimeter noise" to "measurement" data
    if case==0:
        popt, perr = fit_line(t,PyCW+noise, plot=True)
        comment = '0'
    else:
        check = True
        while check:
            try:
                popt, perr = fit_sine(t,PyCW+noise, plot=False)
                comment = ' '
            except:
                popt, perr = fit_line(t,PyCW+noise, plot=True)
                comment = '!'
            check = np.any(perr>1)
            noise = np.random.normal(0,1e-3,len(PyCW)) # redo noise
    return popt, perr, tilt_mu, tilt_std, comment


if __name__ == '__main__':
    
    cases = np.arange(57)
    
    data = np.zeros(len(cases),
                        dtype=list(zip(['a','f','Sa','Sf','mean_tilt','tilt_std','comment'],['float']*6+['object']))
                    )
    for case in cases:
        print('case ', case)
        popt, perr, tilt_mu, tilt_std, comment = process(case)
        data[case] = popt[0],popt[1],perr[0],perr[1], tilt_mu, tilt_std, comment

    # hypothesis testing the null case
    z_score = data[0]['f']/data[0]['Sf']  # assuming the slope is actually zero (b/c no tilts)
    p_value = stats.norm.sf(abs(z_score)) # look for the probability to get the estimated mean
    fig_hyp, ax_hyp = plot_gauss(data['f'][0],data['Sf'][0])
    ax_hyp.vlines(0,0,500,colors='red')

                       #                                COMMENT                                          #

                       # in our case, the z-score is >2, the P-value ~ 2% < 5%, which is somewhat statistically significant
                       # this could be an indication that there's indeed a QFS-augmentation to the Wx rotator-frequency
                       # we will assume that the obtained value data[0]['f'] is this QFS-augmentation
                       # (in other words, that if the Nuclotron operated in FS the z-score would be less;
                       # which can be tested by comparing the data[0]['f'] in the 8-period Nuclotron vs the 16-period).
                       # This will allow us to extract the tilt-augmentation from the data['f'] data by elimitating the data[0]['f']
                       # (under the assumption that the total Wx = Wx[EDM] + Wx[QFS] + Wx[tilt],
                       #                                       and Wx[EDM] = 0 in these simulations).

    hyp_Wx_tilt = data['f'] - data['f'][0]
    hyp_Wx_tilt_std = np.sqrt(data['Sf']**2 + data['Sf'][0]**2)
    plt.errorbar(data['mean_tilt'],hyp_Wx_tilt,yerr=hyp_Wx_tilt,xerr=data['tilt_std'])
