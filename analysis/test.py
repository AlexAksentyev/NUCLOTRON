import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt; plt.ion()
from analysis import load_data, guess_freq, guess_phase
import numpy.lib.recfunctions as rfn
from scipy.optimize import curve_fit

mpl.rcParams['font.size']=14

LATTICE = 'MOD2'

DATDIR = '../data/'+LATTICE+'/'

Wcyc = .5822942764643650e6 # cyclotron frequency [Hz = rev/sec]
TAU = 1/Wcyc

rate_per_sec = lambda rate_per_turn: rate_per_turn*Wcyc

def load(case):
    datCW = load_data(DATDIR, 'TRPRAY:CW_CASE_{}.dat'.format(case))
    spdatCW = load_data(DATDIR, 'TRPSPI:CW_CASE_{}.dat'.format(case))
    datCCW = load_data(DATDIR, 'TRPRAY:CCW_CASE_{}.dat'.format(case))
    spdatCCW = load_data(DATDIR, 'TRPSPI:CCW_CASE_{}.dat'.format(case))
    return datCW, spdatCW, datCCW, spdatCCW

def fit_sine(x,y):
    f0 = guess_freq(x,y)
    sine = lambda x,a,f: a*np.sin(2*np.pi*f*x) #longitudinal beam injection hence phase 0
    popt, pcov = curve_fit(sine,x,y,p0=(1,f0))
    perr = np.sqrt(np.diag(pcov))
    yfit = sine(x,popt[0],popt[1])
    plt.plot(x,y,'-')
    plt.plot(x,yfit,'--r')
    return popt, perr

def process(case):
    datCW, spdatCW, datCCW, spdatCCW = load(case)
    tilts = np.loadtxt(DATDIR+'TILTS:CASE_{}.in'.format(case))/np.pi * 180  # rad -> deg

    t = spdatCW['iteration'][:,0]*TAU
    PyCW = spdatCW['S_Y'].mean(axis=1)
    noise = np.random.normal(0,1e-3,len(PyCW))  
    popt, perr = fit_sine(t,PyCW+noise)
    return popt, perr


if __name__ == '__main__':
    
    cases = [0,1,2]
    
    data = np.zeros(len(cases),dtype=list(zip(['a','f','Sa','Sf'],['float']*4)))
    for case in cases:
        popt, perr = process(case)
        data[case] = popt[0],popt[1],perr[0],perr[1]
