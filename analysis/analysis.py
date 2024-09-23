import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import sympy
from scipy.optimize import curve_fit, least_squares
from statsmodels.nonparametric.smoothers_lowess import lowess

################# constants ##########################
INTTYPE = ['iteration', 'PID', 'EID', 'ray']
HOMEDIR = '/Users/alexaksentyev/REPOS/NUCLOTRON/'
# ELNAMES = np.insert(np.load('nica_element_names.npy'),0,'INJ')
# ELNAMES = np.insert(ELNAMES, 1,'RF')

GAMMA = 1.14
L = 250
BETA = np.sqrt(1 - 1/GAMMA**2)
CLIGHT = 3e8
v = CLIGHT*BETA
TAU = L/v
############### function definintions #####################
def _read_header(fileaddress):
    with open(fileaddress) as f:
        nray_line = f.readline()
        dtype_line = f.readline()
    number = nray_line.strip().split(":")[1]
    nray = int(number) if number.isdigit() else int(number.split()[0])
    dtype = dtype_line.strip().split()[1:]
    for i, e in enumerate(dtype):
        if (e in INTTYPE):
            dtype[i] = (e, int)
        else:
            dtype[i] = (e, float)
    return nray, dtype

def _shape_up(dat, nrays):
    dat.shape = (-1, nrays)
    dat = dat[:, 1:]
    return dat
    
def load_data(path, filename):
    nray, d_type = _read_header(path+filename)
    ps = np.loadtxt(path+filename, d_type, skiprows=2)
    ps = _shape_up(ps, nray)
    return ps

def guess_freq(time, signal): # estimating the initial frequency guess
    zci = np.where(np.diff(np.sign(signal)))[0] # find indexes of signal zeroes
    delta_phase = np.pi*(len(zci)-1)
    delta_t = time[zci][-1]-time[zci][0]
    guess = delta_phase/delta_t/2/np.pi
    return guess

def guess_phase(time, sine):
    ds = sine[1]-sine[0]
    dt = time[1]-time[0]
    sg = np.sign(ds/dt)
    phase0 = np.arcsin(sine[0]) if sg>0 else np.pi-np.arcsin(sine[0])
    return phase0

def fit_line(x,y): # this is used for evaluating the derivative
    # resid = lambda p, x,y: p[0] + p[1]*x - y
    # # initial parameter estimates
    # a0 = y[0]; b0 = (y[-1]-y[0])/(x[-1]-x[0])
    # # fitting
    # result = least_squares(resid, [a0, b0], args=(x,y), loss='soft_l1', f_scale=.1)
    # popt = result.x
    # # computing the parameter errors
    # J = result.jac
    # pcov = np.linalg.inv(J.T.dot(J))*result.fun.std()
    # perr = np.sqrt(np.diagonal(pcov))
    ## same with curve_fit
    line = lambda x,a,b: a + b*x
    data_size = len(x)
    n_skip = int(.1*data_size)
    ii = slice(0, None) if len(x)<100 else slice(n_skip,-1*n_skip)
    popt, pcov = curve_fit(line, x[ii], y[ii])
    perr = np.sqrt(np.diag(pcov))
    return popt, perr

def project_spin_nbar(spdata, tssdata, ftype='CO'): # either CO, or mean
                                        # DON'T USE MEAN, I only generate data for one turn here
   def make_nbar_seq(component, repnum):
       num_el = len(component)
       pick_i = np.unique(spdata['EID'][:,0])%num_el
       x = component[pick_i]
       x0 = x[0]
       x = x[1:] #np.append(x[1:], x[:2])[:-1]
       x = np.tile(x, repnum) # x is 1d; [1,2,3,..., 1, 2, 3, ..., 1, 2, 3,... ]
       return np.insert(x, 0, x0)
   def normalize(nbar):
       norm_nbar = np.sqrt(nbar['X']**2 + nbar['Y']**2 + nbar['Z']**2)
       if np.any(abs(norm_nbar-1))>1e-6:
           print('********** nbar norm is suspect! {}, {}'.format(norm_nbar.min(), norm_nbar.max()))
       return {lbl: nbar[lbl]/norm_nbar for lbl in ['X','Y','Z']}
   s = {lbl:spdata['S_'+lbl] for lbl in ['X','Y','Z']}
   ntrn = np.unique(spdata['iteration'][1:,0])[-1]
   if ftype=='CO':
       n = {lbl:make_nbar_seq(tssdata['N'+lbl][:,0], ntrn) for lbl in ['X','Y','Z']}
   elif ftype=='mean':
       n = {lbl:make_nbar_seq(np.mean(tssdata['N'+lbl], axis=1), ntrn) for lbl in ['X','Y','Z']}
   n = normalize(n)
   prod = {lbl: (s[lbl].T*n[lbl]).T for lbl in ['X','Y','Z']}
   return prod['X']+prod['Y']+prod['Z']

def project_spin_axis(spdata, axis=[0,0,1]):
    s = {lbl:spdata['S_'+lbl] for lbl in ['X','Y','Z']}
    n = dict(zip(['X','Y','Z'], axis))
    prod = {lbl: (s[lbl].T*n[lbl]).T for lbl in ['X','Y','Z']}
    return prod['X']+prod['Y']+prod['Z']

############## class definitions ####################
class Data:
    def __init__(self, path, filename):
        self._data = load_data(path, filename)

    @property
    def data(self):
        return self._data
    @property
    def co(self):
        return self._data[:,0]
    def __getitem__(self, key):
        return self._data[key]

class TSS(Data):
    def plot(self, fun=lambda x: x[:,0]):
        norma = np.sqrt(self['NY']**2 + self['NZ']**2)
        sin_psi = self['NY']/norma
        psi = np.rad2deg(np.arcsin(sin_psi))
        fig, ax = plt.subplots(3,1, sharex=True)
        ax[0].plot(fun(self['NU']))
        ax[0].set_ylabel(r'$f(\nu_s)$')
        ax[1].set_ylabel(r'$f(\bar n_{\alpha})$')
        for v in ['NX','NY','NZ']:
            ax[1].plot(fun(self[v]), label=v)
        ax[1].legend()
        ax[2].plot(fun(psi))
        ax[2].set_ylabel(r'$\angle(\bar n,\vec v)$ [deg]')
        for i in range(3):
            ax[i].ticklabel_format(axis='both', style='sci', scilimits=(0,0), useMathText=True)
            ax[i].grid(axis='x')
        return fig, ax
        
class DAVEC:
    VARS = ['X','A','Y','B','T','D']
    def __init__(self, path):
        X,A,Y,B,T,D = sympy.symbols(self.VARS)
        self._dtype = list(zip(['i', 'coef', 'ord'] + self.VARS, [int]*9))
        self._dtype[1] = ('coef', float)
        self._data = np.loadtxt(path, skiprows=1,  dtype=self._dtype, comments='-----')
        self.const = self._data[0]['coef']
        cc = self._data['coef']
        e = {}
        for var in self.VARS:
            e[var] = self._data[var]
        expr = cc*(X**e['X'] * A**e['A'] * Y**e['Y'] * B**e['B'] * T**e['T'] * D**e['D'])
        self.coefs = cc
        self.expr = expr
        self.poly = sympy.poly(expr.sum()) # wanted to improve this with list and Poly, but
        # "list representation is not supported," what?
        
    def __call__(self, ps_arr):
        # vals = np.array([self.poly(*v) for v in ps_arr]) # for some reason this doesn't work properly
        vals = np.array([self.poly.eval(dict(zip(ps_arr.dtype.names, v))) for v in ps_arr]) # this does
        return vals

    def __sub__(self, other):
        return self.poly.sub(other.poly)

    def __add__(self, other):
        return self.poly.add(other.poly)

class NBAR:
    def __init__(self, folder, mrkr):
        self._dict = {}
        for i, lbl in [(1,'X'),(2,'Y'),(3,'Z')]:
            self._dict.update({lbl:DAVEC(folder+'NBAR{:d}:{}.da'.format(i, mrkr))})
        self._mean = np.array([self._dict[e].const for e in ['X','Y','Z']])
        self._norm = np.sqrt(np.sum(self._mean**2))
    @property
    def mean(self):
        return self._mean
    @property
    def norm(self):
        return self._norm

class Polarization(Data):
    def __init__(self, iteration, eid, value, spin_proj):
        self._data = np.array(list(zip(iteration, eid, value)),
                                  dtype = [('iteration', int), ('EID', int), ('Value', float)])
        self._spin_proj = spin_proj

    @classmethod
    def on_nbar(cls, spdata, tssdata):
        sp_proj = project_spin_nbar(spdata, tssdata, ftype='CO')
        return cls._initializer(spdata, sp_proj)

    @classmethod
    def on_axis(cls, spdata, axis=[0,0,1]):
        sp_proj = project_spin_axis(spdata, axis)
        return cls._initializer(spdata, sp_proj)

    @classmethod
    def _initializer(cls, spdata, sp_proj):
        it = spdata['iteration'][:,0]
        try:
            eid = spdata['EID'][:,0]
        except:
            eid = np.ones(it.shape)
        nray = sp_proj.shape[1]
        pol = sp_proj.sum(axis=1)/nray
        return cls(it, eid, pol, sp_proj)
    
    @property
    def spin_proj(self):
        return self._spin_proj
    @property
    def co(self):
        return self._data
    def plot(self, eid, xlab='sec'):
        jj = self['EID']==eid
        y = self['Value'][jj]
        it = self['iteration'][jj]
        t = it*TAU
        par, err = fit_line(t, y)
        fit = lowess(y[0:None:100], t[0:None:100])
        fig, ax = plt.subplots(1,1)
        if xlab=='sec':
            x = t
        elif xlab=='turn':
            x = it
            par, err =  (TAU*e for e in [par, err])
        else:
            x = t
            xlab = 'sec'
        ax.plot(x,y, '.')
        ax.plot(fit[:,0], fit[:,1], '-k')
        ax.plot(x, par[0] + x*par[1], '-r',
                    label=r'$slp = {:4.2e} \pm {:4.2e}$ [u/{}]'.format(par[1], err[1], xlab))
        ax.set_ylabel(r'$\sum_i(\vec s_i, \bar n_{})$'.format(eid))
        ax.set_xlabel(xlab)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
        ax.legend()
        return fig, ax
