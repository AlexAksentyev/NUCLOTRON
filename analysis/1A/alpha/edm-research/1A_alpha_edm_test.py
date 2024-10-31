import numpy as np
from numpy import pi, cos, sin
from scipy.spatial.transform import Rotation as R
from pandas import DataFrame
import matplotlib.pyplot as plt; plt.ion()

ntilt = lambda phi, theta: R.from_rotvec(phi/cos(theta)*np.array([sin(theta), cos(theta), 0])) # tilted element's proper axis
rkick = lambda dummy, theta: R.from_rotvec(theta*np.array([1, 0, 0])) # radial kick
vbend = lambda phi: R.from_rotvec(phi*np.array([0, 1, 0])) # vertical bend
pair  = lambda phi, theta: rkick(0, theta)*vbend(phi) # bend followed by r-kick

N_elem = 40 # total number of elements
tilt = np.random.normal(0, 1e-4, N_elem)
tilt -= tilt.mean() # make strict zero mean tilt distribution

element_array = np.zeros(N_elem, dtype=object)

tilt_rev_sign = -1
to_sec = 1e6

def fill(array, tilt_, method=ntilt):
    N = array.shape[0]
    for i in range(0, N, 2): # even members
        array[i] = method(+pi/N/2, tilt_[i])
        array[i+1] = method(-pi/N/2, tilt_[i+1])
    return array

def shift(array): # this will modify the original passed array
    N = array.shape[0]
    a0 = array[0]
    for p in range(1, N):
        array[p-1] = array[p]
    array[-1] = a0

def multiply(array):
    return np.cumprod(array)[-1]

def total_field(array):
    N = array.shape[0]
    field = np.zeros(N, dtype=object)
    for p in range(N-1):
        print(p)
        field[p] = multiply(array) # 0 - OG,
        shift(array)              # 1 - shift, 2 - shift^2, ...
    print(p+1)
    field[p+1] = multiply(array)    # N-1 - shift^(N-1)
    return field

def normalize(vec):
    norm = np.sqrt(np.sum(vec**2))
    return norm, vec/norm

def output(array, res_direct_vec, res_reverse_vec, reduced=False):
    if not reduced:
        print('typical element rotation axis-angle representation')
        print('even element')
        print(array[1].as_rotvec())
        print('odd element')
        print(array[2].as_rotvec())
        print(' ')

    print('full ring rotation axis-angle representation')
    print('direct (CW)')
    print(res_direct_vec)
    print('reverse (CCW)')
    print(res_reverse_vec)
    print('absolute difference')
    print(1e6*(np.abs(res_direct_vec) - np.abs(res_reverse_vec)), ' [rad/sec]')
    print('sum Wx')
    print(1e6*(res_direct_vec[0] + res_reverse_vec[0]), ' [rad/sec]')
    print(' ')

    if not reduced:
        print('spin frequency and n-bar axis (normalized)')
        print('direct (CW)')
        a, vec = normalize(res_direct_vec)
        print(a*1e6, ' [rad/s]'); print(vec)
        print('reverse (CCW)')
        a, vec = normalize(res_reverse_vec)
        print(a*1e6, ' [rad/s]'); print(vec)
        print(' ')

def base(array, tilt, edm, method):
    array = fill(array, tilt + edm, method)
    res_direct = multiply(array) # total direct spin-rotation
    array = fill(array, tilt_rev_sign*tilt + edm, method)
    res_reverse = multiply(array[::-1]) # total reverse spin-rotation
    return res_direct.as_rotvec(), res_reverse.as_rotvec() # converted to frequencies [angle*axis]

def run_through(edm_spectrum):
    N = edm_spectrum.shape[0]
    direct_av_array  = np.zeros(N, dtype=[('abs', float), ('x', float), ('y', float), ('z', float)])
    reverse_av_array = np.zeros(N, dtype=[('abs', float), ('x', float), ('y', float), ('z', float)])
    for i, edm in enumerate(edm_spectrum):
        res_direct_vec, res_reverse_vec = base(element_array, tilt, edm, method) # integral frequency vectors [angle/turn * axis]
        direct_av_array[i]  = normalize(res_direct_vec)[0], res_direct_vec[0], *res_direct_vec[1:]
        reverse_av_array[i] = normalize(res_reverse_vec)[0], res_reverse_vec[0], *res_reverse_vec[1:]
    return direct_av_array, reverse_av_array # av = [angle*axis]
    
    
if __name__ == '__main__':
    print('quasi FS model'); method = ntilt
    print(' ')
    
    # null to eyeball
    res_direct_vec, res_reverse_vec = base(element_array, tilt, 0, method) # total frequency vectors [angle*axis]
    output(element_array, res_direct_vec, res_reverse_vec)

    # varying EDM
    N_pnt = 13*5; edm_spectrum = np.linspace(1e-16, 1e-4, N_pnt)
    direct_av_array, reverse_av_array = run_through(edm_spectrum)

    fig, ax = plt.subplots(2,2, sharex='col')
    ax[0,0].plot(edm_spectrum, to_sec*direct_av_array['x'], '-.', label='CW')
    ax[0,0].plot(edm_spectrum, -to_sec*reverse_av_array['x'], '-.', label='-CCW')
    ax[1,0].plot(edm_spectrum, to_sec*(direct_av_array['x'] + reverse_av_array['x']), '.')
    ax[0,0].set_ylabel(r'$\Omega_x$ [rad/s]'); ax[0,0].legend()
    ax[1,0].set_ylabel(r'$\Sigma \Omega_x$ [rad/s]')
    
    ax[0,1].plot(edm_spectrum, to_sec*direct_av_array['abs'], '-.', label='CW')
    ax[0,1].plot(edm_spectrum, to_sec*reverse_av_array['abs'], '-.', label='CCW')
    ax[1,1].plot(edm_spectrum, to_sec*(direct_av_array['abs'] - reverse_av_array['abs']), '.')
    ax[0,1].set_ylabel(r'$||\Omega||$ [rad/s]');   ax[0,1].legend()
    ax[1,1].set_ylabel(r'$\Delta ||\Omega||$ [rad/s]')
    for i in range(2):
        ax[1,i].set_xlabel(r'$\eta$')
        for j in range(2):
            ax[i,j].grid()
            ax[i,j].ticklabel_format(style='sci', scilimits=(0,0), useMathText=True, axis='both')

