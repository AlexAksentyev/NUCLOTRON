import numpy as np
from numpy import pi, cos, sin
from scipy.spatial.transform import Rotation as R
from pandas import DataFrame
import matplotlib.pyplot as plt; plt.ion()

ntilt = lambda phi, theta: R.from_rotvec(phi/cos(theta)*np.array([sin(theta), cos(theta), 0])) # tilted element's proper axis
rkick = lambda dummy, theta: R.from_rotvec(theta*np.array([1, 0, 0])) # radial kick
vbend = lambda phi: R.from_rotvec(phi*np.array([0, 1, 0])) # vertical bend
pair  = lambda phi, theta: rkick(0, theta)*vbend(phi) # bend followed by r-kick

N_elem = 4 # total number of elements
element_array = np.zeros(N_elem, dtype=object)

tilt_rev_sign = -1
to_sec = 1e6 # turns angle rad/turn to frequency rad/sec

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

def output(array, res_direct, res_reverse, reduced=False):
    if not reduced:
        print('typical element rotation axis-angle representation')
        print('even element')
        print(array[2].as_rotvec())
        print('odd element')
        print(array[3].as_rotvec())
        print(' ')

    print('full ring rotation axis-angle representation')
    print('direct (CW)')
    print(res_direct.as_rotvec())
    print('reverse (CCW)')
    print(res_reverse.as_rotvec())
    print('absolute difference')
    print(1e6*(np.abs(res_direct.as_rotvec()) - np.abs(res_reverse.as_rotvec())), ' [rad/sec]')
    print('sum Wx')
    print(1e6*(res_direct.as_rotvec()[0] + res_reverse.as_rotvec()[0]), ' [rad/sec]')
    print(' ')

    if not reduced:
        print('spin frequency and n-bar axis (normalized)')
        print('direct (CW)')
        a, vec = normalize(res_direct.as_rotvec())
        print(a*3e6, ' [rad/s]'); print(vec)
        print('reverse (CCW)')
        a, vec = normalize(res_reverse.as_rotvec())
        print(a*3e6, ' [rad/s]'); print(vec)
        print(' ')

def base(array, tilt, edm, method):
    array = fill(array, tilt + edm, method)
    res_direct = multiply(array)
    array = fill(array, tilt_rev_sign*tilt + edm, method)
    res_reverse = multiply(array[::-1])
    return res_direct, res_reverse

def test_local(method, phi, tlit, edm_array):
    N = edm_array.shape[0]
    element_W = np.zeros(N, dtype = [('abs', float), ('x', float), ('y', float), ('z', float)])
    element_nbar = np.zeros(N, dtype= [('x', float), ('y', float), ('z', float)])
    for i, edm in enumerate(edm_array):
        vec = ntilt(phi, tlit + edm).as_rotvec()
        a, nbar = normalize(vec)
        element_W[i] = a, vec[0], vec[1], vec[2] # in [rad/turn]
        element_nbar[i] = nbar[0], nbar[1], nbar[2]
    return element_W, element_nbar 
    
if __name__ == '__main__':
    print('quasi FS model'); method = ntilt
    print(' ')
    
    tilt = np.random.normal(0, 1e-4, N_elem)
    tilt -= tilt.mean() # make strict zero mean tilt distribution

    res_direct, res_reverse = base(element_array, tilt, 0, method) # null to eyeball
    output(element_array, res_direct, res_reverse)

    # varying EDM
    N_pnt = 13*5; edm_spectrum = np.linspace(-5e-9, 5e-9, N_pnt) 
    W_array1, nbar_array1 = test_local(method, pi/16, 1e-5, edm_spectrum) # in [rad/turn]
    W_array2, nbar_array2 = test_local(method, pi/16, 1e-4, edm_spectrum)
    W_array3, nbar_array3 = test_local(method, pi/16, 5e-4, edm_spectrum)
    W_array0, nbar_array0 = test_local(method, pi/16, 0, edm_spectrum)


