import numpy as np
from numpy import pi, cos, sin
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt; plt.ion()

ntilt = lambda phi, theta: R.from_rotvec(phi/cos(theta)*np.array([sin(theta), cos(theta), 0])) # tilted element's proper axis
rkick = lambda dummy, theta: R.from_rotvec(theta*np.array([1, 0, 0])) # radial kick
vbend = lambda phi: R.from_rotvec(phi*np.array([0, 1, 0])) # vertical bend
pair  = lambda phi, theta: rkick(0, theta)*vbend(phi) # bend followed by r-kick

N_elem = 80 # total number of elements
tilt = np.random.normal(0, 1e-4, N_elem)
tilt -= tilt.mean() # make strict zero mean tilt distribution

element_array = np.zeros(N_elem, dtype=object)
kick_array = np.zeros(N_elem, dtype=object)
pair_array = np.zeros(N_elem, dtype=object)

tilt_rev_sign = -1
edm = 0

def normalize(vec):
    norm = np.sqrt(np.sum(vec**2))
    return norm, vec/norm

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
        print(a*1e6, ' [rad/s]'); print(vec)
        print('reverse (CCW)')
        a, vec = normalize(res_reverse.as_rotvec())
        print(a*1e6, ' [rad/s]'); print(vec)
        print(' ')

def test_base(array, tilt, method):
    array = fill(array, tilt, method)
    res_direct = multiply(array)
    array = fill(array, tilt_rev_sign*tilt, method)
    res_reverse = multiply(array[::-1])
    output(array, res_direct, res_reverse, reduced=True)
    return res_direct.as_rotvec(), res_reverse.as_rotvec()

def test_periodicity(array_num): # lattice periodicity effect test
    print('quasi FS model (N = {})'.format(array_num))
    method = ntilt
    
    tilt = np.random.normal(0, 1e-4, array_num); tilt -= tilt.mean();# tilt += 1e-5
    array = np.zeros(array_num, dtype=object)
    test_base(array, tilt, method)

def test_distvar(): # varying distributions
    print('quasi FS model')
    method = ntilt
    
    array_num = 16
    case_num = 16
    rd = np.zeros(case_num); rr = np.zeros(case_num)
    for case in range(case_num):
        print('case # {}'.format(case))
        tilt = np.random.normal(0, 1e-4, array_num); tilt -= tilt.mean(); tilt += case*1e-5
        array = np.zeros(array_num, dtype=object)
        rd_, rr_ = test_base(array, tilt, method)
        rd[case] = rd_[0]; rr[case] = rr_[0]
    return rd, rr

def test_permutations(): # same distro but different order
    print('quasi FS model')
    method = ntilt
    
    array_num = 16
    tilt = np.random.normal(0, 1e-4, array_num); tilt -= tilt.mean(); tilt += 1e-5
    rd = np.zeros(array_num); rr = np.zeros(array_num)
    for case in range(array_num):
        print('case # {}'.format(case))
        val = tilt[case]; tilt[case] = tilt[-case-1]; tilt[-case-1] = val
        array = np.zeros(array_num, dtype=object)
        rd_, rr_ = test_base(array, tilt, method)
        rd[case] = rd_[0]; rr[case] = rr_[0] # x-component of rotation vectors
    return rd, rr

if __name__ == '__main__':
    print('QFS tilted model')
    print(' ')
    element_array = fill(element_array, tilt+edm, ntilt)
    res_direct_tilt = multiply(element_array)
    element_array = fill(element_array, tilt_rev_sign*(tilt-edm), ntilt)
    res_reverse_tilt = multiply(element_array[::-1])
    output(element_array, res_direct_tilt, res_reverse_tilt)

    print('strict FS (r-kick) model')
    print(' ')
    tilt = np.random.normal(0, 1e-4, N_elem)
    kick_array = fill(kick_array, tilt+edm, rkick)
    res_direct_fs = multiply(kick_array)
    kick_array = fill(kick_array, tilt_rev_sign*(tilt-edm), rkick)
    res_reverse_fs = multiply(kick_array[::-1])
    output(kick_array, res_direct_fs, res_reverse_fs)

    # print('PAIR (Rx*Ry) model')
    # print(' ')
    # pair_array = fill(pair_array, tilt, pair)
    # res_direct_pair = multiply(pair_array)
    # pair_array = fill(pair_array, tilt_rev_sign*tilt, pair)
    # res_reverse_pair = multiply(pair_array[::-1])
    # output(pair_array, res_direct_pair, res_reverse_pair)

    
    # for n in range(4,17, 4):
    #     test_periodicity(n)

    # rd1, rr1 = test_distvar()
    # # rd2, rr2 = test_permutations()
    # fig, ax = plt.subplots(2,2, sharey='row')
    # ax[0,0].plot(rd1, '-', label='CW'); ax[0,0].plot(rr1, '-', label='CCW'); ax[0,0].set_title('different distros')
    # #ax[0,1].plot(rd2, '-', label='CW'); ax[0,1].plot(rr2, '-', label='CCW'); ax[0,1].set_title('distro permutations')
    # ax[1,0].plot(rd1+rr1, '-');# ax[1,1].plot(rd2+rr2, '-');  ax[1,0].set_ylabel('CW + CCW')
    # ax[0,1].legend()
    # for i in range(2):
    #     for j in range(2):
    #         ax[i,j].ticklabel_format(style='sci', scilimits=(0,0), axis='y', useMathText=True)
