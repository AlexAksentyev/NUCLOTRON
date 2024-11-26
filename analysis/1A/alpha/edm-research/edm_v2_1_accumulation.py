## at this point I wll assume that since v2 I do edm-modification of element spin-rotation correctly
## hence fill_edm_mod() and base_edm_mode() will be called simply fill() and base()

import numpy as np
from numpy import pi, cos, sin
from scipy.spatial.transform import Rotation as R
from pandas import DataFrame
import matplotlib.pyplot as plt; plt.ion()

ntilt = lambda phi, theta: R.from_rotvec(phi/cos(theta)*np.array([sin(theta), cos(theta), 0])) # tilted element's proper axis
rkick = lambda dummy, theta: R.from_rotvec(theta*np.array([1, 0, 0])) # radial kick

tilt_rev_sign = -1
turns_per_sec = 1e6 

edm_modify_as_rkick = lambda mdm_roter, edm_kick: R.from_rotvec(mdm_roter.as_rotvec() + rkick(0, edm_kick).as_rotvec()) # modifies mdm-rotation adding edm-effect

def edm_modify(mdm_roter, edm_kick):
    z_axis = np.array([0,0,1])
    mdm_vec = mdm_roter.as_rotvec();       mdm_a, mdm_nbar = normalize(mdm_vec)
    if mdm_a == 0: #   if the element doesn't do any mdm spin-rotation (this only happens in ideal/"untilted" FS),
        edm_vec = edm_kick*np.array([1,0,0]) # then edm would be the only spin-vector-driving force, and do a radial rotation
    elif mdm_nbar[1] == 0: #   if the element does an mdm-rotation, but no vertical component, then we're working with FS
        edm_vec = mdm_nbar[0]*edm_kick*mdm_nbar # and edm_vec // mdm_vec; mdm_nbar[0] factor to keep edm constant-direction
    else: #   otherwise compute edm kick axis as velocity x magnetic field
        edm_nbar = np.cross(z_axis, mdm_nbar);    edm_vec = edm_kick*edm_nbar
    return R.from_rotvec(mdm_vec + edm_vec)

def fill(mdm_method, tilt_array, edm_kick):
    N = tilt_array.shape[0]
    array = np.zeros(N, dtype=object)
    ###print('edm_kick = ', edm_kick)
    for i in range(0, N, 2): # even members
        mdm_rotation_even = mdm_method(+pi/N/2, tilt_array[i])
        mdm_rotation_odd = mdm_method(-pi/N/2, tilt_array[i+1])
        ###print('mdm rotation (even)', mdm_rotation_even.as_rotvec())
        ###print('edm kick', rkick(0, edm_kick).as_rotvec())
        array[i] = edm_modify(mdm_rotation_even, edm_kick) # rotations
        array[i+1] = edm_modify(mdm_rotation_odd, edm_kick)
        ###print('mdm+edm rotation (even)', array[i].as_rotvec())
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
    print(to_sec*(np.abs(res_direct.as_rotvec()) - np.abs(res_reverse.as_rotvec())), ' [rad/sec]')
    print('sum Wx')
    print(to_sec*(res_direct.as_rotvec()[0] + res_reverse.as_rotvec()[0]), ' [rad/sec]')
    print(' ')

    if not reduced:
        print('spin frequency and n-bar axis (normalized)')
        print('direct (CW)')
        a, vec = normalize(res_direct.as_rotvec())
        print(a*to_sec, ' [rad/s]'); print(vec)
        print('reverse (CCW)')
        a, vec = normalize(res_reverse.as_rotvec())
        print(a*to_sec, ' [rad/s]'); print(vec)
        print(' ')

def test_nbar_response(edm_kick_array, method, elem_num):
    N = edm_kick_array.shape[0]
    nbar_deviation_angle_array = np.zeros(N)
    for i, edm_kick_per_element in enumerate(edm_kick_array):
        print('kick', edm_kick_per_element)
        ideal_lattice = fill(method, np.zeros(elem_num), edm_kick_per_element) # introduce edm_kick into ideal lattice
        vec = multiply(ideal_lattice).as_rotvec()
        a, nbar = normalize(vec)
        print(a, nbar)
        nbar_deviation_angle_array[i] = nbar[0]
    return nbar_deviation_angle_array
        
if __name__ == '__main__':
    model = 'quasi FS'
    print(model+' model'); method = ntilt
    print(' ')

    N_elem = 4; edm_kick_per_element = 1e-6

    total_ring_time = 1/turns_per_sec # seconds per turn
    element_pass_time = total_ring_time/N_elem
    schedule = np.arange(1,N_elem+1)*element_pass_time # seconds/segment

    mdm_roter_ideal  = edm_modify(method(pi/16, 0),    edm_kick_per_element) # for reference
    mdm_roter_tilted = edm_modify(method(pi/16, 3e-5), edm_kick_per_element)

    print('investigation of nbar response to edm-presence (ideal lattice)')
    edm_kick_spectrum = np.linspace(5e-3, 5e-2, 10)
    deviation_angle_array = test_nbar_response(edm_kick_spectrum, method, N_elem)
    fig, ax = plt.subplots(1,1)
    ax.plot(edm_kick_spectrum, deviation_angle_array, '.')
    ax.set_xlabel('edm kick per element [rad]')
    ax.set_ylabel(r'$\bar n_x,$ [rad]')
    ax.ticklabel_format(style='sci', scilimits=(0,0), axis='both')

    if False:
        print('investigation (accumulation)')
    
        tilt_array = np.random.normal(0, 1e-4, N_elem)
        tilt_mean = tilt_array.mean(); tilt_std = tilt_array.std()
    
        element_array_ideal_no_edm    = fill(method, np.zeros(N_elem),                     0)
        element_array_ideal_with_edm  = fill(method, np.zeros(N_elem),  edm_kick_per_element)
        element_array_tilted_no_edm   = fill(method, tilt_array,                           0)
        element_array_tilted_with_edm = fill(method, tilt_array,        edm_kick_per_element)
    
        products_ideal_no_edm    = np.cumprod(element_array_ideal_no_edm)
        products_ideal_with_edm  = np.cumprod(element_array_ideal_with_edm)
        products_tilted_no_edm   = np.cumprod(element_array_tilted_no_edm)
        products_tilted_with_edm = np.cumprod(element_array_tilted_with_edm)

        extract_x_angle   = np.vectorize(lambda product: product.as_rotvec()[0]) # in [rad/segment]
        extract_abs_angle = np.vectorize(lambda product: normalize(product.as_rotvec())[0])

        Wx_array_ideal_no_edm     = extract_x_angle(products_ideal_no_edm)/schedule # in [rad/sec]
        Wx_array_ideal_with_edm   = extract_x_angle(products_ideal_with_edm)/schedule
        Wx_array_tilted_no_edm    = extract_x_angle(products_tilted_no_edm)/schedule
        Wx_array_tilted_with_edm  = extract_x_angle(products_tilted_with_edm)/schedule
        absW_array_ideal_no_edm = extract_abs_angle(products_ideal_no_edm)/schedule
        absW_array_ideal_with_edm = extract_abs_angle(products_ideal_with_edm)/schedule
        absW_array_tilted_no_edm = extract_abs_angle(products_tilted_no_edm)/schedule
        absW_array_tilted_with_edm = extract_abs_angle(products_tilted_with_edm)/schedule

        fig, ax = plt.subplots(1,1)
        ax.set_title(r'{} model of {} elements with {:4.2e} $\pm$ {:4.2e} [rad] tilt'.format(model, N_elem, tilt_mean, tilt_std))
        ax.plot(schedule, Wx_array_tilted_with_edm,   '-.', label=r'$\Omega_x$')
        ax.plot(schedule, absW_array_tilted_with_edm, '-.', label=r'$||\Omega||$')
        ax.plot(schedule, np.abs(Wx_array_tilted_with_edm),   '-.', label=r'$||\Omega_x||$')
        ax.set_xlabel('time within one ring turn [sec]')
        ax.set_ylabel('[rad/sec]')
        ax.ticklabel_format(style='sci', scilimits=(0,0), axis='both')
        ax.legend()
        
        fig, ax = plt.subplots(3,3)
        ax[0,0].plot(Wx_array_ideal_no_edm -  Wx_array_ideal_no_edm, '-.', label='dfference')
        ax[0,0].plot(Wx_array_ideal_no_edm, '-.', label='proper')
        ax[0,1].plot(Wx_array_ideal_no_edm - Wx_array_tilted_no_edm, '-.', label='difference')
        ax[0,1].plot(-Wx_array_tilted_no_edm, '-.', label='-proper')
        ax[0,2].plot(Wx_array_ideal_no_edm - Wx_array_tilted_with_edm, '-.')
        ax[1,0].plot(Wx_array_ideal_with_edm  - Wx_array_ideal_no_edm,  '-.')
        ax[1,1].plot(Wx_array_ideal_with_edm  - Wx_array_tilted_no_edm, '-.', label='difference')
        ax[1,1].plot(-Wx_array_tilted_no_edm, '-.', label='-only tilt')
        ax[1,1].plot(-Wx_array_tilted_with_edm, '-.', label='-both')
        ax[1,2].plot(Wx_array_ideal_with_edm - Wx_array_tilted_with_edm, '-.')
        ax[2,0].plot(Wx_array_tilted_with_edm - Wx_array_ideal_no_edm,  '-.', label='difference')
        ax[2,0].plot(Wx_array_tilted_with_edm,  '-.', label='proper')
        ax[2,1].plot(Wx_array_tilted_with_edm - Wx_array_tilted_no_edm, '-.', label='difference')
        ax[2,1].plot(Wx_array_ideal_with_edm, '-.', label='only edm')
        ax[2,2].plot(Wx_array_tilted_with_edm - Wx_array_tilted_with_edm, '-.')
        ax[0,0].set_title( 'no tilt, no edm'); ax[0,1].set_title('only tilt'); ax[0,2].set_title('both tilt + edm')
        ax[0,0].set_ylabel('no tilt, no edm')
        ax[1,0].set_ylabel('only edm') 
        ax[2,0].set_ylabel('both tilt + edm')  
        ax[0,0].legend(); ax[0,1].legend()
        ax[1,1].legend();
        ax[2,0].legend(); ax[2,1].legend()
        for i in range(3):
            for j in range(3):
                ax[i,j].ticklabel_format(style='sci', scilimits=(0,0), axis='y')
                ax[i,j].grid()

