import numpy as np
from numpy import pi, cos, sin
from scipy.spatial.transform import Rotation as R
from pandas import DataFrame
import matplotlib.pyplot as plt; plt.ion()

ntilt = lambda phi, theta: R.from_rotvec(phi/cos(theta)*np.array([sin(theta), cos(theta), 0])) # tilted element's proper axis
rkick = lambda dummy, theta: R.from_rotvec(theta*np.array([1, 0, 0])) # radial kick

N_elem = 4 # total number of elements
tilt_array = np.zeros(N_elem) #np.random.normal(0, 1e-4, N_elem)
#tilt_array -= tilt_array.mean() # make strict zero mean tilt distribution

tilt_rev_sign = -1
to_sec = 1e6 # turns angle rad/turn to frequency rad/sec

def fill(tilt_array, method=ntilt):
    N = tilt_array.shape[0]
    array = np.zeros(N, dtype=object)
    for i in range(0, N, 2): # even members
        array[i] = method(+pi/N/2, tilt_array[i])
        array[i+1] = method(-pi/N/2, tilt_array[i+1])
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

def base(tilt, method):
    array = fill(tilt, method)
    res_direct = multiply(array) # total direct spin-rotation
    array = fill(tilt_rev_sign*tilt, method)
    res_reverse = multiply(array[::-1]) # total reverse spin-rotation
    return array, res_direct, res_reverse # returned as rotations

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
    

def fill_edm_mod(mdm_method, tilt_array, edm_kick):
    N = tilt_array.shape[0]
    array = np.zeros(N, dtype=object)
    print('edm_kick = ', edm_kick)
    for i in range(0, N, 2): # even members
        mdm_rotation_even = mdm_method(+pi/N/2, tilt_array[i])
        mdm_rotation_odd = mdm_method(-pi/N/2, tilt_array[i+1])
        print('mdm rotation (even)', mdm_rotation_even.as_rotvec())
        print('edm kick', rkick(0, edm_kick).as_rotvec())
        array[i] = edm_modify(mdm_rotation_even, edm_kick) # rotations
        array[i+1] = edm_modify(mdm_rotation_odd, edm_kick)
        print('mdm+edm rotation (even)', array[i].as_rotvec())
    return array

def base_edm_mode(tilt_array, mdm_method, edm_kick):
    print('+tilt')
    array = fill_edm_mod(mdm_method, tilt_array, edm_kick); print(' ')
    res_direct = multiply(array) # total direct spin-rotation
    print('-tilt')
    array = fill_edm_mod(mdm_method, tilt_rev_sign*tilt_array, edm_kick); print(' ')
    res_reverse = multiply(array[::-1]) # total reverse spin-rotation
    return array, res_direct, res_reverse # returned as rotations

def test_inner(mdm_roter, edm_kick_array): # "local" frequency (proper to element) can also be called by "inner"
    N = edm_kick_array.shape[0]
    element_W = np.zeros(N, dtype = [('abs', float), ('x', float), ('y', float), ('z', float)])
    element_nbar = np.zeros(N, dtype= [('x', float), ('y', float), ('z', float)])
    for i, edm_kick in enumerate(edm_kick_array):
        modified_roter_vec = edm_modify(mdm_roter, edm_kick).as_rotvec() # vector [angle/turn * axis]
        a, nbar = normalize(modified_roter_vec)
        element_W[i] = a, modified_roter_vec[0], modified_roter_vec[1], modified_roter_vec[2] # in [rad/turn]
        element_nbar[i] = nbar[0], nbar[1], nbar[2]
    return element_W, element_nbar

def test_outer(tilt_array, method, edm_kick_array): # with respect to the "local" I will refer to the total frequency as "outer"
    N = edm_kick_array.shape[0]
    direct_W_array  = np.zeros(N, dtype = [('abs', float), ('x', float), ('y', float), ('z', float)])
    reverse_W_array = np.zeros(N, dtype = [('abs', float), ('x', float), ('y', float), ('z', float)])
    for i, edm_kick in enumerate(edm_kick_array):
        element_array, res_direct, res_reverse = base_edm_mode(tilt_array, method, edm_kick) # rotations
        res_direct_vec  = res_direct.as_rotvec() # turn to vec
        res_reverse_vec = res_reverse.as_rotvec()
        a_dir, nbar = normalize(res_direct_vec) # normalze -> absolute value
        a_rev, nbar = normalize(res_reverse_vec)
        direct_W_array[i] = a_dir, res_direct_vec[0], res_direct_vec[1], res_direct_vec[2] # record
        reverse_W_array[i] = a_rev, res_reverse_vec[0], res_reverse_vec[1], res_reverse_vec[2]
    return direct_W_array, reverse_W_array  # in [rad/turn]

if __name__ == '__main__':
    model = 'quasi FS'
    print(model+' model'); method = ntilt
    print(' ')

    # null to eyeball
    print('orginal') # no edm
    element_array0, res_direct0, res_reverse0 = base(tilt_array, method)
    output(element_array0, res_direct0, res_reverse0)
    print('modified') # for cross-check use edm_kick = 0
    element_array1, res_direct1, res_reverse1 = base_edm_mode(tilt_array, method, 1e-6) # total rotations
    output(element_array1, res_direct1, res_reverse1)

    ### varying EDM
    edm_kick_spectrum = np.linspace(-5e-5, 5e-5, 1000)

    ## local (element-proper) frequency
    element_tilt = 1e-5 # tilt producing the MDM-caused radial effect
    element_bend = pi/16 # ideally set element's spin-rotation angle
    mdm_roter = method(element_bend, element_tilt)
    
    W_array, nbar_array = test_inner(mdm_roter, edm_kick_spectrum) # W entries in [rad/turn]

    fig_inner, ax_inner = plt. subplots(1,1)
    ax_inner.set_title('test: inner (local frequency)')
    ax_inner.plot(edm_kick_spectrum, W_array['abs'], '.')
    ax_inner.set_xlabel('edm kick'); ax_inner.set_ylabel(r'$\Omega$ [rad/turn]')
    ax_inner.ticklabel_format(style='sci', scilimits=(0,0), useMathText=True, axis='both')
    ax_inner.grid()
    
    ## total (ring-proper) frequency
    direct_av_array, reverse_av_array = test_outer(tilt_array, method, edm_kick_spectrum)

    fig, ax = plt.subplots(2,2, sharex='col')
    ax[0,0].set_title('test: outer (total ring frequency)')
    ax[0,1].set_title(r'ring: {}, $\langle tilt\rangle = {:4.2e} [rad]$'.format(model, tilt_array.mean()))
    ax[0,0].plot(edm_kick_spectrum, to_sec*direct_av_array['x'], '-.', label='CW')
    ax[0,0].plot(edm_kick_spectrum, to_sec*reverse_av_array['x'], '-.', label='CCW')
    ax[1,0].plot(edm_kick_spectrum, to_sec*(direct_av_array['x'] + reverse_av_array['x']), '.')
    ax[0,0].set_ylabel(r'$\Omega_x$ [rad/s]'); ax[0,0].legend()
    ax[1,0].set_ylabel(r'$\Sigma \Omega_x$ [rad/s]')
    
    ax[0,1].plot(edm_kick_spectrum, to_sec*direct_av_array['abs'], '-.', label='CW')
    ax[0,1].plot(edm_kick_spectrum, to_sec*reverse_av_array['abs'], '-.', label='CCW')
    ax[1,1].plot(edm_kick_spectrum, to_sec*(direct_av_array['abs'] - reverse_av_array['abs']), '.')
    ax[0,1].set_ylabel(r'$||\Omega||$ [rad/s]');   ax[0,1].legend()
    ax[1,1].set_ylabel(r'$\Delta ||\Omega||$ [rad/s]')
    for i in range(2):
        ax[1,i].set_xlabel(r'edm kick [angle/turn]')
        for j in range(2):
            ax[i,j].grid()
            ax[i,j].ticklabel_format(style='sci', scilimits=(0,0), useMathText=True, axis='both')


    
    


