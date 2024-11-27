import numpy as np
from numpy import pi, cos, sin
from scipy.spatial.transform import Rotation as R
from scipy.stats import linregress
from pandas import DataFrame
import matplotlib.pyplot as plt; plt.ion()

ntilt = lambda phi, theta: R.from_rotvec(phi/cos(theta)*np.array([sin(theta), cos(theta), 0])) # tilted element's proper axis
rkick = lambda dummy, theta: R.from_rotvec(theta*np.array([1, 0, 0])) # radial kick

N_elem = 4 # total number of elements

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

def base_edm_mode(tilt_array, mdm_method, edm_kick):
    ###print('+tilt')
    array = fill_edm_mod(mdm_method, tilt_array, edm_kick); ###print(' ')
    res_direct = multiply(array) # total direct spin-rotation
    ###print('-tilt')
    array = fill_edm_mod(mdm_method, tilt_rev_sign*tilt_array, edm_kick); ###print(' ')
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

def prep_tilt_distro(mean, sigma, zero_mean=False):
    tilt_array =  np.random.normal(mean, sigma, N_elem)
    factical_sigma = tilt_array.std(); factical_mean = tilt_array.mean()
    if zero_mean:
        tilt_array -= factical_mean; factical_mean = 0 # make strict zero mean tilt distribution
    return tilt_array, factical_mean, factical_sigma

def analyze(x_fit_range, direct_av_array, reverse_av_array):
    delta_av_observation = direct_av_array['x'] + reverse_av_array['x'] # in [rad/turn]
    delta_x_fit_range = x_fit_range[-1] - x_fit_range[0] # I give an ordered, min-to-max edm-kick range; in [rad/turn]
    observation_range = delta_av_observation.max() - delta_av_observation.min() # in [rad/turn]
    ratio_slope = linregress(x_fit_range, delta_av_observation).slope
    ratio_rough = observation_range/delta_x_fit_range
    return ratio_slope, ratio_rough # sentir1 and sentir2
    
def test_distro(tilt_mean_array, edm_kick_spectrum, method):
    N = tilt_mean_array.shape[0]
    effect_array  = np.zeros(N, dtype=[('tilt_sigma', float), ('tilt_mean', float), ('sentir1', float), ('sentir2', float)])
    for i, mean in enumerate(tilt_mean_array):
        tilt_distro, factical_mean, factical_sigma = prep_tilt_distro(mean, 1e-4, zero_mean=False) # tilt_distro not mean-zeroed
        direct_av_array, reverse_av_array = test_outer(tilt_distro, method, edm_kick_spectrum) # in [rad/turn]
        edm_kick_fit_range = edm_kick_spectrum * N_elem             # [rad/elem] * N [elem/turn] = [rad/turn]
        sensitivity_slope, sensitivity_rough = analyze(edm_kick_fit_range, direct_av_array, reverse_av_array)
        effect_array[i] = factical_sigma, factical_mean, sensitivity_slope, sensitivity_rough
    return effect_array
        

if __name__ == '__main__':
    model = 'quasi FS'
    print(model+' model'); method = ntilt
    print(' ')

    edm_kick_spectrum = np.linspace(-5e-5, 5e-5, 10) # [rad/element]
    tilt_sigma_array = np.linspace(1e-5, 1e-3, 100)
    tilt_mean_array = np.linspace(-1e-5, -1e-5, 100)
    effect_array = test_distro(tilt_mean_array, edm_kick_spectrum, method)
    
    effect_df = DataFrame(effect_array)
    effect_df = effect_df.sort_values('tilt_mean')

    sentir1 = effect_array['sentir1']
    reg = linregress(effect_array['tilt_mean'], sentir1) # fit sensitivity on the tilt-kick distribution's mean
    min_sentir1 = reg.intercept # ideal lattice's sensitivity

    fig, ax = plt.subplots(1,2)
    ax[0].set_title('{} model (min sensvitity {:4.2e})'.format(model, min_sentir1))
    ax[1].set_title('[{} elements in lattice]'.format(N_elem))
    ax[0].plot(effect_df['tilt_mean'],  (effect_df['sentir1']), '-.', label='fine')
    ax[0].plot(effect_df['tilt_mean'],  effect_df['sentir2'], '.', label='rough')
    ax[1].plot(effect_df['tilt_sigma'], (effect_df['sentir1']), '-.', label='fine')
    ax[1].plot(effect_df['tilt_sigma'], effect_df['sentir2'], '.', label='rough')
    ax[0].set_xlabel('tilt mean [rad]'); ax[1].set_xlabel('tilt sigma [rad]')
    ax[0].set_ylabel('lattice sensitivity to edm presence estimate')
    for i in range(2):
        ax[i].ticklabel_format(style='sci', scilimits=(0,0), axis='both')
        ax[i].legend(); ax[i].grid()
    
    # checking weird 0 sentir1 when tilt's mean is 0
    tilt_array_case0 = prep_tilt_distro(0, 1e-4, zero_mean=True)[0]
    direct_av_array, reverse_av_array = test_outer(tilt_array_case0, method, edm_kick_spectrum)
    
    fig, ax = plt.subplots(2,2, sharex='col')
    ax[0,0].set_title('test: outer (total {} ring frequency)'.format(model))
    ax[0,1].set_title(r'$tilt = {:4.2e} \pm {:4.2e} [rad]$'.format(tilt_array_case0.mean(), tilt_array_case0.std()))
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

