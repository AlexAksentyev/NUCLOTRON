from numpy import pi, sqrt, linspace, zeros, array, random
from model_base import Element, Lattice, add_edm, normalize
import matplotlib.pyplot as plt; plt.ion()

to_sec = 1e6 # turns angle rad/turn to frequency rad/sec

beta = .5    # unitless
gamma = 1/sqrt(1-beta**2)

particle_G = -.14 # deuterons
particle_mass = 2*1.67e-27 # in [kg]
particle_charge = 1*1.602e-19 # in [C]
particle_eta = 1e-14

spin_tune = gamma*particle_G+1 # laboratory coordinates;    "spin" = "mdm"

def set_ideal_element(mdm_angle, mdm_axis, edm_tune):
    tot_angle, tot_axis = add_edm(mdm_angle, array(mdm_axis), edm_tune)
    return Element(tot_angle, tot_axis, 0)


lattice_periodicity = 8
bend_angle = 2*pi/lattice_periodicity
mdm_angle = (spin_tune-1)*bend_angle; mdm_axis = [0,1,0]


edm_eta_spectrum = linspace(-5e-5, 5e-5, 30)
edm_tune_spectrum = edm_eta_spectrum*beta*gamma/2
N = edm_eta_spectrum.shape[0]
direct_av_array  = zeros(N, dtype = [('abs', float), ('x', float), ('y', float), ('z', float)])
reverse_av_array = zeros(N, dtype = [('abs', float), ('x', float), ('y', float), ('z', float)])
for i, edm_tune in enumerate(edm_tune_spectrum):
    print(edm_tune)
    ideal_element_array = [set_ideal_element(mdm_angle*(-1)**i, mdm_axis, edm_tune*(i%2))
                           for i in range(lattice_periodicity*2)]
    ideal_lattice  = Lattice(ideal_element_array)
    print('lattice character')
    print(ideal_lattice.direct.as_rotvec())
    l_direct_vec  = ideal_lattice.direct.as_rotvec() # turn to vec
    l_reverse_vec = ideal_lattice.reverse.as_rotvec()
    a_dir, nbar = normalize(l_direct_vec) # normalze -> absolute value
    a_rev, nbar = normalize(l_reverse_vec)
    direct_av_array[i] = a_dir, l_direct_vec[0], l_direct_vec[1], l_direct_vec[2] # record
    reverse_av_array[i] = a_rev, l_reverse_vec[0], l_reverse_vec[1], l_reverse_vec[2] # in [rad/turn]

fig, ax = plt.subplots(2,2, sharex='col')
ax[0,0].set_title('test: outer (total ring frequency)')
ax[0,1].set_title('ideal lattice')
ax[0,0].plot(edm_tune_spectrum, to_sec*direct_av_array['x'], '-.', label='CW')
ax[0,0].plot(edm_tune_spectrum, to_sec*reverse_av_array['x'], '-.', label='CCW')
ax[1,0].plot(edm_tune_spectrum, to_sec*(direct_av_array['x'] - reverse_av_array['x']), '.')
ax[0,0].set_ylabel(r'$\Omega_x$ [rad/s]'); ax[0,0].legend()
ax[1,0].set_ylabel(r'$\Sigma \Omega_x$ [rad/s]')
    
ax[0,1].plot(edm_tune_spectrum, to_sec*direct_av_array['abs'], '-.', label='CW')
ax[0,1].plot(edm_tune_spectrum, to_sec*reverse_av_array['abs'], '-.', label='CCW')
ax[1,1].plot(edm_tune_spectrum, to_sec*(direct_av_array['abs'] - reverse_av_array['abs']), '.')
ax[0,1].set_ylabel(r'$||\Omega||$ [rad/s]');   ax[0,1].legend()
ax[1,1].set_ylabel(r'$\Delta ||\Omega||$ [rad/s]')
for i in range(2):
    ax[1,i].set_xlabel(r'edm tune')
    for j in range(2):
        ax[i,j].grid()
        ax[i,j].ticklabel_format(style='sci', scilimits=(0,0), useMathText=True, axis='both')

edm_tune_null = 0
ideal_element_array_null = [set_ideal_element(mdm_angle*(-1)**i, mdm_axis, edm_tune_null)
                           for i in range(lattice_periodicity*2)]
ideal_lattice_null  = Lattice(ideal_element_array_null)

def add_edm_lattice_wise(lattice, edm_tune):
    element_plus_array = []
    for element in lattice:
        mdm_angle, mdm_axis = element.angle, element.axis
        tot_angle, tot_axis = add_edm(mdm_angle, array(mdm_axis), edm_tune)
        element_plus_array += Element.from_rotvec(tot_angle*tot_axis)
    return element_plus_array
        
