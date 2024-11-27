import numpy as np
from copy import deepcopy
from numpy import pi, cos, sin, tan
from math import ceil
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt; plt.ion()

CLIGHT = 3e8 # in [m/s]

beta = .5    # unitless
gamma = 1/np.sqrt(1-beta**2)

particle_G = -.14 # deuterons
particle_mass = 2*1.67e-27 # in [kg]
particle_charge = 1*1.602e-19 # in [C]
particle_eta = 0*1e-14

spin_tune = gamma*particle_G+1 # laboratory coordinates;    "spin" = "mdm"
edm_tune = particle_eta*beta*gamma/2

LRING = 251
TOF_R = LRING/(CLIGHT*beta)

def normalize(vec):
    norm = np.sqrt(np.sum(np.square(vec)))
    return (norm, vec/norm) if norm!=0 else (0, vec)

def add_edm(mdm_angle, mdm_axis):
    edm_angle = edm_tune/spin_tune*mdm_angle
    beam_axis = np.array([0, 0, 1])
    edm_axis = np.cross(beam_axis, mdm_axis)

    mdm_rotvec = mdm_angle*mdm_axis
    edm_rotvec = edm_angle*edm_axis
    net_rotvec = mdm_rotvec + edm_rotvec
    spin_rotation_angle, axis = normalize(net_rotvec)
    return spin_rotation_angle, axis

class Element:
    def __init__(self, angle, axis, tilt_angle):
        self.__angle_ideal = angle          # in [rad]
        self.__axis_ideal  = np.array(axis)
        self.__tilt_angle  = tilt_angle     # in [rad]
        
        tilter = R.from_rotvec(self.__tilt_angle*np.array([0, 0, 1]))
        self.__axis = tilter.apply(self.__axis_ideal)
        self.__angle = angle/cos(tilt_angle) # angle amplfied to keep By magnitude <-> c.o. (invariant)

        mirror_axis = np.array([-1,1,1])*self.__axis

        self.__rotator_ideal  = R.from_rotvec(self.__angle_ideal*self.__axis_ideal)
        self.__rotator_active = R.from_rotvec(self.__angle      *self.__axis)
        self.__mirror_rotator = R.from_rotvec(self.__angle      *mirror_axis)

        self._quat = self.__rotator_active._quat # apparently needed for np.cumprod
        self._single = self.__rotator_active._single

    @classmethod
    def from_rotator(cls, rotator):
        vector = rotator.as_rotvec()
        angle = np.sqrt(np.sum(vector**2))
        axis = vector/angle if angle!=0 else np.array([0,0,0])
        return cls(angle, axis, 0)
    @classmethod
    def from_rotvec(cls, rotvec):
        angle, axis = normalize(rotvec)
        return cls(angle, axis, 0)
    @classmethod
    def difference(cls, one, other):
        difference_vector = one.as_rotvec() - other.as_rotvec()
        return np.sqrt(np.sum(np.square(difference_vector)))
    @classmethod
    def abs_difference(cls, one, other):
        abs_difference_vector = np.abs(one.as_rotvec()) - np.abs(other.as_rotvec())
        return np.sqrt(np.sum(np.square(abs_difference_vector)))

    @property
    def axis(self):
        return self.__axis
    @property
    def angle(self):
        return self.__angle
    @property
    def rotator(self):
        return self.__rotator_active
    @property
    def ideal(self):
        return self.__rotator_ideal
    @property
    def mimage(self):
        return self.__mirror_rotator
    
    def as_rotvec(self):
        return self.__rotator_active.as_rotvec()
    def __mul__(self, other):
        return Element.from_rotator(self.__rotator_active * other.__rotator_active)
    def act(self, vector):
        return self.__rotator_active.apply(vector)

class Bend(Element): ## pure magnetic element, both EDM- & MDM-actions
    def __init__(self, bend_angle, tilt_angle):
        mdm_angle = (spin_tune-1)*bend_angle # center-of-mass coordinates, hence -1
        mdm_axis = np.array([0, 1, 0])
        
        spin_rotation_angle, axis = add_edm(mdm_angle, mdm_axis) # for bend's ideal
        
        super().__init__(spin_rotation_angle, axis, tilt_angle)
        
class Wien(Element): ## no EDM-action (Lorentz force zero)
    def __init__(self, spin_correction_angle, tilt_angle):
        super().__init__(spin_correction_angle, [0, 1, 0], tilt_angle)

class Combined(Element): ## no MDM-action (by design)
    def __init__(self, bend_angle, tilt_angle, tilt_rule='whole'):
        spin_rotation_angle = bend_angle*gamma*beta * particle_eta/2
        if tilt_rule!='whole': # E & B are not perfectly aligned w/in element <=> exists uncompensated radial MDM
            spin_rotation_angle += (spin_tune-1)*bend_angle * tan(tilt_angle)
            
        super().__init__(spin_rotation_angle, [1, 0, 0], tilt_angle)
    
class Quad(Element): # linear approximation!  ## pure magnetic element, both EDM- & MDM-actions
    def __init__(self, xshift, yshift, grad, length=1):
        deviation = np.sqrt(xshift**2 + yshift**2)
        restoring_force = deviation*grad  # G [T/m] * d [m] = B [T]
        force_axis = np.array([xshift, yshift, 0])
        if deviation!=0:
            force_axis /= deviation # normalize
        
        v = beta*CLIGHT # in [m/s]
        tof = length/v  # in [s]
        bend_angle = tof*restoring_force/gamma*particle_charge/particle_mass
        mdm_angle = (spin_tune-1)*bend_angle
            
        spin_rotation_angle, axis = add_edm(mdm_angle, force_axis)

        super().__init__(spin_rotation_angle, axis, 0)
        

class RotatorField(np.ndarray):
    def __new__(cls, size):
        obj = super().__new__(cls, (size,3))
        return obj
    def plot(self):
        pass
    
    @property
    def x(self):
        return self[:,0]
    @property
    def y(self):
        return self[:1]
    @property
    def z(self):
        return self[:2]
    
class Lattice:
    def __init__(self, element_array):
        self.__direct_array = element_array # list of Elements
        self.__reverse_array = [Element.from_rotator(e.mimage) for e in element_array[::-1]] # list of Elements
        self.__cumprod = np.cumprod(element_array)
        self.__revprod = np.cumprod(self.__reverse_array)

        ## compute n-field
        N = len(element_array)
        self.__nfield = RotatorField(N)
        runner = deepcopy(element_array)
        for p in range(N-1):
            rvec = np.cumprod(runner)[-1].as_rotvec()
            self.__nfield[p] = rvec#[0], *rvec[1:]
            e0 = runner[0]; runner.pop(0); runner += [e0] # shift by one element
        rvec = np.cumprod(runner)[-1].as_rotvec()
        self.__nfield[p+1] = rvec#[0], *rvec[1:]

    def __getitem__(self, key):
        return self.__direct_array[key]
    
    @property
    def direct(self):
        return self.__cumprod[-1]
    @property
    def reverse(self):
        return self.__revprod[-1]
    @property
    def nfield(self):
        return self.__nfield

def track_co(svec, element, nturns, npnts=5000): # this is only valid for c.o. tracking
    streak_power = ceil(nturns/npnts)
    streak = np.cumprod(np.repeat(element, streak_power))
    streak = streak[-1]
    
    output = np.zeros(npnts+1, dtype = [('x', float), ('y', float), ('z', float), ('n', int)])
    svec_runing = deepcopy(svec)
    for n in range(npnts+1):
        output[n] = svec_runing[0], *svec_runing[1:], n*streak_power
        svec_runing = streak.act(svec_runing)
    output[n] = svec_runing[0], *svec_runing[1:], n*streak_power
    return output


if __name__ == '__main__':
    lattice_periodicity = 8
    bend_angle = 2*pi/lattice_periodicity

    q = Quad(3e-4, 1e-3, 4) # testing elements
    b = Bend(bend_angle, -1e-4)
    
    tilts   = np.random.normal(0, 1e-4, lattice_periodicity*2)
    xshifts = np.random.normal(0, 1e-3, lattice_periodicity)
    yshifts = np.random.normal(0, 1e-3, lattice_periodicity)

    bend_array = [Bend( bend_angle, tilts[i*2])                 for i in range(lattice_periodicity)]
    wien_array = [Wien(-bend_angle*(spin_tune-1), tilts[i*2+1]) for i in range(lattice_periodicity)]
    quad_array = [Quad(xshifts[i], yshifts[i], .4)              for i in range(lattice_periodicity)]
    free_array = [Element.from_rotvec([0,0,0])                  for i in range(lattice_periodicity)]
    comb_array = [Combined(bend_angle, tilts[i], 'partial')     for i in range(lattice_periodicity*2)]

    # fiirst way to define, most abstractly
    element_array1 = [Element((spin_tune-1)*bend_angle*(-1)**i, [0,1,0], tilts[i]) for i in range(lattice_periodicity*2)]
    reverse_array1 = [Element.from_rotator(e.mimage) for e in element_array1[::-1]] # for testing
    l10 = Lattice(element_array1)
    # a more concrete way to defne a lattice array
    element_array2  = list(np.array([bend_array, wien_array, quad_array]).flatten('F'))
    element_array20 = list(np.array([bend_array, wien_array, free_array]).flatten('F')) # for testing, no quads
    element_array1 = np.array(element_array1); element_array1.shape=(lattice_periodicity, 2) # for testing l2 - l1
    quad_array = np.array(quad_array)
    element_array1 = list(np.column_stack([element_array1, quad_array]).flatten('C'))

    l1  = Lattice(element_array1)
    l2  = Lattice(element_array2)
    l20 = Lattice(element_array20)
    l_fs = Lattice(comb_array)

    s = np.array([0, 0, 1]) # longitudinal spin-vector
    tracking = track_co(s, l_fs.direct, 300000, 5000)
    plt.plot(tracking['n']*TOF_R, tracking['y'])
    plt.xlabel('time [s]'); plt.ylabel('s_y')
    plt.ticklabel_format(style='sci', scilimits=(0,0), axis='both')

