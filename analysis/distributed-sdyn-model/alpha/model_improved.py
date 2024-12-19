from scipy.spatial.transform import Rotation as R
import numpy as np; from numpy import pi, cos, sin

CLIGHT = 3e8 # in [m/s]

beta = .5    # unitless
gamma = 1/np.sqrt(1-beta**2)

particle_G = -.14 # deuterons
particle_mass = 2*1.67e-27 # in [kg]
particle_charge = 1*1.602e-19 # in [C]
particle_eta = 2e-5

spin_tune = gamma*particle_G+1 # laboratory coordinates;    "spin" = "mdm"
edm_tune = particle_eta*beta*gamma/2

LRING = 251
TOF_R = LRING/(CLIGHT*beta)

def normalize(vec):
    norm = np.sqrt(np.sum(np.square(vec)))
    return (norm, vec/norm) if norm!=0 else (0, vec)

class Rotator:
    def __init__(self, angle, axis):
        self.__angle = angle          # in [rad]
        self.__axis  = np.array(axis)

        self.__rotator  = R.from_rotvec(self.__angle*self.__axis)
        self.__mirror_rotator = None

        self._quat = self.__rotator._quat # apparently needed for np.cumprod
        self._single = self.__rotator._single

    def __repr__(self):
        return str(self.as_rotvec())
    @classmethod
    def from_rotator(cls, rotator):
        vector = rotator.as_rotvec()
        angle = np.sqrt(np.sum(vector**2))
        axis = vector/angle if angle!=0 else np.array([0,0,0])
        return cls(angle, axis)
    @classmethod
    def from_rotvec(cls, rotvec):
        angle, axis = normalize(rotvec)
        return cls(angle, axis)
    @classmethod
    def difference(cls, one, other):
        difference_vector = one.as_rotvec() - other.as_rotvec()
        return np.sqrt(np.sum(np.square(difference_vector)))
    @classmethod
    def abs_difference(cls, one, other):
        abs_difference_vector = np.abs(one.as_rotvec()) - np.abs(other.as_rotvec())
        return np.sqrt(np.sum(np.square(abs_difference_vector)))
    @classmethod
    def perp(cls, rotator, quotient):
        beam_axis  = np.array([0, 0, 1])
        perp_angle = quotient*rotator.angle
        perp_axis = np.cross(beam_axis, rotator.axis)

        return cls(perp_angle, perp_axis)

    @property
    def axis(self):
        return self.__axis
    @property
    def angle(self):
        return self.__angle
    @property
    def rotator(self):
        return self.__rotator
    @rotator.setter
    def rotator(self, value):
        self.__rotator = value
    @property
    def mimage(self):
        return self.__mirror_rotator
    @mimage.setter
    def mimage(self, value):
        self.__mirror_rotator = value
    
    def as_rotvec(self):
        return self.__rotator.as_rotvec()
    def __mul__(self, other):
        return Rotator.from_rotator(self.__rotator * other.__rotator)
    def act(self, vector):
        return self.__rotator.apply(vector)

class MDMRotator(Rotator):
    def __init__(self, angle, axis):
        super().__init__(angle, axis)
        
        mirror_axis = np.array([-1,1,1])*self.axis # mirror symmetry not violated
        mirror_angle = self.angle
        self.mimage = R.from_rotvec(mirror_angle      *mirror_axis)

class EDMRotator(Rotator):
    def __init__(self, angle, axis):
        super().__init__(angle, axis)

        mirror_axis = np.array([1,1,1])*self.axis # mirror symmetry violated
        mirror_angle = self.angle
        self.mimage = R.from_rotvec(mirror_angle      *mirror_axis)

class Element(Rotator):
    def __init__(self, mdm_rotator, edm_rotator):
        self.__mdm_rotator = mdm_rotator
        self.__edm_rotator = edm_rotator
        self.rotator = R.from_rotvec(self.__mdm_rotator.as_rotvec() + self.__edm_rotator.as_rotvec())
        self.mimage  = R.from_rotvec(self.__mdm_rotator.mimage.as_rotvec() + self.__edm_rotator.mimage.as_rotvec())

    @classmethod
    def from_mdm(cls, angle, axis, tilt, quotient_=0):
        tilter = R.from_rotvec(tilt*np.array([0, 0, 1]))
        
        mdm_tilted_angle = angle/cos(tilt) # in [rad]
        mdm_ideal_axis = np.array(axis)
        mdm_tilted_axis  = tilter.apply(mdm_ideal_axis)
        
        mdm_rotator = MDMRotator(mdm_tilted_angle, mdm_tilted_axis)
        edm_rotator = EDMRotator.perp(mdm_rotator, quotient_)
        return cls(mdm_rotator, edm_rotator)

class RotatorField(np.ndarray):
    def __new__(cls, lattice):
        size = lattice.size
        obj = super().__new__(cls, (size, 3))
        runner = lattice.element_array
        for p in range(size-1):
            rvec = np.prod(runner, 0).as_rotvec()
            obj[p] = rvec#[0], *rvec[1:]
            e0 = runner[0]; runner.pop(0); runner += [e0] # shift by one element
        rvec = np.prod(runner, 0).as_rotvec()
        obj[p+1] = rvec#[0], *rvec[1:]
        return obj
    def plot(self):
        pass
    
    @property
    def x(self):
        return self[:,0]
    @property
    def y(self):
        return self[:,1]
    @property
    def z(self):
        return self[:,2]

class Lattice:
    def __init__(self, element_array):
        self.__size = len(element_array)
        self.__direct_array = element_array # list of Elements
        self.__reverse_array = [Rotator.from_rotator(e.mimage) for e in element_array[::-1]] # list of Elements
        self.__dirprod = np.prod(element_array, 0)
        self.__revprod = np.prod(self.__reverse_array, 0)

    def __getitem__(self, key):
        return self.__direct_array[key]
    @property
    def size(self):
        return self.__size
    @property
    def element_array(self):
        return self.__direct_array
    @property
    def direct(self):
        return self.__dirprod
    @property
    def reverse(self):
        return self.__revprod


if __name__ == '__main__':

    lattice_periodicity = 8
    bend_angle = 2*pi/lattice_periodicity

    ## no tilting
    tilts   = np.zeros(lattice_periodicity*2)
    element_array = [Element.from_mdm((spin_tune-1)*bend_angle*(-1)**i, [0,1,0], tilts[i],
                                          edm_tune/spin_tune*(i%2)) # WFs don't generate EDM-effect
                         for i in range(lattice_periodicity*2)]                        
    l_ideal  = Lattice(element_array)

    ## yes tilting

    tilts   = np.random.normal(0, 1e-4, lattice_periodicity*2)
    element_array = [Element.from_mdm((spin_tune-1)*bend_angle*(-1)**i, [0,1,0], tilts[i],
                                          edm_tune/spin_tune*(i%2)) # WFs don't generate EDM-effect
                         for i in range(lattice_periodicity*2)]                        
    l_tilted  = Lattice(element_array)
