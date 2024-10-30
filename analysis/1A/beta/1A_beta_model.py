import numpy as np
from numpy import pi, cos, sin, sqrt
from scipy.spatial.transform import Rotation as R

ntotal = lambda omega, nx, ny, k: omega*np.array([nx, ny, sqrt(1 - nx**2 - ny**2)]) 
