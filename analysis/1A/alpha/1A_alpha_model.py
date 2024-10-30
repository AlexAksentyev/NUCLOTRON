import numpy as np
from numpy import pi, cos, sin
from scipy.spatial.transform import Rotation as R

def normalize(vec):
    norm = np.sqrt(np.sum(vec**2))
    print(norm*1e6, ' [rad/s]')
    return vec/norm

# for QFS
print('QFS')
print(' ')

n = lambda phi,theta: phi/cos(theta)*np.array([sin(theta), cos(theta), 0])

tilt = np.random.normal(0,1e-4,80)
tilt -= tilt.mean() # make strict zero mean tilt distribution

s = np.array([0,0,1]) # spin vector

element_array = np.zeros(80, dtype=object)

# for direct, positive +tilt
for i in range(0,80,2):
    element_array[i] = R.from_rotvec(n(+pi/40, tilt[i]))
for i in range(1,80,2):
    element_array[i] = R.from_rotvec(n(-pi/40, tilt[i]))
res_direct_qfs  = np.cumprod(element_array)[-1]

#for reverse -- negative -tilt
for i in range(0,80,2):
    element_array[i] = R.from_rotvec(n(+pi/40, -tilt[i]))
for i in range(1,80,2):
    element_array[i] = R.from_rotvec(n(-pi/40, -tilt[i]))
res_reverse_qfs = np.cumprod(element_array[::-1])[-1]

print('typical even (+pi/40 about Y) element axis-angle representation')
print(element_array[14].as_rotvec())
print('typical odd (-pi/40 about Y) element axis-angle representation')
print(element_array[73].as_rotvec())
print(' ')

print('direct (CW) QFS n-bar axis')
print(res_direct_qfs.as_rotvec())
print('reverse (CCW) QFS n-bar axis')
print(res_reverse_qfs.as_rotvec())
print(' ')

print('action on spin = <0,0,1>, direct')
print(res_direct_qfs.apply(s))
print('reverse')
print(res_reverse_qfs.apply(s))
print(' ')

# SFS
print('FS')
print(' ')

rk = lambda theta: theta*np.array([1, 0, 0]) # radial kick

kick_array = np.zeros(80, dtype=object)

# direct : positive +tlit
for i in range(80):
    kick_array[i] = R.from_rotvec(rk(tilt[i]+1e-5)) # add a non-zero mean kick 1e-5
res_direct_fs = np.cumprod(kick_array)[-1]
# reverse : negative -tilt
for i in range(80):
    kick_array[i] = R.from_rotvec(rk(-tilt[i]-1e-5)) # add a non-zero mean kick -1e-5 (negative)
res_reverse_fs = np.cumprod(kick_array[::-1])[-1]

print('typical kick axis-angle representation, corresponding tilt')
print(kick_array[14].as_rotvec(), tilt[14])
print('same for another kick')
print(kick_array[73].as_rotvec(), tilt[73])
print(' ')

print('direct (CW) FS n-bar axis')
print(res_direct_fs.as_rotvec())
print('reverse (CCW) FS n-bar axis')
print(res_reverse_fs.as_rotvec())
print(' ')

print('action on spin = <0,0,1>, direct')
print(res_direct_fs.apply(s))
print('reverse')
print(res_reverse_fs.apply(s))
print(' ')

# consequent
print('Rx*Ry model')
print(' ')

tn =  lambda phi: phi*np.array([0, 1, 0]) # vertical bend
pair = lambda phi, theta: R.from_rotvec(rk(theta))*R.from_rotvec(tn(phi))

pair_array = np.zeros(80, dtype=object)

# for direct : positive +tlit
for i in range(0,80,2):
    pair_array[i] = pair(+pi/40, tilt[i])
for i in range(1,80,2):
    pair_array[i] = pair(-pi/40, tilt[i])
res_direct_pr = np.cumprod(pair_array)[-1]
# for reverse : negative -tilt
for i in range(0,80,2):
    pair_array[i] = pair(+pi/40, -tilt[i])
for i in range(1,80,2):
    pair_array[i] = pair(-pi/40, -tilt[i])
res_reverse_pr = np.cumprod(pair_array[::-1])[-1]

print('typical even (+pi/40 about Y) PAIR axis-angle representation')
print(pair_array[14].as_rotvec())
print('typical odd (-pi/40 about Y) PAIR axis-angle representation')
print(pair_array[73].as_rotvec())
print(' ')

print('direct (CW) PAIR n-bar axis')
print(res_direct_pr.as_rotvec())
print('reverse (CCW) PAIR n-bar axis')
print(res_reverse_pr.as_rotvec())
print(' ')

print('action on spin = <0,0,1>, direct')
print(res_direct_pr.apply(s))
print('reverse')
print(res_reverse_pr.apply(s))
print(' ')
