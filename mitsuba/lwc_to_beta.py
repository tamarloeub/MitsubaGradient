import numpy as np
import scipy.io as sio

data = open("/home/tamarl/MitsubaGradient/mitsuba/CloudsSim/jpl/cloud_field/rico20sn1024x1024x65.txt", 'r')
lines = data.readlines()

nx = int(lines[3].split()[0])
ny = int(lines[3].split()[1])
nz = int(lines[3].split()[2])
medium = np.zeros((nx, ny, nz))
lwc    = np.zeros((nx, ny, nz))
r_eff  = np.zeros((nx, ny, nz))

#btmp = lines[4].split()
#bounds = [btmp[0], btmp[1], btmp[2], ]
lines = lines[5:]
for line in lines:
    line_array = line.split()
    inds = np.array(line_array[:3], dtype=int)
    vals = np.array(line_array[3:], dtype=float)
    lwc[ inds[0] - 1,    inds[1] - 1, inds[2] - 1 ] = vals[0]
    r_eff[ inds[0] - 1,  inds[1] - 1, inds[2] - 1 ] = vals[1]
    medium[ inds[0] - 1, inds[1] - 1, inds[2] - 1 ] = 1.5 / vals[1] * vals[0] * 1000 # in [1/km]

data.close()

sio.savemat("/home/tamarl/MitsubaGradient/mitsuba/CloudsSim/jpl/cloud_field/rico20sn1024x1024x65_approximated_beta_km.mat", {'beta' : medium})