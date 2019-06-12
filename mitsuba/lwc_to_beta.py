import numpy as np

data = open("/home/tamarl/MitsubaGradient/mitsuba/CloudsSim/jpl/cloud_field/rico20sn1024x1024x65.txt", 'r')
lines = data.readlines()

nx = lines[3].split()[0]
ny = lines[3].split()[1]
nz = lines[3].split()[2]
medium = np.zeros((nx, ny, nz))
lwc    = np.zeros((nx, ny, nz))
r_eff  = np.zeros((nx, ny, nz))

#btmp = lines[4].split()
#bounds = [btmp[0], btmp[1], btmp[2], ]

for line in lines:
    line_array = line.split()
    lwc[ line_array[0], line_array[1], line_array[2] ]    = line_array[3]
    r_eff[ line_array[0], line_array[1], line_array[2] ]  = line_array[4]
    medium[ line_array[0], line_array[1], line_array[2] ] = 1.5 / r_eff * line_array[3] * 1000 # in []1/km]

data.close()
