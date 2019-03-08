import mitsuba
from mitsuba.core import *
from mitsuba.render import *
from mitsuba.core import PluginManager

import os, sys
import multiprocessing
import datetime
import numpy as np
from struct import pack, unpack
import re, time
import scipy.io as sio

from mtspywrapper import *
def sceneLoadFromFile(xml_filename):
    # Get a reference to the thread's file resolver
    fileResolver = Thread.getThread().getFileResolver()

    # Register any searchs path needed to load scene resources (optional)
    fileResolver.appendPath('Myscenes')

    # Load the scene from an XML file
    scene        = SceneHandler.loadScene(fileResolver.resolve(xml_filename), StringMap())    
    return scene
def set_density_from_vol(filename):
    """
    Generates 3D matrix (ndarray) from a binary of .vol type
    Output
      density_mat: 3D matrix of float representing the voxels values of the object
    """     

    fid = open(filename)

    # Reading first 48 bytes of volFileName as header , count begins from zero  
    header = fid.read(48)  

    # Converting header bytes 8-21 to volume size [xsize,ysize,zsize] , type = I : 32 bit integer
    size = unpack(3*'I', bytearray(header[8:20]))

    # Converting data bytes 49-* to a 3D matrix size of [xsize,ysize,zsize], 
    # type = f : 32 bit float   
    binary_data = fid.read()
    nCells = size[0] * size[1] * size[2]
    density_mat = np.array(unpack(nCells*'f', bytearray(binary_data)))
    density_mat = density_mat.reshape(size, order='F')
    fid.close()

    for ax in range(3):
        u_volume, counts =  np.unique(density_mat, axis=ax, return_counts=True)
        if np.all(counts==2):
            density_mat = u_volume
   
    return density_mat 


def transformLookAt(cam_pos, target, up):
    #forward = (target - cam_pos) / np.linalg.norm(target - cam_pos)
    forward = (cam_pos - target) / np.linalg.norm(cam_pos - target)
    right   = np.cross(up, forward)
    right   = right / np.linalg.norm(right)
    newUp   = np.cross(forward, right)

    T         = np.zeros([4,4])
    T[0:3, 0] = right
    T[0:3, 1] = newUp    
    T[0:3, 2] = forward
    T[0:3, 3] = cam_pos
    T[3, 3]   = 1.0
    
    return newUp, T

def get_grad_from_output_file(filename):
    f = open(filename)
    lines = f.readlines()
    vals  = [ re.sub('[\[\],]', '', ' '.join(line.split()[4:7])) for line in lines ]

    grad = np.zeros((len(vals), 3)) # Spectrum, one pixel
    for grid_point in range(len(vals)):
        grad[grid_point] = [float(val) for val in vals[grid_point].split()]

    f.close()
    return grad

def render_scene(scene, output_filename, n_cores, grid_size, n_pixels_w, n_pixels_h):    
    queue = RenderQueue()

    # Create a queue for tracking render jobs
    film   = scene.getFilm()
    size   = film.getSize() 
    bitmap = Bitmap(Bitmap.ELuminance, Bitmap.EFloat32, size) # for radiance
    #bitmap = Bitmap(Bitmap.ERGB, Bitmap.EUInt8, size) # for RGB image
    blocksize = max(np.divide(max(size.x, size.y), n_cores), 1)
    scene.setBlockSize(blocksize) 

    scene.setDestinationFile(output_filename)    

    # Create a render job and insert it into the queue
    job = RenderJob('render', scene, queue)
    job.start()

    # Wait for all jobs to finish and release resources
    queue.waitLeft(0)

    # Develop the camera's film 
    film.develop(Point2i(0, 0), size, Point2i(0, 0), bitmap)

    ## Write to an output file
    #outFile = FileStream('renderedResult.png', FileStream.ETruncReadWrite)
    #bitmap.write(Bitmap.EPNG, outFile)
    #outFile.close()

    radiance   = np.array(bitmap.buffer()) 
    inner_grad = np.zeros([grid_size, 3, n_pixels_w, n_pixels_h])
    for pw in range(n_pixels_w):
        for ph in range(n_pixels_h):
            inner_grad[:, :, pw, ph] = get_grad_from_output_file('output_' + str(pw) + '_' + str(ph)+ '.txt')

    # End session
    queue.join() 

    return radiance, inner_grad



## Parameters
# load density from vol file

## Eshkol's cloud
#beta_gt = set_density_from_vol('/home/tamarl/MitsubaGradient/mitsuba/CloudsSim/eshkol/50CNN_128x128x100_beta_cutted_vol_2_2_2.vol')
#x_spacing = 50 # in meters
#y_spacing = 50 #/ in meters
#z_spacing = 40 # in meters

#[ nx, ny, nz ] = beta_gt.shape
#bounds  = [-nx * x_spacing / 2, -ny * y_spacing / 2, 0, nx * x_spacing / 2, ny * y_spacing / 2, nz * z_spacing]#-250, -250, 0, 250, 250, 80]   # bounding box = [xmin, ymin, zmin, xmax, ymax, zmax] in meters units 

#beta0_diff     = np.array([0.1])#, max_val, 0.1]) 

## load jpl's cloud
jpl_full_cloud = np.load('CloudsSim/jpl/jpl_ext.npy')
beta_gt   = jpl_full_cloud[15:17,17:19, 13:15]
x_spacing = 0.02 # in km
y_spacing = 0.02 # in km
z_spacing = 0.04 # in km

[ nx, ny, nz ] = beta_gt.shape
bounds = [-nx * x_spacing / 2, -ny * y_spacing / 2, 0, nx * x_spacing / 2, ny * y_spacing / 2, nz * z_spacing]   # bounding box = [xmin, ymin, zmin, xmax, ymax, zmax] in km units 

beta0_diff     = np.array([2])#, max_val, 0.1]) 

beta_gt_flat = beta_gt.flatten('F')

# algorithm parameters
max_iterations = 1500
n_unknowns     = np.prod(beta_gt.shape)
max_val        = np.max(beta_gt_flat)
grid_size      = np.prod(beta_gt.shape)

# sensors parameters
n_sensors  = 3
n_pixels_w = 2 #TBD
n_pixels_h = 2 #TBD
n_pixels   = n_pixels_h * n_pixels_w

# optimizer parameters - ADAM
alpha   = 0.01 # randomly select alpha hyperparameter -> r = -a * np.random.rand() ; alpha = 10**(r)                                - randomly selected from a logaritmic scale
beta1   = 0.9  # randomly select beta1 hyperparameter -> sample (1-beta1), r = -a * np.random.uniform(-3, -1) ; beta1 = 1 - 10**(r) - randomly selected from a logaritmic scale
epsilon = 1e-8
beta2   = 0.999

Np_vector     = np.array([512]) * 128#np.array([8192 / 4])
scene_gt      = [ None ] * n_sensors # create an empty list
algo_scene    = [ None ] * n_sensors # create an empty list
sensors_pos   = [ None ] * n_sensors # create an empty list
gt_grad       = np.zeros((n_sensors, grid_size, 3, n_pixels_h, n_pixels_w))
I_gt          = np.zeros((n_sensors, n_pixels_h, n_pixels_w))
I_algo        = np.zeros((n_sensors, len(beta0_diff), max_iterations, n_pixels_h, n_pixels_w))
runtime       = np.zeros((len(beta0_diff), max_iterations))
cost_gradient = np.zeros((len(beta0_diff), max_iterations, grid_size))#, n_pixels_h, n_pixels_w))
betas         = np.zeros((len(beta0_diff), max_iterations, grid_size))

f_multi         = True
output_filename = 'renderedResult'

if f_multi: # Set parallel job or run on 1 cpu only
    n_cores = multiprocessing.cpu_count()
else:
    n_cores = 1

# Start up the scheduling system with one worker per local core
scheduler = Scheduler.getInstance()
for i in range(0, n_cores):
    scheduler.registerWorker(LocalWorker(i, 'wrk%i' % i))
scheduler.start()

import matplotlib.pyplot as plt
additional_str = 'crop cloud'

if n_sensors > 1:
    sensors_s = ' ' + str(n_sensors) + ' sensors'
else:
    sensors_s = ''

TOA      = bounds[5]
H        = z_spacing * 2
t_const  = np.array([0, 0, TOA])
up_const = np.array([-1, 0, 0])

# Ground Truth:
sensors_radius = nx * x_spacing / 2
o = np.zeros((n_sensors, 3))
o[0] = np.array([0, 0, TOA + H]) 
for ss in range(n_sensors - 1):
    theta     = 2 * np.pi / (n_sensors - 1) * ss
    o[ss + 1] = np.array([round(sensors_radius * np.cos(theta), 2), round(sensors_radius * np.sin(theta), 2), TOA + H])             

for ss in range(n_sensors):
    newUp, _        = transformLookAt(o[ss], t_const, up_const)
    sensors_pos[ss] = { 'origin' : Point(o[ss][0],   o[ss][1],   o[ss][2]),
                        'target' : Point(t_const[0], t_const[1], t_const[2]), 
                        'up'     : Vector(newUp[0],  newUp[1],   newUp[2]) }
    
    scene_gt[ss] = pyScene()
    scene_gt[ss].create_new_scene(beta=beta_gt, g=0.85, origin=sensors_pos[ss]['origin'], target=sensors_pos[ss]['target'], 
                                  up=sensors_pos[ss]['up'], nSamples=Np_vector[0]*4*4, sensorType='perspective', bounding_box=bounds, fov_f=True, 
                                  width=n_pixels_w, height=n_pixels_h)

    I_gt[ss], gt_grad[ss] = render_scene(scene_gt[ss]._scene, output_filename, n_cores, grid_size, n_pixels_w, n_pixels_h)

    ##scene        = sceneLoadFromFile(os.path.realpath(scene_gt[ss]._medium._medium_path) + '/scene.xml')
    ##I_gt[ss], gt_grad[ss] = render_scene(scene, output_filename, n_cores, grid_size, n_pixels_w, n_pixels_h)
    
#sio.loadmat('/home/tamarl/MitsubaGradient/mitsuba/crop cloud jpl ground truth grid 8 4 pixels.m.mat')    
sio.savemat('crop cloud jpl ground truth grid 8 4 pixels 1048576 photons.m', {'beta_gt': beta_gt, 'I_gt': I_gt, 'scene_gt': scene_gt, 'gt_grad':gt_grad})

for bb in range(len(beta0_diff)):
    # optimizer parameters - ADAM
    first_moment  = 0 #m0
    second_moment = 0 #v0

    beta0 = beta_gt + np.ones(beta_gt.shape) * beta0_diff[bb]

    # for now beta is not a Spectrum:
    inner_grad_float = np.zeros((n_unknowns, 1))

    # Gradient descent loop
    beta  = np.copy(beta0)
    start = time.time()

    for iteration in range(max_iterations):               
        betas[bb, iteration] = beta.flatten('F')                
        cost_grad            = np.zeros(grid_size)                     

        for ss in range(n_sensors):
            # Create scene with given beta
            algo_scene[ss] = pyScene()
            algo_scene[ss].create_new_scene(beta=beta, g=0.85, origin=sensors_pos[ss]['origin'], target=sensors_pos[ss]['target'], 
                                            up=sensors_pos[ss]['up'], nSamples=Np_vector[0], sensorType='perspective', fov_f=True,
                                            bounding_box=bounds, width=n_pixels_w, height=n_pixels_h)
            
            [ I_algo[ss, bb, iteration], inner_grad ] = render_scene(algo_scene[ss]._scene, output_filename, n_cores, grid_size,
                                                                     n_pixels_w, n_pixels_h)
            
            #scene        = sceneLoadFromFile(os.path.realpath(algo_scene[ss]._medium._medium_path) + '/scene.xml')
            #[ I_algo[ss, bb, iteration], inner_grad ] = render_scene(scene, output_filename, n_cores, grid_size, n_pixels_w, n_pixels_h)            

            ### beta is not a Spectrum, for now:
            inner_grad_float = np.mean(inner_grad, 1)

            tmp        =  (-1) * ( I_algo[ss, bb, iteration] - I_gt[ss] )                    
            cost_grad += np.sum(np.sum(inner_grad_float * tmp, 2), 1)

        cost_gradient[bb, iteration] = cost_grad

        cost_grad_mat = np.reshape(cost_grad, beta.shape, 'F')

        ## ADAM implementation
        first_moment  = beta1 * first_moment  + (1 - beta1) * cost_grad_mat
        second_moment = beta2 * second_moment + (1 - beta2) * cost_grad_mat**2
        #second_moment = beta2 * second_moment + (1 - beta2) * np.power(cost_grad_mat, 2)

        first_moment_bar  = first_moment  / ( 1 - beta1**(iteration + 1) )
        second_moment_bar = second_moment / ( 1 - beta2**(iteration + 1) )

        beta -= alpha * first_moment_bar / (np.sqrt(second_moment_bar) + epsilon)    

        if beta[beta <= 0].size:
            beta_before = beta + alpha * first_moment_bar / (np.sqrt(second_moment_bar) + epsilon)    
            beta[beta <= 0] = beta_before[beta <= 0]
            print('fixed beta!')

        end = time.time()

        runtime[bb, iteration] = end - start 
    
    out_path = '/home/tamarl/MitsubaGradient/Gradient wrapper/small cloud/'
        
    bb0 = beta0_diff[bb]                   
    diff = 0
    for ss in range(n_sensors):
        diff += np.sum(np.sum( I_gt[ss] * np.ones(I_algo[ss, bb].shape) - I_algo[ss, bb], 2 ), 1) ##TBD
            
    cost     = 0.5 * diff**2
    gradient = cost_gradient[bb]
    tmp = np.ones(beta_gt_flat.shape)
    tmp[beta_gt_flat >0] = beta_gt_flat[beta_gt_flat >0]    
    betas_err = (betas[bb] - beta_gt_flat) / tmp * 100 # np.repmat(beta_gt)
    
    out_name = out_path+'_small cloud jpl density grid 8 4 pixels ' + sensors_s + ' dependent grad and fwd Np '+ str(Np_vector[0]) + ' adam beta0 ' + str(bb0)+' photonSpec_F.mat'
    sio.savemat(out_name, {'beta': beta, 'beta0': beta0_diff[bb], 'runtime': runtime, 'gradient': gradient, 
                           'cost': cost, 'betas_err':betas_err, 'I_algo': I_algo[:, bb]})
    
scheduler.stop()

print('end')
