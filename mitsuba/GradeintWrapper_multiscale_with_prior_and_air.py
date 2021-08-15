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
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy import ndimage
import scipy
import datetime
import shutil

from mtspywrapper import *


def loadMask(flags, params, n_sensors):
    ## Space curving mask
    fname  = 'mask original ' + str(flags.team) + 'with air ' + flags.crop_s + str(params.nx * params.ny * params.nz) + ' grid points ' 
    fname += str(n_sensors) + ' sensors above the medium 1 cycle '
    if flags.crop_s == '' or flags.crop_s == 'cloud1 ' or flags.crop_s == 'cloud2 ':
        fname += str(np.power(np.max([np.min((params.nx * 2, 100)) , np.min((params.ny * 2, 100)) ]), 2)) + ' pixels.mat'
    else:
        fname += str(np.power(np.max([params.nx, params.ny]), 2)) + ' pixels.mat'
    mask      = sio.loadmat(fname)
    load_mask = mask['mask']
    load_mask[load_mask <= 0.07] = 0
    load_mask[load_mask >  0]    = 1
    
    return load_mask    
    
def selectCloudScene(flags, params, n_sensors):    
    params.x_spacing = 0.02 # in km
    params.y_spacing = 0.02 # in km
        
    if flags.team == 'jpl ':
        params.z_spacing = 0.04 # in km
        flags.out_path = '../Gradient wrapper/jpl/multiscale/'        
        minval         = 2
        
        if flags.crop_s == '':            # aviad's crop
            full_cloud = np.load('CloudsSim/jpl/jpl_ext.npy')
            npad = ((1, 1), (1, 1), (1, 1)) # (n_before, n_after) in each dimension
        
        elif flags.crop_s == 'c2 ':       # aviad's 2nd crop
            full_cloud = sio.loadmat('CloudsSim/jpl/jpl_bigger_cloud.mat')
            full_cloud = full_cloud['beta']
            
            full_cloud[ 0, :, :] = params.padVal
            full_cloud[-1, :, :] = params.padVal
            full_cloud[ :, 0, :] = params.padVal
            full_cloud[ :,-1, :] = params.padVal
            full_cloud[ :, :, 0] = params.padVal
            full_cloud[ :, :,-1] = params.padVal
            npad = ((5, 5), (0, 0), (0, 0)) # (n_before, n_after) in each dimension

        elif flags.crop_s == 'small cf ': # small cloudfield
            full_cloud = sio.loadmat('CloudsSim/jpl/small_cloud_field.mat')
            full_cloud = full_cloud['beta_smallcf']
            
            full_cloud[:, :,  0] = params.padVal
            full_cloud[:, :, -1] = params.padVal
            npad = ((4, 3), (1, 1), (0, 0))  # ((2, 1), (2, 2), (0, 0))
            
        elif flags.crop_s == 'cloud1 ':
            full_cloud = sio.loadmat('CloudsSim/jpl/cloud1.mat')
            full_cloud = full_cloud['beta_cloud1']
            full_cloud = full_cloud[:, :, 12:38] # assume we not to bound this cloud hieght and top
            npad = ((1, 1), (1, 1), (1, 1)) # (n_before, n_after) in each dimension
            
        elif flags.crop_s == 'cloud2 ':
            full_cloud = sio.loadmat('CloudsSim/jpl/cloud2.mat')
            full_cloud = full_cloud['beta_cloud2']
            full_cloud = full_cloud[:, :, 12:47] # assume we not to bound this cloud hieght and top
            npad = ((1, 1), (1, 1), (1, 1)) # (n_before, n_after) in each dimension

        full_cloud  = np.pad(full_cloud, pad_width=npad, mode='constant', constant_values=params.padVal)
        
        [ params.nx, params.ny, params.nz ] = full_cloud.shape
        
        beta_gt = full_cloud
        mask    = loadMask(flags, params, n_sensors)    
        
        if flags.crop_f and (flags.crop_s == '' or flags.crop_s == 'cloud2 '):  
            ns      = 8    
            beta_gt = beta_gt[12:12+ns, 14:14+ns, 10:10+ns]
            mask    = mask[12:12+ns,  14:14+ns, 10:10+ns]
    
        if flags.crop_s == 'cloud2 ': 
            npad    = ((0, 0), (1, 1), (0, 0))  
            beta_gt = np.pad(beta_gt, pad_width=npad, mode='constant', constant_values=params.padVal)
            mask    = np.pad(mask, pad_width=npad, mode='constant', constant_values=0)        
            
        elif flags.enlarged_f and (flags.crop_s == '' or flags.crop_s == 'cloud2 '):
            enlarge_scale  = 3.
            beta_gt        = zoom(beta_gt, enlarge_scale)
            mask           = zoom(mask, enlarge_scale)
            mask[mask < 0] = 0
            beta_gt[beta_gt < 0.01] = params.padVal
    
        else:
            npad    = ((2, 2), (0, 0), (5, 6))
            beta_gt = np.pad(beta_gt, pad_width=npad, mode='constant', constant_values=params.padVal)
            mask    = np.pad(mask, pad_width=npad, mode='constant', constant_values=0)
        
    else:    
        params.z_spacing = 0.02 # in km
        flags.out_path  = '../Gradient wrapper/Eshkol/multiscale/'        
        
        ## Eshkol's cloud
        # beta_gt = set_density_from_vol('/home/tamarl/MitsubaGradient/mitsuba/CloudsSim/eshkol/50CNN_128x128x100_beta_cutted_vol_2_2_2.vol')
        # x_spacing = 50 # in meters
        # y_spacing = 50 #/ in meters
        # z_spacing = 40 # in meters
        
        # [ nx, ny, nz ] = beta_gt.shape
        # bounds  = [-nx * x_spacing / 2, -ny * y_spacing / 2, 0, nx * x_spacing / 2, ny * y_spacing / 2, nz * z_spacing]#-250, -250, 0, 250, 250, 80]   # bounding box = [xmin, ymin, zmin, xmax, ymax, zmax] in meters units
        
        ## single cloud
        # full_cloud = sio.loadmat('CloudsSim/eshkol/Beta_BOMEX_1CLD_512x512x320_500CCN_10m_7dT0.1_0000003240_pdfVARS2.mat')
        ## croping the cloud so it will have even voxels in each axis and to cut off all the zeros voxels
        ##full_cloud = full_cloud['beta'][9:, 11:, 48:168] # for eshkol's single cloud snapshot 3600
        # full_cloud = full_cloud['beta'][13:,13:95,49:129] # for eshkol's single cloud snapshot 3240
        ## downsample the cloud to have less voxels
        # full_cloud = zoom(full_cloud, (.5, .5, .5))
        # flags.crop_s     = 'singlecloud '
        
        # minval = 3.4621

    beta_gt[beta_gt < minval] = params.padVal # zeros set small density values
    beta_gt[mask == 0]        = params.padVal # making the mask a binary map
    
    # bounding box = [xmin, ymin, zmin, xmax, ymax, zmax] in km units
    [ params.nx, params.ny, params.nz ] = beta_gt.shape        
    bounds = [0, 0, 0, params.nx * params.x_spacing, params.ny * params.y_spacing, params.nz * params.z_spacing] # bounds = [-nx * x_spacing / 2, -ny * y_spacing / 2, 0, nx * x_spacing / 2, ny * y_spacing / 2, nz * z_spacing]
    
    return beta_gt, mask, bounds


def updateBetaGTSize(flags, params, beta_gt, mask, npad):
    beta_gt = np.pad(beta_gt, pad_width=npad, mode='constant', constant_values=params.padVal)
    mask    = np.pad(mask, pad_width=npad, mode='constant', constant_values=0)
    
    beta_gt[mask == 0] = params.padVal
    
    [ params.nx, params.ny, params.nz ] = beta_gt.shape     
    beta_gt_flat   = beta_gt.flatten('F')
    
    grid_ms, res_ms = setGridMultiscale(flags, params)

    return beta_gt, mask, beta_gt_flat, grid_ms, res_ms

def updateBetaGTValue(params, air_params, beta_gt):
    params.beta_a = 1.09e-3 * air_params.H_air * np.power(air_params.wavelength * 1e-3, -4.05) \
                                                              * (1 - np.exp(-air_params.TOA / air_params.H_air)) / air_params.TOA

    beta_gt      += params.beta_a
    beta_gt_flat  = beta_gt.flatten('F')
    
    return beta_gt, beta_gt_flat


def setGridMultiscale(flags, params):
    #fractal_dim = 2 - 0.4
    ## grid_ms  = nx / np.round(np.array([ 5., 5.*(fractal_dim), 5.*(fractal_dim)**2, 5.*(fractal_dim)**3, float(nx) ]))
    if flags.crop_s == 'cloud1 ' or flags.crop_s == '':
        res_ms = np.round(np.array([4., 4. * np.sqrt(2), 4. * np.sqrt(2) ** 2, 4. * np.sqrt(2) ** 3, 4. * np.sqrt(2) ** 4,
                                    4. * np.sqrt(2) ** 5, 4. * np.sqrt(2) ** 6, float(params.nx)]))
    
    #elif flags.crop_s == '':
    #    res_ms = np.round(np.array([4 * 2, 4 * 3, 4 * 4, 4 * 5, 4 * 6, 4 * 7, 4 * 8, float(params.nx)]))
    
    elif flags.crop_s == 'cloud2 ': # or  flags.crop_s == '':
        res_ms = np.round(np.array([4, 4 * 2, 4 * 2**2, 4 * 2**3, float(params.nx)]))

    else:
        res_ms = np.round(np.array([4, 4 * 2, 4 * 2**2, 4 * 2**3, 4 * 2**4, float(params.nx)]))
        
    params.n_stages  = len(res_ms)
    
    grid_ms        = np.zeros((params.n_stages, 3))
    grid_ms[:, 0]  = params.nx / res_ms
    res_ms_tmp     = np.copy(res_ms)
    res_ms_tmp[-1] = float(params.ny)
    grid_ms[:, 1]  = params.ny / res_ms_tmp
    res_ms_tmp[-1] = float(params.nz)
    grid_ms[:, 2]  = params.nz / res_ms_tmp
    
    return grid_ms, res_ms

def setSchedulerWorkers(flags):
    if flags.parallel_f:  # Set parallel job or run on 1 cpu only
        flags.n_cores = multiprocessing.cpu_count() # T if others are running
    else:
        flags.n_cores = 1
    
    # Start up the scheduling system with one worker per local core
    scheduler = Scheduler.getInstance()
    for i in range(0, flags.n_cores):
        scheduler.registerWorker(LocalWorker(i, 'wrk%i' % i))
    
    return scheduler    

def setSensorsPositionsAndFOV(flags, params, bounds, n_sensors):
    params.TOA = bounds[5]
    up_const   = np.array([-1, 0, 0])
    
    if flags.team == 'jpl ':
        if flags.crop_s == 'c2 ':
            params.H = params.z_spacing * params.nz / 2.
            t_const  = np.array([bounds[3] / 2., bounds[4] / 2., params.TOA * 3. / 4.])
            
        elif flags.crop_s == 'small cf ':
            params.H = params.z_spacing * params.nz
            t_const  = np.array([bounds[3] / 2., bounds[4] / 2., params.TOA / 4.])
            
        elif flags.crop_s == 'cloud1 ':
            params.H = params.z_spacing * params.nz / 2.
            t_const  = np.array([bounds[3] / 2., bounds[4] / 2., params.TOA * 2. / 3. ])
            
        elif flags.crop_s == 'cloud2 ':
            params.H = params.z_spacing * params.nz * 5 / 12.
            t_const  = np.array([bounds[3] / 2., bounds[4] / 2., params.TOA * 2. / 3. ])    
            
        else:
            params.H = params.z_spacing * params.nz / 4 # 12  
            t_const  = np.array([bounds[3] / 2., bounds[4] / 2., params.TOA * 2. / 3.])
    
    else:    # for eshkol's cloud
        params.H = params.z_spacing * params.nz / 2.
        t_const  = np.array([bounds[3] / 2., bounds[4] / 2., params.TOA])
        
    # Ground Truth:
    o = np.zeros((n_sensors, 3))
    t = np.zeros((n_sensors, 3))
    u = np.zeros((n_sensors, 3))
    
    o[0] = np.array([round(bounds[3], 1) / 2., round(bounds[4], 1) / 2., params.TOA + params.H])
    t[0] = t_const
    u[0] = up_const
    
    ## Setting the formation of the cameras - a dome or a cycle:
    params.n_cycles   = 1
    flags.formation_s = 'dome '  # 'cycle '
    r_toa             = np.linalg.norm(o[0] - t_const)  # creating a Dome and not a Circle
    
    for rr in range(params.n_cycles):
        if flags.crop_s == 'c2 ' or flags.crop_s == 'small cf ':
            sensors_radius = params.y_spacing * (params.ny + 16) / 2
        else:
            sensors_radius = params.y_spacing * (params.ny + 12) / 2
            
        for ss in range(n_sensors - 1):
            theta = 2 * np.pi / (n_sensors - 1) * ss
            o[ss + 1 + rr * 4] = np.array([round(o[0][0] + sensors_radius * np.cos(theta), 2),
                                           round(o[0][1] + sensors_radius * np.sin(theta), 2), params.TOA + params.H])
            
            if (flags.formation_s == 'dome ') and (flags.crop_s is not '') :
                o[ss + 1 + rr * 4][2] = np.sqrt(r_toa ** 2 - (o[ss + 1 + rr * 4][0] - o[0][0]) ** 2 - \
                                                (o[ss + 1 + rr * 4][1] - o[0][1]) ** 2) + t_const[2]
            t[ss + 1 + rr * 4] = t_const
            u[ss + 1 + rr * 4] = up_const
    
    sensors_pos = [None] * n_sensors  # create an empty list
    for ss in range(n_sensors):
        newUp, _ = transformLookAt(o[ss], t[ss], u[ss])
        sensors_pos[ss] = {'origin': Point(o[ss][0],  o[ss][1], o[ss][2]),
                           'target': Point(t[ss][0],  t[ss][1], t[ss][2]),
                           'up'    : Vector(newUp[0], newUp[1], newUp[2])}
    
    # FOV calc: fov is set by axis x
    max_medium    = np.array([bounds[3], bounds[4], bounds[5]])
    min_medium    = np.array([bounds[0], bounds[1], bounds[2]])
    medium_center = (max_medium + min_medium) / 2
    
    L = np.max([max_medium[0] - min_medium[0], max_medium[1] - min_medium[1]]) / 2  # camera's FOV covers the whole medium
    if flags.crop_s == 'c2 ':
        fov_rad = 2 * np.arctan(L / ((params.TOA + params.H) / 3))
    elif flags.crop_s == 'small cf ':
        fov_rad = 2 * np.arctan(L * 1.75 / ((params.TOA + params.H)))
    elif flags.crop_s == 'cloud1 ' or flags.crop_s == 'cloud2 ':
        fov_rad = 2 * np.arctan(L / ((params.TOA + params.H) / 4) )
    else:
        fov_rad = 2 * np.arctan(L / ((params.TOA + params.H) / 4))
    fov_deg = 180 * fov_rad / np.pi
        
    return sensors_pos, fov_deg
    
def sceneLoadFromFile(xml_filename):
    # Get a reference to the thread's file resolver
    fileResolver = Thread.getThread().getFileResolver()

    # Register any searchs path needed to load scene resources (optional)
    fileResolver.appendPath('Myscenes')

    # Load the scene from an XML file
    scene = SceneHandler.loadScene(fileResolver.resolve(xml_filename), StringMap())
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
    size = unpack(3 * 'I', bytearray(header[8:20]))

    # Converting data bytes 49-* to a 3D matrix size of [xsize,ysize,zsize],
    # type = f : 32 bit float
    binary_data = fid.read()
    nCells = size[0] * size[1] * size[2]
    density_mat = np.array(unpack(nCells * 'f', bytearray(binary_data)))
    density_mat = density_mat.reshape(size, order='F')
    fid.close()

    for ax in range(3):
        u_volume, counts = np.unique(density_mat, axis=ax, return_counts=True)
        if np.all(counts == 2):
            density_mat = u_volume

    return density_mat


def transformLookAt(cam_pos, target, up):
    # forward = (target - cam_pos) / np.linalg.norm(target - cam_pos)
    forward = (cam_pos - target) / np.linalg.norm(cam_pos - target)
    right   = np.cross(up, forward)
    right   = right / np.linalg.norm(right)
    newUp   = np.cross(forward, right)

    T         = np.zeros([4, 4])
    T[0:3, 0] = right
    T[0:3, 1] = newUp
    T[0:3, 2] = forward
    T[0:3, 3] = cam_pos
    T[3, 3]   = 1.0

    return newUp, T


def get_grad_from_output_file(filename):
    f = open(filename)
    lines = f.readlines()
    vals = [re.sub('[\[\],]', '', ' '.join(line.split()[4:7])) for line in lines]

    grad = np.zeros((len(vals), 3))  # Spectrum, one pixel
    for grid_point in range(len(vals)):
        grad[grid_point] = [float(val) for val in vals[grid_point].split()]

    f.close()
    return grad


def render_scene(scene, output_filename, n_cores, n_pixels_w, n_pixels_h, Igt=np.array([0])):
    queue = RenderQueue()

    # Create a queue for tracking render jobs
    film   = scene.getFilm()
    size   = film.getSize() 
    bitmap = Bitmap(Bitmap.ELuminance, Bitmap.EFloat32, size) # for radiance
    # bitmap = Bitmap(Bitmap.ERGB, Bitmap.EUInt8, size) # for RGB image
    blocksize = max(np.divide(max(size.x, size.y), n_cores), 1)
    scene.setBlockSize(blocksize)

    scene.setDestinationFile(output_filename)

    # for p in range(n_pixels_h*n_pixels_w):
        # scene.setSpecGt(float(Igt[p]), p)
    if not Igt.all():
        Igt = np.zeros((n_pixels_w, n_pixels_h)).flatten('F')

    scene.setSpecGt(list(Igt))  # , int(n_pixels_h*n_pixels_w))
    # Create a render job and insert it into the queue
    job = RenderJob('render', scene, queue)
    job.start()

    # Wait for all jobs to finish and release resources
    queue.waitLeft(0)
    # End session
    queue.join()

    # Develop the camera's film
    film.develop(Point2i(0, 0), size, Point2i(0, 0), bitmap)

    ## Write to an output file
    # outFile = FileStream('renderedResult.png', FileStream.ETruncReadWrite)
    # bitmap.write(Bitmap.EPNG, outFile)
    # outFile.close()

    radiance = np.array(bitmap.buffer())
    if sum(Igt) == 0:
        inner_grad = None
    else:
        inner_grad = np.array(scene.getTotalGradient())
        # inner_grad = np.reshape(inner_grad_tmp, [grid_size, n_pixels_w * n_pixels_h], 'F')
        # inner_grad = np.reshape(inner_grad_tmp, [grid_size, n_pixels_w, n_pixels_h])
        
    # prob_density_return = scene.getProbDensity()

    return radiance, inner_grad



class Flags():
    def __init__(self):
        self.render_gt_f = False
        self.parallel_f  = True
        self.crop_f      = False
        self.enlarged_f  = False
        self.debug_f     = False 
        
        self.crop_s = ''
        self.formation_s = ''
        
        self.team        = 'jpl '
        self.out_path    = ''
        
        self.n_cores     = 10
        self.output_filename = 'renderedResult'        
        
        self.air_model = False


class Parameters():
    def __init__(self):
        self.x_spacing = 0.02 # km
        self.y_spacing = 0.02 # km
        self.z_spacing = 0.04 # km
        self.nx = 1
        self.ny = 1
        self.nz = 1        
        
        self.TOA = 0
        self.H   = 0
        self.sun_irradiance = 10
        
        self.n_cycles = 1

        self.n_stages = 1
        self.padVal   = 0.01
        self.beta_a   = 0.01

class AirModel():
    def __init__(self):
        self.wavelength = 0    # nm
        self.H_air      = 10.4 # km
        self.TOA        = 0    # km
        self.beta_const = 0
        

## flags and params
flags  = Flags()
params = Parameters()

flags.team   = 'jpl '
flags.crop_s = ''
#flags.render_gt_f = True
flags.air_model = True

gt_out      = ''
air_model_s = ''   
rr_depth    = None
rr_str      = ''
n_sensors   = 9

if flags.air_model:
    params.padVal = 0
    air_model_s = 'and const air model '

if flags.enlarged_f:
    gt_out += 'enlarged '
    
if flags.crop_s == 'cloud2 ':
    rr_depth = 10000
    rr_str = ' rrDepth ' + str(rr_depth)

##--------------------------------------------
##         Loading the Cloud and Mask
##--------------------------------------------
beta_gt, mask, bounds = selectCloudScene(flags, params, n_sensors)
beta_gt_flat = beta_gt.flatten('F')

##--------------------------------------------
##           Prior parameters
##--------------------------------------------
if flags.air_model is False:
    slambda_v = np.array([1])
    mlambda_v = np.array([0.5])
    
elif flags.crop_s == '' or flags.crop_s == 'cloud1 ' or flags.crop_s == 'cloud2 ':
    slambda_v = np.array([0.01]) #, 0.01, 0, 0.01, 0.1, 0.25])
    mlambda_v = np.array([0.1]) #01,   0, 0, 0.01, 0.1, 0.25])
else: 
    slambda_v = np.array([1])
    mlambda_v = np.array([1])    
    
laplacian3d          = np.ones((3, 3, 3)) / 26
laplacian3d[1, 1, 1] = -1

# monotonically increasing by z prior
kz = np.array([[[1, 0, -1], [1, 0, -1], [1, 0, -1]],
               [[1, 0, -1], [1, 0, -1], [1, 0, -1]],
               [[1, 0, -1], [1, 0, -1], [1, 0, -1]]])
kz = kz / np.linalg.norm(kz.flatten(), ord=1)

##--------------------------------------------
##            Algorithm parameters
##--------------------------------------------
max_iterations = 200
beta0_diff     = np.array([2])

runtime     = np.zeros((len(slambda_v), max_iterations))
cost_data   = np.zeros_like(runtime)
cost_mean   = np.zeros_like(runtime)
smooth_term = np.zeros_like(runtime)
monoZ_term  = np.zeros_like(runtime)
mb          = np.zeros_like(runtime)
mass_err    = np.zeros_like(runtime)

gmz_mean = np.zeros_like(runtime)
gs_mean  = np.zeros_like(runtime)
gJ_mean  = np.zeros_like(runtime)

##--------------------------------------------
##           Multiscale parameters
##--------------------------------------------
grid_ms, res_ms = setGridMultiscale(flags, params)
iters_ms = np.zeros(params.n_stages, dtype=int)

##--------------------------------------------
##         Adam Optimizer parameters
##--------------------------------------------
if flags.crop_s == '' or flags.crop_s == 'cloud1 ':# or crop_s == 'cloud2 ':
    alpha = 1      # randomly select alpha hyperparams -> r = -a * np.random.rand() ; alpha = 10**(r) - randomly selected from a logarithmic scale
else:
    alpha = 0.3    # randomly select alpha hyperparams -> r = -a * np.random.rand() ; alpha = 10**(r) - randomly selected from a logarithmic scale

alpha_scale = 1    
beta1       = 0.9  # randomly select beta1 hyperparams -> sample (1-beta1), r = -a * np.random.uniform(-3, -1) ; beta1 = 1 - 10**(r) - randomly
                   # selected from a logarithmic scale
epsilon     = 1e-8
beta2       = 0.999

##--------------------------------------------
##          Scene parameters
##--------------------------------------------
grid_size = np.prod(beta_gt.shape)

if flags.crop_s == '' or flags.crop_s == 'cloud1 ' or flags.crop_s == 'cloud2 ':
    n_pixels_w = np.min([params.nx * 2, 100]) # resolution ~= spacing / 2 = 10 meters
    n_pixels_h = np.min([params.ny * 2, 100])
else:
    n_pixels_w = params.nx # resolution ~= spacing = 20 meters
    n_pixels_h = params.ny 

n_pixels = n_pixels_h * n_pixels_w

Np_vector = np.array([512]) / 4  # * 2
if flags.crop_s == '' or flags.crop_s == 'cloud1 ':
    gt_Np_fac = 8 
else:
    gt_Np_fac = 8 / 2

sensors_pos, fov_deg = setSensorsPositionsAndFOV(flags, params, bounds, n_sensors)

##--------------------------------------------
##                Air Model
##--------------------------------------------
if flags.air_model:
    air_p = AirModel()
    ### update beta to reach the camera with air
    npad = ((0, 0), (0, 0), (0, int(np.floor((params.TOA + params.H - bounds[5]) / params.z_spacing))))
    beta_gt, mask, beta_gt_flat, grid_ms, res_ms = updateBetaGTSize(flags, params, beta_gt, mask, npad)
    bounds[5] = params.z_spacing * (int((params.TOA + params.H) / params.z_spacing) - 1) #params.TOA + params.H
    
    air_p.TOA             = bounds[5] # km
    air_p.wavelength      = 475       # blue channel, in nm
    params.sun_irradiance = 1.98 * 10 # blue channel
    
    beta_gt_cloud = np.copy(beta_gt)
    beta_gt, beta_gt_flat = updateBetaGTValue(params, air_p, beta_gt)
    grid_size = np.prod(beta_gt.shape)

scheduler = setSchedulerWorkers(flags)
scheduler.start()

gt_out += 'ground truth original ' + str(flags.team) + 'cloud with ocean ' + air_model_s + flags.crop_s + str(grid_size)
gt_out += ' grid points ' + str(n_sensors) + ' sensors all above the medium 1 ' + flags.formation_s + str(n_pixels) + ' pixels '
gt_out += str(Np_vector[0] * gt_Np_fac) + ' photons' + rr_str + '.mat'

if (not os.path.isfile(gt_out)) or flags.render_gt_f:
    flags.render_gt_f = True
    I_gt = np.zeros((n_sensors, n_pixels_h, n_pixels_w))
    for ss in range(n_sensors):
        ##scene        = sceneLoadFromFile(os.path.realpath(scene_gt[ss]._medium._medium_path) + '/scene.xml')
        ##I_gt[ss], gt_grad[ss] = render_scene(scene, output_filename, n_cores, grid_size, n_pixels_w, n_pixels_h)        
        scene_gt = pyScene()
        scene_gt.create_new_scene(beta=beta_gt, g=0.85, origin=sensors_pos[ss]['origin'], target=sensors_pos[ss]['target'],
                                  up=sensors_pos[ss]['up'], nSamples=int(Np_vector[0] * gt_Np_fac), sensorType='perspective',
                                  bounding_box=bounds, fov=fov_deg, width=n_pixels_w, height=n_pixels_h, rrDepth=rr_depth, 
                                  irradiance=params.sun_irradiance)

        I_gt[ss], _ = render_scene(scene_gt._scene, flags.output_filename, flags.n_cores, n_pixels_w, n_pixels_h)
        sio.savemat(gt_out, {'beta_gt': beta_gt, 'I_gt': I_gt})
        print(ss)
        
    sio.savemat(gt_out, {'beta_gt': beta_gt, 'I_gt': I_gt})

else:
    gt = sio.loadmat(gt_out)
    I_gt = gt['I_gt']

## ------------------------------------------------------------------------------------------------------- ##
paths = []

for ll in range(len(slambda_v)):
    slambda = slambda_v[ll]
    mlambda = mlambda_v[ll]
    
    add_path_out = ''
    if mlambda > 0.:
        add_path_out += 'monoz/'
    if slambda > 0.:
        add_path_out += 'smooth/'
    if flags.crop_f:
        add_path_out += 'crop/'
    if flags.enlarged_f:
        add_path_out += 'enlarged/'
    add_path_out += datetime.datetime.now().strftime("%y_%m_%d_%H:%M") + '/'
    paths.append(flags.out_path + add_path_out)
    
    if not os.path.exists(flags.out_path + add_path_out):
        os.makedirs(flags.out_path + add_path_out)
        shutil.copyfile(os.path.realpath(__file__), flags.out_path + add_path_out + 'exe_script.py')

    out_name  = 'stages ' + str(params.n_stages) + ' l1 ' + str(slambda) + ' l2 ' + str(mlambda) + ' w ocean '+ air_model_s 
    out_name += flags.crop_s + str(grid_size) + ' grid points ' + str(n_sensors) + ' sensors in a ' + flags.formation_s 
    out_name += str(n_pixels) + ' pixels ' + rr_str + str(Np_vector[0]) + ' photons multiscale 2nd mom SC mask beta0 '

    # Loop
    for bb in range(len(beta0_diff)):
        out_name += str(beta0_diff[bb])

        # optimizer parameters - ADAM
        first_moment    = 0  # m0
        second_moment   = 0  # v0
        last_stage_iter = 0

        # Gradient descent loop
        up_scale  = 0
        stage     = 0
        cost_iter = 0
        cost_window_size = 10 

        #beta = np.ones(np.array(beta_gt.shape / grid_ms[stage], dtype=int)) * beta0_diff[bb]       
        beta_zoom_out = np.ones_like(beta_gt) * beta0_diff[bb]
        if beta_zoom_out[mask == 0].size:
            beta_zoom_out[mask == 0] = params.padVal

        if flags.air_model:
            beta_zoom_out += params.beta_a

        mask_ms = zoom(mask, 1./grid_ms[stage])       
        beta    = zoom(beta_zoom_out, 1./grid_ms[stage])
        [nx_ms, ny_ms, nz_ms] = beta.shape
        
        
        if flags.crop_s == '' or flags.crop_s == 'cloud1 ':
            n_pixels_w = np.min([np.max([nx_ms, ny_ms]) * 2, 100])  # resolution ~= spacing / 2 = 10 meters
            n_pixels_h = np.min([np.max([nx_ms, ny_ms]) * 2, 100])
        else:
            n_pixels_w = np.max([nx_ms, ny_ms]) # resolution ~= spacing / 2 = 10 meters
            n_pixels_h = np.max([nx_ms, ny_ms])
        n_pixels = n_pixels_h * n_pixels_w

        Np_scale  = 2 ** np.count_nonzero(np.mod(np.log2(res_ms),1) == 0)
        prior_w   = 1 #4 ** (-(params.n_stages - 1))
        grid_size = np.prod(beta.shape)

        I_algo  = np.zeros((n_sensors, len(beta0_diff), n_pixels_h, n_pixels_w))
        I_gt_ms = zoom(I_gt, (1, 1 / grid_ms[stage, 0], 1 / grid_ms[stage, 1]))

        min_mb = 0
         
        start = time.time()

        for iteration in range(max_iterations):
            if iteration > 0:
                if stage == params.n_stages - 1:
                    beta_zoom_out = beta
                else:
                    beta_zoom_out = zoom(beta, grid_ms[stage])                    

            mb[ll, iteration]       = np.sum(np.sum(np.sum(abs(beta_zoom_out - beta_gt), 2), 1), 0) / np.sum(beta_gt_flat) * 100
            mass_err[ll, iteration] = (np.sum(abs(beta_zoom_out.flatten('F'))) - np.sum(abs(beta_gt_flat))) / np.sum(beta_gt_flat) * 100

            if flags.debug_f:
                print("iteration = " + str(iteration))
                print("error = " + str(np.round(mb[ll, iteration], 2)))

            if (mb[ll, iteration] < min_mb):
                min_mb = mb[ll, iteration]
                bb0 = beta0_diff[bb]
                sio.savemat(flags.out_path + add_path_out + out_name + ' min_mb.mat',
                            {'beta0_diff': beta0_diff[bb], 'mask': mask, 'beta_gt': beta_gt, 'params': params,# 'air_p': air_p,  
                             'flags': flags,                                               # Scene pre-fixed params
                             'alpha': alpha, 'beta1': beta1, 'beta2': beta2,               # Optimization hyper-params
                             'smooth_lambda': slambda, 'smooth_term': smooth_term[ll],     # Smooth prior
                             'monoz_lambda': mlambda, 'monoZ_term': monoZ_term[ll],        # Monotonically increasing z prior
                             'first_moment': first_moment, 'second_moment': second_moment, # Optimization iters params
                             'stage': stage, 'iters_ms': iters_ms, 
                             'grid_ms': grid_ms, 'alpha_scale': alpha_scale,               # Multiscale
                             'cost_data': cost_data[ll], 'cost_mean': cost_mean[ll], 'I_algo': I_algo, 
                             'runtime': runtime[ll], 'iteration': iteration, 'mass_err': mass_err[ll], 'mb': mb[ll], 
                             'beta': beta})                                                # algorithm calculated variables

            cost_grad = np.zeros(grid_size)
            cost_iter = 0
            for ss in range(n_sensors):
                I_gt_zoomed = zoom(I_gt[ss], n_pixels_h / np.array(I_gt[ss].shape, dtype=float))
                algo_scene  = pyScene()
                algo_scene.create_new_scene(beta=np.copy(beta), origin=sensors_pos[ss]['origin'], target=sensors_pos[ss]['target'],  ##MAYBE NOT COPT BETA?!?
                                            up=sensors_pos[ss]['up'], nSamples=int(Np_vector[0] * Np_scale), g=0.85, fov=fov_deg,
                                            sensorType='perspective', bounding_box=bounds, width=n_pixels_w, height=n_pixels_h,
                                            rrDepth=rr_depth, irradiance=params.sun_irradiance)

                [ I_algo[ss, bb], inner_grad ] = render_scene(algo_scene._scene, flags.output_filename, flags.n_cores, n_pixels_w,
                                                              n_pixels_h, I_gt_zoomed.flatten('F'))
                # remove the scene's dir
                if os.path.exists(os.path.realpath(algo_scene._medium._medium_path)):
                    shutil.rmtree(os.path.realpath(algo_scene._medium._medium_path))

                cost_grad += inner_grad
                cost_iter += np.linalg.norm(I_gt_zoomed - I_algo[ss, bb], ord=2)

            cost_grad_mat = np.zeros_like(beta)

            ## Beta cloud prior calculations - beta_c
            if flags.air_model:
                beta_c = np.copy(beta) - np.ones_like(beta) * params.beta_a
            else:
                tmp = np.zeros_like(beta)
                if stage == params.n_stages - 1:
                    tmp[ mask == 0 ]    = params.beta_a
                else:
                    tmp[ mask_ms <= 0.07 ] = params.beta_a
                beta_c = np.copy(beta) - tmp
            beta_c[ beta_c < 0 ] = 0
                
            ## smooth prior
            smooth_term[ll, iteration] = np.linalg.norm(ndimage.convolve(beta_c, laplacian3d, mode='nearest'))

            ## monotonically increasing in z prior
            c1_mz = np.max(beta_c.flatten())
            #c2_mz = 10.
            indic_b = np.tanh(beta_c.flatten('F') / c1_mz)
            indic_b[beta_c.flatten('F') <= .0] = .0
            _, __, Dzb = np.gradient(beta_c)
            Dzbf       = Dzb.flatten('F')
            c2_mz = np.max( np.abs(Dzbf) )
            monoZ_term[ll, iteration] = np.dot(np.transpose(indic_b), np.tanh(Dzbf / c2_mz))

            #if stage >= n_stages - 2:
            smooth_grad = slambda * prior_w * ndimage.convolve(ndimage.convolve(beta_c, laplacian3d, mode='nearest'), laplacian3d,
                                                               mode='nearest')

            dindic_b = np.cosh(beta_c.flatten('F') / c1_mz)**(-2) / c1_mz
            dindic_b[beta_c.flatten('F') <= .0] = .0            ## <=0.01?
            mg1f       = np.dot(np.diag(np.cosh(Dzbf / c2_mz)**(-2)), indic_b) / c2_mz
            _, __, mg1 = np.gradient(np.reshape(mg1f, [nx_ms, ny_ms, nz_ms], 'F'))                
            mgf        = -mlambda * prior_w  * ( mg1.flatten('F') + np.dot(np.diag(dindic_b), np.tanh(Dzbf / c2_mz)) )

            mz_grad = np.reshape(mgf, [nx_ms, ny_ms, nz_ms], 'F')
            
            if stage == params.n_stages - 1:
                smooth_grad[mask == 0] = 0.0
                mz_grad[mask == 0]     = 0.0
            else:            
                smooth_grad[mask_ms <= 0.07] = 0 # 0] = .0
                mz_grad[mask_ms <= 0.07]     = 0 # 0]     = .0            

            gmz_mean[ll, iteration] = np.mean(mz_grad.flatten())
            gs_mean[ll, iteration]  = np.mean(smooth_grad.flatten())
            ## end of prior calculations
            
            cost_grad_mat += smooth_grad + mz_grad

            cost_data[ll, iteration] = .5 * cost_iter
            cost_mean[ll, iteration] = .5 * cost_iter / n_pixels + prior_w * ( .5 * slambda * smooth_term[ll, iteration]
                                                                    - mlambda * monoZ_term[ll, iteration] ) / float(np.prod(beta.shape))
            
            # Multi scale:        
            if ((stage == 0) and (iteration > cost_window_size * 2)) or ((stage > 0) and 
                                                                         (iteration - iters_ms[stage - 1] > cost_window_size * 2)):
                if ( abs(np.mean(cost_mean[ll, iteration - cost_window_size:iteration]) - 
                         np.mean(cost_mean[ll, iteration - cost_window_size * 2:iteration - cost_window_size])) < 0.0005 ): #0.0005 ) ):
                     #and (np.std(cost_mean[bb, iteration - cost_window_size * 2 : iteration]) < 0.002) ):
                    up_scale = cost_window_size + 1
                else:  # cost is saturated (mean) X times in a row
                    up_scale = 0

            if stage == params.n_stages - 1:
                cost_grad[ mask.flatten('F') == 0 ] = 0
            else:
                cost_grad[ mask_ms.flatten('F') <= 0.07 ] = 0 #0 ] = .0
                
            gJ_mean[ll, iteration]  = np.mean(cost_grad)            
                
            cost_grad_mat += np.reshape(cost_grad, [nx_ms, ny_ms, nz_ms], 'F')

            ## ADAM implementation
            first_moment  = beta1 * first_moment  + (1 - beta1) * cost_grad_mat
            second_moment = beta2 * second_moment + (1 - beta2) * np.power(cost_grad_mat, 2)

            first_moment_bar  = first_moment  / ( 1 - beta1**(iteration-last_stage_iter + 1) )
            second_moment_bar = second_moment / ( 1 - beta2**(iteration-last_stage_iter + 1) )

            if stage == params.n_stages - 1:
                first_moment_bar[ mask == 0 ]  = 0.
                second_moment_bar[ mask == 0 ] = 1.
            else:            
                first_moment_bar[mask_ms <= 0.07]  = 0 # 0] = .0
                second_moment_bar[mask_ms <= 0.07] = 1 # 0]     = .0      
                
            beta -= alpha * first_moment_bar / (np.sqrt(second_moment_bar) + epsilon)

            if beta[beta < params.beta_a].size:
                beta_before = np.copy(beta) + alpha * first_moment_bar / (np.sqrt(second_moment_bar) + epsilon)
                beta[beta < params.beta_a] = beta_before[beta < params.beta_a]

                if beta[beta < params.beta_a].size:  # if still smaller than beta air
                    beta[beta < params.beta_a] = params.beta_a
                print('fixed beta!')

            end = time.time()

            runtime[ll, iteration] = end - start
            if beta.max() > 500:
                print("ERROR")

            # Multi scale in grid resolution:
            if (up_scale >= cost_window_size + 1) and (stage < (params.n_stages - 1)):
                iters_ms[stage] = iteration
                last_stage_iter = iteration
                prior_w        *= 1#4
                up_scale        = 0                
                stage          += 1 
                zoom_ms         = grid_ms[stage - 1] / grid_ms[stage] * np.ones(3)
                beta            = zoom(beta, zoom_ms, mode='nearest')  
                
                if stage < params.n_stages - 1:
                    mask_ms = zoom(mask, 1./grid_ms[stage])
                    beta[ mask_ms <= 0.07 ] = params.beta_a # 0 ] = params.beta_a
                    if np.mod(np.log2(nx_ms),1) == 0:
                        Np_scale /= 2.
                else:
                    mask_ms = mask  
                    beta[ mask == 0 ] = params.beta_a
                    Np_scale = 4              
                    tmp = 8
                    #if flags.air_model:
                    #    tmp += int(np.floor((params.TOA + params.H - params.z_spacing * params.nx) / params.z_spacing))
                    #beta[ :, :, params.nz - tmp ]       = params.beta_a
                    #beta[ :, :, params.nz - (tmp + 1) ] = params.beta_a
                    #min_mb        = np.sum(np.sum(np.sum(abs(beta_zoom_out - beta_gt), 2), 1), 0) / np.sum(beta_gt_flat) * 100  
                    
                [nx_ms, ny_ms, nz_ms] = beta.shape

                if beta[beta < params.beta_a].size:
                    beta[beta < params.beta_a] = params.beta_a

                beta_f = beta.flatten('F')

                if flags.crop_s == '' or flags.crop_s == 'cloud1 ':
                    n_pixels_w = np.max([nx_ms, ny_ms]) * 2 # resolution ~= spacing / 2 = 10 meters
                    n_pixels_h = np.max([nx_ms, ny_ms]) * 2
                else:
                    n_pixels_w = np.max([nx_ms, ny_ms])  # resolution ~= spacing / 2 = 10 meters
                    n_pixels_h = np.max([nx_ms, ny_ms])
                n_pixels  = n_pixels_w * n_pixels_h  # !! T
                grid_size = np.prod(beta.shape)

                I_algo  = np.zeros((n_sensors, len(beta0_diff), n_pixels_h, n_pixels_w))
                I_gt_ms = zoom(I_gt, (1, 1 / grid_ms[stage, 0], 1 / grid_ms[stage, 1]), mode='nearest')

                first_moment  = 0  # m0
                second_moment = 0
                
                alpha *= alpha_scale

                sio.savemat(flags.out_path + add_path_out + out_name + ' start of stage ' + str(stage) + '.mat',
                            {'beta0_diff': beta0_diff[bb], 'mask': mask, 'beta_gt': beta_gt, 'params': params, #'air_p': air_p,  
                             'flags': flags,                                               # Scene pre-fixed params
                             'alpha': alpha, 'beta1': beta1, 'beta2': beta2,               # Optimization hyper-params
                             'smooth_lambda': slambda, 'smooth_term': smooth_term[ll],     # Smooth prior
                             'monoz_lambda': mlambda, 'monoZ_term': monoZ_term[ll],        # Monotonically increasing z prior
                             'first_moment': first_moment, 'second_moment': second_moment, # Optimization iters params
                             'stage': stage, 'iters_ms': iters_ms, 
                             'grid_ms': grid_ms, 'alpha_scale': alpha_scale,               # Multiscale
                             'cost_data': cost_data[ll], 'cost_mean': cost_mean[ll], 'I_algo': I_algo, 
                             'runtime': runtime[ll], 'iteration': iteration, 'mass_err': mass_err[ll], 'mb': mb[ll], 
                             'beta': beta})                                                # algorithm calculated variables

            if (np.mod(iteration, 10) == 0) and (iteration > 0):
                bb0 = beta0_diff[bb]
                sio.savemat(flags.out_path + add_path_out + out_name + ' iter.mat',
                            {'beta0_diff': beta0_diff[bb], 'mask': mask, 'beta_gt': beta_gt, 'params': params, #'air_p': air_p,  
                             'flags': flags,                                               # Scene pre-fixed params
                             'alpha': alpha, 'beta1': beta1, 'beta2': beta2,               # Optimization hyper-params
                             'smooth_lambda': slambda, 'smooth_term': smooth_term[ll],     # Smooth prior
                             'monoz_lambda': mlambda, 'monoZ_term': monoZ_term[ll],        # Monotonically increasing z prior
                             'first_moment': first_moment, 'second_moment': second_moment, # Optimization iters params
                             'stage': stage, 'iters_ms': iters_ms, 
                             'grid_ms': grid_ms, 'alpha_scale': alpha_scale,               # Multiscale
                             'cost_data': cost_data[ll], 'cost_mean': cost_mean[ll], 'I_algo': I_algo, 
                             'runtime': runtime[ll], 'iteration': iteration, 'mass_err': mass_err[ll], 'mb': mb[ll], 
                             'beta': beta})                                                # algorithm calculated variables

    curr_iter = iteration

scheduler.stop()    

print('end')
