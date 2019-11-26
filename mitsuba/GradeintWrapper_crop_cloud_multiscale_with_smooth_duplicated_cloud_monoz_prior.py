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

def render_scene(scene, output_filename, n_cores, grid_size, n_pixels_w, n_pixels_h, Igt=None):    
    queue = RenderQueue()

    # Create a queue for tracking render jobs
    film   = scene.getFilm()
    size   = film.getSize() 
    bitmap = Bitmap(Bitmap.ELuminance, Bitmap.EFloat32, size) # for radiance
    #bitmap = Bitmap(Bitmap.ERGB, Bitmap.EUInt8, size) # for RGB image
    blocksize = max(np.divide(max(size.x, size.y), n_cores), 1)
    scene.setBlockSize(blocksize) 

    scene.setDestinationFile(output_filename)    

    #for p in range(n_pixels_h*n_pixels_w):
        #scene.setSpecGt(float(Igt[p]), p)
    scene.setSpecGt(list(Igt))#, int(n_pixels_h*n_pixels_w))
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
    #outFile = FileStream('renderedResult.png', FileStream.ETruncReadWrite)
    #bitmap.write(Bitmap.EPNG, outFile)
    #outFile.close()

    radiance   = np.array(bitmap.buffer()) 
    inner_grad = np.array(scene.getTotalGradient())
    #inner_grad_tmp = scene.getTotalGradient()
    #inner_grad = np.reshape(inner_grad_tmp, [grid_size, n_pixels_w * n_pixels_h], 'F')
    #inner_grad = np.reshape(inner_grad_tmp, [grid_size, n_pixels_w, n_pixels_h])

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

## flags
render_gt_f = False
parallel_f  = True
crop_f      = False
enlarged_f  = False
debug_f     = True

### Loading a cloud

##--------------------------
##           JPL
##--------------------------
out_path  = '../Gradient wrapper/jpl/'

team       = 'jpl '

x_spacing = 0.02 # in km
y_spacing = 0.02 # in km
z_spacing = 0.04 # in km

minval    = 2

## aviad's crop
#full_cloud = np.load('CloudsSim/jpl/jpl_ext.npy')
#crop_s     = ''

### aviad's 2nd crop
#full_cloud = sio.loadmat('CloudsSim/jpl/jpl_bigger_cloud.mat')
#full_cloud = full_cloud['beta']
#crop_s     = 'c2 '

### small cloudfield
full_cloud = sio.loadmat('CloudsSim/jpl/small_cloud_field.mat')
full_cloud = full_cloud['beta_smallcf']
crop_s     = 'small cf '


## cloud 1
#full_cloud = sio.loadmat('CloudsSim/jpl/cloud1.mat')
#full_cloud = full_cloud['beta_cloud1']
#full_cloud = full_cloud[:, :, 12:38] #assume we not to bound this cloud hieght and top
#crop_s     = 'cloud1 '

## cloud 2
#full_cloud = sio.loadmat('CloudsSim/jpl/cloud2.mat')
#full_cloud = full_cloud['beta_cloud2']
#full_cloud = full_cloud[:,:,12:47] #assume we not to bound this cloud hieght and top
#crop_s     = 'cloud2 '

##--------------------------
##           Eshkol
##--------------------------
#out_path  = '../Gradient wrapper/Eshkol/'
#
#team = 'eshkol '
#
#x_spacing = 0.02 # in km
#y_spacing = 0.02 # in km
#z_spacing = 0.02 # in km

## single cloud
#full_cloud = sio.loadmat('CloudsSim/eshkol/Beta_BOMEX_1CLD_512x512x320_500CCN_10m_7dT0.1_0000003240_pdfVARS2.mat')
## croping the cloud so it will have even voxels in each axis and to cut off all the zeros voxels
##full_cloud = full_cloud['beta'][9:, 11:, 48:168] # for eshkol's single cloud snapshot 3600
#full_cloud = full_cloud['beta'][13:,13:95,49:129] # for eshkol's single cloud snapshot 3240
## downsample the cloud to have less voxels
#full_cloud = zoom(full_cloud, (.5, .5, .5))
#crop_s     = 'singlecloud '

#minval = 3.4621

##--------------------------
##        Padding
##--------------------------
if crop_s == 'c2 ':
    full_cloud[ 0, :, :] = 0.01
    full_cloud[-1, :, :] = 0.01
    full_cloud[ :, 0, :] = 0.01
    full_cloud[ :,-1, :] = 0.01
    full_cloud[ :, :, 0] = 0.01
    full_cloud[ :, :,-1] = 0.01
    # npad is a tuple of (n_before, n_after) for each dimension    
    npad = ((5, 5), (0, 0), (0, 0))

elif crop_s == 'small cf ':   
    full_cloud[:, :,  0] = 0.01
    full_cloud[:, :, -1] = 0.01
    npad = ((4, 3), (1, 1), (0, 0))#((2, 1), (2, 2), (0, 0))

else:
    # npad is a tuple of (n_before, n_after) for each dimension    
    npad = ((1, 1), (1, 1), (1, 1))

full_cloud     = np.pad(full_cloud, pad_width=npad, mode='constant', constant_values=0.01)
[ nx, ny, nz ] = full_cloud.shape

# manually creates mask
ind0x, ind0y, ind0z    = np.where(full_cloud < minval)
non0_x, non0_y, non0_z = np.where(full_cloud >= minval)
ind0f                  = np.where(full_cloud.flatten('F') < minval) 
non0_f                 = np.where(full_cloud.flatten('F') >= minval)

full_cloud[ind0x, ind0y, ind0z] = 0.01

n_sensors  = 9

## Update Mask to space curving mask
mask_name  = 'mask original ' + str(team) + 'with air ' + crop_s + str(np.prod(full_cloud.shape)) + ' grid points ' + str(n_sensors) 
mask_name += ' sensors above the medium 1 cycle ' + str(np.power(np.max([nx, ny]), 2)) + ' pixels.mat'
mask       = sio.loadmat(mask_name)
load_mask  = mask['mask']

if crop_s == 'small cf ' or crop_s == 'c2 ': 
    beta_gt = full_cloud
    mask    = load_mask    

else:
    if crop_f:  
        ns      = 8    
        beta_gt = full_cloud[12:12+ns, 14:14+ns, 10:10+ns]
        mask    = load_mask[12:12+ns,  14:14+ns, 10:10+ns]

    else:
        beta_gt = full_cloud
        mask    = load_mask

    if enlarged_f:
        enlarge_scale = 3.
        beta_gt       = zoom(beta_gt, enlarge_scale)
        mask          = zoom(mask,    enlarge_scale)
        mask[mask < 0]          = 0
        beta_gt[beta_gt < 0.01] = 0.01

    if crop_f or enlarged_f:
        npad    = ((4, 4), (4, 4), (4, 4))
        beta_gt = np.pad(beta_gt, pad_width=npad, mode='constant', constant_values=0.01)    
        mask    = np.pad(mask,    pad_width=npad, mode='constant', constant_values=0)

    else:    
        npad    = ((2, 2), (0, 0), (5, 6))
        beta_gt = np.pad(beta_gt, pad_width=npad, mode='constant', constant_values=0.01) 
        mask    = np.pad(mask,    pad_width=npad, mode='constant', constant_values=0)

    ind0x, ind0y, ind0z    = np.where(mask <= 0.07) # == 0)
    non0_x, non0_y, non0_z = np.where(mask > 0.07) # 0)
    ind0f  = np.where(mask.flatten('F') <= 0.07) # == 0)
    non0_f = np.where(mask.flatten('F') > 0.07) # 0)

beta_gt[ind0x, ind0y, ind0z] = 0.01
beta_gt_flat = beta_gt.flatten('F')

[ nx, ny, nz ] = beta_gt.shape     

# bounding box = [xmin, ymin, zmin, xmax, ymax, zmax] in km units 
#bounds = [-nx * x_spacing / 2, -ny * y_spacing / 2, 0, nx * x_spacing / 2, ny * y_spacing / 2, nz * z_spacing]
bounds = [ 0, 0, 0, nx * x_spacing, ny * y_spacing, nz * z_spacing ]

##--------------------------------------------
##            Algorithm parameters
##--------------------------------------------
max_iterations = 500 * 2
max_val        = np.max(beta_gt_flat)
grid_size      = np.prod(beta_gt.shape)

n_unknowns = grid_size
mb         = np.zeros(max_iterations)
mass_err   = np.zeros(max_iterations)

# sensors parameters
n_pixels_w = nx #* 2 # resolution ~= spacing / 2 = 10 meters
n_pixels_h = ny #* 2
n_pixels   = n_pixels_h * n_pixels_w

# multi-scale parameters
fractal_dim = 2 - 0.4
#grid_ms  = nx / np.round(np.array([ 5., 5.*(fractal_dim), 5.*(fractal_dim)**2, 5.*(fractal_dim)**3, float(nx) ]))
#grid_ms  = nx / np.round(np.array([ 4., 4.*np.sqrt(2), 4.*np.sqrt(2)**2, 4.*np.sqrt(2)**3, 4.*np.sqrt(2)**4, 4.*np.sqrt(2)**5,
                                    #4.*np.sqrt(2)**6, float(nx) ]))

grid_ms_x  = nx / np.round(np.array([ 4., 4.*2, 4.*2**2, 4.*2**3,  4.*2**4, float(nx) ]))				    
n_stages = len(grid_ms_x)

grid_ms  = np.zeros((n_stages, 3))
grid_ms[:, 0] = grid_ms_x
grid_ms[:, 1] = ny / np.round(np.array([ 4., 4.*2, 4.*2**2, 4.*2**3,  4.*2**4, float(ny) ]))	
grid_ms[:, 2] = nz / np.round(np.array([ 4., 4.*2, 4.*2**2, float(nz), float(nz), float(nz) ]))	

iters_ms = np.zeros(n_stages, dtype=int)

# optimizer parameters - ADAM
alpha       = 0.3 # 0.08 # randomly select alpha hyperparams -> r = -a * np.random.rand() ; alpha = 10**(r) - randomly selected from a logarithmic scale
alpha_scale = 1   # (0.15/0.5)**(1. / (n_stages - 1))
beta1       = 0.9 # randomly select beta1 hyperparams -> sample (1-beta1), r = -a * np.random.uniform(-3, -1) ; beta1 = 1 - 10**(r) - randomly 
                  # selected from a logarithmic scale
epsilon     = 1e-8
beta2       = 0.999

#photons_total = np.array([512 * 18 *18])
Np_vector = np.array([512*2]) # photons_total / n_pixels
gt_Np_fac = 8/2

beta0_diff = np.array([2])

sensors_pos = [ None ] * n_sensors # create an empty list

output_filename = 'renderedResult'

if parallel_f: # Set parallel job or run on 1 cpu only
    n_cores = 20 #multiprocessing.cpu_count() # T if others are running
else:
    n_cores = 1

## Connect to a socket on a named host or IP address
## 7754 is the default port of 'mtssrv'
#stream = SocketStream('132.68.58.133', 7554)

## Create a remote worker instance that communicates over the stream
#remoteWorker = RemoteWorker('netWorker', stream)

# Start up the scheduling system with one worker per local core
scheduler = Scheduler.getInstance()
for i in range(0, n_cores):
    scheduler.registerWorker(LocalWorker(i, 'wrk%i' % i))
#n_cores_remote = 10
# register remote workers
#for i in range(n_cores, n_cores_remote+n_cores):
    #scheduler.registerWorker(remoteWorker(i, 'wrk%i' % i))
#scheduler.registerWorker(remoteWorker)

scheduler.start()

# scenes params
TOA      = bounds[5]
up_const = np.array([-1, 0, 0])

## for jpl's cloud
if crop_s == 'c2 ':
    H        = z_spacing * nz / 2.
    t_const  = np.array([bounds[3] / 2., bounds[4] / 2., TOA * 3. / 4. ])
elif crop_s == 'small cf ':
    H        = z_spacing * nz
    t_const  = np.array([bounds[3] / 2., bounds[4] / 2., TOA / 4. ])
else:
    H        = z_spacing * nz / 4.
    t_const  = np.array([bounds[3] / 2., bounds[4] / 2., TOA * 2. / 3. ])

## for eshkol's cloud
#H        = z_spacing * nz / 2. 
#t_const  = np.array([bounds[3] / 2., bounds[4] / 2., TOA])

# Ground Truth:
o = np.zeros((n_sensors, 3))
t = np.zeros((n_sensors, 3))
u = np.zeros((n_sensors, 3))

o[0] = np.array([round(bounds[3], 1) / 2., round(bounds[4], 1) / 2., TOA + H]) 
t[0] = t_const
u[0] = up_const

# FOV calc: fov is set by axis x
max_medium    = np.array([bounds[3], bounds[4], bounds[5]])
min_medium    = np.array([bounds[0], bounds[1], bounds[2]])
medium_center = ( max_medium + min_medium ) / 2

L = np.max([max_medium[0] - min_medium[0], max_medium[1] - min_medium[1]]) / 2 #camera's FOV covers the whole medium
if crop_s == 'c2 ':
    fov_rad = 2 * np.arctan(L / ((TOA + H) / 3) )
elif crop_s == 'small cf ':
    fov_rad = 2 * np.arctan(L *1.75 / ((TOA + H)))    
else:
    fov_rad = 2 * np.arctan(L / ((TOA + H) / 4) )
fov_deg = 180 * fov_rad / np.pi

## Setting the formation of the cameras - a dome or a cycle:
n_cycles = 1
#formation_s = 'cycle '
formation_s = 'dome '
r_toa       = np.linalg.norm(o[0]-t_const) # creating a Dome and not a Circle

for rr in range(n_cycles):
    if crop_s == 'c2 'or crop_s == 'small cf ':
        sensors_radius = y_spacing * (ny + 16) / 2     
    else:
        sensors_radius = y_spacing * (ny + 12) / 2
    for ss in range(n_sensors-1):
        theta = 2 * np.pi / (n_sensors - 1) * ss
        o[ss + 1 + rr*4] = np.array([round(o[0][0] + sensors_radius * np.cos(theta), 2), 
                                     round(o[0][1] + sensors_radius * np.sin(theta), 2), TOA + H])     

        # dome
        o[ss + 1 + rr*4][2] = np.sqrt(r_toa**2 - (o[ss + 1 + rr*4][0] - o[0][0])**2 - (o[ss + 1 + rr*4][1]-o[0][1])**2 ) + t_const[2]
        t[ss + 1 + rr*4]    = t_const
        u[ss + 1 + rr*4]    = up_const

gt_out = ''
if enlarged_f:
    gt_out += 'enlarged '

rr_depth = None

if rr_depth is not None:
    rr_str = ' rrDepth ' + str(rr_depth) 
else:
    rr_str = ''

gt_out += 'ground truth original ' + str(team) + 'cloud with air and ocean ' + crop_s + str(grid_size) + ' grid points '
gt_out += str(n_sensors) + ' sensors all above the medium 1 ' + formation_s + str(n_pixels) + ' pixels '
gt_out += str(Np_vector[0] * gt_Np_fac) + ' photons' + rr_str + '.mat'

if (not os.path.isfile(gt_out)) or render_gt_f:
    I_gt     = np.zeros((n_sensors, n_pixels_h, n_pixels_w))

for ss in range(n_sensors):
    newUp, _        = transformLookAt(o[ss], t[ss], u[ss])
    sensors_pos[ss] = { 'origin' : Point(o[ss][0],   o[ss][1],   o[ss][2]),
                        'target' : Point(t[ss][0],   t[ss][1],   t[ss][2]), 
                        'up'     : Vector(newUp[0],  newUp[1],   newUp[2]) }

    if (not os.path.isfile(gt_out)) or render_gt_f:
        scene_gt = pyScene()
        scene_gt.create_new_scene(beta=beta_gt, g=0.85, origin=sensors_pos[ss]['origin'], target=sensors_pos[ss]['target'], 
                                  up=sensors_pos[ss]['up'], nSamples=int(Np_vector[0] * gt_Np_fac), sensorType='perspective', 
                                  bounding_box=bounds, fov=fov_deg, width=n_pixels_w, height=n_pixels_h, rrDepth=rr_depth)

        I_gt[ss], _ = render_scene(scene_gt._scene, output_filename, n_cores, grid_size, n_pixels_w, n_pixels_h)
        sio.savemat(gt_out, {'beta_gt': beta_gt, 'I_gt': I_gt})
        print(ss)

    ##scene        = sceneLoadFromFile(os.path.realpath(scene_gt[ss]._medium._medium_path) + '/scene.xml')
    ##I_gt[ss], gt_grad[ss] = render_scene(scene, output_filename, n_cores, grid_size, n_pixels_w, n_pixels_h)

if (not os.path.isfile(gt_out)) or render_gt_f:
    sio.savemat(gt_out, {'beta_gt': beta_gt, 'I_gt': I_gt})

else:
    gt = sio.loadmat(gt_out)
    I_gt = gt['I_gt']

slambda_v = np.array([1])# 0.1, 1e-2,1e-3, 1e-4, 1e-5])
laplacian3d          = np.ones((3, 3, 3)) / 26
laplacian3d[1, 1, 1] = -1

# monotonically increasing by z prior
mlambda_v = np.array([1])
kz = np.array([[[1, 0, -1], [1, 0, -1], [1, 0, -1]], 
               [[1, 0, -1], [1, 0, -1], [1, 0, -1]],
               [[1, 0, -1], [1, 0, -1], [1, 0, -1]]])
kz = kz / np.linalg.norm(kz.flatten(), ord=1)

for ll in range(len(slambda_v)):
    slambda = slambda_v[ll]
    mlambda = mlambda_v[ll]

    add_path_out  = 'multiscale/'
    if mlambda > 0.:
        add_path_out += 'monoz/'        
    if slambda > 0.:
        add_path_out += 'smooth/'
    if crop_f:
        add_path_out += 'crop/'
    if enlarged_f:
        add_path_out += 'enlarged/'        

    add_path_out += datetime.datetime.now().strftime("%y_%m_%d_%H:%M") + '/'
    out_path     += add_path_out

    if not os.path.exists(out_path):
        os.makedirs(out_path)
        shutil.copyfile(os.path.realpath(__file__), out_path + 'exe_script.py')

    out_name  = 'stages ' + str(n_stages) + ' lambda1 ' + str(slambda) + ' lambda2 ' + str(mlambda) + ' w air and ocean '
    out_name += crop_s + str(grid_size) + ' grid points ' + str(n_sensors) + ' sensors in 1 ' + formation_s 
    out_name += str(n_pixels) + ' pixels ' + str(Np_vector[0]) + rr_str + ' photons changes w scale w 2nd mom SC mask beta0 '

    # Updating the mask to teh grid stages
    #ind0x_ms = [ None ] * n_stages
    #ind0y_ms = [ None ] * n_stages
    #ind0z_ms = [ None ] * n_stages
    #ind0f_ms = [ None ] * n_stages

    #ind0x_ms[-1], ind0y_ms[-1], ind0z_ms[-1] = np.where(mask == 0)
    #ind0f_ms[-1] = ind0f

    #for st in range(n_stages - 1):
        #mask_ms = zoom(mask, 1 / grid_ms[st] * np.ones(3))
        #mask_ms[mask_ms <= 1e-15] = 0
        #ind0x_ms[st], ind0y_ms[st], ind0z_ms[st] = np.where(mask_ms <= 0)
        #ind0f_ms[st]                = np.where(mask_ms.flatten('F') <= 0)

    runtime     = np.zeros((len(beta0_diff), max_iterations))
    cost_data   = np.zeros((len(beta0_diff), max_iterations))  
    cost_mean   = np.zeros((len(beta0_diff), max_iterations))        
    smooth_term = np.zeros((len(beta0_diff), max_iterations))
    monoZ_term  = np.zeros((len(beta0_diff), max_iterations))


    # Loop   
    for bb in range(len(beta0_diff)): 
        out_name += str(beta0_diff[bb])

        # optimizer parameters - ADAM
        first_moment  = 0 #m0
        second_moment = 0 #v0
        last_stage_iter = 0

        # Gradient descent loop
        Np_ms     = 1#8 * 2
        up_scale  = 0
        stage     = 0    
        cost_iter = 0
        cost_window_size = 10#0/4

        beta  = np.ones( np.array(beta_gt.shape / grid_ms[stage], dtype=int) ) * beta0_diff[bb]
        if beta[beta < 0.01].size:
            beta[beta < 0.01] = 0.01

        beta0 = zoom(beta, grid_ms[stage])# * np.ones(3))

        beta_zoom_out        = np.copy(beta0)
        [nx_ms, ny_ms, nz_ms] = beta.shape
        n_pixels_w            = np.max([nx_ms, ny_ms]) #* 2 # resolution ~= spacing / 2 = 10 meters
        n_pixels_h            = np.max([nx_ms, ny_ms]) #* 2  
        n_pixels              = n_pixels_h * n_pixels_w

        Np_scale  = (n_stages - 1) * 2 / 2 / 2 # np.array([2, 1, 1, 1, 0.5, 0.25, 0.25, 1])#
        prior_w   = 5**(-(n_stages-1))
        grid_size = np.prod(beta.shape)

        I_algo  = np.zeros((n_sensors, len(beta0_diff), n_pixels_h, n_pixels_w))
        I_gt_ms = zoom(I_gt, (1, 1 / grid_ms[stage,0], 1 / grid_ms[stage,1]))    

        start   = time.time()

        for iteration in range(max_iterations):               
            if iteration > 0:
                beta_zoom_out = zoom(beta, grid_ms[stage])#(grid_ms[stage], grid_ms[stage], grid_ms[stage]))

            mb[iteration] = np.sum(np.sum(np.sum( abs(beta_zoom_out - beta_gt), 2), 1), 0 ) / np.sum( beta_gt_flat ) * 100
            mass_err[iteration] = (np.sum(abs(beta_zoom_out.flatten('F'))) - np.sum(abs(beta_gt_flat))) / np.sum( beta_gt_flat ) * 100
            
            if debug_f:
    			print("iteration = " + str(iteration))
                print("error = " + str(np.round(mb[iteration],2)))

            cost_grad = np.zeros(grid_size)

            cost_iter = 0
            for ss in range(n_sensors):
                I_gt_zoomed = zoom(I_gt[ss], n_pixels_h/np.array(I_gt[ss].shape, dtype=float))
                algo_scene = pyScene()
                algo_scene.create_new_scene(beta=np.copy(beta), origin=sensors_pos[ss]['origin'], target=sensors_pos[ss]['target'], ##MAYBE NOT COPT BETA?!?
                                            up=sensors_pos[ss]['up'], nSamples=int(Np_vector[0]/Np_ms*Np_scale), g=0.85, fov=fov_deg, 
                                            sensorType='perspective', bounding_box=bounds, width=n_pixels_w, height=n_pixels_h, 
                                            rrDepth=rr_depth)

                [ I_algo[ss, bb], inner_grad ] = render_scene(algo_scene._scene, output_filename, n_cores, grid_size, n_pixels_w,
                                                              n_pixels_h, I_gt_zoomed.flatten('F'))
                # remove the scene's dir
                if os.path.exists(os.path.realpath(algo_scene._medium._medium_path)):
                    shutil.rmtree(os.path.realpath(algo_scene._medium._medium_path))
                #
                #tmp        =  (-1) * ( zoom(I_algo[ss, bb], np.array(I_gt[ss].shape, dtype=float)/n_pixels_h ) - I_gt[ss] ) #I_gt_ms[ss])                
                #tmp        = zoom(tmp, n_pixels_h / np.array(I_gt[ss].shape, dtype=float))
                #cost_iter += np.linalg.norm(tmp, ord=2)
                cost_grad += inner_grad#np.dot(inner_grad, tmp.flatten('F'))
                cost_iter += np.linalg.norm(I_gt_zoomed - I_algo[ss, bb], ord=2)

            cost_grad_mat  = np.zeros_like(beta)

            ## smooth prior
            smooth_term[bb, iteration] = np.linalg.norm(ndimage.convolve(beta, laplacian3d, mode='nearest'))

            ## monotonically increasing in z prior
            indic_b  = np.tanh(beta.flatten('F') / 20.)
            indic_b[beta.flatten('F') <= 0.01] = 0
            #Dzb  = ndimage.convolve(beta, kz)
            _, __, Dzb = np.gradient(beta)
            Dzbf       = Dzb.flatten('F')
            monoZ_term[bb, iteration] = np.dot(np.transpose(indic_b), np.tanh(Dzbf / 5.))

            #if stage >= n_stages - 2:
            smooth_grad = slambda * prior_w  * ndimage.convolve(ndimage.convolve(beta, laplacian3d, mode='nearest'), laplacian3d, 
                                                                mode='nearest')

            dindic_b = np.cosh(beta.flatten('F') / 20.)**(-2) / 20.
            dindic_b[beta.flatten('F') <= 0.01] = 0            
            mg1f     = np.dot(np.diag(np.cosh(Dzbf / 5.)**(-2)), indic_b) / 5.
            #mg1     = ndimage.convolve(np.reshape(mg1f, [nx_ms, ny_ms, nz_ms], 'F'), kz)
            _, __, mg1 = np.gradient(np.reshape(mg1f, [nx_ms, ny_ms, nz_ms], 'F'))                
            mgf        = -mlambda * prior_w  * ( mg1.flatten('F') + np.dot(np.diag(dindic_b), np.tanh(Dzbf / 5.)) )

            mz_grad = np.reshape(mgf, [nx_ms, ny_ms, nz_ms], 'F')
            if stage == n_stages - 1:        
                smooth_grad[ind0x, ind0y, ind0z] = 0.0
                mz_grad[ind0x, ind0y, ind0z]     = 0.0

            cost_grad_mat += smooth_grad + mz_grad


            # cost_data[bb, iteration]    = 0.5 * cost_iter + slambda * smooth_term[bb, iteration] + mlambda * monoZ_term[bb, iteration]	  	
            # cost_mean[bb, iteration] = 0.5 * cost_iter / n_pixels                
            cost_data[bb, iteration] = 0.5 * cost_iter
            cost_mean[bb, iteration] = 0.5 * cost_iter / n_pixels + prior_w * ( 0.5 * slambda * smooth_term[bb, iteration]
                                                                      - mlambda * monoZ_term[bb, iteration] ) / float(np.prod(beta.shape))
            # Multi scale:        
            if ((stage == 0) and (iteration > cost_window_size * 2)) or ((stage > 0) and 
                                                                         (iteration - iters_ms[stage - 1] > cost_window_size * 2)):
                if ( abs(np.mean(cost_mean[bb, iteration-cost_window_size:iteration]) - 
                         np.mean(cost_mean[bb, iteration-cost_window_size*2:iteration-cost_window_size]) ) < 0.001):
                    up_scale = cost_window_size + 1
                else: # cost is saturated (mean) X times in a row
                    up_scale = 0

            if stage == n_stages - 1:
                cost_grad[ ind0f ] = 0.0 

            cost_grad_mat += np.reshape(cost_grad, [nx_ms, ny_ms, nz_ms], 'F')

            ## ADAM implementation
            first_moment  = beta1 * first_moment  + (1 - beta1) * cost_grad_mat
            second_moment = beta2 * second_moment + (1 - beta2) * np.power(cost_grad_mat, 2)

            first_moment_bar  = first_moment  / ( 1 - beta1**(iteration-last_stage_iter + 1) )
            second_moment_bar = second_moment / ( 1 - beta2**(iteration-last_stage_iter + 1) )

            if stage == n_stages-1:            
                first_moment_bar[ind0x, ind0y, ind0z]  = 0.
                second_moment_bar[ind0x, ind0y, ind0z] = 1.        

            beta -= alpha * first_moment_bar / (np.sqrt(second_moment_bar) + epsilon)

            if beta[beta < 0.01].size:
                beta_before       = beta + alpha * first_moment_bar #/ (np.sqrt(second_moment_bar) + epsilon)    
                beta[beta < 0.01] = beta_before[beta < 0.01]

                if beta[beta < 0.01].size: # if still smaller than beta of "air"
                    beta[beta < 0.01] = 0.01
                print('fixed beta!')

            end = time.time()

            runtime[bb, iteration] = end - start

            # Multi scale in grid resolution:   
            if (up_scale >= cost_window_size + 1) and (stage < (n_stages - 1)):
                iters_ms[stage] = iteration   
                last_stage_iter = iteration
                prior_w        *= 5
                up_scale        = 0                
                stage          += 1 
                zoom_ms         = grid_ms[stage - 1] / grid_ms[stage] #* np.ones(3)
                beta            = zoom(beta, zoom_ms, mode='nearest')        

                if stage == n_stages - 1 : 
                    beta[ ind0x, ind0y, ind0z ] = 0.01
                    Np_scale = 1                    
                else:
                    Np_scale /= 2.

                if beta[ beta < 0.01 ].size:
                    beta[ beta < 0.01 ] = 0.01    

                beta_f = beta.flatten('F')

                [nx_ms, ny_ms, nz_ms] = beta.shape
                n_pixels_w            = np.max([nx_ms, ny_ms]) #* 2 # resolution ~= spacing / 2 = 10 meters
                n_pixels_h            = np.max([nx_ms, ny_ms]) #* 2    
                n_pixels              = n_pixels_w * n_pixels_h # !! T
                grid_size             = np.prod(beta.shape)            

                I_algo  = np.zeros((n_sensors, len(beta0_diff), n_pixels_h, n_pixels_w))                            
                I_gt_ms = zoom(I_gt, (1, 1/grid_ms[stage, 0], 1/grid_ms[stage, 1]), mode='nearest')

                #min1m = abs(first_moment).min() # !!!CHECK!!!
                #first_moment  = zoom(first_moment, zoom_ms, mode='nearest')
                #first_moment[abs(first_moment) < min1m] = 0.0
                first_moment  = 0 #m0
                #min2m = second_moment.min()
                #second_moment = zoom(second_moment, zoom_ms, mode='nearest')                 
                #second_moment[second_moment < min2m] = 0.0
                second_moment = 0
                alpha        *= alpha_scale

                [ nx_ms, ny_ms, nz_ms ] = beta.shape
                grid_size               = np.prod(beta.shape)
                #cost_window_size       *= 2

                sio.savemat(out_path + out_name + ' start of stage ' + str(stage) + '.mat', 
                            { 'beta0_diff' : beta0_diff[bb], 'mask' : mask, 'beta_gt' : beta_gt,   # Scene pre-fixed params
                              'alpha' : alpha, 'beta1' : beta1, 'beta2' : beta2,                   # Optimization hyper-params
                              'smooth_lambda' : slambda, 'smooth_term' : smooth_term,              # Smooth prior
                              'monoz_lambda' : mlambda, 'monoZ_term' : monoZ_term,                 # Monotonically increasing z prior
                              'first_moment' : first_moment, 'second_moment' : second_moment,      # Optimization iters params
                              'stage' : stage, 'iters_ms' : iters_ms, 'alpha_scale' : alpha_scale, 
                              'Np_ms' : Np_ms, 'grid_ms' : grid_ms,                                  # Multiscale
                              'cost_data' : cost_data, 'mb' : mb, 'I_algo': I_algo, 'runtime' : runtime, 'beta' : beta,
                              'iteration' : iteration, 'mass_err' : mass_err, 'cost_mean' : cost_mean }) # algorithm calculated variables

            if (np.mod(iteration, 10) == 0) and (iteration > 0):
                bb0 = beta0_diff[bb]     
                sio.savemat(out_path + out_name + ' iter.mat', 
                            { 'beta0_diff' : beta0_diff[bb], 'mask' : mask, 'beta_gt' : beta_gt,   # Scene pre-fixed params
                              'alpha' : alpha, 'beta1' : beta1, 'beta2' : beta2,                   # Optimization hyper-params
                              'smooth_lambda' : slambda, 'smooth_term' : smooth_term,              # Smooth prior
                              'monoz_lambda' : mlambda, 'monoZ_term' : monoZ_term,                 # Monotonically increasing z prior
                              'first_moment' : first_moment, 'second_moment' : second_moment,      # Optimization iters params
                              'stage' : stage, 'iters_ms' : iters_ms, 'alpha_scale' : alpha_scale, 
                              'Np_ms' : Np_ms, 'grid_ms' : grid_ms,                                  # Multiscale
                              'cost_data' : cost_data, 'mb' : mb, 'I_algo': I_algo, 'runtime' : runtime, 'beta' : beta,
                              'iteration' : iteration, 'mass_err' : mass_err, 'cost_mean' : cost_mean }) # algorithm calculated variables


    curr_iter = iteration
    #I_algo_p  = I_algo[:, :, 0:curr_iter]
    ##betas_p   = betas[:, 0:curr_iter]
    #cost_p    = cost[:, 0:curr_iter]

    #iters   = np.linspace(1, curr_iter, curr_iter)    

    #for bb_p in range(len(beta0_diff)):        
        #bb0      = beta0_diff[bb_p]                   
        #t_2      = np.argmin(cost_p[bb_p])    
        #gradient = cost_gradient[bb_p]
        #tmp_p    = np.reshape(betas_p[bb_p], [curr_iter, nx_ms, ny_ms, nz_ms], 'F')
        #grid_f   = np.power(betas_p[bb_p,curr_iter-1].shape[0], 1/3.) / nx
        #if abs(grid_f - 1) < 10e-10:
            #beta_gt_zoom = beta_gt
        #else:
            #beta_gt_zoom = zoom(beta_gt, (grid_f, grid_f, grid_f), mode='nearest')

        #betas_full_scale = zoom(tmp_p, (1, 1./grid_f, 1./grid_f, 1./grid_f), mode='nearest')

        #betas_err = (betas_full_scale - beta_gt) / beta_gt * 100    
        #sio.savemat(out_path + out_name + '.mat', 
                    #{ 'beta0_diff' : beta0_diff[bb], 'mask' : mask, 'beta_gt' : beta_gt,          # Scene pre-fixed params
                        #'alpha' : alpha, 'beta1' : beta1, 'beta2' : beta2,                          # Optimization hyper-params
                        #'slambda' : slambda,# 'smooth_g_s0' : smooth_grad_s0, 'smooth_g_s1' : smooth_grad_s1, 
                        ##'smooth_g_s2' : smooth_grad_s2, 'smooth_g_s3' : smooth_grad_s3,            # smooth prior
                        #'first_moment' : first_moment, 'second_moment' : second_moment,             # Optimization iters params
                        #'Np_ms' : Np_ms, 'grid_ms' : grid_ms, 'stage' : stage, 'iters_ms' : iters_ms, #'cost_sat' : cost_sat,
                        ##'betas_s0' : betas_stage0, 'betas_s1' : betas_stage1, 'betas_s2' : betas_stage2, 
                        ##'betas_s3' : betas_stage3, 'betas_s4' : betas_stage4, 'betas_s5' : betas_stage5, 
                        ##'I_algo_s0' : I_algo_s0, 'I_algo_s1' : I_algo_s1, 'I_algo_s2' : I_algo_s2, 'I_algo_s3' : I_algo_s3,
                        #'mb' : mb, 'alpha_scale' : alpha_scale, #'I_algo_s4' : I_algo_s4, 'I_algo_s5': I_algo_s5, 
                        ##'cost_grad_s0' : cost_grad_stage0, 'cost_grad_s1' : cost_grad_stage1, 'cost_grad_s2' : cost_grad_stage2,
                        ## 'cost_grad_s3' : cost_grad_stage3,  'betas_zoom_out' : betas_zoom_out,   # Multiscale
                        #'cost' : cost, 'I_algo': I_algo, 'runtime' : runtime, #'gradient' : cost_gradient, 'betas' : betas, 
                        #'iteration' : iteration, 'beta' : beta, 'cost_mean' : cost_mean })             # algorithm calculated variables

        #mean_betas_err = np.sum(np.sum(np.sum(abs(betas_full_scale - beta_gt), 3), 2), 1) / np.sum(beta_gt.flatten('F'))*100
#
        #plt.figure(figsize=(19,9))    
        #plt.plot(iters, mean_betas_err, '--',  marker='o', markersize=5)
        #plt.title('mean error of beta in %', fontweight='bold')  
        #plt.grid(True)
        #plt.xlim(left=0)
        #plt.ylabel('[%]', fontweight='bold')
        #plt.savefig(out_path + out_name + ' mean error over iteration.png', dpi=300)               
#
        #plt.figure(figsize=(19,9))    
        #plt.plot(iters, cost_p[bb_p], '--',  marker='o', markersize=5)
        #plt.title('Cost', fontweight='bold')  
        #plt.grid(True)
        #plt.xlim(left=0)
        #plt.savefig(out_path + out_name + ' cost.png', dpi=300)               
#
        #plt.figure(figsize=(19,9))    
        #for yy in range(ny-2):                
            #maxc = np.max([np.max(np.array([betas_err[0].flatten(), betas_err[t_2].flatten(), betas_err[-1].flatten()])), 100])
            #minc = np.min([np.min(np.array([betas_err[0].flatten(), betas_err[t_2].flatten(), betas_err[-1].flatten()])), 0])        
#
            #plt.subplot(ny-2, 3, 3 * yy + 1)
            #plt.imshow(betas_err[ 0, :, yy+1, :])
            ##plt.clim(minc, maxc)     
            #plt.colorbar()        
            #plt.axis('off')
            #non0_errs = np.mean(abs(betas_err[0, non0_x, yy + 1, non0_z].flatten()))
            #plt.title('slice y = ' + str(yy + 2) + ' initial status, mean error = ' + str(round(non0_errs, 2)) + '%')
#
            #plt.subplot(ny-2, 3, 3 * yy + 2)
            #plt.imshow(betas_err[ t_2, :, yy+1, :])
            ##plt.clim(minc, maxc)     
            #plt.colorbar()
            #plt.axis('off')
            #non0_errs = np.mean(abs(betas_err[t_2, non0_x, yy + 1, non0_z].flatten()))
            #plt.title('iteration = ' + str(t_2) + ', mean error = ' + str(round(non0_errs, 2)) + '%')
#
            #plt.subplot(ny-2, 3, 3 * yy + 3)
            #plt.imshow(betas_err[ -1, :, yy+1, :])
            ##plt.clim(minc, maxc)     
            #plt.colorbar()
            #plt.axis('off')
            #non0_errs = np.mean(abs(betas_err[-1, non0_x, yy + 1, non0_z].flatten()))
            #plt.title('iteration = ' + str(curr_iter) + ', mean error = ' + str(round(non0_errs, 2)) + '%')
#
        #plt.suptitle('Beta error [%]', fontweight='bold')
        #plt.savefig(out_path + out_name + ' errors heat maps.png', dpi=300)               
#
        #plt.figure(figsize=(19,9))    
        #for yy in range(ny-2):               
            #plt.subplot(ny-2, 4, 4 * yy + 1)
            ##plt.imshow(beta_gt_zoom[ :, yy+1, :])
            #plt.imshow(beta_gt[ :, yy+1, :])
            #plt.colorbar()
            #plt.axis('off')
            #plt.title('Original Beta, slice y = ' + str(yy + 2))
#
            #plt.subplot(ny - 2, 4, 4 * yy + 2)
            ##plt.imshow(tmp_p[ 0, :, yy+1, :])
            #plt.imshow(betas_full_scale[ 0, :, yy+1, :])        
            #plt.colorbar()        
            #plt.axis('off')
            #plt.title('Initial status')
#
            #plt.subplot(ny-2, 4, 4 * yy + 3)
            ##plt.imshow(tmp_p[ t_2, :, yy+1, :])
            #plt.imshow(betas_full_scale[ t_2, :, yy+1, :])
            #plt.colorbar()
            #plt.axis('off')
            #plt.title('iteration = ' + str(t_2))
#
            #plt.subplot(ny - 2, 4, 4 * yy + 4)
            ##plt.imshow(tmp_p[ -1, :, yy + 1, :])
            #plt.imshow(betas_full_scale[ -1, :, yy + 1, :])
            ##plt.clim(minc, maxc)     
            #plt.colorbar()
            #plt.axis('off')
            #plt.title('iteration = ' + str(curr_iter))      
#
        #plt.suptitle('Beta values [1/km]', fontweight='bold')
        #plt.savefig(out_path + out_name + ' density heat maps.png', dpi=300)                       
#
        #plt.figure(figsize=(19,9))    
        #for ss in range(n_sensors):
            #plt.subplot(n_sensors, 4, 4 * ss + 1)
            #plt.imshow(I_gt[ss])
            #maxc = np.max(np.array([I_gt[ss], I_algo_p[ss, bb, 0], I_algo_p[ss, bb, t_2], I_algo_p[ss, bb, -1]]))
            ##plt.clim(0, maxc)
            #plt.colorbar()
            #plt.axis('off')
            #plt.title('Ground truth')
#
            #plt.subplot(n_sensors, 4, 4 * ss + 2)
            #plt.imshow(I_algo_p[ss, bb, 0])
            ##plt.clim(0, maxc)
            #plt.colorbar()
            #plt.axis('off')
            #err = sum(sum((I_gt[ss] - I_algo_p[ss, bb, 0])**2))
            #plt.title('Initial output, error = ' + str(round(err, 7)))
#
            #plt.subplot(n_sensors, 4, 4 * ss + 3)
            #plt.imshow(I_algo_p[ss, bb, t_2])
            ##plt.clim(0, maxc)
            #plt.colorbar()
            #plt.axis('off')
            #err = sum(sum((I_gt[ss] - I_algo_p[ss, bb, t_2])**2))
            #plt.title('iter = ' + str(t_2) + ', error = ' + str(round(err, 7)))    
#
            #plt.subplot(n_sensors, 4, 4 * ss + 4)
            #plt.imshow(I_algo_p[ss, bb, -1])
            ##plt.clim(0, maxc)
            #plt.colorbar()
            #plt.axis('off')
            #err = sum(sum((I_gt[ss] - I_algo_p[ss, bb, -1])**2)) 
            #plt.title('Final output, error = ' +  str(round(err, 7)))
#
        #plt.savefig(out_path + out_name + ' images from sensors.png', dpi=300)


scheduler.stop()    

print('end')
