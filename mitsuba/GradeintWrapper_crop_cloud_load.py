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
    #blocksize = max(np.divide(max(size.x, size.y), n_cores), 1)
    #scene.setBlockSize(blocksize) 

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
    inner_grad_tmp = scene.getTotalGradient()
    inner_grad = np.reshape(inner_grad_tmp, [grid_size, n_pixels_w * n_pixels_h], 'F')
    #inner_grad = np.reshape(inner_grad_tmp, [grid_size, n_pixels_w, n_pixels_h])

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

## flags
render_gt_f = False
f_multi     = True
crop_f      = True

## load jpl's cloud
jpl_full_cloud = np.load('CloudsSim/jpl/jpl_ext.npy')
x_spacing = 0.02 # in km
y_spacing = 0.02 # in km
z_spacing = 0.04 # in km

# npad is a tuple of (n_before, n_after) for each dimension
npad           = ((1, 1), (1, 1), (1, 1))
jpl_full_cloud = np.pad(jpl_full_cloud, pad_width=npad, mode='constant', constant_values=0)
[ nx, ny, nz ] = jpl_full_cloud.shape

# manually creates mask
ind0x, ind0y, ind0z    = np.where(jpl_full_cloud < 2)
non0_x, non0_y, non0_z = np.where(jpl_full_cloud >= 2)
ind0f  = np.where(jpl_full_cloud.flatten('F') < 2)
non0_f = np.where(jpl_full_cloud.flatten('F') >= 2)

jpl_full_cloud[ind0x, ind0y, ind0z] = 0.01

#norm_factor  = sum(beta_gt * z_spacing, 2) / 5.
#beta_gt     /= np.mean(norm_factor)

# Update Mask to space curving mask
n_sensors  = 9
# mask resolution
mask_p     = np.power(np.max([nx, ny]) * 2, 2)
mask_name  = 'mask original jpl with air ' + str(np.prod(jpl_full_cloud.shape)) + ' grid points ' + str(n_sensors) + ' sensors above the '
mask_name += 'medium 1 cycle ' + str(mask_p) + ' pixels.mat'
mask       = sio.loadmat(mask_name)
load_mask  = mask['mask']

# crop cloud 
if crop_f:
    beta_gt = np.pad(jpl_full_cloud[12:19, 14:21, 10:17], pad_width=npad, mode='constant', constant_values=0)
    mask    = np.pad(load_mask[12:19, 14:21, 10:17],      pad_width=npad, mode='constant', constant_values=0)
    [ nx, ny, nz ] = beta_gt.shape    
    
else:
    beta_gt = jpl_full_cloud
    mask    = load_mask

# bounding box = [xmin, ymin, zmin, xmax, ymax, zmax] in km units 
bounds = [0, 0, 0, nx * x_spacing, ny * y_spacing, nz * z_spacing]#[-nx * x_spacing / 2, -ny * y_spacing / 2, 0, nx * x_spacing / 2, ny * y_spacing / 2, nz * z_spacing]

ind0x, ind0y, ind0z    = np.where(mask == 0)
non0_x, non0_y, non0_z = np.where(mask > 0)
ind0f  = np.where(mask.flatten('F') == 0)
non0_f = np.where(mask.flatten('F') > 0)

beta_gt[ind0x, ind0y, ind0z] = 0.01
beta_gt_flat = beta_gt.flatten('F')

# algorithm parameters
max_iterations = 500 * 3 * 5 + 1 #500 *3
max_val        = np.max(beta_gt_flat)
grid_size      = np.prod(beta_gt.shape)

n_unknowns     = grid_size

# sensors parameters
n_pixels_w = np.max([nx, ny]) * 2 # resolution ~= spacing / 2 = 10 meters
n_pixels_h = np.max([nx, ny]) * 2
n_pixels   = n_pixels_h * n_pixels_w

# optimizer parameters - ADAM
alpha   = 0.08 # randomly select alpha hyperparameter -> r = -a * np.random.rand() ; alpha = 10**(r)                                - randomly selected from a logaritmic scale
beta1   = 0.9  # randomly select beta1 hyperparameter -> sample (1-beta1), r = -a * np.random.uniform(-3, -1) ; beta1 = 1 - 10**(r) - randomly selected from a logaritmic scale
epsilon = 1e-8
beta2   = 0.999

Np_vector  = np.array([512]) 
gt_Np_fac  = 2
beta0_diff = np.array([40])

sensors_pos   = [ None ] * n_sensors # create an empty list
algo_scene    = [ None ] * n_sensors # create an empty list
I_algo        = np.zeros((n_sensors, len(beta0_diff), max_iterations, n_pixels_h, n_pixels_w))
runtime       = np.zeros((len(beta0_diff), max_iterations))
cost_gradient = np.zeros((len(beta0_diff), max_iterations, grid_size))
betas         = np.zeros((len(beta0_diff), max_iterations, grid_size))
cost          = np.zeros((len(beta0_diff), max_iterations))

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

# scenes params
TOA      = bounds[5]
H        = z_spacing * nz / 4.
t_const  = np.array([bounds[3] / 2., bounds[4] / 2., TOA * 2. / 3. ])
up_const = np.array([-1, 0, 0])

# Ground Truth:
o = np.zeros((n_sensors, 3))
t = np.zeros((n_sensors, 3))
u = np.zeros((n_sensors, 3))

o[0] = np.array([round(bounds[3], 1) / 2., round(bounds[3], 1) / 2., TOA + H]) 
t[0] = t_const
u[0] = up_const

# FOV calc:
# fov is set by axis x
max_medium    = np.array([bounds[3], bounds[4], bounds[5]])
min_medium    = np.array([bounds[0], bounds[1], bounds[2]])
medium_center = ( max_medium + min_medium ) / 2

L       = np.max([max_medium[0] - min_medium[0], max_medium[1] - min_medium[1]]) / 2 #camera's FOV covers the whole medium
fov_rad = 2 * np.arctan(L / ((TOA + H) / 4) )
fov_deg = 180 * fov_rad / np.pi

for rr in range(1):#2):
    #if rr == 0:
    #    sensors_radius = y_spacing * (ny + 1) / 2  # angles to medium = 51 deg
    #else:
    sensors_radius = y_spacing * (ny + 12) / 2
        
    for ss in range(n_sensors-1):#(n_sensors - 1)/2):
        theta     = 2 * np.pi / (n_sensors - 1) * ss
        o[ss + 1 + rr*4] = np.array([round(o[0][0] + sensors_radius * np.cos(theta), 2), 
                                     round(o[0][1] + sensors_radius * np.sin(theta), 2), TOA + H])     
        t[ss + 1 + rr*4] = t_const
        u[ss + 1 + rr*4] = up_const

gt_out  = 'ground truth original jpl cloud with air and ocean ' + str(grid_size) + ' grid points ' + str(n_sensors) + ' sensors all above the '
gt_out += 'medium 1 cycle ' + str(n_pixels) + ' pixels ' + str(Np_vector[0] * gt_Np_fac) + ' photons.mat'

if (not os.path.isfile(gt_out)) or render_gt_f:
    scene_gt = [ None ] * n_sensors # create an empty list
    I_gt     = np.zeros((n_sensors, n_pixels_h, n_pixels_w))

for ss in range(n_sensors):
    newUp, _        = transformLookAt(o[ss], t[ss], u[ss])
    sensors_pos[ss] = { 'origin' : Point(o[ss][0],   o[ss][1],   o[ss][2]),
                        'target' : Point(t[ss][0],   t[ss][1],   t[ss][2]), 
                        'up'     : Vector(newUp[0],  newUp[1],   newUp[2]) }

    if (not os.path.isfile(gt_out)) or render_gt_f:
        scene_gt[ss] = pyScene()
        scene_gt[ss].create_new_scene(beta=beta_gt, g=0.85, origin=sensors_pos[ss]['origin'], target=sensors_pos[ss]['target'], 
                                 up=sensors_pos[ss]['up'], nSamples=Np_vector[0] * gt_Np_fac, sensorType='perspective', bounding_box=bounds,
                                 fov=fov_deg, width=n_pixels_w, height=n_pixels_h)

        I_gt[ss], _ = render_scene(scene_gt[ss]._scene, output_filename, n_cores, grid_size, n_pixels_w, n_pixels_h)

    ##scene        = sceneLoadFromFile(os.path.realpath(scene_gt[ss]._medium._medium_path) + '/scene.xml')
    ##I_gt[ss], gt_grad[ss] = render_scene(scene, output_filename, n_cores, grid_size, n_pixels_w, n_pixels_h)

if (not os.path.isfile(gt_out)) or render_gt_f:
    sio.savemat(gt_out, {'beta_gt': beta_gt, 'I_gt': I_gt, 'scene_gt': scene_gt})

else:
    gt = sio.loadmat(gt_out)
    I_gt = gt['I_gt']

out_path  = '/home/tamarl/MitsubaGradient/Gradient wrapper/jpl/'

out_name = ''
if crop_f:
    out_name += 'crop '
    
out_name += 'org jpl cloud with air and ocean ' + str(grid_size) + ' grid points ' + str(n_unknowns) + ' unknowns ' + str(n_sensors) + ' sensors all '
out_name += 'above the medium 1 cycle ' + str(n_pixels) + ' pixels ' + str(Np_vector[0]) + ' photons ' + str(alpha) + ' adam alpha with space '
out_name += 'carving mask fix air'

load_path = out_path
load_name = out_name + '.mat'
#"air and ocean 729 grid points 729 unknowns 9 sensors all above the medium 1 cycle 324 pixels 512 photons 0.08 adam alpha with space carving mask fix air.mat"
for bb in range(len(beta0_diff)):

    beta0 = np.ones(beta_gt.shape) * beta0_diff[bb]
    beta0[ind0x, ind0y, ind0z] = beta_gt[ind0x, ind0y, ind0z]  

    # load results 
    load_res  = sio.loadmat(load_path + load_name)
    
    # algo params
    prev_iter = load_res['iteration'][0,0] + 1     
    
    # ADAM params
    alpha = load_res['alpha'][0,0]
    beta1 = load_res['beta1'][0,0]
    beta2 = load_res['beta2'][0,0]
    first_moment      = load_res['first_moment'][0,0]
    second_moment     = load_res['second_moment'][0,0]
    #first_moment_bar  = load_res['first_moment_bar']
    #second_moment_bar = load_res['second_moment_bar']
    
    # scene params
    runtime[:, 0:prev_iter]       = load_res['runtime'][:, 0:prev_iter]
    cost[: ,0:prev_iter]          = load_res['cost'][:, 0:prev_iter]
    I_algo[:, :, 0:prev_iter]     = load_res['I_algo'][:, :, 0:prev_iter]
    betas[:, 0:prev_iter]         = load_res['betas'][:, 0:prev_iter]
    cost_gradient[:, 0:prev_iter] = load_res['gradient'][:, 0:prev_iter]
    
    # Gradient descent loop
    beta_prev = np.reshape(betas[:,prev_iter-1], beta_gt.shape, 'F')
    beta  = np.copy(beta_prev)
    start = time.time()

    for ii in range(max_iterations - prev_iter): 
        iteration = ii + prev_iter
        betas[bb, iteration] = beta.flatten('F')                
        cost_grad            = np.zeros(grid_size)

        cost_iter = 0
        for ss in range(n_sensors):
            # Create scene with given beta
            algo_scene[ss] = pyScene()
            algo_scene[ss].create_new_scene(beta=np.copy(beta), g=0.85, origin=sensors_pos[ss]['origin'], 
                                            target=sensors_pos[ss]['target'], up=sensors_pos[ss]['up'], nSamples=Np_vector[0], 
                                            sensorType='perspective', fov=fov_deg, bounding_box=bounds, width=n_pixels_w, 
                                            height=n_pixels_h)

            [ I_algo[ss, bb, iteration], inner_grad ] = render_scene(algo_scene[ss]._scene, output_filename, n_cores, grid_size,
                                                                     n_pixels_w, n_pixels_h)
            # remove the scene's dir
            import shutil
            if os.path.exists(os.path.realpath(algo_scene[ss]._medium._medium_path)):
                shutil.rmtree(os.path.realpath(algo_scene[ss]._medium._medium_path))
            #
            tmp        =  (-1) * ( I_algo[ss, bb, iteration] - I_gt[ss] )                    
            cost_grad += np.dot(inner_grad, tmp.flatten('F'))
            cost_iter += np.linalg.norm(tmp,ord=2)				

        cost[bb, iteration] = 0.5 * cost_iter
		
        cost_grad[ind0f]             = 0.0            
        cost_gradient[bb, iteration] = cost_grad
        cost_grad_mat                = np.reshape(cost_grad, beta.shape, 'F')

        ## ADAM implementation
        first_moment  = beta1 * first_moment  + (1 - beta1) * cost_grad_mat
        second_moment = beta2 * second_moment + (1 - beta2) * np.power(cost_grad_mat, 2)

        first_moment_bar  = first_moment  / ( 1 - beta1**(iteration + 1) )
        second_moment_bar = second_moment / ( 1 - beta2**(iteration + 1) )

        beta -= alpha * first_moment_bar / (np.sqrt(second_moment_bar) + epsilon)

        if beta[beta <= 0].size:
            beta_before = beta + alpha * first_moment_bar / (np.sqrt(second_moment_bar) + epsilon)    
            beta[beta <= 0] = beta_before[beta <= 0]
            print('fixed beta!')

        end = time.time()

        runtime[bb, iteration] = end - start 
        if (np.mod(iteration, 500) == 0) and (iteration > 0):
            bb0 = beta0_diff[bb]          
            sio.savemat(out_path + out_name + ' iter ' + str(iteration) + '.mat', 
                        { 'beta0_diff' : beta0_diff[bb], 'mask' : mask, 'beta_gt' : beta_gt, # Scene pre-fixed params
                          'alpha' : alpha, 'beta1' : beta1, 'beta2' : beta2,                 # Optimization hyper-params
                          'first_moment' : first_moment, 'second_moment' : second_moment,    # Optimization iters params
                          'cost' : cost, 'I_algo': I_algo, 'betas' : betas, 'gradient' : cost_gradient, 'runtime' : runtime, 
                          'iteration' : iteration})                                          # algorithm calculated variables

            iters = np.linspace(1, iteration, iteration) 
            plt.figure(figsize=(19,9))    
            plt.plot(iters, cost[bb,0:iteration], '--',  marker='o', markersize=5)
            plt.title('Cost', fontweight='bold')  
            plt.grid(True)
            plt.xlim(left=0)
            plt.savefig(out_path + out_name + ' iter ' + str(iteration) + ' cost.png', dpi=300)                

curr_iter = iteration
I_algo_p  = I_algo[:, :, 0:curr_iter]
betas_p   = betas[:, 0:curr_iter]
cost_p    = cost[:, 0:curr_iter]

iters   = np.linspace(1, curr_iter, curr_iter)    

for bb_p in range(len(beta0_diff)):        
    bb0 = beta0_diff[bb_p]                   
    t_2      = np.argmin(cost_p[bb_p])    
    gradient = cost_gradient[bb_p]
    #tmp      = np.ones(beta_gt_flat.shape)
    #tmp[beta_gt_flat >0] = beta_gt_flat[beta_gt_flat >0]    
    tmp_p = np.reshape(betas_p[bb_p], [curr_iter, nx, ny, nz], 'F')
    betas_err = (tmp_p - beta_gt) / beta_gt * 100
    
    sio.savemat(out_path + out_name + '.mat', 
                        { 'beta0_diff' : beta0_diff[bb_p], 'mask' : mask, 'beta_gt' : beta_gt, # Scene pre-fixed params
                          'alpha' : alpha, 'beta1' : beta1, 'beta2' : beta2,                   # Optimization hyper-params
                          'first_moment' : first_moment, 'second_moment' : second_moment,      # Optimization iters params
                          'cost' : cost, 'I_algo': I_algo, 'betas' : betas, 'gradient' : cost_gradient, 'runtime' : runtime, 
                          'iteration' : iteration})                                            # algorithm calculated variables

    mean_betas_err = np.sum(np.sum(np.sum(abs((tmp_p - beta_gt)), 3), 2), 1) / np.sum(beta_gt.flatten('F'))*100

    plt.figure(figsize=(19,9))    
    plt.plot(iters, mean_betas_err, '--',  marker='o', markersize=5)
    plt.title('mean error of beta in %', fontweight='bold')  
    plt.grid(True)
    plt.xlim(left=0)
    plt.ylabel('[%]', fontweight='bold')
    plt.savefig(out_path + out_name + ' mean error over iteration.png', dpi=300)               

    plt.figure(figsize=(19,9))    
    plt.plot(iters, cost_p[bb_p], '--',  marker='o', markersize=5)
    plt.title('Cost', fontweight='bold')  
    plt.grid(True)
    plt.xlim(left=0)
    plt.savefig(out_path + out_name + ' cost.png', dpi=300)               

    plt.figure(figsize=(19,9))    
    for yy in range(ny-2):                
        maxc = np.max([np.max(np.array([betas_err[0].flatten(), betas_err[t_2].flatten(), betas_err[-1].flatten()])), 100])
        minc = np.min([np.min(np.array([betas_err[0].flatten(), betas_err[t_2].flatten(), betas_err[-1].flatten()])), 0])        

        plt.subplot(ny-2, 3, 3 * yy + 1)
        plt.imshow(betas_err[ 0, :, yy+1, :])
        #plt.clim(minc, maxc)     
        plt.colorbar()        
        plt.axis('off')
        non0_errs = np.mean(abs(betas_err[0, non0_x, yy + 1, non0_z].flatten()))
        plt.title('slice y = ' + str(yy + 2) + ' initial status, mean error = ' + str(round(non0_errs, 2)) + '%')

        plt.subplot(ny-2, 3, 3 * yy + 2)
        plt.imshow(betas_err[ t_2, :, yy+1, :])
        #plt.clim(minc, maxc)     
        plt.colorbar()
        plt.axis('off')
        non0_errs = np.mean(abs(betas_err[t_2, non0_x, yy + 1, non0_z].flatten()))
        plt.title('iteration = ' + str(t_2) + ', mean error = ' + str(round(non0_errs, 2)) + '%')

        plt.subplot(ny-2, 3, 3 * yy + 3)
        plt.imshow(betas_err[ -1, :, yy+1, :])
        #plt.clim(minc, maxc)     
        plt.colorbar()
        plt.axis('off')
        non0_errs = np.mean(abs(betas_err[-1, non0_x, yy + 1, non0_z].flatten()))
        plt.title('iteration = ' + str(curr_iter) + ', mean error = ' + str(round(non0_errs, 2)) + '%')

    plt.suptitle('Beta error [%]', fontweight='bold')
    plt.savefig(out_path + out_name + ' errors heat maps.png', dpi=300)               

    plt.figure(figsize=(19,9))    
    for yy in range(ny-2):               
        plt.subplot(ny-2, 4, 4 * yy + 1)
        plt.imshow(beta_gt[ :, yy+1, :])
        plt.colorbar()
        plt.axis('off')
        plt.title('Original Beta, slice y = ' + str(yy + 2))

        plt.subplot(ny - 2, 4, 4 * yy + 2)
        plt.imshow(tmp_p[ 0, :, yy+1, :])
        plt.colorbar()        
        plt.axis('off')
        plt.title('Initial status')

        plt.subplot(ny-2, 4, 4 * yy + 3)
        plt.imshow(tmp_p[ t_2, :, yy+1, :])
        plt.colorbar()
        plt.axis('off')
        plt.title('iteration = ' + str(t_2))

        plt.subplot(ny - 2, 4, 4 * yy + 4)
        plt.imshow(tmp_p[ -1, :, yy + 1, :])
        #plt.clim(minc, maxc)     
        plt.colorbar()
        plt.axis('off')
        plt.title('iteration = ' + str(curr_iter))      

    plt.suptitle('Beta values [1/km]', fontweight='bold')
    plt.savefig(out_path + out_name + ' density heat maps.png', dpi=300)                       

    plt.figure(figsize=(19,9))    
    for ss in range(n_sensors):
        plt.subplot(n_sensors, 4, 4 * ss + 1)
        plt.imshow(I_gt[ss])
        maxc = np.max(np.array([I_gt[ss], I_algo_p[ss, bb, 0], I_algo_p[ss, bb, t_2], I_algo_p[ss, bb, -1]]))
        #plt.clim(0, maxc)
        plt.colorbar()
        plt.axis('off')
        plt.title('Ground truth')

        plt.subplot(n_sensors, 4, 4 * ss + 2)
        plt.imshow(I_algo_p[ss, bb, 0])
        #plt.clim(0, maxc)
        plt.colorbar()
        plt.axis('off')
        err = sum(sum((I_gt[ss] - I_algo_p[ss, bb, 0])**2))
        plt.title('Initial output, error = ' + str(round(err, 7)))

        plt.subplot(n_sensors, 4, 4 * ss + 3)
        plt.imshow(I_algo_p[ss, bb, t_2])
        #plt.clim(0, maxc)
        plt.colorbar()
        plt.axis('off')
        err = sum(sum((I_gt[ss] - I_algo_p[ss, bb, t_2])**2))
        plt.title('iter = ' + str(t_2) + ', error = ' + str(round(err, 7)))    

        plt.subplot(n_sensors, 4, 4 * ss + 4)
        plt.imshow(I_algo_p[ss, bb, -1])
        #plt.clim(0, maxc)
        plt.colorbar()
        plt.axis('off')
        err = sum(sum((I_gt[ss] - I_algo_p[ss, bb, -1])**2)) 
        plt.title('Final output, error = ' +  str(round(err, 7)))

    plt.savefig(out_path + out_name + ' images from sensors.png', dpi=300)


scheduler.stop()    

print('end')
