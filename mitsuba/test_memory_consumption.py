import mitsuba
from mitsuba.core import *
from mitsuba.render import *
from mitsuba.core import PluginManager

import multiprocessing
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from mtspywrapper import *

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
    if not Igt:
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
    #outFile = FileStream('renderedResult.png', FileStream.ETruncReadWrite)
    #bitmap.write(Bitmap.EPNG, outFile)
    #outFile.close()

    radiance   = np.array(bitmap.buffer()) 
    if sum(Igt) == 0:
        inner_grad = None
    else:
        inner_grad = np.array(scene.getTotalGradient())
    #inner_grad_tmp = scene.getTotalGradient()
    #inner_grad = np.reshape(inner_grad_tmp, [grid_size, n_pixels_w * n_pixels_h], 'F')
    #inner_grad = np.reshape(inner_grad_tmp, [grid_size, n_pixels_w, n_pixels_h])

    return radiance, inner_grad




@profile
def forward_Mitsuba(ii):
    output_filename = 'renderedResult'

    x_spacing = 0.02 # in km
    y_spacing = 0.02 # in km
    z_spacing = 0.04 # in km
    
    ## aviad's crop
    beta_gt = np.load('CloudsSim/jpl/jpl_ext.npy')
    # npad is a tuple of (n_before, n_after) for each dimension
    npad = ((1, 1), (1, 1), (1, 1))
    
    beta_gt     = np.pad(beta_gt, pad_width=npad, mode='constant', constant_values=0.01)
    
    # manually creates mask
    ind0x, ind0y, ind0z    = np.where(beta_gt < 2)

    beta_gt[ind0x, ind0y, ind0z] = 0.01
    
    npad    = ((3, 3), (1, 1), (6, 7))
    beta_gt = np.pad(beta_gt, pad_width=npad, mode='constant', constant_values=0.01)
    
    [ nx, ny, nz ] = beta_gt.shape     
    
    # bounding box = [xmin, ymin, zmin, xmax, ymax, zmax] in km units 
    bounds = [ 0, 0, 0, nx * x_spacing, ny * y_spacing, nz * z_spacing ]
    
    sensor_pos = [None]  # create an empty list
    
    Np_vector = np.array([512 * 2])  # photons_total / n_pixels
    gt_Np_fac = 8 / 2
    
    # scenes params
    TOA      = bounds[5]
    up_const = np.array([-1, 0, 0])
    
    ## for jpl's cloud
    H       = z_spacing * nz / 12  # 4.
    t_const = np.array([bounds[3] / 2., bounds[4] / 2., TOA * 2. / 3.])
    
    o = np.array([round(bounds[3], 1) / 2., round(bounds[4], 1) / 2., TOA + H])
    t = t_const
    u = up_const
    
    # FOV calc: fov is set by axis x
    max_medium    = np.array([bounds[3], bounds[4], bounds[5]])
    min_medium    = np.array([bounds[0], bounds[1], bounds[2]])
    
    L = np.max([max_medium[0] - min_medium[0], max_medium[1] - min_medium[1]]) / 2  # camera's FOV covers the whole medium
    fov_rad = 2 * np.arctan(L / ((TOA + H) / 4))
    fov_deg = 180 * fov_rad / np.pi

    beta = zoom(beta_gt, zoom_ms[ii], mode='nearest')
    if beta[beta < 0.01].size:
        beta[beta < 0.01] = 0.01
    
    [ nx, ny, nz ] = beta.shape
    grid_size = np.prod(beta.shape)
    
    n_pixels_w = np.max([nx, ny]) # resolution ~= spacing / 2 = 10 meters
    n_pixels_h = np.max([nx, ny])
    
    newUp, _   = transformLookAt(o, t, u)
    sensor_pos = {'origin': Point(o[0],  o[1], o[2]),
                  'target': Point(t[0],  t[1], t[2]),
                  'up'    : Vector(newUp[0], newUp[1], newUp[2])}

    scene = pyScene()
    scene.create_new_scene(beta=beta, g=0.85, origin=sensor_pos['origin'], target=sensor_pos['target'],
                              up=sensor_pos['up'], nSamples=int(Np_vector[0] * gt_Np_fac), sensorType='perspective',
                              bounding_box=bounds, fov=fov_deg, width=n_pixels_w, height=n_pixels_h, rrDepth=None)

    Is, _ = render_scene(scene._scene, output_filename, n_cores, grid_size, n_pixels_w, n_pixels_h)


## flags
if __name__ == '__main__':
    
    render_gt_f = False
    parallel_f  = True
    crop_f      = False
    enlarged_f  = False
    debug_f     = True
    
    if parallel_f:  # Set parallel job or run on 1 cpu only
        n_cores = multiprocessing.cpu_count() # T if others are running
    else:
        n_cores = 1
    
    # Start up the scheduling system with one worker per local core
    scheduler = Scheduler.getInstance()
    for i in range(0, n_cores):
        scheduler.registerWorker(LocalWorker(i, 'wrk%i' % i))
    scheduler.start()
    
    #Is_ms = zoom(Is, (1, 1 / grid_ms[stage, 0], 1 / grid_ms[stage, 1]))
    zoom_ms = np.array([0.25, 0.5, 1, 2, 4, 8])  #grid_ms[stage - 1] / grid_ms[stage]  # * np.ones(3)
    
    ii = 3 #4
    
    start = time.time()
    
    forward_Mitsuba(ii)
    print('end')
    
    end = time.time()
    print(( end - start )/60.)
    
    scheduler.stop()    
    
    
