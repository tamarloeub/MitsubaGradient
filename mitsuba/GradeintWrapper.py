import mitsuba
from mitsuba.core import *
from mitsuba.render import *

import os, sys
import multiprocessing
import datetime
import numpy as np
from struct import pack, unpack
import re, time

from mtspywrapper import *#pyMedium, pyParallelRaySensor, pySolarEmitter

from mitsuba.core import PluginManager

def get_grad_from_output_file(filename):
    f = open(filename)
    lines = f.readlines()
    vals  = [ re.sub('[\[\],]', '', ' '.join(line.split()[4:7])) for line in lines ]
    
    grad = np.zeros((len(vals), 3)) # Spectrum, one pixel
    for grid_point in range(len(vals)):
        grad[grid_point] = [float(val) for val in vals[grid_point].split()]
    
    f.close()
    return grad
    
def render_scene(scene, output_filename, n_cores):    
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
    inner_grad = get_grad_from_output_file('output.txt')
    
    # End session
    queue.join() 
    
    return radiance, inner_grad


### Loading from an XML file
#def sceneLoadFromFile(xml_filename):
    ## Get a reference to the thread's file resolver
    #fileResolver = Thread.getThread().getFileResolver()
    
    ## Register any searchs path needed to load scene resources (optional)
    #fileResolver.appendPath('Myscenes')
    
    ## Load the scene from an XML file
    #scene        = SceneHandler.loadScene(fileResolver.resolve(xml_filename), StringMap())    
    #return scene
#xml_filename = "medium1Voxel.xml"
#scene        = sceneLoadFromFile(xml_filename)

## Parameters
# algorithm parameters
max_iterations = 150
#step_size      = 5#.5
n_unknowns     = 1
beta_gt_factor = np.array([1, 2, 5, 10])
beta0_diff     = np.array([-2, -0.5, 0.5, 2])
grid_size = 8

# optimizer parameters - ADAM
alpha   = 0.01
beta1   = 0.9
beta2   = 0.999
epsilon = 1e-8
first_moment  = 0 #m0
second_moment = 0 #v0

# mitsuba parameters
n_sensors     = 1#4
scene_gt      = [ None ] * n_sensors # create an empty list
algo_scene    = [ None ] * n_sensors # create an empty list
radiance_gt   = np.zeros((len(beta_gt_factor), n_sensors, 1))
sensors_pos   = [ None ] * n_sensors # create an empty list
algo_radiance = np.zeros((len(beta_gt_factor), len(beta0_diff), max_iterations, n_sensors))
betas         = np.zeros((len(beta_gt_factor), len(beta0_diff), max_iterations, n_unknowns))
time_per_iter = np.zeros((len(beta_gt_factor), len(beta0_diff), max_iterations))
inner_grad    = np.zeros((len(beta_gt_factor), len(beta0_diff), max_iterations, n_sensors, grid_size, 3))

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

for bb_gt in range(len(beta_gt_factor)):
    beta0_factor   = beta_gt_factor[bb_gt] + beta0_diff
    
    # Ground Truth:
    scene_gt[0]    = pyScene()
    scene_gt[0].create_new_scene(beta=beta_gt_factor[bb_gt], nSamples=4096)
    sensors_pos[0] = scene_gt[0]._sensor.get_world_points()
    beta_gt        = scene_gt[0].get_scene_beta()
    
    # Render Ground Truth Scene
    radiance_gt[bb_gt][0], _ = render_scene(scene_gt[0]._scene, output_filename, n_cores)
    
    ## Add more sensors
    #sensors_pos[1] = { 'origin' : Point(0, 0, -3), 
                       #'target' : sensors_pos[0]['target'], 
                       #'up'     : Vector(-1, 0, 0) }
    #sensors_pos[2] = { 'origin' : Point(0, 3, 0), 
                       #'target' : sensors_pos[0]['target'], 
                       #'up'     : Vector(0, 0, 1) }
    #sensors_pos[3] = { 'origin' : Point(0, -3, 0), 
                       #'target' : sensors_pos[0]['target'], 
                       #'up'     : Vector(0, 0, -1) }
    
    #scene_gt[1]    = scene_gt[0].copy_scene_with_different_sensor_position(sensors_pos[1]['origin'], sensors_pos[1]['target'], sensors_pos[1]['up'])
    #scene_gt[2]    = scene_gt[0].copy_scene_with_different_sensor_position(sensors_pos[2]['origin'], sensors_pos[2]['target'], sensors_pos[2]['up'])
    #scene_gt[3]    = scene_gt[0].copy_scene_with_different_sensor_position(sensors_pos[3]['origin'], sensors_pos[3]['target'], sensors_pos[3]['up'])
    
    
    #radiance_gt[1], _ = render_scene(scene_gt[1]._scene, output_filename, n_cores)
    #radiance_gt[2], _ = render_scene(scene_gt[2]._scene, output_filename, n_cores)
    #radiance_gt[3], _ = render_scene(scene_gt[3]._scene, output_filename, n_cores)
    
    #print(radiance_gt)
    
    for bb in range(len(beta0_diff)):
        beta0  = np.ones(beta_gt.shape) * beta0_factor[bb]
        
        # for now beta is not a Spectrum:
        inner_grad_float = np.zeros((n_unknowns, 1))
        
        # Gradient descent loop
        beta = np.copy(beta0)
        
        for iteration in range(max_iterations):
            #print(iteration)
            #print(beta)
            start = time.time()
            cost_grad = np.zeros((n_unknowns, 1))
            
            for ss in range(n_sensors):
                # Create scene with given beta
                algo_scene[ss] = scene_gt[ss].copy_scene_with_different_density(beta)
                #[ algo_radiance[bb_gt][bb][iteration][ss], 
                  #inner_grad[bb_gt][bb][iteration][ss] ] = render_scene(algo_scene[ss]._scene, output_filename, n_cores)
                [ algo_radiance[bb_gt][bb][iteration][ss], _ ] = render_scene(algo_scene[ss]._scene, output_filename, n_cores)
                [ _, inner_grad[bb_gt][bb][iteration][ss] ]    = render_scene(algo_scene[ss]._scene, output_filename, n_cores)
                
                # beta is not a Spectrum, for now:
                # one unknown
                tmp = inner_grad[bb_gt][bb][iteration]
                inner_grad_float = np.mean(np.mean(tmp))
                
                cost_grad += inner_grad_float * np.mean( radiance_gt[bb_gt][ss] - algo_radiance[bb_gt][bb][iteration][ss] )
                
            #print(algo_radiance[bb_gt][bb][iteration])
                
            #cost_grad_mat = np.reshape(cost_grad, beta.shape, 'C') #CHECK - I thins it's column stack but it's not C style 'F' ot 'C'
            cost_grad_mat = np.ones(beta.shape) * cost_grad
            
            ## ADAM implementation
            first_moment  = beta1 * first_moment  + (1 - beta1) * cost_grad_mat
            second_moment = beta2 * second_moment + (1 - beta2) * cost_grad_mat**2
            
            first_moment_bar  = first_moment  / ( 1 - beta1**(iteration + 1) )
            second_moment_bar = second_moment / ( 1 - beta2**(iteration + 1) )
            
            beta     -= alpha * first_moment_bar / (np.sqrt(second_moment_bar) + epsilon)    
            
            ## for fixed step size
            #beta     += step_size * cost_grad_mat
            
            betas[bb_gt][bb][iteration] = np.mean(np.mean(beta))
            if betas[bb_gt][bb][iteration] <= 0:
                beta = np.ones(beta.shape) * 0.1
                
            #if iteration == 0:
            time_per_iter[bb_gt][bb][iteration] = time.time() - start + time_per_iter[bb_gt][bb][iteration - 1]
            # Print some statistics about the rendering process
            #print(Statistics.getInstance().getStats())
            
scheduler.stop()
    
#print(algo_radiance)
#print(betas)

import matplotlib.pyplot as plt
iters = np.linspace(1, max_iterations, max_iterations)

for bb_gt in range(len(beta_gt_factor)):
    beta0_factor = beta_gt_factor[bb_gt] + beta0_diff
    for bb in range(len(beta0_diff)):
        diff         = np.squeeze(algo_radiance[bb_gt][bb] - radiance_gt[bb_gt] * np.ones(algo_radiance[bb_gt][bb].shape))
        cost         = 0.5 * diff**2
        tmp          = inner_grad[bb_gt][bb]
        gradient     = (np.mean(np.squeeze(np.mean(tmp, 2)), 1)) * diff
        betas_err    = abs(betas[bb_gt][bb] - beta_gt_factor[bb_gt]) / beta_gt_factor[bb_gt] * 100
        
        plt.figure(1)
        plt.subplot(4, 1, 1)
        plt.plot(time_per_iter[bb_gt][bb], cost)
        plt.title('Cost value, original beta = ' + str(beta_gt_factor[bb_gt]) + ', starting with beta0 = ' + str(beta0_factor[bb]) )
        plt.grid(True)
        plt.xlim(left=0)
        
        plt.subplot(4, 1, 2)
        plt.plot(time_per_iter[bb_gt][bb], gradient)
        plt.title('Gradient value')
        plt.grid(True)
        plt.xlim(left=0)
        
        plt.subplot(4, 1, 3)
        plt.plot(time_per_iter[bb_gt][bb], betas[bb_gt][bb])
        plt.title('Beta')
        plt.grid(True)
        plt.xlim(left=0)
        
        plt.subplot(4, 1, 4)
        plt.plot(time_per_iter[bb_gt][bb], betas_err)
        plt.title('Beta error')
        plt.xlabel('Running time [sec]')
        plt.ylabel('error [%]')
        plt.grid(True)
        plt.xlim(left=0)
        
        plt.show()

print('end')