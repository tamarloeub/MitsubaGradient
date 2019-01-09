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

class pyScene(object):
    def __init__(self):
        self._scene  = Scene()         
        self._pmgr   = PluginManager.getInstance()
        self._medium = None
        self._sensor = None
        
        self._scene_set      = False        
        self._integrator_str = None
        self._emitter_str    = None
    
    def copy_scene(self):
        new_scene = pyScene()
        new_scene._medium = self._medium
        new_scene._sensor = self._sensor
        
        new_scene._scene_set      = self._scene_set
        new_scene._integrator_str = self._integrator_str
        new_scene._emitter_str    = self._emitter_str
        return new_scene
        
        
    def get_scene_beta(self):
        if self._medium == None:
            return None
        medium = self._medium
        return medium.get_density_data()
    
    def set_sensor_film_sampler(self, origin, target, up, nSamples):        
        # Create a sensor, film & sample generator
        self._sensor = pySensor()
        self._sensor.set_sensor_type('radiancemeter')
        self._sensor.set_sampler(nSamples)
        self._sensor.set_to_world(origin, target, up)
               
    def set_integrator(self):
        # Create integrator
        self._integrator_str = self._pmgr.create({
            'type' : 'volpath_simple', 
            'maxDepth' : -1    
        })

    def set_emitter(self):
        # Create a light source
        self._emitter_str = self._pmgr.create({
            'type' : 'directional',
            'direction' : Vector(0, 0, -1),
            'intensity' : Spectrum(1)
        })
            
    def set_medium(self, beta):    
        # Create medium with bounding box
        self._medium = pyMedium()
        self._medium.set_phase(0.85)
        self._medium.set_albedo(1)
    
        # Define the extinction field (\beta) in [km^-1]
        bounding_box = [-1, -1, -1, 1, 1, 1]   # [xmin, ymin, zmin, xmax, ymax, zmax] in km units 
        
        beta_parameter = 0.9
        if isinstance(beta, int):
            beta = float(beta)
        
        if isinstance(beta, float):
            beta_parameter = beta
            beta = ()
            
        if beta == ():
            res                   = [2, 2, 2]               # [xres, yrex, zres]
            geometrical_thickness = bounding_box[5] - bounding_box[2]
            tau  = beta_parameter * geometrical_thickness * np.ones(res)
            beta = tau / geometrical_thickness    
    
        self._medium.set_density(beta, bounding_box)   
        
    def set_scene(self, beta=(), origin=None, target=None, up=None, nSamples=4096):
        if (origin is None) and (target is None) and (up is None):
            origin = Point(0, 0, 3)
            target = Point(0, 0, 1)
            up     = Vector(1, 0, 0)    
        else:
            assert (origin is not None) and (target is not None) or (up is not None), "One of teh toWorld points is not define (origin \ target \ up)"
            
        self.set_sensor_film_sampler(origin, target, up, nSamples)        
        self.set_integrator()
        self.set_emitter()
        self.set_medium(beta)
        self._scene_set = True
        
    def configure_scene(self):            
        # Set the sensor, film & sample generator
        self._scene.addChild(self._sensor.sensor_to_mitsuba())        
    
        # Set the integrator
        self._scene.addChild(self._integrator_str)

        # Set the emiter - light source
        self._scene.addChild(self._emitter_str)
        
        # Set bounding box
        self._scene.addChild(self._medium.bounding_box_to_mitsuba())    
    
        # Set medium
        self._scene.addChild(self._medium.medium_to_mitsuba())  
            
        self._scene.configure()        
        
        return self._scene
    
    def create_new_scene(self, beta=(), origin=None, target=None, up=None, nSamples=4096):
        self.set_scene(beta, origin, target, up, nSamples)
        return self.configure_scene()
        
    def copy_scene_with_different_density(self, beta):
        assert (self._scene_set is True), "Can't copy unset scene"
        new_scene = self.copy_scene()        
        new_scene.set_medium(beta)
        new_scene.configure_scene()
        return new_scene
    
    def copy_scene_with_different_sensor_position(self, origin, target, up):
        assert (self._scene_set is True), "Can't copy unset scene"
        new_scene = self.copy_scene()        
        new_scene.set_sensor_film_sampler(origin, target, up)
        new_scene.configure_scene()
        return new_scene        

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
max_iterations = 50
step_size      = 15#.5
n_unknowns     = 1

# mitsuba parameters
n_sensors     = 1#4
scene_gt      = [ None ] * n_sensors # create an empty list
algo_scene    = [ None ] * n_sensors # create an empty list
radiance_gt   = np.zeros((n_sensors, 1))
sensors_pos   = [ None ] * n_sensors # create an empty list
algo_radiance = np.zeros((max_iterations, n_sensors))
betas         = np.zeros((max_iterations, n_unknowns))
time_per_iter = np.zeros((max_iterations, 1))

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

# Ground Truth:
scene_gt[0] = pyScene()
scene_gt[0].create_new_scene(beta=4.0, nSamples=4096*4)

sensors_pos[0] = scene_gt[0]._sensor.get_toWord_points()

# Render Ground Truth Scene
radiance_gt[0], _ = render_scene(scene_gt[0]._scene, output_filename, n_cores)

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


print(radiance_gt)

beta_gt = scene_gt[0].get_scene_beta()
beta0 = np.ones(beta_gt.shape) * 5

# for now beta is not a Spectrum:
inner_grad_float = np.zeros((n_unknowns, 1))
inner_grad    = np.zeros((max_iterations, n_sensors, np.prod(beta_gt.shape), 3))

# Gradient descent loop
beta = beta0

# ADAM parameters
alpha   = 0.001
beta1   = 0.9
beta2   = 0.999
epsilon = 1e-8
first_moment  = 0
second_moment = 0

for iteration in range(max_iterations):
    print(iteration)
    print(beta)
    start = time.time()
    #cost_grad = np.zeros((np.prod(beta.shape), 1))
    cost_grad = np.zeros((n_unknowns, 1))

    for ss in range(n_sensors):
        # Create scene with given beta
        algo_scene[ss] = scene_gt[ss].copy_scene_with_different_density(beta)
        [ algo_radiance[iteration][ss], inner_grad[iteration][ss] ] = render_scene(algo_scene[ss]._scene, output_filename, n_cores)
        
        # beta is not a Spectrum, for now:
        # one unknown
        inner_grad_float = np.mean(np.mean(inner_grad[iteration][ss]))
        #inner_grad_float = np.zeros((inner_grad[ss].shape[0], 1))
        #for jj in range(inner_grad[ss].shape[0]):
            #inner_grad_float[jj] = np.mean(inner_grad[ss][jj])
        
        #cost_grad += inner_grad_float * (algo_radiance[ss] - radiance_gt[ss])        
        cost_grad += inner_grad_float * np.mean(algo_radiance[iteration][ss] - radiance_gt[ss])        
        
    print(algo_radiance[iteration])
        
    #cost_grad_mat = np.reshape(cost_grad, beta.shape, 'C') #CHECK - I thins it's column stack but it's not C style 'F' ot 'C'
    cost_grad_mat = np.ones(beta.shape) * cost_grad
    
    ## ADAM implementation
    #first_moment = beta1 * first_moment + (1 - beta1) * cost_grad_mat
    
    beta         -= step_size * cost_grad_mat
    
    betas[iteration] = np.mean(np.mean(beta))
    if betas[iteration] <= 0:
        beta = np.ones(beta.shape) * 0.1
        
    #if iteration == 0:
    time_per_iter[iteration] = time.time() - start + time_per_iter[iteration - 1]
    #else:
        #time_per_iter[iteration] = time.time() - time_per_iter[iteration - 1]
    # Print some statistics about the rendering process
    #print(Statistics.getInstance().getStats())
scheduler.stop()

print(algo_radiance)
print(betas)

import matplotlib.pyplot as plt
iters = np.linspace(1, max_iterations, max_iterations)

diff         = np.squeeze(algo_radiance - radiance_gt * np.ones(algo_radiance.shape))
cost         = 0.5 * diff**2
gradient     = (np.mean(np.squeeze(np.mean(inner_grad, 2)), 1)) * diff
mean_gt_beta = np.mean(beta_gt.flatten())
betas_err    = abs(betas - mean_gt_beta) / mean_gt_beta * 100

plt.figure(1)
plt.subplot(4, 1, 1)
plt.plot(time_per_iter, cost)
plt.title('Cost value, original beta = 0.9, starting with beta0 = 0.1')
plt.grid(True)
plt.xlim(left=0)

plt.subplot(4, 1, 2)
plt.plot(time_per_iter, gradient)
plt.title('Gradient value')
plt.grid(True)
plt.xlim(left=0)

plt.subplot(4, 1, 3)
plt.plot(time_per_iter, betas)
plt.title('Beta')
plt.grid(True)
plt.xlim(left=0)

plt.subplot(4, 1, 4)
plt.plot(time_per_iter, betas_err)
plt.title('Beta error')
plt.xlabel('Running time [sec]')
plt.ylabel('error [%]')
plt.grid(True)
plt.xlim(left=0)

plt.show()

print('end')