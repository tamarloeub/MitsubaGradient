import mitsuba
from mitsuba.core import *
from mitsuba.render import *

import os, sys
import multiprocessing
import datetime
import numpy as np
from struct import pack, unpack
import re

from mtspywrapper import *#pyMedium, pyParallelRaySensor, pySolarEmitter

from mitsuba.core import PluginManager

class pyScene(object):
    def __init__(self):
        self._scene = Scene()         
        self._pmgr  = PluginManager.getInstance()
        self._medium = None
        
    def get_scene_beta(self):
        if self._medium == None:
            return None
        medium = self._medium
        return medium.get_density_data()
    
    def create_sensor_film_sampler(self, origin, target, up):        
        # Create a sensor, film & sample generator
        sensor = pySensor()
        sensor.set_sensor_type('radiancemeter')
        sensor.set_to_world(origin, target, up)
        objects = sensor.sensor_to_mitsuba()
        
        # Set the sensor, film & sample generator
        self._scene.addChild(objects)
               
    def create_integrator(self):
        # Create integrator
        integrator = self._pmgr.create({
            'type' : 'volpath_simple', 
            'maxDepth' : -1    
        })
               
        # Set the integrator
        self._scene.addChild(integrator)
        
    def create_emitter(self):
        # Create a light source
        lightSource = self._pmgr.create({
            'type' : 'directional',
            'direction' : Vector(0, 0, -1),
            'intensity' : Spectrum(1)
        })
    
        # Add a light source
        self._scene.addChild(lightSource)
            
    def create_medium(self, beta=()):    
        # Create medium with bounding box
        self._medium = pyMedium()
        self._medium.set_phase(0.85)
        self._medium.set_albedo(1)
    
        # Define the extinction field (\beta) in [km^-1]
        bounding_box = [-1, -1, -1, 1, 1, 1]   # [xmin, ymin, zmin, xmax, ymax, zmax] in km units 
        
        if beta == ():
            res                   = [2, 2, 2]               # [xres, yrex, zres]
            geometrical_thickness = bounding_box[5] - bounding_box[2]
            tau  = 0.3 * geometrical_thickness * np.ones(res)
            beta = tau / geometrical_thickness    
    
        self._medium.set_density(beta, bounding_box)
        medium_str = self._medium.medium_to_mitsuba()  
        
        # Set bounding box
        self._scene.addChild(self._medium.bounding_box_to_mitsuba())    
    
        # Set medium
        self._scene.addChild(medium_str)    
        
    def create_scene(self, beta=(), origin=None, target=None, up=None):     
        if (origin is None) and (target is None) and (up is None):
            origin = Point(0, 0, 3)
            target = Point(0, 0, 1)
            up     = Vector(1, 0, 0)    
        else:
            assert (origin is not None) and (target is not None) or (up is not None), "One of teh toWorld points is not define (origin \ target \ up)"
            
        self.create_sensor_film_sampler(origin, target, up)        
        self.create_integrator()
        self.create_emitter()
        self.create_medium(beta)
        
        self._scene.configure()        
        
        return self._scene
        
    def create_scene_with_different_density(self, beta):
        self.create_scene(beta)
        return self._scene
    
    def create_scene_with_different_sensor_position(self, origin, target, up):
        self.create_scene(self._medium.get_density_data(), origin, target, up)
        return self._scene    

def getGradFromOutputFile(filename):
    f = open(filename)
    lines = f.readlines()
    vals  = [ re.sub('[\[\],]', '', ' '.join(line.split()[4:7])) for line in lines ]
    
    grad = np.zeros((len(vals), 3)) # Spectrum, one pixel
    for grid_point in range(len(vals)):
        grad[grid_point] = [float(val) for val in vals[grid_point].split()]
    
    f.close()
    return grad
    
#def create_scene():
    #pmgr  = PluginManager.getInstance()
    #scene = Scene()
    
    ## Create a sensor, film & sample generator
    #objects = pmgr.create({
        #'type' : 'radiancemeter',
        #'toWorld' : Transform.lookAt(
            #Point(0, 0, 3),   # Camera origin
            #Point(0, 0, 1),     # Camera target
            #Vector(1, 0, 0)     # 'up' vector
            #),
        #'film' : {
            #'type' : 'mfilm'
        #},
        #'sampler' : {
            #'type' : 'ldsampler',
            #'sampleCount' : 4096
        #}
    #})
    
    ## Create integrator
    #integrator = pmgr.create({
        #'type' : 'volpath_simple', 
        #'maxDepth' : -1    
    #})
    
    ## Create a light source
    #lightSource = pmgr.create({
        #'type' : 'directional',
        #'direction' : Vector(0, 0, -1),
        #'intensity' : Spectrum(1)
    #})
    
    ## Create medium with bounding box
    #medium = pyMedium()
    #medium.set_phase(0.85)
    #medium.set_albedo(1)
    
    ## Define the extinction field (\beta) in [km^-1]
    #res                   = [2, 2, 2]               # [xres, yrex, zres]
    #bounding_box          = [-1, -1, -1, 1, 1, 1]   # [xmin, ymin, zmin, xmax, ymax, zmax] in km units 
    #geometrical_thickness = bounding_box[5] - bounding_box[2]
    #tau  = 0.9 * geometrical_thickness * np.ones(res)
    #beta = tau / geometrical_thickness    
    
    #medium.set_density(beta, bounding_box)
    #medium_str = medium.medium_to_mitsuba()       
    
    ## Set the sensor, film & sample generator
    #scene.addChild(objects)
    
    ## Set the integrator
    #scene.addChild(integrator)
    
    ## Add a light source
    #scene.addChild(lightSource)
    
    ## Set bounding box
    #scene.addChild(medium.bounding_box_to_mitsuba())    
    
    ## Set medium
    #scene.addChild(medium_str)    
    
    #scene.configure()
    
    #return scene


def render_scene(scene, output_filename, f_multi):    
    queue = RenderQueue()
    
    # Create a queue for tracking render jobs
    film   = scene.getFilm()
    size   = film.getSize() 
    bitmap = Bitmap(Bitmap.ELuminance, Bitmap.EFloat32, size) # for radiance
    #bitmap = Bitmap(Bitmap.ERGB, Bitmap.EUInt8, size) # for RGB image
    blocksize = max(np.divide(max(size.x, size.y), ncores), 1)
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
    inner_grad = getGradFromOutputFile('output.txt')
    
    # End session
    queue.join() 
    #scheduler.stop()
    
    return radiance, inner_grad

## Parameters
# algorithm parameters
max_iterations  = 10 * 2
step_size       = 100 / 2
# Ground Truth:
scene_gt = pyScene()
scene_gt.create_scene()

# mitsuba parameters
f_multi         = True
output_filename = 'renderedResult'
if f_multi: # Set parallel job or run on 1 cpu only
    ncores = 1
else:
    ncores = multiprocessing.cpu_count()

# Start up the scheduling system with one worker per local core
scheduler = Scheduler.getInstance()
for i in range(0, ncores):
    scheduler.registerWorker(LocalWorker(i, 'wrk%i' % i))
scheduler.start()

#sceneResID = scheduler.registerResource(scene_gt._scene)

## Render Ground Truth Scene
radiance_gt, _ = render_scene(scene_gt._scene, output_filename, f_multi)
#scheduler.stop()
beta_gt = scene_gt.get_scene_beta()
beta0 = np.ones(beta_gt.shape) * 0.1
# for now beta is not a Spectrum:
inner_grad_float = np.zeros((beta_gt.shape[0], 1))
    
# Gradient descent loop
beta = beta0

for iteration in range(max_iterations):
    print(iteration)
    print(beta)
    print(inner_grad_float)
    # Create scene with given beta
    algo_scene = pyScene()
    algo_scene.create_scene_with_different_density(beta)   
    
    [ radiance, inner_grad ] = render_scene(algo_scene._scene, output_filename, f_multi)
    # for now beta is not a Spectrum:
    inner_grad_float = np.zeros((inner_grad.shape[0], 1))    
    for i in range(inner_grad.shape[0]):
        inner_grad_float[i] = np.mean(inner_grad[i])
    
    cost_grad     = inner_grad_float * (radiance - radiance_gt)
    cost_grad_mat = np.reshape(cost_grad, beta.shape, 'C') #CHECK - I thins it's column stack but it's not C style 'F' ot 'C'
    beta         -= step_size * cost_grad_mat
    print(radiance)
    # Print some statistics about the rendering process
    print(Statistics.getInstance().getStats())