import mitsuba
from mitsuba.core import *
from mitsuba.render import *

import os, sys
import multiprocessing
import datetime
import numpy as np
from struct import pack, unpack

from mtspywrapper import pyMedium, pyParallelRaySensor, pySolarEmitter

def create_scene():
    pmgr  = PluginManager.getInstance()
    scene = Scene()
    
    # Create a sensor, film & sample generator
    objects = pmgr.create({
        'type' : 'radiancemeter',
        'toWorld' : Transform.lookAt(
            Point(0, 0, 3),   # Camera origin
            Point(0, 0, 1),     # Camera target
            Vector(1, 0, 0)     # 'up' vector
            ),
        'film' : {
            'type' : 'mfilm'
        },
        'sampler' : {
            'type' : 'ldsampler',
            'sampleCount' : 4096
        }
    })
    
    # Create integrator
    integrator = pmgr.create({
        'type' : 'volpath_simple', 
        'maxDepth' : -1    
    })
    
    # Create a light source
    lightSource = pmgr.create({
        'type' : 'directional',
        'direction' : Vector(0, 0, -1),
        'intensity' : Spectrum(1)
    })
    
    # Create medium with bounding box
    medium = pyMedium()
    medium.set_phase(0.85)
    medium.set_albedo(1)
    
    # Define the extinction field (\beta) in [km^-1]
    res                   = [2, 2, 2]               # [xres, yrex, zres]
    bounding_box          = [-1, -1, -1, 1, 1, 1]   # [xmin, ymin, zmin, xmax, ymax, zmax] in km units 
    geometrical_thickness = bounding_box[5] - bounding_box[2]
    tau  = 0.9 * geometrical_thickness * np.ones(res)
    beta = tau / geometrical_thickness    
    
    medium.set_density(beta, bounding_box)
    medium_str = medium.medium_to_mitsuba()       
    
    # Set the sensor, film & sample generator
    scene.addChild(objects)
    
    # Set the integrator
    scene.addChild(integrator)
    
    # Add a light source
    scene.addChild(lightSource)
    
    # Set bounding box
    scene.addChild(medium.bounding_box_to_mitsuba())    
    
    # Set medium
    scene.addChild(medium_str)    
    
    scene.configure()
    
    return scene


def render_scene(scene, queue, output_filename, f_multi):    
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
    
    radiance = np.array(bitmap.buffer()) 
    
    return radiance

## Parameters
# algorithm parameters
max_iterations  = 1

# mitsuba parameters
f_multi         = False
output_filename = 'renderedResult'
# Set parallel job or run on 1 cpu only
if f_multi:
    ncores = 1
else:
    ncores = multiprocessing.cpu_count()
   
    
# Gradient descent loop
for iteration in range(max_iterations):
    # Create scene
    scene = create_scene()   
    
    scheduler = Scheduler.getInstance()
    # Start up the scheduling system with one worker per local core
    for i in range(0, ncores):
        scheduler.registerWorker(LocalWorker(i, 'wrk%i' % i))
    scheduler.start()
    
    queue    = RenderQueue()
    radiance = render_scene(scene, queue, output_filename, f_multi)
    
    # End session
    queue.join() 
    print(radiance)
    # Print some statistics about the rendering process
    print(Statistics.getInstance().getStats())