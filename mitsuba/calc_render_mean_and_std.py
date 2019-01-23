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

from mtspywrapper import *

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
    
    # End session
    queue.join() 
    
    return radiance


## Loading from an XML file
def sceneLoadFromFile(xml_filename):
    # Get a reference to the thread's file resolver
    fileResolver = Thread.getThread().getFileResolver()
    
    # Register any searchs path needed to load scene resources (optional)
    fileResolver.appendPath('Myscenes')
    
    # Load the scene from an XML file
    scene        = SceneHandler.loadScene(fileResolver.resolve(xml_filename), StringMap())    
    return scene


xml_filename = "medium1Voxel.xml"
scene        = sceneLoadFromFile(xml_filename)

n_times  = 1000
radiance = np.zeros((n_times, 1))

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

for ii in range(n_times):
    radiance[ii] = render_scene(scene, output_filename, n_cores)
    
    
mean_radiance = np.mean(radiance)
std_radiance  = np.std(radiance)

print("mean radiance: " + str(mean_radiance))
print("std radiance: " + str(std_radiance))