from mtspywrapper import *

# Mitsuba imports
from mitsuba.core import *
from mitsuba.render import Scene, RenderQueue, RenderJob
from mitsuba.core import PluginManager, Vector, Point, Transform, Spectrum

# Other imports 
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from tqdm import tqdm
import multiprocessing


def is_out_valid(outM, mean, std):
    with open (outM, "r") as myfile:
        data=myfile.readlines()
    data_string = data[0][data[0].find('['):data[0].find(']')+1]
    data_string = data_string.replace("[","").replace("]","")    
    data_val    = float(data_string)
    if (data_val <= mean + std) and (data_val >= mean - std):
        return True
    else:
        return False
    
def is_beta_valid(pscene, beta):
    if np.prod(beta == pscene.get_scene_beta()):
        return True
    else:
        return False

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
    
    radiance   = np.array(bitmap.buffer()) 
    
    # End session
    queue.join() 
    
    return radiance



################### Wrapper Test ###################
n_cores   = 30
scheduler = Scheduler.getInstance()

for i in range(0, n_cores):
    scheduler.registerWorker(LocalWorker(i, 'wrk%i' % i))
scheduler.start()

vol_shape = np.array([2, 2, 2])

## Random Beta - float ##
beta   = np.random.rand(1) * 99 + 1
pscene = pyScene()
pscene.create_new_scene(beta=beta, nSamples=256)
radiance_float   = render_scene(pscene._scene, "out_bfloat.m", n_cores)
float_beta_valid = is_beta_valid(pscene, beta)

##Random beta - grid 
beta   = np.random.rand(vol_shape[0], vol_shape[1], vol_shape[2]) * 99 + 1
pscene = pyScene()
pscene.create_new_scene(beta=beta,nSamples=256)
radiance_grid   = render_scene(pscene._scene, "out_bgrid.m", n_cores)
grid_beta_valid = is_beta_valid(pscene, beta)

scheduler.stop()

################### XML Test ###################

mean = 0.09506165280565619
std  = 0.011925707400626996

xml_path = "data/medium1Voxel.xml"
cmd      = "mitsuba " + xml_path + " -q"
os.system(cmd)

xml_res = is_out_valid("data/medium1Voxel.m", mean, std)

print("")
print("---------------------------------------")
print("Basic Tests Before Check-in Output:")
print("---------------------------------------")

if (radiance_float > 0) and (float_beta_valid):
    print("constvol beta test SUCCESS")
else:
    print("constvol beta test FAILED")
    if radiance_float <= 0:
        print("radiance <= 0")
    if not float_beta_valid:
        print("beta does not match")
   
if (radiance_grid > 0) and (grid_beta_valid):
    print("gridvol beta test SUCCESS")
else:
    print("gridvol beta test FAILED")
    if radiance_grid <= 0:
        print("radiance <= 0")
    if not grid_beta_valid:
        print("beta does not match")

if xml_res:
    print("XML test SUCCESS")
else:
    print("XML test FAILED")

## Remove output files
os.remove("out_bfloat.m")
os.remove("out_bgrid.m")
os.remove("output.txt")
os.remove("data/medium1Voxel.m")
os.remove("mitsuba.vislbatch2.log")

    
    