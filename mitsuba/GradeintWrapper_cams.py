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

def transformLookAt(p, t, up):
    dir_vec = (t-p)/np.linalg.norm(t-p)
    left  = np.cross(up, dir_vec)
    left  = left/np.linalg.norm(left)
    newUp = np.cross(dir_vec, left)

    T = np.zeros([4,4])
    T[0:3,0]  = - left
    T[0:3,1]  = newUp    
    T[0:3,2] = dir_vec
    T[0:3,3] = p
    T[3,3]   = 1.0

    q = np.dot(T.transpose()[:-1,:-1] , p.transpose())
        
    T_inverse = np.zeros([4,4])
    T_inverse[0,0:3] = left
    T_inverse[1,0:3] = newUp    
    T_inverse[2,0:3] = dir_vec
    T_inverse[0:3,3] = -q
    T_inverse[3,3]   = 1.0

    return newUp, T, T_inverse



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


## Parameters
# algorithm parameters
max_iterations = 400 * 3
n_unknowns     = 4
beta_gt_factor = np.array([2])#, 5, 10]) #np.array([1, 2, 5, 10])
beta0_diff     = np.array([1, 2, 4]) #np.array([-0.5, 0, 0.5, 2, 5, 1, 2, 5])

grid_size = 8

# mitsuba parameters
n_sensors     = 4#4
n_pixels      = 1
# several Nps:
#Np_vector     = np.array([1, 2, 4, 8, 16, 32]) * 128
# single Np:
Np_vector     = np.array([512]) * 128
scene_gt      = [ None ] * n_sensors # create an empty list
algo_scene    = [ None ] * n_sensors # create an empty list
sensors_pos   = [ None ] * n_sensors # create an empty list
I_gt          = np.zeros((n_sensors, len(Np_vector), len(beta_gt_factor), 1))
I_algo        = np.zeros((n_sensors, len(Np_vector), len(beta_gt_factor), len(beta0_diff), max_iterations))
runtime       = np.zeros((len(Np_vector), len(beta_gt_factor), len(beta0_diff), max_iterations))
cost_gradient = np.zeros((len(Np_vector), len(beta_gt_factor), len(beta0_diff), max_iterations, grid_size, 3))

## 1 unknown
#betas         = np.zeros((len(Np_vector), len(beta_gt_factor), len(beta0_diff), max_iterations, n_unknowns))
## 4 unknowns
betas          = np.zeros((len(Np_vector), len(beta_gt_factor), len(beta0_diff), max_iterations, n_unknowns*2))
beta_4unknowns = np.zeros((len(Np_vector), len(beta_gt_factor), n_unknowns))
                          
f_multi         = True
output_filename = 'renderedResult2'

if f_multi: # Set parallel job or run on 1 cpu only
    n_cores = multiprocessing.cpu_count()
else:
    n_cores = 1

# Start up the scheduling system with one worker per local core
scheduler = Scheduler.getInstance()
for i in range(0, n_cores):
    scheduler.registerWorker(LocalWorker(i, 'wrk%i' % i))
scheduler.start()

additional_str = ' camera'

for nps in range(len(Np_vector)):
    for bb_gt in range(len(beta_gt_factor)):
        beta0_factor   = beta_gt_factor[bb_gt] + beta0_diff
        
        ## 4 unknowns
        tmp = [round(x, 1) for x in beta_gt_factor[bb_gt] * np.random.rand(4, 1) + np.ones([4, 1]) * 0.51]
        beta_4unknowns[0, bb_gt] = tmp
        beta_gt          = np.zeros([2, 2, 2])
        beta_gt[0, :, :] = np.reshape(tmp, [2, 2])
        beta_gt[1, :, :] = np.reshape(tmp, [2, 2])
        
        # Ground Truth:
        scene_gt[0]    = pyScene()
        
        ## 1 unknown
        #scene_gt[0].create_new_scene(beta=beta_gt_factor[bb_gt], origin=Point(0, 3, 0), target=None, up=Vector(0, 0, 1), nSamples=Np_vector[nps])
        #beta_gt        = scene_gt[0].get_scene_beta()
        ## 4 unknowns     
        scene_gt[0].create_new_scene(beta=beta_gt, sensorType='prespective', nSamples=Np_vector[nps], width=2, height=2)
        sensors_pos[0] = scene_gt[0]._sensor.get_world_points()
        
        # Render Ground Truth Scene
        I_gt[0][nps][bb_gt], _ = render_scene(scene_gt[0]._scene, output_filename, n_cores)
        
        ## Add more sensors
        #sensors_pos[1] = { 'origin' : Point(0, 0, -3),  ## NOT A GOOD POSITION!!!
                           #'target' : sensors_pos[0]['target'], 
                           #'up'     : Vector(-1, 0, 0) }
        newUp1, _, _ = transformLookAt(np.array([3, 0, 0]), 
                                      np.array([sensors_pos[0]['target'][0], sensors_pos[0]['target'][1], sensors_pos[0]['target'][2]]), 
                                      np.array([0, 1, 0]))
        sensors_pos[1] = { 'origin' : Point(3, 0, 0),
                           'target' : sensors_pos[0]['target'], 
                           'up'     : Vector(newUp1[0], newUp1[1], newUp1[2]) }
        
        newUp2, _, _ = transformLookAt(np.array([0, 3, 0]), 
                                       np.array([sensors_pos[0]['target'][0], sensors_pos[0]['target'][1], sensors_pos[0]['target'][2]]), 
                                       np.array([0, 1, 0]))        
        sensors_pos[2] = { 'origin' : Point(0, 3, 0), 
                           'target' : sensors_pos[0]['target'], 
                           'up'     : Vector(newUp2[0], newUp2[1], newUp2[2]) }
        
        newUp3, _, _ = transformLookAt(np.array([0, -3, 0]), 
                                       np.array([sensors_pos[0]['target'][0], sensors_pos[0]['target'][1], sensors_pos[0]['target'][2]]), 
                                       np.array([0, 1, 0]))        
        sensors_pos[3] = { 'origin' : Point(0, -3, 0), 
                           'target' : sensors_pos[0]['target'], 
                           'up'     : Vector(newUp3[0], newUp3[1], newUp3[2]) }
        
        scene_gt[1]    = scene_gt[0].copy_scene_with_different_sensor_position(sensors_pos[1]['origin'], sensors_pos[1]['target'], sensors_pos[1]['up'])
        scene_gt[2]    = scene_gt[0].copy_scene_with_different_sensor_position(sensors_pos[2]['origin'], sensors_pos[2]['target'], sensors_pos[2]['up'])
        scene_gt[3]    = scene_gt[0].copy_scene_with_different_sensor_position(sensors_pos[3]['origin'], sensors_pos[3]['target'], sensors_pos[3]['up'])
        
        
        I_gt[1][nps][bb_gt], _ = render_scene(scene_gt[1]._scene, output_filename, n_cores)
        I_gt[2][nps][bb_gt], _ = render_scene(scene_gt[2]._scene, output_filename, n_cores)
        I_gt[3][nps][bb_gt], _ = render_scene(scene_gt[3]._scene, output_filename, n_cores)
        
        print(beta_gt_factor[bb_gt])
        
        for bb in range(len(beta0_diff)):
            # optimizer parameters - ADAM
            alpha   = 0.01
            beta1   = 0.9
            beta2   = 0.999
            epsilon = 1e-8
            first_moment  = 0 #m0
            second_moment = 0 #v0
            
            ## 1 unknown
            #beta0  = np.ones(beta_gt.shape) * beta0_factor[bb]
            ## 4 unknowns
            beta0 = np.ones(beta_gt.shape) * beta0_diff[bb]

            #beta0  = beta_gt + np.ones(beta_gt.shape) * beta0_diff[bb] 
            #if bb > 4:
                #beta0 = np.ones(beta_gt.shape) * beta0_diff[bb]
                
            # for now beta is not a Spectrum:
            inner_grad_float = np.zeros((n_unknowns, 1))
            
            # Gradient descent loop
            beta  = np.copy(beta0)
            start = time.time()
            
            for iteration in range(max_iterations):
                #print(iteration)
                print(beta)
                ## 1 unknown
                #cost_grad = np.zeros((n_unknowns, n_pixels))
                ## 4 unknowns
                cost_grad = np.zeros((n_unknowns*2, n_pixels))
                
                for ss in range(n_sensors):
                    # Create scene with given beta
                    algo_scene[ss] = scene_gt[ss].copy_scene_with_different_density(beta)
                    [ I_algo[ss][nps][bb_gt][bb][iteration], 
                      inner_grad ] = render_scene(algo_scene[ss]._scene, output_filename, n_cores)
                    
                    ### beta is not a Spectrum, for now:
                    ## 1 unknown
                    #inner_grad_float = np.mean(np.mean(inner_grad))
                    #tmp              = (-1) * ( I_algo[ss][nps][bb_gt][bb][iteration] - I_gt[ss][nps][bb_gt] )
                    #cost_grad       += inner_grad_float * tmp[:, None] #np.matmul(inner_grad_float, (-1)* np.mean( I_algo[ss][nps][bb_gt][bb][iteration] - I_gt[ss][nps][bb_gt] ))
                    
                    ## 4 unknowns
                    inner_grad_float = np.mean(inner_grad, 1)
                    inner_grad_float[0] = (inner_grad_float[0] + inner_grad_float[4]) / 2
                    inner_grad_float[1] = (inner_grad_float[1] + inner_grad_float[5]) / 2
                    inner_grad_float[2] = (inner_grad_float[2] + inner_grad_float[6]) / 2
                    inner_grad_float[3] = (inner_grad_float[3] + inner_grad_float[7]) / 2
                    inner_grad_float[4] = inner_grad_float[0]
                    inner_grad_float[5] = inner_grad_float[1]
                    inner_grad_float[6] = inner_grad_float[2]
                    inner_grad_float[7] = inner_grad_float[3]
                    
                    tmp        = (-1) * ( I_algo[ss][nps][bb_gt][bb][iteration] - I_gt[ss][nps][bb_gt] )                    
                    cost_grad += inner_grad_float[:, None] * tmp[:, None] #np.matmul(inner_grad_float, (-1)* np.mean( I_algo[ss][nps][bb_gt][bb][iteration] - I_gt[ss][nps][bb_gt] ))
                    
                #print(I_algo[bb_gt][bb][iteration])
                
                cost_gradient[nps][bb_gt][bb][iteration] = cost_grad
                ## 1 unknown
                #cost_grad_mat = np.ones(beta.shape) * cost_grad
                ## 4 unknowns
                cost_grad_mat = np.reshape(cost_grad, beta.shape, 'C') #CHECK - I thins it's column stack but it's not C style 'F' ot 'C'

                
                ## ADAM implementation
                first_moment  = beta1 * first_moment  + (1 - beta1) * cost_grad_mat
                second_moment = beta2 * second_moment + (1 - beta2) * cost_grad_mat**2
                
                first_moment_bar  = first_moment  / ( 1 - beta1**(iteration + 1) )
                second_moment_bar = second_moment / ( 1 - beta2**(iteration + 1) )
                
                beta     -= alpha * first_moment_bar / (np.sqrt(second_moment_bar) + epsilon)    
                
                ## for fixed step size
                #beta     += step_size * cost_grad_mat
                
                ## 1 unknown
                #betas[nps][bb_gt][bb][iteration] = np.mean(np.mean(beta))
                ## 4 unknowns
                betas[nps][bb_gt][bb][iteration] = beta.flatten()

                if beta[beta <= 0].size:
                    ## 1 unknown
                    #beta = np.ones(beta.shape) * 0.1
                    ## 4 unknowns
                    beta[beta <= 0] = 0.1
                    print('fixed beta!')
                
                end = time.time()
    
                #if iteration == 0:
                runtime[nps][bb_gt][bb][iteration] = end - start 
                #else:
                    #runtime[bb_gt][bb][iteration] = end - start + runtime[bb_gt][bb][iteration - 1]
                # Print some statistics about the rendering process
                #print(Statistics.getInstance().getStats())
            
scheduler.stop()
    
#print(I_algo)
#print(betas)

import matplotlib.pyplot as plt
iters = np.linspace(1, max_iterations, max_iterations)
if alpha is not 0.01:
    alpha_s = ' alpha ' + str(alpha)
else:
    alpha_s = ''

if n_sensors > 1:
    sensors_s = ' ' + str(n_sensors) + ' sensors'
else:
    sensors_s = ''
    
### one Np
    
## 1 unknown    
#out_path = '/home/tamarl/MitsubaGradient/Gradient wrapper/plots/'
#for bb_gt in range(len(beta_gt_factor)):
    #beta0_factor = beta_gt_factor[bb_gt] + beta0_diff
    #for bb in range(len(beta0_diff)):
        #diff = 0
        #for ss in range(n_sensors):
            #diff += np.squeeze( I_gt[ss][0][bb_gt] * np.ones(I_algo[ss][0][bb_gt][bb].shape) - I_algo[ss][0][bb_gt][bb] )
        #cost      = 0.5 * diff**2
        #tmp       = cost_gradient[0][bb_gt][bb]
        #gradient  = np.mean(np.mean(tmp, 2), 1)
        #betas_err = abs(betas[0][bb_gt][bb] - beta_gt_factor[bb_gt]) / beta_gt_factor[bb_gt] * 100

        #plt.figure(figsize=(19,9))    
        
        #plt.subplot(2, 2, 1)
        #plt.plot(iters, cost, '--',  marker='o', markersize=5)
        #plt.title('Cost', fontweight='bold')              
        #plt.grid(True)
        #plt.xlim(left=0)
                
        #plt.subplot(2, 2, 3)
        #plt.plot(iters, gradient, '--',  marker='o', markersize=5)
        #plt.title('Gradient', fontweight='bold')
        #plt.xlabel('iteration')                
        #plt.grid(True)
        #plt.xlim(left=0)
    
        #plt.subplot(2, 2, 2)
        #plt.plot(iters, betas[0][bb_gt][bb], '--',  marker='o', markersize=5)
        #plt.title('Beta', fontweight='bold')
        #plt.grid(True)
        #plt.xlim(left=0)
    
        #plt.subplot(2, 2, 4)
        #plt.plot(iters, betas_err, '--',  marker='o', markersize=5)
        #plt.title('Beta error', fontweight='bold')
        #plt.xlabel('iteration')
        #plt.ylabel('[%]', fontweight='bold')
        #plt.grid(True)
        #plt.xlim(left=0)
    
        #plt.suptitle('Original beta = ' + str(beta_gt_factor[bb_gt]) + ', starting with beta0 = ' + str(beta0_factor[bb]) + ', Np = ' + str(Np_vector[0]), fontweight='bold')
        ##plt.show()
        
        #out_name = 'fix grad' + sensors_s + ' dependent grad and fwd Np '+ str(Np_vector[0]) + ' adam unknown 1 beta gt ' + str(beta_gt_factor[bb_gt]) + ' beta0 ' + str(beta0_factor[bb]) + alpha_s + additional_str +'.png'
        #plt.savefig(out_path + out_name, dpi=300)

## 4 unknowns:
out_path = '/home/tamarl/MitsubaGradient/Gradient wrapper/plots/'
for bb_gt in range(len(beta_gt_factor)):       
    for bb in range(len(beta0_diff)):
        diff = 0
        for un in range(4):
            for ss in range(n_sensors):
                diff += np.squeeze( I_gt[ss][0][bb_gt] * np.ones(I_algo[ss][0][bb_gt][bb].shape) - I_algo[ss][0][bb_gt][bb] )            
            cost      = 0.5 * diff**2
            tmp       = np.mean(cost_gradient[0][bb_gt][bb], 2)
            gradient  = (np.transpose(tmp)[un] + np.transpose(tmp)[un+4]) / 2
            tmp       = betas[0][bb_gt][bb]
            betas_un  = (np.transpose(tmp)[un] + np.transpose(tmp)[un+4]) / 2
            betas_err = abs(betas_un - beta_4unknowns[0][bb_gt][un]) / beta_4unknowns[0][bb_gt][un] * 100
    
            plt.figure(figsize=(19,9))    
            
            plt.subplot(2, 2, 1)
            plt.plot(iters, cost, '--',  marker='o', markersize=5)
            plt.title('Cost', fontweight='bold')              
            plt.grid(True)
            plt.xlim(left=0)
                    
            plt.subplot(2, 2, 3)
            plt.plot(iters, gradient, '--',  marker='o', markersize=5)
            plt.title('Gradient', fontweight='bold')
            plt.xlabel('iteration')                
            plt.grid(True)
            plt.xlim(left=0)
        
            plt.subplot(2, 2, 2)
            plt.plot(iters, betas_un, '--',  marker='o', markersize=5)
            plt.title('Beta', fontweight='bold')
            plt.grid(True)
            plt.xlim(left=0)
        
            plt.subplot(2, 2, 4)
            plt.plot(iters, betas_err, '--',  marker='o', markersize=5)
            plt.title('Beta error', fontweight='bold')
            plt.xlabel('iteration')
            plt.ylabel('[%]', fontweight='bold')
            plt.grid(True)
            plt.xlim(left=0)
            
            bb0 = beta0_diff[bb]
            #bb0 = beta_4unknowns[0][bb_gt][un] + beta0_diff[bb]
            #if bb > 4:
                #bb0 = beta0_diff[bb]
            plt.suptitle('Original beta = ' + str(beta_4unknowns[0][bb_gt][un]) + ', starting with beta0 = ' + str(bb0) + ', Np = ' + str(Np_vector[0]) + ', unknown ' + str(un + 1), fontweight='bold')
            #plt.show()
            
            out_name = 'fix grad 4 unknowns' + sensors_s + ' dependent grad and fwd Np '+ str(Np_vector[0]) + ' adam beta gt ' + str(beta_gt_factor[bb_gt]) + ' beta0 ' + str(bb0) + alpha_s + additional_str + '_' + str(un + 1) +'.png'
            plt.savefig(out_path + out_name, dpi=300)

### several Nps
#out_path = '/home/tamarl/MitsubaGradient/Gradient wrapper/plots/'
    
##for bb_gt in range(len(beta_gt_factor)):
    ##beta0_factor = beta_gt_factor[bb_gt] + beta0_diff
    ##for bb in range(len(beta0_diff)):
        ##for np_p in range(2):        
            ###numeric_grad = [None] * 3
            ##diff      = [None] * 3
            ##cost      = [None] * 3
            ##tmp       = [None] * 3
            ##gradient  = [None] * 3
            ##betas_err = [None] * 3
    
            ##plt.figure(figsize=(8,10))
            ##for np_i in range(3):
                ###numeric_grad[np_i] = np.gradient(cost)
                ##diff[np_i]      = np.squeeze(I_algo[3 * np_p + np_i][bb_gt][bb] - I_gt[3 * np_p + np_i][bb_gt] * np.ones(I_algo[3 * np_p + np_i][bb_gt][bb].shape))
                ##cost[np_i]      = 0.5 * diff[np_i]**2
                ##tmp[np_i]       = cost_gradient[3 * np_p + np_i][bb_gt][bb]
                ##gradient[np_i]  = (np.mean(np.squeeze(np.mean(tmp[np_i], 2)), 1))
                ##betas_err[np_i] = abs(betas[3 * np_p + np_i][bb_gt][bb] - beta_gt_factor[bb_gt]) / beta_gt_factor[bb_gt] * 100
            
                ##plt.subplot(3, 3, 3 * np_i + 1)
                ###plt.plot(runtime[bb_gt][bb], cost, '--',  marker='o', markersize=5)
                ##plt.plot(iters, cost[np_i], '--',  marker='o', markersize=5)
                ##plt.title('Cost, Np = ' + str(Np_vector[3 * np_p + np_i]), fontweight='bold')
                ###plt.title('Cost value')
                ###plt.ylabel('Cost value')
                ##if np_i == 2:
                    ##plt.xlabel('iteration')                
                ##plt.grid(True)
                ##plt.xlim(left=0)
                        
                ##plt.subplot(3, 3, 3 * np_i + 2)
                ###plt.plot(runtime[bb_gt][bb], gradient, '--',  marker='o', markersize=5)
                ##plt.plot(iters, gradient[np_i], '--',  marker='o', markersize=5)
                ###plt.plot(iters, numeric_grad[np_i], 'r--',  marker='o', markersize=3)        
                ###plt.title('Gradient value, Np = ' + str(Np_vector[3 * np_p + np_i]))
                ##plt.title('Gradient', fontweight='bold')
                ##if np_i == 2:
                    ##plt.xlabel('iteration')                
                ##plt.grid(True)
                ##plt.xlim(left=0)
            
                ##plt.subplot(6, 3, 6 * np_i + 3)
                ###plt.plot(runtime[bb_gt][bb], betas[bb_gt][bb], '--',  marker='o', markersize=5)
                ##plt.plot(iters, betas[3 * np_p + np_i][bb_gt][bb], '--',  marker='o', markersize=5)
                ##plt.title('Beta', fontweight='bold')
                ###plt.ylabel('Beta')
                ##if np_i == 2:
                    ##plt.xlabel('iteration')                
                ##plt.grid(True)
                ##plt.xlim(left=0)
            
                ##plt.subplot(6, 3, 6 * (np_i + 1))
                ###plt.plot(runtime[bb_gt][bb], betas_err, '--',  marker='o', markersize=5)
                ##plt.plot(iters, betas_err[np_i], '--',  marker='o', markersize=5)
                ###plt.title('Beta error')
                ###plt.xlabel('Running time [sec]')
                ##if np_i == 2:
                    ##plt.xlabel('iteration')
                ###plt.ylabel('Beta error [%]')
                ##plt.ylabel('Error [%]', fontweight='bold')
                ##plt.grid(True)
                ##plt.xlim(left=0)
            
            ##plt.suptitle('Original beta = ' + str(beta_gt_factor[bb_gt]) + ', starting with beta0 = ' + str(beta0_factor[bb]), fontweight='bold')
            ##if np_p == 1:
                ##s_p  = ' 2'
            ##else:
                ##s_p  = ''
                
            ##out_name = 'several Nps dependent unknown 1 grad and fwd beta gt ' + str(beta_gt_factor[bb_gt]) + ' beta0 ' + str(beta0_factor[bb]) + s_p + '.png'
            ##plt.savefig(out_path + out_name, dpi=300)
            
print('end')
