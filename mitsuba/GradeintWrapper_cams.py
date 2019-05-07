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

def plot_data(show_not_save, beta_gt_factor, beta0_diff, n_sensors, n_pixels, n_pixels_h, n_pixels_w, I_gt, I_algo, cost_gradient, betas, beta_4unknowns, Np_vector, sensors_s, alpha_s, beta1_s,  beta2_s, additional_str, sensors_pos):
    out_path = '/home/tamarl/MitsubaGradient/Gradient wrapper/plots/'
    iters = np.linspace(1, max_iterations, max_iterations)    
    for bb_gt in range(len(beta_gt_factor)):       
        for bb in range(len(beta0_diff)):                   
            bb0 = beta0_diff[bb]                   
            diff = 0
            for ss in range(n_sensors):
                for pp in range(n_pixels):
                    ppx, ppy = np.unravel_index(pp, (n_pixels_h, n_pixels_w))
                    diff    += np.squeeze( I_gt[ss, 0, bb_gt, ppx, ppy] * np.ones(I_algo[ss, 0, bb_gt, bb, :, ppx, ppy].shape)
                                           - I_algo[ss, 0, bb_gt, bb, :, ppx, ppy] )
            cost      = 0.5 * diff**2
            tmp       = np.mean(cost_gradient[0][bb_gt][bb], 2)

            plt.figure(figsize=(19,9))    

            plt.subplot(3, 1, 1)
            plt.plot(iters, cost, '--',  marker='o', markersize=5)
            plt.title('Cost', fontweight='bold')              
            plt.grid(True)
            plt.xlim(left=1)            
            for un in range(4):
                gradient  = ( tmp[:, un] + tmp[:, un + 4] ) / 2
                plt.subplot(3, 2, 3 + un)
                plt.plot(iters, gradient, '--',  marker='o', markersize=5)
                plt.title('Gradient at voxel ' + str(un + 1), fontweight='bold')
                if un > 1:
                    plt.xlabel('iteration')                
                plt.grid(True)
                plt.xlim(left=1)

            if bb > 2:
                plt.suptitle('Original beta = [' + str(beta_4unknowns[0, bb_gt, 0]) + ', ' + str(beta_4unknowns[0, bb_gt, 1]) + '; ' + 
                             str(beta_4unknowns[0, bb_gt, 2]) + ', ' + str(beta_4unknowns[0, bb_gt, 3]) + '] starting with beta0 = ' + 
                             str(bb0) + ', Np = ' + str(Np_vector[0]), fontweight='bold')
            else:
                plt.suptitle('Original beta = [' + str(beta_4unknowns[0, bb_gt, 0]) + ', ' + str(beta_4unknowns[0, bb_gt, 1]) + '; ' + 
                             str(beta_4unknowns[0, bb_gt, 2]) + ', ' + str(beta_4unknowns[0, bb_gt, 3]) +  '] starting with beta0 = gt + ' + 
                             str(bb0) + ', Np = ' + str(Np_vector[0]), fontweight='bold')
            out_name = 'cost and grad of 4 unknowns' + sensors_s + ' dependent grad and fwd Np '+ str(Np_vector[0]) + ' adam beta gt ' + str(beta_gt_factor[bb_gt])
            if bb > 2:
                out_name += ' const beta0 = ' + str(bb0) + alpha_s + beta1_s + beta2_s + additional_str + '.png'
            else:
                out_name += ' beta0 = gt + '  + str(bb0) + alpha_s + beta1_s + beta2_s + additional_str + '.png'

            if show_not_save:
                plt.show()
            else:
                plt.savefig(out_path + out_name, dpi=300)   


            plt.figure(figsize=(19,10))
            unp = 0
            for un in range(4):
                #gradient  = ( tmp[:, un] + tmp[:, un + 4] ) / 2
                betas_un  = ( betas[0, bb_gt, bb, :, un] + betas[0, bb_gt, bb, :, un + 4] ) / 2
                betas_err = (betas_un - beta_4unknowns[0, bb_gt, un]) / beta_4unknowns[0, bb_gt, un] * 100
                if un > 1:
                    unp = 2
                plt.subplot(4, 2, un + unp + 1)
                plt.plot(iters, betas_un, '--',  marker='o', markersize=5)
                plt.title('Beta at voxel ' + str(un+1), fontweight='bold')
                plt.grid(True)
                plt.xlim(left=1)

                plt.subplot(4, 2, un +3 + unp)
                plt.plot(iters, betas_err, '--',  marker='o', markersize=5)
                plt.title('Error', fontweight='bold')
                if un > 1:
                    plt.xlabel('iteration')
                plt.ylabel('[%]', fontweight='bold')
                plt.grid(True)
                plt.xlim(left=1)

                #plt.figure(figsize=(19,9))    

                #plt.subplot(2, 2, 1)
                #plt.plot(iters, cost, '--',  marker='o', markersize=5)
                #plt.title('Cost', fontweight='bold')              
                #plt.grid(True)
                #plt.xlim(left=1)

                #plt.subplot(2, 2, 3)
                #plt.plot(iters, gradient, '--',  marker='o', markersize=5)
                #plt.title('Gradient', fontweight='bold')
                #plt.xlabel('iteration')                
                #plt.grid(True)
                #plt.xlim(left=1)

                #plt.subplot(2, 2, 2)
                #plt.plot(iters, betas_un, '--',  marker='o', markersize=5)
                #plt.title('Beta', fontweight='bold')
                #plt.grid(True)
                #plt.xlim(left=1)

                #plt.subplot(2, 2, 4)
                #plt.plot(iters, betas_err, '--',  marker='o', markersize=5)
                #plt.title('Beta error', fontweight='bold')
                #plt.xlabel('iteration')
                #plt.ylabel('[%]', fontweight='bold')
                #plt.grid(True)
                #plt.xlim(left=1)

            if bb > 2:
                plt.suptitle('Original beta = ' + str(beta_4unknowns[0, bb_gt, un]) + ', starting with beta0 = ' + str(bb0) + ', Np = ' + str(Np_vector[0]) + ', unknown ' + str(un + 1), fontweight='bold')
            else:
                plt.suptitle('Original beta = ' + str(beta_4unknowns[0, bb_gt, un]) + ', starting with beta0 = gt + ' + str(bb0) + ', Np = ' + str(Np_vector[0]) + ', unknown ' + str(un + 1), fontweight='bold')

            out_name = 'fix grad 4 unknowns' + sensors_s + ' dependent grad and fwd Np '+ str(Np_vector[0]) + ' adam beta gt ' + str(beta_gt_factor[bb_gt])
            if bb > 2:
                out_name += ' const beta0 = ' + str(bb0) + alpha_s + beta1_s + beta2_s + additional_str + '_' + str(un + 1) + '.png'
            else:
                out_name += ' beta0 = gt + '  + str(bb0) + alpha_s + beta1_s + beta2_s + additional_str + '_' + str(un + 1) + '.png'

            if show_not_save:
                plt.show()
            else:
                plt.savefig(out_path + out_name, dpi=300)

            plt.figure(figsize=(19,9))    
            for ss in range(n_sensors):
                plt.subplot(n_sensors + 1, 3, 3 * ss + 1)
                plt.imshow(I_gt[ss, 0, bb_gt])
                maxc = np.max(np.array([I_gt[ss, 0, bb_gt], I_algo[ss, 0, bb_gt, bb, 0], I_algo[ss, 0, bb_gt, bb, -1]]))
                plt.clim(0, maxc)
                plt.colorbar()
                plt.axis('off')
                plt.title('Ground truth, Sensor at (' + str(sensors_pos[ss]['origin'][0]) + ', ' + str(sensors_pos[ss]['origin'][1]) + 
                          ', ' + str(sensors_pos[ss]['origin'][2]) + ')')

                plt.subplot(n_sensors + 1, 3, 3 * ss + 2)
                plt.imshow(I_algo[ss, 0, bb_gt, bb, 0])
                plt.clim(0, maxc)
                plt.colorbar()
                plt.axis('off')
                err = sum(sum((I_gt[ss, 0, bb_gt] - I_algo[ss, 0, bb_gt, bb, 0])**2))
                plt.title('Initial output, error = ' + str(round(err, 7)))

                plt.subplot(n_sensors + 1, 3, 3 * ss + 3)
                plt.imshow(I_algo[ss, 0, bb_gt, bb, -1])
                plt.clim(0, maxc)
                plt.colorbar()
                plt.axis('off')
                err = sum(sum((I_gt[ss, 0, bb_gt] - I_algo[ss, 0, bb_gt, bb, -1])**2)) 
                plt.title('Final output, error = ' +  str(round(err, 7)))

            plt.subplot(n_sensors + 1, 3, 3 * n_sensors + 1)
            plt.imshow(np.mean(I_gt, 0)[0, bb_gt])
            maxc = np.max(np.array([np.mean(I_gt, 0)[0, bb_gt], np.mean(I_algo, 0)[0, bb_gt, bb, 0], np.mean(I_algo, 0)[0, bb_gt, bb, -1]]))
            plt.clim(0, maxc)        
            plt.colorbar()
            plt.axis('off')
            plt.title('Mean Ground Truth', fontweight='bold')

            plt.subplot(n_sensors + 1, 3, 3 * n_sensors + 2)
            plt.imshow(np.mean(I_algo, 0)[0, bb_gt, bb, 0])
            plt.clim(0, maxc)                
            plt.colorbar()
            plt.axis('off')
            err = sum(sum((np.mean(I_gt, 0)[0, bb_gt] - np.mean(I_algo, 0)[0, bb_gt, bb, 0])**2))
            plt.title('Mean Initial Output, error = ' +  str(round(err, 6)), fontweight='bold')

            plt.subplot(n_sensors + 1, 3, 3 * n_sensors + 3)
            plt.imshow(np.mean(I_algo, 0)[0, bb_gt, bb, -1])
            plt.clim(0, maxc)                
            plt.colorbar()
            plt.axis('off')
            err = sum(sum((np.mean(I_gt, 0)[0, bb_gt] - np.mean(I_algo, 0)[0, bb_gt, bb, -1])**2))
            plt.title('Mean Final Output, error = ' +  str(round(err, 6)), fontweight='bold')

            if bb > 2:
                plt.suptitle('Sensors images, original beta = [' + str(beta_4unknowns[0, bb_gt, 0]) + ', ' + str(beta_4unknowns[0, bb_gt, 1]) + '; ' + str(beta_4unknowns[0, bb_gt, 2]) + ', ' + str(beta_4unknowns[0, bb_gt, 3]) +  '] starting with beta0 = ' + str(bb0) + ', Np = ' + str(Np_vector[0]), fontweight='bold')
            else:
                plt.suptitle('Sensors images, original beta = [' + str(beta_4unknowns[0, bb_gt, 0]) + ', ' + str(beta_4unknowns[0, bb_gt, 1]) + '; ' + str(beta_4unknowns[0, bb_gt, 2]) + ', ' + str(beta_4unknowns[0, bb_gt, 3]) +  '] starting with beta0 = gt + ' + str(bb0) + ', Np = ' + str(Np_vector[0]), fontweight='bold')

            out_name = 'fix grad 4 unknowns' + sensors_s + ' dependent grad and fwd Np '+ str(Np_vector[0]) + ' adam beta gt ' + str(beta_gt_factor[bb_gt])
            if bb > 2:
                out_name += ' const beta0 = ' + str(bb0) + alpha_s + beta1_s + beta2_s + additional_str + ' sensors images.png'
            else:
                out_name += ' beta0 = gt + '  + str(bb0) + alpha_s + beta1_s + beta2_s + additional_str + ' sensors images.png'

            if show_not_save:
                plt.show()
            else:            
                plt.savefig(out_path + out_name, dpi=300)

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

    #q = np.dot(T.transpose()[:-1, :-1] , cam_pos.transpose())

    #T_inverse         = np.zeros([4,4])
    #T_inverse[0, 0:3] = right
    #T_inverse[1, 0:3] = newUp    
    #T_inverse[2, 0:3] = forward
    #T_inverse[0:3, 3] = -q
    #T_inverse[3, 3]   = 1.0

    return newUp, T#, T_inverse

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
    inner_grad_tmp = scene.getTotalGradient()
    inner_grad_tmp = np.reshape(inner_grad_tmp, [grid_size, n_pixels_w * n_pixels_h], 'F')
    inner_grad = np.reshape(inner_grad_tmp, [grid_size, n_pixels_w, n_pixels_h])    
    #inner_grad = np.zeros([grid_size, 3, n_pixels_w, n_pixels_h])
    #for pw in range(n_pixels_w):
        #for ph in range(n_pixels_h):
            #inner_grad[:, :, pw, ph] = get_grad_from_output_file('output_' + str(pw) + '_' + str(ph)+ '.txt')

    # End session
    queue.join() 

    return radiance, inner_grad


## Parameters
# algorithm parameters
max_iterations = 200#500 * 3 #* 2
n_unknowns     = 4
n_cases        = 3
beta_gt_factor = np.array([5])#, 4, 6])#, 5, 10]) #np.array([1, 2, 5, 10])
beta0_diff     = np.array([1, 2, 3])
#beta0_diff     = np.array([-0.5, -0.2, 0, 0.2, 0.5]) 
grid_size      = 8

# sensors parameters
n_sensors  = 4
n_pixels_w = 2
n_pixels_h = 2
n_pixels   = n_pixels_h * n_pixels_w

# optimizer parameters - ADAM
alpha   = 0.01 # randomly select alpha hyperparameter -> r = -a * np.random.rand() ; alpha = 10**(r)                                - randomly selected from a logaritmic scale
beta1   = 0.9 # randomly select beta1 hyperparameter -> sample (1-beta1), r = -a * np.random.uniform(-3, -1) ; beta1 = 1 - 10**(r) - randomly selected from a logaritmic scale
epsilon = 1e-8
beta2   = 0.999#, 0.99, 0.999, 0.9999])

for case in range(n_cases):
    # several Nps:
    #Np_vector     = np.array([1, 2, 4, 8, 16, 32]) * 128
    # single Np:
    Np_vector     = np.array([512]) * 128
    scene_gt      = [ None ] * n_sensors # create an empty list
    algo_scene    = [ None ] * n_sensors # create an empty list
    sensors_pos   = [ None ] * n_sensors # create an empty list
    I_gt          = np.zeros((n_sensors, len(Np_vector), len(beta_gt_factor), n_pixels_h, n_pixels_w))
    I_algo        = np.zeros((n_sensors, len(Np_vector), len(beta_gt_factor), len(beta0_diff), max_iterations, n_pixels_h, n_pixels_w))
    runtime       = np.zeros((len(Np_vector), len(beta_gt_factor), len(beta0_diff), max_iterations))
    cost_gradient = np.zeros((len(Np_vector), len(beta_gt_factor), len(beta0_diff), max_iterations, grid_size, n_pixels))
    cost          = np.zeros((len(beta0_diff), max_iterations))

    ## 1 unknown
    #betas         = np.zeros((len(Np_vector), len(beta_gt_factor), len(beta0_diff), max_iterations, n_unknowns))
    ## 4 unknowns
    betas          = np.zeros((len(Np_vector), len(beta_gt_factor), len(beta0_diff), max_iterations, n_unknowns*2))
    beta_4unknowns = np.zeros((len(Np_vector), len(beta_gt_factor), n_unknowns))

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

    import matplotlib.pyplot as plt
    #additional_str = ' camera more itersations'
    additional_str = ''

    if alpha is not 0.01:
        alpha_s = ' alpha ' + str(alpha)
    else:
        alpha_s = ''

    if beta1 is not 0.9:
        beta1_s = ' adam beta1 ' + str(beta1)
    else:
        beta1_s = ''

    if beta2 is not 0.999:
        beta2_s = ' adam beta2 ' + str(beta2)
    else:
        beta2_s = ''

    if n_sensors > 1:
        sensors_s = ' ' + str(n_sensors) + ' sensors'
    else:
        sensors_s = ''

    for nps in range(len(Np_vector)):
        for bb_gt in range(len(beta_gt_factor)):

            ## 4 unknowns
            #tmp = [round(x, 1) for x in np.random.uniform(0.51, beta_gt_factor[bb_gt] + 0.09, 4)]
            if   case == 0:
                tmp = [1.1, 2.3, 2.4, 1.7] * 2 
                additional_str = ' case 1'                
            elif case == 1:
                tmp = [1.6, 2.3, 1.0, 3.1] * 2
                additional_str = ' case 2'                
            elif case == 2:
                tmp = [3.5, 6.5, 1.6, 7.3] * 2                 
                additional_str = ' case 3'                
            else:
                tmp = [1.2, 1.1, 0.9, 1.1] * 2#[1.5, 1.5, 1.2, 1.4]
            tmp     = np.array(tmp) 
            beta_gt =  np.reshape(tmp, [2, 2, 2], 'F')
            beta_4unknowns[0, bb_gt] = tmp[0:4]            
            #beta_gt = np.reshape(tmp, [2, 2, 2])

            up_const     = np.array([-1, 0, 0])
            #target_const = np.array([0.01, 0.01, 0.01])

            # Ground Truth:
            scene_gt[0]    = pyScene()

            ## 1 unknown
            #scene_gt[0].create_new_scene(beta=beta_gt_factor[bb_gt], origin=Point(0, 3, 0), target=None, up=Vector(0, 0, 1), nSamples=Np_vector[nps])
            #beta_gt        = scene_gt[0].get_scene_beta()
            ## 4 unknowns
            t1 =  np.array([0.5, 0.5, 1.01])#0.01])
            newUp1, _ = transformLookAt(np.array([0.5, 0.5, 3]), 
                                        t1, 
                                        up_const)

            sensors_pos[0] = { 'origin' : Point(0.5, 0.5, 3),
                               'target' : Point(t1[0], t1[1], t1[2]), 
                               'up'     : Vector(newUp1[0], newUp1[1], newUp1[2]) }

            scene_gt[0].create_new_scene(beta=beta_gt, g=0, origin=sensors_pos[0]['origin'], target=sensors_pos[0]['target'], 
                                         up=sensors_pos[0]['up'], nSamples=Np_vector[nps] * 4, sensorType='perspective', 
                                         width=n_pixels_w, height=n_pixels_h)

            # Render Ground Truth Scene
            I_gt[0, nps, bb_gt], _ = render_scene(scene_gt[0]._scene, output_filename, n_cores, grid_size, n_pixels_w, n_pixels_h)

            ## Add more sensors
            #sensors_pos[1] = { 'origin' : Point(0, 0, -3),  ## NOT A GOOD POSITION!!!
                                #'target' : sensors_pos[0]['target'], 
                                #'up'     : Vector(-1, 0, 0) }
            t2  =  np.array([0.5, -0.5, 1.01])#0.01])
            newUp2, _ = transformLookAt(np.array([0.5, -0.5, 3]), 
                                        t2, 
                                        up_const)     
            sensors_pos[1] = { 'origin' : Point(0.5, -0.5, 3), 
                               'target' : Point(t2[0], t2[1], t2[2]),
                               'up'     : Vector(newUp2[0], newUp2[1], newUp2[2]) }

            t3 = np.array([-0.5, 0.5, 1.01])#0.01])
            newUp3, _ = transformLookAt(np.array([-0.5, 0.5, 3]), 
                                        t3,
                                        up_const)       
            sensors_pos[2] = { 'origin' : Point(-0.5, 0.5, 3), 
                               'target' : Point(t3[0], t3[1], t3[2]), 
                               'up'     : Vector(newUp3[0], newUp3[1], newUp3[2]) }

            t4 = np.array([-0.5, -0.5, 1.01])#0.01])
            newUp4, _ = transformLookAt(np.array([-0.5, -0.5, 3]), 
                                        t4, 
                                          up_const)
            sensors_pos[3] = { 'origin' : Point(-0.5, -0.5, 3),
                               'target' : Point(t4[0], t4[1], t4[2]), 
                               'up'     : Vector(newUp4[0], newUp4[1], newUp4[2]) }

            scene_gt[1]    = pyScene()
            scene_gt[1].create_new_scene(beta=beta_gt, g=0, origin=sensors_pos[1]['origin'], target=sensors_pos[1]['target'], 
                                         up=sensors_pos[1]['up'], nSamples=Np_vector[nps] * 4, sensorType='perspective', 
                                         width=n_pixels_w, height=n_pixels_h)
            scene_gt[2]    = pyScene()
            scene_gt[2].create_new_scene(beta=beta_gt, g=0, origin=sensors_pos[2]['origin'], target=sensors_pos[2]['target'], 
                                         up=sensors_pos[2]['up'], nSamples=Np_vector[nps] * 4, sensorType='perspective', 
                                         width=n_pixels_w, height=n_pixels_h)
            scene_gt[3]    = pyScene()
            scene_gt[3].create_new_scene(beta=beta_gt, g=0, origin=sensors_pos[3]['origin'], target=sensors_pos[3]['target'], 
                                         up=sensors_pos[3]['up'], nSamples=Np_vector[nps] * 4, sensorType='perspective', 
                                         width=n_pixels_w, height=n_pixels_h)

            I_gt[1, nps, bb_gt], _ = render_scene(scene_gt[1]._scene, output_filename, n_cores, grid_size, n_pixels_w, n_pixels_h)
            I_gt[2, nps, bb_gt], _ = render_scene(scene_gt[2]._scene, output_filename, n_cores, grid_size, n_pixels_w, n_pixels_h)
            I_gt[3, nps, bb_gt], _ = render_scene(scene_gt[3]._scene, output_filename, n_cores, grid_size, n_pixels_w, n_pixels_h)

            print(beta_gt_factor[bb_gt])

            for bb in range(len(beta0_diff)):
                # optimizer parameters - ADAM
                first_moment  = 0 #m0
                second_moment = 0 #v0

                ## 1 unknown
                #beta0  = np.ones(beta_gt.shape) * beta0_factor[bb]
                ## 4 unknowns
                #beta0 = beta_gt + np.ones(beta_gt.shape) * beta0_diff[bb] 
                if bb > -1:
                    beta0 = np.ones(beta_gt.shape) * beta0_diff[bb]

                # for now beta is not a Spectrum:
                inner_grad_float = np.zeros((n_unknowns, 1))

                # Gradient descent loop
                beta  = np.copy(beta0)
                start = time.time()

                for iteration in range(max_iterations):               

                    #print(iteration)
                    print(beta)
                    ## 1 unknown
                    #betas[nps, bb_gt, bb, iteration] = np.mean(np.mean(beta))
                    #cost_grad = np.zeros((n_unknowns, n_pixels))
                                ## 4 unknowns
                    betas[nps, bb_gt, bb, iteration] = beta.flatten('F')                
                    #betas[nps, bb_gt, bb, iteration] = beta.flatten() 
                    cost_grad = np.zeros((n_unknowns*2, 1))                     

                    for ss in range(n_sensors):
                        # Create scene with given beta
                        algo_scene[ss]    = pyScene()
                        algo_scene[ss].create_new_scene(beta=beta, g=0, origin=sensors_pos[ss]['origin'], target=sensors_pos[ss]['target'], 
                                                        up=sensors_pos[ss]['up'], nSamples=Np_vector[nps], 
                                                        sensorType='perspective', width=n_pixels_w, height=n_pixels_h)
                        
                        [ I_algo[ss, nps, bb_gt, bb, iteration], 
                          inner_grad ] = render_scene(algo_scene[ss]._scene, output_filename, n_cores, grid_size, n_pixels_w, n_pixels_h)

                        ### beta is not a Spectrum, for now:
                        ## 1 unknown
                        #inner_grad_float = np.mean(np.mean(inner_grad))
                        #tmp              = (-1) * ( I_algo[ss, nps, bb_gt, bb, iteration] - I_gt[ss, nps, bb_gt] )
                        #cost_grad       += inner_grad_float * tmp[:, None] #np.matmul(inner_grad_float, (-1)* np.mean( I_algo[ss][nps][bb_gt][bb][iteration] - I_gt[ss][nps][bb_gt] ))

                        ## 4 unknowns
                        #inner_grad_float    = np.mean(inner_grad, 1)
                        inner_grad[0] = (inner_grad[0] + inner_grad[4]) / 2
                        inner_grad[1] = (inner_grad[1] + inner_grad[5]) / 2
                        inner_grad[2] = (inner_grad[2] + inner_grad[6]) / 2
                        inner_grad[3] = (inner_grad[3] + inner_grad[7]) / 2
                        inner_grad[4] = inner_grad[0]
                        inner_grad[5] = inner_grad[1]
                        inner_grad[6] = inner_grad[2]
                        inner_grad[7] = inner_grad[3]

                        tmp        =  (-1) * ( I_algo[ss, nps, bb_gt, bb, iteration] - I_gt[ss, nps, bb_gt] ) 
                        cost_grad += np.sum(np.sum(inner_grad * tmp, 2), 1)[:, None]                       
                        #cost_grad += np.matmul(inner_grad_float, tmp.flatten('F'))[:, None]
                        #cost_grad += inner_grad_float * tmp
                        cost_iter += np.linalg.norm(tmp,ord=2)				

                    cost[bb, iteration] = 0.5 * cost_iter

                    cost_gradient[nps, bb_gt, bb, iteration] = cost_grad
                    ## 1 unknown
                    #cost_grad_mat = np.ones(beta.shape) * cost_grad
                    ## 4 unknowns
                    cost_grad_mat = np.reshape(cost_grad, beta.shape, 'F') #CHECK - I thins it's column stack but it's not C style 'F' ot 'C'
                    #cost_grad_mat = np.reshape(cost_grad, beta.shape, 'C')    

                    ## ADAM implementation
                    first_moment  = beta1 * first_moment  + (1 - beta1) * cost_grad_mat
                    second_moment = beta2 * second_moment + (1 - beta2) * cost_grad_mat**2
                    #second_moment = beta2 * second_moment + (1 - beta2) * np.power(cost_grad_mat, 2)

                    first_moment_bar  = first_moment  / ( 1 - beta1**(iteration + 1) )
                    second_moment_bar = second_moment / ( 1 - beta2**(iteration + 1) )

                    beta -= alpha * first_moment_bar / (np.sqrt(second_moment_bar) + epsilon)    

                    ## for fixed step size
                    #beta     += step_size * cost_grad_mat

                    if beta[beta <= 0].size:
                        ## 1 unknown
                        #beta = np.ones(beta.shape) * 0.1
                        ## 4 unknowns
                        beta_before = beta + alpha * first_moment_bar / (np.sqrt(second_moment_bar) + epsilon)    
                        beta[beta <= 0] = beta_before[beta <= 0]
                        print('fixed beta!')

                    end = time.time()

                    runtime[nps, bb_gt, bb, iteration] = end - start 

    scheduler.stop()


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

            #out_name = 'fix grad ' + sensors_s + 'dependent grad and fwd Np '+ str(Np_vector[0]) + ' adam unknown 1 beta gt ' + str(beta_gt_factor[bb_gt]) + ' beta0 ' + str(beta0_factor[bb]) + alpha_s + additional_str +'.png'
            #plt.savefig(out_path + out_name, dpi=300)

    ## 4 unknowns:
    out_path = '/home/tamarl/MitsubaGradient/Gradient wrapper/plot_tmp_cmp/cams/'
    iters = np.linspace(1, max_iterations, max_iterations)    

    for bb_gt in range(len(beta_gt_factor)):       
        for bb in range(len(beta0_diff)):    
            bb0 = beta0_diff[bb]                   
            diff = 0
            for ss in range(n_sensors):
                for pp in range(n_pixels):
                    ppx, ppy = np.unravel_index(pp, (n_pixels_h, n_pixels_w))
                    diff    += np.squeeze( I_gt[ss, 0, bb_gt, ppx, ppy] * np.ones(I_algo[ss, 0, bb_gt, bb, :, ppx, ppy].shape)
                                           - I_algo[ss, 0, bb_gt, bb, :, ppx, ppy] )
            cost      = 0.5 * diff**2
            tmp       = np.mean(cost_gradient[0, bb_gt, bb], 2)   

            for un in range(4):
                gradient  = ( tmp[:, un] + tmp[:, un + 4] ) / 2
                betas_un  = ( betas[0, bb_gt, bb, :, un] + betas[0, bb_gt, bb, :, un + 4] ) / 2
                betas_err = (betas_un - beta_4unknowns[0, bb_gt, un]) / beta_4unknowns[0, bb_gt, un] * 100

                plt.figure(figsize=(19,9))    

                #plt.subplot(2, 2, 1)
                #plt.plot(iters, cost, '--',  marker='o', markersize=5)
                #plt.title('Cost', fontweight='bold')  
                #plt.grid(True)
                #plt.xlim(left=0)

                #plt.subplot(2, 2, 3)
                plt.subplot(2, 2, 4)
                plt.plot(iters, gradient, '--',  marker='o', markersize=5)
                #w = np.ones(10) / 10
                #mean_grad = np.convolve(gradient.flatten(), w, 'same')
                #plt.plot(iters, mean_grad * np.ones(iters.shape), 'r-')
                plt.title('Gradient', fontweight='bold')
                plt.xlabel('iteration')                
                plt.grid(True)
                plt.xlim(left=0)

                #plt.subplot(2, 2, 2)
                plt.subplot(2, 2, 2)
                plt.plot(iters, betas_un, '--',  marker='o', markersize=5)
                plt.title('Beta', fontweight='bold')
                plt.grid(True)
                plt.xlim(left=0)

                #plt.subplot(2, 2, 4)
                plt.subplot(1, 2, 1)
                plt.plot(iters, betas_err, '--',  marker='o', markersize=5)
                plt.title('Beta error', fontweight='bold')
                plt.xlabel('iteration')
                plt.ylabel('[%]', fontweight='bold')
                plt.grid(True)
                plt.xlim(left=0)    

                #if bb > 2:
                plt.suptitle('Original beta = ' + str(beta_4unknowns[0, bb_gt, un]) + ', starting with beta0 = ' + str(bb0) + ', Np = ' + str(Np_vector[0]) + ', unknown ' + str(un + 1), fontweight='bold')
                #else:
                #plt.suptitle('Original beta = ' + str(beta_4unknowns[0, bb_gt, un]) + ', starting with beta0 = gt + ' + str(bb0) + ', Np = ' + str(Np_vector[0]) + ', unknown ' + str(un + 1), fontweight='bold')
                #plt.show()

                #out_name = '_fix grad 4 unknowns' + sensors_s + ' dependent grad and fwd Np '+ str(Np_vector[0]) + ' adam beta gt ' + str(beta_gt_factor[bb_gt]) + ' beta0 ' + str(bb0) + alpha_s + beta1_s + beta2_s + additional_str + '_' + str(un + 1) +'photonSpec_F.png'
                out_name = '_fix grad 4 unknowns' + sensors_s + ' dependent grad and fwd Np '+ str(Np_vector[0]) + ' adam' + additional_str + ' const beta0 ' + str(bb0) + alpha_s + beta1_s + beta2_s + '_' + str(un + 1) +' photonSpec_F.png'
                plt.savefig(out_path + out_name, dpi=300)

            plt.figure(figsize=(19,9))    
            for ss in range(n_sensors):
                plt.subplot(n_sensors + 1, 3, 3 * ss + 1)
                plt.imshow(I_gt[ss, 0, bb_gt])
                maxc = np.max(np.array([I_gt[ss, 0, bb_gt], I_algo[ss, 0, bb_gt, bb, 0], I_algo[ss, 0, bb_gt, bb, -1]]))
                plt.clim(0, maxc)
                plt.colorbar()
                plt.axis('off')
                plt.title('Ground truth')

                plt.subplot(n_sensors + 1, 3, 3 * ss + 2)
                plt.imshow(I_algo[ss, 0, bb_gt, bb, 0])
                plt.clim(0, maxc)
                plt.colorbar()
                plt.axis('off')
                err = sum(sum((I_gt[ss, 0, bb_gt] - I_algo[ss, 0, bb_gt, bb, 0])**2))
                plt.title('Initial output, error = ' + str(round(err, 7)))

                plt.subplot(n_sensors + 1, 3, 3 * ss + 3)
                plt.imshow(I_algo[ss, 0, bb_gt, bb, -1])
                plt.clim(0, maxc)
                plt.colorbar()
                plt.axis('off')
                err = sum(sum((I_gt[ss, 0, bb_gt] - I_algo[ss, 0, bb_gt, bb, -1])**2)) 
                plt.title('Final output, error = ' +  str(round(err, 7)))

            plt.subplot(n_sensors + 1, 3, 3 * n_sensors + 1)
            plt.imshow(np.mean(I_gt, 0)[0, bb_gt])
            maxc = np.max(np.array([np.mean(I_gt, 0)[0, bb_gt], np.mean(I_algo, 0)[0, bb_gt, bb, 0], np.mean(I_algo, 0)[0, bb_gt, bb, -1]]))
            plt.clim(0, maxc)        
            plt.colorbar()
            plt.axis('off')
            plt.title('Mean Ground Truth', fontweight='bold')

            plt.subplot(n_sensors + 1, 3, 3 * n_sensors + 2)
            plt.imshow(np.mean(I_algo, 0)[0, bb_gt, bb, 0])
            plt.clim(0, maxc)                
            plt.colorbar()
            plt.axis('off')
            err = sum(sum((np.mean(I_gt, 0)[0, bb_gt] - np.mean(I_algo, 0)[0, bb_gt, bb, 0])**2))
            plt.title('Mean Initial Output, error = ' +  str(round(err, 6)), fontweight='bold')

            plt.subplot(n_sensors + 1, 3, 3 * n_sensors + 3)
            plt.imshow(np.mean(I_algo, 0)[0, bb_gt, bb, -1])
            plt.clim(0, maxc)                
            plt.colorbar()
            plt.axis('off')
            err = sum(sum((np.mean(I_gt, 0)[0, bb_gt] - np.mean(I_algo, 0)[0, bb_gt, bb, -1])**2))
            plt.title('Mean Final Output, error = ' +  str(round(err, 6)), fontweight='bold')

            #if bb > 2:
            plt.suptitle('Sensors images, original beta = [' + str(beta_4unknowns[0, bb_gt, 0]) + ', ' + str(beta_4unknowns[0, bb_gt, 1]) + '; ' + str(beta_4unknowns[0, bb_gt, 2]) + ', ' + str(beta_4unknowns[0, bb_gt, 3]) +  '] starting with beta0 = ' + str(bb0) + ', Np = ' + str(Np_vector[0]), fontweight='bold')
            #else:
            #plt.suptitle('Sensors images, original beta = [' + str(beta_4unknowns[0, bb_gt, 0]) + ', ' + str(beta_4unknowns[0, bb_gt, 1]) + '; ' + str(beta_4unknowns[0, bb_gt, 2]) + ', ' + str(beta_4unknowns[0, bb_gt, 3]) +  '] starting with beta0 = gt + ' + str(bb0) + ', Np = ' + str(Np_vector[0]), fontweight='bold')

            out_name = '_fix grad 4 unknowns' + sensors_s + ' dependent grad and fwd Np '+ str(Np_vector[0]) + ' adam beta gt ' + str(beta_gt_factor[bb_gt])
            #if bb > 2:
            out_name += ' const beta0 = ' + str(bb0) + alpha_s + beta1_s + beta2_s + additional_str + ' sensors images.png'
            #else:
            #out_name += ' beta0 = gt + '  + str(bb0) + alpha_s + beta1_s + beta2_s + additional_str + ' sensors images.png'

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
