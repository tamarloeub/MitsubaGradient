
## '../Gradient wrapper/jpl/multiscale/monoz/smooth/19_09_09_15:13/stages 8 lambda1 0.2 lambda2 1 with air and ocean 54872 grid points 54872 unknowns 9 sensors 5776 pixels 1024 rrDepth 3 photons changes with scale space carving mask beta0 2' + ' iter.mat'


zz_max = [ np.max([ beta[:, :, zz], beta_gt[:, :, zz] ]) for zz in range(nz_ms) ]

plt.figure(figsize=(19,9))
for zz in range(nz_ms):     
    plt.subplot(int(np.ceil(np.sqrt(nz_ms))), int(np.ceil(np.sqrt(nz_ms))), zz + 1)               
    plt.imshow(beta[:, :, zz])
    plt.title( 'z = ' + str(zz+1) )
    plt.clim(0.01, zz_max[zz])
    plt.colorbar()
    plt.axis('off')
plt.suptitle('curr beta [1/km]', fontweight='bold')

plt.figure(figsize=(19,9))
for zz in range(nz_ms):    
    plt.subplot(int(np.ceil(np.sqrt(nz_ms))), int(np.ceil(np.sqrt(nz_ms))), zz + 1)               
    plt.imshow(beta_gt[:, :, zz])
    plt.title( 'z = ' + str(zz+1) )
    plt.clim(0.01, zz_max[zz])
    plt.colorbar()
    plt.axis('off')
plt.suptitle('beta gt [1/km]', fontweight='bold')

diff_map = (beta_gt - beta)
diff_map[abs(diff_map) <= 0.05] =0

plt.figure(figsize=(19,9))
for zz in range(nz_ms):    
    plt.subplot(int(np.ceil(np.sqrt(nz_ms))), int(np.ceil(np.sqrt(nz_ms))), zz + 1)               
    plt.imshow(diff_map[:, :, zz])
    plt.title( 'z = ' + str(zz+1) )
    plt.clim(-np.max(abs(diff_map[:, :, zz])), np.max(abs(diff_map[:, :, zz])))
    plt.colorbar()
    plt.axis('off')    
plt.suptitle('difference map = (beta_gt - curr_beta) [1/km]', fontweight='bold')

plt.show()


mean_filter = np.ones(cost_window_size)/cost_window_size
plt.figure()
plt.plot(np.linspace(1,iteration, iteration+1), cost_mean[0, :iteration+1],'--',  marker='o', 
         markersize=5,color='blue') 
plt.title('Cost', fontweight='bold')  
plt.grid(True)
#plt.xlim((0, runtime[0,iteration]/60./60.))
plt.xlabel('iterations')

plt.figure()
plt.plot(np.linspace(1,iteration, iteration+1), np.transpose(mb[:iteration+1]),'--',  marker='o', markersize=5,color='blue') 
plt.title('mean beta error', fontweight='bold')  
plt.grid(True)
#plt.xlim((0, runtime[0,iteration]/60./60.))
plt.xlabel('iterations')
plt.ylabel('[%]')
plt.show()


plt.figure()
plt.plot(np.transpose(runtime[:,:iteration])/60./60., cost_mean[0, :iteration],'--',  marker='o', 
         markersize=5,color='blue') 
plt.title('Cost', fontweight='bold')  
plt.grid(True)
plt.xlim((0, runtime[0,iteration-1]/60./60.))
plt.xlabel('runtime [hours]')

plt.figure()
plt.plot(np.transpose(runtime[:,:iteration])/60./60., np.transpose(mb[:iteration]),'--',  marker='o', markersize=5,color='blue') 
plt.title('mean beta error', fontweight='bold')  
plt.grid(True)
plt.xlim((0, runtime[0,iteration-1]/60./60.))
plt.xlabel('runtime [hours]')
plt.ylabel('[%]')
plt.show()




plt.figure()
plt.plot(np.linspace(1,iteration-1,iteration),np.transpose(runtime[:,:iteration])/60./60.,'--',  marker='o', 
         markersize=5,color='blue') 
plt.title('runtime', fontweight='bold')  
plt.grid(True)
#plt.xlim((0, runtime[0,iteration-1]/60./60.))
plt.xlabel('iterations')
plt.ylabel('runtime [hours]')
plt.show()
