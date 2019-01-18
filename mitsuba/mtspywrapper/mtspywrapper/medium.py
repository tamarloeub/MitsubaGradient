from struct import pack, unpack
import os
from tempfile import NamedTemporaryFile
from mitsuba.core import PluginManager, Transform, Vector, Spectrum
import numpy as np
from mitsuba.render import Medium
import datetime


class pyMedium(object):
    def __init__(self):
        self._pmgr = PluginManager.getInstance()
        self._density = None
        self._albedo = 1
        self._phase = None
        self._scale = 1
        self._bounding_box = None
        self._shape = ()
        self._medium_path = 'Myscenes/' + datetime.datetime.now().strftime("%y_%m_%d_%H:%M")
        self._files = dict()
        self._boundaries = {'x' : 'open', 'y' : 'open' }
        
    def create_boundingbox(self, bounds):
        #input :  bounds = [xmin, ymin, zmin, xmax, ymax, zmax]
        self._bounding_box = bounds
        if not os.path.exists(self._medium_path):
            os.makedirs(self._medium_path)
            
        self._files['bounding_box'] = self._medium_path + '/bounds.obj'
        f = open(self._files['bounding_box'], 'w')
        
        # Define all the vertices 
        f.write('v ' + str(bounds[3]) + ' ' + str(bounds[4]) + ' ' + str(bounds[2]) + '\n') 
        f.write('v ' + str(bounds[3]) + ' ' + str(bounds[1]) + ' ' + str(bounds[2]) + '\n') 
        f.write('v ' + str(bounds[0]) + ' ' + str(bounds[1]) + ' ' + str(bounds[2]) + '\n') 
        f.write('v ' + str(bounds[0]) + ' ' + str(bounds[4]) + ' ' + str(bounds[2]) + '\n')
        f.write('v ' + str(bounds[3]) + ' ' + str(bounds[4]) + ' ' + str(bounds[5]) + '\n') 
        f.write('v ' + str(bounds[3]) + ' ' + str(bounds[1]) + ' ' + str(bounds[5]) + '\n')
        f.write('v ' + str(bounds[0]) + ' ' + str(bounds[1]) + ' ' + str(bounds[5]) + '\n') 
        f.write('v ' + str(bounds[0]) + ' ' + str(bounds[4]) + ' ' + str(bounds[5]) + '\n')
        
        # Define all the normals  
        f.write('vn 0.000000 1.000000 0.000001\n')
        f.write('vn 0.000000 1.000000 0.000000\n')
        f.write('vn -1.000000 0.000000 -0.000000\n')
        f.write('vn -0.000000 -1.000000 -0.000001\n')
        f.write('vn -0.000000 -1.000000 0.000000\n')
        f.write('vn 1.000000 0.000000 -0.000001\n')
        f.write('vn 1.000000 -0.000001 0.000001\n')
        f.write('vn -0.000000 -0.000000 1.000000\n')
        f.write('vn 0.000000 0.000000 -1.000000\n')
        
        # Define the connectivity
        f.write('f 5//1 1//1 4//1\n')
        f.write('f 5//2 4//2 8//2\n')
        f.write('f 3//3 7//3 8//3\n')
        f.write('f 3//3 8//3 4//3\n')
        f.write('f 2//4 6//4 3//4\n')
        f.write('f 6//5 7//5 3//5\n')
        f.write('f 1//6 5//6 2//6\n')
        f.write('f 5//7 6//7 2//7\n')
        f.write('f 5//8 8//8 6//8\n')
        f.write('f 8//8 7//8 6//8\n')
        f.write('f 1//9 2//9 3//9\n')
        f.write('f 1//9 3//9 4//9\n')
        f.close()

    def dup_volume(self, volume, dups):
        # duplicated volume according to the dups list 
        # if dups == [1, 1, 1, 1] then dup_volume = volume
        dup_volume = np.tile(volume, dups)    
        return dup_volume
    
    def set_vol_file(self, volume, vol_type):
        # this function gets volume data and write it to .vol file and updates the file full path to self._files
        assert (type(vol_type) == str), "Input vol_type need to be type string"
        assert ((vol_type is 'albedo') or (vol_type is 'density') or (vol_type is 'phase')),\
               "Python wrapper don't support volumetric " + vol_type

        if not os.path.exists(self._medium_path):
            os.makedirs(self._medium_path)
        self._files[vol_type] = self._medium_path + '/' + vol_type + '.vol'

        f = open(self._files[vol_type], 'w')
        
        f.write('VOL')         # Bytes 1-3 ASCII Bytes 'V', 'O', and 'L'
        f.write(pack('B',3))   # Byte 4 File format version number (currently 3)
        f.write(pack('I',1))   # Bytes 5-8 Encoding identifier (32-bit integer).The following choices are available:
                               #   1. Dense float32-based representation
                               #   2. Dense float16-based representation (currently not supported by this implementation)       
                               #   3. Dense uint8-based representation (The range 0..255 will be mapped to 0..1)
                               #   4. Dense quantized directions. The directions are stored in spherical coordinates with a total storage cost of 16 bit per entry.
    
    
        # Add dimensions to reach a 4D structure (fourth dimention for multi-spectral data)
        for i in range(volume.ndim, 4):
            volume = volume[..., np.newaxis]
        
        ## need to be in seperate function.
        #Duplicate dimensions with 1 cell (currently mitsuba accepts only >2 grid points per dimension)
        shape = volume.shape
        dup = [1, 1, 1, 1]
        for i in range(3):
            # Singelton on that dimension - requieres duplication 
            if (shape[i] == 1): 
                #self._ndim -= 1   
                dup[i] = 2
        volume = np.tile(volume, dup)

        shape = volume.shape
        ncells = np.prod(shape)
        f.write(pack(4*'I', *shape))              # Bytes 9-24 Number of cells along the X,Y,Z axes (32 bit integer); Bytes 21-24 Number of channels (32 bit integer, supported values: 1 or 3)           
        f.write(pack(6*'f', *self._bounding_box)) # Bytes 25-48 Axis-aligned bounding box of the data stored in single precision order: (xmin, ymin, zmin, xmax, ymax, zmax)
    
        # Write the data: Bytes 49-*
        # Binary data of the volume stored in the specified encoding. The data are ordered so that the following C-style indexing operation makes sense
        # after the file has been mapped into memory: data[((zpos*yres + ypos)*xres + xpos)*channels + chan]
        # where (xpos, ypos, zpos, chan) denotes the lookup location.
        f.write(pack('f'*ncells, *volume.ravel(order='F')))
        f.close()
    
    #def set_scale(self, value):
        #self._scale = float(value)
        
    def set_albedo(self, data):
        volume = np.array(data)
        assert ( (volume.max() <= 1) and (volume.min() >= 0) ), "Values of albedo should be between 0 to 1" 

        if volume.ndim == 4:
            assert (volume.shape[3] == 3),"albedo fourth dimention should be of size 3"
            
        if volume.shape == ():
            self._albedo = data
        else:
            if self._shape == () : 
                self.set_volume(volume.shape)
            else:
                assert self._shape[:,:,:] == volume.shape, "Grid volume does not agree with medium shape" 
            if (volume.ndim <= 3):
                self._albedo = dup_volume(volume, [1, 1, 1, 3])
            self.set_vol_file(self._albedo, 'albedo')
    
    def set_phase(self, data=0.85):
        volume = np.array(data)
        assert ( (volume.max() < 1) and (volume.min() > -1) ), "Values of g should be between -1 to 1, but not equal"    
        if volume.shape == ():
            self._phase = float(data)
        else:
            if self._shape == () : 
                self.set_volume(volume.shape)
            else:
                assert self._shape == volume.shape, "Grid volume does not agree with medium shape"                    
                
            self._phase = volume
            self.set_vol_file(volume, 'phase')
        
    def set_volume(self, shape):    
        if shape == ():
            self._shape = (1,1,1)            
        else:
            self._shape = shape
    
    def set_boundary(self, x='open', y='open'):
        # Set boundary conditions to open / periodic 
        # Input - x: 'open' / 'periodic' ;  y: 'open' / 'periodic'
        self._boundaries['x'] = x
        self._boundaries['y'] = y
        
    def set_density(self, data, bounds):
        # Scale the data to the interval [0,1]: needed for woodcock integration
        # this scale is later used as a parameters in mitsuba
        # also sets the medium shape
        density_mat = np.array(data)
        if self._shape == () : 
            self.set_volume(density_mat.shape)
        else:
            assert self._shape == density_mat.shape, "Grid volume does not agree with medium shape"  
                   
        self.create_boundingbox(bounds)
        self._scale = float(density_mat.max())
                                    
        density_norm = density_mat / self._scale
        
        if density_mat.shape == ():
            self._density = float(data)
        else:
            assert self._shape == density_mat.shape, "Grid volume does not agree with medium shape"         
            self._density = density_mat
            self.set_vol_file(density_norm, 'density')
    
    def get_albedo_data(self):
        return self._albedo
               
    def get_boundary_conditions(self):
        if self._boundaries == None:
            self._boundaries = {'x' : 'open', 'y' : 'open' }
        return self._boundaries
    
    def get_density_data(self):
        return self._density
   
    def get_phase_data(self):
        return self._phase    
        
    def get_world_transform(self):
        bb           = self._bounding_box
        bottom_left  = Vector(bb[0], bb[1], bb[2])
        top_right    = Vector(bb[3], bb[4], bb[5])
        scale_v      = (top_right - bottom_left) / 2.0
        translate_v  = bottom_left + scale_v
        transform    = Transform.translate(translate_v) * Transform.scale(scale_v)
        return transform       

    def albedo_to_mitsuba(self):
        volume = self.get_albedo_data()
        if np.array(volume).shape == ():
            albedo_str = {
                'type' : 'constvolume',
                'value' : Spectrum(volume)
            }            
        else:
            albedo_str = {
                'type' : 'gridvolume',
                'filename' : self._files['albedo']
            }
        return albedo_str
    
    def density_to_mitsuba(self):
        volume = self.get_density_data()
        if np.array(volume).shape == ():
            density_str = {
                'type' : 'constvolume',
                'value' : volume
            }            
        else:
            density_str = {
                'type' : 'gridvolume',
                'filename' : self._files['density']
            }
        return density_str

    def phase_to_mitsuba(self):
        volume = self.get_phase_data()
        if np.array(volume).shape == ():
            phase_str = {
                'type' : 'hg',
                'g' : {
                    'type' : 'constvolume',
                    'value' : volume
                }
            }            
        else:
            phase_str = {
                'type' : 'hg',
                'g' : {
                    'type' : 'gridvolume',
                    'filename' : self._files['phase']
                }
            }
        return phase_str

    def bounding_box_to_mitsuba(self, scene_type='smallMedium'):
        if scene_type is 'step':
            bounding_box_str = self._pmgr.create({
                'type'     : 'cube',
                'toWorld'  : self.get_world_transform(),
                'interior' : self.medium_to_mitsuba(scene_type)
            })
        else:
            bounding_box_str = self._pmgr.create({
                'type'     : 'obj',
                'filename' : self._files['bounding_box'],
                'interior' : self.medium_to_mitsuba(scene_type)
            })         
            
        return bounding_box_str
            
    def medium_to_mitsuba(self, scene_type='smallMedium'):
        bc = self.get_boundary_conditions()
        if scene_type is 'step':
            medium = self._pmgr.create({
                'type'      : 'heterogeneous',
                'method'    : 'simpson',
                'xBoundary' : bc['x'],
                'yBoundary' : bc['y'],
                'density'   : self.density_to_mitsuba(),
                'albedo'    : self.albedo_to_mitsuba(),
                'scale'     : self.scale,
                'phase'     : self.phase_to_mitsuba()
                })
        else:
            medium = self._pmgr.create({
                'type'      : 'heterogeneous',
                'method'    : 'woodcock',
                'xBoundary' : bc['x'],
                'yBoundary' : bc['y'],
                'density'   : self.density_to_mitsuba(),
                'albedo'    : self.albedo_to_mitsuba(),
                'scale'     : self.scale,                
                'phase'     : self.phase_to_mitsuba()
            })            
        return medium    

    def density_to_dict(self):
        file_name  = os.path.realpath(self._files['density'])
        dictionary = {
            'name'   : 'density',
            'type'   : 'gridvolume',
            'string' : {
                'name'  : 'filename',
                'value' : file_name
            }
        }
        return dictionary

    def phase_to_dict(self):
        volume = np.array(self.get_phase_data())
        if volume.shape == ():
            dictionary = {
                'type'   : 'hg',
                'volume' : {
                    'name'  : 'g',
                    'type'  : 'constvolume',
                    'float' : {
                        'name'  : 'value',
                        'value' : float(volume)
                    }
                }
            }
            
        else:
            file_name  = os.path.realpath(self._files['phase'])
            dictionary = {
                'type'   : 'hg',
                'volume' : {
                    'name'   : 'g',
                    'type'   : 'gridvolume',
                    'string' : {
                        'name'  : 'filename',
                        'value' : file_name
                    }
                }
            }
        return dictionary

    def albedo_to_dict(self):
        volume     = self.get_albedo_data()
        dictionary = {
            'name'     : 'albedo',
            'type'     : 'constvolume',
            'spectrum' : {
                'name'  : 'value',
                'value' : float(volume)
            }
        }
        return dictionary

    def bounding_to_dict(self):
        _id        = 'OneVoxel'
        dictionary = {
            'type'   : 'obj',
            'string' : {
                'name'  : 'filename',
                'value' : os.path.realpath(self._files['bounding_box'])
            },
            'ref':{
                'name' : 'interior',
                'id'   : _id
            }
        }
        return dictionary
    
    def to_dict(self):
        _id        = 'OneVoxel'
        dictionary = {
            'type'  : 'heterogeneous',
            'id'    : _id,
            'string': {
                'name'  : 'method',
                'value' : 'woodcock'
            },
            ('volume',0) : self.density_to_dict(),
            ('volume',1) : self.albedo_to_dict(),
            'phase'      : self.phase_to_dict(),
            'float'      : {
                'name'  : 'scale',
                'value' : self._scale
            }
        }
        return dictionary
    
    @property
    def scale(self):
        return self._scale
   
    @property
    def albedo(self):
        return self._albedo

    @property
    def phase(self):
        return self._phase
   
    @property
    def density(self):
        return self._density
    
    @property
    def shape(self):
        return self._shape
    
    @property 
    def bounding_box(self):
        return self._bounding_box