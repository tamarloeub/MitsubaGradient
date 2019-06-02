import os
import mitsuba
import mitsuba.core 
from mitsuba.render import Scene
from mtspywrapper import *
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString 

class Ocean(object):
    def __init__(self):
        self._shape   = None
        self._diffuse = None
        self._transz  = None
        self._scalexy = None
        
    def set_diffuse(self, val):
        self._diffuse = val
    
    def set_transz(self, val):
        self._transz = val
    
    def set_scalexy(self, val):
        self._scalexy = float(val)
        
    def get_diffuse(self):
        return self._diffuse
    
    def get_transz(self):
        return self._transz
    
    def get_scalexy(self):
        return self._scalexy
        
        
class pyScene(object):
    def __init__(self):
        self._scene  = Scene()         
        self._pmgr   = PluginManager.getInstance()
        self._medium = None
        self._sensor = None

        self._scene_set      = False        
        self._integrator_str = None
        self._emitter_str    = None
        self._ocean          = Ocean()

    def get_scene_beta(self):
        if self._medium == None:
            return None
        medium = self._medium
        return medium.get_density_data()

    def calculate_fov(self, points):
        #bounds = [xmin, ymin, zmin, xmax, ymax, zmax]
        # fov is set by axis x
        bounds        = self._medium._bounding_box
        max_medium    = np.array([bounds[3], bounds[4], bounds[5]])
        min_medium    = np.array([bounds[0], bounds[1], bounds[2]])
        medium_center = ( max_medium + min_medium ) / 2
        focal_point   = np.array([points['target'][0], points['target'][1], points['target'][2]])
        sensor_origin = np.array([points['origin'][0], points['origin'][1], points['origin'][2]])

        L       = np.max([max_medium[0] - min_medium[0], max_medium[1] - min_medium[1]]) / 2 #camera's FOV covers the whole medium
        H       = np.linalg.norm(sensor_origin - focal_point) 
        fov_rad = 2 * np.arctan(L / H)
        fov_deg = 180 * fov_rad / np.pi
        return fov_deg

    def set_sensor_film_sampler(self, sensorType, points, nSamples, fov_f=False, fov=None, width=None, height=None):        
        # Create a sensor, film & sample generator
        self._sensor = pySensor()
        if sensorType is None:
            sensorType = 'radiancemeter'
        self._sensor.set_type(sensorType)
        self._sensor.set_sampler(nSamples)
        self._sensor.set_to_world(points) 
        self._sensor.set_medium(self._medium)
        if sensorType is 'perspective':
            if (fov is None) and (fov_f is True):
                #else use Mitsuba's default FOV
                fov = self.calculate_fov(points)
            if fov is not None:
                self._sensor.set_fov(fov)
            self._sensor.set_film(width=width, height=height)

    def set_integrator(self):
        # Create integrator
        self._integrator_str = self._pmgr.create({
            'type' : 'volpath_simple', 
            'maxDepth' : -1    
        })
    
    def set_ocean(self):
        zmin = self._medium._bounding_box[2]
        zmax = self._medium._bounding_box[-1]
        #tz   = (zmax - zmin)
        tz   = 1
        self._ocean.set_diffuse(Spectrum(0.01))
        self._ocean.set_transz(-tz)
        self._ocean.set_scalexy(100)
        
        scalexy = self._ocean.get_scalexy()
        self._ocean._shape = self._pmgr.create({
            'type' : 'rectangle',
            'bsdf' : {
                    'type' : 'diffuse',
                    'reflectance' : self._ocean.get_diffuse()
            },
            'toWorld' : Transform.translate(Vector(0, 0, -tz)).scale(Vector(scalexy, scalexy, 1.))#Vector(1., 1.,20.))
        }) 
   
    def set_emitter(self):
        # Create a light source
        self._emitter_str = self._pmgr.create({
            'type' : 'directional',
            'direction' : Vector(0, 0, -1),
            #'toWorld' :  Transform.rotate(Vector(0,0,1), 180),            
            'irradiance' : Spectrum(100)
        })

    def set_medium(self, beta, albedo=None, bounding_box=None, g=None):    
        # Create medium with bounding box
        self._medium = pyMedium()
        if g is None:
            g = 0.85
        if albedo is None:
            albedo = 1             
        self._medium.set_phase(g)
        self._medium.set_albedo(albedo)

        # Define the extinction field (\beta) in [km^-1]
        if bounding_box is None:
            bounding_box = [-1, -1, -1, 1, 1, 1]   # [xmin, ymin, zmin, xmax, ymax, zmax] in km units 
            #bounding_box = [-250, -250, 0, 250, 250, 400]

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

    def set_scene(self, beta=(), albedo=None, sensorType=None, points=None, nSamples=4096, fov_f=False, fov=None, width=None, height=None, bounding_box=None, g=None):
        if points is None:
            points = dict()
            points['origin'] = Point(0, 0, 3)
            points['target'] = Point(0, 0, 1)            
            points['up']     = Vector(1, 0, 0)   

        if points['origin'] is None:
            points['origin'] = Point(0, 0, 3)            
        if points['target'] is None:
            points['target'] = Point(0, 0, 1)
        if points['up']     is None:
            points['up']     = Vector(1, 0, 0)           

        self.set_medium(beta, albedo, bounding_box, g)
        self.set_sensor_film_sampler(sensorType, points, nSamples, fov_f, fov, width, height)        
        self.set_integrator()
        self.set_emitter()
        self.set_ocean()
        self._scene_set = True

    def configure_scene(self, scene_type='smallMedium'):            
        # Set the sensor, film & sample generator
        self._scene.addChild(self._sensor.sensor_to_mitsuba())        

        # Set the integrator
        self._scene.addChild(self._integrator_str)

        # Set the emiter - light source
        self._scene.addChild(self._emitter_str)

        # Set bounding box
        self._scene.addChild(self._medium.bounding_box_to_mitsuba(scene_type))    

        # Set medium
        self._scene.addChild(self._medium.medium_to_mitsuba(scene_type))  
        
        # Set ocean
        self._scene.addChild(self._ocean._shape)        

        self._scene.configure()        

        return self._scene

    def create_new_scene(self, beta=(), albedo=None, sensorType=None, origin=None, target=None, up=None, nSamples=4096, fov_f=False, fov=None, width=None, height=None, scene_type='smallMedium', bounding_box=None, g=None):
        if (origin is not None) and (target is not None) and (up is not None):
            points           = dict()
            points['origin'] = origin
            points['target'] = target            
            points['up']     = up     
        else:
            points = None
        self.set_scene(beta, albedo, sensorType, points, nSamples, fov_f, fov, width, height, bounding_box, g)
        self.configure_scene(scene_type)
        self.scene_to_xml()
        return

    def copy_scene(self):
        new_scene         = pyScene()
        new_scene._medium = self._medium
        new_scene._sensor = self._sensor

        new_scene._scene_set      = self._scene_set
        new_scene._integrator_str = self._integrator_str
        new_scene._emitter_str    = self._emitter_str
        return new_scene

    def copy_scene_with_different_density(self, beta):
        assert (self._scene_set is True), "Can't copy unset scene"
        new_scene = self.copy_scene()        
        new_scene.set_medium(beta)
        new_scene.configure_scene()
        return new_scene

    def copy_scene_with_different_sensor_position(self, origin, target, up, nSamples=None):
        assert (self._scene_set is True), "Can't copy unset scene"
        if nSamples is None:
            sampler = self._sensor.get_sampler()
            nSamples = sampler.get_sample_count()

        points           = dict()
        points['origin'] = origin
        points['target'] = target            
        points['up']     = up                    

        new_scene  = self.copy_scene() 
        sensorType = self._sensor.get_type()
        fov        = self._sensor.get_fov()
        width      = self._sensor._film.get_width()
        height     = self._sensor._film.get_height()
        new_scene.set_sensor_film_sampler(sensorType, points, nSamples, fov, width, height)
        new_scene.configure_scene()
        return new_scene        

    def emitter_to_dict(self): ##update
        dictionary = {
            'type'   : 'directional',
            'vector' : {
                'name' : 'direction',
                'x'    : '0',
                'y'    : '0',
                'z'    : '-1'
                },
            'spectrum' : {
                'name'  : 'irradiance',
                'value' : '100'
            }
        }
        return dictionary
    
    def ocean_to_dict(self):
        #zmin = self._medium._bounding_box[2]
        #zmax = self._medium._bounding_box[-1]
        #tz   = (zmax - zmin)
        
        dictionary = {
            'type' : 'rectangle',
            'bsdf' : {
                'type'  : 'diffuse',                
                'spectrum' : {
                    'name' : 'reflectance',
                    'value' : str(self._ocean.get_diffuse()[0]) #'0.05'
                },
            },
            'transform' : {
                'name' : 'toWorld',
                'scale' : {
                    'x' : str(self._ocean.get_scalexy()),
                    'y' : str(self._ocean.get_scalexy())
                },
                'translate' : {
                    'z' : str(self._ocean.get_transz())
                }
            } 
        }
        return dictionary
    
    def integrator_to_dict(self):
        dictionary = {
            'type'    : 'volpath_simple',
            'integer' : {
                'name'  : 'maxDepth',
                'value' : '-1'
            }
        }
        return dictionary      

    def to_dict(self):
        dictionary = {
            'version'    : '0.5.0',
            'medium'     : self._medium.to_dict(),
            #'shape'      : self._medium.bounding_to_dict(),
            ('shape', 0) : self._medium.bounding_to_dict(),
            'sensor'     : self._sensor.to_dict(),
            'emitter'    : self.emitter_to_dict(),
            'integrator' : self.integrator_to_dict(),
            ('shape', 1) : self.ocean_to_dict()
        }
        return dictionary

    def xml_rec(self, elem, dictionary):
        # first export medium to xml:
        if 'medium' in dictionary.keys():
            if type(dictionary['medium']) == dict:
                sub_elem = ET.SubElement(elem,'medium')
                sub_elem = self.xml_rec(sub_elem,dictionary['medium'])
            else:
                if type(dictionary['medium']) != str:
                    value = str(dictionary['medium'])
                else:
                    value = dictionary['medium']
                elem.set('medium', value)            

        # export the rest of the dict to xml:
        for atr in dictionary.keys():
            if atr == 'medium':
                continue
            
            if type(atr) == tuple:
                atr_temp = atr[0]
            else:
                atr_temp = atr

            if type(dictionary[atr]) == dict:
                sub_elem = ET.SubElement(elem,atr_temp)
                sub_elem = self.xml_rec(sub_elem,dictionary[atr])
            else:
                if type(dictionary[atr]) != str:
                    value = str(dictionary[atr])
                else:
                    value = dictionary[atr]
                elem.set(atr, value)
        return elem

    def scene_to_xml(self):
        filename  = os.path.realpath(self._medium._medium_path) + '/scene.xml'
        f         = open(filename, 'w')         
        root      = self.xml_rec(ET.Element("scene"), self.to_dict())
        rough_str = ET.tostring(root, 'utf-8')
        reparsed  = parseString(rough_str)
        xml_txt   = reparsed.toprettyxml(indent="\t")
        f.write(xml_txt)
        f.close()
        return filename

