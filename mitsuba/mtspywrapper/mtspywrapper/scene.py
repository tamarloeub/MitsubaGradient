import os
import mitsuba
import mitsuba.core 
from mitsuba.render import Scene
from mtspywrapper import *
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString 


class pyScene(object):
    def __init__(self):
        self._scene  = Scene()         
        self._pmgr   = PluginManager.getInstance()
        self._medium = None
        self._sensor = None
        
        self._scene_set      = False        
        self._integrator_str = None
        self._emitter_str    = None
        
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
            
    def set_medium(self, beta, bounding_box=None, g=None):    
        # Create medium with bounding box
        self._medium = pyMedium()
        if g is None:
            g = 0.85
            
        self._medium.set_phase(g)
        self._medium.set_albedo(1)
    
        # Define the extinction field (\beta) in [km^-1]
        if bounding_box is None:
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
        
    def set_scene(self, beta=(), origin=None, target=None, up=None, nSamples=4096, bounding_box=None, g=None):
        if origin is None:
            origin = Point(0, 0, 3)
        if target is None:
            target = Point(0, 0, 1)
        if up     is None:
            up     = Vector(1, 0, 0)    

        self.set_sensor_film_sampler(origin, target, up, nSamples)        
        self.set_integrator()
        self.set_emitter()
        self.set_medium(beta, bounding_box, g)
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
            
        self._scene.configure()        
        
        return self._scene
    
    def create_new_scene(self, beta=(), origin=None, target=None, up=None, nSamples=4096, scene_type='smallMedium', bounding_box=None, g=None):
        self.set_scene(beta, origin, target, up, nSamples, bounding_box, g)
        return self.configure_scene(scene_type)
        
    def copy_scene(self):
        new_scene = pyScene()
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
            
        new_scene = self.copy_scene()        
        new_scene.set_sensor_film_sampler(origin, target, up, nSamples)
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
                'value' : '1'
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
            'shape'      : self._medium.bounding_to_dict(),
            'sensor'     : self._sensor.to_dict(),
            'emitter'    : self.emitter_to_dict(),
            'integrator' : self.integrator_to_dict()                                
        }
        return dictionary
    
    def xml_rec(self, elem, dictionary):
        for atr in dictionary.keys():
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
    
    