import numpy as np
from mtspywrapper import pyMedium
from mitsuba.core import PluginManager, Vector, Point, Transform, Spectrum
from mitsuba.render import Sensor

class pySampler(object):
    def __init__(self):
        self._type = 'ldsampler'
        self._sampleCount = 4096
        
    def set_sampler(self, stype='ldsampler', sampleCount=4096):
        self._type = stype
        self._sampleCount = sampleCount
        
    def get_type(self):
        return self._type

    def get_sample_count(self):
        return self._sampleCount
    
    def sample_to_mitsuba(self):
        sample_str = self._pmgr.create({
            'type'        : self.get_type(),
            'sampleCount' : self.get_sample_count()
        })
        return sample_str
    
class pyFilm(object):
    def __init__(self):
        self._type       = 'mfilm'
        self._width      = None
        self._height     = None       
        self._fileFormat = None
        
    def set_film(self, ftype='mfilm', width=None, height=None, fileFormat=None):
        self._type = ftype
        if width is not None:
            if height is not None:
                self._width  = width
                self._height = height
        else:
            assert ( (width is None) and (height is None) ), "width and height need to be set together (one of them is None)"    
            
        if fileFormat is not None:
            self._fileFormat = fileFormat
        
    def get_type(self):
        return self._type
    
    def film_to_mitsuba(self):
        if self._width is not None:
            if self._fileFormat is not None:
                film_str = self._pmgr.create({
                    'type' : self.get_type(),
                    'fileFormat' : self._fileFormat,               
                    'width' : self._width,
                    'height' : self._height
                })
            else:
                film_str = self._pmgr.create({
                    'type' : self.get_type(),
                    'width' : self._width,
                    'height' : self._height
                })
        else:
            film_str = self._pmgr.create({
                'type'        : self.get_type()
            })
        return film_str

class pySensor(object):
    def __init__(self):
        self._pmgr    = PluginManager.getInstance() 
        self._type    = None
        self._film    = pyFilm()
        self._sampler = pySampler()
    
    def set_sensor_type(self, sensor_type):
        self._type = sensor_type
        
    def get_sensor_type(self):
        return self._type
    
    def set_film(self, ftype='mfilm', width=None, height=None, fileFormat=None):
        self._film.set_film(ftype, width, height, fileFormat)
    
    def set_to_world(self, origin, target, up):
        self._toWorld = Transform.lookAt(
                origin, # Camera origin
                target, # Camera target
                up )    # 'up' vector
    
    def get_film(self):
        return self._film
    
    def set_sampler(self, num_samples, sampler_type='ldsampler'):
        self._sampler.set_sampler(sampler_type, num_samples)
        
        
    def get_sampler(self):
        return self._sampler
    
    def get_world_transform(self):
        return self._toWorld
        
    def sensor_to_mitsuba(self):
        sampler = self.get_sampler()
        film    = self.get_film()
        sensor_str = self._pmgr.create({
            'type'    : self.get_sensor_type(),
            'toWorld' : self.get_world_transform(),
            'film'    : film.film_to_mitsuba(),
            'sampler' : sampler.sample_to_mitsuba()
        })
        return sensor_str
    
    
class pyParallelRaySensor(pySensor):
    
    def __init__(self, medium):
        """"""
        super(pyParallelRaySensor, self).__init__()
        self._medium_bounding_box = medium.bounding_box
        width = medium.shape[0]
        try:
            height = medium.shape[1]
        except IndexError:
            height = 1
        self.set_sensor_type('orthographic')
        self.set_world_transform(0.0, 0.0)
        self.set_film(width, height)

    def set_world_transform(self, view_zenith, view_azimuth):
        """ 
        Set a parallal ray sensor world transform 
        Input: 
          view_zenith [deg]
          view_azimuth [deg]  
        """
        [xmin, ymin, zmin, xmax, ymax, zmax] = self._medium_bounding_box

        cos_azimuth, sin_azimuth = np.cos(np.deg2rad(view_azimuth)), np.sin(np.deg2rad(view_azimuth))
        cos_zenith, sin_zenith = np.cos(np.deg2rad(view_zenith)), np.sin(np.deg2rad(view_zenith))

        assert(cos_zenith > 0.0), "Error: cos(zenith) < 0.0 ===> zenith > 90.0"
        scale_vector = Vector(
            (xmax/2) * (cos_zenith + (1 - cos_zenith) * np.abs(sin_azimuth)), 
            (ymax/2) * (cos_zenith + (1 - cos_zenith) * np.abs(cos_azimuth)), 
            1.0
        )
        translation_vector = Vector(
            cos_azimuth * sin_zenith, 
            sin_azimuth * sin_zenith, 
            cos_zenith
        )        
        target = Point(xmax/2, ymax/2, zmax)
        origin = target + (xmax**2 + ymax**2) * translation_vector 
        up = Vector(0,1,0)

        self._world_transform = Transform.lookAt(origin, target, up) * Transform.scale(scale_vector)
        
    def get_world_transform(self):
        return self._world_transform        