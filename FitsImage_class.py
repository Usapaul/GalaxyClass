
# ----------- Class AstroImage -----------------

import imfit_model
import imfit_module
import add_functions
import general_functions
import fits2image

import numpy as np
from astropy.io import fits

import json

from datetime import datetime
import tempfile

import os
import shutil
import subprocess


#-----------------------------------------------

def isfloat(input_string):
    ''' Returns True if the input_string string
        can be interpreted as a float number and returns False otherwise 
    '''
    try:
        float(input_string)
    except ValueError:
        return False
    else:
        return True




#-----------------------------------------------



#-----------------------------------------------
#           * ^ * ^ * ^ * ^ * ^ * ^ *
# -------------- Class FitsImage ---------------
#-----------------------------------------------


class FitsImage(object):
    ''' A class which describes fits-image
        based on an existing fits file or specified by numpy 2d-array
    '''
    # default:
    _filename = None # name of fits file used
    _tmp_filename = None # a temporary file for FITS, deleting after work
    _abspath = None # fits file name with full abs path location
    _dirname = None # directory which contains fits file
    _data = None # numpy array of pixel values, could be 3-dimension
    _header = None # header of fits file
    #
    _pixelscale = None # pixelscale in arcsec/pixel or another scale
    _pixelscale_unit = None
    #
    _wavelength = None # the wavelength of the passband in nm (may change)
    _wavelength_unit = 'nm' # nm, micron, cm, m and etc. nm — as default
    _passbandName = None # the passband name of the passband for certain image    
    #
    _ra = None # right ascension of the image center
    _dec = None # declination of the image center
    #
    _pixel_unit = None # e/sec, adu and anything else
    _exposure = None # always in seconds
    #
    comment = None # if needed
    history = None # manually written history of the GalaxyImage (1 line text)


    def __init__(self, input_image, **kwargs):
        ''' Input: filename or numpy.ndarray or hdu object'''
        if isinstance(input_image, str):
            # it's a name of fits file
            abspath = os.path.abspath(input_image)
            dirname, basename = os.path.split(abspath)
            self._abspath = abspath
            self._dirname = dirname
            self._filename = basename
            hdul = fits.open(input_image)
        elif isinstance(input_image, np.ndarray):
            data = input_image
            fd, tmp_filename = tempfile.mkstemp()
            fits.writeto(tmp_filename, data, overwrite=True)
            hdul = fits.open(tmp_filename)

        elif isinstance(input_image, fits.hdu.hdulist.HDUList):
            hdul = input_image
        else:
            raise ValueError('Unknown type of input')
        #
        self._data = np.array(hdul[0].data)
        self._header = hdul[0].header.copy()
        hdul.close()
        for key, value in kwargs.items():
            setattr(self, key, value)


    @classmethod
    def from_data(cls, data):
        return cls(data)


    @classmethod
    def from_file(cls, filename, **kwargs):
        return cls(filename, **kwargs)

    @classmethod
    def from_json(cls, json_filename):
        with open(json_filename) as file:
            json_dict = json.load(file)
        _abspath = json_dict.get('_abspath')
        _filename = json_dict.get('_filename')
        if _abspath is not None:
            return cls.from_file(_abspath, **json_dict)
        elif _filename is not None:
            return cls.from_file(_filename, **json_dict)
        else:
            raise ValueError("No abspath/filename is presented in json")


    def __str__(self):
        return f"{self.__class__.__name__} {self.filename}"
        
    
    def __repr__(self):
        output_str = self.__class__.__name__
        interesting_params = [
            self.filename, self.shape, self.ra, self.dec,
            self.pixelscale, self.pixelscale_unit]
        for item in interesting_params:
            if item is not None:
                output_str += ', ' + str(item)
        return output_str

    def __copy__(self):
        data = self.data.copy()
        kwargs_dict = self.__dict__.copy()
        for key in ['_filename', '_abspath', '_dirname']:
            popped = kwargs_dict.pop(key, d=None)
        return self.__class__(data, **kwargs_dict)

    def copy(self):
        return self.__copy__()



    def convert_to(self, input_type):
        ''' Converts data array to the specified type
            (int, float (=np.float64), np.float32)
        '''
        if isinstance(input_type, str):
            input_type_str = input_type
        else:
            input_type_str = str(input_type.dtype)
        #
        if input_type_str in ['float', 'np.float64',
                              'numpy.float64', 'float64']:
            input_type = np.float64
            bitpix = -64
        elif input_type_str in ['np.float32', 'numpy.float32', 'float32']:
            input_type = np.float32
            bitpix = -32
        elif input_type_str in ['int', 'integer', 'np.int','numpy.int',
                                'np.int64', 'numpy.int64', 'int64']:
            input_type = np.int64
            bitpix = 64
        elif input_type_str in ['np.int32', 'numpy.int32', 'int32']:
            input_type = np.int32
            bitpix = 32
        elif input_type_str in ['np.int16', 'numpy.int16', 'int16']:
            input_type = np.int16
            bitpix = 16
        elif input_type_str in ['np.uint8', 'numpy.uint8', 'uint8']:
            input_type = np.uint8
            bitpix = 8
        else:
            raise ValueError("input type (string) is unknown")
        #
        old_data = self.data
        self.data = np.array(old_data, dtype=input_type)
        self.header['BITPIX'] = bitpix


    # - - - - - - - - - - - - - - - - - - - - - - -

    @property
    def data(self):
        ''' numpy array of pixel values data'''
        return self._data

    @data.setter
    def data(self, input_array):
        self._data = np.array(input_array)

    @data.deleter
    def data(self):
        self._data = None

    # - - - - - - - - - - - - - - - - - - - - - - -

    @property
    def header(self):
        ''' header of fits image'''
        return self._header

    @header.setter
    def header(self, input_header):
        if input_header is not None:
            self._header = fits.header.Header(input_header)
        else:
            self._header = None

    @header.deleter
    def header(self):
        self._header = None


    def create_header(self):
        ''' create a header attribute of the GalaxyImage object
            which takes almost empty header_template.dat file
            and changes NAXIS1/2 parameters to the known image shape
        '''
        file_dir = os.path.abspath(__file__)
        header_template_abspath = os.path.join(file_dir, 'header_template.dat')
        header = fits.header.Header.from_file(header_template_abspath)
        header['NAXIS1'] = self.xN
        header['NAXIS2'] = self.yN
        header['DATE'] = datetime.today().strftime("%Y-%m-%dT%H:%M:%S")
        self._header = header


    # - - - - - - - - - - - - - - - - - - - - - - -

    @property
    def abspath(self):
        ''' absolute filename system path (os.path.abspath)'''
        return self._abspath

    @abspath.setter
    def abspath(self, input_str):
        abs_path = os.path.abspath(input_str)
        dirname, basename = os.path.split(abs_path)
        self._abspath = abs_path
        self._dirname = dirname
        self._filename = basename


    @abspath.deleter
    def abspath(self):
        self._abspath = None
        self._dirname = None
        self._filename = None


    # - - - - - - - - - - - - - - - - - - - - - - -

    @property
    def filename(self):
        ''' filename on disk (os.path.basename)'''
        return self._filename

    @filename.setter
    def filename(self, filename):
        self._filename = filename
        if os.path.exists(filename):
            abspath = os.path.abspath(filename)
            dirname = os.path.dirname(filename)
        else:
            dirname = os.getcwd()
            abspath = os.path.join(dirname, filename)
        self._abspath = abspath
        self._dirname = dirname

    @filename.deleter
    def filename(self):
        self._filename = None
        self._abspath = None
        self._dirname = None


    @property
    def name(self):
        return self.filename

    # - - - - - - - - - - - - - - - - - - - - - - -

    @property
    def dirname(self):
        ''' directory name of image file on disk (os.path.dirname)'''
        return self._dirname

    @dirname.setter
    def dirname(self, input_str):
        dirname = os.path.abspath(input_str)
        if self.filename is not None:
            basename = self.filename
        else:
            basename = 'tmp.fits'
        abspath = os.path.join(dirname,basename)
        self._abspath = abspath
        self._dirname = dirname
        self._filename = basename

    @dirname.deleter
    def dirname(self):
        # delete only directory (temporary may be)
        self._dirname = None


    # - - - - - - - - - - - - - - - - - - - - - - -


    def update_from_file(self, filename):
        ''' Updates the GalaxyImage object with another fits file 
            (wavelength and pixelscale are remain the same)
        '''
        if os.path.exists(filename):
            old_shape = self.shape
            hdul = fits.open(filename)
            new_data = hdul[0].data.copy()
            if new_data.shape != old_shape:
                raise ValueError("New image in file has different shape")
            self.data = new_data
            self.header = hdul[0].header.copy()
            self.filename = filename
            hdul.close()
            # other parameters would not be setted manually again
        else:
            raise ValueError(f"File {filename} does not exist")


    def update(self):
        ''' Updates the GalaxyImage object with data 
            in the same fits file
        '''
        if self.filename is not None:
            filename = self.filename
            self.update_from_file(filename)            
        else:
            raise ValueError(f"Name of the updating file is not specified")


    def save(self):
        ''' Saves Image to the already specified fits file'''
        if self.filename is None:
            raise ValueError(f"GalaxyImage object does not have a filename")
        fits.writeto(self.abspath, self.data, self.header, overwrite=True)


    def writeto(self, filename):
        ''' Saves Image to the [filename] fits file'''
        self.filename = filename
        self.save()
        

    def copyto(self, filename):
        ''' Copies fits-image to a fits file with another [filename] '''
        fits.writeto(filename, self.data, self.header, overwrite=True)


    def to_jpg(self, filename=None, overwrite=False):
        ''' It uses fits2image() in module fits2image'''
        if self.abspath is None:
            if self.data is None:
                raise ValueError("No data or fits file")
            fd, tmp_filename = tempfile.mkstemp()
            fits_file = os.path.abspath(tmp_filename)
            self.copyto(fits_file)
        else:
            fits_file = self.abspath 
        if filename is None:
            splitted = fits_file.split('.')
            file = '.'.join(splitted[:-1])
            extension = splitted[-1]
            output_file = f"{file}_jpg.{extension}"
            if os.path.exists(output_file):
                if overwrite is True:
                    os.remove(output_file)
                else:
                    raise ValueError(f"Specify the filename"
                                     " or set overwrite=True")
        else:
            if os.path.exists(filename):
                if overwrite is True:
                    os.remove(filename)
                    output_file = filename
                else:
                    raise ValueError(f"File {filename} exists."
                                     " Give a new name or set overwrite=True")
        #
        if os.path.dirname(output_file) == os.getcwd():
            output_file = os.path.basename(output_file)
        fits2image.fits2image(fits_file, output_file)
        #
        return f"Saved as JPG to file {output_file}"


    def open_ds9(self, command_string=None):
        ''' Runs DS9 with current fits data
            and special DS9 parameters in command_string
        '''
        command = f"ds9 {self.abspath} {command_string}"
        popen_output = subprocess.Popen(command, shell=True)

    def refresh_ds9(self):
        ''' NEED USE PYDS9 LIBRARY!'''
        return None

    # - - - - - - - - - - - - - - - - - - - - - - -

    @property
    def galaxy_name(self):
        ''' The name of the galaxy associated with the given image'''
        return self._galaxy_name

    @galaxy_name.setter
    def galaxy_name(self, input_str):
        self._galaxy_name = input_str

    @galaxy_name.deleter
    def galaxy_name(self, input_str):
        self._galaxy_name = None

    # - - - - - - - - - - - - - - - - - - - - - - -

    @property
    def wavelength(self):
        ''' Returnes known wavelength for the GalaxyImage in specified units'''
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value):
        ''' float value of wavelength '''
        if isfloat(value):
            self._wavelength = float(value)
        elif value is None:
            self._wavelength = None
        else:
            raise ValueError(f"Imput value for wavelength is NaN")

    @wavelength.deleter
    def wavelength(self):
        self._wavelength = None

    @property
    def wave(self):
        ''' Shows known wavelength for the GalaxyImage in specified units'''
        return self.wavelength

    @property
    def wavelength_int(self):
        return int(self.wavelength)


    # - - - - - - - - - - - - - - - - - - - - - - -

    @property
    def wavelength_unit(self):
        ''' Shows unit of the known wavelength for the GalaxyImage 
            (nm as default but may change)'''
        return self._wavelength_unit

    @wavelength_unit.setter
    def wavelength_unit(self, input_str_unit):
        if input_str_unit is not None:
            self._wavelength_unit = str(input_str_unit)
        else:
            self._wavelength_unit = None

    @wavelength_unit.deleter
    def wavelength_unit(self):
        self._wavelength_unit = None

    @ property
    def wavelength_nm(self):
        ''' The wavelength of the passband in nanometres '''
        if self.wavelength_unit is None or self.wavelength is None:
            return None
        coeff = add_functions.wave_unit_ratio_to_nm(self.wavelength_unit)
        if coeff is None:
            raise ValueError("wavelength_unit is not correct")
        wave_in_nm = coeff * self.wavelength
        return wave_in_nm

    # - - - - - - - - - - - - - - - - - - - - - - -


    @property
    def passbandName(self):
        return self._passbandName

    @passbandName.setter
    def passbandName(self, input_str_unit):
        if input_str_unit is not None:
            self._passbandName = str(input_str_unit)
        else:
            self._passbandName = None

    @passbandName.deleter
    def passbandName(self):
        self._wavelength_unit = None

    @property
    def band(self):
        return self.passbandName

    @property
    def filter(self):
        return self.passbandName

    # - - - - - - - - - - - - - - - - - - - - - - -

    @property
    def pixelscale(self):
        return self._pixelscale

    @pixelscale.setter
    def pixelscale(self, input_scale):
        if isfloat(input_scale):
            self._pixelscale = float(input_scale)
        elif input_scale is None:
            self._pixelscale = None
        else:
            raise ValueError("input for pixelscale is NaN")

    @pixelscale.deleter
    def pixelscale(self):
        self._pixelscale = None

    @property
    def scale(self):
        return self.pixelscale


    @property
    def pixelscale_unit(self):
        return self._pixelscale_unit

    @pixelscale_unit.setter
    def pixelscale_unit(self, input_str_unit):
        if input_str_unit is not None:
            self._pixelscale_unit = str(input_str_unit)
        else:
            self._pixelscale_unit = None

    @pixelscale_unit.deleter
    def pixelscale_unit(self):
        self._pixelscale_unit = None

    @ property
    def pixelscale_arcsec(self):
        ''' The pixelscale in units arcsec/pixel '''
        if self.pixelscale_unit is None or self.pixelscale is None:
            return None
        coeff = add_functions.unit_ratio_to_arcsec(self.pixelscale_unit)
        if coeff is None:
            raise ValueError("pixelscale_unit is not correct")
        pixelscale_in_arcsec = coeff * self.pixelscale
        return pixelscale_in_arcsec

    def pix2arcsec(self, pixelsN):
        ''' Converts input (amount of pixels) to arcsec
            with given pixelscale'''
        return pixelsN * self.pixelscale_arcsec

    def arcsec2pix(self, arcsec):
        ''' Converts input (angular size in arcsec) to pixel number
            with given pixelscale'''
        return arcsec / self.pixelscale_arcsec


    # - - - - - - - - - - - - - - - - - - - - - - -


    @property
    def ra(self):
        ''' RA of the image center'''
        return self._ra

    @ra.setter
    def ra(self, input_ra):
        if isfloat(ra):
            self._ra = float(input_ra)
        else:
            raise ValueError('Input right ascension is NaN')

    @ra.deleter
    def ra(self):
        self._ra = None


    @property
    def dec(self):
        ''' DEC of the image center'''
        return self._dec

    @dec.setter
    def dec(self, input_dec):
        if isfloat(dec):
            self._dec = float(input_dec)
        else:
            raise ValueError('Input declination is NaN')

    @dec.deleter
    def dec(self):
        self._dec = None


    @property
    def pixel_unit(self):
        return self._pixel_unit

    @pixel_unit.setter
    def pixel_unit(self, input_pixel_unit):
        if input_pixel_unit is not None:
            self._pixel_unit = str(input_pixel_unit)
        else:
            self._pixel_unit = None

    @pixel_unit.deleter
    def pixel_unit(self):
        self._pixel_unit = None


    @property
    def exposure(self):
        return self._exposure

    @exposure.setter
    def exposure(self, input_exposure):
        if isfloat(input_exposure):
            self._exposure = float(input_exposure)
        else:
            raise ValueError('Input exposure is NaN')

    @exposure.deleter
    def exposure(self):
        self._exposure = None


    # - - - - - - - - - - - - - - - - - - - - - - -

    @property
    def shape(self):
        ''' shape of data array '''
        return self.data.shape

    @property
    def yN(self):
        ''' size of image y-axis (integer) '''
        return self.data.shape[0]

    @property
    def xN(self):
        ''' size of image x-axis (integer) '''
        return self.data.shape[1]

    @property
    def yC(self):
        ''' row y-center of image (float) '''
        return (self.data.shape[0] - 1) / 2

    @property
    def xC(self):
        ''' column x-center of image (float) '''
        return (self.data.shape[1] - 1) / 2

    @property
    def data1(self):
        ''' the same as data but x and y indices starts from 1 '''
        data1 = np.zeros((data.shape[0]+1, data.shape[1]+1))
        data1[1:,1:] = self.data
        return data1

    @property
    def yC1(self):
        ''' the same as yC but like indices starts from 1 (not 0) '''
        return (self.yC + 1)

    @property
    def xC1(self):
        ''' the same as xC but like indices starts from 1 (not 0) '''
        return (self.xC + 1)

    @property
    def shape_arcsec(self):
        if self.pixelscale is None or self.pixelscale_unit is None:
            return None
        else:
            yN, xN = self.shape
            scale = self.pixelscale_arcsec
            shape_in_arcsec = (yN * scale, xN * scale)
            return shape_in_arcsec


    @property
    def as_dict(self):
        output_dict = {}
        for key, value in self.__dict__.items():
            if (isinstance(key, str | int | float)
                and general_functions.is_python_simple_type(value)):
                output_dict[key] = value
        return output_dict


    def get_dict_from_attrs(self, *args):
        ''' Returns self.as_dict but with only
            named attributes names (in *args) '''
        output_dict = {}
        for arg in args:
            attr_name = arg
            attr_value = getattr(self, attr_name)
            output_dict[attr_name] = attr_value
        return output_dict


    @property
    def as_json(self):
        dictionary = self.as_dict
        json_text = json.dumps(dictionary, indent=4)
        return json_text

    def save_json(self, filename):
        with open(filename, "w") as outfile:
            outfile.write(self.as_json)




#-----------------------------------------------
#           * ^ * ^ * ^ * ^ * ^ * ^ *
# -------------- Class MaskImage ---------------
#-----------------------------------------------



class MaskImage(object):
    ''' Mask for astronomical image, True and False values in each pixel
    '''
    _array = None # numpy array of the mask
    _false_value = False # values which the False pixels will be filled
    _true_value = True # values which the False pixels will be filled
    _filename = None
    #
    def __init__(self, input_mask):
        ''' Input: numpy array or filename'''
        if isinstance(input_mask, np.ndarray):
            self._array = input_mask
        elif isinstance(input_mask, str):
            # it's a filename
            hdul = fits.open(input_mask)
            data = hdul[0].data
            self._array = data.copy()
            hdul.close()
        

