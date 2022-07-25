
# ----------- Class Galaxy -----------------

import imfit_model
import imfit_module
import add_functions
import general_functions
import fits2image

import numpy as np
from astropy.io import fits

from datetime import datetime
import tempfile

import os


#-----------------------------------------------


def isfloat(input):
    ''' Returns True if the input string
        can be interpreted as a float number and returns False otherwise 
    '''
    try:
        float(input)
    except ValueError:
        return False
    else:
        return True



#-----------------------------------------------




class GalaxyImage(object):
    ''' One galaxy may have multiple images
    This class is exactly for a single image:
    it has size/shape, pixelscale, X, Y, PA, ra, dec of the center,
    pixnumber, HISTORY, COMMENT (as lines of the FITS file header)
    
    and methods:
    to jpg(), setNaNs(float_value),
    replace(condition/maskedBOOLEAN_array, new_value, tmp=False) — 
    — replaces the GalaxyImage.data or returns the new_array if tmp is True
    with replaced values
    #
    it would also have images...
    images should be fits hdul files with its own properties
    [...]_image = hdul[0].copy() /// 
    /// but may have method update (image) on the hard drive
    mask_image (mask image array)
    property masked_image(mask_value=np.nan)
    error_image
    psf image
    regions_list — list where each item is region in DS9
    regions_groups — dict where each item is named regions list
    #
    ds9_load_dict — a dict with recommended ds9 options:
    scale=log, scale limits = ..., contours = True and others
    ds9_load_string - a string of ds9_load_dict
    #
    properties :
    (ra, dec) of left bottom corner (pixel center or pixel border??)
    '''



#-----------------------------------------------
#           * ^ * ^ * ^ * ^ * ^ * ^ *
# ----------- Class Galaxy -------------
#-----------------------------------------------


class GalaxyImage(object):
    ''' A class which describes galaxy fits-image
        based on an existing fits file or specified by numpy 2d-array
    '''
    # default:
    _wavelength = None # the wavelength of the passband in nm (may change)
    _wavelength_unit = 'nm' # nm, micron, cm, m and etc. nm — as default
    _passbandName = None # the passband name of the passband for certain image
    #
    _filename = None # name of fits file used
    _abspath = None # fits file name with full abs path location
    _dirname = None # directory which contains fits file
    _galaxy_name = None # name of the Galaxy (optional)
    _data = None # numpy array of pixel values, could be 3-dimension
    _Nchannels = None # number of color channels (RGB, for example)
    _header = None # header of fits file
    #
    _pixelscale = None # pixelscale in arcsec/pixel or another scale
    _pixelscale_unit = None
    #
    _ra = None # right ascension of the image center
    _dec = None # declination of the image center
    #
    model = None # ImfitModelCombination object — known imfit model
    model_filename = None # where to write self.model.as_file_text
    #
    comment = None # if needed
    history = None # manually written history of the GalaxyImage (1 line text)


    def __init__(self, input_image):
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
        self._data = hdul[0].data.copy()
        self._header = hdul[0].header.copy()
        hdul.close()


    @classmethod
    def from_data(cls, data):
        return cls(data)


    @classmethod
    def from_file(cls, filename):
        return cls(filename)


    @classmethod
    def to_jpg(self):
        ''' will use fits2image() in module fits2image'''
        return None


    def __str__(self):
        return self.filename
        
    
    def __repr__(self):
        output_str = 'GalaxyImage'
        interesting_params = [
            self.filename, self.galaxy_name, 
            self.shape, self.wavelength_nm,
            self.ra, self.dec, self.pixelscale]
        for item in interesting_params:
            if item is not None:
                output_str += ', ' + str(item)
        return output_str


    # - - - - - - - - - - - - - - - - - - - - - - -
    #      if new data or filename specified
    #      — change everything

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
        D = {}
        attrs_list = [
            'wavelength', 'wavelength_unit', 'passbandName',
            'filename', 'abspath', 'galaxy_name',
            'pixelscale', 'pixelscale_unit', 'ra', 'dec',
            'comment', 'history']
        for attribute in attrs_list:
            attr_value = getattr(self, attribute)
            if attr_value is not None:
                D[attribute] = attr_value
        model = self.model
        model_filename = self.model_filename
        if model is not None:
            if model_filename is not None:
                D['model_filename'] = model_filename
            D['model'] = model.as_dicts

        if model is not None:
            D['model'] = True
            model_text = model.as_file_text
            if model_filename is not None:
                D['model_filename'] = model_filename
            else:
                model_filename = general_functions.uniq_filename(
                                    prefix='galModel_', ext='.dat',
                                    check_dirname=os.getcwd())
            with open(model_filename, 'w') as file:
                file.write(model_text)
        return D


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
    def as_file_text(self):
        lines = []
        data_dict = self.as_dict
        attrs_list = data_dict.keys()
        model
        for attr in attrs_list:



        if model is not None:
            D['model'] = True
            model_text = model.as_file_text
            if model_filename is not None:
                D['model_filename'] = model_filename
            else:
                model_filename = general_functions.uniq_filename(
                                    prefix='galModel_', ext='.dat',
                                    check_dirname=os.getcwd())
            with open(model_filename, 'w') as file:
                file.write(model_text)









#-----------------------------------------------
#           * ^ * ^ * ^ * ^ * ^ * ^ *
# ----------- Class Galaxy -------------
#-----------------------------------------------


class Galaxy(object):
    ''' A class for saving a lot of information about single galaxy
        and connected to ImfitModel class
        Input: a dict with parameters 
            or named function arguments with their values
        Example:     Galaxy({'name': 'M31', 'ID': 224})
        Example (2): Galaxy(name='NGC224', fits_name='galaxy.fits')
    '''
    # Default:
    name = None # galaxy name
    ID = None # galaxy unique identificator
    number = None # local galaxy unique identificator
    id_list = [] # list of different galaxy IDs
    ra = None
    dec = None
    commonDir = None # main directory where the galaxy files are keeping
    fits_name = None # name of galaxy fits file
    fits = None # astropy fits object
    data = None # galaxy image array
    wavelength = None # the wavelength of the passband for certain image
    passbandName = None # the passband name of the passband for certain image
    distance = None # distance to the galaxy in parsec
    velocity = None
    redshift = None
    size_arcsec = None
    size_parsec = None
    imfit_config_name = None
    imfit_bestfit_name = None
    imfitModel = None # ImfitModelCombination object
    #
    #--------------------------------------------
    #
    def __init__(self, *args, **kwargs):
        main_dict = {}
        if len(kwargs) > 0:
            main_dict.update(kwargs)
        if len(args) == 1 and isinstance(args[0],dict):
            input_dict = args[0]
            main_dict.update(input_dict)
        else:
            raise ValueError("Galaxy.__init__ takes only a dict"
                             "or named parameters/variables")
        #
        for key, value in main_dict.items():
            setattr(self, key, value)

























