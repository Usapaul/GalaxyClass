
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

from FitsImage_class import FitsImage
from geometry import Point2D, Vector2D

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
# -------------- Class MaskImage ---------------
#-----------------------------------------------



class MaskImage(FitsImage):
    ''' Mask for astronomical image, True and False values in each pixel
    '''
    # mask_value is 1, and not masked pixels have value 0
    # type — unsigned integer (numpy.uint8) as the most appropriate
    #
    def __init__(self, input_mask):
        fitsImage = FitsImage(input_mask)
        fitsImage.data = np.where(fitsImage.data != 0, 1, 0)
        fitsImage.convert_to(np.uint8)
        return fitsImage

    # - - - - - - - - - - - - - - - - - - - - - - -

    def invert(self):
        self.data = np.where(self.data == 1, 0, 1)

    def add_box(self, x, y, w, h):
        if w%2 == 1:
            w += 1
        if h%2 == 1:
            h += 1
        self.data[y-h/2:y+h/2,x-w/2:x+w/2]

    def add_circle(self, x_center, y_center, radius):
        x_center = int(np.floor(x_center))
        y_center = int(np.floor(y_center))
        r = int(np.ceil(radius))
        radial_dist = lambda x, y : np.sqrt((x-x_center)**2 + (y-y_center)**2)
        for x in range(x-r, x+r+1):
            y_abs = np.sqrt(radius**2 - x**2)
            self.data[y-y_abs:y+y_abs+1,x] = 1

    def add_ellipse(self, x_center, y_center, ellA, ellB, theta):
        x_center = int(np.floor(x_center))
        y_center = int(np.floor(y_center))
        a = int(np.ceil(a))
        b = int(np.ceil(b))
        if pa == 0:
            for x in range(x_center-a, x_center+a+1):
                y_abs = b * np.sqrt(1 - x**2/a**2)
                self.data[y-y_abs:y+y_abs+1,x] = 1
        else:




    # - - - - - - - - - - - - - - - - - - - - - - -
    
    def as_type(self, input_type):
        return self.data.astype(input_type)

    @property
    def as_boolean(self):
        return self.data.astype(bool)

    @property
    def fraction(self):
        return np.count_nonzero(self.data) / self.data.size


