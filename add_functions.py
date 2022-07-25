
''' Additional helpful functions are here in this module
'''

import numpy as np
from astropy.io import fits


def wave_unit_ratio_to_nm(input_str):
    ''' Returns a coefficient for which the wavelength 
        with different metric unit should be multiplied. 
        Examples: 
        wave_unit_ratio_to_nm('m') = 1e9; 
        wave_unit_ratio_to_nm('mcm') = 1000; 
        wave_unit_ratio_to_nm('A') = 0.1
    '''
    metre_list = ['m', 'metre', 'metr', 'meter', '']
    angstrem_list = ['Å', 'a', 'an', 'ang', 'angstrem']
    prefixes_dict = {}
    prefixes_dict['nano'] = ['n', 'nano', 'nan']
    prefixes_dict['micro'] = ['mc', 'mk', 'micro', 'micron', 'mu', '𝜇']
    prefixes_dict['mili'] = ['m', 'milli', 'mili']
    prefixes_dict['centi'] = ['c', 'cen', 'centi']
    prefixes_dict['kilo'] = ['k', 'kilo']
    metrics_dict = {metric: [prefix + suffix 
                          for prefix in prefixes_dict[metric]
                          for suffix in metre_list]
                    for metric in prefixes_dict.keys()}
    metrics_dict['metre'] = metre_list
    metrics_dict['angstrem'] = angstrem_list
    coeff_dict = {}
    coeff_dict['angstrem'] = 0.1
    coeff_dict['nano'] = 1
    coeff_dict['micro'] = 1e3
    coeff_dict['mili'] = 1e6
    coeff_dict['centi'] = 1e7
    coeff_dict['metre'] = 1e9
    coeff_dict['kilo'] = 1e12
    #
    input_metric = input_str.lower().replace(' ','')
    for metric in metrics_dict.keys():
        if input_metric in metrics_dict[metric]:
            coeff = coeff_dict[metric]
            return coeff
    else:
        return None




def unit_ratio_to_arcsec(input_str):
    ''' Returns a coefficient for which the pixelscale 
        with different from arcsec/pixel unit should be multiplied. 
        Examples: 
        unit_ratio_to_arcsec('deg') = 3600; 
        unit_ratio_to_arcsec('minutes') = 60; 
        unit_ratio_to_arcsec('mas') = 0.001
    '''
    mas_list = ['mas', 'miliarcsec', 'ms', 'milisec', 'milis']
    arcsec_list = ['arcsec', 's', 'sec', 'secs', 'second', 'seconds', '"']
    arcmin_list = ['arcmin', 'm', 'min', 'mins', 'minute', 'minutes' '\'']
    degree_list = ['d', 'deg', 'degs', 'degree', 'degrees', '°']
    radian_list = ['r', 'rad', 'rads', 'radian', 'radians']
    #
    names_dict = {}
    names_dict['mas'] = (0.001, mas_list)
    names_dict['arcsec'] = (1, arcsec_list)
    names_dict['arcmin'] = (60, arcmin_list)
    names_dict['degree'] = (3600, degree_list)
    names_dict['radian'] = (206264.8, radian_list)
    #
    input_scale = input_str.lower().replace(' ','')
    for item in names_dict.values():
        coeff, names_list = item
        if input_scale in names_list:
            return coeff
    else:
        return None



