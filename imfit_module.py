# A code with useful functions to work with Imfit

import numpy as np
import astropy

import re

from astropy.io import ascii
from astropy.table import Table, Column

from astropy.io import fits

from pathlib import Path
import shutil
import argparse
import os
import sys
import subprocess
import tempfile
from datetime import datetime

import astropy.units as u


#=======================================================
#=======================================================


#---------------------
with open('imfit_parameters_dictionary.txt') as file:
    lines = [line.strip() for line in file.readlines()]

imfit_models_parameters_dict = {}

for line in lines:
    model = line.split(":")[0].strip()
    parameters_list = line.split(":")[1].strip().split()
    imfit_models_parameters_dict[model] = parameters_list

all_imfit_functions = imfit_models_parameters_dict.keys()

parameters_set = set()
for parameters_list in imfit_models_parameters_dict.values():
    parameters_set |= set(parameters_list)

all_imfit_parameters = list(parameters_set)
all_imfit_parameters.sort()
all_imfit_parameters += ['X0', 'Y0']
#---------------------

def imfit_parameters_list_of(modelName):
    ''' This function returns a list with parameters names which
        is specified exactly for the modelName Imfit 2D-function
        Note that parameter order is important (imfit requires)
    '''
    if modelName not in all_imfit_functions:
        raise ValueError(f"Unknown function for Imfit: {modelName}")
    #
    parameters_list = imfit_models_parameters_dict[modelName]
    return parameters_list


#========================================================


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


def init_imfit_Table():
    ''' Creates an empty astropy table with special Imfit parameters
    Some column names are presented below. All of them have a good-presenting
    name but here is an explanation of some of them:
    'modelName' — the first model in bestfit_imfit_parameters.dat file
    the first because a galaxy may have multiple components
    'modelName': BrokenExponential3D, GaussianRing, etc.
    modelCombFlag — 1 if there are more than one component
    fitting in an image. 0 otherwise
    maskCenterFlag — 1 if the central part of a galaxy is masked
    corresponding a known fact that there is a flat light distribution
    in a galaxy 1D-profile. 0 otherwise
    dateTimePOSIX — POSIX (UNIX) time stamp in seconds (from 01.01.1970)
    '''
    table = Table()
    #
    # technical parameters:
    table.add_columns([
        Column(name='galaxyName', dtype='U20'),
        Column(name='passbandName', dtype='U20'),
        Column(name='wave', dtype='f4'),
        Column(name='wave_unit', dtype='U5'),
        Column(name='wavelength_m', dtype='f4'),
        Column(name='modelName', dtype='U25'),
        Column(name='timePOSIX', dtype='f8'),
        Column(name='dateTime', dtype='U30'),
        Column(name='comment', dtype='U300')
        ])
    # model parameters and their errors, limits and comments after '#':
    for parameter in all_imfit_parameters:
        table.add_columns([
            Column(name=parameter, dtype='f4'),
            Column(name="err_"+parameter, dtype='f4'),
            Column(name="lim_"+parameter, dtype='U20'),
            Column(name="comment_"+parameter, dtype='U100'),
            ])
    table.add_column(Column(name='fit_stat', dtype='f4'))
    table.add_column(Column(name='file_text', dtype='U3000'))
    return table




#----------------------------------------------
# -------- for reading config file ------------
#----------------------------------------------
def read_parameter_line(input_str):
    ''' Returns a dict with keys: 'name', 'value', 'limits', 'error', 'comment'
        A parameter line should be in clear format, not commented
    '''
    parameter_line_dict = {}
    words = input_str.strip().split()
    words = [word.strip() for word in words]
    parameter_line_dict['name'] = words[0]    
    parameter_line_dict['value'] = float(words[1])
    #
    Nwords = len(words)
    #
    limits = None
    error = None
    comment = None
    #
    if Nwords > 2:
        if not words[2].startswith('#'):
            limits = words[2]
            if Nwords > 3:
                if words[3].startswith('#'):
                    comment = ' '.join(words[3:])
                    comment = comment[1:] # without '#' at the beginning
                else:
                    raise ValueError(f"Comment {comment} starts not from '#'")
            #
        else: # words[2].startswith('#') is True — it's a comment or error
            if Nwords > 4 and words[3] == '+/-' and isfloat(words[4]):
                error = words[4]
                if Nwords > 5:
                    comment = ' '.join(words[5:])
            else:
                comment = ' '.join(words[2:])
                comment = comment[1:] # without '#' at the beginning
    #
    parameter_line_dict['limits'] = limits
    parameter_line_dict['error'] = error
    parameter_line_dict['comment'] = comment
    # Returns a dict with keys: 'name', 'value', 'limits', 'error', 'comment'
    return parameter_line_dict


def extended_parameter_dict(parameter_line_dict):
    ''' Returns a dict with special parameter keys
        for specified parameter name.
        For example, parameter_line_dict = 
        {'name': 'h1', 'value': 12.0, 'limits': '10,15', 'error': '0.001',
            'comment': 'abc'}
        — function returns {'h1': 12.0, 'limits_h1': '10,15', 'err_h1': 0.001,
            'comment_h1': 'abc'
    '''
    result_dict = {}
    name = parameter_line_dict['name']
    value = parameter_line_dict['value']
    limits = parameter_line_dict['limits']
    error = parameter_line_dict['error']
    comment = parameter_line_dict['comment']
    #
    result_dict[name] = value
    if limits is not None:
        result_dict['limits_'+name] = limits
    if error is not None:
        result_dict['err_'+name] = error
    if comment is not None:
        result_dict['comment_'+name] = comment
    #
    return result_dict



def read_config(input_file):
    ''' Input: list of lines in the config file 
          or   str filename (config/bestfit)
        which constitute a IMFIT-modelFUNCTION blocks (not only one)
        (X0, Y0, FUNCTION+name and parameters lines — is a block)
        Return: list of tuples where each tuple
        is a combination of dicts (may be only one dict and only one tuple)
        — where a dict is an imfit-model dict: modelName
        and parameters with their values, limits, errors and comments
            Example: [(dict, dict,..), (...), ...]
        dict = {'modelName':'PointSource', 'I_tot':1.05, 'X0':120, 'Y0':121}
        (as an example)
    '''
    if isinstance(input_file, list):
        file_lines = [line.strip() for line in input_file]
    elif isinstance(input_file, str):
        if not '\n' in input_file:
            # string filename
            with open(input_file) as file:
                file_lines = [line.strip() for line in file.readlines()]
        else:
            # file content (multiline text)
            file_lines = input_file.split('\n')
    #
    file_text = ';'.join(file_lines)
    clear_lines = [line for line in file_lines
                   if line != '' and not line.startswith('#')]
    #
    result_models_list = [] # will be an output of the read_config function
    #
    current_block = [] # will be converted to tuple
    current_model = {} # imfit model dict
    #
    lines = iter(clear_lines)
    #
    line = next(lines)
    while lines.__length_hint__() > 0:
        X0_dict = read_parameter_line(line)
        X0_data = extended_parameter_dict(X0_dict)
        #
        Y0_dict = read_parameter_line(next(lines))
        Y0_data = extended_parameter_dict(Y0_dict)
        #
        line = next(lines)
        while not line.startswith('X0'):
            if line.startswith('FUNCTION'):
                modelName = re.search(r"FUNCTION\s+(\w+)\b", line)[1]
                current_model['modelName'] = modelName
            else:
                raise ValueError(f"Expected FUNCTION line but got '{line}'")
            #
            parameters_list = imfit_parameters_list_of(modelName)
            for parameterName in parameters_list:
                next_parameter = read_parameter_line(next(lines))
                if next_parameter['name'] != parameterName:
                    raise ValueError(f"The {next_parameter['name']} parameter"
                                      " is in the wrong position")
                current_model.update(extended_parameter_dict(next_parameter))
            #
            current_model.update(X0_data)
            current_model.update(Y0_data)
            current_model['file_text'] = file_text
            #
            current_block.append(current_model.copy())
            current_model.clear()
            if lines.__length_hint__() > 0:
                line = next(lines)
            else:
                break
        #
        result_models_list.append(tuple(current_block))
        current_block.clear()
    #
    return result_models_list




#-----------------------------------------------
# ------- for writing config file --------------
#-----------------------------------------------
def zero_filled_model_dict(modelName):
    ''' Input: str imfit model name
        Return: dict with zero-filled values for each parameter
        (it helps to create variable model_dict faster)
    '''
    result_dict = dict(X0=0.0, Y0=0.0)
    result_dict['modelName'] = modelName
    parameters_list = imfit_parameters_list_of(modelName)
    result_dict.update({name: 0.0 for name in parameters_list})
    return result_dict



def str_parameter_line(name, value=None, limits=None, error=None, comment=None):
    ''' From a given name, parameter value and limits
        or comment returns a config line in a proper format
    '''
    name_part = name.ljust(12)
    if value is not None:
        value_part = str(value).ljust(15)
    else:
        value_part = ''.ljust(15)
    #
    if limits is None or ''.join(limits).strip() == '':
        limits_part = ''
    else:
        # check limits format:
        error_message = f"Error in {name} limits '{limits}'"
        #
        if isinstance(limits,tuple) or isinstance(limits,list):
            if not isfloat(limits[0]) or not isfloat(limits[1]):
                raise ValueError(error_message)
            limit_part = ','.join(list(map(str,limits)))
            #
        elif isinstance(limits,str):
            if limits.strip() == 'fixed':
                limits_part = 'fixed'
                #
            else:
                limits = limits.split(',')
                if not isfloat(limits[0]) or not isfloat(limits[1]):
                    raise ValueError(error_message)
                limits_part = ','.join(list(map(str,limits)))
    #
    line = name_part + value_part + limits_part
    if error is not None:
        line += ' # +/-' + str(error)
    if comment is not None:
        line += ' # ' + comment
    return line



def extract_coords_from_block(models_block):
    ''' Input: list|tuple of multiple imfit model dicts
        or list|tuple with just only one model dict
        The function will check that all models in the models block
            have the same 'X0' and 'Y0' data. 
            If not, then it raises ValueError
        Return: dict with keys which contain 'X0' and 'Y0' data
    '''
    result_dict = {}
    for parameter in ['X0', 'Y0']:
        for prefix in ['', 'err_', 'limits_', 'comment_']:
            coord_key = f"{prefix}{parameter}"
            all_coord_values = [dict_.get(coord_key) for dict_ in models_block
                          if dict_.get(coord_key) is not None]
            all_coord_values = set(all_coord_values)
            if len(all_coord_values) > 1:
                raise ValueError(f"Different X0 or Y0 data"
                                 f" in the block: \n {models_block}")
            elif len(all_coord_values) == 1:
                value = all_coord_values[0]
                result_dict[key] = value
            #
    return result_dict




def write_config_file(config_filename, input_models_list, input_comment=None):
    ''' Creates a config file for Imfit based on the data in input_models_list
        Input: list of tuples of dicts
        Input Example: [({'X0':10},{'X0':10}), ({'X0':15},)]
            input comment — if present, will be written
            in the head position of the config file
        Return: [nothing] — creating file with specified config_filename
        Note: config_filename may be None — then the config file text
            will be returned as simple text
    '''
    file_content_lines = [] # the entire config file content will be here
    file_header = ('# This config was created by a Python program\n\n')
    file_content_lines.append(file_header)
    if input_comment is not None:
        file_content_lines.append(f"# {input_comment}\n\n")
    #
    for current_block in input_models_list:
        coords_dict = extract_coords_from_block(current_block)
        for parameter in ['X0', 'Y0']:
            coord_line = str_parameter_line(
                            name=parameter, value=coords_dict[parameter],
                            limits=coords_dict.get('limits_'+parameter),
                            error=coords_dict.get('err_'+parameter),
                            comment=coords_dict.get('comment_'+parameter)
                            )
            file_content_lines.append(coord_line)
        #
        for current_model in current_block:
            modelName = current_model['modelName']
            file_content_lines.append(f"FUNCTION {modelName}")
            #
            parameters_list = imfit_parameters_list_of(modelName)
            for parameter in parameters_list:
                parameter_line = str_parameter_line(
                            name=parameter, value=current_model[parameter],
                            limits=current_model.get('limits_'+parameter),
                            error=current_model.get('err_'+parameter),
                            comment=current_model.get('comment_'+parameter)
                            )
                file_content_lines.append(parameter_line)
        # add an empty line between blocks:
        file_content_lines.append('')
    #
    file_content = '\n'.join(file_content_lines)
    if config_filename is not None:
        with open(config_filename, 'w') as file:
            file.write(file_content)
    else:
        return file_content



def makeimage(config, output='modelimage.fits', verbosity=False, **kwargs):
    ''' Just run imfit-makeimage and save output file with model image
        or return an array if output is None
    '''
    if '\n' in config:
        # then the input is just file text content, not filename
        fd, tmp_filename = tempfile.mkstemp()
        with os.fdopen(fd, 'w') as file:
            file.write(config)
        config_filename = tmp_filename
        config_is_tmp = True
    else:
        config_filename = config
        config_is_tmp = False
    #
    command = f"makeimage {config_filename}"
    attributes = kwargs.copy()
    #
    if output is None:
        fd, tmp_filename = tempfile.mkstemp()
        output = tmp_filename
        output_is_tmp = True
    else:
        output_is_tmp = False
    #
    command += f" --output={output}"
    #
    refimage = attributes.pop('refimage', None)
    nrows = attributes.pop('nrows', None)
    ncols = attributes.pop('ncols', None)
    #
    if refimage is not None:
        if isinstance(refimage, str):
            # it's a filename of refimage
            command += f" --refimage={refimage}"
        else:
            # it's a numpy array with data
            nrows, ncols = refimage.shape
            command += f" --nrows={nrows} --ncols={ncols}"
            #
    elif ncols is not None and nrows is not None:
        command += f" --nrows={int(nrows)} --ncols={int(ncols)}"
    else:
        raise ValueError("refimage or nrows and ncols are not specified")
    #
    for parameter, value in attributes.items():
        command += f" --{parameter}={value}"
    #
    if verbosity is True:
        makeimage_output = None  # std output of subprocess.run()
    else:
        makeimage_output = subprocess.PIPE
    #
    run_result = subprocess.run(command, shell=True, stdout=makeimage_output,
                                  stderr=makeimage_output, encoding='utf-8')
    if run_result.returncode != 0:
        if verbosity is not True:
            print('ERROR', '-'*50,
                run_result.stdout, run_result.stderr, sep='\n')
        raise ValueError("makeimage ERROR")
    #
    if output_is_tmp:
        data = fits.getdata(output)
        os.remove(output)
        return data
    else:
        with open(config_filename) as file:
            config_lines = [line.strip() for line in file.readlines()]
        #
        clear_lines = [line for line in config_lines
                        if line != '' and not line.startswith('#')]
        with fits.open(output, mode='update') as hdul:
            header = hdul[0].header
            header.add_history('This image was made by makeimage from:')
            for line in clear_lines:
                header.add_history(line)
            hdul.flush()
    #
    if config_is_tmp:
        os.remove(config_filename)





















