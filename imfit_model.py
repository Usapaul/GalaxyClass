
import imfit_module

import numpy as np


#-----------------------------------------------


config_examples_file = 'config_examples.dat'


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
#           * ^ * ^ * ^ * ^ * ^ * ^ *
# ----------- Class ImfitParameter -------------
#-----------------------------------------------


class ImfitParameter(object):
    ''' The only one parameter of an imfit model 
        as a class with limits, comment and error data
        attributes:
            name, value, limits, error, comment
        also: n, v, lim, err, left_lim, right_lim    
    '''
    # Default:
    _name = None
    _value = None
    _limits = None
    _error = None
    _comment = None

    def __init__(self, name=None,
                 value=None, limits=None, comment=None, error=None,
                 **kwargs):
        if name is None:
            # then there is something like 'PA=90.0' in kwargs
            # and also something like 'err_PA=0.1', 'limits_PA=0,90'
            name_filter = filter(lambda x: not any(x.startswith(prefix)
                                for prefix in ['limits_', 'err_', 'comment_']),
                                kwargs.keys())
            name_list = list(name_filter)
            if len(name_list) > 1:
                raise ValueError("Multiple keys for ImfitParameter"
                                 " which assume to be a parameter name:"
                                 f" {name_list}")
            name = name_list[0]
            value = kwargs[name]
            limits = kwargs.get('limits_' + name)
            error = kwargs.get('err_' + name)
            comment = kwargs.get('comment_' + name)
        #
        self._name = name
        self._value = value
        self._limits = limits
        self._error = error
        self._comment = comment


    @classmethod
    def fromtext(cls, input_str):
        ''' Initialize ImfitParameter from str parameter line'''
        parameter_line_dict = imfit_module.read_parameter_line(input_str)
        return cls(**parameter_line_dict)


    @classmethod
    def from_dict(cls, input_dict):
        parameter_dict = {}
        if 'name' in input_dict.keys():
            parameter_dict['name'] = input_dict['name']
            for key in ['value', 'limits', 'error', 'comment']:
                parameter_dict[key] = input_dict.get(key)
        else:
            for key in input_dict.keys():
                if all(not key.startswith(prefix)
                        for prefix in ['err_', 'limits_', 'comment_']):
                    parameterName = key
                    break
            if parameterName not in imfit_module.all_imfit_parameters:
                raise ValueError(f"Unknown imfit parameter: {parameterName}")
            #
            parameter_dict['name'] = parameterName
            parameter_dict['value'] = input_dict[parameterName]
            # ------
            limits = input_dict.get('limits_'+parameterName)
            error = input_dict.get('err_'+parameterName)
            comment = input_dict.get('comment_'+parameterName)
            #
            parameter_dict['limits'] = limits
            parameter_dict['error'] = error
            parameter_dict['comment'] = comment
        #
        return cls(**parameter_dict)

    @classmethod
    def zero(cls, parameterName):
        ''' Input: imfit parameter name
            Return: ImfitParameter class object with
            value = 0.0 and no limits, error or comment
        '''
        return cls(name=parameterName, value=0.0)

    @classmethod
    def zero_fixed(cls, parameterName):
        return cls(name=parameterName, value=0.0, limits='fixed')


    @classmethod
    def example(cls, parameterName='X0'):
        return cls(name=parameterName, value=0.58, limits='0.0,1.0',
                    comment='example')


    # - - - - - - - - - - - - - - - - - - - - - - -
    #           ImfitParameter str and repr

    def __str__(self):
        name_str = self.name if self.name is not None else ''
        parameterString = imfit_module.str_parameter_line(
            name=name_str, value=self.value, limits=self.limits,
            error=self.error, comment=self.comment)
        return parameterString


    def __repr__(self):
        name_str = self.name if self.name is not None else '(unnamed)'
        output_str = f"{name_str}:"
        if self.value is not None:
            output_str += f" {self.value}"
        if self.limits is not None:
            output_str += f" [{self.limits}]"
        if self.error is not None:
            output_str += f" (+/- {self.error})"
        if self.comment is not None:
            output_str += f" # {self.comment}"
        parameterString = (f"[ImfitParameter] {output_str}")
        return parameterString

    # - - - - - - - - - - - - - - - - - - - - - - -

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        if name is None:
            pass
        elif name in imfit_module.all_imfit_parameters:
            pass
        else:
            raise ValueError(f"Input {name} for setting name"
                             f" is not correct imfit parameter name")
        self._name = name

    @name.deleter
    def name(self):
        self._name = None

    # - - - - - - - - - - - - - - - - - - - - - - -

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if self.name is None and value is not None:
            raise ValueError("Parameter should have name before setting value")
        if value is None:
            pass
        elif isfloat(value):
            value = float(value)
        else:
            raise ValueError(f"Input {value} is NaN")
        #
        self._value = value

    @value.deleter
    def value(self):
        self._value = None

    # - - - - - - - - - - - - - - - - - - - - - - -
    #             set limits (not simple)

    @property
    def limits(self):
        return self._limits

    @limits.setter
    def limits(self, limits):
        ''' limits is string representing two numbers separated by comma
            Example: '10.4,10.5' '''
        if self.name is None and limits is not None:
            raise ValueError("Parameter should have name"
                             " before setting limits")
        if self.value is None and limits is not None:
            raise ValueError("Parameter should have value"
                             " before setting limits")
        # ----------------
        if limits is None:
            left_lim = None
            right_lim = None
        elif isinstance(limits,tuple) or isinstance(limits,list):
            left_lim, right_lim = limits
            if isfloat(left_lim) and isfloat(right_lim):
                left_lim = float(left_lim)
                right_lim = float(right_lim)
                limits = f"{left_lim},{right_lim}"
            else:
                raise ValueError(f"NaN in limits list: '{limits}'")
            #
        elif isinstance(limits,str):
            limits = limits.strip()
            if limits == 'fixed':
                left_lim = None
                right_lim = None
            else:
                left_lim, right_lim = limits.split(',')
                if isfloat(left_lim) and isfloat(right_lim):
                    left_lim = float(left_lim)
                    right_lim = float(right_lim)
                    limits = f"{left_lim},{right_lim}"
                else:
                    raise ValueError(f"NaN in limits: '{limits}'")
        else:
            raise ValueError(f"Unknown type of input limits {limits}")
        #
        if limits is not None and limits != 'fixed':
            if right_lim < left_lim:
                raise ValueError(f"Right limit {right_lim} less"
                                 f" than left limit {left_lim}")
            if self.value < left_lim or self.value > right_lim:
                raise ValueError(f"Value {self.value} is not in"
                                 f" limits range {limits}")
        #
        self._limits = limits

    @limits.deleter
    def limits(self):
        self._limits = None


    def set_fixed(self):
        self._limits = 'fixed'

    def set_free(self):
        self._limits = None

    # - - - - - - - - - - - - - - - - - - - - - - -

    @property
    def error(self):
        return self._error

    @error.setter
    def error(self, error):
        if self.name is None and error is not None:
            raise ValueError("Parameter should have name before setting error")
        if self.value is None and error is not None:
            raise ValueError("Parameter should have value"
                             " before setting error")        
        if error is None:
            pass
        elif isfloat(error):
            error = float(error)
        else:
            raise ValueError(f"Input {error} for setting error is NaN")
        #
        self._error = error

    @error.deleter
    def error(self):
        self._error = None

    # - - - - - - - - - - - - - - - - - - - - - - -

    @property
    def comment(self):
        return self._comment

    @comment.setter
    def comment(self, comment):
        if self.name is None and comment is not None:
            raise ValueError("Parameter should have name before .comment")
        if comment is not None:
            comment = str(comment).strip()
            self._comment = comment
        else:
            self._comment = None

    @comment.deleter
    def comment(self, comment):
        self._comment = None

    # - - - - - - - - - - - - - - - - - - - - - - -
    #  ImfitParameter copy: based on copy method of dict objects

    def __copy__(self):
        return self.__class__.from_dict(self.as_dict.copy())

    def copy(self):
        return self.__copy__()

    # - - - - - - - - - - - - - - - - - - - - - - -
    #           short attributes names

    @property
    def n(self):
        return self.name

    @property
    def v(self):
        return self.value

    @property
    def lim(self):
        return self.limits

    @property
    def limits_tuple(self):
        return tuple(map(float,self.limits.split(',')))

    @property
    def left_lim(self):
        return float(self.limits.split(',')[0])

    @property
    def right_lim(self):
        return float(self.limits.split(',')[1])

    @property
    def err(self):
        return self.error

    # - - - - - - - - - - - - - - - - - - - - - - -
    #                checking block

    @property
    def is_fixed(self):
        if self.limits == 'fixed':
            return True
        else:
            return False

    # - - - - - - - - - - - - - - - - - - - - - - -
    #     multiple types of presenting parameter

    @property
    def as_string(self):
        return self.__str__()

    @property
    def as_dict(self):
        parameter_line_dict = {
            'name':self.name,
            'value': self.value,
            'limits': self.limits,
            'error': self.error,
            'comment': self.comment
            }
        return parameter_line_dict

    @property
    def as_extended_dict(self):
        extended_parameter_dict = {
            self.name: self.value,
            'limits_' + self.name: self.limits,
            'err_' + self.name: self.error,
            'comment_' + self.name: self.comment
            }
        return extended_parameter_dict



#-----------------------------------------------
#           * ^ * ^ * ^ * ^ * ^ * ^ *
# ------------- Class ImfitModel ---------------
#-----------------------------------------------



class ImfitModel(object):
    ''' The full imfit model (one 2D-function)
        object with model name and parameters
        It contains X0 and Y0 parameters as separated ones
        and model parameters (as ImfitParameter objects)
        which are setted by the modelName ('Gauss', 'Exponential', ..)
    '''
    X0 = None # as ImfitParameter or None
    Y0 = None # as ImfitParameter or None
    modelName = None
    model_parameters_list = None # only model, not X0, Y0
    
    def __init__(self, modelName=None, **kwargs):
        if modelName is None:
            raise ValueError("ImfitModel must have modelName parameter")
        self.modelName = modelName
        #
        model_parameters_list = imfit_module.imfit_parameters_list_of(modelName)
        parameters_list_and_X0Y0 = ['X0', 'Y0'] + model_parameters_list
        #
        if not all(name in kwargs.keys() for name in model_parameters_list):
            raise ValueError("ImfitModel got not enough parameters")
        else:
            self.model_parameters_list = model_parameters_list
        #
        if 'X0' in kwargs.keys():
            coord_dict = {key: value for key, value in kwargs.items()
                          if 'X0' in key}
            self.X0 = ImfitParameter(**coord_dict)
        if 'Y0' in kwargs.keys():
            coord_dict = {key: value for key, value in kwargs.items()
                          if 'Y0' in key}
            self.Y0 = ImfitParameter(**coord_dict)
        if self.X0 is None and self.Y0 is not None:
            print('Warning: ImfitModel got X0 but no Y0')
        if self.Y0 is None and self.X0 is not None:
            print('Warning: ImfitModel got Y0 but no X0')
        #
        for parameterName in model_parameters_list:
            parameter_dict = {}
            parameter_dict['name'] = parameterName
            parameter_dict['value'] = kwargs[parameterName]
            #
            available_parameter_keys = [prefix + parameterName
                        for prefix in ['err_', 'limits_', 'comment_']]
            for key in available_parameter_keys:
                parameter_dict[key] = kwargs.get(key)
            #
            setattr(self, parameterName, ImfitParameter(**parameter_dict))


    @classmethod
    def from_dict(cls, model_dict):
        return cls(**model_dict)


    @classmethod
    def fromfile(cls, input_file):
        # read the first model from the input_file (may be just text)
        models_list = imfit_module.read_config(input_file)
        first_block = models_list[0]
        first_model = first_block[0]
        model_dict = first_model # it's dict (see imfit_module.read_config)
        if len(models_list) > 1 or len(first_block) > 1:
            print('Warning: ImfitModel read from file with multiple models')
        return cls(**model_dict)


    @classmethod
    def zero_fixed(cls, modelName):
        ''' Input: imfit FUNCTION model name
            Return: ImfitModel class object where for every parameter:
            value = 0.0, limits = 'fixed'
        '''
        parameters_list = imfit_module.imfit_parameters_list_of(modelName)
        parameters_list_and_X0Y0 = ['X0', 'Y0'] + parameters_list
        #
        parameters_dict = {}
        for parameterName in parameters_list_and_X0Y0:
            zero_parameter = ImfitParameter.zero_fixed(parameterName)
            zero_parameter_dict = zero_parameter.as_extended_dict
            parameters_dict.update(zero_parameter_dict)
        #
        parameters_dict['modelName'] = modelName
        #
        return cls.from_dict(parameters_dict)

    @classmethod
    def example(cls):
        ''' Returns example ImfitModel (taken from special file)'''
        models_list = imfit_module.read_config(config_examples_file)
        randint = np.random.randint(1,len(models_list))
        model_dict = models_list[randint][0] # =tuple_object[0]
        return cls(**model_dict)


    @property
    def parameters_list(self):
        ''' list of model parameters and X0, Y0 '''
        parameters_list_and_X0Y0 = ['X0', 'Y0'] + self.model_parameters_list
        return parameters_list_and_X0Y0


    def write_to_file(self, filename=None):
        models_list = [(self.as_dict,),]
        imfit_module.write_config_file(filename, models_list)


    @property
    def x0(self):
        if self.X0 is None:
            return None
        else:
            return self.X0.value

    @property
    def y0(self):
        if self.Y0 is None:
            return None
        else:
            return self.Y0.value

    @property
    def x0y0(self):
        return (self.x0, self.y0)


    def makeimage(self, output='modelimage.fits', verbosity=False, **kwargs):
        attrs = kwargs.copy()
        if (attrs.get('refimage') is None and
            (attrs.get('nrows') is None or attrs.get('ncols') is None)):
            # refimage is None; assuming X0 and Y0 as a center of image
            x0 = self.x0
            y0 = self.y0
            ncols, nrows = map(lambda x: np.ceil(x) * 2, (x0, y0))
            attrs['nrows'] = nrows
            attrs['ncols'] = ncols
        #
        if output is not None:
            imfit_module.makeimage(self.as_file_text, output=output,
                                    verbosity=verbosity, **attrs)
        else:
            # no image will be saved, just return an array
            data = imfit_module.makeimage(self.as_file_text, output=None,
                                             verbosity=verbosity, **attrs)
            return data


    # - - - - - - - - - - - - - - - - - - - - - - -

    def __str__(self):
        modelName_str = self.modelName
        parameters_str_parts = []
        for parameterName in self.parameters_list:
            parameter_class_item = getattr(self, parameterName)
            if parameter_class_item is None:
                # in case of X0 or Y0 is None
                continue
            value = parameter_class_item.value
            fix = 'F' if parameter_class_item.is_fixed else ''
            parameter_short_string = f"{parameterName}={value}{fix}"
            parameters_str_parts.append(parameter_short_string)
        #
        output_str = (f"{modelName_str}({','.join(parameters_str_parts)})")
        return output_str


    def __repr__(self):
        return '\n'.join(self.as_config_lines)

    # - - - - - - - - - - - - - - - - - - - - - - -

    def set_fixed(self):
        for parameterName in self.parameters_list:
            parameter_class_item = getattr(self, parameterName)
            if parameter_class_item is not None:
                parameter_class_item.set_fixed()

    def set_free(self):
        for parameterName in self.parameters_list:
            parameter_class_item = getattr(self, parameterName)
            if parameter_class_item is not None:
                parameter_class_item.set_free()

    def del_comments(self):
        for parameterName in self.parameters_list:
            parameter_class_item = getattr(self, parameterName)
            if parameter_class_item is not None:
                del parameter_class_item.comment
                del parameter_class_item.error


    # - - - - - - - - - - - - - - - - - - - - - - -
    #  ImfitModel copy: based on copy method of dict objects

    def __copy__(self):
        return self.__class__.from_dict(self.as_dict.copy())

    def copy(self):
        return self.__copy__()

    # - - - - - - - - - - - - - - - - - - - - - - -
    #           short attributes names

    @property
    def name(self):
        return self.modelName

    @property
    def list(self):
        return self.parameters_list

    @property
    def model_parameters(self):
        ''' Return a parameters_list without 'X0' and 'Y0' '''
        return self.model_parameters_list

    # - - - - - - - - - - - - - - - - - - - - - - -
    #  different types of presenting an ImfitModel object

    @property
    def short_model_notation(self):
        modelName = self.modelName
        parameters = ','.join(self.model_parameters_list)
        return f"{modelName}({parameters})"

    @property
    def short_notation(self):
        x0 = self.x0
        y0 = self.y0
        return f"{x0}, {y0} {self.short_model_notation}"

    @property
    def model_notation(self):
        modelName = self.modelName
        parameters_strings_list = []
        for name in self.model_parameters_list:
            imfitParameter = getattr(self, name)
            value = imfitParameter.value
            fix = 'F' if imfitParameter.is_fixed else ''
            parameters_strings_list.append(f'{name}[{value}{fix}]')
        #
        return f"{modelName}: {' '.join(parameters_strings_list)}"

    @property
    def notation(self):
        x0 = self.x0
        y0 = self.y0
        return f"X0={x0} Y0={y0} {self.model_notation}"

    @property
    def as_dict(self):
        model_dict = {}
        model_dict['modelName'] = self.modelName
        for parameterName in self.parameters_list:
            imfitParameter = getattr(self, parameterName)
            if imfitParameter is not None:
                extended_parameter_dict = imfitParameter.as_extended_dict
                model_dict.update(extended_parameter_dict)
        #
        return model_dict

    @property
    def as_simple(self):
        return self.as_dict

    @property
    def as_dict_noX0Y0(self):
        model_dict = self.as_dict.copy()
        model_dict_keys = list(model_dict.keys())
        for key in model_dict_keys:
            if any(coord in key for coord in ['X0', 'Y0']):
                model_dict.pop(key)
        return model_dict

    @property
    def as_file_text(self):
        models_list = [(self.as_dict,),]
        file_text = imfit_module.write_config_file(None, models_list)
        return file_text

    @property
    def as_config_lines(self):
        text_lines = self.as_file_text.split('\n')
        clear_lines = [line.strip() for line in text_lines
                       if not (line.startswith('#') or line.strip() == '')]        
        return clear_lines

    @property
    def as_ImfitModelBlock(self):
        return ImfitModelBlock((self,))

    @property
    def as_block(self):
        return self.as_ImfitModelBlock

    @property
    def as_ImfitModelCombination(self):
        return ImfitModelCombination([(self,),])

    @property
    def as_combination(self):
        return self.as_ImfitModelCombination

    @property
    def as_models_list(self):
        return [(self.as_dict,),]





#-----------------------------------------------
#           * ^ * ^ * ^ * ^ * ^ * ^ *
# ----------- Class ImfitModelBlock ------------
#-----------------------------------------------


class ImfitModelBlock(object):
    ''' A block with multiple imfit functions 
        which have the same (common) X0 and Y0 coordinates
    '''
    models_tuple = None # as tuple of ImfitModel objects
    _X0 = ImfitParameter.zero('X0') # as ImfitParameter with...
    _Y0 = ImfitParameter.zero('Y0') # ...value=0.0 — as default

    def __init__(self, *args):
        ''' Input: ImfitModel class objects (= args)
            Returned class object — a tuple of ImfitModelBlock objects
        '''
        models_block_list = []
        if len(args) == 1 and isinstance(args[0], tuple):
            args = tuple([arg for arg in args[0]])
        #
        for arg in args:
            if isinstance(arg, ImfitModel):
                models_block_list.append(arg)
            elif isinstance(arg, dict):
                models_block_list.append(ImfitModel.from_dict(arg))
            else:
                raise ValueError("ImfitModelBlock input is not tuple,"
                                 " ImfitModel or model parameters dict")
        #
        models_block = tuple(models_block_list)
        models_dicts = tuple([model.as_dict for model in models_block])
        coords_dict = imfit_module.extract_coords_from_block(models_dicts)
        #
        self.models_tuple = models_block
        X0_dict = {key: value 
                   for key, value in coords_dict.items()
                   if 'X0' in key}
        Y0_dict = {key: value 
                   for key, value in coords_dict.items()
                   if 'Y0' in key}
        if len(X0_dict) > 0:
            self._X0 = ImfitParameter(**X0_dict)
        if len(Y0_dict) > 0:
            self._Y0 = ImfitParameter(**Y0_dict)


    @classmethod
    def example(cls, size=2):
        ''' Returns example ImfitModelBlock (taken from special file)'''
        first_model_list = [ImfitModel.example().as_dict]
        other_models_list = [ImfitModel.example().as_dict_noX0Y0 
                            for i in range(1,size)]
        models_block = first_model_list + other_models_list
        coords_dict = imfit_module.extract_coords_from_block(first_model_list)
        for model_dict in models_block:
            model_dict.update(coords_dict)
        #
        return cls(*models_block)


    @classmethod
    def fromfile(cls, input_file):
        # read the first block from the input_file (may be just text)
        models_list = imfit_module.read_config(input_file)
        first_block = models_list[0]
        models_block = first_block
        if len(models_list) > 1:
            print('Warning: ImfitModelBlock is given from file'
                  ' with multiple blocks')
        return cls(*models_block)

    # - - - - - - - - - - - - - - - - - - - - - -

    @property
    def X0(self):
        return self._X0

    @X0.setter
    def X0(self, input_X0):
        if isinstance(input_X0, ImfitParameter):
            X0_parameter = input_X0
        elif isinstance(input_X0, int) or isinstance(input_X0, float):
            X0_parameter = ImfitParameter(name='X0', value=input_X0)
        elif isinstance(input_X0, dict):
            X0_parameter = ImfitParameter.from_dict(input_X0)
        elif isinstance(input_X0, str):
            X0_parameter = ImfitParameter.fromtext(input_X0)
        else:
            raise ValueError(f"Input {input_X0} for setting X0"
                             f" is not correct")
        self._X0 = X0_parameter
        for model in self.models_tuple:
            model.X0 = X0_parameter

    @X0.deleter
    def X0(self):
        zero_X0_parameter = ImfitParameter.zero('X0')
        self._X0 = zero_X0_parameter
        for model in self.models_tuple:
            model.X0 = zero_X0_parameter


    @property
    def Y0(self):
        return self._Y0

    @Y0.setter
    def Y0(self, input_Y0):
        if isinstance(input_Y0, ImfitParameter):
            Y0_parameter = input_Y0
        elif isinstance(input_Y0, int) or isinstance(input_Y0, float):
            Y0_parameter = ImfitParameter(name='Y0', value=input_Y0)
        elif isinstance(input_Y0, dict):
            Y0_parameter = ImfitParameter.from_dict(input_Y0)
        elif isinstance(input_Y0, str):
            Y0_parameter = ImfitParameter.fromtext(input_Y0)
        else:
            raise ValueError(f"Input {input_Y0} for setting Y0"
                             f" is not correct")
        self._Y0 = Y0_parameter
        for model in self.models_tuple:
            model.Y0 = Y0_parameter

    @Y0.deleter
    def Y0(self):
        zero_Y0_parameter = ImfitParameter.zero('Y0')
        self._Y0 = zero_Y0_parameter
        for model in self.models_tuple:
            model.Y0 = zero_Y0_parameter

    # - - - - - - - - - - - - - - - - - - - - - -

    def write_to_file(self, filename):
        models_list = [self.as_tuple_of_dicts,]
        imfit_module.write_config_file(filename, models_list)


    def makeimage(self, output='modelimage.fits', verbosity=False, **kwargs):
        attrs = kwargs.copy()
        if (attrs.get('refimage') is None and
            (attrs.get('nrows') is None or attrs.get('ncols') is None)):
            # refimage is None; assuming X0 and Y0 as a center of image
            ncols, nrows = map(lambda x: np.ceil(x) * 2, self.x0y0)
            attrs['nrows'] = nrows
            attrs['ncols'] = ncols
        #
        if output is not None:
            imfit_module.makeimage(self.as_file_text, output=output,
                                    verbosity=verbosity, **attrs)
        else:
            # no image will be saved, just return an array
            data = imfit_module.makeimage(self.as_file_text, output=None,
                                             verbosity=verbosity, **attrs)
            return data


    # - - - - - - - - - - - - - - - - - - - - - - -

    def __str__(self):
        coords_str = f"{self.x0}, {self.y0}"
        models_notes = [model.short_model_notation
                        for model in self.models_tuple]
        models_str = ' '.join(models_notes)
        return f"{coords_str} {models_str}"


    def __repr__(self):
        return '\n'.join(self.as_config_lines)

    # - - - - - - - - - - - - - - - - - - - - - - -
    #     short attributes names and properties

    @property
    def coords_dict(self):
        output_dict = {}
        output_dict.update(self.X0.as_extended_dict)
        output_dict.update(self.Y0.as_extended_dict)
        return output_dict

    @property
    def modelNames_list(self):
        return [model.modelName for model in self.models_tuple]

    @property
    def X0_dict(self):
        return self.X0.as_extended_dict

    @property
    def Y0_dict(self):
        return self.Y0.as_extended_dict

    @property
    def x0(self):
        return self.X0.value

    @property
    def y0(self):
        return self.Y0.value

    @property
    def x0y0(self):
        return (self.x0, self.y0)

    @property
    def tuple(self):
        return self.models_tuple

    @property
    def as_tuple(self):
        return self.models_tuple

    @property
    def models(self):
        return self.models_tuple

    @property
    def models_block(self):
        return self.models_tuple

    @property
    def size(self):
        return len(self.models_tuple)

    # - - - - - - - - - - - - - - - - - - - - - - -
    #  ImfitModelBlock copy: based on copy method of dict objects

    def __copy__(self):
        list_of_models = [model.copy() for model in self.models_tuple]
        X0 = self.X0.copy()
        Y0 = self.Y0.copy()
        for model in list_of_models:
            model.X0 = X0
            model.Y0 = Y0
        #
        return self.__class__(*list_of_models)

    def copy(self):
        return self.__copy__()

    # - - - - - - - - - - - - - - - - - - - - - - -
    #   different methods for presenting the object

    @property
    def as_file_text(self):
        models_list = [self.as_tuple_of_dicts,]
        file_text = imfit_module.write_config_file(None, models_list)
        return file_text

    @property
    def as_config_lines(self):
        text_lines = self.as_file_text.split('\n')
        clear_lines = [line.strip() for line in text_lines
                       if not (line.startswith('#') or line.strip() == '')]        
        return clear_lines

    @property
    def short_notation(self):
        coords_str = f"{self.x0}, {self.y0}"
        models_notes = [model.short_model_notation
                        for model in self.models_tuple]
        models_str = '\n'.join(models_notes)
        return f"{coords_str}\n{models_str}"

    @property
    def notation(self):
        coords_str = f"------  X0={self.x0}  Y0={self.y0}  ------"
        models_notes = [model.model_notation for model in self.models_tuple]
        models_str = '\n'.join(models_notes)
        return f"{coords_str}\n{models_str}"

    @property
    def as_tuple_of_dicts(self):
        models_block_list = [model.as_dict for model in self.models_tuple]
        return tuple(models_block_list)

    @property
    def first_model(self):
        return self.models_tuple[0]

    @property
    def as_simple(self):
        return self.as_tuple_of_dicts

    @property
    def as_ImfitModel(self):
        if self.size == 1:
            return self.models_tuple[0]
        else:
            raise ValueError("The block has multiple ImfitModel-s")

    @property
    def as_ImfitModelCombination(self):
        return ImfitModelCombination([self,])

    @property
    def as_combination(self):
        return self.as_ImfitModelCombination

    @property
    def as_models_list(self):
        return [self.as_tuple_of_dicts,]





#-----------------------------------------------
#           * ^ * ^ * ^ * ^ * ^ * ^ *
# ------- Class ImfitModelCombination ----------
#-----------------------------------------------



class ImfitModelCombination(object):
    ''' The full imfit model object which contains multiple
        imfit FUNCTION blocks and there are not only one
        imfit FUNCTION in one block
        ImfitModelCombination is a list of ImfitModelBlock objects
        but if input is a list of tuples
        then each tuple automatically converts to ImfitModelBlock
    '''

    def __init__(self, *args):
        ''' Input: ImfitModel class objects
            args — it is a list of tuples
            one arg — it's a tuple
            where tuple is a function block combination with common X0 and Y0
            and other args list items — nother blocks of functions
            with different X0 and Y0
        '''
        models_list = []
        if (len(args) == 1 and 
            (isinstance(args[0], list) or isinstance(args[0], tuple))):
            args = [arg for arg in args[0]]
        #
        for arg in args:
            if not isinstance(arg, ImfitModelBlock):
                arg = ImfitModelBlock(arg)
            models_list.append(arg)
        #
        self.blocks_list = models_list


    @classmethod
    def example(cls, shape=(3,2,1)):
        ''' Returns example ImfitModelCombination (taken from special file)'''
        models_list = [ImfitModelBlock.example(size=size) for size in shape]
        return cls(*models_list)


    @classmethod
    def fromfile(cls, input_file):
        result_models_list = imfit_module.read_config(input_file)
        return cls(result_models_list)


    def write_to_file(self, filename):
        models_list = self.blocks_list
        imfit_module.write_config_file(filename, models_list)


    def makeimage(self, output='modelimage.fits', verbosity=False, **kwargs):
        attrs = kwargs.copy()
        if (attrs.get('refimage') is None and
            (attrs.get('nrows') is None or attrs.get('ncols') is None)):
            # refimage is None; assuming X0 and Y0 as a center of image
            x0y0_list = [block.x0y0 for block in self.blocks_list]
            ncols = max([x0y0[0] for x0y0 in x0y0_list]) * 2
            nrows = max([x0y0[1] for x0y0 in x0y0_list]) * 2
            ncols = int(ncols)
            nrows = int(nrows)
            attrs['nrows'] = nrows
            attrs['ncols'] = ncols
        #
        if output is not None:
            imfit_module.makeimage(self.as_file_text, output=output,
                                    verbosity=verbosity, **attrs)
        else:
            # no image will be saved, just return an array
            data = imfit_module.makeimage(self.as_file_text, output=None,
                                             verbosity=verbosity, **attrs)
            return data

    # - - - - - - - - - - - - - - - - - - - - - - -

    def __str__(self):
        modelNames_in_blocks_list = []
        for models_block in self.blocks_list:
            modelNames_in_models_block = []
            for imfitModel in models_block:
                modelName = imfitModel.modelName
                modelNames_in_models_block.append(modelName)
            #
            tuple_names = tuple(modelNames_in_models_block)
            modelNames_in_blocks_list.append(tuple_names)
        #
        output_str = ''
        for tuple_names in modelNames_in_blocks_list:
            output_str += f" ({','.join(tuple_names)})"
        #
        return output_str

    def __repr__(self):
        output_lines = []
        for models_block in self.blocks_list:
            output_lines.append(models_block.notation)
        output_str = '\n'.join(output_lines)
        return output_str


    # - - - - - - - - - - - - - - - - - - - - - - -
    #  ImfitModelComb copy: based on copy method of dict objects

    def __copy__(self):
        list_of_blocks = [block.copy() for block in self.blocks_list]
        return self.__class__(*list_of_blocks)

    def copy(self):
        return self.__copy__()

    # - - - - - - - - - - - - - - - - - - - - - - -

    @property
    def shape(self):
        return tuple([block.size for block in self.blocks_list])

    @property
    def blocks_number(self):
        return len(self.blocks_list)

    @property
    def models_number(self):
        return sum(self.shape)

    @property
    def size(self):
        return self.models_number

    @property
    def modelNames_list(self):
        output_list = []
        for block in self.blocks_list:
            output_list.extend(block.modelNames_list)
        return output_list

    @property
    def as_file_text(self):
        models_list = [block.as_tuple_of_dicts for block in self.blocks_list]
        file_text = imfit_module.write_config_file(None, models_list)
        return file_text

    @property
    def as_config_lines(self):
        text_lines = self.as_file_text.split('\n')
        clear_lines = [line.strip() for line in text_lines
                       if not (line.startswith('#') or line.strip() == '')]        
        return clear_lines

    @property
    def notation(self):
        output_lines = []
        for models_block in self.blocks_list:
            output_lines.append(models_block.short_notation)
        output_str = '\n'.join(output_lines)
        return output_str

    @property
    def list(self):
        return self.blocks_list

    @property
    def blocks(self):
        return self.blocks_list

    @property
    def as_list_of_tuples_of_dicts(self):
        return [block.as_tuple_of_dicts for block in self.blocks_list]

    @property
    def as_models_list(self):
        return self.as_list_of_tuples_of_dicts

    @property
    def as_simple(self):
        return self.as_list_of_tuples_of_dicts

    @property
    def first_block(self):
        return self.blocks_list[0]

    @property
    def first_model(self):
        return self.first_block.first_model

    @property
    def main_model(self):
        return self.first_model

    @property
    def main(self):
        return self.main_model

    @property
    def as_ImfitModel(self):
        if self.size == 1:
            return self.first_model
        else:
            raise ValueError("The combination has multiple Models")

    @property
    def as_ImfitModelBlock(self):
        if self.blocks_number == 1:
            return self.first_block
        else:
            raise ValueError("The combination has multiple Blocks")





