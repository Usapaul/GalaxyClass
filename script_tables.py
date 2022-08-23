# ----------- Class AstroImage -----------------

import os

from datetime import datetime

#-----------------------------------------------


#-----------------------------------------------
#           * ^ * ^ * ^ * ^ * ^ * ^ *
# -------------- Class ScriptList --------------
#-----------------------------------------------

class ScriptList(object):
    ''' 
    '''
    #functions_list = []
    #methods_list = []
    items_list = []
    history_list = []
    comments_list = []
    def __init__(self, iterable_input):
        self.items_list = list(iterable_input)

    def run_function(self, fun, history=None, comment=None, fast=False,
                     *args, **kwargs):
        if not callable(fun):
            raise TypeError(f"Expected python function in input {fun}")
        #
        if fast is False:
            for idx, item in enumerate(self.items_list):
                if hasattr(item, 'copy'):
                    item_copy = item.copy()
                elif isinstance(item, str | int | float | bool | NoneType):
                    item_copy = item
                else:
                    raise TypeError("item {idx} can't be copied")
                #
                try:
                    dump_output = fun(item_copy)
                except:
                    error_message = (f"{fun.__name__}(item_list[{idx}])"
                                     " didn't passed")
                    raise TypeError(error_message)
        #
        for item in self.items_list:
            fun(item, *args, **kwargs)
        #
        if history is not None:
            history_text = history
        else:
            history_text = f"function {fun.__name__}"
        self.update_log(history=history_text, comment=comment)


    def run_method(self, meth, history=None, comment=None, fast=False,
                   *args, **kwargs):
        if any(not hasattr(item, meth)
               or not callable(getattr(item, meth))
               for item in self.items_list):
            raise TypeError(f"Some items don't have the input method")
        #
        if fast is False:
            for idx, item in enumerate(self.items_list):
                if hasattr(item, 'copy'):
                    item_copy = item.copy()
                elif isinstance(item, str | int | float | bool | NoneType):
                    item_copy = item
                else:
                    raise TypeError("item {idx} can't be copied")
                #
                try:
                    item_attr_copy = getattr(item_copy, meth)
                    dump_output = item_copy_attr()
                except:
                    error_message = (f"item_list[{idx}].{meth}()"
                                     " didn't passed")
                    raise TypeError(error_message)
        #
        for item in self.items_list:
            item_method = getattr(item, meth)
            item_method(*args, **kwargs)
        #
        #
        if history is not None:
            history_text = history
        else:
            history_text = f"method {meth}"
        self.update_log(history=history_text, comment=comment)


    def update_log(self, history=None, comment=None):
        oldN = len(self.history_list) - 1
        newN = oldN + 1
        current_time = datetime.today().strftime("%y%m%d_%H%M%S")
        if history is not None:
            history_text = history
        else:
            history_text = "--"
        self.history_list.append((newN, current_time, history_text))
        if comment is not None:
            comment_text = comment
        else:
            comment_text = ''
        self.comment_list.append((newN,comment_text))


    @property
    def size(self):
        return len(self.items_list)


    @property
    def history_length(self):
        return len(history_length)
