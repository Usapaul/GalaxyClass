
from datetime import datetime

def dict_pair_to_text(key, value):
    ''' Returnes a text block (4 lines) 
        which fully describes the key and the value of a dict
        Examples:
        -------------------
        key:galaxyName
        type:<class 'str'>
        value:NGC224
        type:<class 'str'>
        -------------------
        key:NGC_number
        type:<class 'str'>
        value:224
        type:<class 'int'>
        -------------------
        key:32
        type:<class 'int'>
        value:Andromeda galaxy satellite (Messier catalog number)
        type:<class 'str'>
    '''
    block_list = []
    block_list.append(f"key:{key}")
    block_list.append(f"type:{type(key)}")
    block_list.append(f"value:{value}")
    block_list.append(f"type:{type(value)}")
    text_block = '\n'.join(block_list)
    return text_block


def read_value_with_type(input_value, type_str):
    ''' Just returnes a value in the type if is specified in text
        Example: fun(150, "<class 'float'>") = 150.0
        Avaliable:
        type <class>               examples
        ---------------  -------------------------------
        <class 'bool'>    True / False
        <class 'int'>     100
        <class 'float'>   3.14
        <class 'str'>     "hello"
    '''
    match type_str:
        case "<class 'bool'>"
            value = bool(input_value)
        case "<class 'int'>"
            value = int(input_value)
        case "<class 'float'>"
            value = float(input_value)
        case "<class 'str'>"
            value = str(input_value)
    return value


def text_block_to_dict_pair(input_text_block):
    ''' Reads the input_text_block (list or str) and returnes pair (key, value)'''
    if isinstance(input_text_block, str):
        text_lines = input_text_block.split('\n')
    elif isinstance(input_text_block, list):
        text_lines = input_text_block
    else:
        raise ValueError("input dict text_block has not correct type")
    #
    if len(text_lines) != 4:
        raise ValueError("dict text_block is not correct")
    #
    line_key = text_lines[0].strip()
    line_key_type = text_lines[1].strip()
    line_value = text_lines[2].strip()
    line_value_type = text_lines[3].strip()
    if (not line_key.startswith('key:')
            or not line_key_type.startswith('type:')
            or not line_value.startswith('value:')
            or not line_value_type.startswith('type:')):
        raise ValueError("dict text_block is not correct")
    #
    try:
        key = line_key.replace('key:', '')
        key_type = line_key_type.replace('type:', '')
        key = read_value_with_type(key, key_type)
        value = line_value.replace('value:', '')
        value_type = line_value_type.replace('type:', '')
        value = read_value_with_type(value, key_type)
    except:
        raise ValueError("dict text_block is not correct")
    #
    return (key, value)


def dict_as_file_text(input_dict):
    ''' Returns a text which contains full description
        of every single pair "key:value" for the input dict.
        And each pair is separated by an empty line
        Values can have the following types:
        str, int, float, bool
        #
        Example of an output text:
        >>>
        key:galaxyName
        type:<class 'str'>
        value:NGC224
        type:<class 'str'>

        key:NGC_number
        type:<class 'str'>
        value:224
        type:<class 'int'>

        key:32
        type:<class 'int'>
        value:Andromeda galaxy satellite (Messier catalog number)
        type:<class 'str'>
        <<<
    '''
    blocks = []
    for key, value in input_dict.items():
        blocks.append(dict_pair_to_text(key, value))
    full_text = '\n\n'.join(blocks)
    return full_text


def read_dict_from_text(input_str):
    ''' Returns a dict which had read from a text written in
        the special format — it contains blocks,
        each block for every single pair "key:value"
        but it knows a variable type (int, float, str, bool)
        '''
    if isinstance(input_str, list):
        # if text lines in a list
        text = '\n'.join(input_str)
    elif not '\n' in input_str:
        # it's a filename
        with open(input_str) as file:
            text = file.read()
    else:
        # the variable is an entire text like a file
        text = input_str
    #
    output_dict = {}
    text_lines = text.split('\n')
    text_lines = [line for line in text_lines if line.strip() != '']
    Nlines = len(text_lines)
    for i in range(0, Nlines, step=4):
        dict_text_block = text_lines[i:i+4]
        key, value = text_block_to_dict_pair(dict_text_block)
        output_dict[key] = value
    #
    return output_dict



# ------------------------------------------------
# ------------------------------------------------


def datetime_human(input_str):
    ''' Returnes string with human readable date time
        with specified interesting data
        #
        Keys:
        %Y(y)   — year: (20)22
        %m(b,B) — month: 01(Jan_uary) – 12(Dec_ember)
        %d(j)   — day of the month: 01 – 31 (of the year 000-366)
        %H(I,p) — hours: 00-24(01-12,AM/PM)
        %M      — minutes: 00-59
        %S(f)   — seconds: 00-59 (microsec 123456)
        %c(x,X) — datetime: {locale} datetime(date,time)
        %z(Z)   — timezone: /name/(/offset/)
        %w(a,A) — weekday: 0(Sun_day) – 6(Sat_urday)
        %W(U)   — week number: 00-53, Monday(Sunday) is 1st and >="01"
        %%      — literal '%' character
    '''
    return datetime.today().strftime(input_str)




def uniq_filename(prefix=None, suffix=None, ext=None,
                  check_dirname=None):
    ''' Returnes string which is a filename but with special note
        — date and time note which is neccesary for unique name
        Also it checks does the specified dirname contains a
        file with a simple name and then creates more difficult name
        Example:
        uniq_filename(prefix='data_',ext='txt') = 'data_220101.txt'
        uniq_filename(prefix='data_',ext='txt') = 'data_220203_235959.txt'
    '''
    # output_parts = [prefix, datetime, suffix, ext]
    output_parts = ['', '', '', '']
    #
    if prefix is not None:
        output_parts[0] = prefix
    #
    date_time_f = datetime.today().strftime("%y%m%d_%H%M%S_%f")
    date, time, microseconds = date_time_f.split('_')
    output_parts[1] = date
    #
    if suffix is not None:
        output_parts[2] = suffix
    #
    if ext is None:
        ext = '.txt'
    else:
        if not ext.startswith('.'):
            ext = '.' + ext
    output_parts[3] = ext
    # --------
    # IF don't check dirname
    if check_dirname is None:
        return ''.join(output_parts)
        #---------------------------
    # ELSE:
    filename = ''.join(output_parts)
    file_abspath = os.path.join(check_dirname, filename)
    # --------
    # IF file does not exist:
    if not os.path.exist(file_abspath):
        return filename
        #---------------------------
    # ELSE:
    output_parts[2] = f"{date}_{time}"
    filename = ''.join(output_parts)
    file_abspath = os.path.join(check_dirname, filename)
    # --------
    # IF file does not exist:
    if not os.path.exist(file_abspath):
        return filename
        #---------------------------
    # ELSE:
    output_parts[2] = f"{date}_{time}_{microseconds}"
    filename = ''.join(output_parts)
    file_abspath = os.path.join(check_dirname, filename)
    # --------
    # IF file does not exist:
    if not os.path.exist(file_abspath):
        return filename
        #---------------------------
    # ELSE:   (WOW!! EQUAL MICROSECONDS!?!?)
    # date_time_f = f"{date}_{time}_{microseconds}"
    i = 1 # the existing file is first
    while os.path.exist(file_abspath):
        i += 1
        file_abspath = os.path.join(check_dirname, filename+'_'+str(i))
    filename += str(i) # found!
    return filename











