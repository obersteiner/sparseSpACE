import numpy as np
from typing import List, Set, Dict, Tuple, Optional, Union, Sequence, Generator
from itertools import product
import logging
import time
import sys
import types

if sys.version_info[0] == 3 and sys.version_info[1] >= 7:
    timing = time.time_ns
else:
    timing = time.time


def get_cross_product(one_d_arrays: Sequence[Sequence[Union[float, int]]]) \
        -> Generator[Tuple[Union[float, int], ...], None, None]:
    #dim = len(one_d_arrays)
    #return list(zip(*[g.ravel() for g in
    #          np.meshgrid(*[one_d_arrays[d] for d in range(dim)], indexing="ij")]))
    return product(*one_d_arrays)


def get_cross_product_list(one_d_arrays: Sequence[Sequence[Union[float, int]]]) -> List[Tuple[Union[float, int], ...]]:
    return list(get_cross_product(one_d_arrays))

def get_cross_product_numpy_array(one_d_arrays: Sequence[Sequence[Union[float, int]]]) -> np.ndarray:
    return np.array(get_cross_product_list(one_d_arrays))

def get_cross_product_range(one_d_arrays: Sequence[Sequence[int]]) -> Generator[Tuple[int, ...], None, None]:
    return get_cross_product([range(one_d_array) for one_d_array in one_d_arrays])


def get_cross_product_range_list(one_d_arrays: Sequence[Sequence[int]]) -> List[Tuple[int, ...]]:
    return get_cross_product_list([range(one_d_array) for one_d_array in one_d_arrays])

# Default log config
log_filename = 'log_sg'
log_format = '%(asctime)s,%(msecs)d %(name)s %(levelname)s :: %(message)s'
log_date_format = '%H:%M:%S'
logging.basicConfig(filename=log_filename,
                    filemode='a',
                    format=log_format,
                    datefmt=log_date_format,
                    level=logging.ERROR)
logger = logging.getLogger('util')
logger.setLevel(logging.DEBUG)

class print_levels:
    ERROR = 40
    WARNING = 30
    INFO = 20
    DEBUG = 10
    NONE = 0


class log_levels:
    ERROR = 40
    WARNING = 30
    INFO = 20
    DEBUG = 10
    NONE = 0


class LogUtility:
    """
    Helper class for logging and printing.
    logging to file and printing to console can take a substantial amount of time relative to calculating.
    Make sure to set the print and log levels appropriately.
    """
    def __init__(self, log_level=logging.WARNING, print_level=logging.WARNING):

        # keep intial definition in __init__ to silence PEP 8 formatting complaints
        self.log_error_fn = lambda x: None
        self.log_warning_fn = lambda x: None
        self.log_info_fn = lambda x: None
        self.log_debug_fn = lambda x: None

        self.print_error = lambda x: None
        self.print_warning = lambda x: None
        self.print_info = lambda x: None
        self.print_debug = lambda x: None

        self.time_func = lambda fn, *kwargs: fn(*kwargs)

        self.print_prefix = ''
        self.print_delimiter = ' ## '
        self.log_prefix = ''
        self.log_delimiter = ' ## '

        self.print_level = print_level
        self.log_level = log_level
        self.update_log_function()
        self.update_print_function()

    def set_print_level(self, level):
        self.print_level = level
        self.update_print_function()

    def set_print_delimiter(self, delimiter):
        self.print_delimiter = delimiter

    def set_print_prefix(self, prefix):
        self.print_prefix = prefix

    def set_log_level(self, level):
        self.log_level = level
        if level is log_levels.ERROR:
            logger.setLevel(logging.ERROR)
        elif level is log_levels.WARNING:
            logger.setLevel(logging.WARNING)
        elif level is log_levels.INFO:
            logger.setLevel(logging.INFO)
        elif level is log_levels.DEBUG:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.NOTSET)
        self.update_log_function()

    def set_log_delimiter(self, delimiter):
        self.log_delimiter = delimiter

    def set_log_prefix(self, prefix):
        self.log_prefix = prefix

    def update_log_function(self):
        self.log_error_fn = lambda x: logger.error(self.log_prefix + self.log_delimiter + x) if self.log_level >= logging.ERROR \
            else lambda x: None
        self.log_warning_fn = lambda x: logger.warning(self.log_prefix + self.log_delimiter + x) if self.log_level >= logging.WARNING \
            else lambda x: None
        self.log_info_fn = lambda x: logger.info(self.log_prefix + self.log_delimiter + x) if self.log_level >= logging.INFO \
            else lambda x: None
        self.log_debug_fn = lambda x: logger.debug(self.log_prefix + self.log_delimiter + x) if self.log_level >= logging.DEBUG \
            else lambda x: None

        self.time_func = self.timing_wrapper if self.log_level >= log_levels.INFO \
            else lambda msg, fn, *kwargs: fn(*kwargs)

    def update_print_function(self):
        self.print_error = self.print_message if self.print_level >= print_levels.ERROR else lambda x: None
        self.print_warning = self.print_message if self.print_level >= print_levels.WARNING else lambda x: None
        self.print_info = self.print_message if self.print_level >= print_levels.INFO else lambda x: None
        self.print_debug = self.print_message if self.print_level >= print_levels.DEBUG else lambda x: None

        self.time_func = self.timing_wrapper if self.print_level >= print_levels.INFO \
            else lambda msg, fn, *kwargs: fn(*kwargs)

    def print_message(self, message):
        print(self.print_prefix + self.print_delimiter + message)

    def log_error(self, message: str = '') -> None:
        self.log_error_fn(message)
        self.print_error(message)

    def log_warning(self, message: str = '') -> None:
        self.log_warning_fn(message)
        self.print_warning(message)

    def log_info(self, message: str = '') -> None:
        self.log_info_fn(message)
        self.print_info(message)

    def log_debug(self, message: str = '') -> None:
        self.log_debug_fn(message)
        self.print_debug(message)

    def timing_wrapper(self, message, function, *kwargs):
        """ Wrapper function for timing functions
        :param message: The message to print to log (and to console)
        :param function: The function to be timed
        :param kwargs: Arguments for the function to be timed

        """
        start = timing()
        ret = function(*kwargs)
        result = timing() - start
        self.print_info("{0} : {1}".format(message, result))
        self.log_info_fn("{0} : {1}".format(message, result))
        return ret

    def clear_log(self):
        open(log_filename, 'w').close()


# the standard log utility for execution scripts
# source files should get their own log utility instance and set their log and printing levels individually
logUtil = LogUtility()


def clear_log():
    open(log_filename, 'w').close()


def change_log_file(name):
    file = open(name, 'w+')  # create logfile if it doesn't exist
    file.close()
    fileh = logging.FileHandler(name, 'a')
    formatter = logging.Formatter(log_format)
    fileh.setFormatter(formatter)

    log = logging.getLogger()  # root logger
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)
    log.addHandler(fileh)  # set the new handler


class ProfileStuff:
    """
    Helper class for keeping profiling files in order.
    """
    def __init__(self):
        self.data_set_used = 'Unnamed'
        self.fileNameExtension = ''
        self.class_counter = 0
        self.used_std_points = 0

    def set_data_set_used(self, n):
        self.data_set_used = n

    def set_file_name_extension(self, n):
        self.fileNameExtension = n

    def get_data_set_used(self):
        return self.data_set_used

    def get_file_name_extension(self):
        return self.fileNameExtension

    def get_class_counter(self):
        return self.class_counter

    def increment_class_counter(self):
        self.class_counter += 1

    def reset_class_counter(self):
        self.class_counter = 0

pStuff = ProfileStuff()



