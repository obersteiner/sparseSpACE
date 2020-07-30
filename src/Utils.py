import numpy as np
from typing import List, Set, Dict, Tuple, Optional, Union, Sequence, Generator
from itertools import product
import logging


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


log_filename = 'log_sg'
logging.basicConfig(filename=log_filename,
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s :: %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.ERROR)
logger = logging.getLogger('util')
logger.setLevel(logging.DEBUG)


def log_error(message: str = '', do_print: bool = False):
    logger.error(message)
    if do_print:
        print(message)


def log_warning(message: str = '', do_print: bool = False):
    logger.warning(message)
    if do_print:
        print(message)


def log_info(message: str = '', do_print: bool = False):
    logger.info(message)
    if do_print:
        print(message)


def log_debug(message: str = '', do_print: bool = False):
    logger.debug(message)
    if do_print:
        print(message)

def clear_log():
    open(log_filename, 'w').close()

