import numpy
cimport numpy



def cython_first_index(arr:numpy.ndarray, val:int):
    '''
    This module is meant to be an alternative to the numba compiled "first_index" in the main code.
    The reason is that
    '''
    cdef int iii
    cdef int value
    cdef int length
    list = [(index[0], index[1]) for index in numpy.ndenumerate(arr)]
    length = len(list)
    for iii in range(0, length):
        value = list[iii][1]
        if val == value:
             return list[iii][0]