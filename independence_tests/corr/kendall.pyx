from libc.math cimport fabs
import numpy as np
import cython

DTYPE = np.float_

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef kendall_cy(double[:] x, double[:] y):
    """ 
    Simple implementation of Kenndall Tau.
    Note: This implementation is O(n ** 2). One should prefer the O(n * log n)
    `scipy.stats.kendalltau` implementation over this code.
    """
    cdef double sign = 0.0
    cdef double c = 0.0
    cdef Py_ssize_t n = x.shape[0]
    cdef Py_ssize_t i
    cdef Py_ssize_t j

    for i in range(n):
        for j in range(i + 1, n):
            c = (x[i] - x[j]) * (y[i] - y[j])
            if c != 0.0:
                sign += c / fabs(c)
    return sign * 2.0 / (n * (n - 1))


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def kendall_window(double[:] x, double[:] y, Py_ssize_t window_size=10):
    cdef double sign = 0.0
    cdef double c = 0.0
    cdef Py_ssize_t m = x.shape[0]
    cdef Py_ssize_t i
    cdef Py_ssize_t j

    for i in range(window_size, m):
        for j in range(i-window_size, i):
            c = (x[i] - x[j]) * (y[i] - y[j])
            if c != 0.0:
                sign += c / fabs(c)
    return sign / ((m - window_size) * window_size)
