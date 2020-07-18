from libc.math cimport fabs

cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float_
ctypedef np.double_t DTYPE_t

np.import_array()


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef dcorr(double[:] x, double[:] y, bint scale=True):
    """
    Cython code for distance correlation. 
    Equivalent result to Python version:

    >>> a = np.abs(x - x[:, None])
    >>> a_mean = a.mean(axis=0, keepdims=True)
    >>> a[:] += - a_mean - a_mean.T + a_mean.mean()
    >>> 
    >>> b = np.abs(y - y[:, None])
    >>> b_mean = b.mean(axis=0, keepdims=True)
    >>> b[:] += - b_mean - b_mean.T + b_mean.mean()
    >>> 
    >>> cov_ab = np.einsum('ij,ij', a, b) ** 0.5
    >>> var_a = np.einsum('ij,ij', a, a) ** 0.25
    >>> var_b = np.einsum('ij,ij', b, b) ** 0.25
    """
    cdef Py_ssize_t n = x.shape[0]
    cdef Py_ssize_t i
    cdef Py_ssize_t j

    if n == 0:
        return 1.0

    cdef np.ndarray[DTYPE_t, ndim=1] a_mean_1 = np.zeros(n, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] b_mean_1 = np.zeros(n, dtype=DTYPE)
    cdef DTYPE_t a_mean_2 = 0.0
    cdef DTYPE_t b_mean_2 = 0.0
    cdef DTYPE_t cov_ab = 0.0
    cdef DTYPE_t var_a = 0.0
    cdef DTYPE_t var_b = 0.0
    cdef DTYPE_t a_ij = 0.0
    cdef DTYPE_t b_ij = 0.0

    for i in range(n):
        for j in range(n):
            a_mean_1[i] += fabs(x[i] - x[j])
            b_mean_1[i] += fabs(y[i] - y[j])
        a_mean_1[i] /= n
        b_mean_1[i] /= n
        a_mean_2 += a_mean_1[i]
        b_mean_2 += b_mean_1[i]
    a_mean_2 /= n
    b_mean_2 /= n

    if not scale:
        for i in range(n):
            for j in range(n):
                a_ij = fabs(x[i] - x[j]) - a_mean_1[i] - a_mean_1[j] + a_mean_2
                b_ij = fabs(y[i] - y[j]) - b_mean_1[i] - b_mean_1[j] + b_mean_2
                cov_ab += a_ij * b_ij
        return cov_ab ** 0.5 / n

    for i in range(n):
        for j in range(n):
            a_ij = fabs(x[i] - x[j]) - a_mean_1[i] - a_mean_1[j] + a_mean_2
            b_ij = fabs(y[i] - y[j]) - b_mean_1[i] - b_mean_1[j] + b_mean_2
            cov_ab += a_ij * b_ij
            var_a += a_ij ** 2
            var_b += b_ij ** 2

    if var_a * var_b > 0.0:
        return cov_ab ** 0.5 / (var_a * var_b) ** 0.25
    return 1.0
