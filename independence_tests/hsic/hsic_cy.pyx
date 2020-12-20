from libc.math cimport exp

cimport cython
import numpy as np
cimport numpy as np
from scipy.spatial.distance import pdist

DTYPE = np.float_
ctypedef np.double_t DTYPE_t

np.import_array()

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef hsic(DTYPE_t[:] x, DTYPE_t[:] y, DTYPE_t sigma_x=0.0,
           DTYPE_t sigma_y=0.0, bint scale=False):
    """
    Cython code for HSIC (with RBF kernel).
    Equivalent result to Python version:

    >>> k_x -= np.mean(k_x, axis=1)
    >>> k_y -= np.mean(k_y, axis=1)
    >>>
    >>> trace_xy = np.einsum('ji,ij', k_x, k_y)
    >>> if scale:
    >>>     trace_xx = np.einsum('ji,ij', k_x, k_x)
    >>>     trace_yy = np.einsum('ji,ij', k_y, k_y)
    >>>     hsic_score = trace_xy / (trace_xx * trace_yy) ** 0.5
    >>> else:
    >>>     hsic_score = trace_xy / (k_x.shape[0] - 1) ** 2
    """
    cdef Py_ssize_t n = x.shape[0]
    cdef Py_ssize_t i
    cdef Py_ssize_t j

    if n == 0:
        return 1.0

    if sigma_x == 0.0:
        sigma_x = np.median(pdist(np.array(x).reshape(-1, 1),
                                  metric='sqeuclidean'))
    if sigma_y == 0.0:
        sigma_y = np.median(pdist(np.array(y).reshape(-1, 1),
                                  metric='sqeuclidean'))

    cdef np.ndarray[DTYPE_t, ndim=1] kx_mean_1 = np.zeros(n, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] ky_mean_1 = np.zeros(n, dtype=DTYPE)
    cdef DTYPE_t trace_xy = 0.0
    cdef DTYPE_t trace_xx = 0.0
    cdef DTYPE_t trace_yy = 0.0
    cdef DTYPE_t kx_ij = 0.0
    cdef DTYPE_t ky_ij = 0.0

    for i in range(n):
        for j in range(n):
            kx_ij = exp(-0.5 * (x[i] - x[j]) ** 2 / sigma_x)
            ky_ij = exp(-0.5 * (y[i] - y[j]) ** 2 / sigma_y)
            kx_mean_1[i] += kx_ij
            ky_mean_1[i] += ky_ij
        kx_mean_1[i] /= n
        ky_mean_1[i] /= n

    if not scale:
        for i in range(n):
            for j in range(n):
                kx_ij = exp(-0.5 * (x[i] - x[j]) ** 2 / sigma_x)
                ky_ij = exp(-0.5 * (y[i] - y[j]) ** 2 / sigma_y)
                trace_xy += (kx_ij - kx_mean_1[i]) * (ky_ij - ky_mean_1[j])
        return trace_xy / (n - 1) ** 2

    for i in range(n):
        for j in range(n):
            kx_ij = exp(-0.5 * (x[i] - x[j]) ** 2 / sigma_x)
            ky_ij = exp(-0.5 * (y[i] - y[j]) ** 2 / sigma_y)
            trace_xy += (kx_ij - kx_mean_1[i]) * (ky_ij - ky_mean_1[j])
            trace_xx += (kx_ij - kx_mean_1[i]) * (kx_ij - kx_mean_1[j])
            trace_yy += (ky_ij - ky_mean_1[i]) * (ky_ij - ky_mean_1[j])

    if trace_xx * trace_yy > 0.0:
        return trace_xy / (trace_xx * trace_yy) ** 0.5
    return 1.0
