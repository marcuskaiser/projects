from libc.math cimport exp

cimport cython
import numpy as np
cimport numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import gamma

DTYPE = np.float_
ctypedef np.double_t DTYPE_t

np.import_array()

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef hsic(DTYPE_t[:] x, DTYPE_t[:] y, DTYPE_t sigma_x=0.0,
           DTYPE_t sigma_y=0.0, bint scale=False, Py_ssize_t dof=0):
    """
    Cython code for HSIC (with RBF kernel).

       (Hilbert-Schmidt Independence Criterion):

    .. math::
        HSIC = (m - 1)^{-1} tr(k_x k_y}

    where :math:`m` is the number of samples and :math:`k_x` and :math:`k_y`
    are centred kernel gram matrices for x and y, respectively.

    This formula can be derived as the squared Hilbert-Schmidt norm of
    the cross-covariance operator (with feature maps :math:`\\phi` and
    :math:`\\psi` and mean )

    .. math::

        C_{xy} = E_{xy}[\\phi_x(x) \\otimes \\phi_y(y)]
                 - \\mu_x \\otimes \\mu_y

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

    Parameters
    ----------
    x : array-like, shape (n_samples,)
        First vector to be compared.

    y : array-like, shape (n_samples,)
        Second vector to be compared.

    scale : bool, optional (default=False)
        If true, the result is scaled (from 0 to 1) as

        .. math::
            result = hsic(x, y) / (hsic(x, x) * hsic(y, y)) ** 0.5

    sigma_x : float or None, optional (default=None)
        Bandwidth for the kernel of the x variable.
        By default, the median euclidean distance between points is used.

    sigma_y : float or None, optional (default=None)
        Bandwidth for the kernel of the y variable.
        By default, the median euclidean distance between points is used.

    Returns
    -------
    hsic_score : float
        HSIC score. (The test statistic is equal to n_samples time the score.)

    p_value : float
        P-value estimated via gamma approximation.

    Notes
    -----

        HSIC was first introduced in
        http://www.gatsby.ucl.ac.uk/~gretton/papers/GreBouSmoSch05.pdf

        The Gamma approximation for p-values is e.g. described in
        https://papers.nips.cc/paper/2007/hash/d5cfead94f5350c12c322b5b664544c1-Abstract.html

    """
    cdef Py_ssize_t m = x.shape[0]
    cdef Py_ssize_t i
    cdef Py_ssize_t j

    if m == 0:
        return 0.0

    if sigma_x == 0.0:
        sigma_x = np.median(pdist(np.array(x).reshape(-1, 1),
                                  metric='sqeuclidean'))
    if sigma_y == 0.0:
        sigma_y = np.median(pdist(np.array(y).reshape(-1, 1),
                                  metric='sqeuclidean'))

    cdef np.ndarray[DTYPE_t, ndim=1] kx_mean_1 = np.zeros(m, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] ky_mean_1 = np.zeros(m, dtype=DTYPE)
    cdef DTYPE_t trace_xy = 0.0
    cdef DTYPE_t trace_xx = 0.0
    cdef DTYPE_t trace_yy = 0.0
    cdef DTYPE_t kx_ij = 0.0
    cdef DTYPE_t ky_ij = 0.0
    cdef DTYPE_t kx_mean = 0.0
    cdef DTYPE_t ky_mean = 0.0
    cdef DTYPE_t hsic_mu = 0.0
    cdef DTYPE_t hsic_var = 0.0

    kx_mean = 0.0
    ky_mean = 0.0
    for i in range(m):
        for j in range(m):
            kx_ij = exp(-(x[i] - x[j]) ** 2 / sigma_x)
            ky_ij = exp(-(y[i] - y[j]) ** 2 / sigma_y)
            kx_mean_1[i] += kx_ij
            ky_mean_1[i] += ky_ij
        kx_mean_1[i] /= m
        ky_mean_1[i] /= m
        kx_mean += kx_mean_1[i]
        ky_mean += ky_mean_1[i]
    kx_mean /= m
    ky_mean /= m

    for i in range(m):
        for j in range(m):
            kx_ij = exp(-(x[i] - x[j]) ** 2 / sigma_x)
            ky_ij = exp(-(y[i] - y[j]) ** 2 / sigma_y)
            trace_xy += (kx_ij - kx_mean_1[i]) * (ky_ij - ky_mean_1[j])
            trace_xx += (kx_ij - kx_mean_1[i]) * (kx_ij - kx_mean_1[j])
            trace_yy += (ky_ij - ky_mean_1[i]) * (ky_ij - ky_mean_1[j])
    trace_xy /= (m - dof) ** 2
    trace_xx /= (m - dof) ** 2
    trace_yy /= (m - dof) ** 2

    # Calculate p-value for test statistic, based on gamma approximation,
    hsic_mu = (1.0 + kx_mean * ky_mean - kx_mean - ky_mean) / m
    hsic_var = 2.0 * (m - 4) * (m - 5) / (m * (m - 1) * (m - 2) * (m - 3))
    hsic_var *= trace_xx * trace_yy
    alpha = hsic_mu ** 2 / hsic_var
    beta = m * hsic_var / hsic_mu
    p_value = 1.0 - gamma.cdf(m * trace_xy, alpha, scale=beta)

    if not scale:
        return trace_xy, p_value
    elif trace_xx * trace_yy > 0.0:
        return trace_xy / (trace_xx * trace_yy) ** 0.5, p_value
    else:
        return 0.0, 1.0
