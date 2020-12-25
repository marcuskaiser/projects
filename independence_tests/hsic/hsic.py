import numpy as np
from scipy.stats import gamma


def _gaussian_kernel(x, sigma=None):
    """
    Gaussian / RBF kernel. Allows to add extra noise to the diagonal in order
    to make the resulting Gram matrix positive definite.
    The returned matrix is ``k(x, x)``.

    Parameters
    ----------
    x : array-like, shape (n_samples,)
        Vector for which the kernel Gram matrix will be calculated.

    sigma : float or None, optional (default=None)
        Bandwidth for the kernel. By default, the median euclidean
        distance between points is used.

    Returns
    -------
    k_ : array-like
        Resulting Gram matrix.
    """
    d = x - x[:, None]
    if sigma is None:
        sigma = np.median(d[d > 0])

    k_ = np.exp(- (d / sigma) ** 2)
    return k_


def _laplace_kernel(x, sigma=None):
    """
    Laplace kernel. Allows to add extra noise to the diagonal in order
    to make the resulting Gram matrix positive definite.
    The returned matrix is ``k(x, x)``.

    Parameters
    ----------
    x : array-like, shape (n_samples,)
        Vector for which the kernel Gram matrix will be calculated.

    sigma : float or None, optional (default=None)
        Bandwidth for the kernel. By default, the median euclidean
        distance between points is used.

    Returns
    -------
    k_ : array-like
        Resulting Gram matrix.
    """
    d = x - x[:, None]
    if sigma is None:
        sigma = np.median(d[d > 0])

    k_ = np.exp(- np.abs(d / sigma))
    return k_


def _rational_quadratic_kernel(x, alpha=1.0, sigma=None):
    """
    Rational Quadratic kernel. Can be interpreted as a mixture of RBF kernels.
    Allows to add extra noise to the diagonal in order to make the resulting
    Gram matrix positive definite.
    The returned matrix is ``k(x, x)``.

    Parameters
    ----------
    x : array-like, shape (n_samples,)
        Vector for which the kernel Gram matrix will be calculated.

    alpha : float, (default=1.0)
        Scale mixture parameter.

    sigma : float or None, optional (default=None)
        Bandwidth for the kernel. By default, the median euclidean
        distance between points is used.

    Returns
    -------
    k_ : array-like
        Resulting Gram matrix.
    """
    d = x - x[:, None]
    if sigma is None:
        sigma = np.median(d[d > 0])

    k_ = (1.0 + (d / sigma) ** 2.0 / (2.0 * alpha)) ** (-alpha)
    return k_


KERNEL_MAP = {
    'gaussian': _gaussian_kernel,
    'laplace': _laplace_kernel,
    'rational_quadratic': _rational_quadratic_kernel
}


def _hsic_kernel(k_x, k_y, scale=False, dof=0):
    """
    Calculate the HSIC score from the provided gram matrices prior created
    using a SPD kernel.

    Parameters
    ----------
    k_x : array-like, shape (n_samples, n_samples)
        The Gram matrix for the x variable.

    k_y : array-like, shape (n_samples, n_samples)
        The Gram matrix for the y variable.

    scale : bool, optional (default=False)
        If true, the result is scaled (from 0 to 1) as

        .. math::
            result = hsic(x, y) / (hsic(x, x) * hsic(y, y)) ** 0.5

    dof : int, optional (default=1)
        Degree of freedom for scaling the non-normalized score. Default is 0.

    Returns
    -------
    hsic_score : float
        HSIC score
    """
    k_x = k_x - np.mean(k_x, axis=1)
    k_y = k_y - np.mean(k_y, axis=1)

    trace_xy = np.einsum('ji,ij', k_x, k_y)
    if scale:
        trace_xx = np.einsum('ji,ij', k_x, k_x)
        trace_yy = np.einsum('ji,ij', k_y, k_y)
        hsic_score = trace_xy / (trace_xx * trace_yy) ** 0.5
    else:
        hsic_score = trace_xy / (k_x.shape[0] - dof) ** 2
    return hsic_score


def hsic(x, y, scale=False, sigma_x=None, sigma_y=None, kernel='gaussian',
         dof=0):
    """
    (Fairly efficient) Python implementation of HSIC
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

    kernel : str, optional (default='gaussian')
        The type of kernel to be used. Choices are ``gaussian``, ``laplace``
        and ``rational_quadratic``.

    dof : int, optional (default=1)
        Degree of freedom for scaling the non-normalized score. Default is 0.

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
    kernel_fn = KERNEL_MAP[kernel]
    k_x = kernel_fn(x, sigma=sigma_x)
    k_y = kernel_fn(y, sigma=sigma_y)
    hsic_score = _hsic_kernel(k_x, k_y, scale, dof=dof)
    hsic_score_xx = _hsic_kernel(k_x, k_x, scale, dof=dof)
    hsic_score_yy = _hsic_kernel(k_y, k_y, scale, dof=dof)

    m = x.shape[0]

    # Construct p-value:
    kx_mean = k_x.mean()
    ky_mean = k_y.mean()
    hsic_mu = (1.0 + kx_mean * ky_mean - kx_mean - ky_mean) / m
    hsic_var = 2.0 * (m - 4) * (m - 5) / (m * (m - 1) * (m - 2) * (m - 3))
    hsic_var *= hsic_score_xx * hsic_score_yy
    alpha = hsic_mu ** 2 / hsic_var
    beta = m * hsic_var / hsic_mu
    p_value = 1.0 - gamma.cdf(m * hsic_score, alpha, scale=beta)

    return hsic_score, p_value


def _hsic_naive(x, y, scale=False, sigma_x=None, sigma_y=None,
                kernel='gaussian', dof=0):
    """
    Naive (slow) implementation of HSIC (Hilbert-Schmidt Independence
    Criterion). This function is only used to assert correct results of the
    faster method ``hsic``.

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

    kernel : str, optional (default='gaussian')
        The type of kernel to be used. Choices are ``gaussian``, ``laplace``
        and ``rational_quadratic``.

    dof : int, optional (default=1)
        Degree of freedom for scaling the non-normalized score. Default is 1.

    Returns
    -------
    hsic_score : float
        HSIC score
    """
    kernel_fn = KERNEL_MAP[kernel]

    m = x.shape[0]
    h = np.eye(m) - 1 / m
    k_x = kernel_fn(x, sigma=sigma_x) @ h
    k_y = kernel_fn(y, sigma=sigma_y) @ h
    trace_xy = np.sum(k_x.T * k_y)

    if scale:
        trace_xx = np.sum(k_x.T * k_x)
        trace_yy = np.sum(k_y.T * k_y)
        hsic_score = trace_xy / (trace_xx * trace_yy) ** 0.5
    else:
        hsic_score = trace_xy / (m - dof) ** 2
    return hsic_score
