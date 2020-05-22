import numpy as np


def _gaussian_kernel(x, y=None, sigma=None, diag_noise=None):
    """
    Gaussian / RBF kernel. Allows to add extra noise to the diagonal in order
    to make the resulting Gram matrix positive definite.
    The returned matrix is ``k(x, y)`` if ``y`` is provided, and ``k(x, x)``
    otherwise.

    Parameters
    ----------
    x : array-like, shape (n_samples,)
        Vector for which the kernel Gram matrix will be calculated.

    y : array-like or None, optional (default=None), shape (n_samples,)
        Optional, second vector for calculating the kernel Gram matrix.

    sigma : float or None, optional (default=None)
        Bandwidth for the kernel. By default, the median euclidean
        distance between points is used.

    diag_noise : float or None, optional (default=None)
        Additional noise factor added to the diagonal in order to make the
        resulting Gram matrix positive definite.

    Returns
    -------
    k_ : array-like
        Resulting Gram matrix.
    """
    if y is None:
        y = x
    d = x - y[:, None]
    if sigma is None:
        sigma = np.median(d[d > 0])

    k_ = np.exp(-0.5 * (d / sigma) ** 2)
    if diag_noise is None:
        return k_

    assert diag_noise >= 0.0, f'noise >=0 required. Got: {diag_noise}!'
    np.fill_diagonal(k_, k_.diagonal() + diag_noise)
    return k_


def _laplace_kernel(x, y=None, sigma=None, diag_noise=None):
    """
    Laplace kernel. Allows to add extra noise to the diagonal in order
    to make the resulting Gram matrix positive definite.
    The returned matrix is ``k(x, y)`` if ``y`` is provided, and ``k(x, x)``
    otherwise.

    Parameters
    ----------
    x : array-like, shape (n_samples,)
        Vector for which the kernel Gram matrix will be calculated.

    y : array-like or None, optional (default=None), shape (n_samples,)
        Optional, second vector for calculating the kernel Gram matrix.

    sigma : float or None, optional (default=None)
        Bandwidth for the kernel. By default, the median euclidean
        distance between points is used.

    diag_noise : float or None, optional (default=None)
        Additional noise factor added to the diagonal in order to make the
        resulting Gram matrix positive definite.

    Returns
    -------
    k_ : array-like
        Resulting Gram matrix.
    """
    if y is None:
        y = x
    d = x - y[:, None]
    if sigma is None:
        sigma = np.median(d[d > 0])

    k_ = np.exp(- 1.0 * np.abs(d / sigma))
    if diag_noise is None:
        return k_

    assert diag_noise >= 0.0, f'noise >=0 required. Got: {diag_noise}!'
    np.fill_diagonal(k_, k_.diagonal() + diag_noise)
    return k_


def _rational_quadratic_kernel(x, y=None, alpha=1.0, sigma=None,
                               diag_noise=None):
    """
    Rational Quadratic kernel. Can be interpreted as a mixture of RBF kernels.
    Allows to add extra noise to the diagonal in order to make the resulting
    Gram matrix positive definite.
    The returned matrix is ``k(x, y)`` if ``y`` is provided, and ``k(x, x)``
    otherwise.

    Parameters
    ----------
    x : array-like, shape (n_samples,)
        Vector for which the kernel Gram matrix will be calculated.

    y : array-like or None, optional (default=None), shape (n_samples,)
        Optional, second vector for calculating the kernel Gram matrix.

    alpha : float, (default=1.0)
        Scale mixture parameter.

    sigma : float or None, optional (default=None)
        Bandwidth for the kernel. By default, the median euclidean
        distance between points is used.

    diag_noise : float or None, optional (default=None)
        Additional noise factor added to the diagonal in order to make the
        resulting Gram matrix positive definite.

    Returns
    -------
    k_ : array-like
        Resulting Gram matrix.
    """
    if y is None:
        y = x
    d = x - y[:, None]
    if sigma is None:
        sigma = np.median(d[d > 0])

    k_ = (1 + (d / sigma) ** 2 / (2.0 * alpha)) ** (-alpha)
    if diag_noise is None:
        return k_

    assert diag_noise >= 0.0, f'noise >=0 required. Got: {diag_noise}!'
    np.fill_diagonal(k_, k_.diagonal() + diag_noise)
    return k_


KERNEL_MAP = {
    'gaussian': _gaussian_kernel,
    'laplace': _laplace_kernel,
    'rational_quadratic': _rational_quadratic_kernel
}


def _hsic_kernel(k_x, k_y, scaled=False):
    """
    Calculate the HSIC score from the provided gram matrices prior created
    using a SPD kernel.

    Parameters
    ----------
    k_x : array-like, shape (n_samples, n_samples)
        The Gram matrix for the x variable.

    k_y : array-like, shape (n_samples, n_samples)
        The Gram matrix for the y variable.

    scaled : bool, optional (default=False)
        If true, the result is scaled (from 0 to 1) as

        .. math::
            result = hsic(x, y) / (hsic(x, x) * hsic(y, y)) ** 0.5

    Returns
    -------
    hsic_score : float
        HSIC score
    """
    k_x -= np.mean(k_x, axis=1)
    k_y -= np.mean(k_y, axis=1)

    trace_xy = np.einsum('ji,ij', k_x, k_y)
    if scaled:
        trace_xx = np.einsum('ji,ij', k_x, k_x)
        trace_yy = np.einsum('ji,ij', k_y, k_y)
        hsic_score = trace_xy / (trace_xx * trace_yy) ** 0.5
    else:
        hsic_score = trace_xy / (k_x.shape[0] - 1) ** 2
    return hsic_score


def hsic(x, y, scaled=False, sigma=None, kernel='gaussian'):
    """
    (Fairly efficient) implementation of HISC (Hilbert-Schmidt Independence
    Criterion):

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

    scaled : bool, optional (default=False)
        If true, the result is scaled (from 0 to 1) as

        .. math::
            result = hsic(x, y) / (hsic(x, x) * hsic(y, y)) ** 0.5

    sigma : float or None, optional (default=None)
        Bandwidth for the kernel. By default, the median euclidean
        distance between points is used.

    kernel : str, optional (default='gaussian')
        The type of kernel to be used. Choices are ``gaussian``, ``laplace``
        and ``rational_quadratic``.

    Returns
    -------
    hsic_score : float
        HSIC score

    Notes
    -----
        HSIC was first introduced in
        http://www.gatsby.ucl.ac.uk/~gretton/papers/GreBouSmoSch05.pdf
    """
    kernel_fn = KERNEL_MAP[kernel]
    k_x, k_y = kernel_fn(x, sigma=sigma), kernel_fn(y, sigma=sigma)
    hsic_score = _hsic_kernel(k_x, k_y, scaled)
    return hsic_score


def _hsic_naive(x, y, scaled=False, sigma=None, kernel='gaussian'):
    """
    Naive (slow) implementation of HISC (Hilbert-Schmidt Independence
    Criterion). This function is only used to assert correct results of the
    faster method ``hsic``.

    Parameters
    ----------
    x : array-like, shape (n_samples,)
        First vector to be compared.

    y : array-like, shape (n_samples,)
        Second vector to be compared.

    scaled : bool, optional (default=False)
        If true, the result is scaled (from 0 to 1) as

        .. math::
            result = hsic(x, y) / (hsic(x, x) * hsic(y, y)) ** 0.5

    sigma : float or None, optional (default=None)
        Bandwidth for the kernel. By default, the median euclidean
        distance between points is used.

    kernel : str, optional (default='gaussian')
        The type of kernel to be used. Choices are ``gaussian``, ``laplace``
        and ``rational_quadratic``.

    Returns
    -------
    hsic_score : float
        HSIC score
    """
    kernel_fn = KERNEL_MAP[kernel]

    m = x.shape[0]
    h = np.eye(m) - 1 / m
    k_x = kernel_fn(x, sigma=sigma) @ h
    k_y = kernel_fn(y, sigma=sigma) @ h
    trace_xy = np.sum(k_x.T * k_y)

    if scaled:
        trace_xx = np.sum(k_x.T * k_x)
        trace_yy = np.sum(k_y.T * k_y)
        hsic_score = trace_xy / (trace_xx * trace_yy) ** 0.5
    else:
        hsic_score = trace_xy / (1 - m) ** 2
    return hsic_score
