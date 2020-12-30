import numpy as np

from scipy.spatial.distance import pdist, squareform


def _kernel_grad_inplace(x, grad, heuristic='mean'):
    """
    Calculate SVGD update direction (gradient) :math:`\\phi^{\\star}`.
    Based on RBF kernel. Modifies grad.

    Parameters
    ----------
    x : array-like, shape=(n_samples, n_parameters)
        Initial estimate for the distribution.

    grad: array-like, shape=(n_samples, n_parameters)
        Gradient of the objective function.

    heuristic: str, optional (default='mean')
        Heuristic for the kernel. Can be one of `mean` and `median`.
    """
    if x.shape[0] == 1:
        return grad
    d2 = pdist(x, metric='sqeuclidean')
    if heuristic == 'median':
        sigma_sq = np.median(d2)
    elif heuristic == 'mean':
        sigma_sq = np.mean(d2)
    else:
        raise ValueError(f'Unknown heuristic=`{heuristic}`!')

    k_xy = np.exp(-squareform(d2 / sigma_sq))
    grad[:] = k_xy @ (grad - x / sigma_sq)
    grad[:] += k_xy.sum(axis=1, keepdims=True) * x / sigma_sq
    grad[:] /= x.shape[0]


def _adam_update_step_inplace(i, grad, m, v, beta1=0.9, beta2=0.999, eps=1e-8):
    """ Adam update routine with inplace updates. Modifies grad, m and v. """
    m[:] = beta1 * m + (1.0 - beta1) * grad
    v[:] = beta2 * v + (1.0 - beta2) * grad ** 2
    m_adj = m / (1.0 - beta1 ** (i + 1))
    v_adj = v / (1.0 - beta2 ** (i + 1))
    grad[:] = m_adj / (np.sqrt(v_adj) + eps)


def svgd_adam(x_init,
              objective_grad,
              n_iter,
              bandwidth_heuristic='mean',
              eta=1e-3,
              tol=1e-5,
              beta1=0.9,
              beta2=0.999):
    x = x_init.copy()
    m, v = np.zeros_like(x), np.zeros_like(x)

    for i in range(n_iter):
        grad = objective_grad(x)
        _kernel_grad_inplace(
            x=x, grad=grad, heuristic=bandwidth_heuristic)
        _adam_update_step_inplace(
            i=i, grad=grad, m=m, v=v, beta1=beta1, beta2=beta2)

        if np.abs(grad).mean() < tol:
            break
        x += eta * grad
    return x


class SVGD:
    """
    Implementation of the **Stein Variational Gradient Descent** algorithm
    using the Adam Optimizer.

    Parameters
    ----------
    objective_grad : callable
        Gradient of the objective function.

    bandwidth_heuristic: str, optional (default='mean')
        One of 'median' or 'mean' for calculating the bandwidth of the
        Gaussian kernel.

    eta : float, optional (default=1e-3)
        Hyperparameter for optimizer - Learning rate.

    tol : float, optional (default=1e-5)
        Hyperparameter for optimizer - Parameter for early stopping.
        Early stopping is applied when the mean absolute difference between
        two steps is less than this threshold.

    beta1 : float, optional (default=0.9)
        Hyperparameter for Adam optimizer - Decay rate for mean.
        Smaller value means faster decay.

    beta2 : float, optional (default=0.999)
        Hyperparameter for Adam optimizer - Decay rate for variance.
        Smaller value means faster decay.

    Notes
    -----
    References:

    1) https://papers.nips.cc/paper/6338-stein-variational-gradient-descent-a-general-purpose-bayesian-inference-algorithm
    2) http://papers.nips.cc/paper/6904-stein-variational-gradient-descent-as-gradient-flow
    3) http://www.cs.utexas.edu/~qlearning/project.html?p=svgd
    4) https://github.com/dilinwang820/Stein-Variational-Gradient-Descent
    5) http://proceedings.mlr.press/v48/chwialkowski16
    """

    def __init__(self, objective_grad,
                 bandwidth_heuristic='mean',
                 eta=1e-3,
                 tol=1e-5,
                 beta1=0.9,
                 beta2=0.999):
        assert eta > 0.0
        assert tol > 0.0
        assert 0.0 < beta1 < 1.0
        assert 0.0 < beta2 < 1.0

        self._objective_grad = objective_grad
        self._bandwidth_heuristic = bandwidth_heuristic
        self._eta = eta
        self._tol = tol
        self._beta1 = beta1
        self._beta2 = beta2

    def run(self, x_init, n_iter=1000):
        """
        Main method to run the algorithm.

        Parameters
        ----------
        x_init : array-like, shape=(n_samples, n_parameters)
            Initial estimate for the distribution.

        n_iter : int, optional (default=1000)
            Number of iterations to perform.

        Returns
        -------
        x : array-like
            Estimate of the solution.
        """
        assert n_iter > 0
        return svgd_adam(x_init=x_init,
                         objective_grad=self._objective_grad,
                         n_iter=n_iter,
                         tol=self._tol,
                         eta=self._eta,
                         beta1=self._beta1,
                         beta2=self._beta2,
                         bandwidth_heuristic=self._bandwidth_heuristic)
