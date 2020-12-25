from scipy.linalg import norm
from scipy.special._ufuncs import btdtr


def _p_value(r, num_samples):
    r = max(min(r, 1.0), -1.0)

    # Calculate beta distribution with both parameters equal to n / 2 - 1:
    shape_param = 0.5 * num_samples - 1.0
    p = 2 * btdtr(shape_param, shape_param, 0.5 * (1.0 - abs(r)))
    return p


def pearson(x, y):
    """
    Fast implementation of Pearson Correlation. Expects clean data.
    Implementation follows ideas from ``scipy.stats.pearsonr``.
    """
    x = x - x.mean()
    y = y - y.mean()

    r = x.dot(y) / norm(x) / norm(y)
    p = _p_value(r, x.shape[0])
    return r, p
