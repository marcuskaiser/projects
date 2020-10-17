from scipy.linalg import norm
from scipy.special._ufuncs import btdtr


def pearson(x, y):
    """
    Fast implementation of Pearson Correlation. Expects clean data.
    Implementation follows ideas from ``scipy.stats.pearsonr``.
    """
    x_ = x - x.mean()
    y_ = y - y.mean()

    r = x_.dot(y_) / norm(x_) / norm(y_)
    r = max(min(r, 1.0), -1.0)

    # Calculate beta distribution with both parameters equal to n / 2 - 1:
    shape_param = 0.5 * x.shape[0] - 1.0
    p = 2 * btdtr(shape_param, shape_param, 0.5 * (1.0 - abs(r)))
    return r, p
