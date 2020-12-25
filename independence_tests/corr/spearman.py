import numpy as np
from scipy.linalg import norm

from independence_tests.corr.pearson import _p_value


def fast_rank(x, y):
    """ Fast calculation of ranks, as used for e.g. Spearman Correlation. """
    arg_sort = np.empty_like(x, dtype=np.int_)

    arg_sort[:] = x.argsort()
    x_arg_rev = np.empty_like(arg_sort, dtype=np.float_)
    x_arg_rev[arg_sort] = np.arange(len(x))

    arg_sort[:] = y.argsort()
    y_arg_rev = np.empty_like(arg_sort, dtype=np.float_)
    y_arg_rev[arg_sort] = np.arange(len(y))
    return x_arg_rev, y_arg_rev


def spearman(x, y):
    """
    Fast implementation of Spearman Correlation. Expects clean data.
    Note: This does not handle repeated data points. The results diverge
    from the "classical" ``scipy.stats.spearmanr`` implementation with
    averaged ranks for multiple occurrences of variable values
    (e.g. constant arrays).
    """
    x_rank, y_rank = fast_rank(x, y)

    # calculate mean and standard-deviation for uniform distribution:
    mean = (x_rank.shape[0] - 1.0) / 2.0
    std = ((x_rank.shape[0] ** 2 - 1) / 12) ** 0.5

    x_rank -= mean
    x_rank /= std
    y_rank -= mean
    y_rank /= std

    r = x_rank.dot(y_rank) / x.shape[0]
    p_value = _p_value(r, x.shape[0])
    return r, p_value
