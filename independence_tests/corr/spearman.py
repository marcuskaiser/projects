import numpy as np

from independence_tests.corr.pearson import pearson


def fast_rank(x, y):
    """ Fast calculation of ranks, as used for e.g. Spearman Correlation. """
    arg_sort = np.empty_like(x, dtype=np.int_)

    arg_sort[:] = x.argsort()
    x_arg_rev = np.empty_like(arg_sort)
    x_arg_rev[arg_sort] = np.arange(len(x))

    arg_sort[:] = y.argsort()
    y_arg_rev = np.empty_like(arg_sort)
    y_arg_rev[arg_sort] = np.arange(len(y))
    return x_arg_rev, y_arg_rev


def spearman(x, y):
    """ Fast implementation of Spearman Correlation. Expects clean data. """
    x_rank, y_rank = fast_rank(x, y)
    return pearson(x_rank, y_rank)
