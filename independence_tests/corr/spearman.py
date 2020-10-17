import numpy as np

from independence_tests.corr.pearson import pearson


def spearman(x, y):
    """ Fast implementation of Spearman Correlation. Expects clean data. """
    arg_sort = np.empty_like(x, dtype=np.int_)

    arg_sort[:] = x.argsort()
    x_arg_rev = np.empty_like(arg_sort)
    x_arg_rev[arg_sort] = np.arange(len(x))

    arg_sort[:] = y.argsort()
    y_arg_rev = np.empty_like(arg_sort)
    y_arg_rev[arg_sort] = np.arange(len(y))

    return pearson(x_arg_rev, y_arg_rev)
