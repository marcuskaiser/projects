from math import log
import numpy as np

from independence_tests.corr import pearson


def _check_xy(x, y):
    assert x.shape[0] == y.shape[0]
    if x.ndim > 1:
        assert x.shape[1] == 1
        x = x.ravel()
    if y.ndim > 1:
        assert y.shape[1] == 1
        y = y.ravel()
    return x, y


def _get_bins(bins, x, y):
    """
    If bins is not set, estimate them according to
    Hacine-Gharbi and Ravier (2018);
    cf. "de Prado, Marcos Lopez. Machine Learning for Asset Managers.
    Cambridge University Press, 2020."
    """
    if bins is None:
        corr = pearson(x, y)[0]
        if abs(corr) < 1.0:
            # optimal number of bins (Hacine-Gharbi and Ravier (2018))
            ratio_ = (24.0 * x.shape[0]) / (1.0 - corr ** 2)
            bins_ = (0.5 * (1.0 + (1.0 + ratio_) ** 0.5)) ** 0.5
            bins = int(round(bins_))
        else:
            bins = 10
    return bins


def _mi(count_xy):
    """
    Following sklearn.metrics.mutual_info_score
    """
    count_x = count_xy.sum(axis=1)
    count_y = count_xy.sum(axis=0)
    count_total = count_x.sum()

    nz_x, nz_y = np.nonzero(count_xy)
    nz_values = count_xy[nz_x, nz_y]
    outer = count_x[nz_x] * count_y[nz_y]

    mi = nz_values * (np.log(nz_values) - np.log(outer) + log(count_total))
    return mi.sum() / count_total, count_x, count_y
