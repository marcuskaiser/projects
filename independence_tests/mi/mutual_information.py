import numpy as np

from independence_tests.mi.utils import _check_xy, _get_bins, _mi
from independence_tests.utils import fast_rank


def mutual_information(x, y, bins=None, rank=False):
    """
    Estimate of Mutual Information by discretizing the data into bins.
    First creates a histogram of the data, and then applies a discrete
    estimator for mutual information.
    """
    x, y = _check_xy(x, y)

    if rank is True:
        x, y = fast_rank(x, y)

    bins = _get_bins(bins=bins, x=x, y=y)
    count_xy = np.lib.histogramdd([x, y], bins)[0]
    return _mi(count_xy=count_xy)[0]
