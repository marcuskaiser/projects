import numpy as np
from scipy.stats import entropy

from independence_tests.mi.utils import _check_xy, _get_bins, _mi
from independence_tests.utils import fast_rank


def var_info(x, y, bins=None, rank=False, norm=False):
    """
    Variation of Information
    https://en.wikipedia.org/wiki/Variation_of_information

    References
    - "de Prado, Marcos Lopez. Machine Learning for Asset Managers.
      Cambridge University Press, 2020."
    - Hacine-Gharbi and Ravier (2018)

    First creates a histogram of the data, and then applies a discrete
    estimators for mutual information and entropy terms.
    """
    x, y = _check_xy(x, y)

    if rank is True:
        x, y = fast_rank(x, y)

    bins = _get_bins(bins=bins, x=x, y=y)
    count_xy = np.lib.histogramdd([x, y], bins)[0]

    i_xy, count_x, count_y = _mi(count_xy=count_xy)
    h_x = entropy(count_x)
    h_y = entropy(count_y)

    v_xy = max(h_x + h_y - 2.0 * i_xy, 0.0)
    if norm:
        h_xy = max(h_x + h_y - i_xy, 1e-8)
        return v_xy / h_xy
    return v_xy
