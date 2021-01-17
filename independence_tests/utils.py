import numpy as np
from scipy.special._ufuncs import btdtr


def fast_rank(arr, kind='quicksort'):
    """
    Following ``scipy.stats.rankdata``.
    """
    sorter = np.argsort(arr, kind=kind)
    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size, dtype=np.intp)

    arr = arr[sorter]
    obs = np.r_[True, arr[1:] != arr[:-1]]

    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], len(obs)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def pearson_p_value(r, num_samples):
    """ P-value for Pearson R. """
    r = max(min(r, 1.0), -1.0)

    # Calculate beta distribution with both parameters equal to n / 2 - 1:
    shape_param = 0.5 * num_samples - 1.0
    p = 2 * btdtr(shape_param, shape_param, 0.5 * (1.0 - abs(r)))
    return p
