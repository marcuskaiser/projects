from scipy.linalg import norm

from independence_tests.utils import fast_rank, pearson_p_value


def spearman(x, y):
    """
    Fast implementation of Spearman Correlation. Expects clean data.
    """
    # calculate mean for uniform distribution.
    # Note, we would like to use std = ((x.shape[0] ** 2 - 1) / 12) ** 0.5,
    # but for non-unique ranks, the score diverges. Therefore we only use
    # the analytic mean:
    mean = 0.5 * (x.shape[0] + 1.0)

    x_rank = fast_rank(x)
    x_rank -= mean
    y_rank = fast_rank(y)
    y_rank -= mean

    r = x_rank.dot(y_rank) / norm(x_rank) / norm(y_rank)
    r = min(1.0, max(0.0, r))
    p_value = pearson_p_value(r, x.shape[0])
    return r, p_value
