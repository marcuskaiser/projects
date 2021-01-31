from scipy.linalg import norm

from independence_tests.utils import pearson_p_value


def pearson(x, y):
    """
    Fast implementation of Pearson Correlation. Expects clean data.
    Implementation follows ideas from ``scipy.stats.pearsonr``.
    """
    x = x - x.mean()
    y = y - y.mean()

    r = x.dot(y) / norm(x) / norm(y)
    r = min(1.0, max(0.0, r))
    p = pearson_p_value(r, x.shape[0])
    return r, p
