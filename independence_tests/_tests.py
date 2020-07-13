import time
import unittest

import numpy as np

from independence_tests.hsic import _hsic_naive, hsic
from independence_tests.hsic_cy import hsic as hsic_cy
from independence_tests.dcorr_cy import dcorr


def test_run(fn, n_samples=1000, lambda_=0.3, n_iter=10, seed=53):
    assert n_iter > 0
    assert n_samples > 0
    assert 0.0 <= lambda_ <= 1.0

    np.random.seed(seed)
    x = np.random.randn(n_samples)
    y = np.sin(1.0 * np.random.randn(n_samples) * lambda_
               + 2 * x * (1 - lambda_))
    result = None
    time_ = time.time()
    for i in range(n_iter):
        result = fn(x, y)
    time_ = time.time() - time_
    return result, time_


class TestHSIC(unittest.TestCase):
    def test_gaussian(self):
        hsic_0, t0 = test_run(lambda x, y: hsic(x, y, scaled=True,
                                                kernel='gaussian'))
        hsic_1, t1 = test_run(lambda x, y: _hsic_naive(x, y, scaled=True,
                                                       kernel='gaussian'))
        hsic_2, t2 = test_run(lambda x, y: hsic_cy(x, y, scale=True))

        assert np.allclose(hsic_0, hsic_1)
        assert np.allclose(hsic_0, hsic_2)
        assert t0 < t1, f'{t0} > {t1}!'
        print(t0, t1, t0 / t1)
        print(t2, t0, t2 / t0)

        hsic_0, t0 = test_run(lambda x, y: hsic(x, y, scaled=False,
                                                kernel='gaussian'))
        hsic_1, t1 = test_run(lambda x, y: _hsic_naive(x, y, scaled=False,
                                                       kernel='gaussian'))
        hsic_2, t2 = test_run(lambda x, y: hsic_cy(x, y, scale=False))

        assert np.allclose(hsic_0, hsic_1)
        assert np.allclose(hsic_0, hsic_2)
        assert t0 < t1, f'{t0} > {t1}!'
        print(t0, t1, t0 / t1)
        print(t2, t0, t2 / t0)

    def test_laplace(self):
        hsic_0, t0 = test_run(lambda x, y: hsic(x, y, scaled=True,
                                                kernel='laplace'))
        hsic_1, t1 = test_run(lambda x, y: _hsic_naive(x, y, scaled=True,
                                                       kernel='laplace'))
        assert np.allclose(hsic_0, hsic_1)
        assert t0 < t1, f'{t0} > {t1}!'
        print(t0, t1, t0 / t1)

        hsic_0, t0 = test_run(lambda x, y: hsic(x, y, scaled=False,
                                                kernel='laplace'))
        hsic_1, t1 = test_run(lambda x, y: _hsic_naive(x, y, scaled=False,
                                                       kernel='laplace'))
        assert np.allclose(hsic_0, hsic_1)
        assert t0 < t1, f'{t0} > {t1}!'
        print(t0, t1, t0 / t1)

    def test_rational_quadratic(self):
        hsic_0, t0 = test_run(lambda x, y: hsic(x, y, scaled=True,
                                                kernel='rational_quadratic'))
        hsic_1, t1 = test_run(lambda x, y: _hsic_naive(
            x, y, scaled=True, kernel='rational_quadratic'))
        assert np.allclose(hsic_0, hsic_1)
        assert t0 < t1, f'{t0} > {t1}!'
        print(t0, t1, t0 / t1)

        hsic_0, t0 = test_run(lambda x, y: hsic(x, y, scaled=False,
                                                kernel='rational_quadratic'))
        hsic_1, t1 = test_run(lambda x, y: _hsic_naive(
            x, y, scaled=False, kernel='rational_quadratic'))

        assert np.allclose(hsic_0, hsic_1)
        assert t0 < t1, f'{t0} > {t1}!'
        print(t0, t1, t0 / t1)


class TestDCorr(unittest.TestCase):
    def test(self):
        hsic_0, t0 = test_run(lambda x, y: dcorr(x, y, scale=True))
        print(hsic_0, t0)

        hsic_1, t1 = test_run(lambda x, y: dcorr(x, y, scale=False))
        print(hsic_1, t1)


if __name__ == '__main__':
    unittest.main()
