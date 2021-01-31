import time
import unittest

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr, t

from independence_tests import (pearson, spearman, dcorr, hsic as hsic_cy,
                                kendall, mutual_information)
from independence_tests.hsic.hsic import _hsic_naive, hsic as hsic_py


def test_run(fn, n_samples=1000, lambda_=0.3, n_iter=10, seed=53):
    assert n_iter > 0
    assert n_samples > 0
    assert 0.0 <= lambda_ <= 1.0

    np.random.seed(seed)
    x = np.random.randn(n_samples)
    y = np.sin(1.0 * np.random.randn(n_samples) * lambda_
               + 2 * x * (1.0 - lambda_))
    result = None
    time_ = time.time()
    for i in range(n_iter):
        result = fn(x, y)
    time_ = time.time() - time_
    return result, time_


class TestHSIC(unittest.TestCase):
    def test_gaussian(self):
        (hsic_0, p0), t0 = test_run(lambda x, y: hsic_py(x, y, scale=True,
                                                         kernel='gaussian'))
        hsic_1, t1 = test_run(lambda x, y: _hsic_naive(x, y, scale=True,
                                                       kernel='gaussian'))
        (hsic_2, p2), t2 = test_run(lambda x, y: hsic_cy(x, y, scale=True))

        self.assertAlmostEqual(hsic_0, hsic_1)
        self.assertAlmostEqual(hsic_0, hsic_2)
        self.assertAlmostEqual(p0, p2)
        self.assertLess(t0, t1)
        print(t0, t1, t0 / t1)
        print(t2, t0, t2 / t0)

        (hsic_0, p0), t0 = test_run(lambda x, y: hsic_py(x, y, scale=False,
                                                         kernel='gaussian'))
        hsic_1, t1 = test_run(lambda x, y: _hsic_naive(x, y, scale=False,
                                                       kernel='gaussian'))
        (hsic_2, p2), t2 = test_run(lambda x, y: hsic_cy(x, y, scale=False))

        self.assertAlmostEqual(hsic_0, hsic_1)
        self.assertAlmostEqual(hsic_0, hsic_2)
        self.assertAlmostEqual(p0, p2)
        self.assertLess(t0, t1)
        print(t0, t1, t0 / t1)
        print(t2, t0, t2 / t0)

    def test_laplace(self):
        (hsic_0, p0), t0 = test_run(lambda x, y: hsic_py(x, y, scale=True,
                                                         kernel='laplace'))
        hsic_1, t1 = test_run(lambda x, y: _hsic_naive(x, y, scale=True,
                                                       kernel='laplace'))
        self.assertAlmostEqual(hsic_0, hsic_1)
        self.assertLess(t0, t1)
        print(t0, t1, t0 / t1)

        (hsic_0, p0), t0 = test_run(lambda x, y: hsic_py(x, y, scale=False,
                                                         kernel='laplace'))
        hsic_1, t1 = test_run(lambda x, y: _hsic_naive(x, y, scale=False,
                                                       kernel='laplace'))
        self.assertAlmostEqual(hsic_0, hsic_1)
        self.assertLess(t0, t1)
        print(t0, t1, t0 / t1)

    def test_rational_quadratic(self):
        (hsic_0, p0), t0 = test_run(lambda x, y: hsic_py(
            x, y, scale=True, kernel='rational_quadratic'))
        hsic_1, t1 = test_run(lambda x, y: _hsic_naive(
            x, y, scale=True, kernel='rational_quadratic'))
        self.assertAlmostEqual(hsic_0, hsic_1)
        self.assertLess(t0, t1)
        print(t0, t1, t0 / t1)

        (hsic_0, p0), t0 = test_run(lambda x, y: hsic_py(
            x, y, scale=False, kernel='rational_quadratic'))
        hsic_1, t1 = test_run(lambda x, y: _hsic_naive(
            x, y, scale=False, kernel='rational_quadratic'))

        self.assertAlmostEqual(hsic_0, hsic_1)
        self.assertLess(t0, t1)
        print(t0, t1, t0 / t1)


class TestDCorr(unittest.TestCase):
    def test(self):
        dcorr_0, t0 = test_run(lambda x, y: dcorr(x, y, scale=True))
        print(dcorr_0, t0)

        dcorr_1, t1 = test_run(lambda x, y: dcorr(x, y, scale=False))
        print(dcorr_1, t1)


class TestCorr(unittest.TestCase):
    def test(self):
        n_iter = 100
        n_samples = [10, 20, 50, 100, 200, 300, 400, 500, 800, 1000,
                     2000, 3000, 5000, 8000, 10000, 20000, 50000, 100000]

        for n in n_samples:
            x = t.rvs(df=2, size=(n, 2))
            x_ = x[:, 0]
            y_ = x[:, 1] + np.sign(x[:, 0]) * np.abs(x[:, 0]) ** 1.3

            np.testing.assert_allclose(pearson(x_, y_),
                                       tuple(pearsonr(x_, y_)))
            np.testing.assert_allclose(spearman(x_, y_),
                                       tuple(spearmanr(x_, y_)))
            np.testing.assert_allclose(kendall(x_, y_),
                                       kendalltau(x_, y_)[0])

            t0 = time.time()
            for i in range(n_iter):
                spearman(x_, y_)
            t1 = (time.time() - t0) / n_iter

            t0 = time.time()
            for i in range(n_iter):
                spearmanr(x_, y_)
            t2 = (time.time() - t0) / n_iter

            self.assertLess(t1, t2)

            if n > 20:
                x[:10, :] = 0.0

                np.testing.assert_allclose(pearson(x_, y_),
                                           tuple(pearsonr(x_, y_)))
                np.testing.assert_allclose(spearman(x_, y_),
                                           tuple(spearmanr(x_, y_)))


if __name__ == '__main__':
    unittest.main()
