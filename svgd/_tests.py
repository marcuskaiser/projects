import time
import unittest

import numpy as np
from scipy.stats import random_correlation

from svgd import SVGD


def get_normal(n_dims=10, seed=111):
    np.random.seed(seed)
    eig = np.random.rand(n_dims)
    eig *= n_dims / eig.sum()
    precision_matrix = 0.1 * random_correlation.rvs(eig)
    mu = np.zeros(n_dims)
    fn_grad = lambda theta: - (theta - mu) @ precision_matrix
    return fn_grad, mu, precision_matrix


class TestSVGD(unittest.TestCase):

    def test_svgd(self):
        n_dims = 10
        n_samples = 300

        fn_grad, mu, precision_matrix = get_normal(n_dims=n_dims, seed=111)
        mu[:4] = np.array([-0.6871, 3.8010, 13.0, 3.0])

        t0 = time.time()

        theta0 = np.random.normal(0, 1, [n_samples, n_dims])
        theta1 = SVGD(objective_grad=fn_grad, eta=0.05).run(theta0, n_iter=3000)
        print("\nsvgd_adam:\n", np.mean(theta1, axis=0), np.var(theta1, axis=0))
        print("ground truth:\n", mu, np.linalg.inv(precision_matrix).diagonal())

        print(time.time() - t0)


if __name__ == '__main__':
    unittest.main()
