import unittest

import numpy as np
from scipy.stats import random_correlation
from scipy.linalg import norm, inv

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
    def test_svgd_median(self):
        n_dims = 50
        n_samples = 100

        fn_grad, mu, precision_matrix = get_normal(n_dims=n_dims, seed=111)
        mu[:4] = np.array([-0.6871, 3.8010, 13.0, 3.0])
        sigma = inv(precision_matrix).diagonal()

        np.random.seed(17)
        theta0 = np.random.normal(0, 1, [n_samples, n_dims])
        theta1 = SVGD(objective_grad=fn_grad, eta=0.05,
                      bandwidth_heuristic='median').run(theta0, n_iter=3000)

        theta_mu = np.mean(theta1, axis=0)
        theta_var = np.var(theta1, axis=0)
        print('\nerror:', norm(mu - theta_mu) / mu.shape[0], norm(theta_var - sigma) / sigma.shape[0])
        print('n_samples:', theta1.shape[0])
        print('svgd_adam:\n', theta_mu, theta_var)
        print('ground truth:\n', mu, sigma)

    def test_svgd_mean(self):
        n_dims = 50
        n_samples = 200

        fn_grad, mu, precision_matrix = get_normal(n_dims=n_dims, seed=111)
        mu[:4] = np.array([-0.6871, 3.8010, 13.0, 3.0])
        sigma = inv(precision_matrix).diagonal()

        np.random.seed(17)
        theta0 = np.random.normal(0, 1, [n_samples, n_dims])
        theta1 = SVGD(objective_grad=fn_grad, eta=0.05,
                      bandwidth_heuristic='mean').run(theta0, n_iter=3000)

        theta_mu = np.mean(theta1, axis=0)
        theta_var = np.var(theta1, axis=0)
        print('\nerror:', norm(mu - theta_mu) / mu.shape[0], norm(theta_var - sigma) / sigma.shape[0])
        print('n_samples:', theta1.shape[0])
        print('svgd_adam:\n', theta_mu, theta_var)
        print('ground truth:\n', mu, sigma)

    def test_svgd_mean_median(self):
        n_dims = 50
        n_samples = 100

        fn_grad, mu, precision_matrix = get_normal(n_dims=n_dims, seed=111)
        mu[:4] = np.array([-0.6871, 3.8010, 13.0, 3.0])
        sigma = inv(precision_matrix).diagonal()

        np.random.seed(17)
        theta0 = np.random.normal(0, 1, [n_samples, n_dims])
        theta1 = SVGD(objective_grad=fn_grad, eta=0.05,
                      bandwidth_heuristic='mean').run(theta0, n_iter=2500)

        theta2 = SVGD(objective_grad=fn_grad, eta=0.05,
                      bandwidth_heuristic='median').run(theta1, n_iter=500)

        theta_mu = np.mean(theta2, axis=0)
        theta_var = np.var(theta2, axis=0)
        print('\nerror:', norm(mu - theta_mu) / mu.shape[0], norm(theta_var - sigma) / sigma.shape[0])
        print('n_samples:', theta2.shape[0])
        print('svgd_adam:\n', theta_mu, theta_var)
        print('ground truth:\n', mu, sigma)

    def test_svgd_mean_double(self):
        n_dims = 50
        n_samples = 20

        fn_grad, mu, precision_matrix = get_normal(n_dims=n_dims, seed=111)
        mu[:4] = np.array([-0.6871, 3.8010, 13.0, 3.0])
        sigma = inv(precision_matrix).diagonal()

        np.random.seed(17)
        theta0 = np.random.normal(0, 1, [n_samples, n_dims])
        theta1 = SVGD(objective_grad=fn_grad, eta=0.05,
                      bandwidth_heuristic='mean').run(theta0, n_iter=1000)
        for n_iter in [500, 500]:
            theta1 = np.vstack([theta1,
                                theta1 + np.random.normal(scale=0.1, size=theta1.shape),
                                theta1 + np.random.normal(scale=0.1, size=theta1.shape)
                                ])
            theta1 = SVGD(objective_grad=fn_grad, eta=0.05,
                          bandwidth_heuristic='mean').run(theta1, n_iter=n_iter)

        theta_mu = np.mean(theta1, axis=0)
        theta_var = np.var(theta1, axis=0)
        print('\nerror:', norm(mu - theta_mu) / mu.shape[0], norm(theta_var - sigma) / sigma.shape[0])
        print('n_samples:', theta1.shape[0])
        print('svgd_adam:\n', theta_mu, theta_var)
        print('ground truth:\n', mu, sigma)


if __name__ == '__main__':
    unittest.main()
