import unittest
from functools import partial
import time

import numpy as np
from sklearn.linear_model import Lasso

from models.linear_model import LinearModel
from models.linear_model._linear import _linear_loss_fn, fit_linear_lbfgs
from models.linear_model.utils import lbfgs_fit, LOSS_TYPES


def compare_grad(x, y, loss_type, quantile=None, l1_w=0.0, l2_w=0.0):
    w_init = np.zeros((x.shape[1], y.shape[1]))
    loss = partial(_linear_loss_fn, x=x, y=y, loss_type=loss_type,
                   quantile=quantile, l1_w=l1_w, l2_w=l2_w)
    w_star = lbfgs_fit(loss_fn=loss, w_init=w_init)

    loss_no_grad = lambda w: loss(w)[0]
    w_no_grad = lbfgs_fit(loss_fn=loss_no_grad, w_init=w_init, grad=False)
    np.testing.assert_array_less(np.abs(w_star - w_no_grad).mean(), 1e-3)


class TestLowLevel(unittest.TestCase):
    def test_grad(self):
        np.random.seed(51465)
        n = 1000
        x_ = np.random.normal(size=(n, 4))
        y_ = x_[:, 0] + 0.3 * x_[:, 3]
        y_ = y_.reshape(-1, 1)
        compare_grad(x_, y_, loss_type='l1')
        compare_grad(x_, y_, loss_type='l1', l1_w=0.03)
        compare_grad(x_, y_, loss_type='l1', l2_w=0.025)
        compare_grad(x_, y_, loss_type='l1', l1_w=0.01, l2_w=0.025)
        compare_grad(x_, y_, loss_type='l2')
        compare_grad(x_, y_, loss_type='l2', l1_w=0.03)
        compare_grad(x_, y_, loss_type='l2', l2_w=0.025)
        compare_grad(x_, y_, loss_type='l2', l1_w=0.01, l2_w=0.025)

        compare_grad(x_, y_, loss_type='q', quantile=0.3)
        compare_grad(x_, y_, loss_type='q', quantile=0.3, l1_w=0.03)
        compare_grad(x_, y_, loss_type='q', quantile=0.3, l2_w=0.025)
        compare_grad(x_, y_, loss_type='q', quantile=0.3, l1_w=0.01,
                     l2_w=0.025)
        compare_grad(x_, y_, loss_type='q', quantile=0.45)
        compare_grad(x_, y_, loss_type='q', quantile=0.45, l1_w=0.03)
        compare_grad(x_, y_, loss_type='q', quantile=0.45, l2_w=0.025)
        compare_grad(x_, y_, loss_type='q', quantile=0.45, l1_w=0.01,
                     l2_w=0.025)
        compare_grad(x_, y_, loss_type='q', quantile=0.89)
        compare_grad(x_, y_, loss_type='q', quantile=0.89, l1_w=0.03)
        compare_grad(x_, y_, loss_type='q', quantile=0.89, l2_w=0.025)
        compare_grad(x_, y_, loss_type='q', quantile=0.89, l1_w=0.01,
                     l2_w=0.025)

    def test_insample(self):
        np.random.seed(51465)
        n = 1000
        x_ = np.random.normal(size=(n, 4))
        y_ = x_[:, 0] + 0.3 * x_[:, 3]
        truth = np.zeros((4, 1))
        truth[0] = 1.0
        truth[3] = 0.3

        rtol = 1e-4
        np.testing.assert_allclose(
            fit_linear_lbfgs(x_, y_, loss_type='l1')[0], truth, rtol=1e-5)
        np.testing.assert_allclose(
            fit_linear_lbfgs(x_, y_, loss_type='l2')[0], truth, rtol=1e-5)
        np.testing.assert_allclose(
            fit_linear_lbfgs(x_, y_, loss_type='l1', scale=False)[0],
            truth, rtol=rtol)
        np.testing.assert_allclose(
            fit_linear_lbfgs(x_, y_, loss_type='l2', scale=False)[0],
            truth, rtol=rtol)
        np.testing.assert_allclose(
            fit_linear_lbfgs(x_, y_, loss_type='quantile', quantile=0.2,
                             scale=False)[0], truth, rtol=rtol)
        np.testing.assert_allclose(
            fit_linear_lbfgs(x_, y_, loss_type='quantile', quantile=0.2,
                             scale=False)[0], truth, rtol=rtol)
        np.testing.assert_allclose(
            fit_linear_lbfgs(x_, y_, loss_type='quantile', quantile=0.5,
                             scale=False)[0], truth, rtol=rtol)
        np.testing.assert_allclose(
            fit_linear_lbfgs(x_, y_, loss_type='quantile', quantile=0.5,
                             scale=False)[0], truth, rtol=rtol)
        np.testing.assert_allclose(
            fit_linear_lbfgs(x_, y_, loss_type='quantile', quantile=0.8,
                             scale=False)[0], truth, rtol=rtol)
        np.testing.assert_allclose(
            fit_linear_lbfgs(x_, y_, loss_type='quantile', quantile=0.8,
                             scale=False)[0], truth, rtol=rtol)

    def test_bias(self):
        np.random.seed(4509)
        n = 100
        x_ = np.random.normal(size=(n, 4))
        y_ = x_[:, 0] + 0.3 * x_[:, 3] + 0.5

        rtol = 1e-4
        np.testing.assert_allclose(
            fit_linear_lbfgs(x_, y_, loss_type='l1', scale=False)[0],
            np.array([[1.0, 0.0, 0.0, 0.3]]).T, rtol=rtol)
        np.testing.assert_allclose(
            fit_linear_lbfgs(x_, y_, loss_type='l1', scale=False)[2],
            np.array([0.5]), rtol=rtol)

        np.testing.assert_allclose(
            fit_linear_lbfgs(x_, y_, loss_type='l1', scale=False)[0],
            fit_linear_lbfgs(x_, y_, loss_type='l1')[0], rtol=rtol)
        np.testing.assert_allclose(
            fit_linear_lbfgs(x_, y_, loss_type='l2', scale=False)[0],
            fit_linear_lbfgs(x_, y_, loss_type='l2')[0], rtol=rtol)
        np.testing.assert_allclose(
            fit_linear_lbfgs(x_, y_, 'quantile', quantile=0.3, scale=False)[0],
            fit_linear_lbfgs(x_, y_, 'quantile', quantile=0.3)[0], rtol=rtol)
        np.testing.assert_allclose(
            fit_linear_lbfgs(x_, y_, 'quantile', quantile=0.7, scale=False)[0],
            fit_linear_lbfgs(x_, y_, 'quantile', quantile=0.7)[0], rtol=rtol)
        np.testing.assert_allclose(
            fit_linear_lbfgs(x_, y_, loss_type='l2', scale=False)[0],
            fit_linear_lbfgs(x_, y_, loss_type='l2')[0], rtol=rtol)

    def test_qr(self):
        np.random.seed(4509)
        n = 120
        x_ = np.random.normal(size=(n, 4))
        y_ = x_[:, 0] + 0.3 * x_[:, 3] - 1.5 + 0.6 * np.random.normal(size=n)

        last_percentage = 0.0
        for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            w, _, bias = fit_linear_lbfgs(x_[:80], y_[:80],
                                          loss_type='quantile', quantile=q)
            new_percentage = (x_[80:] @ w + bias > y_[80:]).mean()
            self.assertLess(last_percentage, new_percentage)
            last_percentage = new_percentage

        last_percentage = 0.0
        for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            w, _, bias = fit_linear_lbfgs(x_[:80], y_[:80],
                                          loss_type='quantile', quantile=q,
                                          l1_w=0.03, l2_w=0.05)
            new_percentage = (x_[80:] @ w + bias > y_[80:]).mean()
            self.assertLess(last_percentage, new_percentage)
            last_percentage = new_percentage

        last_percentage = 0.0
        for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            w, _, bias = fit_linear_lbfgs(x_[:80], y_[:80],
                                          loss_type='quantile', quantile=q,
                                          l1_w=0.01, l2_w=0.05)
            new_percentage = (x_[80:] @ w + bias > y_[80:]).mean()
            self.assertLess(last_percentage, new_percentage)
            last_percentage = new_percentage

    def test_multiple_targets(self):
        np.random.seed(4509)
        n = 100
        x_ = np.random.normal(size=(n, 4))
        y_ = np.random.normal(size=(n, 2))
        y_[:, 0] += x_[:, 0] + 0.3 * x_[:, 3] + 0.5
        y_[:, 1] += x_[:, 0] + 0.3 * x_[:, 3] + 0.5
        fit_2 = fit_linear_lbfgs(x_, y_, loss_type='l2', scale=False)[0]
        fit_1_0 = fit_linear_lbfgs(x_, y_[:, 0], loss_type='l2', scale=False)[0]
        fit_1_1 = fit_linear_lbfgs(x_, y_[:, 1], loss_type='l2', scale=False)[0]

        rtol = 5 * 1e-3
        np.testing.assert_allclose(fit_1_0.ravel(), fit_2[:, 0], rtol=rtol)
        np.testing.assert_allclose(fit_1_1.ravel(), fit_2[:, 1], rtol=rtol)

        fit_3 = fit_linear_lbfgs(x_, y_, loss_type='l2', scale=True)[0]


class TestHighLevel(unittest.TestCase):
    def test_hl_comparison(self):
        np.random.seed(4509)
        n = 100
        x_ = np.random.normal(size=(n, 4))
        y_ = x_[:, 0] + 0.3 * x_[:, 3] + 0.5 * np.random.normal(size=n)

        rtol = 1e-4
        for scale in [True, False]:
            for loss in LOSS_TYPES:
                np.testing.assert_allclose(
                    LinearModel(loss_type=loss, quantile=0.3,
                                scale=scale).fit(x_, y_)._weights,
                    fit_linear_lbfgs(x_, y_, loss_type=loss, scale=scale,
                                     quantile=0.3)[0],
                    rtol=rtol)

        np.testing.assert_allclose(
            LinearModel(loss_type='quantile',
                        quantile=0.3).fit(x_, y_)._weights,
            fit_linear_lbfgs(x_, y_, loss_type='quantile', quantile=0.3)[0],
            rtol=rtol)
        np.testing.assert_allclose(
            LinearModel(loss_type='quantile',
                        quantile=0.7).fit(x_, y_)._weights,
            fit_linear_lbfgs(x_, y_, loss_type='quantile', quantile=0.7)[0],
            rtol=rtol)
        np.testing.assert_allclose(
            LinearModel(loss_type='l2', scale=False).fit(x_, y_)._weights,
            fit_linear_lbfgs(x_, y_, loss_type='l2')[0], rtol=rtol)

        np.testing.assert_allclose(
            LinearModel(loss_type='l2', scale=False).fit(x_, y_)._bias,
            fit_linear_lbfgs(x_, y_, loss_type='l2')[2], rtol=rtol)

    def test_hl_large(self):
        np.random.seed(4509)
        n = 1_000_000
        x_ = np.random.normal(size=(n, 100))
        y_ = x_[:, 0] + 0.3 * x_[:, 3] + 0.5 * np.random.normal(size=n)
        y_ += x_[:, 5] * (0.3 + x_[:, 6]) * 0.3
        y_ += 0.03 * x_[:, 10]

        t0 = time.time()
        linear_model = LinearModel(loss_type='l2', l1_w=0.01, scale=True).fit(x_, y_)
        print('time:', time.time() - t0)
        print(linear_model._weights.ravel())

        t0 = time.time()
        lm = Lasso(alpha=0.1).fit(x_, y_)
        print('time:', time.time() - t0)
        print(lm.coef_)


if __name__ == '__main__':
    unittest.main()
