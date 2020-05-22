import unittest

import numpy as np

from mgcv import MGCV


class Test1(unittest.TestCase):
    def setUp(self):
        n_train_samples, n_test_samples = 1000, 100
        n_dims = 10

        np.random.seed(113)
        x_train = np.random.normal(size=(n_train_samples, n_dims))
        self.y_train = x_train[:, 0] ** 3 + np.random.normal(
            size=x_train.shape[0])
        self.x_train = x_train
        self.x_test = np.random.normal(size=(n_test_samples, n_dims))

    def test_splines(self):
        mgcv = MGCV().fit(self.x_train, self.y_train)
        details = mgcv.get_details()
        self.assertEqual(details['formula'],
                         'y ~ s(x1) + s(x2) + s(x3) + s(x4) + s(x5) '
                         '+ s(x6) + s(x7) + s(x8) + s(x9) + s(x10)')

        for formula_ in ['y ~ x1 + s(x1)',
                         'y ~ ti(x1) + ti(x2)',
                         'y ~ t2(x1) + s(x2)',
                         'y ~ s(x1, x2, x3)',
                         'y ~ s(x1, x2) + ti(x3)']:
            mgcv = MGCV(formula=formula_).fit(self.x_train, self.y_train)
            details = mgcv.get_details()
            self.assertEqual(details['formula'], formula_)

            formula_ = details['formula'].split('~')[1].replace(' ', '')
            formula_parts = formula_.split('+')
            for fp in formula_parts:
                if 'Intercept' in fp:
                    continue
                if fp.startswith('s') or fp.startswith('ti') \
                        or fp.startswith('t2'):
                    assert fp in details['smooth_terms']['edf']
                else:
                    assert fp in details['parametric_terms']['Estimate']

    def test_splies_similarity(self):
        mgcv1 = MGCV(formula='y ~ s(x1, bs="tp")').fit(self.x_train,
                                                       self.y_train)
        mgcv2 = MGCV(formula='y ~ s(x1)').fit(self.x_train, self.y_train)
        self.assertEqual(
            np.linalg.norm(mgcv1.predict(self.x_test) -
                           mgcv1.predict(self.x_test)), 0.0)
        self.assertEqual(
            np.linalg.norm(mgcv1.predict(self.x_test) -
                           mgcv2.predict(self.x_test)), 0.0)

        mgcv3 = MGCV(formula='y ~ s(x1, bs="ps")').fit(self.x_train,
                                                       self.y_train)
        self.assertLess(
            np.abs(mgcv1.predict(self.x_test) -
                   mgcv3.predict(self.x_test)).mean(), 0.06)

        mgcv4 = MGCV(formula='y ~ s(x1, bs="ad")').fit(self.x_train,
                                                       self.y_train)
        self.assertLess(
            np.abs(mgcv1.predict(self.x_test) -
                   mgcv4.predict(self.x_test)).mean(), 0.06)

    def test_consistency(self):
        mgcv1 = MGCV(formula='y ~ s(x1)').fit(self.x_train, self.y_train)
        mgcv2 = MGCV(formula='y ~ s(x1)').fit(self.x_train, self.y_train)
        self.assertEqual(
            np.linalg.norm(mgcv1.predict(self.x_test) -
                           mgcv1.predict(self.x_test)), 0.0)
        self.assertEqual(
            np.linalg.norm(mgcv1.predict(self.x_test) -
                           mgcv2.predict(self.x_test)), 0.0)

        mgcv3 = MGCV(formula='y ~ s(x1)',
                     gam_type='bam').fit(self.x_train, self.y_train)
        self.assertLess(
            np.linalg.norm(mgcv1.predict(self.x_test) -
                           mgcv3.predict(self.x_test)), 0.02)

    def test(self):
        x_train = np.linspace(-5, 5, 20)
        x_test = np.linspace(-6, 6, 1000)

        def f(x):
            return 0.01 * x ** 3 + np.log(np.abs(x))

        y_train = f(x_train)
        y_test = f(x_test)

        x_train = x_train.reshape(-1, 1)
        x_test = x_test.reshape(-1, 1)

        mgcv = MGCV().fit(x_train, y_train)
        self.assertLess(np.abs(mgcv.predict(x_test) - y_test).mean(), 0.15)
        self.assertIsInstance(mgcv.get_details(), dict)


if __name__ == '__main__':
    unittest.main()
