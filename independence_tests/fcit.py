import math

import numpy as np
from scipy.stats import ttest_1samp
from sklearn.ensemble import RandomForestRegressor


def rfcit(y, x, z=None, train_amount=0.3, seed=14, n_reps=3, **fit_args):
    """
    Adaption of the Fast Conditional Independence test (FCIT) using Random
    Forests. Based on ideas from https://arxiv.org/abs/1804.02747
    and the associated code https://github.com/kjchalup/fcit.

    If ``z=None``, then we test if ``x`` can be used to predict ``y` by
    comparing how well ``x`` predicts ``y`` to a random permutation of ``x``.
    The results are evaluated with a one-sided t-test. This experiment is
    repeated ``n_reps`` times and the resulting p-values are averaged.

    If ``z`` is a numpy array with features, then we compare if ``x`` contains
    additional information on ``y`` given ``z``, by comparing how well
    ``[x, z]`` predicts ``y`` vs. just using ``z`` for the prediction of ``y`.

    Parameters
    ----------
    y : array-like, shape (n_samples, n_features)
        Target to be predicted

    x : array-like, shape (n_samples, n_features)
        Additional variables to be checked for predictive power of ``y``.

    z : array-like, shape (n_samples, n_features)
        Variables we condition on for prediction of ``y``.

    train_amount : float (default=0.3)
        Percentage of data to be used for training of the model. THe first
        train_amount percent of data are used for training.

    seed : int (default=14)
        Random seed for model fitting.

    n_reps : int (default=3)
        Number of repetitions when ``z=None``.

    fit_args : kwargs
        Additional keyword arguments to be passed to the RandomForestRegressor
        instance.
    """
    assert x.shape[0] == y.shape[0]
    n_samples_train = int(x.shape[0] * train_amount)

    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    y = y - y.mean(axis=0)
    y /= (y ** 2).mean(axis=0) ** 0.5

    if z is None:
        random = np.random.default_rng(seed=seed)
        p_val = 0.0

        maes = _fit_rf_medae(xs=x,
                             y=y,
                             n_samples_train=n_samples_train,
                             **fit_args)

        for _ in range(n_reps):
            maes2 = _fit_rf_medae(xs=x[random.permutation(x.shape[0])],
                                  y=y,
                                  n_samples_train=n_samples_train,
                                  **fit_args)
            p_val += _one_sided_t_test(scores=maes2 - maes)

        # Return the average of the estimated p-values:
        return p_val / n_reps

    else:
        assert x.shape[0] == z.shape[0]
        if z.ndim == 1:
            z = z[:, None]

        maes = _fit_rf_medae(xs=np.hstack((x, z)),
                             y=y,
                             n_samples_train=n_samples_train,
                             **fit_args)
        maes2 = _fit_rf_medae(xs=z,
                              y=y,
                              n_samples_train=n_samples_train,
                              **fit_args)
        return _one_sided_t_test(scores=maes2 - maes)


def _one_sided_t_test(scores):
    """
    One-sided t-test for the scores to be zero,
    as in https://github.com/kjchalup/fcit.
    """
    t, p_value = ttest_1samp(scores, 0.0)
    if t < 0.0:
        return 1 - 0.5 * p_value
    return 0.5 * p_value


def _fit_rf_medae(xs, y, n_samples_train=None, **fit_args):
    if n_samples_train is None:
        n_samples_train = xs.shape[0] // 2
    if y.shape[1] == 1:
        y = y.ravel()

    _fit_args = {
        'n_estimators': 30,
        'criterion': 'mae',
        'max_features': 'auto',
        'min_samples_split': max(2, int(math.log(y.shape[0]))),
        'max_depth': 3,
        'random_state': 15,
    }
    _fit_args.update(fit_args)

    model = RandomForestRegressor(**_fit_args)
    model.fit(xs[:n_samples_train], y[:n_samples_train])

    # Return array with the median errors for each tree in the RF:
    return np.array([
        np.median(np.abs(e.predict(xs[n_samples_train:]) - y[n_samples_train:]))
        for e in model.estimators_
    ])


if __name__ == '__main__':
    r = np.random.default_rng(seed=10)
    n_samples = 200

    z_ = r.normal(size=n_samples)
    x = 0.3 * r.normal(size=n_samples)
    y = 0.5 * (1 + np.sin(2.0 * np.pi * np.arange(n_samples))) * r.normal(size=n_samples)

    # y depends on z_
    y[z_ < 0.1] += 0.3 * z_[z_ < 0.1] ** 2.0

    # x depends on z_
    x[np.abs(z_) < 0.3] += z_[np.abs(z_) < 0.3]

    # y and x depend on latent
    latent = r.normal(size=n_samples)
    y[latent < 1.6] += latent[latent < 1.6]
    x[latent > 0.3] += latent[latent > 0.3] * 0.3

    print(rfcit(y, x))
    print(rfcit(y, z_))
    print(rfcit(y, x, z_))
    print(rfcit(y, z_, x))
