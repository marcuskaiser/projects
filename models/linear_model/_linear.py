from functools import partial

import numpy as np

from models.linear_model.utils import (_regularization_loss, _residual_loss_fn,
                                       lbfgs_fit, LOSS_TYPES, scale_std)


def _linear_loss_fn(w, x, y, loss_type, quantile, l1_w, l2_w):
    """
    Assemble loss function. Create L-BFGS objective using ``partial``.
    """
    w = w.reshape((x.shape[1], y.shape[1]))

    loss_, d_loss_ = _residual_loss_fn(x, y, w, loss_type, quantile)
    if l1_w > 0.0 or l2_w > 0.0:
        loss_w, d_loss_w = _regularization_loss(w, l1_w, l2_w)
        loss_ += loss_w
        d_loss_ += d_loss_w
    return loss_, d_loss_.ravel()


def fit_linear_lbfgs(x, y, loss_type='l2', quantile=None,
                     l1_w=0.0, l2_w=0.0, clip_weights=0.025, fit_bias=True,
                     scale=True, copy_xy=True):
    """
    Simple liner model fitting routine. This function is not particularly fast,
    but flexible in the sense that it can be used to fit a linear model with
    various loss functions and penalization terms.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        Feature Matrix.

    y : array-like, shape (n_samples, n_targets) or (n_samples,)
        Target Matrix.

    loss_type : str
        Loss function to be used for optimization.

    quantile : float or None, (default=None)
        Quantile in range (0, 1). Only used when ``loss_type = 'quantile'``.

    l1_w : float, optional (default=0.0)
        L1 regularization weight.

    l2_w : float, optional (default=0.0)
        L2 regularization weight.

    clip_weights : float, optional (default=0.025)
        If  `clip_weights > 0.0`, weights with
        ``abs(weight) < clip_weights * max(abs(weights))`` are set to zero.

    fit_bias : bool (default=True)
        If true, a bias term is fit.

    scale : bool (default=True)
        If true, the data is scaled to unit variance.

    copy_xy : bool (default=True)
        If true, the x and y arrays will be copied.

    Returns
    -------
    w_orig : array-like, shape (n_features, n_targets) or (n_features)
        Weights for linear model on original scale.

    w_star : array-like, shape (n_features, n_targets) or (n_features)
        Weights for linear model before scaling back to original scale.
        These weights can be used to e.g. calculate feature importances.

    bias : array-like, shape (n_targets,)
        Bias term.
    """
    # Check in puts and reshape data to 2d:
    assert loss_type.lower() in LOSS_TYPES, \
        f'Expected loss_type in {LOSS_TYPES}. Got: loss_type={loss_type}!'
    if loss_type == 'quantile':
        assert quantile is not None and 0.0 < quantile < 1.0, \
            f'Expected 0.0 < quantile < 1.0. Got: quantile={quantile}!'

    assert x.shape[0] == y.shape[0], f'{x.shape[0]} != {y.shape[0]}'
    assert x.ndim == 2

    if copy_xy is True:
        x = x.copy()
        y = y.copy()
    if y.ndim == 1:
        y = y[:, None]
    else:
        assert y.ndim == 2

    # Scale data and optionally adjust for bias term:
    x, x_std = scale_std(x, scale=scale)
    y, y_std = scale_std(y, scale=scale)
    if fit_bias is True:
        x = np.hstack((x, np.ones((x.shape[0], 1))))

    # Solve convex optimization problem:
    loss = partial(_linear_loss_fn, x=x, y=y, loss_type=loss_type,
                   quantile=quantile, l1_w=l1_w, l2_w=l2_w)
    w_init = np.zeros((x.shape[1], y.shape[1]))
    if fit_bias is True:
        w_init[-1, :] = y.mean(axis=0)
    w_star = lbfgs_fit(loss_fn=loss, w_init=w_init)

    # Adjust bias terms, weight clipping and re-scaling of weights:
    if fit_bias is True:
        bias = y_std * w_star[-1, :]
        w_star = w_star[:-1, :]
    else:
        bias = np.zeros(y.shape[1])

    if clip_weights > 0.0:
        w_max = np.abs(w_star).max()
        w_star[np.abs(w_star) < w_max * clip_weights] = 0.0
    w_orig = w_star * y_std / x_std[:, None]
    return w_orig, w_star, bias
