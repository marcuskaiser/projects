import numpy as np
from scipy.optimize import minimize

LOSS_TYPES = ['l1', 'median', 'mae', 'l2', 'mse', 'quantile', 'q']


def _residual_loss_fn(x, y, w, loss_type, quantile):
    res = y - x @ w
    if loss_type.lower() in ['l2', 'mse']:
        loss_ = np.mean(res ** 2) / 2
        d_loss_ = - x.T @ res / res.shape[0]
    elif loss_type.lower() in ['l1', 'median', 'mae']:
        loss_ = np.abs(res).mean()
        d_loss_ = - x.T @ np.sign(res) / res.shape[0]
    elif loss_type.lower() in ['quantile', 'q']:
        quantiles = np.where(res < 0, quantile - 1, quantile)
        loss_ = (quantiles * res).mean()
        d_loss_ = - x.T @ quantiles / res.shape[0]
    else:
        raise ValueError(f'Unknown loss_type=`{quantile}`!')
    return loss_, d_loss_


def _regularization_loss(w, l1_w, l2_w):
    loss_ = 0.0
    d_loss_ = np.zeros_like(w)
    if l1_w > 0.0:
        loss_ += l1_w * np.abs(w).sum()
        d_loss_ += l1_w * np.sign(w)
    if l2_w > 0.0:
        loss_ += l2_w * (w ** 2).sum() / 2
        d_loss_ += l2_w * w
    return loss_, d_loss_


def scale_std(x, scale=True):
    """ Scale data by its standard deviation. """
    if scale is True:
        x_std = x.std(axis=0) + 1e-12
        x /= x_std
    else:
        x_std = np.ones(x.shape[1])
    return x, x_std


def check_and_prepare(x, y, loss_type, quantile, scale, fit_bias, copy_xy):
    """ Check in puts and reshape data to 2d. """
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
    return x, x_std, y, y_std


def lbfgs_fit(loss_fn, w_init, grad=True):
    """ Simple wrapper for L-BFGS scipy routine. """
    original_shape = w_init.shape
    w_init = w_init.ravel()
    w_star = minimize(fun=loss_fn, x0=w_init, method='L-BFGS-B', jac=grad).x
    w_star = w_star.reshape(original_shape)
    return w_star


def postprocess_parameters(w_star, x_std, y_std, fit_bias, clip_weights):
    """ Adjust bias terms, weight clipping and re-scaling of weights. """
    if fit_bias is True:
        bias = y_std * w_star[-1, :]
        w_star = w_star[:-1, :]
    else:
        bias = np.zeros(w_star.shape[1])
    if clip_weights > 0.0:
        w_max = np.abs(w_star).max()
        w_star[np.abs(w_star) < w_max * clip_weights] = 0.0
    w_orig = w_star * y_std / x_std[:, None]
    return bias, w_orig, w_star
