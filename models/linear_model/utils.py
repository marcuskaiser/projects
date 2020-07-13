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


def lbfgs_fit(loss_fn, w_init, grad=True):
    """ Simple wrapper for L-BFGS scipy routine. """
    original_shape = w_init.shape
    w_init = w_init.ravel()
    w_star = minimize(fun=loss_fn, x0=w_init, method='L-BFGS-B', jac=grad).x
    w_star = w_star.reshape(original_shape)
    return w_star
