import numpy as np

from models.linear_model.utils import LOSS_TYPES
from models.linear_model._linear import fit_linear_lbfgs


class LinearModel:
    """
    Fit a linear model with various loss functions and optional L1 and L2
    penalization. Based on L-BFGS routine.

    Parameters
    ----------
    loss_type : array-like, shape (n_samples,)
        Vector for which the kernel Gram matrix will be calculated.

    quantile : float or None, (default=None)
        Quantile in range (0, 1). Only used when ``loss_type='quantile'``.

    l1_w : float, optional (default=0.0)
        L1 regularization weight.

    l2_w : float, optional (default=0.0)
        L2 regularization weight.

    clip_weights : float, optional (default=0.025)
        If > 0.0, weights with
        ``abs(weight) < clip_weights * max(abs(weights))`` are set to zero.

    fit_bias : bool (default=True)
        If true, a bias term is fit.

    scale : bool (default=True)
        If true, the data is scaled to unit variance.
    """

    def __init__(self, loss_type='l2', quantile=None, l1_w=0.0, l2_w=0.0,
                 clip_weights=0.025, fit_bias=True, scale=True):
        assert loss_type.lower() in LOSS_TYPES, \
            f'Expected loss_type in {LOSS_TYPES}. Got: loss_type={loss_type}!'
        if loss_type == 'quantile':
            assert quantile is not None and 0.0 < quantile < 1.0, \
                f'Expected 0.0 < quantile < 1.0. Got: quantile={quantile}!'

        self._loss_type = loss_type
        self._l1_w = l1_w
        self._l2_w = l2_w
        self._clip_weights = clip_weights
        self._quantile = quantile
        self._fit_bias = fit_bias
        self._scale = scale

        self._weights = None
        self._weights_scaled = None
        self._bias = None

    def fit(self, x, y):
        """
        Fit Linear model.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Feature values used to fit the model.

        y : array-like, shape (n_samples, n_features) or (n_samples,)
            Target values used to fit the model.

        Returns
        -------
        self
        """
        self._weights, self._weights_scaled, self._bias = fit_linear_lbfgs(
            x=x, y=y, loss_type=self._loss_type, quantile=self._quantile,
            l1_w=self._l1_w, l2_w=self._l2_w, clip_weights=self._clip_weights,
            fit_bias=self._fit_bias, scale=self._scale)
        return self

    def predict(self, x):
        """
        Make prediction after fitting the model.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Features values for predictions.

        Returns
        -------
        out_ : array-like, shape (n_samples, n_targets) or (n_samples,)
            Predictions from the model.
        """
        assert self._weights is not None, 'Call .fit() first!'
        out_ = x @ self._weights + self._bias
        return out_

    @property
    def weights(self):
        assert self._weights is not None, 'Call .fit() first!'
        return self._weights

    @property
    def bias(self):
        assert self._bias is not None, 'Call .fit() first!'
        return self._bias

    @property
    def feature_importance(self):
        """
        Returns the feature importance equal to the absolute value of the
        scaled weights (the weights returned by the linear model before the
        (optional) scaling is inverted).
        """
        assert self._weights is not None, 'Call .fit() first!'
        return np.abs(self._weights_scaled)
