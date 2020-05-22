import numpy as np
import pandas as pd
import rpy2.robjects
import rpy2.robjects.packages
import rpy2.robjects.pandas2ri

rpy2.robjects.pandas2ri.activate()
rpy2.robjects.packages.importr('mgcv')

FIT_GAM = rpy2.robjects.r("""
    function(formula, df, gam_type='gam', ...) {
      if(gam_type == 'gam') {
        gam(formula=as.formula(formula), data=df, ...)
      } else if(gam_type == 'bam') {
        bam(formula=as.formula(formula), data=df, ...)
      }
    }
    """)
PREDICT_GAM = rpy2.robjects.r('predict')
SUMMARY_GAM = rpy2.robjects.r("""
    function(gam) {
      summary_ <- summary(gam)
      list(p=data.frame(summary_$p.table),
           s=data.frame(summary_$s.table),
           coefs=data.frame(coef(gam)),
           vcov=data.frame(vcov(gam))
      )
    }
    """)


class MGCV:
    def __init__(self, formula=None, gam_type='gam'):
        """
        Wrapper for the R package ``mgcv`` for fitting
        Generalized Additive Models (GAMs).

        Parameters
        ----------
        formula : str or None, optional (default=None)
            An optional string formula. For example, a simple
            linear regression with 3 covariates (features) can be specified as

            >>> 'y ~ x1 + x2 + x3'

            By default (if no formula is specified), the formula

            >>> 'y ~ s(x1) + s(x2) + s(x3)'

            is generated. This implies that we instead apply smooths based on
            the individual variables ``x1``, ``x2`` and ``x3``.
            A mix of linear and smooth terms can e.g. be specified via

            >>> 'y ~ x1 + s(x2) + s(x3)'

            An interaction term between the variabels ``x1`` and ``x2`` can
            be included via ``s(x1, x2)``:

            >>> 'y ~ s(x1, x2) + s(x3)'

            The default smooth corresponds to a thin spate spline, which can
            explictily be defined as ``s(x1, bs="ps")``. There are multiple
            alternatives for the smooth terms, such as p-splines via
            ``s(x1, bs="ps")`` and adaptive smoothers ``s(x1, bs="ad")``.
            For full details (including tensor smooths and Gaussian Processes),
            see the references below.

        gam_type : {'gam', 'bam'}, optional (default='gam')
            The gam type to be fitted. ``bam` is more memory efficient, and
            can be faster on large data sets (>> 10k data points).

        Notes
        -----
        References:

        1) https://cran.r-project.org/web/packages/mgcv/
        2) https://doi.org/10.1201/9781315370279
        3) https://people.maths.bris.ac.uk/~sw15190/mgcv/smooth-toolbox.pdf
        4) https://stat.ethz.ch/R-manual/R-patched/library/mgcv/html/smooth.terms.html
        """
        self._formula = formula
        if self._formula is not None:
            if not self._formula.replace(' ', '').startswith('y~'):
                raise ValueError('formula has to be of the form '
                                 '`y ~ f(x1) + ...`. Got: {formula}')
        self._gam_type = gam_type
        assert self._gam_type in ['gam', 'bam']

        self._feature_names = None
        self._gam = None

    @property
    def formula(self):
        return self._formula

    def reset(self):
        self._feature_names = None
        self._gam = None

    def fit(self, x, y):
        """
        Fit Method.

        Parameters
        __________
        x : array-like, shape=(n_samples, n_features)
            Features matrix to be used for training.

        y : array_like, shape=(n_samples,)
            Target vector to be used for training.

        Returns
        -------
        self : object
        """
        assert x.ndim == 2, (
            f'Dimension mismatch: Got x.shape={x.shape}! '
            f'x has to be 2-dimensional with shape (n_samples, n_features)!')
        assert y.ndim == 1, (
            f'Dimension mismatch: Got y.shape={y.shape}! '
            'y has to be 1-dimensional: (n_samples,)! '
            '[Hint: You can modify your y vector using y.ravel()!]')

        self._feature_names = [f'x{i + 1}' for i in range(x.shape[1])]

        if self._formula is None:
            features_ = ' + '.join([f's({x_})' for x_ in self._feature_names])
            self._formula = 'y ~ ' + features_

        df_ = pd.DataFrame(np.concatenate([x, y.reshape(-1, 1)], axis=1),
                           columns=self._feature_names + ['y'])
        self._gam = FIT_GAM(formula=self._formula,
                            gam_type=self._gam_type, df=df_)
        return self

    def predict(self, x):
        """
        Prediction method.

        Parameters
        __________
        x : array-like, shape=(n_samples, n_features)
            Features matrix for which to make predictions.

        Returns
        -------
        y_pred : array-like
            Predicted values
        """
        assert x.ndim == 2, (
            f'Dimension mismatch: Got x.shape={x.shape}! '
            f'x has to be 2-dimensional with shape (n_samples, n_features)!')
        assert x.shape[1] == len(self._feature_names), (
            f'Dimension mismatch: Got x.shape={x.shape}! '
            f'Second dimension has to coincide with n_features='
            f'{len(self._feature_names)}!')
        assert self._gam is not None, (
            f'{self.__class__.__name__} has not been trained yet. '
            f'Call {self.__class__.__name__}.fit() first!')

        df_ = pd.DataFrame(x, columns=self._feature_names)
        y_pred = PREDICT_GAM(self._gam, df_)
        return y_pred

    def _get_summary(self):
        summary_ = SUMMARY_GAM(self._gam)
        return dict(zip(summary_.names, list(summary_)))

    def get_details(self):
        assert self._gam is not None, (
            f'{self.__class__.__name__} has not been trained yet. '
            f'Call {self.__class__.__name__}.fit() first!')
        summary = self._get_summary()
        details = {
            'formula': self._formula,
            'parametric_terms': summary['p'].to_dict(),
            'smooth_terms': summary['s'].to_dict(),
            'coefs': summary['coefs'].to_dict(),
            'vcov': summary['vcov'].to_dict(),
        }
        return details
