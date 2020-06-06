# Wrapper for ``mgcv``

This code is a (simple) ``Python`` wrapper for the ``R`` package ``mgcv``, for fitting Generalized Additive Models (GAMs). The wrapper follows the sklearn API.

## Details

In order for this code to work, an existing ``R`` version has to be installed
(from [CRAN](https://cran.r-project.org/)).  
Calls from ``Python`` into ``R`` are based on ``rpy2`` (see [here](https://github.com/rpy2/rpy2)).

The code has been tested with ``R`` version 3.6.1 and ``rpy2`` version 3.2.6.

## Examples


``` python

from mgcv import MGCV

# Data has to be in the format x_train.shape = (n_samples, n_features) and y_train.shape = (n_samples,)
x_train = ...
y_train = ...

mgcv = MGCV()
mgcv.fit(x_train, y_train)

# A brief summary of the model fit in terms of a dictionary can be obtained via:
detail = mgcv.get_details()
print(details)

# Prediction on new data can be made as follows:
x_test = ...
mgcv.predict(x_test)
```


Simple example of how the interaction terms can be used:

``` python 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import ARDRegression

from mgcv import MGCV

# Create synthetic data:
n_samples = 1000
x_1 = np.log(np.random.exponential(size=n_samples))
y = 0.3 * np.random.normal(size=n_samples) + x_1
x_2 = np.sin(np.linspace(0, 100, n_samples))
x_1 = (1 + x_1) * x_2

x = np.zeros(shape=(n_samples, 2))
x[:, 0] = x_1
x[:, 1] = x_2

# Split data:
x_train, x_test = x[:400], x[400:]
y_train, y_test = y[:400], y[400:]

def test_run(model):
    model.fit(x_train, y_train)
    y_hat = model.predict(x_test)
    plt.plot(y_hat, alpha=0.8)
    plt.plot(y_test, alpha=0.5)
    plt.show()
    print((np.abs(y_hat - y_test)).mean())

# Baseline: ARDRegression
test_run(ARDRegression())

# MGCV with independent variables:
test_run(MGCV())

# MGCV with interaction term:
test_run(MGCV(formula='y~s(x1,x2)'))
```

## References

- https://cran.r-project.org/web/packages/mgcv/
- https://doi.org/10.1201/9781315370279
- https://people.maths.bris.ac.uk/~sw15190/mgcv/smooth-toolbox.pdf
- https://stat.ethz.ch/R-manual/R-patched/library/mgcv/html/smooth.terms.html
