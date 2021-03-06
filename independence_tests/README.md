# Independence tests

## Summary

- Hilbert-Schmidt Independence Criterion (HSIC)
- Distance Correlation (dcorr)
- Fast Conditional Independence Test (FCIT)

Note: Cython code can be compiled with
```shell script
python setup.py build_ext
```

## Hilbert-Schmidt Independence Criterion

Exact python implementation of the **Hilbert-Schmidt Independence Criterion (HSIC)**.
This implementation is fairly efficient (at least 30% faster than typical implementations).

The result can be scaled, in which case the returned quantity is

```math
HSIC(x, y) / (HSIC(x, x) * HSIC(y, y))^{1/2}
```

which ranges from 0.0 (independent) to 1.0 (highly correlated / identical).


### Example

The Cython version can be used as follows:

```python
from independence_tests import hsic

x = ...
y = ...

out1 = hsic(x, y, scale=True)
```

The pure Python version allows the use of three different kernels:

```python
from independence_tests.hsic import hsic

x = ...
y = ...

out1 = hsic(x, y, scale=True, kernel='gaussian')

out2 = hsic(x, y, scale=True, kernel='laplace')

out3 = hsic(x, y, scale=True, kernel='rational_quadratic')
```

### Details

Consider two vectors`x` and `y` and let `K` and `L` be their corresponding (kernel) Gram matrices. Further, define the centering matrix `H = I - m^{-1} 11^T`. Then the (empirical estimator for the) HSIC criterion is given by

```math
HSIC(x, y) = (m-1)^{-2} trace(KHLH)
```

### Reference

- http://www.gatsby.ucl.ac.uk/~gretton/papers/GreBouSmoSch05.pdf


## Distance Correlation

```python
from independence_tests import dcorr

x = ...
y = ...

out1 = dcorr(x, y, scale=True)
```

### Reference

- https://en.wikipedia.org/wiki/Distance_correlation


## Fast Conditional Independence Test

Adaption of the Fast Conditional Independence Test (FCIT) using random forests.
We call this implementation ``rfcit``.
The test has two different modes. When only ``y`` and ``x`` are provided, we test if
``x`` can be used to predict ``y``. This is done via comparing the prediction of ``y`` using ``x`` to
the prediction of ``y`` using a random shuffle of ``x``. We then apply a one-sided t-test to identify if the performance
for the individual trees improved. In order to improve robustness, this procedure is iterated three times and
the resulting p-values are averaged.

```python
from independence_tests import rfcit

x = ...
y = ...

out1 = rfcit(y=y, x=x)
```

If apart from ``x`` and ``y`` also ``z`` is provided, we compare if using ``x`` together with ``z`` improves the
prediction of ``y`` over just using ``z`` for predicting ``y``. In other words, we ask if ``[x, z]`` is better
than just ``z`` at predicting ``y``:

```python
from independence_tests import rfcit

x = ...
y = ...
z = ...

out1 = rfcit(y=y, x=x, z=z)
````

### Reference

- https://arxiv.org/abs/1804.02747