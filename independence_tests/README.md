# Fairly efficient implementation of HSIC

Exact python implementation of the **Hilbert-Schmidt Independence Criterion (HSIC)**.
This implementation is fairly efficient (at least 30% faster than typical implementations).

The result can be scaled, in which case the returned quantity is

```math
HSIC(x, y) / (HSIC(x, x) * HSIC(y, y))^{1/2}
```

which ranges from 0.0 (independent) to 1.0 (highly correlated / identical).


## Example

```python
from independence_tests import hsic

x = ...
y = ...

out1 = hsic(x, y, scaled=True, kernel='gaussian')

out2 = hsic(x, y, scaled=True, kernel='laplace')

out3 = hsic(x, y, scaled=True, kernel='rational_quadratic')
```

## Details

Consider two vectors`x` and `y` and let `K` and `L` be their corresponding (kernel) Gram matrices. Further, define the centering matrix `H = I - m^{-1} 11^T`. Then the (empirical estimator for the) HSIC criterion is given by

```math
HSIC(x, y) = (m-1)^{-2} trace(KHLH)
```

## Reference

- http://www.gatsby.ucl.ac.uk/~gretton/papers/GreBouSmoSch05.pdf