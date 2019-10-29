"""Place to keep track all the components."""

import numpy as np

from scipy.optimize import brent
from mpmath import gamma as Gamma
from mpmath import gammainc as GammaInc

__all__ = ['Sersic', 'b_n_exact']


def Sersic(R, n, I_e, r_e):
    """Compute intensity at radius r for a Sersic profile, given the specified
    """
    return I_e * np.exp(-1 * b_n_exact(n) * (pow(R / r_e, 1.0 / n) - 1.0))


def b_n_exact(n):
    """Exact calculation of the Sersic derived parameter b_n, via solution
    of the function
            Gamma(2n) = 2 gamma_inc(2n, b_n)
    where Gamma = Gamma function and gamma_inc = lower incomplete gamma function.
    If n is a list or Numpy array, the return value is a 1-d Numpy array
    """
    def myfunc(bn, n):
        return abs(float(2 * GammaInc(2*n, 0, bn) - Gamma(2*n)))

    if np.iterable(n):
        b = [brent(myfunc, (nn,)) for nn in n]
        b = np.array(b)
    else:
        b = brent(myfunc, (n,))
    return b
