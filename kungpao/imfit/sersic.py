from __future__ import division, print_function


import numpy as np
from scipy.special import gammaincinv, gamma

from .core import SERSIC_PARAMS
from ..utils import pixscale

__all__ = ['Sersic']


class Sersic(object):
    """
    Two dimensional Sersic surface brightness profile.

    Parameters
    ----------
    params : dict
        Sersic parameters. Uses imfit's convention:
        I_e, r_e, n, X0, Y0, ell, and PA
    zpt : float, optional
        Magnitude zero point.

    Notes
    -----
    - See Graham & Driver 2005 for a concise review of derived
      quantities of Sersic profiles.
      (http://adsabs.harvard.edu/abs/2005PASA...22..118G)
    - PA is in degrees and is defined with respect to the +y axis.
    - theta is in degrees and is defined with respect to the +x axis.
    """

    def __init__(self, params, zpt=27.0):
        """
        Initialize and calculate a bunch of useful quantities.
        """
        self.params = params
        self.I_e = params['I_e']
        self.r_e = params['r_e']
        self.n = params['n']
        self.X0 = params['X0']
        self.Y0 = params['Y0']
        self.PA = params['PA']
        self.ell = params['ell']
        self.q = 1 - self.ell
        self.theta = self.PA+90
        self.r_circ = self.r_e*np.sqrt(self.q)
        self.b_n = gammaincinv(2.*self.n, 0.5)
        self.mu_e = zpt - 2.5*np.log10(self.I_e/pixscale**2)
        self.mu_0 = self.mu_e - 2.5*self.b_n/np.log(10)
        f_n = gamma(2*self.n)*self.n*np.exp(self.b_n)/self.b_n**(2*self.n)
        self.mu_e_ave = self.mu_e - 2.5*np.log10(f_n)
        A_eff = np.pi*(self.r_circ*pixscale)**2
        self.m_tot = self.mu_e_ave - 2.5*np.log10(2*A_eff)
        if 'reduced_chisq' in list(params.keys()):
            self.reduced_chisq = params['reduced_chisq']
        for k, v in params.items():
            if 'err' in k:
                setattr(self, k, v)

    def __call__(self, x, y):
        """
        Evaluate Sersic.
        """
        a, b = self.r_e, (1 - self.ell) * self.r_e
        theta = self.theta*np.pi/180.0
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        x_maj = (x - self.X0) * cos_theta + (y - self.Y0) * sin_theta
        x_min = -(x - self.X0) * sin_theta + (y - self.Y0) * cos_theta
        z = np.sqrt((x_maj/a) ** 2 + (x_min/b) ** 2)
        img = self.I_e* np.exp(-self.b_n * (z ** (1/self.n) - 1))
        return img

    def array(self, shape, logscale=False):
        """
        Get 2D array of Sersic model.

        Parameters
        ----------
        shape : array-like
            Shape of output image.
        logscale : bool, optional
            If True, convert image to log scale.

        Returns
        -------
        img : 2D ndarray
            Model array with input shape.
        """
        y, x = np.indices(shape)
        img = self.__call__(x, y)
        return np.log10(img) if logscale else img
