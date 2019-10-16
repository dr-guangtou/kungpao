"""Module to deal with model parameters"""
from __future__ import print_function, division, unicode_literals

from copy import deepcopy

import numpy as np

from kungpao.model import priors

__all__ = ['ProfileParams']


param_template = {'name': '', 'ini': 0.5, 'sig': 1.0, 'min': 0.0, 'max': 1.0,
                  'type': 'flat', 'label': ''}


class ProfileParams(object):
    """Object to deal with model parameters.

    Based on the `parameters` module from `prospector`:

        https://github.com/bd-j/prospector/blob/master/prospect/models/parameters.py

    """
    def __init__(self, cfg):
        """Constructor.

        Parameters
        ----------
        cfg : dict
            Configuration for model parameters.
        """
        self.cfg_ini = deepcopy(cfg)

        # Check if the parameters are valid
        self.check()

        # Assign prior distributins
        self.assign_prior_distribution()

        # Number of parameters
        self.n_param = len(self.cfg_ini.items())

        # Name of the parameters
        self._names = [param['name'] for _, param in self.cfg_ini.items()]

        # Type of the parameters
        self._types = [param['type'] for _, param in self.cfg_ini.items()]

        # Labels used for plots
        self._labels = [param['label'] for _, param in self.cfg_ini.items()]

        # Distributions of priors
        self._distr = [param['distr'] for _, param in self.cfg_ini.items()]

    def check(self):
        """Check if the parameter is in the right format"""
        for label, param in self.cfg_ini.items():
            if ('name' not in param) or ('type' not in param):
                raise Exception("# Need parameter name or prior type!")
            if 'label' not in param:
                param['label'] = label

            # Check the ini and sig for parameters with Student-T distribution
            if (param['type'] == 'student') and ('ini' not in param or 'sig' not in param):
                raise Exception("# You need both ini and sig for Student T prior !")
            # Check the min and max for parameters with TopHat distribution
            if (param['type'] == 'flat') and ('min' not in param or 'max' not in param):
                raise Exception("# You need both min and max for TopHat prior !")

            # Sanity checks
            if 'sig' in param and param['sig'] <= 0:
                raise Exception("# Dispersion parameter sig can not be zero or negative !")

            if 'min' in param and 'max' in param and param['min'] >= param['max']:
                raise Exception("# Min needs to be samller than Max !")

    def assign_prior_distribution(self):
        """Configure model parameters.
        Parameters
        ----------
        cfg : dict
            Configurations for the model parameters.

        Return
        ------
            Dictionary that contains necessary information of model parameters.

        """
        for label in self.cfg_ini:
            param = self.cfg_ini[label]
            if param['type'] == 'flat':
                param['distr'] = priors.TopHat(low=param['min'], upp=param['max'])
            elif param['type'] == 'student':
                param['distr'] = priors.StudentT(loc=param['ini'], scale=param['sig'], df=1)
            else:
                raise Exception("# Wrong type of prior distribution: [flat|student]")

    def sample(self, nsamples=1):
        """Sampling the prior distributions

        Parameters
        ----------
        nsamples : int, optional
            Number of samples to return. Default: 1

        Return
        ------
        samples : array with the shape (nsamples, n_param)
            Random samples following the prior distributions.

        """
        return np.asarray([distr.sample(nsample=nsamples) for distr in self._distr]).T

    def perturb(self, nsamples=1, level=0.05):
        """Generate samples of parameters using small perturbation around the mean.

        Parameters
        ----------
        nsamples : int, optional
            Number of samples to return. Default: 1
        level : float, optional
            Percents of perturbation

        Return
        ------
        samples : array with the shape (nsamples, n_param)
            Random samples following the prior distributions.

        """
        return np.asarray(
            [self.transform(np.random.randn(self.n_param) * level + 0.5) for i in range(nsamples)])

    def lnprior(self, theta, nested=False):
        """Public version of _ln_prior.

        Parameters
        ----------
        theta : list or array
            A set of parameter values.
        nested : bool, optional
            True is using nested sampling instead. Will return 0 for any finite ln(prior).

        Return
        ------
        ln_prior : float
            Sum of the ln(prior)

        """
        lnp = self._lnprior(theta)

        if nested and np.isfinite(lnp):
            return 0.0

        return lnp

    def transform(self, theta_cube):
        """Go from unit cube to parameter space, for nested sampling.

        Parameters
        ----------
        theta_cuba : list or array
            A set of parameter values in the unit hyper-cube.
        nested : bool, optional
            True is using nested sampling instead. Will return 0 for any finite ln(prior).

        Return
        ------
        theta : array
            Parameters values.

        """
        return np.array([d.unit_transform(u) for (u, d) in zip(theta_cube, self._distr)])

    def _lnprior(self, theta):
        """Return a scalar which is the ln of the product of the prior
        probabilities for each element of theta.

        Parameters
        ----------
        theta : list or array
            A set of parameter values.

        Return
        ------
        ln_prior : float
            Sum of the ln(prior)

        """
        return np.sum([d.lnp(p) for (p, d) in zip(theta, self._distr)])

    def get_ini(self):
        """Return an array of initial values for parameters"""
        return np.array([distr.get_mean()for distr in self._distr])

    def get_low(self):
        """Get the lower boundary of the parameters"""
        return np.array([distr.low for distr in self._distr])

    def get_upp(self):
        """Get the upper boundary of the parameters"""
        return np.array([distr.upp for distr in self._distr])

    @property
    def names(self):
        """Parameter names used in the model."""
        return self._names

    @property
    def labels(self):
        """Parameter labels used for plots."""
        return self._labels

    @property
    def types(self):
        """Types of priors for parameters."""
        return self._types

    @property
    def distr(self):
        """Distributions of priors for parameters."""
        return self._distr

    @property
    def ini(self):
        """Mean values of the prior distributions."""
        return self.get_ini()

    @property
    def low(self):
        """Lower limits of parameters."""
        return self.get_low()

    @property
    def upp(self):
        """Upper limits of parameters."""
        return self.get_upp()
