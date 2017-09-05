"""
Random utilities
"""
from __future__ import division, print_function

import os
import numpy as np

import yaml

from astropy.table import Table
from collections import namedtuple


def get_time_label():
    """
    Return time label for new files & directories.

    From: https://github.com/johnnygreco/hugs/blob/master/hugs/utils.py
    """
    import time

    label = time.strftime("%Y%m%d-%H%M%S")

    return label


def check_random_state(seed):
    """
    Turn seed into a `numpy.random.RandomState` instance.
    Parameters
    ----------
    seed : `None`, int, list of ints, or `numpy.random.RandomState`
        If ``seed`` is `None`, return the `~numpy.random.RandomState`
        singleton used by ``numpy.random``.  If ``seed`` is an `int`,
        return a new `~numpy.random.RandomState` instance seeded with
        ``seed``.  If ``seed`` is already a `~numpy.random.RandomState`,
        return it.  Otherwise raise ``ValueError``.
    Returns
    -------
    random_state : `numpy.random.RandomState`
        RandomState object.
    Notes
    -----
    This routine is adapted from scikit-learn.  See
    http://scikit-learn.org/stable/developers/utilities.html#validation-tools.
    """
    import numbers

    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    if type(seed)==list:
        if type(seed[0])==int:
            return np.random.RandomState(seed)

    raise ValueError('{0!r} cannot be used to seed a numpy.random.RandomState'
                     ' instance'.format(seed))
