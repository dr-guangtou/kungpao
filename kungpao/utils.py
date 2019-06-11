"""Misc utilities."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys
import math
import time
import random
import string

import numpy as np

# Erin Sheldon's cosmology library
import cosmology as cosmology_erin

from tqdm import tqdm

# Astropy related
from astropy import units as u
from astropy.coordinates import SkyCoord

cosmo_erin = cosmology_erin.Cosmo(H0=70.0, omega_m=0.30)

__all__ = ['rad2deg', 'deg2rad', 'hr2deg', 'deg2hr',
           'normalize_angle', 'dist_elliptical', 'weighted_mean',
           'numpy_weighted_mean', 'weighted_median',
           'numpy_weighted_median', 'simple_poly_fit',
           'get_time_label', 'check_random_state', 'random_string',
           'kpc_scale_astropy', 'kpc_scale_erin', 'angular_distance',
           'angular_distance_single', 'angular_distance_astropy']


def rad2deg(rad):
    """Convert radians into degrees."""
    return rad * 180.0 / np.pi


def deg2rad(deg):
    """Convert degrees into radians."""
    return deg * np.pi / 180.0


def hr2deg(deg):
    """Convert degrees into hours."""
    return deg * (24.0 / 360.0)


def deg2hr(hr):
    """Convert hours into degrees."""
    return hr * 15.0


def normalize_angle(num, lower=0, upper=360, b=False):
    """Normalize number to range [lower, upper) or [lower, upper].

    Parameters
    ----------
    num : float
        The number to be normalized.
    lower : int
        Lower limit of range. Default is 0.
    upper : int
        Upper limit of range. Default is 360.
    b : bool
        Type of normalization. Default is False. See notes.

    Returns
    -------
    n : float
        A number in the range [lower, upper) or [lower, upper].

    """
    from math import floor, ceil

    # abs(num + upper) and abs(num - lower) are needed, instead of
    # abs(num), since the lower and upper limits need not be 0. We need
    # to add half size of the range, so that the final result is lower +
    # <value> or upper - <value>, respectively.
    res = num
    if not b:
        if lower >= upper:
            raise ValueError("Invalid lower and upper limits: (%s, %s)" %
                             (lower, upper))

        res = num
        if num > upper or num == lower:
            num = lower + abs(num + upper) % (abs(lower) + abs(upper))
        if num < lower or num == upper:
            num = upper - abs(num - lower) % (abs(lower) + abs(upper))

        res = lower if num == upper else num
    else:
        total_length = abs(lower) + abs(upper)
        if num < -total_length:
            num += ceil(num / (-2 * total_length)) * 2 * total_length
        if num > total_length:
            num -= floor(num / (2 * total_length)) * 2 * total_length
        if num > upper:
            num = total_length - num
        if num < lower:
            num = -total_length - num

        res = num

    res *= 1.0  # Make all numbers float, to be consistent

    return res


def dist_elliptical(x, y, x0, y0, pa=0.0, q=0.9):
    """Distance to center in elliptical coordinate."""
    theta = (pa * np.pi / 180.0)

    distA = ((x - x0) * np.cos(theta) +
             (y - y0) * np.sin(theta)) ** 2.0
    distB = (((y - y0) * np.cos(theta) -
              (x - x0) * np.sin(theta)) / q) ** 2.0

    return np.sqrt(distA + distB)


def weighted_mean(data, weights=None):
    """Calculate the weighted mean of a list."""
    if weights is None:
        return np.mean(data)

    total_weight = float(sum(weights))
    weights = [weight / total_weight for weight in weights]
    w_mean = 0
    for i, weight in enumerate(weights):
        w_mean += weight * data[i]

    return w_mean


def numpy_weighted_mean(data, weights=None):
    """Calculate the weighted mean of an array/list using numpy."""
    weights = np.array(weights).flatten() / float(sum(weights))

    return np.dot(np.array(data), weights)


def weighted_median(data, weights=None):
    """Calculate the weighted median of a list."""
    if weights is None:
        return np.median(data)

    midpoint = 0.5 * sum(weights)
    if any([j > midpoint for j in weights]):
        return data[weights.index(max(weights))]
    if any([j > 0 for j in weights]):
        sorted_data, sorted_weights = zip(*sorted(zip(data, weights)))
        cumulative_weight = 0
        below_midpoint_index = 0
        while cumulative_weight <= midpoint:
            below_midpoint_index += 1
            cumulative_weight += sorted_weights[below_midpoint_index-1]
        cumulative_weight -= sorted_weights[below_midpoint_index-1]
        if cumulative_weight - midpoint < sys.float_info.epsilon:
            bounds = sorted_data[below_midpoint_index-2:below_midpoint_index]
            return sum(bounds) / float(len(bounds))
        return sorted_data[below_midpoint_index-1]


def numpy_weighted_median(data, weights=None):
    """Calculate the weighted median of an array/list using numpy."""
    if weights is None:
        return np.median(np.array(data).flatten())
    data, weights = np.array(data).flatten(), np.array(weights).flatten()
    if any(weights > 0):
        sorted_data, sorted_weights = map(np.array,
                                          zip(*sorted(zip(data, weights))))
        midpoint = 0.5 * sum(sorted_weights)
        if any(weights > midpoint):
            return (data[weights == np.max(weights)])[0]
        cumulative_weight = np.cumsum(sorted_weights)
        below_midpoint_index = np.where(cumulative_weight <= midpoint)[0][-1]
        if (cumulative_weight[below_midpoint_index] -
                midpoint) < sys.float_info.epsilon:
            return np.mean(sorted_data[below_midpoint_index:
                                       below_midpoint_index+2])
        return sorted_data[below_midpoint_index+1]


def simple_poly_fit(x, y, order=4):
    """Fit 1-D polynomial."""
    if len(x) != len(y):
        raise Exception("### X and Y should have the same size")

    coefficients = np.polyfit(x, y, order)
    polynomial = np.poly1d(coefficients)
    fit = polynomial(x)

    return fit


def get_time_label():
    """Return time label for new files & directories.

    From: https://github.com/johnnygreco/hugs/blob/master/hugs/utils.py
    """
    return time.strftime("%Y%m%d-%H%M%S")


def check_random_state(seed):
    """Turn seed into a `numpy.random.RandomState` instance.

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
    if isinstance(seed, (numbers.Real)):
        # Song Huang: In case the seed is a float number, convert it to int
        return np.random.RandomState(int(seed))
    if isinstance(seed, np.random.RandomState):
        return seed
    if isinstance(seed, list):
        if isinstance(seed[0], (numbers.Integral, np.integer)):
            return np.random.RandomState(seed)

    raise ValueError('{0!r} cannot be used to seed a numpy.random.RandomState'
                     ' instance'.format(seed))


def random_string(length=5, chars=string.ascii_uppercase + string.digits):
    """Random string generator.

    Based on:
    http://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits-in-python

    Parameters:
    -----------
    length: int
        Length of the string. Default: 5
    chars: string object
        Types of characters allowed in the random string. Default: ASCII_Uppercase + Digits.
    """
    return ''.join(random.choice(chars) for _ in range(length))


def kpc_scale_astropy(cosmo, redshift):
    """Kpc / arcsec using Astropy cosmology."""
    return (1.0 / cosmo.arcsec_per_kpc_proper(redshift).value)


def kpc_scale_erin(cosmo, redshift):
    """Kpc / arcsec using cosmology by Erin Sheldon."""
    return (cosmo.Da(0.0, redshift) / 206.264806)


def angular_distance(ra_1, dec_1, ra_arr_2, dec_arr_2):
    """Angular distances between coordinates.

    This is just the most straightforward Python code.
    Based on: https://github.com/phn/angles/blob/master/angles.py

    Return
    ------
        Angular distance in unit of arcsec

    """
    d2r = math.pi / 180.0

    xyz_1 = np.asarray([np.cos(dec_1 * d2r) * np.cos(ra_1 * d2r),
                        np.cos(dec_1 * d2r) * np.sin(ra_1 * d2r),
                        np.sin(dec_1 * d2r)]).transpose()

    xyz_2 = np.asarray([np.cos(dec_arr_2 * d2r) * np.cos(ra_arr_2 * d2r),
                        np.cos(dec_arr_2 * d2r) * np.sin(ra_arr_2 * d2r),
                        np.sin(dec_arr_2 * d2r)]).transpose()

    return np.arctan2(np.sqrt(np.sum(np.cross(xyz_1, xyz_2) ** 2.0, axis=1)),
                      np.sum(xyz_1 * xyz_2, axis=1)) / d2r * 3600.0


def angular_distance_single(ra_1, dec_1, ra_2, dec_2):
    """Angular distances between coordinates for single object.

    This is just the most straightforward Python code.
    Based on: https://github.com/phn/angles/blob/master/angles.py

    Return
    ------
        Angular distance in unit of arcsec

    """
    d2r = math.pi / 180.0
    ra_1 *= d2r
    dec_1 *= d2r
    ra_2 *= d2r
    dec_2 *= d2r

    x_1 = math.cos(dec_1) * math.cos(ra_1)
    y_1 = math.cos(dec_1) * math.sin(ra_1)
    z_1 = math.sin(dec_1)

    x_2 = math.cos(dec_2) * math.cos(ra_2)
    y_2 = math.cos(dec_2) * math.sin(ra_2)
    z_2 = math.sin(dec_2)

    d = (x_2 * x_1 + y_2 * y_1 + z_2 * z_1)

    c_x = y_1 * z_2 - z_1 * y_2
    c_y = - (x_1 * z_2 - z_1 * x_2)
    c_z = (x_1 * y_2 - y_1 * x_2)
    c = math.sqrt(c_x ** 2 + c_y ** 2 + c_z ** 2)

    res = math.atan2(c, d)

    return res / d2r * 3600.0


def angular_distance_astropy(ra_1, dec_1, ra_2, dec_2):
    """Compute angular distances using Astropy.

    Return
    ------
        Angular distance in unit of arcsec

    """
    coord1 = SkyCoord(ra_1 * u.degree, dec_1 * u.degree)
    coord2 = SkyCoord(ra_2 * u.degree, dec_2 * u.degree)

    return coord1.separation(coord2).arcsec
