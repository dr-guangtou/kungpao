#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Building 2-D model based on the ellipse result."""

import os
import warnings

from astropy.io import fits

__all__ = ['pyraf_bmodel']

PYRAF_INTERP = ["nearest", "linear", "poly3", "spline"]


def pyraf_bmodel(binary, parent, output=None, highar=False, verbose=False,
                 interp='spline', backgr=0.0):
    """Wrapper of the `Pyraf` `bmodel` function to build 2-D model.

    For details about the `bmodel` task, please see:
        http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?bmodel

    Parameters
    ----------
    binary: str
        Location of the output binary file from `ellipse` task.
    parent: str
        Location of the input image that `ellipse` run is based on.
    output: str, optional
        Name of the output FITS image
    highar: boolen, optional
        Whether to include the higher-order Fourier component. Default: False.
    verbose: boolen, optional
        Verbose mode. Default: False.
    interp: str, optional
        The interpolation mode used by the `trebin` task.
        Allowed values are: nearest | linear | poly3 | spline. Default: spline.

    Returns
    -------
    model: ndarray
        2-D model image of the `ellipse` results.

    """
    # Initiate the isophote module in IRAF/STSDAS
    try:
        from pyraf import iraf
        iraf.stsdas()
        iraf.analysis()
        iraf.isophote()
    except ImportError:
        raise Exception("# Unfortunately, you need to install pyraf first.")

    # Make sure the files are available
    if not os.path.isfile(binary):
        raise FileNotFoundError(
            "# Can not find the binary output file: {}".format(binary))
    if not os.path.isfile(parent):
        raise FileNotFoundError(
            "# Can not find the input image: {}".format(parent))
    
    # Name of the output FITS file
    if output is None:
        output = os.path.splitext(binary)[0] + '_model.fits'

    # Remove the output file if it exists
    if os.path.isfile(output):
        os.remove(output)

    # Check the interpolation method
    if interp not in PYRAF_INTERP:
        raise Exception(
            "# Wrong interpolation method! Choices are {}".format(PYRAF_INTERP))

    # Higher-order harmonic modes
    highar_str = "yes" if highar else "no"
    verbose_str = "yes" if verbose else "no"

    try:
        iraf.bmodel(binary, output, parent=parent, interp=interp, highar=highar_str,
                    backgr=backgr, verbose=verbose_str)
    except Exception:
        warnings.warn("# Something is wrong with the Bmodel task!")
        return None

    if not os.path.isfile(output):
        raise FileNotFoundError("# Cannot find the output file: {}".format(output))
    else:
        model = fits.open(output)[0].data
        return model


def isophote_bmodel(isophote, binary, parent, output=None, highar=False, verbose=False,
                    interp='spline'):
    """Wrapper of the x_isophote.e model procedure to build 2-D model.

    Does not work as expected!

    Parameters
    ----------
    isophote: str
        Location of the `x_isophote.e` executable binary file.

    Returns
    -------
    """
    return
