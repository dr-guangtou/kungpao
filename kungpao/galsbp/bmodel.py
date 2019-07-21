#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Building 2-D model based on the ellipse result."""

__all__ = []

def pyraf_bmodel(binary, parent, output=None, highar=False, verbose=False, interp='spline'):
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
    return 

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
