#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""1-D isophote fitting using `ellipse`."""

from kungpao.isophote import helper

__all__ = []

"""
CHANGES:
    - No Pyraf. Only use the `x_isophote.e`
    - Using configuration structure like dictionary and `yaml` file.
    - Separate the 1-D profile reading and post-processing step.

ELLIPSE:
    - `EllipseIraf()`: using the original IRAF executable file.
    - `EllipsePhot()`: using the `photutils.isophote`.
    - `EllipseHuoguo()`: placeholder for the C/C++ version of `ellipse`.
    - Should include benchmark test.

INPUT:
    - Input image: 2-D array or FITS filename
    - Configuration file: in `yaml` format
        - Include information for the galaxy
    - Optional: Mask image: 2-D array or FITS filename
    - Optional: Input ellipse binary table: for the "force photometry" mode
"""


class EllipseIraf(object):
    """
    Class that deals with elliptical isophote fitting using the STSDAS `ellipse` function.

    To use this class, it requires you have the executable binary files for some IRAF
    tasks **for your computer platform.** The `MacOSX` and `Linux64` versions are provided
    in the `kungpao/iraf` folder as default. The required files include:
        - `x_images.e`: this is one of the core IRAF procedure. We use the `imcopy` task to
            convert `.fits` file to the hardcore `.pl` format.
        - `x_ttools.e`: this part of the STSDAS/Tables package. We use the `tdump` task to
            convert the binary output table of `ellipse` into file for human.
        - `x_isophote.e`: this is the main procedure for isophote fitting.
            - `./x_isophote.e ellipse input` is used for the isophotal analysis.
            - `./x_isophote.e model intab output parent` is used to generate 2-D image.

    For more details, please see:
        http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?ellipse
    """
    def __init__(self, config=None):
        """Read in configuration file and input data."""
