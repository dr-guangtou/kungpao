#!/usr/bin/env python
# encoding: utf-8
"""
Extract information from GALFIT output FITS file.

This is based on astronomeralex's galfit-python-parser:
    https://github.com/astronomeralex/galfit-python-parser
Modified by Song Huang to include more features
"""

import re
import numpy as np

from astropy.io import fits

__all__ = ['GalfitComponent', 'GalfitResults']


class GalfitComponent(object):
    """Stores results from one component of the fit."""

    def __init__(self, galfit_header, component_number, verbose=True):
        """
        Read GALFIT results from output file.

        takes in the fits header from HDU 3 (the galfit model) from a
        galfit output file and the component number to extract
        """
        assert component_number > 0
        assert "COMP_" + str(component_number) in galfit_header

        self.component_type = galfit_header["COMP_" + str(component_number)]
        self.component_number = component_number
        headerkeys = [i for i in galfit_header.keys()]
        comp_params = []

        for i in headerkeys:
            if str(component_number) + '_' in i:
                comp_params.append(i)

        setattr(self, 'good', True)
        for param in comp_params:
            paramsplit = param.split('_')
            val = galfit_header[param]
            """
            we know that val is a string formatted as 'result +/- uncertainty'
            """
            if "{" in val and "}" in val:
                if verbose:
                    print(" ## One parameter is constrained !")
                val = val.replace('{', '')
                val = val.replace('}', '')
                val = val.split()
                if verbose:
                    print(" ## Param - Value : ", param, val)
                setattr(self, paramsplit[1].lower(), float(val[0]))
                setattr(self, paramsplit[1].lower() + '_err', np.nan)
            elif "[" in val and "]" in val:
                if verbose:
                    print(" ## One parameter is fixed !")
                val = val.replace('[', '')
                val = val.replace(']', '')
                val = val.split()
                if verbose:
                    print(" ## Param - Value : ", param, val)
                setattr(self, paramsplit[1].lower(), float(val[0]))
                setattr(self, paramsplit[1].lower() + '_err', np.nan)
            elif "*" in val:
                if verbose:
                    print(" ## One parameter is problematic !")
                val = val.replace('*', '')
                val = val.split()
                if verbose:
                    print(" ## Param - Value : ", param, val)
                setattr(self, paramsplit[1].lower(), float(val[0]))
                setattr(self, paramsplit[1].lower() + '_err', -1.0)
                setattr(self, 'good', True)
            else:
                val = val.split()
                setattr(self, paramsplit[1].lower(), float(val[0]))
                setattr(self, paramsplit[1].lower() + '_err', float(val[2]))


class GalfitResults(object):

    """
    This class stores galfit results information.

    Currently only does one component
    """

    def __init__(self, galfit_fits_file, hduLength=4):
        """Init method for GalfitResults.

        Take in a string that is the name of the galfit output fits file
        """
        hdulist = fits.open(galfit_fits_file)
        # Now some checks to make sure the file is what we are expecting
        assert len(hdulist) == hduLength
        galfit_model = hdulist[hduLength - 2]
        galfit_header = galfit_model.header
        galfit_in_comments = False
        for i in galfit_header['COMMENT']:
            galfit_in_comments = galfit_in_comments or "GALFIT" in i
        assert True == galfit_in_comments
        assert "COMP_1" in galfit_header
        # Now we've convinced ourselves that this is probably a galfit file

        self.galfit_fits_file = galfit_fits_file
        # Read in the input parameters
        self.input_initfile = galfit_header['INITFILE']
        self.input_datain = galfit_header["DATAIN"]
        self.input_sigma = galfit_header["SIGMA"]
        self.input_psf = galfit_header["PSF"]
        self.input_constrnt = galfit_header["CONSTRNT"]
        self.input_mask = galfit_header["MASK"]
        self.input_magzpt = galfit_header["MAGZPT"]

        # Fitting region
        fitsect = galfit_header["FITSECT"]
        fitsect = re.findall(r"[\w']+", fitsect)
        self.box_x0 = fitsect[0]
        self.box_x1 = fitsect[1]
        self.box_y0 = fitsect[2]
        self.box_y1 = fitsect[3]

        # Convolution box
        convbox = galfit_header["CONVBOX"]
        convbox = convbox.split(",")
        self.convbox_x = convbox[0]
        self.convbox_y = convbox[1]

        # Read in the chi-square value
        self.chisq = galfit_header["CHISQ"]
        self.ndof = galfit_header["NDOF"]
        self.nfree = galfit_header["NFREE"]
        self.reduced_chisq = galfit_header["CHI2NU"]
        self.logfile = galfit_header["LOGFILE"]

        # Find the number of components
        num_components = 1
        while True:
            if "COMP_" + str(num_components + 1) in galfit_header:
                num_components = num_components + 1
            else:
                break
        self.num_components = num_components

        for i in range(1, self.num_components + 1):
            setattr(self, "component_" + str(i),
                    GalfitComponent(galfit_header, i, verbose=False))

        hdulist.close()
