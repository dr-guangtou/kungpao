#!/usr/bin/env python
"""
Extract information from GALFIT output FITS file.

This is based on astronomeralex's galfit-python-parser:
    https://github.com/astronomeralex/galfit-python-parser
Modified by Song Huang to include more features
"""

import re
import numpy as np

from astropy.io import fits


class GalfitComponent(object):
    """Stores results from one component of the fit."""

    def __init__(self, galfitheader, component_number, verbose=True):
        """
        Read GALFIT results from output file.

        takes in the fits header from HDU 3 (the galfit model) from a
        galfit output file and the component number to extract
        """
        assert component_number > 0
        assert "COMP_" + str(component_number) in galfitheader

        self.component_type = galfitheader["COMP_" + str(component_number)]
        self.component_number = component_number
        headerkeys = [i for i in galfitheader.keys()]
        comp_params = []

        for i in headerkeys:
            if str(component_number) + '_' in i:
                comp_params.append(i)

        setattr(self, 'good', True)
        for param in comp_params:
            paramsplit = param.split('_')
            val = galfitheader[param]
            """
            we know that val is a string formatted as 'result +/- uncertainty'
            """
            if "{" in val and "}" in val:
                if verbose:
                    print " ## One parameter is constrained !"
                val = val.replace('{', '')
                val = val.replace('}', '')
                val = val.split()
                if verbose:
                    print " ## Param - Value : ", param, val
                setattr(self, paramsplit[1].lower(), float(val[0]))
                setattr(self, paramsplit[1].lower() + '_err', np.nan)
            elif "[" in val and "]" in val:
                if verbose:
                    print " ## One parameter is fixed !"
                val = val.replace('[', '')
                val = val.replace(']', '')
                val = val.split()
                if verbose:
                    print " ## Param - Value : ", param, val
                setattr(self, paramsplit[1].lower(), float(val[0]))
                setattr(self, paramsplit[1].lower() + '_err', np.nan)
            elif "*" in val:
                if verbose:
                    print " ## One parameter is problematic !"
                val = val.replace('*', '')
                val = val.split()
                if verbose:
                    print " ## Param - Value : ", param, val
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

    def __init__(self, galfit_fits_file, hduLength=4, verbose=True):
        """
        Init method for GalfitResults.

        Take in a string that is the name of the galfit output fits file
        """
        hdulist = fits.open(galfit_fits_file)
        # Now some checks to make sure the file is what we are expecting
        assert len(hdulist) == hduLength
        galfitmodel = hdulist[hduLength - 2]
        galfitheader = galfitmodel.header
        galfit_in_comments = False
        for i in galfitheader['COMMENT']:
            galfit_in_comments = galfit_in_comments or "GALFIT" in i
        assert True == galfit_in_comments
        assert "COMP_1" in galfitheader
        # Now we've convinced ourselves that this is probably a galfit file

        self.galfit_fits_file = galfit_fits_file
        # Read in the input parameters
        self.input_initfile = galfitheader['INITFILE']
        self.input_datain = galfitheader["DATAIN"]
        self.input_sigma = galfitheader["SIGMA"]
        self.input_psf = galfitheader["PSF"]
        self.input_constrnt = galfitheader["CONSTRNT"]
        self.input_mask = galfitheader["MASK"]
        self.input_magzpt = galfitheader["MAGZPT"]

        # Fitting region
        fitsect = galfitheader["FITSECT"]
        fitsect = re.findall(r"[\w']+", fitsect)
        self.box_x0 = fitsect[0]
        self.box_x1 = fitsect[1]
        self.box_y0 = fitsect[2]
        self.box_y1 = fitsect[3]

        # Convolution box
        convbox = galfitheader["CONVBOX"]
        convbox = convbox.split(",")
        self.convbox_x = convbox[0]
        self.convbox_y = convbox[1]

        # Read in the chi-square value
        self.chisq = galfitheader["CHISQ"]
        self.ndof = galfitheader["NDOF"]
        self.nfree = galfitheader["NFREE"]
        self.reduced_chisq = galfitheader["CHI2NU"]
        self.logfile = galfitheader["LOGFILE"]

        # Find the number of components
        num_components = 1
        while True:
            if "COMP_" + str(num_components + 1) in galfitheader:
                num_components = num_components + 1
            else:
                break
        self.num_components = num_components

        for i in range(1, self.num_components + 1):
            setattr(self, "component_" + str(i),
                    GalfitComponent(galfitheader, i, verbose=False))

        hdulist.close()
