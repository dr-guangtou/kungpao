"""File Input/Output."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import warnings

try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np

from astropy.io import fits

__all__ = ['save_to_pickle', 'save_to_hickle', 'save_to_csv',
           'save_to_fits', 'parse_reg_ellipse', 'psfex_extract',
           'read_from_pickle', 'save_to_dill', 'read_from_dill']


def read_from_pickle(name):
    """Read the data from Pickle file."""
    return pickle.load(open(name, "rb"))


def save_to_pickle(obj, name):
    """Save an object to a cPickle/Pickle format binary file."""
    output = open(name, 'wb')
    pickle.dump(obj, output, protocol=2)
    output.close()

    return


def save_to_hickle(obj, name):
    """Save an object to a hickle/HDF5 format binary file."""
    try:
        import hickle
    except ImportError:
        raise Exception("### The Hickle package is required!")

    output = open(name, 'wb')
    hickle.dump(obj, output, protocol=2)
    output.close()

    return


def save_to_csv(array, name):
    """Save a numpy array to a CSV file.

    Use the dtype.name as column name if possible
    """
    output = open(name, 'w')
    colNames = array.dtype.names
    output.write("#" + ', '.join(colNames) + '\n')
    for item in array:
        line = ''
        for i in range(0, len(colNames)-1):
            col = colNames[i]
            line += str(item[col]) + ' , '
        line += str(item[colNames[-1]]) + '\n'
        output.write(line)
    output.close()

    return


def save_to_fits(img, fits_file, wcs=None, header=None, overwrite=True):
    """Save an image to FITS file."""
    if wcs is not None:
        wcs_header = wcs.to_header()
        img_hdu = fits.PrimaryHDU(img, header=wcs_header)
    else:
        img_hdu = fits.PrimaryHDU(img)
    if header is not None:
        if 'SIMPLE' in header and 'BITPIX' in header:
            img_hdu.header = header
        else:
            img_hdu.header.extend(header)

    if os.path.islink(fits_file):
        os.unlink(fits_file)

    img_hdu.writeto(fits_file, overwrite=overwrite)

    return


def parse_reg_ellipse(reg_file):
    """Parse a DS9 .reg files.

    convert the Ellipse or Circle regions
    into arrays of parameters for ellipse:
    x, y, a, b, theta
    """
    if os.path.isfile(reg_file):
        raise Exception("### Can not find the .reg file!")

    # Parse the .reg file into lines
    lines = [line.strip() for line in open(reg_file, 'r')]

    # Coordinate type of this .reg file: e.g. 'image'
    coord_type = lines[2].strip()
    # Parse each region
    regs = [reg.split(" ") for reg in lines[3:]]

    xc = []
    yc = []
    ra = []
    rb = []
    theta = []

    for reg in regs:
        if reg[0].strip() == 'ellipse' and len(reg) == 6:
            xc.append(float(reg[1]))
            yc.append(float(reg[2]))
            ra.append(float(reg[3]))
            rb.append(float(reg[4]))
            theta.append(float(reg[5]) * np.pi / 180.0)
        elif reg[0].strip() == 'circle' and len(reg) == 4:
            xc.append(float(reg[1]))
            yc.append(float(reg[2]))
            ra.append(float(reg[3]))
            rb.append(float(reg[3]))
            theta.append(0.0)
        else:
            warnings.warn("Wrong shape, only Ellipse or Circle are availabe")

    xc = np.array(xc, dtype=np.float32)
    yc = np.array(yc, dtype=np.float32)
    ra = np.array(ra, dtype=np.float32)
    rb = np.array(rb, dtype=np.float32)
    theta = np.array(theta, dtype=np.float32)

    return xc, yc, ra, rb, theta, coord_type


def psfex_extract(psfex_file, row, col):
    """Extract PSF image from PSFex result."""
    try:
        import psfex
    except ImportError:
        raise Exception("Need to install PSFex library first!")

    return psfex.PSFEx(psfex_file).get_rec(row, col)


def save_to_dill(obj, name):
    """Save the Python object in a dill file."""
    import dill
    with open(name, "wb") as dill_file:
        dill.dump(obj, dill_file)


def read_from_dill(name):
    """Read saved Python object from a dill file."""
    import dill
    with open(name, "rb") as dill_file:
        content = dill.load(dill_file)

    return content
