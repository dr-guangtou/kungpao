"""
Random utilities
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import copy
import numpy as np

import sep

from .display import display_single, ORG, SEG_CMAP

# Astropy related
from astropy import units as u
from astropy.units import Quantity
from astropy.table import Column
from astropy.coordinates import SkyCoord

from astroquery.gaia import Gaia

# Matplotlib related
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse as mpl_ellip
plt.rc('text', usetex=True)


"""
Angle related functions:

Strongly based on: https://github.com/phn/angles/blob/master/angles.py
by Prasanth Nair

"""
def rad2deg(rad):
    """Convert radians into degrees."""
    return (rad * 180.0 / np.pi)


def deg2rad(deg):
    """Convert degrees into radians."""
    return (deg * np.pi / 180.0)


def hr2deg(deg):
    """Convert degrees into hours."""
    return (deg * (24.0 / 360.0))


def deg2hr(hr):
    """Convert hours into degrees."""
    return (hr * 15.0)


def normAngle(num, lower=0, upper=360, b=False):
    """
    Normalize number to range [lower, upper) or [lower, upper].

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
    Raises
    ------
    ValueError
      If lower >= upper.
    Notes
    -----
    If the keyword `b == False`, then the normalization is done in the
    following way. Consider the numbers to be arranged in a circle,
    with the lower and upper ends sitting on top of each other. Moving
    past one limit, takes the number into the beginning of the other
    end. For example, if range is [0 - 360), then 361 becomes 1 and 360
    becomes 0. Negative numbers move from higher to lower numbers. So,
    -1 normalized to [0 - 360) becomes 359.
    If the keyword `b == True`, then the given number is considered to
    "bounce" between the two limits. So, -91 normalized to [-90, 90],
    becomes -89, instead of 89. In this case the range is [lower,
    upper]. This code is based on the function `fmt_delta` of `TPM`.
    Range must be symmetric about 0 or lower == 0.
    Examples
    --------
    >>> normalize(-270,-180,180)
    90.0
    >>> import math
    >>> math.degrees(normalize(-2*math.pi,-math.pi,math.pi))
    0.0
    >>> normalize(-180, -180, 180)
    -180.0
    >>> normalize(180, -180, 180)
    -180.0
    >>> normalize(180, -180, 180, b=True)
    180.0
    >>> normalize(181,-180,180)
    -179.0
    >>> normalize(181, -180, 180, b=True)
    179.0
    >>> normalize(-180,0,360)
    180.0
    >>> normalize(36,0,24)
    12.0
    >>> normalize(368.5,-180,180)
    8.5
    >>> normalize(-100, -90, 90)
    80.0
    >>> normalize(-100, -90, 90, b=True)
    -80.0
    >>> normalize(100, -90, 90, b=True)
    80.0
    >>> normalize(181, -90, 90, b=True)
    -1.0
    >>> normalize(270, -90, 90, b=True)
    -90.0
    >>> normalize(271, -90, 90, b=True)
    -89.0
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


"""
Geometry Related
"""
def ellipDist(x, y, x0, y0, pa=0.0, q=0.9):
    """Distance to center in elliptical coordinate."""
    theta = (pa * np.pi / 180.0)

    distA = ((x - x0) * np.cos(theta) + (y - y0) * np.sin(theta)) ** 2.0
    distB = (((y - y0) * np.cos(theta) - (x - x0) * np.sin(theta)) / q) ** 2.0

    return np.sqrt(distA + distB)


"""
Weighted mean and median

Based on https://github.com/tinybike/weightedstats
"""
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


"""
PolyNomial Fitting
"""
def polyFit(x, y, order=4):
    """
    Fit polynomial.
    """
    if len(x) != len(y):
        raise Exception("### X and Y should have the same size")
    coefficients = np.polyfit(x, y, order)
    polynomial = np.poly1d(coefficients)
    fit = polynomial(x)

    return fit


"""
File Manipulation

    * Save numpy array to cPickle file format
    * Save numpy array to hickle/HDF5 format
    * Save numpy array to csv file format
"""
def saveToPickle(array, name):
    """Save a numpy array to a cPickle/Pickle format binary file."""
    try:
        import cPickle as pickle
    except:
        import pickle

    output = open(name, 'w')
    pickle.dump(array, output, protocol=2)
    output.close()


def saveToHickle(array, name):
    """Save a numpy array to a hickle/HDF5 format binary file."""
    try:
        import hickle
    except:
        raise Exception("### The Hickle package is required!")

    output = open(name, 'w')
    hickle.dump(array, output, protocol=2)
    output.close()


def saveToCSV(array, name):
    """
    Save a numpy array to a CSV file.

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


def parseRegEllipse(regName):
    """
    Parse a DS9 .reg files.

    convert the Ellipse or Circle regions
    into arrays of parameters for ellipse:
    x, y, a, b, theta
    """
    if os.path.isfile(regName):
        raise Exception("### Can not find the .reg file!")
    # Parse the .reg file into lines
    lines = [line.strip() for line in open(regName, 'r')]
    # Coordinate type of this .reg file: e.g. 'image'
    coordType = lines[2].strip()
    # Parse each region
    regs = [reg.split(" ") for reg in lines[3:]]

    xc = []
    yc = []
    ra = []
    rb = []
    theta = []

    for reg in regs:
        if reg[0].strip() == 'ellipse' and len(reg) is 6:
            xc.append(float(reg[1]))
            yc.append(float(reg[2]))
            ra.append(float(reg[3]))
            rb.append(float(reg[4]))
            theta.append(float(reg[5]) * np.pi / 180.0)
        elif reg[0].strip() == 'circle' and len(reg) is 4:
            xc.append(float(reg[1]))
            yc.append(float(reg[2]))
            ra.append(float(reg[3]))
            rb.append(float(reg[3]))
            theta.append(0.0)

    xc = np.array(xc, dtype=np.float32)
    yc = np.array(yc, dtype=np.float32)
    ra = np.array(ra, dtype=np.float32)
    rb = np.array(rb, dtype=np.float32)
    theta = np.array(theta, dtype=np.float32)

    return xc, yc, ra, rb, theta, coordType


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
    if isinstance(seed, (numbers.Real)):
        # Song Huang: In case the seed is a float number, convert it to int
        return np.random.RandomState(int(seed))
    if isinstance(seed, np.random.RandomState):
        return seed
    if type(seed) == list:
        if type(seed[0]) == int:
            return np.random.RandomState(seed)

    raise ValueError('{0!r} cannot be used to seed a numpy.random.RandomState'
                     ' instance'.format(seed))


def get_pixel_value(img, wcs, ra, dec):
    """
    Return the pixel value from image based on RA, DEC.

    TODO:
        Should be absorbed into the image object later

    Parameters:
        img     : 2-D data array
        wcs     : WCS from the image header
        ra, dec : coordinates, can be array
    """
    px, py = wcs.wcs_world2pix(ra, dec, 0)

    import collections
    if not isinstance(px, collections.Iterable):
        pixValues = img[int(py), int(px)]
    else:
        pixValues = map(lambda x, y: img[int(y), int(x)], px, py)

    return np.asarray(pixValues)


def seg_remove_cen_obj(seg):
    """
    Remove the central object from the segmentation.

    TODO:
        Should be absorbed by objects for segmentation image
    """
    seg_copy = copy.deepcopy(seg)
    seg_copy[seg == seg[int(seg.shape[0] / 2L), int(seg.shape[1] / 2L)]] = 0

    return seg_copy


def seg_index_cen_obj(seg):
    """
    Remove the index array for central object.

    TODO:
        Should be absorbed by objects for segmentation image
    """
    cen_obj = seg[int(seg.shape[0] / 2L), int(seg.shape[1] / 2L)]
    if cen_obj == 0:
        return None
    else:
        return (seg == cen_obj)


def seg_remove_obj(seg, x, y):
    """
    Remove an object from the segmentation given its coordinate.

    TODO:
        Should be absorbed by objects for segmentation image
    """
    seg_copy = copy.deepcopy(seg)
    seg_copy[seg == seg[int(x), int(y)]] = 0

    return seg_copy


def seg_index_obj(seg, x, y):
    """
    Remove the index array for an object given its location.

    TODO:
        Should be absorbed by objects for segmentation image
    """
    obj = seg[int(x), int(y)]
    if obj == 0:
        return None
    else:
        return (seg == obj)


def image_gaia_stars(image,
                     wcs,
                     pixel=0.168,
                     mask_a=694.7,
                     mask_b=4.04,
                     verbose=False,
                     visual=False,
                     size_buffer=1.4):
    """
    Search for bright stars using GAIA catalog.

    TODO:
        Should be absorbed by the object for image later.

    TODO:
        Should have a version that just uses the local catalog.
    """
    # Central coordinate
    ra_cen, dec_cen = wcs.wcs_pix2world(image.shape[0] / 2, image.shape[1] / 2,
                                        0)
    img_cen_ra_dec = SkyCoord(
        ra_cen, dec_cen, unit=('deg', 'deg'), frame='icrs')

    # Width and height of the search box
    img_search_x = Quantity(pixel * (image.shape)[0] * size_buffer, u.arcsec)
    img_search_y = Quantity(pixel * (image.shape)[1] * size_buffer, u.arcsec)

    # Search for stars
    gaia_results = Gaia.query_object_async(
        coordinate=img_cen_ra_dec,
        width=img_search_x,
        height=img_search_y,
        verbose=verbose)

    if len(gaia_results) > 0:
        # Convert the (RA, Dec) of stars into pixel coordinate
        ra_gaia = np.asarray(gaia_results['ra'])
        dec_gaia = np.asarray(gaia_results['dec'])
        x_gaia, y_gaia = wcs.wcs_world2pix(ra_gaia, dec_gaia, 0)

        # Generate mask for each star
        rmask_gaia_arcsec = mask_a * np.exp(
            -gaia_results['phot_g_mean_mag'] / mask_b)

        # Update the catalog
        gaia_results.add_column(Column(data=x_gaia, name='x_pix'))
        gaia_results.add_column(Column(data=y_gaia, name='y_pix'))
        gaia_results.add_column(
            Column(data=rmask_gaia_arcsec, name='rmask_arcsec'))

        if visual:
            fig = plt.figure(figsize=(8, 8))
            ax1 = fig.add_subplot(111)

            ax1 = display_single(image, ax=ax1)
            # Plot an ellipse for each object
            for star in gaia_results:
                smask = mpl_ellip(
                    xy=(star['x_pix'], star['y_pix']),
                    width=(2.0 * star['rmask_arcsec'] / pixel),
                    height=(2.0 * star['rmask_arcsec'] / pixel),
                    angle=0.0)
                smask.set_facecolor(ORG(0.2))
                smask.set_edgecolor(ORG(1.0))
                smask.set_alpha(0.3)
                ax1.add_artist(smask)

            # Show stars
            ax1.scatter(
                gaia_results['x_pix'],
                gaia_results['y_pix'],
                c=ORG(1.0),
                s=100,
                alpha=0.9,
                marker='+')

            ax1.set_xlim(0, image.shape[0])
            ax1.set_ylim(0, image.shape[1])

            return gaia_results, fig
        else:
            return gaia_results
    else:
        return None


def image_clean_up(
        img,
        sig=None,
        bad=None,
        bkg_param_1={'bw': 20,
                     'bh': 20,
                     'fw': 3,
                     'fh': 3},
        det_param_1={'thr': 1.5,
                     'minarea': 40,
                     'deb_n': 128,
                     'deb_c': 0.00001},
        bkg_param_2={'bw': 150,
                     'bh': 150,
                     'fw': 7,
                     'fh': 7},
        det_param_2={'thr': 2.0,
                     'minarea': 20,
                     'deb_n': 64,
                     'deb_c': 0.001},
        bkg_param_3={'bw': 60,
                     'bh': 60,
                     'fw': 5,
                     'fh': 5},
        det_param_3={'thr': 3.5,
                     'minarea': 10,
                     'deb_n': 64,
                     'deb_c': 0.005},
        verbose=False,
        visual=False,
        diagnose=False,
        **kwargs):
    """
    Clean up the image.

    TODO:
        Should be absorbed by object for image later.
    """
    # Measure a very local sky to help detection and deblending
    # Notice that this will remove large scale, and low surface brightness
    # features.
    bkg_1 = sep.Background(
        img,
        mask=bad,
        maskthresh=0,
        bw=bkg_param_1['bw'],
        bh=bkg_param_1['bh'],
        fw=bkg_param_1['fw'],
        fh=bkg_param_1['fh'])
    if verbose:
        print("# BKG 1: Mean Sky / RMS Sky = %10.5f / %10.5f" %
              (bkg_1.globalback, bkg_1.globalrms))

    # Subtract a local sky, detect and deblend objects
    obj_1, seg_1 = sep.extract(
        img - bkg_1.back(),
        det_param_1['thr'],
        err=sig,
        minarea=det_param_1['minarea'],
        deblend_nthresh=det_param_1['deb_n'],
        deblend_cont=det_param_1['deb_c'],
        segmentation_map=True)
    if verbose:
        print("# DET 1: Detect %d objects" % len(obj_1))

    # Detect all pixels above the threshold
    bkg_2 = sep.Background(
        img,
        bw=bkg_param_2['bw'],
        bh=bkg_param_2['bh'],
        fw=bkg_param_2['fw'],
        fh=bkg_param_2['fh'])

    obj_2, seg_2 = sep.extract(
        img,
        det_param_2['thr'],
        err=sig,
        minarea=det_param_2['minarea'],
        deblend_nthresh=det_param_2['deb_n'],
        deblend_cont=det_param_2['deb_c'],
        segmentation_map=True)
    if verbose:
        print("# DET 2: Detect %d objects" % len(obj_2))

    # Estimate the background for generating noise image
    bkg_3 = sep.Background(
        img,
        mask=seg_2,
        maskthresh=0,
        bw=bkg_param_3['bw'],
        bh=bkg_param_3['bh'],
        fw=bkg_param_3['fw'],
        fh=bkg_param_3['fh'])
    if verbose:
        print("# BKG 3: Mean Sky / RMS Sky = %10.5f / %10.5f" %
              (bkg_3.globalback, bkg_3.globalrms))

    if sig is None:
        noise = np.random.normal(
            loc=bkg_3.globalback, scale=bkg_3.globalrms, size=img.shape)
    else:
        sky_val = bkg_3.back()
        sky_sig = bkg_3.rms()
        sky_sig[sky_sig <= 0] = 1E-8
        noise = np.random.normal(loc=sky_val, scale=sky_sig, size=img.shape)

    # Replace all detected pixels with noise
    img_noise_replace = copy.deepcopy(img)
    img_noise_replace[seg_2 > 0] = noise[seg_2 > 0]

    # Detect the faint objects left on the image
    obj_3, seg_3 = sep.extract(
        img_noise_replace,
        det_param_3['thr'],
        err=sig,
        minarea=det_param_3['minarea'],
        deblend_nthresh=det_param_3['deb_n'],
        deblend_cont=det_param_3['deb_c'],
        segmentation_map=True)
    if verbose:
        print("# DET 3: Detect %d objects" % len(obj_3))

    # Combine the two segmentation maps
    seg_comb = (seg_2 + seg_3)

    # Index for the central object
    obj_cen_mask = seg_index_cen_obj(seg_1)
    if verbose:
        if obj_cen_mask is not None:
            print("# Central object: %d pixels" % np.sum(obj_cen_mask))
        else:
            print("# Central object not detected !")

    if obj_cen_mask is not None:
        seg_comb[obj_cen_mask] = 0

    img_clean = copy.deepcopy(img)
    img_clean[seg_comb > 0] = noise[seg_comb > 0]

    if diagnose:
        everything = {
            'img': img,
            'sig': sig,
            "bkg_1": bkg_1,
            "obj_1": obj_1,
            "seg_1": seg_1,
            "bkg_2": bkg_2,
            "obj_2": obj_2,
            "seg_2": seg_2,
            "bkg_3": bkg_3,
            "obj_3": obj_3,
            "seg_3": seg_3,
            "noise": noise
        }
        if visual:
            return img_clean, everything, diagnose_image_clean(
                img_clean, everything, **kwargs)
        else:
            return img_clean, everything
    else:
        return img_clean


def diagnose_image_clean(img_clean, everything,
                         pixel_scale=0.168,
                         physical_scale=None,
                         scale_bar_length=2.0):
    """
    Generate a QA plot for image clean.
    """
    fig = plt.figure(figsize=(18, 18))
    fig.subplots_adjust(
        left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.00, hspace=0.00)

    ax1 = plt.subplot(3, 3, 1)
    if everything['img'] is not None:
        ax1 = display_single(
            everything['img'],
            ax=ax1,
            contrast=0.20,
            scale_bar_length=scale_bar_length,
            pixel_scale=pixel_scale,
            physical_scale=physical_scale,
            color_bar=True)

    ax2 = plt.subplot(3, 3, 2)
    if everything['sig'] is not None:
        ax2 = display_single(
            everything['sig'],
            ax=ax2,
            contrast=0.30,
            scale_bar_length=scale_bar_length,
            pixel_scale=pixel_scale,
            physical_scale=physical_scale,
            color_bar=True)

    ax3 = plt.subplot(3, 3, 3)
    if everything['bkg_1'] is not None:
        ax3 = display_single(
            everything['bkg_1'].back(),
            ax=ax3,
            contrast=0.20,
            scale_bar_length=scale_bar_length,
            pixel_scale=pixel_scale,
            physical_scale=physical_scale,
            color_bar=True)

    ax4 = plt.subplot(3, 3, 4)
    if everything['seg_1'] is not None:
        ax1 = display_single(
            everything['seg_1'],
            ax=ax4,
            contrast=0.10,
            scale_bar_length=scale_bar_length,
            pixel_scale=pixel_scale,
            physical_scale=physical_scale,
            scale_bar_color='k',
            cmap=SEG_CMAP,
            scale='none',
            stretch='linear')

    ax5 = plt.subplot(3, 3, 5)
    if everything['bkg_3'] is not None:
        ax5 = display_single(
            everything['bkg_3'].back(),
            ax=ax5,
            contrast=0.20,
            scale_bar_length=scale_bar_length,
            pixel_scale=pixel_scale,
            physical_scale=physical_scale,
            color_bar=True)

    ax6 = plt.subplot(3, 3, 6)
    if everything['seg_2'] is not None:
        ax6 = display_single(
            everything['seg_2'],
            ax=ax6,
            contrast=0.10,
            scale_bar_length=scale_bar_length,
            pixel_scale=pixel_scale,
            physical_scale=physical_scale,
            scale_bar_color='k',
            scale='none',
            cmap=SEG_CMAP,
            stretch='linear')

    ax7 = plt.subplot(3, 3, 7)
    if everything['seg_3'] is not None:
        ax7 = display_single(
            everything['seg_3'],
            ax=ax7,
            contrast=0.10,
            scale_bar_length=scale_bar_length,
            pixel_scale=pixel_scale,
            physical_scale=physical_scale,
            scale_bar_color='k',
            scale='none',
            cmap=SEG_CMAP,
            stretch='linear')

    ax8 = plt.subplot(3, 3, 8)
    if everything['noise'] is not None:
        ax8 = display_single(
            everything['noise'],
            ax=ax8,
            contrast=0.20,
            scale_bar_length=scale_bar_length,
            pixel_scale=pixel_scale,
            physical_scale=physical_scale,
            color_bar=True)

    ax9 = plt.subplot(3, 3, 9)
    ax9 = display_single(
        img_clean,
        ax=ax9,
        contrast=0.20,
        scale_bar_length=scale_bar_length,
        pixel_scale=pixel_scale,
        physical_scale=physical_scale,
        color_bar=True)

    return fig
