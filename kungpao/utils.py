"""
Random utilities
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import copy
import numpy as np

import sep

# Astropy related
from astropy import wcs
from astropy import units as u
from astropy.io import fits
from astropy.units import Quantity
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS, FK5

from astroquery.gaia import Gaia

# Matplotlib related
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse as mpl_ellip
plt.rc('text', usetex=True)

from .display import display_single, ORG


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


def random_cmap(ncolors=256, random_state=None):
    """
    Generate a matplotlib colormap consisting of random (muted) colors.
    A random colormap is very useful for plotting segmentation images.

    Parameters
    ----------
    ncolors : int, optional
        The number of colors in the colormap.  The default is 256.
    random_state : int or `~numpy.random.RandomState`, optional
        The pseudo-random number generator state used for random
        sampling.  Separate function calls with the same
        ``random_state`` will generate the same colormap.

    Returns
    -------
    cmap : `matplotlib.colors.Colormap`
        The matplotlib colormap with random colors.

    Notes
    -----
    Based on: colormaps.py in photutils
    """

    from matplotlib import colors

    prng = check_random_state(random_state)

    h = prng.uniform(low=0.0, high=1.0, size=ncolors)
    s = prng.uniform(low=0.2, high=0.7, size=ncolors)
    v = prng.uniform(low=0.5, high=1.0, size=ncolors)

    hsv = np.dstack((h, s, v))
    rgb = np.squeeze(colors.hsv_to_rgb(hsv))

    return colors.ListedColormap(rgb)


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
        pixValues = map(lambda x, y: img[int(y), int(x)],
                        px, py)

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


def image_gaia_stars(image, wcs, pixel=0.168,
                     mask_a=694.7, mask_b=4.04,
                     verbose=False, visual=False):
    """
    Search for bright stars using GAIA catalog.

    TODO:
        Should be absorbed by the object for image later.

    TODO:
        Should have a version that just uses the local catalog.
    """
    # Central coordinate
    ra_cen, dec_cen = wcs.wcs_pix2world(image.shape[0]/2,
                                        image.shape[1]/2, 0)
    img_cen_ra_dec = SkyCoord(ra_cen, dec_cen,
                              unit=('deg', 'deg'),
                              frame='icrs')

    # Width and height of the search box
    img_search_x = Quantity(pixel * (image.shape)[0], u.arcsec)
    img_search_y = Quantity(pixel * (image.shape)[1], u.arcsec)

    # Search for stars
    gaia_results = Gaia.query_object_async(coordinate=img_cen_ra_dec,
                                           width=img_search_x,
                                           height=img_search_y,
                                           verbose=verbose)

    if len(gaia_results) > 0:
        # Convert the (RA, Dec) of stars into pixel coordinate
        ra_gaia = np.asarray(gaia_results['ra'])
        dec_gaia = np.asarray(gaia_results['dec'])
        x_gaia, y_gaia = wcs.wcs_world2pix(ra_gaia, dec_gaia, 0)

        # Generate mask for each star
        rmask_gaia_arcsec = mask_a * np.exp(-gaia_results['phot_g_mean_mag'] / mask_b)

        # Update the catalog
        gaia_results.add_column(Column(data=x_gaia, name='x_pix'))
        gaia_results.add_column(Column(data=y_gaia, name='y_pix'))
        gaia_results.add_column(Column(data=rmask_gaia_arcsec, name='rmask_arcsec'))

        if visual:
            fig = plt.figure(figsize=(8, 8))
            ax1 = fig.add_subplot(111)

            show = display_single(image, ax=ax1)
            # Show stars
            ax1.scatter(gaia_results['x_pix'],
                        gaia_results['y_pix'], c=ORG(0.8),
                        s=70, alpha=0.7, marker='+')
            # Plot an ellipse for each object
            for star in gaia_results:
                smask = mpl_ellip(xy=(star['x_pix'],
                                      star['y_pix']),
                                  width=(2.0 * star['rmask_arcsec'] / pixel),
                                  height=(2.0 * star['rmask_arcsec'] / pixel),
                                  angle=0.0)
                smask.set_facecolor(ORG(0.3))
                smask.set_edgecolor(ORG(0.9))
                smask.set_alpha(0.3)
                ax1.add_artist(smask)

            return gaia_results, fig
        else:
            return gaia_results
    else:
        return None


def image_clean_up(img,
                   sig=None,
                   bad=None,
                   bkg_param_1={'bw': 30, 'bh': 30, 'fw': 5, 'fh': 5},
                   det_param_1={'thr': 2.0, 'minarea': 8, 'deb_n': 64, 'deb_c': 0.0001},
                   bkg_param_2={'bw': 100, 'bh': 100, 'fw': 5, 'fh': 5},
                   det_param_2={'thr': 3.0, 'minarea': 10, 'deb_n': 64, 'deb_c': 0.001},
                   verbose=False):
    """
    Clean up the image.

    TODO:
        Should be absorbed by object for image later.
    """
    # Measure a very local sky to help detection and deblending
    # Notice that this will remove large scale, and low surface brightness features.
    bkg_1 = sep.Background(img_swap, mask=bad, maskthresh=0,
                           bw=bkg_param_1['bw'], bh=bkg_param_1['bh'],
                           fw=bkg_param_1['fw'], fh=bkg_param_1['fh'])

    # Subtract a local sky, detect and deblend objects
    obj_1, seg_1 = sep.extract(img - bkg_1.back(), det_param_1['thr'],
                               err=sig,
                               minarea=det_param_1['minarea'],
                               deblend_nthresh=det_param_1['deb_n'],
                               deblend_cont=det_param_1['deb_c'],
                               segmentation_map=True)

    # Detect all pixels above the threshold
    obj_2, seg_2 = sep.extract(img - bkg_1.back(), det_param_1['thr'],
                               err=sig,
                               minarea=det_param_1['minarea'],
                               deblend_nthresh=det_param_1['deb_n'],
                               deblend_cont=det_param_1['deb_c'],
                               segmentation_map=True)
