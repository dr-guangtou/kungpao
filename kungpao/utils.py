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

from .display import display_single, ORG, SEG_CMAP, IMG_CMAP


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


def random_cmap(ncolors=256, random_state=None, background_color='white'):
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

    if background_color is not None:
        if background_color not in colors.cnames:
            raise ValueError('"{0}" is not a valid background color '
                             'name'.format(background_color))
        rgb[0] = colors.hex2color(colors.cnames[background_color])

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


def image_gaia_stars(image, wcs, pixel=0.168,
                     mask_a=694.7, mask_b=4.04,
                     verbose=False, visual=False,
                     size_buffer=1.4):
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
    img_search_x = Quantity(pixel * (image.shape)[0] * size_buffer, u.arcsec)
    img_search_y = Quantity(pixel * (image.shape)[1] * size_buffer, u.arcsec)

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
            # Plot an ellipse for each object
            for star in gaia_results:
                smask = mpl_ellip(xy=(star['x_pix'],
                                      star['y_pix']),
                                  width=(2.0 * star['rmask_arcsec'] / pixel),
                                  height=(2.0 * star['rmask_arcsec'] / pixel),
                                  angle=0.0)
                smask.set_facecolor(ORG(0.2))
                smask.set_edgecolor(ORG(1.0))
                smask.set_alpha(0.3)
                ax1.add_artist(smask)

            # Show stars
            ax1.scatter(gaia_results['x_pix'],
                        gaia_results['y_pix'],
                        c=ORG(1.0),
                        s=100, alpha=0.9, marker='+')

            ax1.set_xlim(0, image.shape[0])
            ax1.set_ylim(0, image.shape[1])

            return gaia_results, fig
        else:
            return gaia_results
    else:
        return None


def image_clean_up(img,
                   sig=None,
                   bad=None,
                   bkg_param_1={'bw': 20, 'bh': 20, 'fw': 5, 'fh': 5},
                   det_param_1={'thr': 5.0, 'minarea': 100, 'deb_n': 128,
                                'deb_c': 0.00001},
                   bkg_param_2={'bw': 60, 'bh': 60, 'fw': 5, 'fh': 5},
                   det_param_2={'thr': 5.0, 'minarea': 10, 'deb_n': 64, 'deb_c': 0.001},
                   det_param_3={'thr': 7.0, 'minarea': 10, 'deb_n': 48, 'deb_c': 0.01},
                   verbose=False,
                   visual=False,
                   diagnose=False):
    """
    Clean up the image.

    TODO:
        Should be absorbed by object for image later.
    """
    # Measure a very local sky to help detection and deblending
    # Notice that this will remove large scale, and low surface brightness features.
    bkg_1 = sep.Background(img, mask=bad, maskthresh=0,
                           bw=bkg_param_1['bw'], bh=bkg_param_1['bh'],
                           fw=bkg_param_1['fw'], fh=bkg_param_1['fh'])
    if verbose:
        print("# BKG 1: Mean Sky / RMS Sky = %10.5f / %10.5f" % (bkg_1.globalback,
                                                                 bkg_1.globalrms))

    # Subtract a local sky, detect and deblend objects
    obj_1, seg_1 = sep.extract(img - bkg_1.back(), det_param_1['thr'],
                               err=sig,
                               minarea=det_param_1['minarea'],
                               deblend_nthresh=det_param_1['deb_n'],
                               deblend_cont=det_param_1['deb_c'],
                               segmentation_map=True)
    if verbose:
        print("# DET 1: Detect %d objects" % len(obj_1))

    # Detect all pixels above the threshold
    obj_2, seg_2 = sep.extract(img, det_param_2['thr'],
                               err=sig,
                               minarea=det_param_2['minarea'],
                               deblend_nthresh=det_param_2['deb_n'],
                               deblend_cont=det_param_2['deb_c'],
                               segmentation_map=True)
    if verbose:
        print("# DET 2: Detect %d objects" % len(obj_2))

    # Estimate the background for generating noise image
    bkg_2 = sep.Background(img, mask=seg_2, maskthresh=0,
                           bw=bkg_param_2['bw'], bh=bkg_param_2['bh'],
                           fw=bkg_param_2['fw'], fh=bkg_param_2['fh'])
    if verbose:
        print("# BKG 2: Mean Sky / RMS Sky = %10.5f / %10.5f" % (bkg_2.globalback,
                                                                 bkg_2.globalrms))

    if sig is None:
        noise = np.random.normal(loc=bkg_2.globalback,
                                 scale=bkg_2.globalrms,
                                 size=img.shape)
    else:
        sky_val = bkg_2.back()
        sky_sig = bkg_2.rms()
        sky_sig[sky_sig <= 0] = 1E-8
        noise = np.random.normal(loc=sky_val,
                                 scale=sky_sig,
                                 size=img.shape)

    # Replace all detected pixels with noise
    img_noise_replace = copy.deepcopy(img)
    img_noise_replace[seg_2 > 0] = noise[seg_2 > 0]

    # Detect the faint objects left on the image
    obj_3, seg_3 = sep.extract(img_noise_replace, det_param_3['thr'],
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

    if seg_comb is not None:
        seg_comb[obj_cen_mask] = 0

    img_clean = copy.deepcopy(img)
    img_clean[seg_comb > 0] = noise[seg_comb > 0]

    if diagnose:
        everything = {"bkg_1": bkg_1,
                      "obj_1": obj_1, "seg_1": seg_1,
                      "obj_2": obj_2, "seg_2": seg_2,
                      "bkg_2": bkg_2,
                      "obj_3": obj_3, "seg_3": seg_3,
                      "noise": noise}
        if visual:
            return img_clean, everything, diagnose_image_clean(img_clean, everything)
        else:
            return img_clean, everything
    else:
        return img_clean


def diagnose_image_clean(img_clean, everything, scale_bar_length=2.0):
    """
    Generate a QA plot for image clean.
    """
    #---------------------------------------------------------------------------#
    fig = plt.figure(figsize=(18, 12))
    fig.subplots_adjust(left=0.01, right=0.99,
                        bottom=0.01, top=0.99,
                        wspace=0.00, hspace=0.00)

    ax1 = plt.subplot(2,3,1)
    if everything['bkg_1'] is not None:
        ax1 = display_single(everything['bkg_1'].back(),
                             ax=ax1,
                             contrast=0.20,
                             scale_bar_length=scale_bar_length,
                             color_bar=True)

    ax2 = plt.subplot(2,3,2)
    if everything['seg_1'] is not None:
        ax1 = display_single(everything['seg_1'],
                             ax=ax2,
                             contrast=0.10,
                             scale_bar_length=scale_bar_length,
                             scale_bar_color='k',
                             cmap=SEG_CMAP,
                             scale='none',
                             stretch='linear')

    ax3 = plt.subplot(2,3,3)
    if everything['bkg_2'] is not None:
        ax3 = display_single(everything['bkg_2'].back(),
                             ax=ax3,
                             contrast=0.20,
                             scale_bar_length=scale_bar_length,
                             color_bar=True)

    ax4 = plt.subplot(2,3,4)
    if everything['seg_2'] is not None:
        ax4 = display_single(everything['seg_2'],
                             ax=ax4,
                             contrast=0.10,
                             scale_bar_length=scale_bar_length,
                             scale_bar_color='k',
                             scale='none',
                             cmap=SEG_CMAP,
                             stretch='linear')

    ax5 = plt.subplot(2,3,5)
    if everything['seg_3'] is not None:
        ax5 = display_single(everything['seg_3'],
                             ax=ax5,
                             contrast=0.10,
                             scale_bar_length=scale_bar_length,
                             scale_bar_color='k',
                             scale='none',
                             cmap=SEG_CMAP,
                             stretch='linear')

    ax6 = plt.subplot(2,3,6)
    ax6 = display_single(img_clean,
                         ax=ax6,
                         contrast=0.20,
                         scale_bar_length=scale_bar_length,
                         color_bar=True)

    return fig
