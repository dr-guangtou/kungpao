"""Useful tool for image reduction."""

from __future__ import print_function, absolute_import, division

import os
import copy

import numpy as np

import scipy.ndimage as ndimage
from scipy.ndimage.filters import gaussian_filter

from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.table import Table, Column
from astropy.modeling import models, fitting

from photutils import DAOStarFinder, IRAFStarFinder

import sep

from .display import diagnose_image_clean, diagnose_image_mask
from .query import image_gaia_stars

__all__ = ['img_cutout', 'get_pixel_value', 'seg_remove_cen_obj',
           'seg_index_cen_obj', 'seg_remove_obj', 'seg_index_obj',
           'img_clean_up', 'seg_to_mask',
           'combine_mask', 'img_obj_mask', 'img_subtract_bright_star',
           'gaia_star_mask', 'iraf_star_mask', 'img_noise_map_conv',
           'mask_high_sb_pixels', 'img_replace_with_noise',
           'img_measure_background', 'img_sigma_clipping']


def gaia_star_mask(img, wcs, pix=0.168, mask_a=694.7, mask_b=4.04,
                   size_buffer=1.4, gaia_bright=18.0,
                   factor_b=1.3, factor_f=1.9):
    """Find stars using Gaia and mask them out if necessary.

    Using the stars found in the GAIA TAP catalog, we build a bright star mask following
    similar procedure in Coupon et al. (2017).

    We separate the GAIA stars into bright (G <= 18.0) and faint (G > 18.0) groups, and
    apply different parameters to build the mask.
    """
    gaia_stars = image_gaia_stars(img, wcs, pixel=pix,
                                  mask_a=mask_a, mask_b=mask_b,
                                  verbose=False, visual=False,
                                  size_buffer=size_buffer)

    # Make a mask image
    msk_star = np.zeros(img.shape).astype('uint8')

    if gaia_stars is not None:
        gaia_b = gaia_stars[gaia_stars['phot_g_mean_mag'] <= gaia_bright]
        sep.mask_ellipse(msk_star, gaia_b['x_pix'], gaia_b['y_pix'],
                        gaia_b['rmask_arcsec'] / factor_b / pix,
                        gaia_b['rmask_arcsec'] / factor_b / pix, 0.0, r=1.0)

        gaia_f = gaia_stars[gaia_stars['phot_g_mean_mag'] > gaia_bright]
        sep.mask_ellipse(msk_star, gaia_f['x_pix'], gaia_f['y_pix'],
                        gaia_f['rmask_arcsec'] / factor_f / pix,
                        gaia_f['rmask_arcsec'] / factor_f / pix, 0.0, r=1.0)

        return gaia_stars, msk_star

    return None, msk_star


def img_noise_map_conv(img, sig, fwhm=1.0, thr_ini=2.5, mask=None,
                       bw_ini=80, bh_ini=80, fw_ini=4, fh_ini=4,
                       bw_glb=240, bh_glb=240, fw_glb=6, fh_glb=6,
                       deb_thr_ini=64, deb_cont_ini=0.001, minarea_ini=25):
    """Identify all objects on the image, and generate a noise map."""
    # Step 1: Image convolution:
    '''
    From Greco et al. 2018:

    Smoothing the image with a circular Gaussian matched to the rms width of the
    point-spread function (PSF).
    Image convolution maximizes the ratio of a source’s peak signal to the local noise level
    (e.g., Irwin 1985; Akhlaghi & Ichikawa 2015), and the PSF scale is formally optimal for the
    idealized case of detecting isolated point sources (Bosch et al. 2017).
    '''
    # Convolve the image with a circular Gaussian kernel with the size of PSF
    # Image convolution
    img_conv = gaussian_filter(img, fwhm / 2.355)

    # Step 2: Detect all objects and build a mask for background measurements
    '''
    Try to detect all pixels above 3 sigma on the image to build a mask.
    Then use such mask to measure the background with different levels of
    smoothing.
    '''
    # Detect all objects on the image
    obj_ini, seg_ini = sep.extract(img_conv, thr_ini, err=sig,
                                   minarea=minarea_ini, mask=mask,
                                   deblend_nthresh=deb_thr_ini,
                                   deblend_cont=deb_cont_ini,
                                   segmentation_map=True)

    print("# Initial detection picks up %d objects" % len(obj_ini))

    # Get an initial object mask
    msk_ini_conv = seg_to_mask(seg_ini, sigma=3.0, msk_max=1.0, msk_thr=0.01)
    if mask is not None:
        msk_ini_conv = (msk_ini_conv | mask)

    # First try of background
    try:
        bkg_ini_conv = sep.Background(img_conv, mask=msk_ini_conv,
                                      bw=bw_ini, bh=bh_ini, fw=fw_ini, fh=fh_ini)

        # Correct the background
        img_conv_cor = img_conv - bkg_ini_conv.back()
    except Exception:
        img_conv_cor = img_conv

    # First try of global background
    bkg_glb_conv = sep.Background(img_conv_cor, mask=msk_ini_conv,
                                  bw=bw_glb, bh=bh_glb, fw=fw_glb, fh=fh_glb)

    bkg_glb = sep.Background(img, mask=msk_ini_conv,
                             bw=bw_glb, bh=bh_glb, fw=fw_glb, fh=fh_glb)

    # Step 3: Generate a noise map using the global background properties
    '''
    The noise values will be used to replace the pixels of bright objects.
    '''
    # Generate a noise map based on the initial background map
    # Replace the negative or zero variance region with huge noise level
    sig_conv = bkg_glb_conv.rms()
    sig_conv[sig_conv <= 0] = 1E-10
    bkg_glb_conv_noise = np.random.normal(loc=bkg_glb_conv.back(),
                                          scale=sig_conv,
                                          size=img_conv_cor.shape)

    sig = bkg_glb.rms()
    sig[sig <= 0] = 1E-10
    bkg_glb_noise = np.random.normal(loc=bkg_glb.back(),
                                     scale=sig,
                                     size=img.shape)

    return img_conv_cor, bkg_glb_conv_noise, bkg_glb_noise


def iraf_star_mask(img, threshold, fwhm, mask=None, bw=500, bh=500, fw=4, fh=4,
                   zeropoint=27.0, mag_lim=24.0):
    """Detect all stellar objects using DAOFind and IRAFStarFinder."""
    bkg_star = sep.Background(img, mask=mask, bw=bw, bh=bh, fw=fw, fh=fh)

    dao_finder = DAOStarFinder(fwhm=fwhm, threshold=threshold * bkg_star.globalrms)
    irf_finder = IRAFStarFinder(fwhm=fwhm, threshold=threshold * bkg_star.globalrms)

    stars_dao = dao_finder(img - bkg_star.globalback)
    stars_irf = irf_finder(img - bkg_star.globalback)

    msk_star = np.zeros(img.shape).astype('uint8')

    if len(stars_irf) > 0:
        stars_irf_use = stars_irf[(-2.5 * np.log10(stars_irf['flux']) + zeropoint) <= mag_lim]
        sep.mask_ellipse(msk_star,
                         stars_irf_use['xcentroid'], stars_irf_use['ycentroid'],
                         fwhm, fwhm, 0.0, r=1.0)
    else:
        stars_irf_use = None

    if len(stars_dao) > 0:
        stars_dao_use = stars_dao[(-2.5 * np.log10(stars_dao['flux']) + zeropoint) <= mag_lim]
        sep.mask_ellipse(msk_star,
                        stars_dao_use['xcentroid'], stars_dao_use['ycentroid'],
                        fwhm, fwhm, 0.0, r=1.0)
    else:
        stars_dao_use = None

    return stars_dao_use, stars_irf_use, msk_star


def img_cutout(img, wcs, coord_1, coord_2, size=60.0, pix=0.168,
               prefix='img_cutout', pixel_unit=False, out_dir=None,
               save=True):
    """Generate image cutout with updated WCS information.

    ----------
    Parameters:
        pixel_unit: boolen, optional
            When True, coord_1, cooord_2 becomes X, Y pixel coordinates.
            Size will also be treated as in pixels.
    """
    if not pixel_unit:
        # imgsize in unit of arcsec
        cutout_size = np.asarray(size) / pix
        cen_x, cen_y = wcs.wcs_world2pix(coord_1, coord_2, 0)
    else:
        cutout_size = np.asarray(size)
        cen_x, cen_y = coord_1, coord_2

    cen_pos = (int(cen_x), int(cen_y))

    # Generate cutout
    cutout = Cutout2D(img, cen_pos, cutout_size, wcs=wcs)

    # Update the header
    header = cutout.wcs.to_header()

    # Build a HDU
    hdu = fits.PrimaryHDU(header=header)
    hdu.data = cutout.data

    # Save FITS image
    if save:
        fits_file = prefix + '.fits'
        if out_dir is not None:
            fits_file = os.path.join(out_dir, fits_file)

        hdu.writeto(fits_file, overwrite=True)

    return cutout


def seg_to_mask(seg, sigma=5.0, msk_max=1000.0, msk_thr=0.01):
    """Convert the segmentation array into an array.

    Parameters
    ----------
        sigma:  Sigma of the Gaussian Kernel

    """
    # Convolve the mask image with a gaussian kernel
    msk_conv = ndimage.gaussian_filter(((seg.copy() > 0) * msk_max),
                                       sigma=sigma, order=0)
    msk_bool = msk_conv > (msk_thr * msk_max)

    return msk_bool.astype('uint8')


def combine_mask(msk1, msk2):
    """Combine two mask images."""
    if (msk1.shape[0] != msk2.shape[0]) or (msk1.shape[1] != msk2.shape[1]):
        raise Exception("### The two masks need to have the same shape!")

    return (msk1 > 0) | (msk2 > 0)


def get_pixel_value(img, wcs, ra, dec):
    """Return the pixel value from image based on RA, DEC.

    TODO:
        Should be absorbed into the image object later

    Parameters
    ----------
        img     : 2-D data array
        wcs     : WCS from the image header
        ra, dec : coordinates, can be array

    """
    px, py = wcs.wcs_world2pix(ra, dec, 0)

    import collections
    if not isinstance(px, collections.Iterable):
        pix = img[int(py), int(px)]
    else:
        pix = [img[int(y), int(x)] for x, y in zip(px, py)]

    return np.asarray(pix)


def seg_remove_cen_obj(seg):
    """Remove the central object from the segmentation.

    TODO
    ----
        Should be absorbed by objects for segmentation image

    """
    seg_copy = copy.deepcopy(seg)
    seg_copy[seg == seg[int(seg.shape[0] / 2.0), int(seg.shape[1] / 2.0)]] = 0

    return seg_copy


def seg_index_cen_obj(seg):
    """Remove the index array for central object.

    TODO
    ----
        Should be absorbed by objects for segmentation image

    """
    cen_obj = seg[int(seg.shape[0] / 2.0), int(seg.shape[1] / 2.0)]
    if cen_obj == 0:
        return None

    return seg == cen_obj


def seg_remove_obj(seg, x, y):
    """Remove an object from the segmentation given its coordinate.

    TODO:
        Should be absorbed by objects for segmentation image
    """
    seg_copy = copy.deepcopy(seg)
    seg_copy[seg == seg[int(x), int(y)]] = 0

    return seg_copy


def seg_index_obj(seg, x, y):
    """Remove the index array for an object given its location.

    TODO
    ----
        Should be absorbed by objects for segmentation image

    """
    obj = seg[int(x), int(y)]
    if obj == 0:
        return None

    return seg == obj


def img_clean_up(
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
    """Clean up the image.

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
        return img_clean, everything

    return img_clean


def img_replace_nan(fits_file, index_hdu=0, inf=True, nan=True, neg_inf=True,
                    replace=0.0, fits_new=None):
    """Replace the infinite value on image."""
    hdu = fits.open(fits_file)
    data = hdu[index_hdu].data

    if nan:
        data[data == np.nan] = replace
    if inf:
        data[data == np.inf] = replace
    if neg_inf:
        data[data == -np.inf] = replace

    if fits_new is None:
        hdu.writeto(fits_file, overwrite=True)
    else:
        hdu.writeto(fits_new, overwrite=True)

    hdu.close()

    return


def img_obj_mask(img, sig=None, bad=None,
                 bkg_param_1={'bw': 20, 'bh': 20, 'fw': 3, 'fh': 3},
                 det_param_1={'thr': 1.5, 'minarea': 40, 'deb_n': 128,
                              'deb_c': 0.00001},
                 bkg_param_2={'bw': 150, 'bh': 150, 'fw': 7, 'fh': 7},
                 det_param_2={'thr': 2.0, 'minarea': 20,
                              'deb_n': 64, 'deb_c': 0.001},
                 bkg_param_3={'bw': 60, 'bh': 60, 'fw': 5, 'fh': 5},
                 det_param_3={'thr': 3.5, 'minarea': 10,
                              'deb_n': 64, 'deb_c': 0.005},
                 sig_msk_1=3.0, sig_msk_2=5.0, sig_msk_3=2.0,
                 thr_msk_1=0.01, thr_msk_2=0.01, thr_msk_3=0.01,
                 object_remove=None,
                 verbose=False, visual=False, diagnose=False, **kwargs):
    """Make object mask."""
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
        img - bkg_2.back(),
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

    # Index for the central object
    if object_remove is None:
        seg_1 = seg_remove_cen_obj(seg_1)
        seg_2 = seg_remove_cen_obj(seg_2)
        seg_3 = seg_remove_cen_obj(seg_3)
    else:
        # TODO: Make it work for an array of objects
        seg_1 = seg_remove_obj(seg_1, object_remove[1], object_remove[0])
        seg_2 = seg_remove_obj(seg_2, object_remove[1], object_remove[0])
        seg_3 = seg_remove_obj(seg_3, object_remove[1], object_remove[0])

    seg_mask_1 = seg_to_mask(seg_1, sigma=sig_msk_1, msk_thr=thr_msk_1)
    seg_mask_2 = seg_to_mask(seg_2, sigma=sig_msk_2, msk_thr=thr_msk_2)
    seg_mask_3 = seg_to_mask(seg_3, sigma=sig_msk_3, msk_thr=thr_msk_3)

    img_mask = (seg_mask_1 | seg_mask_2 | seg_mask_3)

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
            return img_mask, everything, diagnose_image_mask(
                img_mask, everything, **kwargs)
        return img_mask, everything

    return img_mask


def mask_high_sb_pixels(img, pix=0.168, zeropoint=27.0,
                        mu_threshold_1=22.0, mu_threshold_2=23.0,
                        mu_sig_1=8.0, mu_sig_2=1.0):
    """Build a mask for all pixels above certain surface brightness level."""
    np.seterr(invalid='ignore', divide='ignore')
    msk_high_mu_1 = (zeropoint - 2.5 * np.log10(img / (pix ** 2))) < mu_threshold_1
    msk_high_mu_2 = (zeropoint - 2.5 * np.log10(img / (pix ** 2))) < mu_threshold_2

    msk_high_mu_1_conv = seg_to_mask(msk_high_mu_1.astype(int), sigma=mu_sig_1,
                                     msk_max=1000.0, msk_thr=0.01)
    msk_high_mu_2_conv = seg_to_mask(msk_high_mu_2.astype(int), sigma=mu_sig_2,
                                     msk_max=1000.0, msk_thr=0.01)

    msk_high_mu = ((msk_high_mu_1_conv > 0) | (msk_high_mu_2_conv > 0))

    return msk_high_mu


def img_replace_with_noise(img, msk, noise):
    """Replace the mask region with noise."""
    img_clean = copy.deepcopy(img)
    img_clean[msk] = noise[msk]

    return img_clean


def img_sigma_clipping(img, sig, ratio):
    """Return a mask for piexls above certain threshold."""
    return img > (ratio * sig)


def _check_kwargs(kwargs, key, default):
    """Check if key available in kwargs, and go back to default if not."""
    return kwargs[key] if (key in kwargs) else default


def img_measure_background(img, use_sep=True, **kwargs):
    """Estimate sky background of an image.

    For SEP, available parameters are:

        sep_kwargs = {'mask': None,
                      'bw': 20, 'bh': 20, 'fw': 3, 'fh':3, }

    For Photutils, available parameters are:

        phot_kwargs = {'bkg': 'median', 'rms': 'mad', 'mask': None,
                       'clip': True, 'sigma': 3.0, 'iters': 10,
                       'bw': 20, 'bh': 20, 'fw': 3, 'fh':3, }
    """
    if use_sep:
        # Use SEP for background
        sep_back = sep.Background(img, **kwargs)
        return sep_back.back(), sep_back.rms()
    else:
        # Use the photutils.background instead
        if _check_kwargs(kwargs, 'clip', True):
            sigma_clip = SigmaClip(sigma=_check_kwargs(kwargs, 'sigma', 3.0),
                                   iters=_check_kwargs(kwargs, 'iters', 3))
        else:
            sigma_clip = None

        bkg = _check_kwargs(kwargs, 'bkg', 'sextractor')
        rms = _check_kwargs(kwargs, 'rms', 'biweight')

        if bkg == 'biweight':
            from photutils import BiweightLocationBackground
            bkg_estimator = BiweightLocationBackground()
        elif bkg == 'sextractor':
            from photutils import SExtractorBackground
            bkg_estimator = SExtractorBackground()
        elif bkg == 'mmm':
            from photutils import MMMBackground
            bkg_estimator = MMMBackground()
        elif bkg == 'median':
            from photutils import MedianBackground
            bkg_estimator = MedianBackground()
        else:
            raise Exception("# Wrong choice of background estimator!")

        if rms == 'biweight':
            from photutils import BiweightScaleBackgroundRMS
            rms_estimator = BiweightScaleBackgroundRMS()
        elif rms == 'mad':
            from photutils import MADStdBackgroundRMS
            rms_estimator = MADStdBackgroundRMS()
        elif rms == 'std':
            from photutils import StdBackgroundRMS
            rms_estimator = StdBackgroundRMS()
        else:
            raise Exception("# Wrong choice of RMS estimator!")

        bw = kwargs['bw'] if ('bw' in kwargs) else 3
        bh = kwargs['bh'] if ('bh' in kwargs) else 3
        fw = kwargs['fw'] if ('fw' in kwargs) else 3
        fh = kwargs['fh'] if ('fh' in kwargs) else 3

        bkg = Background2D(img,
                           (_check_kwargs(kwargs, 'bh', 100), _check_kwargs(kwargs, 'bw', 100)),
                           filter_size=(_check_kwargs(kwargs, 'fh', 3), _check_kwargs(kwargs, 'fw', 3)),
                           mask=_check_kwargs(kwargs, 'mask', None),
                           sigma_clip=sigma_clip,
                           bkg_estimator=bkg_estimator,
                           bkgrms_estimator=rms_estimator)

        return bkg.background, bkg.background_rms


def img_subtract_bright_star(img, star, x_col='x_pix', y_col='y_pix',
                             gamma=5.0, alpha=6.0, sig=None,
                             x_buffer=4, y_buffer=4, img_maxsize=300):
    """Subtract a bright star from image using a Moffat model."""
    # Use the SLSQP fitter
    fitter_use = fitting.SLSQPLSQFitter()
    
    # Image dimension
    img_h, img_w = img.shape

    # Only fit the stars on the image
    if ((0 + x_buffer < int(star[x_col]) < img_w - x_buffer) and 
        (0 + y_buffer < int(star[y_col]) < img_h - y_buffer)):
        # Get the center of the star
        x_cen, y_cen = int(star[x_col]), int(star[y_col])

        # If the image is too big, cut a part of it
        if (img_h >= img_maxsize) or (img_w >= img_maxsize):
            x_0 = int(x_cen - img_maxsize / 2.0) if (x_cen - img_maxsize / 2.0) > 0 else 0
            x_1 = int(x_cen + img_maxsize / 2.0) if (x_cen + img_maxsize / 2.0) < img_w else (img_w - 1)
            y_0 = int(y_cen - img_maxsize / 2.0) if (y_cen - img_maxsize / 2.0) > 0 else 0
            y_1 = int(y_cen + img_maxsize / 2.0) if (y_cen + img_maxsize / 2.0) < img_h else (img_h - 1)
            x_cen, y_cen = (x_cen - x_0), (y_cen - y_0) 
        else:
            x_0, x_1 = 0, img_w + 1
            y_0, y_1 = 0, img_h + 1
        
        # Determine the weights for the fitting
        img_use = copy.deepcopy(img[y_0:y_1, x_0:x_1])

        weights = (1.0 / sig[y_0:y_1, x_0:x_1]) if (sig is not None) else None
    
        # X, Y grids
        y_size, x_size = img_use.shape
        y_arr, x_arr = np.mgrid[:y_size, :x_size] 
        
        # Initial the Moffat model
        p_init = models.Moffat2D(x_0=x_cen, y_0=y_cen, 
                                 amplitude=(img_use[int(x_cen), int(y_cen)]),
                                 gamma=gamma, alpha=alpha,
                                 bounds={'x_0': [x_cen - x_buffer, x_cen + x_buffer], 
                                         'y_0': [y_cen - y_buffer, y_cen + y_buffer]})
        
        try:
            with np.errstate(all='ignore'):
                best_fit = fitter_use(p_init, x_arr, y_arr, img_use, weights=weights, 
                                      verblevel=0)
                
                img_new = copy.deepcopy(img)
                img_new[y_0:y_1, x_0:x_1] -= best_fit(x_arr, y_arr)
                
            return img_new
        
        except Exception:
            warnings.warn('# Star fitting failed!')
            return img
    else:
        return img
