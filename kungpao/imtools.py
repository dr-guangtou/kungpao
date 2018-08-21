"""Useful tool for image reduction."""

from __future__ import print_function, absolute_import, division

import os
import copy

import numpy as np

import scipy.ndimage as ndimage
from scipy.ndimage.filters import gaussian_filter

from astropy.io import fits
from astropy.table import Table
from astropy.nddata import Cutout2D

from photutils import DAOStarFinder, IRAFStarFinder

import sep

from .display import diagnose_image_clean, diagnose_image_mask
from .query import image_gaia_stars

__all__ = ['img_cutout', 'get_pixel_value', 'seg_remove_cen_obj',
           'seg_index_cen_obj', 'seg_remove_obj', 'seg_index_obj',
           'parse_reg_ellipse', 'img_clean_up', 'seg_to_mask',
           'combine_mask', 'img_obj_mask', 'psfex_extract',
           'gaia_star_mask', 'iraf_star_mask', 'img_noise_map_conv', 
           'mask_high_sb_pixels', 'img_replace_with_noise',
           'detect_high_sb_objects', 'img_sigma_clipping', 
           'get_peak_mu', 'get_avg_mu']


def gaia_star_mask(img, wcs, pixel=0.168, mask_a=694.7, mask_b=4.04, 
                   size_buffer=1.4, gaia_bright=18.0,
                   factor_b=1.3, factor_f=1.9):
    """Find stars using Gaia and mask them out if necessary.
    
    Using the stars found in the GAIA TAP catalog, we build a bright star mask following
    similar procedure in Coupon et al. (2017). 

    We separate the GAIA stars into bright (G <= 18.0) and faint (G > 18.0) groups, and 
    apply different parameters to build the mask.
    """
    gaia_stars = image_gaia_stars(img, wcs, pixel=pixel, 
                                  mask_a=mask_a, mask_b=mask_b,
                                  verbose=False, visual=False, 
                                  size_buffer=size_buffer)
    
    # Make a mask image
    msk_star = np.zeros(img.shape).astype('uint8')

    gaia_b = gaia_stars[gaia_stars['phot_g_mean_mag'] <= gaia_bright]
    sep.mask_ellipse(msk_star, gaia_b['x_pix'], gaia_b['y_pix'], 
                     gaia_b['rmask_arcsec'] / factor_b / pix, 
                     gaia_b['rmask_arcsec'] / factor_b / pix, 0.0, r=1.0)

    gaia_f = gaia_stars[gaia_stars['phot_g_mean_mag'] > gaia_bright]
    sep.mask_ellipse(msk_star, gaia_f['x_pix'], gaia_f['y_pix'], 
                     gaia_f['rmask_arcsec'] / factor_f / pix, 
                     gaia_f['rmask_arcsec'] / factor_f / pix, 0.0, r=1.0)
    
    return gaia_stars, msk_star


def img_noise_map_conv(img, sig, fwhm=1.0, thr_ini=2.5, 
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
                                   minarea=minarea_ini, 
                                   deblend_nthresh=deb_thr_ini,
                                   deblend_cont=deb_cont_ini,  
                                   segmentation_map=True)

    print("# Initial detection picks up %d objects" % len(obj_ini))

    # Get an initial object mask
    msk_ini_conv = seg_to_mask(seg_ini, sigma=3.0, msk_max=1.0, msk_thr=0.01)

    # First try of background
    bkg_ini_conv = sep.Background(img_conv, mask=msk_ini_conv, 
                                  bw=bw_ini, bh=bh_ini, fw=fw_ini, fh=fh_ini)

    # Correct the background 
    img_conv_cor = img_conv - bkg_ini_conv.back()

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
    bkg_glb_conv_noise = np.random.normal(loc=bkg_glb_conv.back(), 
                                          scale=bkg_glb_conv.rms(), 
                                          size=img_conv_cor.shape)

    bkg_glb_noise = np.random.normal(loc=bkg_glb.back(), 
                                     scale=bkg_glb.rms(), 
                                     size=img.shape)
    
    return img_conv_cor, bkg_glb_conv_noise, bkg_glb_noise


def iraf_star_mask(img, threshold, fwhm, bw=500, bh=500, fw=4, fh=4,
                   zeropoint=27.0, mag_lim=24.0):
    """Detect all stellar objects using DAOFind and IRAFStarFinder."""
    bkg_star = sep.Background(img, bw=bw, bh=bh, fw=fw, fh=fh)

    dao_finder = DAOStarFinder(fwhm=fwhm, threshold=threshold * bkg_star.globalrms)
    irf_finder = IRAFStarFinder(fwhm=fwhm, threshold=threshold * bkg_star.globalrms)

    stars_dao = dao_finder(img - bkg_star.globalback)
    stars_irf = irf_finder(img - bkg_star.globalback)
    
    msk_star = np.zeros(img.shape).astype('uint8')
    
    stars_irf_use = stars_irf[(-2.5 * np.log10(stars_irf['flux']) + zeropoint) <= mag_lim]
    stars_dao_use = stars_dao[(-2.5 * np.log10(stars_dao['flux']) + zeropoint) <= mag_lim]

    sep.mask_ellipse(msk_star, 
                     stars_irf_use['xcentroid'], stars_irf_use['ycentroid'], 
                     fwhm, fwhm, 0.0, r=1.0)

    sep.mask_ellipse(msk_star, 
                     stars_dao_use['xcentroid'], stars_dao_use['ycentroid'], 
                     fwhm, fwhm, 0.0, r=1.0)
    
    return stars_dao_use, stars_irf_use, msk_star


def img_cutout(img, wcs, ra, dec, size=60.0, pix=0.168,
               prefix='img_cutout'):
    """Generate image cutout with updated WCS information."""
    # imgsize in unit of arcsec
    cutout_size = size / pix

    cen_x, cen_y = wcs.wcs_world2pix(ra, dec, 0)
    cen_pos = (int(cen_x), int(cen_y))

    # Generate cutout
    cutout = Cutout2D(img, cen_pos, (cutout_size, cutout_size),
                      wcs=wcs)

    # Update the header
    header = cutout.wcs.to_header()

    # Build a HDU
    hdu = fits.PrimaryHDU(header=header)
    hdu.data = cutout.data

    # Save FITS image
    fits_file = prefix + '.fits'
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

    xc = np.array(xc, dtype=np.float32)
    yc = np.array(yc, dtype=np.float32)
    ra = np.array(ra, dtype=np.float32)
    rb = np.array(rb, dtype=np.float32)
    theta = np.array(theta, dtype=np.float32)

    return xc, yc, ra, rb, theta, coord_type


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


def psfex_extract(psfex_file, row, col):
    """Extract PSF image from PSFex result."""
    try:
        import psfex
    except ImportError:
        raise Exception("Need to install PSFex library first!")

    return psfex.PSFEx(psfex_file).get_rec(row, col)


def mask_high_sb_pixels(img, pix=0.168, zeropoint=27.0,
                        mu_threshold_1=22.0, mu_threshold_2=23.0,
                        mu_sig_1=8.0, mu_sig_2=1.0):
    """Build a mask for all pixels above certain surface brightness level."""
    msk_high_mu_1 = (zeropoint - 2.5 * np.log10(img / (pix ** 2))) < mu_threshold_1
    msk_high_mu_2 = (zeropoint - 2.5 * np.log10(img / (pix ** 2))) < mu_threshold_2

    msk_high_mu_1_conv = seg_to_mask(msk_high_mu_1.astype(int), sigma=mu_sig_1, 
                                     msk_max=1000.0, msk_thr=0.01)
    msk_high_mu_2_conv = seg_to_mask(msk_high_mu_2.astype(int), sigma=mu_sig_2, 
                                     msk_max=1000.0, msk_thr=0.01)

    msk_high_mu = ((msk_high_mu_1_conv > 0 ) | (msk_high_mu_2_conv > 0))
    
    return msk_high_mu


def img_replace_with_noise(img, msk, noise):
    """Replace the mask region with noise."""
    img_clean = copy.deepcopy(img)
    img_clean[msk] = noise[msk]
    
    return img_clean


def img_sigma_clipping(img, sig, ratio):
    """Return a mask for piexls above certain threshold."""
    return img > (ratio * sig)


def get_avg_mu(obj, pix=0.176, zero_point=27.0):
    """Get the average surface brightness of a SEP object."""
    return -2.5 * np.log10(obj['flux'] / 
                           (np.pi * obj['a'] * obj['b'] * 
                            (pix ** 2))) + zero_point


def get_peak_mu(obj, pix=0.176, zero_point=27.0):
    """Get the peak surface brightness of a SEP object."""
    return -2.5 * np.log10(obj['cpeak'] /
                           (pix ** 2.0)) + zero_point


def detect_high_sb_objects(img, sig, threshold=30.0, min_area=100,
                           deb_thr_hsig=128, deb_cont_hsig=0.0001,
                           mu_limit=23.0, sig_hsig_1=0.1, sig_hsig_2=4.0):
    """Detect all bright objects and mask them out."""
    # Step 1: Detect bright objects on the image 
    '''
    From Greco et al. 2018:

    Next, we find very bright sources by flagging all pixels that are at least 28σ above the 
    global background level for each patch; for a typical patch, this corresponds to the 
    brightest ∼2% of all pixels. 
    The background and its variance are estimated using several iterations of sigma clipping.

    In this work, we choose to detect two group of bright objects: 
    1:  > 20 sigma, size > 200
    2:  > 15 sigma, size > 10000
    '''
    # Object detection: high threshold, relative small minimum size
    obj_hsig, seg_hsig = sep.extract(img, threshold, err=sig, 
                                     minarea=min_area, 
                                     deblend_nthresh=deb_thr_hsig,
                                     deblend_cont=deb_cont_hsig,  
                                     segmentation_map=True)

    # Remove objects with low peak surface brightness
    idx_low_peak_mu = []
    obj_hsig = Table(obj_hsig)
    for idx, obj in enumerate(obj_hsig):
        if get_peak_mu(obj) >= mu_limit:
            seg_hsig[seg_hsig == (idx + 1)] = 0
            idx_low_peak_mu.append(idx)

    obj_hsig.remove_rows(idx_low_peak_mu)

    print("# Keep %d high surface brightness objects" % len(obj_hsig))

    # Generate a mask
    msk_hsig = seg_to_mask(seg_hsig, sigma=sig_hsig_1, msk_max=1000.0, msk_thr=0.01)
    msk_hsig_large = seg_to_mask(seg_hsig, sigma=sig_hsig_2, msk_max=1000.0, msk_thr=0.005)
    
    return obj_hsig, msk_hsig, msk_hsig_large

