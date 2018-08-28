"""Detect objects on the image."""

from __future__ import (print_function,
                        division,
                        absolute_import)

import copy

import numpy as np
import scipy.stats as st

import scipy.ndimage as ndimage
from scipy.ndimage.filters import gaussian_filter

from astropy.io import fits
from astropy.table import Table, Column

import sep

from kungpao.imtools import *


__all__ = ['sep_detection', 'simple_convolution_kernel', 'get_gaussian_kernel',
           'detect_high_sb_objects', 'detect_low_sb_objects', 
           'obj_avg_mu', 'obj_avg_mu']


def simple_convolution_kernel(kernel):
    """Precomputed convolution kernel for the SEP detections."""
    if kernel == 1:
        # Tophat_3.0_3x3
        convKer = np.asarray([[0.560000, 0.980000,
                               0.560000], [0.980000, 1.000000, 0.980000],
                              [0.560000, 0.980000, 0.560000]])
    elif kernel == 2:
        # Topcat_4.0_5x5
        convKer = np.asarray(
            [[0.000000, 0.220000, 0.480000, 0.220000,
              0.000000], [0.220000, 0.990000, 1.000000, 0.990000, 0.220000],
             [0.480000, 1.000000, 1.000000, 1.000000,
              0.480000], [0.220000, 0.990000, 1.000000, 0.990000, 0.220000],
             [0.000000, 0.220000, 0.480000, 0.220000, 0.000000]])
    elif kernel == 3:
        # Topcat_5.0_5x5
        convKer = np.asarray(
            [[0.150000, 0.770000, 1.000000, 0.770000,
              0.150000], [0.770000, 1.000000, 1.000000, 1.000000, 0.770000],
             [1.000000, 1.000000, 1.000000, 1.000000,
              1.000000], [0.770000, 1.000000, 1.000000, 1.000000, 0.770000],
             [0.150000, 0.770000, 1.000000, 0.770000, 0.150000]])
    elif kernel == 4:
        # Gaussian_3.0_5x5
        convKer = np.asarray(
            [[0.092163, 0.221178, 0.296069, 0.221178,
              0.092163], [0.221178, 0.530797, 0.710525, 0.530797, 0.221178],
             [0.296069, 0.710525, 0.951108, 0.710525,
              0.296069], [0.221178, 0.530797, 0.710525, 0.530797, 0.221178],
             [0.092163, 0.221178, 0.296069, 0.221178, 0.092163]])
    elif kernel == 5:
        # Gaussian_4.0_7x7
        convKer = np.asarray([[
            0.047454, 0.109799, 0.181612, 0.214776, 0.181612, 0.109799,
            0.047454
        ], [
            0.109799, 0.254053, 0.420215, 0.496950, 0.420215, 0.254053,
            0.109799
        ], [
            0.181612, 0.420215, 0.695055, 0.821978, 0.695055, 0.420215,
            0.181612
        ], [
            0.214776, 0.496950, 0.821978, 0.972079, 0.821978, 0.496950,
            0.214776
        ], [
            0.181612, 0.420215, 0.695055, 0.821978, 0.695055, 0.420215,
            0.181612
        ], [
            0.109799, 0.254053, 0.420215, 0.496950, 0.420215, 0.254053,
            0.109799
        ], [
            0.047454, 0.109799, 0.181612, 0.214776, 0.181612, 0.109799,
            0.047454
        ]])
    elif kernel == 6:
        # Gaussian_5.0_9x9
        convKer = np.asarray([[
            0.030531, 0.065238, 0.112208, 0.155356, 0.173152, 0.155356,
            0.112208, 0.065238, 0.030531
        ], [
            0.065238, 0.139399, 0.239763, 0.331961, 0.369987, 0.331961,
            0.239763, 0.139399, 0.065238
        ], [
            0.112208, 0.239763, 0.412386, 0.570963, 0.636368, 0.570963,
            0.412386, 0.239763, 0.112208
        ], [
            0.155356, 0.331961, 0.570963, 0.790520, 0.881075, 0.790520,
            0.570963, 0.331961, 0.155356
        ], [
            0.173152, 0.369987, 0.636368, 0.881075, 0.982004, 0.881075,
            0.636368, 0.369987, 0.173152
        ], [
            0.155356, 0.331961, 0.570963, 0.790520, 0.881075, 0.790520,
            0.570963, 0.331961, 0.155356
        ], [
            0.112208, 0.239763, 0.412386, 0.570963, 0.636368, 0.570963,
            0.412386, 0.239763, 0.112208
        ], [
            0.065238, 0.139399, 0.239763, 0.331961, 0.369987, 0.331961,
            0.239763, 0.139399, 0.065238
        ], [
            0.030531, 0.065238, 0.112208, 0.155356, 0.173152, 0.155356,
            0.112208, 0.065238, 0.030531
        ]])
    else:
        raise Exception("### More options will be available in the future")

    return convKer


def get_gaussian_kernel(size, sig):
    """Return a 2D Gaussian kernel array.

    Based on https://stackoverflow.com/questions/29731726
    """
    interval = (2 * size + 1.) / size
    x = np.linspace(-size - interval / 2.,
                    size + interval / 2., size + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()

    return kernel


def sep_detection(img, threshold, kernel=4, err=None, use_sig=True,
                  subtract_bkg=True, return_bkg=True, return_seg=True,
                  bkg_kwargs=None, **det_kwargs):
    """Object detection using SEP.

    Example of bkg_kwargs:
        {'mask': None, 'bw': 100, 'bh': 100, 'fw': 100, 'fh': 100 }

    Example of det_kwargs:
        {'minarea': 10, 'deblend_nthreshs': 32,
         'deblend_conts': 0.0001, 'filter_type': 'matched'}

    """
    # Determine the kernel used in detection
    if isinstance(kernel, int):
        filter_kernel = simple_convolution_kernel(kernel)
    elif isinstance(kernel, (list, tuple, np.ndarray)):
        filter_kernel = get_gaussian_kernel(kernel[0], kernel[1])
    else:
        raise Exception("Wrong choice for convolution kernel")

    # Estimate background, subtract it if necessary
    if bkg_kwargs is not None:
        bkg, rms = img_measure_background(img, use_sep=True, **bkg_kwargs)
    else:
        bkg, rms = img_measure_background(img, use_sep=True)

    if subtract_bkg:
        img -= bkg

    # If no error or variance array is provided, use the global rms of sky
    if err is None:
        threshold *= rms

    # Make the detection using sigma or variance array
    if use_sig:
        results = sep.extract(img, threshold, err=err,
                              filter_kernel=filter_kernel,
                              segmentation_map=return_seg, **det_kwargs)
    else:
        results = sep.extract(img, threshold, var=err,
                              filter_kernel=filter_kernel,
                              segmentation_map=return_seg, **det_kwargs)

    if return_seg:
        obj, seg = results
        if return_bkg:
            return obj, seg, bkg
        return obj, seg
    else:
        if return_bkg:
            return results, bkg
        return results
        

def obj_avg_mu(obj, pix=0.176, zero_point=27.0):
    """Get the average surface brightness of a SEP object."""
    return -2.5 * np.log10(obj['flux'] /
                           (np.pi * obj['a'] * obj['b'] *
                            (pix ** 2))) + zero_point


def obj_peak_mu(obj, pix=0.176, zero_point=27.0):
    """Get the peak surface brightness of a SEP object."""
    return -2.5 * np.log10(obj['cpeak'] /
                           (pix ** 2.0)) + zero_point


def detect_high_sb_objects(img, sig, threshold=30.0, min_area=100, mask=None,
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
                                     minarea=min_area, mask=mask,
                                     deblend_nthresh=deb_thr_hsig,
                                     deblend_cont=deb_cont_hsig,
                                     segmentation_map=True)

    # Remove objects with low peak surface brightness
    idx_low_peak_mu = []
    obj_hsig = Table(obj_hsig)
    for idx, obj in enumerate(obj_hsig):
        if obj_peak_mu(obj) >= mu_limit:
            seg_hsig[seg_hsig == (idx + 1)] = 0
            idx_low_peak_mu.append(idx)

    obj_hsig.remove_rows(idx_low_peak_mu)

    print("# Keep %d high surface brightness objects" % len(obj_hsig))

    # Generate a mask
    msk_hsig = seg_to_mask(seg_hsig, sigma=sig_hsig_1, msk_max=1000.0, msk_thr=0.01)
    msk_hsig_large = seg_to_mask(seg_hsig, sigma=sig_hsig_2, msk_max=1000.0, msk_thr=0.005)

    return obj_hsig, msk_hsig, msk_hsig_large


def detect_low_sb_objects(img, threshold, sig, msk_hsig_1, msk_hsig_2, noise,
                          minarea=200, mask=None, deb_thr_lsig=64,
                          deb_cont_lsig=0.001, frac_mask=0.2):
    """Detect all the low threshold pixels."""
    # Detect the low sigma pixels on the image
    obj_lsig, seg_lsig = sep.extract(img, threshold, err=sig,
                                     minarea=minarea, mask=mask,
                                     deblend_nthresh=deb_thr_lsig,
                                     deblend_cont=deb_cont_lsig,
                                     segmentation_map=True)

    obj_lsig = Table(obj_lsig)
    obj_lsig.add_column(Column(data=(np.arange(len(obj_lsig)) + 1), name='index'))

    print("# Detection %d low threshold objects" % len(obj_lsig))

    x_mid = (obj_lsig['xmin'] + obj_lsig['xmax']) / 2.0
    y_mid = (obj_lsig['ymin'] + obj_lsig['ymax']) / 2.0

    # Remove the LSB objects whose center fall on the high-threshold mask
    seg_lsig_clean = copy.deepcopy(seg_lsig)
    obj_lsig_clean = copy.deepcopy(obj_lsig)
    img_lsig_clean = copy.deepcopy(img)

    idx_remove = []
    for idx, obj in enumerate(obj_lsig):
        xcen, ycen = int(obj['y']), int(obj['x'])
        xmid, ymid = int(y_mid[idx]), int(x_mid[idx])
        msk_hsig = (msk_hsig_1 | msk_hsig_2)
        if (msk_hsig[xmid, ymid] > 0):
            # Replace the segement with zero
            seg_lsig_clean[seg_lsig == (idx + 1)] = 0
            # Replace the image with noise
            img_lsig_clean[seg_lsig == (idx + 1)] = noise[seg_lsig == (idx + 1)]
            # Remove the object
            idx_remove.append(idx)

    obj_lsig_clean.remove_rows(idx_remove)

    # Remove LSB objects whose segments overlap with the high-threshold mask
    frac_msk = np.asarray([(msk_hsig_1[seg_lsig_clean == idx]).sum() /
                            np.asarray([seg_lsig_clean == idx]).sum()
                           for idx in obj_lsig_clean['index']])

    idx_overlap = []
    for index, idx in enumerate(obj_lsig_clean['index']):
        if frac_msk[index] >= frac_mask:
            # Replace the segement with zero
            seg_lsig_clean[seg_lsig == idx] = 0
            # Replace the image with noise
            img_lsig_clean[seg_lsig == idx] = noise[seg_lsig == idx]
            # Remove the object
            idx_overlap.append(index)

    return seg_lsig_clean, img_lsig_clean
