"""
Functions for displaying images and photometric results.
"""

from __future__ import print_function, \
    division, \
    absolute_import

from astropy.visualization import ZScaleInterval, \
    AsymmetricPercentileInterval

import numpy as np

import matplotlib.pyplot as plt
plt.rc('text', usetex=True)

# About the Colormaps
IMG_CMAP = plt.get_cmap('viridis')
IMG_CMAP.set_bad(color='black')


def prettify(fig, ax, label=None):
    """
    Prettify a figure.

    Thanks to: python-fsps/feature_demo.py by Ben Johnson

    """
    ax.set_xlim(0.9e3, 1e6)
    ax.set_xscale('log')
    ax.set_ylim(0.01, 2)

    ax.set_xlabel('rest-frame $\lambda$ ($\AA$)', fontsize=20)
    ax.set_ylabel('$\lambda \, f_\lambda$', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    if label is not None:
        ax.text(0.63, 0.85, label, transform=ax.transAxes, fontsize=16)

    return fig, ax


def display_single(img,
                   pixel_scale=0.168,
                   xsize=8,
                   ysize=8,
                   ax=None,
                   stretch='arcsinh',
                   scale='zscale',
                   contrast=0.25,
                   no_negative=False,
                   lower_percentile=1.0,
                   upper_percentile=99.0,
                   cmap=IMG_CMAP,
                   scale_bar=True,
                   scale_bar_length=20.0,
                   scale_bar_fontsize=20,
                   scale_bar_y_offset=0.5):
    """
    Display a single image.

    :param img: np 2-D array for image

    :param xsize: int, default = 8
        Width of the image.
    :param ysize: int, default = 8
        Height of the image.
    """
    if ax is None:
        fig = plt.figure(figsize=(xsize, ysize))
        ax1 = fig.add_subplot(111)
    else:
        ax1 = ax

    # Stretch option
    if stretch.strip() is 'arcsinh':
        img_scale = np.arcsinh(img)
    elif stretch.strip() is 'log':
        if no_negative:
            img[img <= 0.0] = 1.0E-10
        img_scale = np.log(img)
    elif stretch.strip() is 'log10':
        if no_negative:
            img[img <= 0.0] = 1.0E-10
        img_scale = np.log10(img)
    elif stretch.strip() is 'linear':
        img_scale = img
    else:
        raise Exception("# Wrong stretch option.")

    # Scale option
    if scale.strip() is 'zscale':
        zmin, zmax = ZScaleInterval(contrast=contrast).get_limits(img_scale)
    elif scale.strip() is 'percentile':
        zmin, zmax = AsymmetricPercentileInterval(
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile).get_limits(img_scale)

    ax1.imshow(img_scale, origin='lower', cmap=cmap, vmin=zmin, vmax=zmax)

    # Hide ticks and tick labels
    ax1.tick_params(
        labelbottom='off',
        labelleft='off',
        axis=u'both',
        which=u'both',
        length=0)

    # Put scale bar on the image
    (img_size_x, img_size_y) = img.shape
    if scale_bar:
        scale_bar_x_0 = int(img_size_x * 0.95 -
                            (scale_bar_length / pixel_scale))
        scale_bar_x_1 = int(img_size_x * 0.95)
        scale_bar_y = int(img_size_y * 0.10)
        scale_bar_text_x = (scale_bar_x_0 + scale_bar_x_1) / 2
        scale_bar_text_y = (scale_bar_y * scale_bar_y_offset)
        scale_bar_text = r'$%d^{\prime\prime}$' % int(scale_bar_length)
        scale_bar_text_size = scale_bar_fontsize

        ax1.plot(
            [scale_bar_x_0, scale_bar_x_1], [scale_bar_y, scale_bar_y],
            linewidth=3,
            c='w',
            alpha=1.0)
        ax1.text(
            scale_bar_text_x,
            scale_bar_text_y,
            scale_bar_text,
            fontsize=scale_bar_text_size,
            horizontalalignment='center',
            color='w')

    if ax is None:
        return fig
    else:
        return ax1
