#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Visulization tools."""

import copy
import pickle
import numpy as np

from pkg_resources import resource_filename

from astropy.visualization import ZScaleInterval
from astropy.visualization import AsymmetricPercentileInterval
from astropy.visualization import make_lupton_rgb

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import gridspec
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rc('text', usetex=True)

__all__ = ['random_cmap', 'display_single', 'display_all', 'diagnose_image_clean',
           'diagnose_image_mask', 'science_cmap', 'img_rgb_figure',
           'IMG_CMAP', 'SEG_CMAP']


def random_cmap(ncolors=256, background_color='white'):
    """Random color maps.

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
    prng = np.random.mtrand._rand

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


# About the Colormaps
IMG_CMAP = copy.copy(plt.get_cmap('viridis'))
IMG_CMAP.set_bad(color='black')
SEG_CMAP = random_cmap(ncolors=512, background_color=u'white')
SEG_CMAP.set_bad(color='white')
SEG_CMAP.set_under(color='white')


def display_single(img,
                   pixel_scale=0.168,
                   physical_scale=None,
                   xsize=8,
                   ysize=8,
                   ax=None,
                   stretch='arcsinh',
                   scale='zscale',
                   contrast=0.25,
                   zmin=None,
                   zmax=None,
                   no_negative=False,
                   lower_percentile=1.0,
                   upper_percentile=99.0,
                   cmap=IMG_CMAP,
                   scale_bar=True,
                   scale_bar_length=5.0,
                   scale_bar_fontsize=20,
                   scale_bar_y_offset=0.5,
                   scale_bar_color='w',
                   scale_bar_loc='left',
                   color_bar=False,
                   color_bar_loc=1,
                   color_bar_width='75%',
                   color_bar_height='5%',
                   color_bar_fontsize=18,
                   color_bar_color='w',
                   add_text=None,
                   text_fontsize=30,
                   text_color='w'):
    """Display single image.

    Parameters
    ----------
        img: np 2-D array for image

        xsize: int, default = 8
            Width of the image.

        ysize: int, default = 8
            Height of the image.

    """
    if ax is None:
        fig = plt.figure(figsize=(xsize, ysize))
        ax1 = fig.add_subplot(111)
    else:
        ax1 = ax

    # Stretch option
    if stretch.strip() == 'arcsinh':
        img_scale = np.arcsinh(img)
    elif stretch.strip() == 'log':
        if no_negative:
            img[img <= 0.0] = 1.0E-10
        img_scale = np.log(img)
    elif stretch.strip() == 'log10':
        if no_negative:
            img[img <= 0.0] = 1.0E-10
        img_scale = np.log10(img)
    elif stretch.strip() == 'linear':
        img_scale = img
    else:
        raise Exception("# Wrong stretch option.")

    # Scale option
    if zmin is not None and zmax is not None:
        zmin, zmax = zmin, zmax
    elif scale.strip() == 'zscale':
        try:
            zmin, zmax = ZScaleInterval(contrast=contrast).get_limits(img_scale)
        except IndexError:
            # TODO: Deal with problematic image
            zmin, zmax = -1.0, 1.0
    elif scale.strip() == 'percentile':
        try:
            zmin, zmax = AsymmetricPercentileInterval(
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile).get_limits(img_scale)
        except IndexError:
            # TODO: Deal with problematic image
            zmin, zmax = -1.0, 1.0
    else:
        zmin, zmax = np.nanmin(img_scale), np.nanmax(img_scale)

    show = ax1.imshow(img_scale, origin='lower', cmap=cmap, interpolation='none',
                      vmin=zmin, vmax=zmax)

    # Hide ticks and tick labels
    ax1.tick_params(
        labelbottom=False,
        labelleft=False,
        axis=u'both',
        which=u'both',
        length=0)

    # Put scale bar on the image
    (img_size_x, img_size_y) = img.shape
    if physical_scale is not None:
        pixel_scale *= physical_scale
    if scale_bar:
        if scale_bar_loc == 'left':
            scale_bar_x_0 = int(img_size_x * 0.04)
            scale_bar_x_1 = int(img_size_x * 0.04 +
                                (scale_bar_length / pixel_scale))
        else:
            scale_bar_x_0 = int(img_size_x * 0.95 -
                                (scale_bar_length / pixel_scale))
            scale_bar_x_1 = int(img_size_x * 0.95)

        scale_bar_y = int(img_size_y * 0.10)
        scale_bar_text_x = (scale_bar_x_0 + scale_bar_x_1) / 2
        scale_bar_text_y = (scale_bar_y * scale_bar_y_offset)
        if physical_scale is not None:
            scale_bar_text = r'$%d\ \mathrm{kpc}$' % int(scale_bar_length)
        else:
            scale_bar_text = r'$%d^{\prime\prime}$' % int(scale_bar_length)
        scale_bar_text_size = scale_bar_fontsize

        ax1.plot(
            [scale_bar_x_0, scale_bar_x_1], [scale_bar_y, scale_bar_y],
            linewidth=3,
            c=scale_bar_color,
            alpha=1.0)
        ax1.text(
            scale_bar_text_x,
            scale_bar_text_y,
            scale_bar_text,
            fontsize=scale_bar_text_size,
            horizontalalignment='center',
            color=scale_bar_color)
    if add_text is not None:
        text_x_0 = int(img_size_x*0.08)
        text_y_0 = int(img_size_y*0.80)
        ax1.text(text_x_0, text_y_0, r'$\mathrm{'+add_text+'}$', fontsize=text_fontsize, color=text_color)

    # Put a color bar on the image
    if color_bar:
        ax_cbar = inset_axes(ax1,
                             width=color_bar_width,
                             height=color_bar_height,
                             loc=color_bar_loc)
        if ax is None:
            cbar = plt.colorbar(show, ax=ax1, cax=ax_cbar,
                                orientation='horizontal')
        else:
            cbar = plt.colorbar(show, ax=ax, cax=ax_cbar,
                                orientation='horizontal')

        cbar.ax.xaxis.set_tick_params(color=color_bar_color)
        cbar.ax.yaxis.set_tick_params(color=color_bar_color)
        cbar.outline.set_edgecolor(color_bar_color)
        plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'),
                 color=color_bar_color, fontsize=color_bar_fontsize)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'),
                 color=color_bar_color, fontsize=color_bar_fontsize)

    if ax is None:
        return fig
    return ax1


def diagnose_image_clean(img_clean, everything,
                         pixel_scale=0.168,
                         physical_scale=None,
                         scale_bar_length=2.0):
    """QA plot for image clean process."""
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


def diagnose_image_mask(img_mask, everything,
                        pixel_scale=0.168,
                        physical_scale=None,
                        scale_bar_length=2.0):
    """QA plot for image clean process."""
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
    if everything['seg_2'] is not None:
        ax5 = display_single(
            everything['seg_2'],
            ax=ax5,
            contrast=0.10,
            scale_bar_length=scale_bar_length,
            pixel_scale=pixel_scale,
            physical_scale=physical_scale,
            scale_bar_color='k',
            scale='none',
            cmap=SEG_CMAP,
            stretch='linear')

    ax6 = plt.subplot(3, 3, 6)
    if everything['seg_3'] is not None:
        ax6 = display_single(
            everything['seg_3'],
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
    if everything['bkg_3'] is not None:
        ax7 = display_single(
            everything['bkg_3'].back(),
            ax=ax7,
            contrast=0.20,
            scale_bar_length=scale_bar_length,
            pixel_scale=pixel_scale,
            physical_scale=physical_scale,
            color_bar=True)

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

    img_clean = everything['img']
    img_clean[img_mask > 0] = np.nan
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


def science_cmap(cmap_name='vik', visual=False, list_maps=False):
    """ Perceptually uniform color maps

    Parameters
    ----------
    cmap_name : string, optional
        Name of the colour map.
        Default: `vik`
    visual : boolen, optional
        Whether to visualize the colour map or not.
        Default: False
    list_maps: boolen, optional
        If `True`, just list the names of available maps
        Default: False

    Returns
    -------
    cmap : `matplotlib.colors.Colormap`
        The matplotlib colormap with random colors.

    Notes
    -----
        Thanks Fabia Cramer for making these available.
            http://www.fabiocrameri.ch/colourmaps.php
    """
    cmap_list = ['bamako', 'batlow', 'berlin', 'bilbao',
                 'davos', 'imola', 'lajolla', 'lapaz',
                 'oslo', 'roma', 'vik']
    if list_maps:
        print(cmap_list)
        return cmap_list

    if cmap_name not in cmap_list:
        raise Exception("Wrong colour map name!")
    else:
        cmap_pkl = resource_filename('kungpao',
                                     '/cmap_data/%s.pkl' % cmap_name)

        with open(cmap_pkl, 'rb') as cmap_file:
            cmap = LinearSegmentedColormap.from_list(cmap_name,
                                                     pickle.load(cmap_file))

    if visual:
        plt.imshow(np.linspace(0, 100, 256)[None, :], aspect='auto',
                   cmap=cmap, interpolation='none')
        plt.show()

    return cmap


def img_rgb_figure(image_r, image_g, image_b, stretch=0.5, Q=20,
                   show=True, save=False, prefix='rgb', shrink=40,
                   scale=0.168, physical=False, scalebar=None,
                   minimum=None,
                   scale_bar_y_offset=0.4, scale_bar_fontsize=15):
    """Making RGB picture using the Lupton algorithm."""

    assert image_r.shape == image_g.shape
    assert image_r.shape == image_b.shape

    img_h, img_w = image_r.shape

    fig_h, fig_w = int(img_h / shrink), int(img_w / shrink)

    if minimum is None:
        minimum = np.asarray(
            [np.percentile(image_r, 1.0), np.percentile(image_g, 1.0),
             np.percentile(image_b, 1.0)])

    image = make_lupton_rgb(image_r, image_g, image_b, minimum=minimum,
                            stretch=stretch, Q=Q)

    if show:
        # --- Display --- #
        fig = plt.figure(figsize=(fig_w, fig_h))
        plt.subplots_adjust(left=0.002, bottom=0.002, right=0.998, top=0.998)
        ax1 = fig.add_subplot(111)

        # Adjust the frame
        ax1.spines['top'].set_linewidth(3.0)
        ax1.spines['top'].set_color('xkcd:silver')
        ax1.spines['bottom'].set_linewidth(3.0)
        ax1.spines['bottom'].set_color('xkcd:silver')
        ax1.spines['left'].set_linewidth(3.0)
        ax1.spines['left'].set_color('xkcd:silver')
        ax1.spines['right'].set_linewidth(3.0)
        ax1.spines['right'].set_color('xkcd:silver')

        # Adjust the ticks
        ## Remove tick labels
        ax1.set_yticklabels([])
        ax1.set_xticklabels([])

        ## Set number of ticks
        ax1.xaxis.set_major_locator(plt.MaxNLocator(8))
        ax1.yaxis.set_major_locator(plt.MaxNLocator(8))
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator())

        ## Adjust the tick legth, locations, width, and colors
        ax1.tick_params(axis="both", direction="in", length=8,
                        width=2.5, color='xkcd:grey',
                        top=True, right=True, which='major')
        ax1.tick_params(axis="both", direction="in", length=5,
                        width=2.5, color='xkcd:grey',
                        bottom=True, left=True,
                        top=True, right=True, which='minor')
        ax1.imshow(image, interpolation='none', origin='lower')

        # Scale bar
        # TODO: the dealing of scale bar parameters is not perfect
        if scalebar is not None:
            # Pixel length
            pixel_length = scalebar / scale

            scale_bar_x_0 = int(img_w * 0.05)
            scale_bar_x_1 = int(img_w * 0.05 + pixel_length)

            scale_bar_y = int(img_h * 0.12)
            scale_bar_text_x = (scale_bar_x_0 + scale_bar_x_1) / 2
            scale_bar_text_y = (scale_bar_y * scale_bar_y_offset)

            if physical:
                scale_bar_text = r'$%d\ \mathrm{kpc}$' % int(scalebar)
            else:
                scale_bar_text = r'$%d^{\prime\prime}$' % int(scalebar)

            ax1.plot(
                [scale_bar_x_0, scale_bar_x_1], [scale_bar_y, scale_bar_y],
                linewidth=2.0, c='w', alpha=1.0)
            ax1.text(scale_bar_text_x, scale_bar_text_y,
                     scale_bar_text, fontsize=scale_bar_fontsize,
                     horizontalalignment='center', color='w')

        if save:
            fig.savefig(prefix + '.png', dpi=90)
            plt.close(fig)

        return image, fig

    return image

def display_all(img_list, n_column=3, img_size=3., hdu_index=None, label_list=None,
                label_x=0.1, label_y=0.9, fontsize=20, **kwargs):
    """Display a list of images."""
    # Number of image to show
    n_img = len(img_list)

    if n_img <= n_column:
        n_col = n_img
        n_row = 1
    else:
        n_col = n_column
        n_row = int(np.ceil(n_img / n_column))

    fig = plt.figure(figsize=(img_size * n_col, img_size * n_row))
    fig.subplots_adjust(left=0., right=1., bottom=0., top=1., wspace=0., hspace=0.)

    gs = gridspec.GridSpec(n_row, n_col)
    gs.update(wspace=0.0, hspace=0.00)

    for ii in range(n_img):
        if hdu_index is None:
            img_show = img_list[ii]
        else:
            img_show = img_list[ii][hdu_index].data

        ax = plt.subplot(gs[ii])
        ax = display_single(img_show, ax=ax, **kwargs)

        if label_list is not None:
            if len(label_list) != n_img:
                print("# Wrong number for labels!")
            else:
                ax.text(label_x, label_y, label_list[ii], fontsize=fontsize,
                        transform=ax.transAxes, color='w')

    return fig
