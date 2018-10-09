"""Functions related to GALFIT modeling."""

from __future__ import division, absolute_import, print_function

import os

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import NullFormatter
from matplotlib.patches import Ellipse

from astropy.io import fits

from palettable.colorbrewer.qualitative import Set1_9

from kungpao.display import display_single

color_comps = Set1_9.mpl_colors
plt.rc('text', usetex=True)

__all__ = ['galfit_naive_aic', 'show_galfit_model']


def galfit_naive_aic(galfit_output):
    """Estimate the AIC, BIC, and HQ for a GALFIT model.

    AIC=-2 ln(L) + 2 k : akaike information criterion
    BIC=-2 ln(L) + ln(n)*k : bayesian information criterion
    HQ=-2 ln(L) + ln(ln(n))*k  hannan-quinn criterion
    """
    chisq = np.float(galfit_output.chisq)
    ndof = np.float(galfit_output.ndof)
    nfree = np.float(galfit_output.nfree)

    # AIC
    aic = chisq + 2.0 * nfree + (2.0 * nfree * (nfree + 1.0)) / (ndof - nfree - 1.0)

    # BIC
    bic = chisq + np.log(ndof) * nfree

    # Hannan-Quinn cirterion
    hq = chisq + np.log(np.log(ndof)) * nfree

    return aic, bic, hq


def show_galfit_model(
        galfit_fits, galfit_out, root=None, verbose=True, vertical=False,
        zoom=True, zoom_limit=6.0, show_title=True, show_chi2=True, zoom_size=None,
        overplot_components=True, mask_residual=True, mask_file=None, show_bic=True,
        xsize=15.0, ysize=5.0, cmap=None, axes=None, show_contour=True,
        title=None, pixel=0.168, **kwargs):
    """Three columns plot of the Galfit models. """
    # Read in the results fits file
    galfit_results = fits.open(galfit_fits)
    fig_name = galfit_fits.replace('.fits', '.png')

    # Colormap
    cmap = plt.get_cmap('viridis') if cmap is None else cmap

    if verbose:
        print(" ## %s ---> %s " % (galfit_fits, fig_name))

    # Set up the figure
    if axes is None:
        fig = plt.figure(figsize=(xsize, ysize))
        if not vertical:
            grid = gridspec.GridSpec(1, 3)
        else:
            grid = gridspec.GridSpec(3, 1)
            xsize, ysize = ysize, xsize
        grid.update(wspace=0.0, hspace=0.0, left=0, right=1, top=1, bottom=0)
        ax1 = plt.subplot(grid[0])
        ax2 = plt.subplot(grid[1])
        ax3 = plt.subplot(grid[2])
    else:
        # Can provide a tuple of three pre-designed axes
        ax1, ax2, ax3 = axes

    # Geometry of each component
    n_comp = galfit_out.num_components
    x_comps, y_comps, re_comps, ba_comps, pa_comps = [], [], [], [], []

    for i in range(1, n_comp+1):
        string_comp = 'component_' + str(i)
        info_comp = getattr(galfit_out, string_comp)
        if info_comp.component_type == 'sersic':
            x_comps.append(info_comp.xc)
            y_comps.append(info_comp.yc)
            re_comps.append(info_comp.re)
            ba_comps.append(info_comp.ar)
            pa_comps.append(info_comp.pa)

    # Image size and scale
    img_ori = galfit_results[1].data
    img_mod = galfit_results[2].data
    img_res = galfit_results[3].data
    img_xsize, img_ysize = img_ori.shape
    img_xcen, img_ycen = img_xsize / 2.0, img_ysize / 2.0

    # Show mask on the residual map
    if mask_residual:
        # Allow user to provide an external mask
        if mask_file is None:
            mask_file = os.path.join(root, galfit_out.input_mask)
        if os.path.isfile(mask_file) or os.path.islink(mask_file):
            img_msk = fits.open(mask_file)[0].data
            img_msk = img_msk[np.int(galfit_out.box_x0)-1: np.int(galfit_out.box_x1),
                              np.int(galfit_out.box_y0)-1: np.int(galfit_out.box_y1)]
            img_res[img_msk > 0] = np.nan
        else:
            print("XXX Can not find the mask file : %s" % mask_file)
            img_msk = None
            img_res = img_res

    r_max = np.max(np.asarray(re_comps)) * zoom_limit
    if zoom_size is not None:
        r_zoom = zoom_size / 2.0
        x0, x1 = int(img_xcen - r_zoom), int(img_xcen + r_zoom)
        y0, y1 = int(img_ycen - r_zoom), int(img_ycen + r_zoom)
    elif img_xcen >= r_max and img_ycen >= r_max and zoom:
        x0, x1 = int(img_xcen - r_max), int(img_xcen + r_max)
        y0, y1 = int(img_ycen - r_max), int(img_ycen + r_max)
        print(" ## Image has been truncated to highlight the galaxy !")
    else:
        x0, x1 = 0, img_xsize - 1
        y0, y1 = 0, img_ysize - 1

    img_ori = img_ori[x0: x1, y0: y1]
    img_mod = img_mod[x0: x1, y0: y1]
    img_res = img_res[x0: x1, y0: y1]
    x_padding, y_padding = x0, y0

    x_comps = np.asarray(x_comps) - np.float(galfit_out.box_x0)
    y_comps = np.asarray(y_comps) - np.float(galfit_out.box_y0)
    re_comps = np.asarray(re_comps)
    ba_comps = np.asarray(ba_comps)
    pa_comps = np.asarray(pa_comps)
    x_comps -= x_padding
    y_comps -= y_padding

    # Show the original image
    ax1.xaxis.set_major_formatter(NullFormatter())
    ax1.yaxis.set_major_formatter(NullFormatter())
    ax1 = display_single(img_ori, ax=ax1, cmap=cmap, pixel_scale=pixel, **kwargs)

    # Overplot the contour of the component
    if overplot_components:
        try:
            for ii, r0 in enumerate(re_comps):
                x0, y0 = x_comps[ii], y_comps[ii]
                q0, pa0 = ba_comps[ii], pa_comps[ii]
                ellip_comp = Ellipse(
                    xy=(x0, y0), width=(r0 * q0 * 2.0), height=(r0 * 2.0), angle=pa0)
                ax1.add_artist(ellip_comp)
                ellip_comp.set_clip_box(ax1.bbox)
                ellip_comp.set_alpha(1.0)
                ellip_comp.set_edgecolor(color_comps[ii])
                ellip_comp.set_facecolor('none')
                ellip_comp.set_linewidth(2.5)
        except Exception:
            print("XXX Can not highlight the components")

    # Show a tile
    if show_title and title is not None:
        str_title = ax1.text(
            0.50, 0.90, r'$\mathrm{%s}$' % title, fontsize=25,
            transform=ax1.transAxes, horizontalalignment='center')
        str_title.set_bbox(dict(facecolor='white', alpha=0.6, edgecolor='white'))

    # Show the model
    ax2.xaxis.set_major_formatter(NullFormatter())
    ax2.yaxis.set_major_formatter(NullFormatter())
    ax2 = display_single(img_mod, ax=ax2, cmap=cmap, pixel_scale=pixel, **kwargs)

    # Show contours of the model
    if show_contour:
        try:
            tam = np.size(img_mod, axis=0)
            contour_x = np.arange(tam)
            contour_y = np.arange(tam)
            ax2.contour(contour_x, contour_y, np.arcsinh(img_mod), colors='c',
                        linewidths=1.5)
        except Exception:
            print("XXX Can not generate the Contour !")

    # Show the reduced chisq
    if show_chi2:
        ax2.text(0.06, 0.92, r'${\chi}^2/N_{DoF} : %s$' % galfit_out.reduced_chisq,
                 fontsize=14, transform=ax2.transAxes)
    if show_bic:
        aic, bic, hq = galfit_naive_aic(galfit_out)
        ax2.text(0.06, 0.82, r'$\mathrm{BIC} : %9.3f$' % bic, fontsize=14, 
                 transform=ax2.transAxes)
        # ax2.text(0.06, 0.87, 'AIC : %9.3f' % aic, fontsize=14, transform=ax2.transAxes)
        # ax2.text(0.06, 0.77, 'HQ : %9.3f' % hq, fontsize=14, transform=ax2.transAxes)

    # Show the residual image
    ax3.xaxis.set_major_formatter(NullFormatter())
    ax3.yaxis.set_major_formatter(NullFormatter())
    ax3 = display_single(img_res, ax=ax3, cmap=cmap, pixel_scale=pixel, **kwargs)

    # Overplot the contour of the component
    if overplot_components:
        try:
            for ii, r0 in enumerate(re_comps):
                x0, y0 = x_comps[ii], y_comps[ii]
                q0, pa0 = ba_comps[ii], pa_comps[ii]
                ellip_comp = Ellipse(
                    xy=(x0, y0), width=(r0 * q0 * 2.0), height=(r0 * 2.0), angle=pa0)
                ax1.add_artist(ellip_comp)
                ellip_comp.set_clip_box(ax3.bbox)
                ellip_comp.set_alpha(1.0)
                ellip_comp.set_edgecolor(color_comps[ii])
                ellip_comp.set_facecolor('none')
                ellip_comp.set_linewidth(2.5)
        except Exception:
            print("XXX Can not highlight the components")

    # Save Figure
    if axes is None:
        fig.savefig(fig_name, dpi=80)
        return fig
    else:
        return ax1, ax2, ax3
