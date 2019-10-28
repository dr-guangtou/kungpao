"""Plotting the 1-D profile from isophotal analysis."""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rcParams
from matplotlib.patches import Ellipse
from matplotlib.ticker import NullFormatter

from kungpao.display import display_single


plt.rc('text', usetex=True)
rcParams.update({'axes.linewidth': 1.5})
rcParams.update({'xtick.direction': 'in'})
rcParams.update({'ytick.direction': 'in'})
rcParams.update({'xtick.minor.visible': 'True'})
rcParams.update({'ytick.minor.visible': 'True'})
rcParams.update({'xtick.major.pad': '7.0'})
rcParams.update({'xtick.major.size': '8.0'})
rcParams.update({'xtick.major.width': '1.5'})
rcParams.update({'xtick.minor.pad': '7.0'})
rcParams.update({'xtick.minor.size': '4.0'})
rcParams.update({'xtick.minor.width': '1.5'})
rcParams.update({'ytick.major.pad': '7.0'})
rcParams.update({'ytick.major.size': '8.0'})
rcParams.update({'ytick.major.width': '1.5'})
rcParams.update({'ytick.minor.pad': '7.0'})
rcParams.update({'ytick.minor.size': '4.0'})
rcParams.update({'ytick.minor.width': '1.5'})
rcParams.update({'axes.titlepad': '10.0'})
rcParams.update({'font.size': 25})


__all__ = ['display_isophote', 'display_center_fourier', 'display_intensity_shape']


def display_isophote(img, ell, iso_color='orangered', zoom=None, **display_kwargs):
    """Visualize the isophotes."""
    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.00, hspace=0.00)
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace=0.0, hspace=0.00)

    # Whole central galaxy: Step 2
    ax1 = fig.add_subplot(gs[0])
    ax1.yaxis.set_major_formatter(NullFormatter())
    ax1.xaxis.set_major_formatter(NullFormatter())

    if zoom is not None:
        x_img, y_img = img.shape[0], img.shape[1] 
        x_size, y_size = int(x_img / zoom), int(y_img / zoom)
        x_off, y_off = int((x_img - x_size) / 2), int((y_img - y_size) / 2)
        img = img[x_off: x_off + x_size, y_off: y_off + y_size]
    else:
        x_off, y_off = 0, 0

    ax1 = display_single(img, ax=ax1, scale_bar=False, **display_kwargs)

    for k, iso in enumerate(ell):
        if k % 2 == 0:
            e = Ellipse(xy=(iso['x0'] - x_off, iso['y0'] - y_off),
                        height=iso['sma'] * 2.0,
                        width=iso['sma'] * 2.0 * (1.0 - iso['ell']),
                        angle=iso['pa'])
            e.set_facecolor('none')
            e.set_edgecolor(iso_color)
            e.set_alpha(0.9)
            e.set_linewidth(2.0)
            ax1.add_artist(e)
    ax1.set_aspect('equal')


def display_center_fourier(ell, x_max=4.0):
    """Display the 1-D profiles."""
    fig = plt.figure(figsize=(10, 10))
    fig.subplots_adjust(left=0.0, right=1.0,
                        bottom=0.0, top=1.0,
                        wspace=0.00, hspace=0.00)

    ax1 = fig.add_axes([0.08, 0.07, 0.85, 0.23])
    ax2 = fig.add_axes([0.08, 0.30, 0.85, 0.23])
    ax3 = fig.add_axes([0.08, 0.53, 0.85, 0.23])
    ax4 = fig.add_axes([0.08, 0.76, 0.85, 0.23])

    # A3 / B3 profile
    ax1.grid(linestyle='--', alpha=0.4, linewidth=2)

    ax1.errorbar((ell['sma'] ** 0.25),
                 ell['a3'],
                 yerr=ell['a3_err'],
                 color='k', alpha=0.7, fmt='o',
                 capsize=4, capthick=1, elinewidth=1)

    ax1.errorbar((ell['sma'] ** 0.25),
                 ell['b3'],
                 yerr=ell['b3_err'],
                 color='r', alpha=0.7, fmt='o',
                 capsize=4, capthick=1, elinewidth=1)

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(25)

    ax1.set_xlim(1.09, x_max)
    ax1.set_ylim(-0.9, 0.9)

    ax1.set_xlabel(r'$R/\mathrm{pixel}^{1/4}$', fontsize=30)
    ax1.set_ylabel(r'${A_3\ {\rm or}\ B_3}$', fontsize=30)

    # A4/ B4 profile
    ax2.grid(linestyle='--', alpha=0.4, linewidth=2)

    ax2.errorbar((ell['sma'] ** 0.25),
                 ell['a4'],
                 yerr=ell['a4_err'],
                 color='k', alpha=0.7, fmt='o',
                 capsize=4, capthick=1, elinewidth=1)

    ax2.errorbar((ell['sma'] ** 0.25),
                 ell['b4'],
                 yerr=ell['b4_err'],
                 color='r', alpha=0.7, fmt='o',
                 capsize=4, capthick=1, elinewidth=1)

    ax2.xaxis.set_major_formatter(NullFormatter())
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(25)

    ax2.set_xlim(1.09, x_max)
    ax2.set_ylim(-0.9, 0.9)
    ax2.set_ylabel(r'${A_4\ {\rm or}\ B_4}$', fontsize=30)

    # X central coordinate profile
    ax3.grid(linestyle='--', alpha=0.4, linewidth=2)

    ax3.errorbar((ell['sma'] ** 0.25),
                 ell['x0'],
                 yerr=ell['x0_err'],
                 color='k', alpha=0.7, fmt='o',
                 capsize=4, capthick=1, elinewidth=1)

    ax3.xaxis.set_major_formatter(NullFormatter())
    for tick in ax3.yaxis.get_major_ticks():
        tick.label.set_fontsize(25)

    ax3.set_xlim(1.09, x_max)
    ax3.set_ylabel(r'$\mathrm{X_0}$', fontsize=32)

    # Position Angle profile
    ax4.grid(linestyle='--', alpha=0.4, linewidth=2)

    ax4.errorbar((ell['sma'] ** 0.25),
                 ell['y0'], yerr=ell['y0_err'],
                 color='k', alpha=0.7, fmt='o',
                 capsize=4, capthick=2, elinewidth=2)

    ax4.xaxis.set_major_formatter(NullFormatter())
    for tick in ax4.yaxis.get_major_ticks():
        tick.label.set_fontsize(25)

    ax4.set_xlim(1.09, x_max)
    ax4.set_ylabel(r'$\mathrm{Y_0}$', fontsize=25)

    return fig


def display_intensity_shape(ell, x_max=4.0):
    """Display the 1-D profiles."""
    fig = plt.figure(figsize=(10, 10))
    fig.subplots_adjust(left=0.0, right=1.0,
                        bottom=0.0, top=1.0,
                        wspace=0.00, hspace=0.00)

    ax1 = fig.add_axes([0.08, 0.07, 0.85, 0.48])
    ax2 = fig.add_axes([0.08, 0.55, 0.85, 0.20])
    ax3 = fig.add_axes([0.08, 0.75, 0.85, 0.20])

    # 1-D profile
    ax1.grid(linestyle='--', alpha=0.4, linewidth=2)

    yerr = np.log10(ell['intens'] + ell['int_err']) - np.log10(ell['intens'])
    ax1.errorbar((ell['sma'] ** 0.25),
                 np.log10(ell['intens']),
                 yerr=yerr,
                 color='k', alpha=0.7, fmt='o',
                 capsize=4, capthick=1, elinewidth=1)

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(25)

    ax1.set_xlim(1.09, x_max)

    ax1.set_xlabel(r'$R/\mathrm{pixel}^{1/4}$', fontsize=30)
    ax1.set_ylabel(r'$\log\ ({\rm Intensity})$', fontsize=30)

    # Ellipticity profile
    ax2.grid(linestyle='--', alpha=0.4, linewidth=2)

    ax2.errorbar((ell['sma'] ** 0.25),
                 ell['ell'],
                 yerr=ell['ell_err'],
                 color='k', alpha=0.7, fmt='o', capsize=4, capthick=2, elinewidth=2)

    ax2.xaxis.set_major_formatter(NullFormatter())
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(25)

    ax2.set_xlim(1.09, x_max)
    ax2.set_ylabel(r'$e$', fontsize=35)

    # Position Angle profile
    ax3.grid(linestyle='--', alpha=0.4, linewidth=2)

    ax3.errorbar((ell['sma'] ** 0.25),
                 ell['pa'], yerr=ell['pa_err'],
                 color='k', alpha=0.7, fmt='o', capsize=4, capthick=2, elinewidth=2)

    ax3.xaxis.set_major_formatter(NullFormatter())
    for tick in ax3.yaxis.get_major_ticks():
        tick.label.set_fontsize(25)

    ax3.set_xlim(1.09, x_max)
    ax3.set_ylabel(r'$\mathrm{PA\ [deg]}$', fontsize=25)

    return fig
