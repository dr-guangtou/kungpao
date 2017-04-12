"""
Collection of visualization functions for imfit. 
"""
from __future__ import division, print_function
                     

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from .sersic import Sersic
from ..utils import pixscale
from toolbox.utils.plotting import ticks_off
from toolbox.image import zscale

__all__ = ['imfit_results']

def img_mod_res(img_fn, mod_params, mask_fn=None, cmap=plt.cm.gray_r, 
                save_fn=None, show=True, band='i', subplots=None, 
                titles=True, **kwargs):
    """
    Show imfit results: image, model, and residual.
    """

    img = fits.getdata(img_fn)

    if subplots is None:
        fig, axes = plt.subplots(1, 3, **kwargs)
        fig.subplots_adjust(wspace=0.08)
    else:
        fig, axes = subplots

    s = Sersic(mod_params)
    model = s.array(img.shape)
    res = img - model

    vmin, vmax = zscale(img)

    param_labels = {}

    if titles:
        titles = ['Original Image', 'Model', 'Residual']
    else:
        titles = ['']*3


    for i, data in enumerate([img, model, res]):
        ticks_off(axes[i])
        axes[i].imshow(data, vmin=vmin, vmax=vmax, origin='lower', 
                       cmap=cmap, aspect='equal', rasterized=True)
        axes[i].set_title(titles[i], fontsize=20, y=1.01)

    if mask_fn is not None:
        mask = fits.getdata(mask_fn)
        mask = mask.astype(float)
        mask[mask==0.0] = np.nan
        axes[0].imshow(mask, origin='lower', alpha=0.4, 
                       vmin=0, vmax=1, cmap='rainbow_r')

    x = 0.05
    y = 0.93
    dy = 0.08
    dx = 0.63
    fs = 17
    m_tot = r'$m_'+band+'='+str(round(s.m_tot, 1))+'$'
    r_e = r'$r_\mathrm{eff}='+str(round(s.r_e*pixscale,1))+'^{\prime\prime}$'
    mu_0 = r'$\mu_0='+str(round(s.mu_0,1))+'$'
    mu_e = r'$\mu_e='+str(round(s.mu_e,1))+'$'
    n = r'$n = '+str(round(s.n,2))+'$'
    chisq = r'$\chi^2_\mathrm{dof} = '+\
            str(round(mod_params['reduced_chisq'],2))+'$' 
    c = 'b'

    axes[1].text(x, y, m_tot, transform=axes[1].transAxes, 
                 fontsize=fs, color=c)
    axes[1].text(x, y-dy, mu_0, transform=axes[1].transAxes, 
                 fontsize=fs, color=c)
    axes[1].text(x, y-2*dy, mu_e, transform=axes[1].transAxes, 
                 fontsize=fs, color=c)
    axes[1].text(x+dx, y, n, transform=axes[1].transAxes, 
                 fontsize=fs, color=c)
    axes[1].text(x+dx, y-dy, r_e, transform=axes[1].transAxes, 
                 fontsize=fs, color=c)
    axes[1].text(x+dx, y-2*dy, chisq, transform=axes[1].transAxes, 
                 fontsize=fs, color=c)
    axes[1].text(0.9, 0.05, band, color='r', transform=axes[1].transAxes,
                 fontsize=25)

    if show:
        try: import RaiseWindow
        except: pass
        plt.show()

    if save_fn is not None:
        fig.savefig(save_fn, bbox_inches='tight')
