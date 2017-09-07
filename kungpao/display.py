"""
Functions for displaying images and photometric results.
"""

from __future__ import division, print_function

import os
import numpy as np


def prettify(fig, ax, label=None):
    """
    Prettify a figure.

    Thanks to: python-fsps/feature_demo.py by Ben Johnson

    """
    ax.set_xlim(0.9e3, 1e6)
    ax.set_xscale('log')
    ax.set_ylim(0.01, 2)
    #ax.set_yscale('log')
    ax.set_xlabel('rest-frame $\lambda$ ($\AA$)', fontsize=20)
    ax.set_ylabel('$\lambda \, f_\lambda$', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    if label is not None:
        ax.text(0.63, 0.85, label, transform=ax.transAxes, fontsize=16)

    return fig, ax
