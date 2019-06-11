#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Galactic extinction."""

from __future__ import print_function, absolute_import, division

import subprocess

from astropy.coordinates import SkyCoord


__all__ = ['radec_extinction']


def _find_sfd_maps(north='SFD_dust_4096_ngp.fits',
                   south='SFD_dust_4096_sgp.fits', index=0):
    """Use locate command to find the SFD files."""
    north_map = subprocess.check_output(
        "locate %s" % north, shell=True).splitlines()
    south_map = subprocess.check_output(
        "locate %s" % south, shell=True).splitlines()

    if north_map:
        north_file = north_map[index].decode('utf-8')
    else:
        north_file = None

    if south_map:
        south_file = south_map[index].decode('utf-8')
    else:
        south_file = None

    return north_file, south_file


def radec_extinction(ra, dec, a_lambda=1.0):
    """Estimate the Galactic extinction for HSC filters.

    -----------
    Parameters:
        (ra, dec): (float, float)
            The input coordinates can be arrays

        a_lambda : optional, default=1.0
            Convert the e_bv value into extinction in magnitude unit.

    """
    coords = SkyCoord(ra, dec, frame='icrs', unit='deg')

    try:
        # mwdust by Jo Bovy
        import mwdust
        sfd = mwdust.SFD(sf10=True)
        ebv = sfd(coords.galactic.l.deg, coords.galactic.b.deg, 0)
    except ImportError:
        try:
            # Try querying the IRSA dust map instead
            from astroquery.irsa_dust import IrsaDust
            extinction_tab = IrsaDust.get_extinction_table(coords)
            ebv = (extinction_tab['A_SFD'] / extinction_tab['A_over_E_B_V_SFD'])[1]
        except ImportError:
            raise Exception("# Need mwdust by Jo Bovy or Astroquery")

    return a_lambda * ebv
