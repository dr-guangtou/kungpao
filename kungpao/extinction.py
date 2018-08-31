"""Galactic extinction."""

from __future__ import print_function, absolute_import, division

from astropy.coordinates import SkyCoord


__all__ = ['radec_extinction']


def radec_extinction(ra, dec, a_lambda=1.0):
    """Estimate the Galactic extinction for HSC filters.

    -----------
    Parameters:
        (ra, dec): (float, float)
            The input coordinates can be arrays

        a_lambda : optional, default=1.0
            Convert the e_bv value into extinction in magnitude unit.

    """
    try:
        import mwdust
        sfd = mwdust.SFD(sf10=True)
        coords = SkyCoord(ra, dec, frame='icrs', unit='deg')
        ebv = sfd(coords.galactic.l.deg, coords.galactic.b.deg, 0)
    except ImportError:
        raise Exception("# Both mwdust and sncosmo are not available")

    return a_lambda * ebv
