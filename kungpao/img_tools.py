"""Useful tool for image reduction."""

from astropy.io import fits
from astropy.nddata import Cutout2D

__all__ = ['img_cutout']


def img_cutout(img, img_wcs, ra, dec, size=60.0, pix=0.168,
               prefix='img_cutout'):
    """Generate image cutout with updated WCS information."""
    # imgsize in unit of arcsec
    cutout_size = size / pix

    cen_x, cen_y = img_wcs.wcs_world2pix(ra, dec, 0)
    cen_pos = (int(cen_x), int(cen_y))

    # Generate cutout
    cutout = Cutout2D(img, cen_pos, (cutout_size, cutout_size),
                      wcs=img_wcs)

    # Update the header
    header = cutout.wcs.to_header()

    # Build a HDU
    hdu = fits.PrimaryHDU(header=header)
    hdu.data = cutout.data

    # Save FITS image
    fits_file = prefix + '.fits'
    hdu.writeto(fits_file, overwrite=True)

    return cutout
