"""Helper functions for isophote analysis."""

import os

__all__ = ['fits_to_pl']


def fits_to_pl(ximage, fits, output=None, verbose=False):
    """Convert FITS image into the IRAF .pl format.

    Parameters
    ----------
    ximage: string
        Location of the x_images.e executable file.
    fits: string
        Input FITS file name.
    output: string, optional
        Output .pl file name. Default: None.
        If None, the file name will be "input.fits.pl".
    verbose: bool, optional
        Blah, Blah.  Default: False.

    """
    if not os.path.isfile(ximage) or not os.path.islink(ximage):
        raise FileNotFoundError("Can not find x_images.e: {}".format(ximage))

    if not os.path.isfile(fits):
        raise FileNotFoundError("Can not find input FITS image: {}".format(fits))

    if output is None:
        output = fits.replace('.fits', '.fits.pl')

    if os.path.isfile(output):
        if verbose:
            print("# Output file exists! Will remove {}".format(output))
        os.remove(output)

    imcopy = "{} imcopy input={} output={} verbose=no".format(
        ximage, fits.strip(), output.strip()
    )

    os.system(imcopy)

    return
