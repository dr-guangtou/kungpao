"""Helper functions for isophote analysis."""

import os
import platform

import kungpao

__all__ = ['fits_to_pl', 'iraf_commands']


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
    if not os.path.isfile(ximage) and not os.path.islink(ximage):
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


def iraf_commands():
    """Locate the exectuble files for IRAF functions.

    Returns
    -------
    iraf: dict
        Dictionary for the IRAF functions.

    """
    if platform.system() == 'Darwin':
        IRAF_DIR = os.path.join(
            os.path.dirname(kungpao.__file__), 'iraf', 'macosx')
    elif platform.system() == 'Linux':
        IRAF_DIR = os.path.join(
            os.path.dirname(kungpao.__file__), 'iraf', 'linux')
    else:
        raise ValueError(
            'Wrong platform: only support MacOSX or Linux now')

    return {
        'ellipse': os.path.join(IRAF_DIR, 'x_isophote.e'),
        'ximages': os.path.join(IRAF_DIR, 'x_images.e'),
        'ttools': os.path.join(IRAF_DIR, 'x_ttools.e'),
    }
