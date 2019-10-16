"""Helper functions for isophote analysis."""

import os
import platform

import numpy as np

from matplotlib.patches import Ellipse

import kungpao

__all__ = ['fits_to_pl', 'iraf_commands', 'fix_pa_profile', 'isophote_to_ellip']


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


def fix_pa_profile(ellipse_output, pa_col='pa', delta_pa=75.0):
    """
    Correct the position angle for large jump.

    Parameters
    ----------
    ellipse_output: astropy.table
        Output table summarizing the result from `ellipse`.
    pa_col: string, optional
        Name of the position angle column. Default: pa
    delta_pa: float, optional
        Largest PA difference allowed for two adjacent radial bins. Default=75.

    Return
    ------
    ellipse_output with updated position angle column.

    """
    pa = ellipse_output[pa_col]

    for i in range(1, len(pa)):
        if (pa[i] - pa[i - 1]) >= delta_pa:
            pa[i] -= 180.0
        elif pa[i] - pa[i - 1] <= (-1.0 * delta_pa):
            pa[i] += 180.0

    ellipse_output[pa_col] = pa

    return ellipse_output


def isophote_to_ellip(ellipse_output, x_pad=0.0, y_pad=0.0):
    """
    Convert ellipse results into ellipses for visualization.

    Parameters
    ----------
    ellipse_output: astropy.table
        Output table summarizing the result from `ellipse`.

    Return
    ------
    ell_list: list
        List of Matplotlib elliptical patches for making plot.

    """
    x = ellipse_output['x0'] - x_pad
    y = ellipse_output['y0'] - y_pad
    pa = ellipse_output['pa']
    a = ellipse_output['sma'] * 2.0
    b = ellipse_output['sma'] * 2.0 * (1.0 - ellipse_output['ell'])

    ell_list = [Ellipse(xy=np.array([x[i], y[i]]), width=np.array(b[i]),
                        height=np.array(a[i]), angle=np.array(pa[i]))
                for i in range(x.shape[0])]

    return ell_list
