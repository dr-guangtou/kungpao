"""Helper functions for isophote analysis."""

import os
import platform
import subprocess

import numpy as np

from matplotlib.patches import Ellipse

import kungpao

__all__ = ['fits_to_pl', 'iraf_commands', 'fix_pa_profile', 'isophote_to_ellip',
           'save_isophote_output', 'remove_index_from_output']


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
        # TODO: Need to test whether .fits.pl or .pl works.
        output = fits.replace('.fits', '.fits.pl')

    if os.path.isfile(output):
        if verbose:
            print("# Output file exists! Will remove {}".format(output))
        os.remove(output)

    imcopy = "{} imcopy input={} output={} verbose=no".format(
        ximage, fits.strip(), output.strip()
    )

    os.system(imcopy)

    return output


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

    return (os.path.join(IRAF_DIR, 'x_isophote.e'),
            os.path.join(IRAF_DIR, 'x_ttools.e'),
            os.path.join(IRAF_DIR, 'x_images.e'))


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


def save_isophote_output(ellip_output, prefix=None, ellip_config=None, location=''):
    """
    Save the Ellipse output to file.

    Parameters
    ----------
    ellip_output: astropy.table
        Output table for the isophote analysis.
    ellip_config: dict
        Configuration parameters for the isophote analysis.
    prefix: string, optional
        Prefix of the output file. Default: None
    location: string, optional
        Directory to keep the output.

    Returns
    -------
    output_file: string
        Name of the output numpy record.

    """
    if prefix is None:
        prefix = 'ellip_output'
    output_file = os.path.join(location, prefix + ".npz")

    # Save the output and configuration parameters in a 'npz'.
    np.savez(output_file, output=ellip_output, config=ellip_config)

    return output_file


def remove_index_from_output(output_tab, replace='NaN'):
    """
    Remove the Indef values from the Ellipse output.

    Parameters:
    """
    if os.path.exists(output_tab):
        subprocess.call(['sed', '-i_back', 's/INDEF/' + replace + '/g', output_tab])
        # Remove the back-up file
        if os.path.isfile(output_tab.replace('.tab', '_back.tab')):
            os.remove(output_tab.replace('.tab', '_back.tab'))
    else:
        raise FileExistsError('Can not find the input catalog: {}'.format(output_tab))

    return output_tab
