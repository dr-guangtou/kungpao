"""Useful tool for image reduction."""

from astropy.io import fits
from astropy.nddata import Cutout2D

__all__ = ['img_cutout', 'get_pixel_value']


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


def get_pixel_value(img, wcs, ra, dec):
    """Return the pixel value from image based on RA, DEC.

    TODO:
        Should be absorbed into the image object later

    Parameters
    ----------
        img     : 2-D data array
        wcs     : WCS from the image header
        ra, dec : coordinates, can be array

    """
    px, py = wcs.wcs_world2pix(ra, dec, 0)

    import collections
    if not isinstance(px, collections.Iterable):
        pixValues = img[int(py), int(px)]
    else:
        pixValues = map(lambda x, y: img[int(y), int(x)], px, py)

    return np.asarray(pixValues)


def seg_remove_cen_obj(seg):
    """
    Remove the central object from the segmentation.

    TODO:
        Should be absorbed by objects for segmentation image
    """
    seg_copy = copy.deepcopy(seg)
    seg_copy[seg == seg[int(seg.shape[0] / 2L), int(seg.shape[1] / 2L)]] = 0

    return seg_copy


def seg_index_cen_obj(seg):
    """
    Remove the index array for central object.

    TODO:
        Should be absorbed by objects for segmentation image
    """
    cen_obj = seg[int(seg.shape[0] / 2L), int(seg.shape[1] / 2L)]
    if cen_obj == 0:
        return None
    else:
        return (seg == cen_obj)


def seg_remove_obj(seg, x, y):
    """
    Remove an object from the segmentation given its coordinate.

    TODO:
        Should be absorbed by objects for segmentation image
    """
    seg_copy = copy.deepcopy(seg)
    seg_copy[seg == seg[int(x), int(y)]] = 0

    return seg_copy


def seg_index_obj(seg, x, y):
    """
    Remove the index array for an object given its location.

    TODO:
        Should be absorbed by objects for segmentation image
    """
    obj = seg[int(x), int(y)]
    if obj == 0:
        return None
    else:
        return (seg == obj)


def parseRegEllipse(regName):
    """
    Parse a DS9 .reg files.

    convert the Ellipse or Circle regions
    into arrays of parameters for ellipse:
    x, y, a, b, theta
    """
    if os.path.isfile(regName):
        raise Exception("### Can not find the .reg file!")
    # Parse the .reg file into lines
    lines = [line.strip() for line in open(regName, 'r')]
    # Coordinate type of this .reg file: e.g. 'image'
    coordType = lines[2].strip()
    # Parse each region
    regs = [reg.split(" ") for reg in lines[3:]]

    xc = []
    yc = []
    ra = []
    rb = []
    theta = []

    for reg in regs:
        if reg[0].strip() == 'ellipse' and len(reg) is 6:
            xc.append(float(reg[1]))
            yc.append(float(reg[2]))
            ra.append(float(reg[3]))
            rb.append(float(reg[4]))
            theta.append(float(reg[5]) * np.pi / 180.0)
        elif reg[0].strip() == 'circle' and len(reg) is 4:
            xc.append(float(reg[1]))
            yc.append(float(reg[2]))
            ra.append(float(reg[3]))
            rb.append(float(reg[3]))
            theta.append(0.0)

    xc = np.array(xc, dtype=np.float32)
    yc = np.array(yc, dtype=np.float32)
    ra = np.array(ra, dtype=np.float32)
    rb = np.array(rb, dtype=np.float32)
    theta = np.array(theta, dtype=np.float32)

    return xc, yc, ra, rb, theta, coordType
