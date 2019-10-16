"""Helper functions for isophote analysis."""


def maskFits2Pl(inputImage, inputMask, replace=False):
    """
    Convert the FITS mask into the IRAF .pl file.

    This is stupid....

    Parameters:
    """
    if not os.path.isfile(inputMask):
        raise Exception("Can not find the FITS mask: %s" % inputMask)
    # Why the hell the .pl mask is not working under
    if not replace:
        outputMask = inputImage.replace('.fits', '.fits.pl')
    else:
        outputMask = inputImage.replace('.fits', '.pl')
    if os.path.isfile(outputMask):
        os.remove(outputMask)
    # Convert the fits format mask into pl format.
    iraf.unlearn('imcopy')
    iraf.imcopy(input=inputMask, output=outputMask, verbose=True)

    return outputMask

