#!/usr/bin/env python
# encoding: utf-8
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import sys
import random
import logging
import warnings
from datetime import datetime

# Matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib.patches import Ellipse

# Numpy
import numpy as np

# GalSim
import galsim
print("# GalSim version : ", galsim.__version__)


def singleSersic(xcen, ycen, flux, reff, nser, q, pa,
                 nx, ny, psf=None, scale=1.0, exptime=1.0,
                 trunc=0, max_fft=None):
    """Generate image for single Sersic component."""
    ser = galsim.Sersic(n=nser, half_light_radius=reff,
                        flux=(flux * exptime), trunc=trunc).shear(
                        q=q, beta=(0.0 * galsim.degrees)).rotate(
                        (90.0 - pa) * galsim.degrees).shift(
                        (xcen - (nx / 2.0)), (ycen - (ny / 2.0)))

    if psf is not None:
        if max_fft is not None:
            gsparams = galsim.GSParams(maximum_fft_size=max_fft)
        else:
            gsparams = galsim.GSParams()
        return (galsim.Convolve([ser, psf],
                                gsparams=gsparams).drawImage(
                nx=nx, ny=ny, method='no_pixel',
                scale=scale)).array
    else:
        return (ser.drawImage(nx=nx, ny=ny,
                              method='no_pixel',
                              scale=scale)).array


def doubleSersic(xcen1, ycen1, flux1, reff1, nser1, q1, pa1,
                 xcen2, ycen2, flux2, reff2, nser2, q2, pa2,
                 nx, ny, psf=None, scale=1.0, exptime=1.0,
                 trunc=0, max_fft=None):
    """Generate image for double Sersic component."""
    ser = galsim.Add([galsim.Sersic(n=nser1, half_light_radius=reff1,
                         flux=(flux1 * exptime), trunc=trunc).shear(
                         q=q1, beta=(0.0 * galsim.degrees)).rotate(
                         (90.0 - pa1) * galsim.degrees).shift(
                         (xcen1 - (nx / 2.0)), (ycen1 - (ny / 2.0))),
                      galsim.Sersic(n=nser2, half_light_radius=reff2,
                         flux=(flux2 * exptime), trunc=trunc).shear(
                         q=q2, beta=(0.0 * galsim.degrees)).rotate(
                         (90.0 - pa2) * galsim.degrees).shift(
                         (xcen2 - (nx / 2.0)), (ycen2 - (ny / 2.0)))])

    if psf is not None:
        if max_fft is not None:
            gsparams = galsim.GSParams(maximum_fft_size=max_fft)
            return (galsim.Convolve([ser, psf],
                                    gsparams=gsparams).drawImage(
                    nx=nx, ny=ny, method='no_pixel',
                    scale=scale)).array
        else:
            return (galsim.Convolve([ser, psf]).drawImage(
                    nx=nx, ny=ny, method='no_pixel',
                    scale=scale)).array
    else:
        return (ser.drawImage(nx=nx, ny=ny,
                              method='no_pixel',
                              scale=scale)).array


def quickFits(data, name):
    hdu = fits.PrimaryHDU(data)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(name)


def mag2Flux(mag, zp, exptime):
    """Convert magnitude into flux"""
    return (10.0 ** ((zp - mag) / 2.5)) * exptime


def savePng(img, pngFile='galFake.png'):
    """Generate a PNG picture of the model.

    @param  img:
        GalSim Image object.

    @param  pngFile:
        Name of the PNG file (default='galFake.png')
    """
    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax1.imshow(np.arcsinh(img.array), aspect='auto', origin='lower')
    ax1.xaxis.set_major_formatter(NullFormatter())
    ax1.yaxis.set_major_formatter(NullFormatter())

    fig.savefig(pngFile, format='png')


def getPsf(fwhm, psfFile='psfFake.fits'):
    """Make a PSF image, and pass the PSF object.

    @param fwhm:
        FWHM of the PSF in pixel
    """
    logging.basicConfig(format="%(message)s", level=logging.INFO,
                        stream=sys.stdout)
    logger = logging.getLogger("psf")

    nx, ny = np.floor(fwhm * 10), np.floor(fwhm * 10)
    scale = 1.0
    logger.info('PSF: FWHM=%4.1f (%d, %d)', fwhm, nx, ny)

    # Make a PSF Image
    psfImg = galsim.ImageF(nx, ny)
    psfCen = psfImg.bounds.trueCenter()
    psfWcs = galsim.OffsetWCS(scale=scale, origin=psfCen)
    psfImg.wcs = psfWcs

    # PSF model
    psf = galsim.Gaussian(fwhm=fwhm)

    # Draw PSF image
    psf.drawImage(psfImg, method='no_pixel')

    # Save the fits image
    galsim.fits.write(psfImg, psfFile)
    logger.info('Write the PSF to %s', psfFile)

    return psf


def galaxy1(sky=0.2, galSNR=5000.0, fwhm=6.0):
    """Make the first fake galaxy.

    @param  sky:
        Per pixel sky value, used to get Poisson noise (default=0.2)

    @param  galSNR:
        The SNR of the galaxy, used to get the noise (default=5000.0)

    @param  fwhm:
        FWHM of the Gaussian PSF used to convolve the image (default=6.0)

    ------
    Three component elliptical galaxy   :

        Component 1: xCen=405.0, yCen=398; FLux=200;
                     Re=6.0;   n=2.5, q=0.9, PA1=10.0

        Component 2: xCen=405.0, yCen=398; FLux=400;
                     Re=30.0;  n=2.2, q=0.8, PA1=20.0

        Component 3: xCen=405.0, yCen=398; FLux=700;
                     Re=100.0; n=2.2, q=0.6, PA1=35.0

        Contamination 1: FLux=100; Re=20.0; n=3.0, q=0.7, PA1=70.0

        Contamination 2: FLux=200; Re=30.0; n=1.0, q=0.3, PA1=75.0
    """
    logging.basicConfig(format="%(message)s", level=logging.INFO,
                        stream=sys.stdout)
    logger = logging.getLogger("galaxy1")

    # Basic information of the image
    nx, ny = 800, 800
    exptime = 100.0
    scale = 1.0
    logger.info('Galaxy1: Sky=%6.2f, SNR=%5d', sky, galSNR)

    # Get the PSF, and save the PSF image
    psf = getPsf(fwhm, psfFile='galaxy1_psf.fits')

    # Make a GalSim Image
    gal1Img = galsim.ImageF(nx, ny)
    imgCen = gal1Img.bounds.trueCenter()
    imgWcs = galsim.OffsetWCS(scale=scale, origin=imgCen)
    gal1Img.wcs = imgWcs

    # Component 1: flux1 = 200; reff1 = 6.0; nser1 = 2.5; q1 = 0.9; pa1 = 10.0
    comp1 = galsim.Sersic(n=2.5, half_light_radius=6.0,
                          flux=(200.0 * exptime)).shear(
                          q=0.9, beta=(0.0 * galsim.degrees)).rotate(
                          10.0 * galsim.degrees).shift(
                          (405.0 - (nx / 2.0)), (398.0 - (ny / 2.0)))

    # Component 1: flux1 = 400; reff1 = 30.0; nser1 = 2.2; q1 = 0.8; pa1 = 20.0
    comp2 = galsim.Sersic(n=2.2, half_light_radius=30.0,
                          flux=(400.0 * exptime)).shear(
                          q=0.8, beta=(0.0 * galsim.degrees)).rotate(
                          20.0 * galsim.degrees).shift(
                          (405.0 - (nx / 2.0)), (398.0 - (ny / 2.0)))

    # Component 3: flux1 = 700; reff1 = 100.0; nser1 = 2.0; q1 = 0.6; pa1 = 35.0
    comp3 = galsim.Sersic(n=2.0, half_light_radius=100.0,
                          flux=(700.0 * exptime)).shear(
                          q=0.6, beta=(0.0 * galsim.degrees)).rotate(
                          35.0 * galsim.degrees).shift(
                          (405.0 - (nx / 2.0)), (398.0 - (ny / 2.0)))

    # Contamination 1:
    cont1 = galsim.Sersic(n=3.0, half_light_radius=20.0,
                          flux=(100.0 * exptime)).shear(
                          q=0.7, beta=(0.0 * galsim.degrees)).rotate(
                          70.0 * galsim.degrees).shift(150, 150)

    # Contamination 2:
    cont2 = galsim.Sersic(n=1.0, half_light_radius=30.0,
                          flux=(200.0 * exptime)).shear(
                          q=0.3, beta=(0.0 * galsim.degrees)).rotate(
                          75.0 * galsim.degrees).shift(-200, -200)

    # Add all components together
    gal1 = galsim.Add([comp1, comp2, comp3, cont1, cont2])

    # Convolution
    gal1Conv = galsim.Convolve([psf, gal1])

    # Draw the image
    gal1Conv.drawImage(gal1Img, method='no_pixel')

    # Add Noise
    rng = random.seed(datetime.now())
    noise = galsim.PoissonNoise(rng, sky_level=sky)
    gal1Img.addNoiseSNR(noise, galSNR)

    # Save the FITS image
    gal1File = 'galaxy1_img.fits'
    logger.info('Write to FITS file : %s', gal1File)
    galsim.fits.write(gal1Img, gal1File)

    # Save the PNG picture
    savePng(gal1Img, pngFile='galaxy1_img.png')

    return gal1Img


def galaxy2(sky=0.15, galSNR=6000.0, fwhm=7.0):
    """Make the second fake galaxy.

    @param  sky:
        Per pixel sky value, used to get Poisson noise (default=0.2)

    @param  galSNR:
        The SNR of the galaxy, used to get the noise (default=5000.0)

    @param  fwhm:
        FWHM of the Gaussian PSF used to convolve the image (default=6.0)

    ------
    Three component edge-on disk galaxy with bulge+disk+halo   :

        Component 1: xCen=420.0, yCen=380; FLux=300;
                     Re=8.0;   n=3.0, q=0.8, PA1=45.0

        Component 2: xCen=420.0, yCen=380; FLux=500;
                     Re=40.0;  n=0.8, q=0.1, PA1=45.0

        Component 3: xCen=420.0, yCen=380; FLux=600;
                     Re=35.0;  n=1.5, q=0.6, PA1=40.0

        Contamination 1: FLux=100; Re=20.0; n=2.5, q=0.7, PA1=0.0

        Contamination 2: FLux=200; Re=30.0; n=2.0, q=0.6, PA1=75.0
    """
    logging.basicConfig(format="%(message)s", level=logging.INFO,
                        stream=sys.stdout)
    logger = logging.getLogger("galaxy2")

    # Basic information of the image
    nx, ny = 800, 800
    exptime = 100.0
    scale = 1.0
    logger.info('Galaxy2: Sky=%6.2f, SNR=%5d', sky, galSNR)

    # Get the PSF, and save the PSF image
    psf = getPsf(fwhm, psfFile='galaxy2_psf.fits')

    # Make a GalSim Image
    gal2Img = galsim.ImageF(nx, ny)
    imgCen = gal2Img.bounds.trueCenter()
    imgWcs = galsim.OffsetWCS(scale=scale, origin=imgCen)
    gal2Img.wcs = imgWcs

    # Component 1: flux1 = 300; reff1 = 8.0; nser1 = 3.0; q1 = 0.8; pa1 = 45.0
    comp1 = galsim.Sersic(n=3.0, half_light_radius=8.0,
                          flux=(300.0 * exptime)).shear(
                          q=0.8, beta=(0.0 * galsim.degrees)).rotate(
                          45.0 * galsim.degrees).shift(
                          (420.0 - (nx / 2.0)), (380.0 - (ny / 2.0)))

    # Component 1: flux1 = 500; reff1 = 40.0; nser1 = 0.8; q1 = 0.1; pa1 = 45.0
    comp2 = galsim.Sersic(n=0.8, half_light_radius=40.0,
                          flux=(500.0 * exptime)).shear(
                          q=0.10, beta=(0.0 * galsim.degrees)).rotate(
                          45.0 * galsim.degrees).shift(
                          (420.0 - (nx / 2.0)), (380.0 - (ny / 2.0)))

    # Component 3: flux1 = 600; reff1 = 35.0; nser1 = 1.5; q1 = 0.6; pa1 = 40.0
    comp3 = galsim.Sersic(n=1.5, half_light_radius=35.0,
                          flux=(600.0 * exptime)).shear(
                          q=0.6, beta=(0.0 * galsim.degrees)).rotate(
                          40.0 * galsim.degrees).shift(
                          (420.0 - (nx / 2.0)), (380.0 - (ny / 2.0)))

    # Contamination 1:
    cont1 = galsim.Sersic(n=2.5, half_light_radius=20.0,
                          flux=(100.0 * exptime)).shear(
                          q=0.7, beta=(0.0 * galsim.degrees)).rotate(
                          0.0 * galsim.degrees).shift(50, 200)

    # Contamination 2:
    cont2 = galsim.Sersic(n=2.0, half_light_radius=30.0,
                          flux=(200.0 * exptime)).shear(
                          q=0.6, beta=(0.0 * galsim.degrees)).rotate(
                          75.0 * galsim.degrees).shift(-300, -190)

    # Add all components together
    gal2 = galsim.Add([comp1, comp2, comp3, cont1, cont2])

    # Convolution
    gal2Conv = galsim.Convolve([psf, gal2])

    # Draw the image
    gal2Conv.drawImage(gal2Img, method='no_pixel')

    # Add Noise
    rng = random.seed(datetime.now())
    noise = galsim.PoissonNoise(rng, sky_level=sky)
    gal2Img.addNoiseSNR(noise, galSNR)

    # Save the FITS file
    gal2File = 'galaxy2_img.fits'
    logger.info('Write to FITS file : %s', gal2File)
    galsim.fits.write(gal2Img, gal2File)

    # Save the PNG picture
    savePng(gal2Img, pngFile='galaxy2_img.png')

    return gal2Img


if __name__ == "__main__":

    gal1 = galaxy1()
    gal2 = galaxy2()
