"""Integrators for Isophote."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import math
import numpy.ma as ma

from .constant import *

__all__ = ['AreaIntegrator', 'BiLinearIntegrator', 'Integrator',
           'MeanIntegrator', 'MedianIntegrator', 'NearestNeighborIntegrator',
           'integrators']


class Integrator(object):

    def __init__(self, image, geometry, angles, radii, intensities):
        '''
        Constructor

        :param image: 2-d numpy array
             image array
        :param geometry: Geometry instance
            object that encapsulates geometry information about current ellipse
        :param angles: list
            output list; contains the angle values along the elliptical path
        :param radii:  list
            output list; contains the radius values along the elliptical path
        :param intensities: list
            output list; contains the extracted intensity values along the elliptical path
        '''
        self._image = image
        self._geometry = geometry

        self._angles = angles
        self._radii = radii
        self._intensities = intensities

        # for bounds checking
        self._i_range = range(0, self._image.shape[1] - 1)
        self._j_range = range(0, self._image.shape[0] - 1)

    def integrate(self, radius, phi):
        '''
        The three input lists are updated with one sample point taken
        from the image by a chosen integration method.

        Sub classes should implement the actual integration method.

        :param radius: float
            length of radius vector in pixels
        :param phi: float
            polar angle of radius vector
        '''
        raise NotImplementedError

    def _reset(self):
        '''
        Starts the results lists anew.

        This method is for internal use and shouldn't
        be used by external callers.
        '''
        self._angles = []
        self._radii = []
        self._intensities = []

    def _store_results(self, phi, radius, sample):
        self._angles.append(phi)
        self._radii.append(radius)
        self._intensities.append(sample)

    def get_polar_angle_step(self):
        raise NotImplementedError

    def get_sector_area(self):
        raise NotImplementedError

    def is_area(self):
        raise NotImplementedError


class NearestNeighborIntegrator(Integrator):

    def integrate(self, radius, phi):

        self._r = radius

        # Get image coordinates of (radius, phi) pixel
        i = int(radius * math.cos(phi + self._geometry.pa) + self._geometry.x0)
        j = int(radius * math.sin(phi + self._geometry.pa) + self._geometry.y0)

        # ignore data point if outside image boundaries
        if (i in self._i_range) and (j in self._j_range):

            sample = self._image[j][i]

            if sample is not ma.masked:
                self._store_results(phi, radius, sample)

    def get_polar_angle_step(self):
        return 1. / self._r

    def get_sector_area(self):
        return 1.

    def is_area(self):
        return False


class BiLinearIntegrator(Integrator):

    def integrate(self, radius, phi):

        self._r = radius

        # Get image coordinates of (radius, phi) pixel
        x_ = radius * math.cos(phi + self._geometry.pa) + self._geometry.x0
        y_ = radius * math.sin(phi + self._geometry.pa) + self._geometry.y0
        i = int(x_)
        j = int(y_)
        fx = x_ - i
        fy = y_ - j

        # ignore data point if outside image boundaries
        if (i in self._i_range) and (j in self._j_range):

            # TODO: in the future, will need to handle masked pixels here
            qx = 1. - fx
            qy = 1. - fy

            try:
                if self._image[j][i] is not ma.masked and \
                   self._image[j+1][i] is not ma.masked and \
                   self._image[j][i+1] is not ma.masked and \
                   self._image[j+1][i+1] is not ma.masked:

                    sample = self._image[j][i]     * qx * qy + \
                             self._image[j+1][i]   * qx * fy + \
                             self._image[j][i+1]   * fx * qy + \
                             self._image[j+1][i+1] * fy * fx

                    # store results
                    self._store_results(phi, radius, sample)
            except Exception:
                print("index %d %d" % (i, j))

    # def _subpix(self, image, i, j, fx, fy):
    #
    #     z1 = image[j][i]
    #     z2 = image[j][i+1]
    #     z3 = image[j+1][i]
    #     z4 = image[j+1][i+1]
    #
    #     sum = 0.
    #     a1  = z2 - z1
    #     a2  = z4 - z3
    #     a3  = 1./ NCELL
    #     correction = 0.5 + a3 / 2.
    #     for j in range(0, NCELL):
    #         y = j * a3 + fy - correction
    #         for i in range(0, NCELL):
    #             x = i * a3 + fx - correction
    #             za = a1 * x + z1
    #             zb = a2 * x + z3
    #             z  = (zb - za) * y + za
    #             sum += z
    #     return sum / NCELL**2

    def get_polar_angle_step(self):
        return 1. / self._r

    def get_sector_area(self):
        return 2.

    def is_area(self):
        return False


class AreaIntegrator(Integrator):

    def __init__(self, image, geometry, angles, radii, intensities):

        super(AreaIntegrator, self).__init__(image, geometry, angles, radii, intensities)

        # build auxiliary bi-linear integrator to be used when
        # sector areas contain a too small number of valid pixels.
        self._bi_linear_integrator = integrators[BI_LINEAR](image, geometry, angles, radii, intensities)

    def integrate(self, radius, phi):

        self._phi = phi

        # Get image coordinates of the four vertices of the elliptical sector.
        vertex_x, vertex_y = self._geometry.initialize_sector_geometry(phi)

        self._sector_area = self._geometry.sector_area

        # step in polar angle to be used by caller next time
        # when updating the current polar angle 'phi' to point
        # to the next sector.
        self._phistep = self._geometry.sector_angular_width

        # define rectangular image area that encompasses the elliptical
        # sector. We have to account for rounding of pixel indices.
        i1 = int(min(vertex_x)) - 1
        j1 = int(min(vertex_y)) - 1
        i2 = int(max(vertex_x)) + 1
        j2 = int(max(vertex_y)) + 1

        # polar angle limits for this sector
        phi1, phi2 = self._geometry.polar_angle_sector_limits()

        # ignore data point if the elliptical sector lies
        # partially, ou totally, outside image boundaries
        if (i1 in self._i_range) and (j1 in self._j_range) and \
           (i2 in self._i_range) and (j2 in self._j_range):

            # Scan rectangular image area, compute sample value.
            npix = 0
            accumulator = self.initialize_accumulator()
            for j in range(j1,j2):
                for i in range(i1, i2):
                    # Check if polar coordinates of each pixel
                    # put it inside elliptical sector.
                    rp, phip = self._geometry.to_polar(i, j)

                    # check if inside angular limits
                    if phip < phi2 and phip >= phi1:

                        # check if radius is inside bounding ellipses
                        sma1, sma2 = self._geometry.bounding_ellipses()
                        aux = (1. - self._geometry.eps) / math.sqrt(((1. - self._geometry.eps) *
                              math.cos(phip))**2 + (math.sin(phip))**2)
                        r1 = sma1 * aux
                        r2 = sma2 * aux

                        if rp < r2 and rp >= r1:
                            # update accumulator with pixel value
                            pix_value = self._image[j][i]
                            if pix_value is not ma.masked:
                                accumulator, npix = self.accumulate(pix_value,
                                                                    accumulator)

            # If 6 or less pixels were sampled, get the bi-linear interpolated value instead.
            if npix in range (0,7):
                # must reset integrator to remove older samples.
                self._bi_linear_integrator._reset()
                self._bi_linear_integrator.integrate(radius, phi)
                # because it was reset, current value is the only one stored
                # internally in the bi-linear integrator instance. Move it
                # from the internal integrator to this instance.
                if len(self._bi_linear_integrator._intensities) > 0:
                    sample_value = self._bi_linear_integrator._intensities[0]
                    self._store_results(phi, radius, sample_value)

            elif npix > 6:
                sample_value = self.compute_sample_value(accumulator)
                self._store_results(phi, radius, sample_value)

    def get_polar_angle_step(self):
        phi1, phi2 = self._geometry.polar_angle_sector_limits()
        phistep = self._geometry.sector_angular_width / 2. + phi2 - self._phi
        return phistep

    def get_sector_area(self):
        return self._sector_area

    def is_area(self):
        return True

    def initialize_accumulator(self):
        raise NotImplementedError

    def accumulate(self, pixel_value, accumulator):
        raise NotImplementedError

    def compute_sample_value(self, accumulator):
        raise NotImplementedError


class MeanIntegrator(AreaIntegrator):

    def initialize_accumulator(self):
        accumulator = 0.
        self._npix = 0
        return accumulator

    def accumulate(self, pixel_value, accumulator):
        accumulator += pixel_value
        self._npix += 1
        return accumulator, self._npix

    def compute_sample_value(self, accumulator):
        return accumulator / self._npix


class MedianIntegrator(AreaIntegrator):

    def initialize_accumulator(self):
        accumulator = []
        self._npix = 0
        return accumulator

    def accumulate(self, pixel_value, accumulator):
        accumulator.append(pixel_value)
        self._npix += 1
        return accumulator, self._npix

    def compute_sample_value(self, accumulator):
        accumulator.sort()
        return accumulator[int(self._npix/2)]


# Specific integrator subclasses can be instantiated from here.

integrators = {
    NEAREST_NEIGHBOR: NearestNeighborIntegrator,
    BI_LINEAR: BiLinearIntegrator,
    MEAN: MeanIntegrator,
    MEDIAN: MedianIntegrator
}
