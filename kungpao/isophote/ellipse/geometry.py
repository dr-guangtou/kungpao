"""Geometry related functions for Isophote."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import math
import numpy as np

from .constant import (PI, TWOPI, DEFAULT_STEP,
                       PHI_MAX, PHI_MIN)

__all__ = ['Geometry', 'normalize_angle']


def normalize_angle(angle):
    '''
    Restore angle to valid range (0 - PI) in radians.

    :param angle: float
         the angle
    :return:  float
         the input angle expressed in the range (0 - PI)
    '''
    while angle > TWOPI:
        angle -= TWOPI

    if angle > PI:
        angle -= PI

    while angle < -1.0 * TWOPI:
        angle += PI

    if angle < -1.0 * PI:
        angle += PI

    if angle < 0.:
        angle += PI

    return angle


def _area(sma, eps, phi, r):
    '''
    Utility function used in the computation of elliptical sector areas.
    '''
    aux = r * math.cos(phi) / sma
    signal = aux / abs(aux)
    if abs(aux) >= 1.:
        aux = signal
    return (abs(sma**2 * (1.0 - eps) / 2.0 * math.acos(aux)))


class Geometry(object):
    def __init__(self,
                 x0,
                 y0,
                 sma,
                 eps,
                 pa,
                 astep=DEFAULT_STEP,
                 linear_growth=False):
        '''
        This is basically a container that allows storage of all parameters
        associated with a given ellipse's geometry.

        Parameters that describe the relationship of a given ellipse with
        other associated ellipses are also encapsulated in this container.
        These associated ellipses may include e.g. the two (inner and outer)
        bounding ellipses that are used to build sectors along the elliptical
        path. These sectors are used as areas for integrating pixel values,
        when area integration mode (mean or median) is used.

        The Geometry object also keeps track of *where* in the ellipse we are,
        when performing an 'extract' operation. This is mostly relevant when
        using an area integration mode (as opposed to a pixel integration mode)

        :param x0: float
            center coordinate in pixels along image row
        :param y0: float
            center coordinate in pixels along image column
        :param sma: float
             semi-major axis in pixels
        :param eps: ellipticity
             ellipticity
        :param pa: float, units radians
             position angle of semi-major axis in relation to the +X axis of
             the image array (rotating towards the +Y axis). Position angles
             are defined in the range 0 < PA <= np.pi. Avoid using as starting
             position angle 'pa0 = 0.', since the fit algorithm may not work
             properly. When the ellipses are such that position angles are near
             either extreme of the range, noise can make the solution jump back
             and forth between successive isophotes, by amounts close to 180
             degrees.
        :param astep: float, default = 0.1
            step value for growing/shrinking the semi-major axis. It can be
            expressed either in pixels (when 'linear_growth'=True) or in
            relative value (when 'linear_growth=False')
        :param linear_growth: boolean, default = False
            semi-major axis growing/shrinking mode
        '''
        self.x0 = x0
        self.y0 = y0
        self.sma = sma
        self.eps = eps
        self.pa = pa

        self.astep = astep
        self.linear_growth = linear_growth

        # variables used in the calculation of the sector angular width.
        sma1, sma2 = self.bounding_ellipses()
        inner_sma = min((sma2 - sma1), 3.)
        self._area_factor = (sma2 - sma1) * inner_sma

        # sma can eventually be zero!
        if self.sma > 0.:
            self.sector_angular_width = max(
                min((inner_sma / self.sma), PHI_MAX), PHI_MIN)
            self.initial_polar_angle = self.sector_angular_width / 2.

            self.initial_polar_radius = self.radius(self.initial_polar_angle)

    def radius(self, angle):
        '''
        Given a polar angle, return the corresponding polar radius.

        :param angle: float
            polar angle (radians)
        :return: float
            polar radius (pixels)
        '''
        try:
            rad = self.sma * (1. - self.eps) / math.sqrt(
                ((1. - self.eps) * math.cos(angle))**2 + (math.sin(angle))**2)
        except Exception:
            rad = 1.0E-4

        return rad

    def initialize_sector_geometry(self, phi):
        '''
        Initialize geometry attributes associated with an elliptical sector
        at polar angle 'phi'.

        Computes:
         - the four vertices that define the elliptical sector on the
           pixel array.
         - sector area (in attribute self.sector_area)
         - sector angular width (in attribute self.sector_angular_width)

        :param phi: float
            polar angle (radians) where the sector is located.
        :return: tuple with two 1-D np arrays
            with the X and Y coordinates of each vertex.
        '''

        # These polar radii bound the region between the inner
        # and outer ellipses that define the sector.
        sma1, sma2 = self.bounding_ellipses()
        eps_ = 1. - self.eps
        # polar vector at one side of the elliptical sector
        self._phi1 = phi - self.sector_angular_width / 2.
        r1 = sma1 * eps_ / math.sqrt((eps_ * math.cos(self._phi1))**2 +
                                     (math.sin(self._phi1))**2)
        r2 = sma2 * eps_ / math.sqrt((eps_ * math.cos(self._phi1))**2 +
                                     (math.sin(self._phi1))**2)
        # polar vector at the other side of the elliptical sector
        self._phi2 = phi + self.sector_angular_width / 2.
        r3 = sma2 * eps_ / math.sqrt((eps_ * math.cos(self._phi2))**2 +
                                     (math.sin(self._phi2))**2)
        r4 = sma1 * eps_ / math.sqrt((eps_ * math.cos(self._phi2))**2 +
                                     (math.sin(self._phi2))**2)

        # sector area
        sa1 = _area(sma1, self.eps, self._phi1, r1)
        sa2 = _area(sma2, self.eps, self._phi1, r2)
        sa3 = _area(sma2, self.eps, self._phi2, r3)
        sa4 = _area(sma1, self.eps, self._phi2, r4)
        self.sector_area = abs((sa3 - sa2) - (sa4 - sa1))

        # angular width of sector. It is calculated such that the sectors
        # come out with roughly constant area along the ellipse.
        self.sector_angular_width = max(
            min((self._area_factor / (r3 - r4) / r4), PHI_MAX), PHI_MIN)

        # compute the 4 vertices that define the elliptical sector.
        vertex_x = np.zeros(shape=4, dtype=float)
        vertex_y = np.zeros(shape=4, dtype=float)

        # vertices are labelled in counterclockwise sequence
        vertex_x[0] = r1 * math.cos(self._phi1 + self.pa) + self.x0
        vertex_y[0] = r1 * math.sin(self._phi1 + self.pa) + self.y0
        vertex_x[1] = r2 * math.cos(self._phi1 + self.pa) + self.x0
        vertex_y[1] = r2 * math.sin(self._phi1 + self.pa) + self.y0
        vertex_x[2] = r4 * math.cos(self._phi2 + self.pa) + self.x0
        vertex_y[2] = r4 * math.sin(self._phi2 + self.pa) + self.y0
        vertex_x[3] = r3 * math.cos(self._phi2 + self.pa) + self.x0
        vertex_y[3] = r3 * math.sin(self._phi2 + self.pa) + self.y0

        return vertex_x, vertex_y

    def bounding_ellipses(self):
        '''
        Compute the semi-major axis of the two ellipses that bound
        the annulus where integrations take place.

        :return: tuple:
            with two floats - the smaller and larger values of
            SMA that define the annulus  bounding ellipses
        '''
        if (self.linear_growth):
            a1 = self.sma - self.astep / 2.
            a2 = self.sma + self.astep / 2.

        else:
            a1 = self.sma * (1. - self.astep / 2.)
            a2 = self.sma * (1. + self.astep / 2.)

        return a1, a2

    def polar_angle_sector_limits(self):
        '''
        Returns the two polar angles that bound the sector.

        The two bounding polar angles only become available after
        calling method initialize_sector_geometry(phi).

        :return: tuple:
            with two floats - the smaller and larger values of
            polar angle that bound the current sector
        '''
        return self._phi1, self._phi2

    def to_polar(self, x, y):
        '''
        Given x,y coordinates on image grid, returns radius
        and polar angle on ellipse coordinate system. Takes
        care of different definitions for pa and phi:

        -PI < pa < PI
        0 < phi  < 2*PI

        Note that radius can be anything; solution is not tied
        to the semi-major axis length, but to the center position
        and tilt angle only.

        :param x: float
            image coordinate
        :param y: float
            image coordinate
        :return: 2 floats
            radius, angle
        '''
        x1 = x - self.x0
        y1 = y - self.y0

        radius = x1**2 + y1**2
        if radius > 0.0:
            radius = math.sqrt(radius)
            angle = math.asin(abs(y1) / radius)
        else:
            radius = 0.
            angle = 1.

        if x1 >= 0. and y1 < 0.:
            angle = 2 * np.pi - angle
        elif x1 < 0. and y1 >= 0.:
            angle = np.pi - angle
        elif x1 < 0. and y1 < 0.:
            angle = np.pi + angle

        pa1 = self.pa
        if self.pa < 0.:
            pa1 = self.pa + 2 * np.pi
        angle = angle - pa1
        if angle < 0.:
            angle = angle + 2 * np.pi

        return radius, angle

    def update_sma(self, step):
        '''
        Return an updated value for the semi-major axis, given the
        current value and the updating step value. The step value must
        be managed by the caller so as to support both modes: grow
        outwards, and shrink inwards.

        :param step: float
            the step value
        :return: float
            the new semi-major axis length
        '''
        if self.linear_growth:
            sma = self.sma + step
        else:
            sma = self.sma * (1. + step)
        return sma

    def reset_sma(self, step):
        '''
        This method should be used whenever one wants to switch the
        direction of semi-major axis growth, from outwards to inwards.

        :param step: float
            the current step value
        :return: float, float
            the new semi-major axis length and the new step value to
            initiate the semi-major axis length shrink inwards. This
            is the step value that should be used when calling method
            update_sma.
        '''
        if self.linear_growth:
            sma = self.sma - step
            step = -step
        else:
            aux = 1. / (1. + step)
            sma = self.sma * aux
            step = aux - 1.

        return sma, step
