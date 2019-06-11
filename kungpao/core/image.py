#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""@file image.py
All the classes that deal with images of galaxies.
"""

class ImageException(Exception):
    """Class for error related to image.
    """
    pass

class Image(object):
    """The base class for all other kinds of images.
    """
    def __init__():

    @property
    def image(self):
        """
        Structured array that describe the mask planes.
        """
        return self._bitmasks

    @bitmasks.setter
    def bitmasks(self, mask_array):
        self._bitmasks = mask_array

#########################################################################################
#
# Our class hierarchy is:
#
#    Image
#        --- ImageSci
#        --- ImageVar
#        --- ImageSig
#        --- ImageMsk
#        --- ImagePsf
#    MultibandImage
#
# Here we define the rest of these classes, and implement some common functions
#
#########################################################################################
