"""@file image.py
All the classes that deal with images of galaxies.
"""


class Image(object):
    """The base class for all other kinds of images.
    """

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
