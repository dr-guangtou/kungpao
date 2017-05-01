"""@file object.py
All the classes that deal with object on the image.
"""


class Object(object):
    """The base class for all other kinds of objects.
    """


#########################################################################################
#
# Our class hierarchy is:
#
#    Object
#        --- Star
#        --- Galaxy
#        --- Diffuse
#    ObjectList
#
# Here we define the rest of these classes, and implement some common functions
#
#########################################################################################
