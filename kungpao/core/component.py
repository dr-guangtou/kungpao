"""@file component.py
All the classes that describe structural components of galaxies.
"""
import numpy
from scipy import special


class Component(object):
    """The base class for all other kinds of structural component.

    A Component is not intended to be constructed directly.
    Normally, you would use whatever derived class is appropriate for the surface
    brightness profile you want:

    Access Methods
    --------------
    """
    def fromConfig(parameters):
        """Placeholder for function that initiates a component from a
        configuration dictionary.
        """
        return None

    def toConfig():
        """Placeholder for function that outputs the parameter of the
        component to a dictionary.
        """
        parameters = {}
        return parameters

    def __init__(self, obj):
        # This guarantees that all Component(s) have a "params"
        if isinstance(obj, Component):
            self.params = obj.params
            if hasattr(obj, 'noise'):
                self.noise = obj.noise
        else:
            raise TypeError("Component must be initialized with another Component!")


#########################################################################################
#
# Our class hierarchy is:
#
#    Component
#        --- Sersic
#           --- deVaucouleurs
#           --- Exponential
#           ---
#        --- Gaussian
#        --- Edge-on Disk
#        --- Others....
#    MultiComponent
#
# Here we define the rest of these classes, and implement some common functions
#
#########################################################################################

class Sersic(Component):
    """The class that describes everything about Sersic profile.

    Initialization
    --------------

    Example:
    -------

    Methods:
    -------
    """
    def __init__(self, params, lower=None, upper=None):
