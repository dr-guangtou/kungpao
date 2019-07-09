#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deal with configuration file for using `Ellipse`."""

__all__ = ['DEFAULT_CONFIG', 'EllipseConfig']

DEFAULT_CONFIG = 'default.yaml'

class EllipseConfig(object):
    """
    Class to deals with the configuration parameters for `Ellipse` in `IRAF`. 
    """
    def __init__(self, config=None):
        """
        
        """