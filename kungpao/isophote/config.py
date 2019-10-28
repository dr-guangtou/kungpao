#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deal with configuration file for using `Ellipse`."""

import os
import yaml

import kungpao

__all__ = ['DEFAULT_CONFIG', 'EllipseConfig']


DEFAULT_CONFIG = os.path.join(os.path.dirname(kungpao.__file__), 'isophote', 'default.yaml')

class EllipseConfig(object):
    """
    Class to deals with the configuration parameters for `Ellipse` in `IRAF`.
    """
    def __init__(self, config=None):
        """
        Setup the configuration parameters, make sure they are reasonable.
        """
        cfg_default = yaml.load(open(DEFAULT_CONFIG))

        if config is not None:
            cfg = yaml.load(open(config))
