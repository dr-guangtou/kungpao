#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re

try:
    from setuptools import setup
    setup
except ImportError:
    from distutils.core import setup
    setup

setup(
    name="kungpao",
    version='0.0.1',
    author="Song Huang",
    author_email="shuang89@ucsc.edu",
    packages=["kungpao"],
    url="https://github.com/dr-guangtou/kungpao",
    license="LICENSE",
    description="Deliciou Galaxies!",
    long_description=open("README.md").read() + "\n\n",
    package_data={"kungpao": ["cmap_data/*pkl"]},
    include_package_data=True,
    #install_requires=["numpy", "scipy >= 0.9", "astropy", "matplotlib", "scikit-learn"],
)