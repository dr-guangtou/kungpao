#!/usr/bin/env python
# encoding: utf-8
"""Cover the INIFILE with the LOGFILE."""

import os
import shutil
import argparse

from astropy.io import fits

__all__ = ['log_to_input']


def log_to_input(input_fits):
    """Convert the GALFIT output log to input file."""
    if not os.path.isfile(input_fits):
        raise Exception("## Can not find the input fits results: %s" %
                        input_fits)

    # Read in the Header of the model HDU"""
    model_header = fits.open(input_fits)[2].header
    # Find the input file"""
    ini_file = model_header['INITFILE']
    # Find the log file"""
    log_file = model_header['LOGFILE']

    # See if the LOGFILE is available"""
    if not os.path.isfile(log_file):
        raise Exception("## Can not find the LOGFILE : %s" % log_file)
    if ini_file not in open(log_file, 'r').read():
        raise Exception("## Wrong LOGFILE?")

    # Copy the LOGFILE to INITFILE"""
    shutil.copyfile(log_file, ini_file)

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Name of the input models")

    args = parser.parse_args()

    log_to_input(args.input)
