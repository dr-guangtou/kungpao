#!/usr/bin/env python
# encoding: utf-8
"""Cover the INIFILE with the LOGFILE."""

import os
import shutil
import argparse

from astropy.io import fits


def run(inputFits):
    """Convert the GALFIT output log to input file."""
    if not os.path.isfile(inputFits):
        raise Exception("## Can not find the input fits results: %s" %
                        inputFits)
    """Read in the Header of the model HDU"""
    modHead = fits.open(inputFits)[2].header
    """Find the input file"""
    iniFile = modHead['INITFILE']
    """Find the log file"""
    logFile = modHead['LOGFILE']
    """See if the LOGFILE is available"""
    if not os.path.isfile(logFile):
        raise Exception("## Can not find the LOGFILE : %s" % logFile)
    if iniFile not in open(logFile, 'r').read():
        raise Exception("## Wrong LOGFILE?")
    """Copy the LOGFILE to INITFILE"""
    shutil.copyfile(logFile, iniFile)

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("inputFits", help="Name of the input models")

    args = parser.parse_args()

    run(args.inputFits)
