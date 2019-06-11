"""Misc utilities."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import copy

import numpy as np

import smatch
import cosmology as cosmology_erin

from tqdm import tqdm

from astropy.table import Table, Column, vstack, unique, join

from .utils import kpc_scale_erin, angular_distance

cosmo_erin = cosmology_erin.Cosmo(H0=70.0, omega_m=0.30)

__all__ = ['table_pair_match_physical', 'filter_healpix_mask', 'smatch_catalog',
           'smatch_catalog_by_field', 'convert_bytes_to_int', 'convert_bytes_to_str']


def table_pair_match_physical(cat1, cat2, z_col='z_best', r_kpc=1E3,
                              cosmo=cosmo_erin, ra_col='ra', dec_col='dec',
                              include=False):
    """Count the pairs within certain distance."""
    num_pair = []
    index_pair = []

    for obj1 in tqdm(cat1):
        scale = kpc_scale_erin(cosmo, obj1[z_col])
        ang_sep = angular_distance(obj1[ra_col], obj1[dec_col],
                                   cat2[ra_col], cat2[dec_col]) * scale

        if include:
            num_pair.append(np.sum(ang_sep < r_kpc) - 1)
        else:
            num_pair.append(np.sum(ang_sep < r_kpc))

        index_pair.append(np.where(ang_sep < r_kpc))

    return np.asarray(num_pair), index_pair


def filter_healpix_mask(mask, catalog, ra='ra', dec='dec', verbose=True):
    """Filter a catalog through a Healpix mask.

    Parameters
    ----------
    mask : healpy mask data
        healpy mask data
    catalog : numpy array or astropy.table
        Catalog that includes the coordinate information
    ra : string
        Name of the column for R.A.
    dec : string
        Name of the column for Dec.
    verbose : boolen, optional
        Default: True

    Return
    ------
        Selected objects that are covered by the mask.
    """
    import healpy

    nside, hp_indices = healpy.get_nside(mask), np.where(mask)[0]

    phi, theta = np.radians(catalog[ra]), np.radians(90. - catalog[dec])

    hp_masked = healpy.ang2pix(nside, theta, phi, nest=True)

    select = np.in1d(hp_masked, hp_indices)

    if verbose:
        print("# %d/%d objects are selected by the mask" % (select.sum(), len(catalog)))

    return catalog[select]


def smatch_catalog(table1, table2, rmatch, index='index',
                   ra1='ra', dec1='dec', ra2='ra', dec2='dec',
                   nside=4096, maxmatch=1, join_type='left',
                   filled=True, verbose=True):
    """Match two catalogs using smatch."""

    # Perform the match using smatch
    matches = smatch.match(
        table1[ra1], table1[dec1], rmatch, table2[ra2], table2[dec2],
        nside=nside, maxmatch=maxmatch
    )

    if verbose:
        print("# Find %d matches !" % len(matches))

    if len(matches) > 1:
        # Add an index column
        cat2 = copy.deepcopy(table2)
        cat2.add_column(Column(data=np.full(len(table2), -999, dtype=np.int64), name=index))
        cat2.add_column(Column(data=np.full(len(table2), 0.0), name='cosdist'))

        cat2[index][matches['i2']] = table1[index][matches['i1']].data
        cat2['cosdist'][matches['i2']] = matches['cosdist']

        # Join two tables
        table_output = join(table1, cat2, keys=index, join_type=join_type)

        # Only keep the unique
        table_output.sort('cosdist')
        table_output.reverse()
        table_output = unique(table_output, keys=index, silent=True, keep='first')
        table_output.remove_column('cosdist')

        if filled:
            return table_output.filled()

        return table_output

    return None


def smatch_catalog_by_field(table1, table2, rmatch, field='field', **kwargs):
    """Use smatch to cross-match two catalogs."""
    list_matched = []

    for fd in np.unique(table1[field]):
        # Select objects in each small field
        flag = table1[field] == fd

        matches = smatch_catalog(
            table1[flag], table2, rmatch, **kwargs
        )

        if matches is not None:
            list_matched.append(matches)

    return vstack(list_matched)


def convert_bytes_to_int(table, column, length=19, fill_str='-99999', int64=False):
    """Convert the long ID in Bytes dtype to int."""
    id_bytes = table[column]

    table.remove_column(column)

    if int64:
        id_int = np.asarray(
            [np.int64(
                str(item).replace('\'', '') .replace('b','').replace(' ' * length, fill_str))
             for item in id_bytes.data])
    else:
        id_int = np.asarray(
            [int(
                str(item).replace('\'', '') .replace('b','').replace(' ' * length, fill_str))
             for item in id_bytes.data])

    table.add_column(Column(data=id_int, name=column))

    return table


def convert_bytes_to_str(table, column, length=9, fill_str=''):
    """Convert the long ID in Bytes dtype to string."""
    id_bytes = table[column]

    table.remove_column(column)

    id_str = np.asarray(
        [str(item).replace('\'', '') .replace('b','').replace(' ' * length, fill_str)
         for item in id_bytes.data])

    table.add_column(Column(data=id_str, name=column))

    return table
