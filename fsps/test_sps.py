# -*- coding: utf-8 -*-

from __future__ import division, print_function

import json
import numpy as np
from numpy.testing import assert_allclose
from .fsps import StellarPopulation
from .__init__ import githashes
try:
    import h5py
except(ImportError):
    pass
try:
    import matplotlib.pyplot as pl
except(ImportError):
    pass

pop = StellarPopulation(zcontinuous=1)
default_params = dict([(k, pop.params[k]) for k in pop.params.all_params])
libs = pop.libraries

def _reset_default_params():
    for k in pop.params.all_params:
        pop.params[k] = default_params[k]


def compare_reference(ref_in, current):
    """This method compares the values in a previously generated HDF5 file to
    the values in a supplied dictionary, and reports the maximum percentage
    change and where that change occurred.

    :param ref_in:
        h5py.File object with keys matching those in the given dictionary of values.

    :param current:
        Dictionary of calculated values that you want to compare to the old
        version.  Keys are the name of the quentities (e.g. maggies,
        stellar_mass, etc.)
    """
    ref_libs = json.loads(ref_in.attrs['libraries'])
    #assert ref_libs == libs

    diffstring = ("The maximum difference from the reference quantities "
                  "is {:4.5f} percent in the quantity {}")
    maxdiff, maxdq = 0, 'ANY'
    for k, v in current.items():
        diff = np.atleast_1d(v /ref_in[k]  - 1).flatten()  # percent difference
        maxdind = np.argmax(np.abs(diff))
        if np.abs(diff[maxdind]) > maxdiff:
            maxdiff = diff[maxdind]
            maxdq = k

    return diffstring.format(maxdiff, maxdq)

        
def test_sps_ssps(reference_in=None, reference_out=None):
    """This text should say what benchmark you're trying to reproduce,
    e.g. Conroy & Gunn (2009) Figure 1.  Also use an informative function name.

    :param reference_in: (optional)
        The name (and path) to a file to use as a refernce for the quantities
        being calculated.  The currently calculated quantities will be compared
        to the ones in this file.

    :param reference_out: (optional)
        Name of hdf5 file to output containing the calculated quantities

    :returns figs:
        List of matplotlib.pyplot figure objects.
    """
    _reset_default_params()

    # Do stuff to make the data you need
    # This example just gets the absolute magnitudes for SSPs
    pop.params["sfh"] = 0
    mag = pop.get_mags(tage=0, bands=["v"])
    # we make it a linear unit so that percent differences are more meaningful
    maggies = 10**(-0.4 * mags)
    masses = pop.stellar_mass

    # Make figures and an output structure
    # The output structure 
    fig, axes = pl.subplots()
    current = {'ssp_maggies': maggies,
               'stellar_masses': masses}

    # Here we compare to an old set of values stored in an hdf5 file
    if reference_in is not None:
        ref = h5py.File(reference_in, 'r')
        diffstring = compare_reference(ref, current)
        print(diffstring)
        ref.close()

    # Here we write out an hdf5 file
    if reference_out is not None:
        ref = h5py.File(reference_out, 'w')
        for k, v in current.items():
            d = ref.create_dataset(k, data=v)
        # now we add useful version things
        ref.attrs['libraries'] = json.dumps(libs)
        ref.attrs['default_params'] = json.dumps(default_params)
        ref.attrs['git_history'] = json.dumps(githashes)
        ref.close()

    return [fig]

def test_csp_sfh(reference_in=None, reference_out=None):
    """This function tests the SFH parameters- sfstart, sftrunc, tau,
    tburst, fburst and constant; by generating the magnitude of each
    CSP under three different filters- Galex-FUV, SDSS-r, WISE-1.  As
    a fourth comparision, the evolution of the stellar mass for each
    sfh is also shown.

    :param reference_in: (optional)
        The name (and path) to a file to use as a refernce for the quantities
        being calculated.  The currently calculated quantities will be compared
        to the ones in this file.

    :param reference_out: (optional)
        Name of hdf5 file to output containing the calculated quantities

    :returns figs:
        List of matplotlib.pyplot figure objects.

    """
    filters = ['wise_w1', 'sdss_r', 'galex_fuv']
    propname = ['WISE W1', 'SDSS R', 'GALEX FUV','MASS']
    ages = pop.ssp_ages

    prop = {};color= {}

    # 1. Constant SFH
    color['const'] = '#e41a1c'
    _reset_default_params()
    pop.params['sfh'] = 1
    pop.params['const'] = 1.0
    prop['const'] = np.concatenate((pop.get_mags(bands=filters),
                                    pop.stellar_mass.reshape(np.size(ages),1)),
                                   axis=1)
    # 2. Constant SFH between 10Myr and 10Gyr
    color['tophat'] = '#377eb8'
    _reset_default_params()
    pop.params['sfh'] = 1
    pop.params['const'] = 1.0
    pop.params['sf_start'] = 1e-3
    pop.params['sf_trunc'] = 10.0
    prop['tophat'] = np.concatenate((pop.get_mags(bands=filters),
                                     pop.stellar_mass.reshape(np.size(ages),1)),
                                    axis=1)
    # 3. Constant SFH + Burst at log-age 10.01
    color['cburst'] = '#4daf4a'
    _reset_default_params()
    pop.params['sfh'] = 1
    pop.params['const'] = 0.5
    pop.params['tburst'] = 10.01
    pop.params['fburst'] = 0.5
    prop['cburst'] = np.concatenate((pop.get_mags(bands=filters),
                                     pop.stellar_mass.reshape(np.size(ages),1)),
                                    axis=1)
    # 4. Exponential SFH with tau= 1Gyr
    color['exp'] = '#984ea3'
    _reset_default_params()
    pop.params['sfh'] = 1
    pop.params['tau'] = 1.0
    prop['exp'] = np.concatenate((pop.get_mags(bands=filters),
                                  pop.stellar_mass.reshape(np.size(ages),1)),
                                 axis=1)
    # 5. Delayed Exponential SFH with tau= 1Gyr
    color['dexp'] = '#ff7f00'
    _reset_default_params()
    pop.params['sfh'] = 4
    pop.params['tau'] = 1.0
    prop['dexp'] = np.concatenate((pop.get_mags(bands=filters),
                                   pop.stellar_mass.reshape(np.size(ages),1)),
                                  axis=1)

    # Store the magnitudes for three filters in linear units for
    # meaningful comparision later.
    current = {}
    for k,v in prop.items():
        current[k] = np.concatenate((np.power(10,prop[k][:,0:np.size(filters)]),
                                    (prop[k][:,np.size(filters)]).reshape(np.size(ages),1)),
                                    axis=1)

    nrows=2;ncols=2
    fig, ax = pl.subplots(nrows=nrows, ncols=ncols,sharex=True,figsize=(15,15))
    ax[0][0].set_ylim((25,0))
    ax[0][1].set_ylim((25,0))
    ax[1][0].set_ylim((25,0))
    ax[1][1].set_yscale('log')

    for row in range(nrows):
        for col in range(ncols):
            ax[row][col].set_ylabel(propname[(nrows*row+col)],fontsize=18)
            ax[row][col].set_xlabel('Log Age[yrs]',fontsize=18)
            for sfh in prop.keys():
                ax[row][col].plot(ages, prop[sfh][:,(nrows*row+col)],c=color[sfh],label=sfh,linewidth=2)

    ax[row][col].legend(loc=4,fontsize=18)
    
    # Here we compare to an old set of values stored in an hdf5 file
    if reference_in is not None:
        ref = h5py.File(reference_in, 'r')
        diffstring = compare_reference(ref, current)
        print(diffstring)
        ref.close()

    # Here we write out an hdf5 file
    if reference_out is not None:
        ref = h5py.File(reference_out, 'w')
        for k, v in current.items():
            d = ref.create_dataset(k, data=v)
        # now we add useful version things
        ref.attrs['libraries'] = json.dumps(libs)
        ref.attrs['default_params'] = json.dumps(default_params)
        ref.attrs['git_history'] = json.dumps(githashes)
        ref.close()

    return [fig]
