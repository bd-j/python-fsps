# -*- coding: utf-8 -*-

from __future__ import division, print_function

import json
import os
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

# As far as we can tell, Conroy & Gunn (2010) Fig.2 uses Vega magnitudes
pop = StellarPopulation(compute_vega_mags=True, zcontinuous=1)
default_params = dict([(k, pop.params[k]) for k in pop.params.all_params])
libs = pop.libraries

def _reset_default_params():
    for k in pop.params.all_params:
        pop.params[k] = default_params[k]


def compare_reference_iso(ref_in, current):
    """This method compares the values in a previously generated HDF5 file to
    the current values, and reports the maximum change in color between the
    input and the current values.
    We assume that the reference file contains J,H,K,I,V,B magnitudes computed
    at the same log(ages) as the ones used by the current model.
    :param ref_in:
        h5py.File object with keys matching those in the given dictionary of values.
    :param current:
        Dictionary of calculated values that you want to compare to the old
        version. These correspond to the J,H,K,I,V,B magnitudes.
        
    Beware: the active isochrone model and the input value isochrone model
    must be the same for any useful comparison!
    """
    ref_libs = json.loads(ref_in.attrs['libraries'])
    for i in range(2):
        assert ref_libs[i] == libs[i]

    J1 = np.asarray(ref_in.get('d0/J'))
    H1 = np.asarray(ref_in.get('d0/H'))
    K1 = np.asarray(ref_in.get('d0/K'))
    I1 = np.asarray(ref_in.get('d0/I'))
    V1 = np.asarray(ref_in.get('d0/V'))
    B1 = np.asarray(ref_in.get('d0/B'))
    Ages1 = np.asarray(ref_in.get('t0/Ages'))
    
    Jdiff = current['J'] - J1
    Hdiff = current['H'] - H1
    Kdiff = current['K'] - K1
    Idiff = current['I'] - I1
    Vdiff = current['V'] - V1
    Bdiff = current['B'] - B1
    
    BVdiff = (current['B']-current['V']) - (B1-V1)
    VIdiff = (current['V']-current['I']) - (V1-I1)
    VKdiff = (current['V']-current['K']) - (V1-K1)
    JHdiff = (current['J']-current['H']) - (J1-H1)
    JKdiff = (current['J']-current['K']) - (J1-K1)
    HKdiff = (current['H']-current['K']) - (H1-K1)

    print('max B-V difference (input vs. current) = ' + str(np.max(np.abs(BVdiff))))
    print('max V-I difference (input vs. current) = ' + str(np.max(np.abs(VIdiff))))
    print('max V-K difference (input vs. current) = ' + str(np.max(np.abs(VKdiff))))
    print('max J-H difference (input vs. current) = ' + str(np.max(np.abs(JHdiff))))
    print('max J-K difference (input vs. current) = ' + str(np.max(np.abs(JKdiff))))
    print('max H-K difference (input vs. current) = ' + str(np.max(np.abs(HKdiff))))
    

        
def test_ssp_iso_color(reference_in=None, reference_out=None):
    """This method will generate a plot of SSP colors as a function of age for
    different isochrone models, reproducing Conroy & Gunn (2010) Figure 2.
    We use the MIST, BaSTI, and Padova (the one currently implemented in 
    python-fsps, not the modified version referenced in Conroy & Gunn (2010))
    isochrone models. 
    NOTE: Because changing between different isochrone models requires
    recompiling FSPS and re-installing python-fsps, we wrote this method to
    compute SSP colors for only ONE isochrone model (the active model). The 
    other two models (the passive models) are provided from a benchmark run on 
    12/06/2016.
    The current version of this method assumes that the active model is MIST 
    and the passive models are BaSTI and Padova. All isochrone models assume 
    that the spectral library is BaSEL. 
    NOTE: We leave it to the user to change FSPS libraries to match our active
    model assumption or to change the active model to suit their needs. Since
    reference_out outputs only the information from the active model, name 
    your file appropriately!
    
    :param reference_in: (optional)
        The name (and path) to a file to use as a reference for the quantities
        being calculated.  The currently calculated quantities will be compared
        to the ones in this file.
    :param reference_out: (optional)
        Name of hdf5 file to output containing the calculated quantities
    :returns figs:
        List of matplotlib.pyplot figure objects.
    """
    _reset_default_params()
    
    # If you want to use a different active model, please change the variable
    # active_model (in addition to changing the libraries inside FSPS and
    # python-fsps). Current options are 'MIST', 'Padova', and 'BaSTI'.
    active_model = 'MIST'

    # Here we will compute SSP J,H,K,I,V,B magnitudes for different ages (and
    # metallicities)
    # Conroy & Gunn (2010) only provide metallicities at 7 ages, so we will do
    # some interpolation - before 1 Gyr, metallicity is constant; after 1 Gyr,
    # we assume do a linear interpolation
    met_inter = np.interp(np.linspace(9.1, 10.1, num=21), [9.0, 9.5, 9.8, 10.1], [-0.28, -0.38, -0.68, -1.5])
    # Full ages and metallicity arrays here
    logage_act = np.concatenate((np.linspace(7.5, 9.05, num=32), np.linspace(9.1, 10.1, num=21)))
    met = np.concatenate((np.ones(32)*(-0.28), met_inter))
    # Magnitude bands
    bands = ['b','v','cousins_i','2mass_j','2mass_ks','2mass_h']
    # Initialize some arrays to store the different bands
    size = len(logage_act)
    B_act = np.empty(size)
    V_act = np.empty(size)
    I_act = np.empty(size)
    J_act = np.empty(size)
    K_act = np.empty(size)
    H_act = np.empty(size)
    # Now compute the magnitudes
    print("Computing magnitudes for {} isochrones. This might take a while!".format(active_model))
    for i in range(size):
        pop.params['logzsol'] = met[i]
        mags = pop.get_mags(tage=10**(logage_act[i]-9), bands=bands)
        B_act[i] = mags[0]
        V_act[i] = mags[1]
        I_act[i] = mags[2]
        J_act[i] = mags[3]
        K_act[i] = mags[4]
        H_act[i] = mags[5]

    # Make figures and an output structure
    # The output structure 
    fig, axarr = pl.subplots(3,2,figsize=(3.2*2*1.5*0.9, 2.8*3*1.5*0.8))
    # Colors and linestyles
    models = {'MIST': {'color':'b', 'linestyle':'-', 'dashes':[8, 4, 2, 4]}, 
    'Padova': {'color':'k', 'linestyle':'-'}, 
    'BaSTI':{'color':'r', 'linestyle':'--'}}
    # Get the passive models data from our benchmark file
    dir = os.path.dirname(__file__)
    bench_path = os.path.join(dir, 'benchmark_iso_color.hdf5')
    f = h5py.File(bench_path, 'r')
    # Plot all the things!
    print('Generating plots')
    for model in models.keys():
        if model != active_model:
            # get data for passive models
            J = np.asarray(f.get(model+'/d0/J'))
            H = np.asarray(f.get(model+'/d0/H'))
            K = np.asarray(f.get(model+'/d0/K'))
            I = np.asarray(f.get(model+'/d0/I'))
            V = np.asarray(f.get(model+'/d0/V'))
            B = np.asarray(f.get(model+'/d0/B'))
            logage = np.asarray(f.get(model+'/t0/Ages'))
        else:
            # get data for active model
            J = J_act
            H = H_act
            K = K_act
            I = I_act
            V = V_act
            B = B_act
            logage = logage_act
        # plotting
        axarr[0,0].plot(logage, B-V, lw=2, **models[model])
        axarr[0,0].set_ylabel(r'${\rm B-V}$', fontsize=13)
        axarr[0,1].plot(logage, V-I, label='FSPS + '+model, lw=2, \
            **models[model])
        axarr[0,1].set_ylabel(r'${\rm V-I}$', fontsize=13)
        axarr[1,0].plot(logage, V-K, lw=2, **models[model])
        axarr[1,0].set_ylabel(r'${\rm V-K}$', fontsize=13)
        axarr[1,1].plot(logage, J-H, lw=2, **models[model])
        axarr[1,1].set_ylabel(r'${\rm J-H}$', fontsize=13)
        axarr[2,0].plot(logage, J-K, lw=2, **models[model])
        axarr[2,0].set_ylabel(r'${\rm J-K}$', fontsize=13)
        axarr[2,1].plot(logage, H-K, lw=2, **models[model])
        axarr[2,1].set_ylabel(r'${\rm H-K}$', fontsize=13)
        
    #Plot beautification here
    axarr[0,0].minorticks_on()
    axarr[0,0].set_xlim(left=7.5,right=10.3)
    axarr[0,0].set_ylim(top=1.0, bottom=0.0)
    axarr[0,1].minorticks_on()
    axarr[0,1].set_xlim(left=7.5,right=10.3)
    axarr[0,1].set_ylim(top=1.41, bottom=0.3)
    axarr[1,0].minorticks_on()
    axarr[1,0].set_xlim(left=7.5,right=10.3)
    axarr[1,0].set_ylim(top=3.51,bottom=1.01)
    axarr[1,1].minorticks_on()
    axarr[1,1].set_xlim(left=7.5,right=10.3)
    axarr[1,1].set_ylim(top=0.91,bottom=0.35)
    axarr[2,0].minorticks_on()
    axarr[2,0].set_xlim(left=7.5,right=10.3)
    axarr[2,0].set_ylim(top=1.3, bottom=0.39)
    axarr[2,1].minorticks_on()
    axarr[2,1].set_xlim(left=7.5,right=10.3)
    axarr[2,1].set_ylim(top=0.501, bottom=0.0)
    for i in range(2):
        axarr[2,i].set_xlabel(r'${\rm log(Age \; [yrs])}$', fontsize=13)
    leg = axarr[0,1].legend(loc=2, frameon=False)
    ltext  = leg.get_texts()
    pl.setp(ltext, fontname='Times New Roman')
    pl.tight_layout()
       
    current = {'B': B_act,
               'V': V_act,
               'I': I_act,
               'J': J_act,
               'K': K_act,
               'H': H_act,
               'Ages': logage_act}

    # Here we compare to an old set of values stored in a benchmark hdf5 file
    # Please make sure that the log(ages) at which you compute the colors
    # match our log(ages) array, or else you'll get a lot of errors
    # We also assume that the input file has data for the magnitudes in the
    # different bands
    if reference_in is not None:
        ref = h5py.File(reference_in, 'r')
        diffstring = compare_reference_iso(ref, current)
        ref.close()

    # Here we write out an hdf5 file
    if reference_out is not None:
        import datetime
        timestamp = 'T'.join(str(datetime.datetime.now()).split())
        ref = h5py.File(reference_out, 'w')
        # write all the data!
        ref['d0/J'] = current['J']
        ref['d0/H'] = current['H']
        ref['d0/K'] = current['K']
        ref['d0/I'] = current['I']
        ref['d0/V'] = current['V']
        ref['d0/B'] = current['B']
        ref['d0'].attrs['units'] = 'mags'
        ref['d0'].attrs['long_name'] = 'Filters J,H,K,I,V,B' 
        for key in ref['d0'].keys():
            ref['d0/%s'%key].attrs['units'] = 'mags'
            ref['d0/%s'%key].attrs['long_name'] = key+'-band magnitude'

        ref['t0/Ages'] = current['Ages']
        ref['t0'].attrs['units'] = 'Log Age (yrs)'
        ref['t0'].attrs['long_name'] = 'Log Age of Population in yrs' 

        # now we add useful version things
        ref.attrs['default'] = 'entry'                        
        ref.attrs['file_name'] = reference_out
        ref.attrs['timestamp'] = timestamp
        ref.attrs['HDF5_Version'] = h5py.version.hdf5_version
        ref.attrs['h5py_Version'] = h5py.version.version
        ref.attrs['libraries'] = json.dumps(libs)
        ref.attrs['default_params'] = json.dumps(default_params)
        ref.attrs['git_history'] = json.dumps(githashes)
        ref.close()

    return [fig]
