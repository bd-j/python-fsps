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
        version. These correspond to the J,H,K,I,V,B magnitudes and the ages
        at which the magnitudes are computed.
        
    Beware: the active isochrone model and the input value isochrone model must be the same for any useful comparison!
    """
    ref_libs = json.loads(ref_in.attrs['libraries'])
    for i in range(2):
        assert ref_libs[i] == libs[i]

    # create empty arrays to store data
    data_in = np.zeros((len(ref_in['d0'].keys())+1, \
        len(np.asarray(ref_in.get('t0/Ages')))))
    data_c = np.zeros((len(current.keys())+1, \
        len(current['Ages'])))

    # populate with ages
    data_in[0,:] = np.asarray(ref_in.get('t0/Ages'))
    data_c[0,:] = current['Ages']
    
    # sort keys
    sorted_keys = current.keys()
    sorted_keys.remove('Ages')
    sorted_keys.sort()
    # populate with magnitudes
    for i, key in enumerate(sorted_keys):
        data_in[i+1,:] = np.asarray(ref_in.get('d0/%s'%key))
        data_c[i+1,:] = current[key]

    # compute color difference between current and input
    BVdiff = (data_c[1,:]-data_c[6,:]) - (data_in[1,:]-data_in[6,:])
    VIdiff = (data_c[6,:]-data_c[3,:]) - (data_in[6,:]-data_in[3,:])
    VKdiff = (data_c[6,:]-data_c[5,:]) - (data_in[6,:]-data_in[5,:])
    JHdiff = (data_c[4,:]-data_c[2,:]) - (data_in[4,:]-data_in[2,:])
    JKdiff = (data_c[4,:]-data_c[5,:]) - (data_in[4,:]-data_in[5,:])
    HKdiff = (data_c[2,:]-data_c[5,:]) - (data_in[2,:]-data_in[5,:])

    #colordiffset = np.asarray(BVdiff, VIdiff, VKdiff, JHdiff, JKdiff, HKdiff)
    
    
    print('max B-V difference (input vs. current) = ' + str(np.max(np.abs(BVdiff))))
    print('max V-I difference (input vs. current) = ' + str(np.max(np.abs(VIdiff))))
    print('max V-K difference (input vs. current) = ' + str(np.max(np.abs(VKdiff))))
    print('max J-H difference (input vs. current) = ' + str(np.max(np.abs(JHdiff))))
    print('max J-K difference (input vs. current) = ' + str(np.max(np.abs(JKdiff))))
    print('max H-K difference (input vs. current) = ' + str(np.max(np.abs(HKdiff))))
    
    #return data_in, data_c, diffset, colordiffset
        
def test_ssp_iso_color(reference_in=None, reference_out=None):
    """This method will generate a plot of SSP colors as a function of age for
    different isochrone models, reproducing Conroy & Gunn (2010) Figure 2.
    We use the MIST, BaSTI, and Padova (the one currently implemented in 
    python-fsps, not the modified version referenced in Conroy & Gunn (2010))
    isochrone models. 
    NOTE: Because changing between different isochrone models requires recompiling FSPS and re-installing python-fsps, we wrote this method to 
    compute SSP colors for only ONE isochrone model (the active model). The 
    other two models (the passive models) are provided from a benchmark run on 
    12/06/2016.
    The current version of this method assumes that the active model is MIST 
    and the passive models are BaSTI and Padova. All isochrone models assume 
    that the spectral library is BaSEL. 
    NOTE: We leave it to the user to change FSPS libraries to match our active model assumption or to change the active model to suit their needs. Since
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
    met_inter = np.interp(np.linspace(9.1, 10.1, num=21), \
        [9.0, 9.5, 9.8, 10.1], [-0.28, -0.38, -0.68, -1.5])
    # Full ages and metallicity arrays here
    logage_act = np.concatenate((np.linspace(7.5, 9.05, num=32), \
        np.linspace(9.1, 10.1, num=21)))
    met = np.concatenate((np.ones(32)*(-0.28), met_inter))
    # Magnitude bands
    bands = ['b','v','cousins_i','2mass_j','2mass_ks','2mass_h']
    # Initialize some arrays to store the different bands
    t_size = len(logage_act)
    colors_act = np.empty([len(bands), t_size])
    # Now compute the magnitudes
    print("Computing magnitudes for {} isochrones. This might take a while!".format(active_model))
    for i in range(t_size):
        pop.params['logzsol'] = met[i]
        colors_act[:,i] = pop.get_mags(tage=10**(logage_act[i]-9), bands=bands)

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
        temp_plot = np.empty([len(bands)+1, t_size])
        if model != active_model:
            # get data for passive models
            temp_plot[0,:] = np.asarray(f.get(model+'/t0/Ages'))
            for i, key in enumerate(['B', 'V', 'I', 'J', 'K', 'H']):
                temp_plot[i+1,:] = np.asarray(f.get(model+'/d0/%s'%key))
        else:
            # get data for active model
            temp_plot[0,:] = logage_act
            for i in range(len(bands)):
                temp_plot[i+1,:] = colors_act[i,:]
        # plotting
        axarr[0,0].plot(temp_plot[0,:], temp_plot[1,:]-temp_plot[2,:], lw=2, \
            **models[model])
        axarr[0,0].set_ylabel(r'${\rm B-V}$', fontsize=13)
        axarr[0,1].plot(temp_plot[0,:], temp_plot[2,:]-temp_plot[3,:], \
            label='FSPS + {}'.format(model), lw=2, **models[model])
        axarr[0,1].set_ylabel(r'${\rm V-I}$', fontsize=13)
        axarr[1,0].plot(temp_plot[0,:], temp_plot[2,:]-temp_plot[5,:], lw=2, \
            **models[model])
        axarr[1,0].set_ylabel(r'${\rm V-K}$', fontsize=13)
        axarr[1,1].plot(temp_plot[0,:], temp_plot[4,:]-temp_plot[6,:], lw=2, \
            **models[model])
        axarr[1,1].set_ylabel(r'${\rm J-H}$', fontsize=13)
        axarr[2,0].plot(temp_plot[0,:], temp_plot[4,:]-temp_plot[5,:], lw=2, \
            **models[model])
        axarr[2,0].set_ylabel(r'${\rm J-K}$', fontsize=13)
        axarr[2,1].plot(temp_plot[0,:], temp_plot[6,:]-temp_plot[5,:], lw=2, \
            **models[model])
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
       
    current = {'B': colors_act[0,:],
               'V': colors_act[1,:],
               'I': colors_act[2,:],
               'J': colors_act[3,:],
               'K': colors_act[4,:],
               'H': colors_act[5,:],
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
        for key in ['B', 'V', 'I', 'J', 'K', 'H']:
            ref['d0/'+key] = current[key]
            
        ref['d0'].attrs['units'] = 'mags'
        ref['d0'].attrs['long_name'] = 'Filters B,V,I,J,K,H' 
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
