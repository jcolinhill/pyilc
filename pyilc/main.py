from __future__ import print_function
import sys
import numpy as np
import os
import healpy as hp
from input import ILCInfo
import time
from wavelets import Wavelets, wavelet_ILC, harmonic_ILC
"""
main script for doing needlet or harmonic (multiply constrained) ILC analysis
"""
##########################
# main input file
### input file containing most specifications ###
tstart = time.time()
try:
    input_file = (sys.argv)[1]
except IndexError:
    input_file = '../input/pyilc_input_example_general.yml'
##########################

##########################
# read in the input file and set up relevant info object
info = ILCInfo(input_file)
##########################
t2=time.time()
##########################
# read in frequency maps
#info.read_maps() # fiona: we only need to do this if we are computing wavelet coeffs, if they haven't already been computed and saved
# read in bandpasses
info.read_bandpasses()
# read in beams
info.read_beams()
##########################

##########################
# construct wavelets
wv = Wavelets(N_scales=info.N_scales, ELLMAX=info.ELLMAX, tol=1.e-6, taper_width=info.taper_width)
if info.wavelet_type == 'GaussianNeedlets':
    ell, filts = wv.GaussianNeedlets(FWHM_arcmin=info.GN_FWHM_arcmin)
elif info.wavelet_type == 'CosineNeedlets': # Fiona added CosineNeedlets
    ell,filts = wv.CosineNeedlets(ellmin = info.ellmin,ellpeaks = info.ellpeaks)
elif info.wavelet_type == 'ScaleDiscretizedWavelets':
    ell,filts = wv.ScaleDiscretizedWavelets(ellboundaries = info.ellboundaries)
elif info.wavelet_type == 'TopHatHarmonic':
    ell,filts = wv.TopHatHarmonic(info.ellbins)
else:
    raise TypeError('unsupported wavelet type')
# example plot -- output in example_wavelet_plot
#wv.plot_wavelets(log_or_lin='lin')
##########################

##########################
# wavelet ILC
if info.wavelet_type == 'TopHatHarmonic':
    info.maps2alms()
    info.alms2cls()
    harmonic_ILC(wv, info, resp_tol=info.resp_tol, map_images=False)
else:
    wavelet_ILC(wv, info, resp_tol=info.resp_tol, map_images=False)
##########################
tend = time.time()-tstart
print("tend is",tend)

##########################
# TODO
# add simple code to cross-correlate the output ILC map with input map specified by the user (useful for simulations/validation)
##########################
