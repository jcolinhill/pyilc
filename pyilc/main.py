from __future__ import print_function
import sys
import numpy as np
import os
import healpy as hp
from input import ILCInfo
from wavelets import Wavelets, wavelet_ILC
"""
main script for doing wavelet (MC)^2ILC analysis
"""
##########################
# main input file
### input file containing most specifications ###
try:
    input_file = (sys.argv)[1]
except IndexError:
    input_file = '../input/pyilc_input_Kristen_example.yml'
##########################

##########################
# read in the input file and set up relevant info object
info = ILCInfo(input_file)
##########################

##########################
# read in frequency maps
info.read_maps()
# read in bandpasses
info.read_bandpasses()
# read in beams
info.read_beams()
##########################

##########################
# construct wavelets
wv = Wavelets(N_scales=info.N_scales, ELLMAX=info.ELLMAX, tol=1.e-6)
if info.wavelet_type == 'GaussianNeedlets':
    ell, filts = wv.GaussianNeedlets(FWHM_arcmin=info.GN_FWHM_arcmin)
# Fiona HILC implementation
elif info.wavelet_type == 'TopHatHarmonic':
    ell,filts = wv.TopHatHarmonic(info.ellbins)
else:
    raise TypeError('unsupported wavelet type')
# example plot -- output in example_wavelet_plot
#wv.plot_wavelets(log_or_lin='lin')
##########################

##########################
# wavelet ILC
# Fiona HILC implementation:
if info.wavelet_type == 'TopHatHarmonic':
    harmonic_ILC(wv, info, ILC_bias_tol=1.e-2, wavelet_beam_criterion=1.e-3, resp_tol=1.e-3, map_images=True)
else:
    wavelet_ILC(wv, info, ILC_bias_tol=1.e-2, wavelet_beam_criterion=1.e-3, resp_tol=1.e-3, map_images=True)
##########################


##########################
# TODO
# add simple code to cross-correlate the output ILC map with input map specified by the user (useful for simulations/validation)
##########################
