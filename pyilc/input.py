from __future__ import print_function
import numpy as np
import yaml
import os
import healpy as hp
"""
module to read in relevant input specified by user
"""
##########################
# wavelet types implemented thus far
# Fiona edit: add TopHatHarmonic wavelets (to do a HILC)
WV_TYPES = ['GaussianNeedlets','TopHatHarmonic']
##########################

##########################
# bandpass types -- either delta functions or actual bandpasses
BP_TYPES = ['DeltaBandpasses','ActualBandpasses']
##########################

##########################
# beam types -- either symmetric gaussians or 1D ell-dependent profiles
BEAM_TYPES = ['Gaussians','1DBeams']
##########################

##########################
# component types implemented thus far
COMP_TYPES = ['CMB','kSZ','tSZ','rSZ','mu','CIB', 'CIB_dbeta'] # Fiona edit: added CIB first moment CIB_dbeta
##########################

##########################
# SED parameter types implemented thus far (that can be varied)
PARAM_TYPES = ['kT_e_keV','beta_CIB','Tdust_CIB']
##########################

##########################
# prior types on SED parameters
PRIOR_TYPES = ['Delta','Gaussian','TopHat']
##########################

##########################
### DEFAULT INPUT FILE ###
# modify this if you want to use your own
# or you can specify it when constructing ILCInfo
default_path = '../input/'
default_input = 'pyilc_input.yml'
##########################

##########################
# simple function for opening the file
def read_dict_from_yaml(yaml_file):
    assert(yaml_file != None)
    with open(yaml_file) as f:
        config = yaml.safe_load(f)
    return config
##########################

##########################
"""
class that contains map info (and associated data), ILC specifications, etc., and handles input
"""
class ILCInfo(object):
    def __init__(self, input_file=None):
        self.input_file = input_file
        if (self.input_file is None):
            # default case
            #fpath = os.path.dirname(__file__)
            self.input_file = default_path+default_input
        else:
            pass
        p = read_dict_from_yaml(self.input_file)
        # output file directory
        self.output_dir = p['output_dir']
        assert type(self.output_dir) is str, "TypeError: output_dir"
        # prefix for output file names
        self.output_prefix = p['output_prefix']
        assert type(self.output_prefix) is str, "TypeError: output_prefix"
        # Fiona edit: add output_suffix. This is only appended to the weights and the maps, not the covmat and invcovmat (and waveletized frequency maps), 
        # so you can (eg) perform different deprojections with different SED specs, while using the same covmat and invcovmat
        if 'output_suffix' in p.keys():
            self.output_suffix = p['output_suffix']
            assert type(self.output_suffix) is str, "TypeError: output_suffix"
        # flag whether to save maps of the ILC weights (if 'yes' then they will be saved; otherwise not)
        self.save_weights = p['save_weights']
        assert type(self.save_weights) is str, "TypeError: save_weights"
        # maximum multipole for this analysis
        self.ELLMAX = p['ELLMAX']
        assert type(self.ELLMAX) is int and self.ELLMAX > 0, "ELLMAX"
        # type of wavelets to use -- see WV_TYPES above
        self.wavelet_type = p['wavelet_type']
        assert type(self.wavelet_type) is str, "TypeError: wavelet_type"
        assert self.wavelet_type in WV_TYPES, "unsupported wavelet type"
        # number of wavelet filter scales used
        # Fiona edit: add if statement for HILC case
        if not self.wavelet_type == 'TopHatHarmonic':
            self.N_scales = p['N_scales']
            assert type(self.N_scales) is int and self.N_scales > 0, "N_scales"
        # parameters for each wavelet type
        # TODO: implement passing of these to the wavelet construction (probably in the main ILC script)
        if self.wavelet_type == 'GaussianNeedlets':
            # FWHM values defining the gaussian needlets
            self.GN_FWHM_arcmin = np.asarray(p['GN_FWHM_arcmin'])
            assert len(self.GN_FWHM_arcmin) == self.N_scales - 1, "GN_FWHM_arcmin"
            assert all(FWHM_val > 0. for FWHM_val in self.GN_FWHM_arcmin), "GN_FWHM_arcmin"
        # Fiona edit: add TopHatHarmonic wavelets (to do a HILC)
        elif self.wavelet_type == 'TopHatHarmonic':
            #  TODO: add functionality for the user to specity arbitrary ell-bins directly
            self.Delta_ell_HILC = p['BinSize']  # the bin sizes for a linearly-ell-binnedHILC
            self.ellbins = np.arange(0,self.ELLMAX+1,self.Delta_ell_HILC)
            self.N_scales = len(self.ellbins)-1
            assert type(self.N_scales) is int and self.N_scales > 0, "N_scales"
        # TODO: implement these
        #elif self.wavelet_type == 'CosineNeedlets':
        #elif self.wavelet_type == 'ScaleDiscretizedWavelets':
            # parameters defining these wavelets
            # TODO: add relevant assertions
            #self.B_param = p['B_param']
            #self.J_min = p['J_min']
        # Fiona cross-ILC implementation
        self.cross_ILC = False
        if 'cross_ILC' in p.keys():
            if p['cross_ILC'].lower() in ['true','yes','y']:
                self.cross_ILC = True
        # number of frequency maps used
        self.N_freqs = p['N_freqs']

        self.ILC_bias_tol = 0.01
        if 'ILC_bias_tol' in p.keys():
            self.ILC_bias_tol = p['ILC_bias_tol']

        assert type(self.N_freqs) is int and self.N_freqs > 0, "N_freqs"
        # delta-function bandpasses or actual bandpasses
        self.bandpass_type = p['bandpass_type']
        assert self.bandpass_type in BP_TYPES, "unsupported bandpass type"
        if self.bandpass_type == 'DeltaBandpasses':
            # delta function bandpasses: frequency values in GHz
            self.freqs_delta_ghz = p['freqs_delta_ghz']
            assert len(self.freqs_delta_ghz) == self.N_freqs, "freqs_delta_ghz"
        elif self.bandpass_type == 'ActualBandpasses':
            # actual bandpasses: list of bandpass file names, each containing two columns: [freq [GHz]] [transmission [arbitrary norm.]]
            self.freq_bp_files = p['freq_bp_files']
            assert len(self.freq_bp_files) == self.N_freqs, "freq_bp_files"
        # frequency map file names
        self.freq_map_files = p['freq_map_files']
        assert len(self.freq_map_files) == self.N_freqs, "freq_map_files"
        # Fiona cross-ILC implementation
        if self.cross_ILC:
            self.freq_map_files_s1 = p['freq_map_files_s1']
            assert len(self.freq_map_files_s1) == self.N_freqs, "freq_map_files_s1"
            self.freq_map_files_s2 = p['freq_map_files_s2']
            assert len(self.freq_map_files_s2) == self.N_freqs, "freq_map_files_s2"
        # beams: symmetric gaussians or 1D ell-dependent profiles
        self.beam_type = p['beam_type']
        assert self.beam_type in BEAM_TYPES, "unsupported beam type"
        if self.beam_type == 'Gaussians':
            # symmetric gaussian beams: FWHM values in arcmin
            self.beam_FWHM_arcmin = np.asarray(p['beam_FWHM_arcmin'])
            assert len(self.beam_FWHM_arcmin) == self.N_freqs, "beam_FWHM_arcmin"
            assert all(FWHM_val > 0. for FWHM_val in self.beam_FWHM_arcmin), "beam_FWHM_arcmin"
            # FWHM assumed to be in strictly decreasing order
            if ( any( i < j for i, j in zip(self.beam_FWHM_arcmin, self.beam_FWHM_arcmin[1:]))):
                raise AssertionError
        elif self.beam_type == '1DBeams':
            # symmetric 1D beams with arbitrary profiles: list of beam file names, each containing two columns: [ell] [b_ell (norm. to 1 at ell=0)]
            self.beam_files = p['beam_files']
            assert len(self.beam_files) == self.N_freqs, "beam_files"
            print("Note: frequency maps are assumed to be in strictly decreasing beam size ordering!")
        # Fiona edit: allow for performing ILC at a user-specified beam / resolution 
        self.perform_ILC_at_beam = None
        if 'perform_ILC_at_beam' in p.keys():
            #perform_ILC_at_beam should be in arcmin. If perform_ILC_at_beam is unspecified, ILC will be performed at resolution of the highest-resolution map
            self.perform_ILC_at_beam = p['perform_ILC_at_beam']  
        # N_side value of highest-resolution input map (and presumed output map N_side)
        # be conservative and assume N_side must be a power of 2 (stricly speaking only necessary for nest-ordering)
        # https://healpy.readthedocs.io/en/latest/generated/healpy.pixelfunc.isnsideok.html
        self.N_side = p['N_side']
        assert hp.pixelfunc.isnsideok(self.N_side, nest=True), "invalid N_side"
        self.N_pix = 12*self.N_side**2
        # ILC: component to preserve
        self.ILC_preserved_comp = p['ILC_preserved_comp']
        assert self.ILC_preserved_comp in COMP_TYPES, "unsupported component type in ILC_preserved_comp"
        # ILC: component(s) to deproject (if any)
        self.N_deproj = p['N_deproj']
        assert type(self.N_deproj) is int and self.N_deproj >= 0, "N_deproj"
        if (self.N_deproj > 0):
            self.ILC_deproj_comps = p['ILC_deproj_comps']
            assert len(self.ILC_deproj_comps) == self.N_deproj, "ILC_deproj_comps"
            assert all(comp in COMP_TYPES for comp in self.ILC_deproj_comps), "unsupported component type in ILC_deproj_comps"
            assert((self.N_deproj + 1) <= self.N_freqs), "not enough frequency channels to deproject this many components"
        ####################
        ### TODO: this block of code with SED parameters, etc is currently not used anywhere
        ###   instead, we currently just get the SED parameter info from fg_SEDs_default_params.yml
        ###   if we wanted to do something fancy like sample over SED parameters, we would want to make use of this code
        # ILC: SED parameters
        self.N_SED_params = p['N_SED_params']
        assert type(self.N_SED_params) is int and self.N_SED_params >= 0, "N_SED_params"
        if (self.N_SED_params > 0):
            #TODO: implement checks that only SED parameters are called here for components that are being explicitly deprojected
            #TODO: more generally, implement some way of associating the parameters with the components
            self.SED_params = p['SED_params']
            assert len(self.SED_params) == self.N_SED_params, "SED_params"
            assert all(param in PARAM_TYPES for param in self.SED_params), "unsupported parameter type in SED_params"
            # get fiducial values (which are also taken to be centers of priors)
            self.SED_params_vals = np.asarray(p['SED_params_vals'])
            assert len(self.SED_params_vals) == self.N_SED_params, "SED_params_vals"
            # get prior ranges (Delta = don't vary)
            self.SED_params_priors = p['SED_params_priors']
            assert len(self.SED_params_priors) == self.N_SED_params, "SED_params_priors"
            assert all(prior in PRIOR_TYPES for prior in self.SED_params_priors), "unsupported prior type in SED_params_priors"
            # Delta -> parameter has no meaning
            # Gaussian -> parameter is std dev
            # TopHat -> parameter is width
            self.SED_params_priors_params = np.asarray(p['SED_params_priors_params'])
            assert len(self.SED_params_priors_params) == self.N_SED_params, "SED_params_priors_params"
        ####################
        ####################
        # TODO: cross-correlation not yet implemented (not hard to do)
        # file names of maps with which to cross-correlate
        self.N_maps_xcorr = p['N_maps_xcorr']
        assert type(self.N_maps_xcorr) is int and self.N_maps_xcorr >= 0, "N_maps_xcorr"
        if (self.N_maps_xcorr > 0):
            self.maps_xcorr_files = p['maps_xcorr_files']
            assert len(self.maps_xcorr_files) == self.N_maps_xcorr, "maps_xcorr_files"
            # file names of masks to use in each cross-correlation
            # masks should be pre-apodized
            self.masks_xcorr_files = p['masks_xcorr_files']
            if self.masks_xcorr_files is not None: #None = no mask to be applied
                assert len(self.masks_xcorr_files) == self.N_maps_xcorr, "masks_xcorr_files"
        ####################

    # method for reading in maps
    def read_maps(self):
        self.maps = np.zeros((self.N_freqs,self.N_pix), dtype=np.float64)
        for i in range(self.N_freqs):
            # TODO: allow reading in of maps not in field=0 in the fits file
            # TODO: allow specification of nested or ring ordering (although will already work here if fits keyword ORDERING is present)
            temp_map = hp.fitsfunc.read_map(self.freq_map_files[i], field=0, verbose=False)
            assert len(temp_map) <= self.N_pix, "input map at higher resolution than specified N_side"
            if (len(temp_map) == self.N_pix):
                self.maps[i] = np.copy(temp_map)
            elif (len(temp_map) < self.N_pix):
                # TODO: should probably upgrade in harmonic space to get pixel window correct
                self.maps[i] = np.copy( hp.pixelfunc.ud_grade(temp_map, nside_out=self.N_side, order_out='RING', dtype=np.float64) )
        # Fiona cross-ILC implementation
        if self.cross_ILC:
            self.maps_s1 = np.zeros((self.N_freqs,self.N_pix), dtype=np.float64)
            self.maps_s2 = np.zeros((self.N_freqs,self.N_pix), dtype=np.float64)
            for i in range(self.N_freqs):
                # TODO: allow reading in of maps not in field=0 in the fits file
                # TODO: allow specification of nested or ring ordering (although will already work here if fits keyword ORDERING is present)
                temp_map_s1 = hp.fitsfunc.read_map(self.freq_map_files_s1[i], field=0, verbose=False)
                assert len(temp_map_s1) <= self.N_pix, "input map at higher resolution than specified N_side"
                temp_map_s2 = hp.fitsfunc.read_map(self.freq_map_files_s2[i], field=0, verbose=False)
                assert len(temp_map_s2) <= self.N_pix, "input map at higher resolution than specified N_side"
                if (len(temp_map_s1) == self.N_pix):
                    self.maps_s1[i] = np.copy(temp_map_s1)
                elif (len(temp_map_s1) < self.N_pix):
                    # TODO: should probably upgrade in harmonic space to get pixel window correct
                    self.maps_s1[i] = np.copy( hp.pixelfunc.ud_grade(temp_map_s1, nside_out=self.N_side, order_out='RING', dtype=np.float64) )
                if (len(temp_map_s2) == self.N_pix):
                    self.maps_s2[i] = np.copy(temp_map_s2)
                elif (len(temp_map_s2) < self.N_pix):
                    # TODO: should probably upgrade in harmonic space to get pixel window correct
                    self.maps_s2[i] = np.copy( hp.pixelfunc.ud_grade(temp_map_s2, nside_out=self.N_side, order_out='RING', dtype=np.float64) )
            del(temp_map_s1)
            del(temp_map_s2)
        # also read in maps with which to cross-correlate, if specified
        if self.N_maps_xcorr != 0:
            # maps
            self.maps_xcorr = np.zeros((self.N_maps_xcorr,self.N_pix), dtype=np.float64)
            for i in range(self.N_maps_xcorr):
                temp_map = hp.fitsfunc.read_map(self.maps_xcorr_files[i], field=0, verbose=False)
                assert len(temp_map) <= self.N_pix, "input map for cross-correlation at higher resolution than specified N_side"
                if (len(temp_map) == self.N_pix):
                    self.maps_xcorr[i] = np.copy(temp_map)
                elif (len(temp_map) < self.N_pix):
                    # TODO: should probably upgrade in harmonic space to get pixel window correct
                    self.maps_xcorr[i] = np.copy( hp.pixelfunc.ud_grade(temp_map, nside_out=self.N_side, order_out='RING', dtype=np.float64) )
            # masks
            if self.masks_xcorr_files is not None: #None = no mask to be applied
                self.masks_xcorr = np.zeros((self.N_maps_xcorr,self.N_pix), dtype=np.float64)
                for i in range(self.N_maps_xcorr):
                    temp_map = hp.fitsfunc.read_map(self.masks_xcorr_files[i], field=0, verbose=False)
                    assert len(temp_map) <= self.N_pix, "input mask for cross-correlation at higher resolution than specified N_side"
                    if (len(temp_map) == self.N_pix):
                        self.masks_xcorr[i] = np.copy(temp_map)
                    elif (len(temp_map) < self.N_pix):
                        self.masks_xcorr[i] = np.copy( hp.pixelfunc.ud_grade(temp_map, nside_out=self.N_side, order_out='RING', dtype=np.float64) )
            else: #no mask
                self.masks_xcorr = np.ones((self.N_maps_xcorr,self.N_pix), dtype=np.float64)

    # method for reading in bandpasses
    # self.bandpasses is a list of length self.N_freqs where each entry is an N x 2 array where N can be different for each frequency channel
    def read_bandpasses(self):
        if self.bandpass_type == 'ActualBandpasses':
            self.bandpasses = [] #initialize empty list
            for i in range(self.N_freqs):
                (self.bandpasses).append(np.loadtxt(self.freq_bp_files[i], unpack=True, usecols=(0,1)))

    # method for reading in beams
    # self.beams is a list of length self.N_freqs where each entry is an (ELLMAX+1) x 2 array
    def read_beams(self):
        if self.beam_type == 'Gaussians':
            self.beams = np.zeros((self.N_freqs,self.ELLMAX+1,2), dtype=np.float64)
            for i in range(self.N_freqs):
                self.beams[i] = np.transpose(np.array([np.arange(self.ELLMAX+1), hp.sphtfunc.gauss_beam(self.beam_FWHM_arcmin[i]*(np.pi/180.0/60.0), lmax=self.ELLMAX)]))
                # Fiona edit: allow for performing ILC at a user-specified beam / resolution
                #we will convolve all maps to the common_beam
                if self.perform_ILC_at_beam is not None:
                    self.common_beam =np.transpose(np.array([np.arange(self.ELLMAX+1), hp.sphtfunc.gauss_beam(self.perform_ILC_at_beam*(np.pi/180.0/60.0), lmax=self.ELLMAX)]))
                else:
                    self.common_beam = self.beams[-1] # if perform_ILC_at_beam is unspecified, convolve to the beam of the highest-resolution map
        elif self.beam_type == '1DBeams':
            self.beams = [] #initialize empty list
            for i in range(self.N_freqs):
                (self.beams).append(np.loadtxt(self.beam_files[i], unpack=True, usecols=(0,1)))
                # check that beam profiles start at ell=0 and extend to self.ELLMAX or beyond
                assert (self.beams)[i][0][0] == 0, "beam profiles must start at ell=0"
                assert (self.beams)[i][-1][0] >= self.ELLMAX, "beam profiles must extend to ELLMAX or higher"
                if ((self.beams)[i][-1][0] > self.ELLMAX):
                    (self.beams)[i] = (self.beams)[i][0:ELLMAX+1]
                assert (len((self.beams)[i]) == ELLMAX+1), "beam profiles must contain all integer ells up to ELLMAX"
