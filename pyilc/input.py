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
# WV_TYPES = ['GaussianNeedlets','TopHatHarmonic']
WV_TYPES = ['GaussianNeedlets','TopHatHarmonic','CosineNeedlets','ScaleDiscretizedWavelets'] # Fiona added CosineNeedlets
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
COMP_TYPES = ['CMB','kSZ','tSZ','rSZ','mu','CIB', 'CIB_dbeta','CIB_dT','radio','radio_dbeta','radio2']
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

        # suffix for output file names (only ILC weights and ILC maps)
        self.output_suffix = ''
        if 'output_suffix' in p.keys():
            self.output_suffix = p['output_suffix']
            assert type(self.output_suffix) is str, "TypeError: output_suffix"
        self.output_suffix_intermediate = self.output_suffix
        if 'output_suffix_intermediate' in p.keys():
            self.output_suffix_intermediate = p ['output_suffix_intermediate']
            assert type(self.output_suffix_intermediate) is str, "TypeError: output_suffix_intermediate"

        self.output_suffix_weights = self.output_suffix
        if 'output_suffix_weights' in p.keys():
            self.output_suffix_weights = p['output_suffix_weights']
            assert type(self.output_suffix_weights) is str, "TypeError: output_suffix_weights"

        # flag whether to save maps of the ILC weights (if 'yes' then they will be saved; otherwise not)
        self.save_weights = p['save_weights']
        assert type(self.save_weights) is str, "TypeError: save_weights"

        #flag whether to save the ILC map at each scale - if not in input file will default to False
        self.save_scale_ILC_maps = False
        if 'save_scale_ILC_maps' in p.keys():
            if p['save_scale_ILC_maps'].lower() in ['yes','true']:
                self.save_scale_ILC_maps = True

        # maximum multipole for this analysis
        self.ELLMAX = p['ELLMAX']
        assert type(self.ELLMAX) is int and self.ELLMAX > 0, "ELLMAX"

        # type of wavelets to use -- see WV_TYPES above
        self.wavelet_type = p['wavelet_type']
        assert type(self.wavelet_type) is str, "TypeError: wavelet_type"
        assert self.wavelet_type in WV_TYPES, "unsupported wavelet type"

        ## number of wavelet filter scales used
        #Remove this an donly read it if the wavelt_type is not TopHatarmonic (below)
        #self.N_scales = p['N_scales']
        #assert type(self.N_scales) is int and self.N_scales > 0, "N_scales"

        # tolerance for the checks for the responses: preserved component should be within resp_tol of 1, 
        # deprojected components should be within resp_tol of 0
        # defalt is 1e-3
        self.resp_tol = 1e-3
        if 'resp_tol' in p.keys():
            self.resp_tol = p['resp_tol']

        # width of high ell taper for filters, set to 0 if no taper desired. Default is 200
        self.taper_width = 200
        if 'taper_width' in p.keys():
            self.taper_width = p['taper_width']
        assert self.ELLMAX - self.taper_width > 10., "desired taper is too broad for given ELLMAX"

        if not self.wavelet_type == 'TopHatHarmonic':
            # Number of scales for the NILC
            self.N_scales = p['N_scales']
            assert type(self.N_scales) is int and self.N_scales > 0, "N_scales"

        # parameters for each wavelet type
        if self.wavelet_type == 'GaussianNeedlets':
            # FWHM values defining the gaussian needlets
            self.GN_FWHM_arcmin = np.asarray(p['GN_FWHM_arcmin'])
            assert len(self.GN_FWHM_arcmin) == self.N_scales - 1, "GN_FWHM_arcmin"
            assert all(FWHM_val > 0. for FWHM_val in self.GN_FWHM_arcmin), "GN_FWHM_arcmin"
            assert 'ellboundaries' not in p.keys()
            assert 'ellpeaks' not in p.keys()
        elif self.wavelet_type == 'CosineNeedlets':  #Fiona added CosineNeedlets
            # ellpeak values defining the cosine needlets
            self.ellpeaks = np.asarray(p['ellpeaks'])
            self.ellmin = np.asarray(p['ellmin'])
            assert len(self.ellpeaks) == self.N_scales - 1, "ellpeaks"
            assert all(ellpeak> 0. for ellpeak in self.ellpeaks), "ellpeaks"
            assert self.ellmin>=0, 'ellmin'
            assert 'GN_FWHM_arcmin' not in p.keys()
            assert 'ellboundaries' not in p.keys()
        elif self.wavelet_type == 'ScaleDiscretizedWavelets':
            self.ellboundaries = np.asarray(p['ellboundaries'])
            assert len(self.ellboundaries) == self.N_scales + 1, "ellpeaks"
            assert all(ellpeak> 0. for ellpeak in self.ellboundaries[1:]), "ellpeaks"
            assert self.ellboundaries[0]==0
            assert 'GN_FWHM_arcmin' not in p.keys()
            assert 'ellpeaks' not in p.keys()
        elif self.wavelet_type == 'TopHatHarmonic':
            # TODO: add functionality for the user to specity arbitrary ell-bins directly
            # the bin sizes for a linearly-ell-binnedHILC
            self.Delta_ell_HILC = p['BinSize']  
            self.ellbins = np.arange(0,self.ELLMAX+1,self.Delta_ell_HILC)
            self.N_scales = len(self.ellbins)-1
            assert type(self.N_scales) is int and self.N_scales > 0, "N_scales"

            # Option to save the harmonic covmat; by default it is False
            self.save_harmonic_covmat = False
            if 'save_harmonic_covmat' in p.keys():
                if p['save_harmonic_covmat'].lower() in ['true','yes','y']:
                    self.save_harmonic_covmat = True
            self.save_alms = False
            if 'save_alms' in p.keys():
                if p['save_alms'].lower() in ['true','yes','y']:
                    self.save_alms = True
        # TODO: implement these
        #elif self.wavelet_type == 'ScaleDiscretizedWavelets':
            # parameters defining these wavelets
            # TODO: add relevant assertions
            #self.B_param = p['B_param']
            #self.J_min = p['J_min']

        
        # flag to perform cross-ILC 
        self.cross_ILC = False
        if 'cross_ILC' in p.keys():
            if p['cross_ILC'].lower() in ['true','yes','y']:
                self.cross_ILC = True

        # number of frequency maps used
        self.N_freqs = p['N_freqs']
        assert type(self.N_freqs) is int and self.N_freqs > 0, "N_freqs"

        # if you want to use some pre-computed covmat and drop certain frequency channels, put this in here
        # as a list of the indices of the frequency channels you want to drop
        if 'drop_channels' in p.keys():
            self.drop_channels  = p['drop_channels']
            assert type(self.drop_channels) is list
            for x in self.drop_channels:
                assert type(x) is int
                assert x < self.N_freqs
        else:
            self.drop_channels = []

        # wavelet_beam_criterion, set to 1e-3 by default. This removes frequencies from the NILC
        # whose beams are a certain fraction smaller than the appropriate needlet filter within the range
        # of ells appropriate for the filter.
        if 'wavelet_beam_criterion' in p.keys():
            self.wavelet_beam_criterion = p['wavelet_beam_criterion']
        else:
            self.wavelet_beam_criterion = 1.e-3

        # override_N_freqs_to_use OVERRIDES information from wavlet_beam_criterion
        # and allows you to explicitly specify how many frequencies to use at each wavelet scale
        # this should be a list of ints, of length N_scales, where each entry in the list specifies 
        # how many frequency channels one should use at the scale corresponding to that entry. 
        # if the entry is less than N_freqs, the lowest resolution maps will be dropped from the NILC
        # such that there are the appropriate number of frequency channels used in each scale
        self.override_N_freqs_to_use = False
        if 'override_N_freqs_to_use' in p.keys():
            self.override_N_freqs_to_use = True
            self.N_freqs_to_use = p['override_N_freqs_to_use']
            assert type(self.N_freqs_to_use) is list
            assert len(self.N_freqs_to_use) == self.N_scales
            for x in self.N_freqs_to_use:
                print(x)
                assert type(x) is int
                assert x>0
                assert x<=self.N_freqs


        # optionally input the param_dict_file. The default is '../input/fg_SEDs_default_params.yml'
        self.param_dict_file = '../input/fg_SEDs_default_params.yml'
        if 'param_dict_file' in p.keys():
            self.param_dict_file = p['param_dict_file']

        # delta-function bandpasses/passbands or actual bandpasses/passbands
        self.bandpass_type = p['bandpass_type']
        assert self.bandpass_type in BP_TYPES, "unsupported bandpass type"
        if self.bandpass_type == 'DeltaBandpasses':
            # delta function bandpasses: frequency values in GHz
            self.freqs_delta_ghz = p['freqs_delta_ghz']
            assert len(self.freqs_delta_ghz) == self.N_freqs, "freqs_delta_ghz"
            for xind, x in enumerate(self.freqs_delta_ghz):
                if x in ["none","None"]:
                    self.freqs_delta_ghz[xind] = None
        elif self.bandpass_type == 'ActualBandpasses':
            # actual bandpasses: list of bandpass file names, each containing two columns: [freq [GHz]] [transmission [arbitrary norm.]]
            self.freq_bp_files = p['freq_bp_files']
            for xind,x in enumerate(self.freq_bp_files):
                if x.lower()=="none":
                    self.freq_bp_files[xind] = None
                if x.lower()=='delta':
                    self.freq_bp_files[xind] = 'delta'

                
            assert len(self.freq_bp_files) == self.N_freqs, "freq_bp_files"
            if 'delta' in self.freq_bp_files:
                self.freqs_delta_ghz = p['freqs_delta_ghz']
                assert len(self.freqs_delta_ghz) == self.N_freqs, "freqs_delta_ghz"
                for xind, x in enumerate(self.freqs_delta_ghz):
                    if x in ["none","None"]:
                        self.freqs_delta_ghz[xind] = None


        # do the wavelet maps already exist as saved files? we can tell the code to skip the check for this, if 
        # we know this alredy. Deafults to False
        self.wavelet_maps_exist = False
        if 'wavelet_maps_exist' in p.keys():
            if p['wavelet_maps_exist'].lower() in ['true','yes','y']:
                self.wavelet_maps_exist = True

        # frequency map file names
        self.freq_map_files = p['freq_map_files']
        assert len(self.freq_map_files) == self.N_freqs, "freq_map_files"

        # some preprocessing maps: Is there one map you want to subtract from all the inputs (eg the kinematic dipole)?
        # put the filename in a a string here
        self.map_to_subtract = None
        if 'map_to_subtract' in p.keys():
            self.map_to_subtract = p['map_to_subtract']
        # Is there a different map you want to subtract from all of the inputs? If so, put them in as a list here
        self.maps_to_subtract = None
        if 'maps_to_subtract'  in p.keys():
            self.maps_to_subtract = p['maps_to_subtract']
            assert len(self.maps_to_subtract) == self.N_freqs
            for xind,x in enumerate(self.maps_to_subtract):
                assert type(x) is str
                if x.lower() == 'none':
                    self.maps_to_subtract[xind] = None


        self.freq_map_field = 0
        if 'freq_map_field' in p.keys():
            self.freq_map_field = p['freq_map_field']

        # S1 and S2 maps for the cross-ILC
        if self.cross_ILC:
            self.freq_map_files_s1 = p['freq_map_files_s1']
            assert len(self.freq_map_files_s1) == self.N_freqs, "freq_map_files_s1"
            self.freq_map_files_s2 = p['freq_map_files_s2']
            assert len(self.freq_map_files_s2) == self.N_freqs, "freq_map_files_s2"

        # Do we want to compute the weights from the covmat (with np.linalg.solve) or from the invcovmat (with np.linalg.inv)?
        # Default is from invcovmat but this is both more numerically unstable, more computationally intensive, and more
        # memory intensive (as both covmats and invcovmats are saved). Possibly change this default to covmat?
        self.weights_from_invcovmat = True
        self.weights_from_covmat = False
        if 'weights_from_covmat' in p.keys():
            if p['weights_from_covmat'].lower() in ['true','yes','y']:
               self.weights_from_covmat =  True
               self.weights_from_invcovmat = False


        # Flag to apply weights to other maps than those used in the ILC weight calculation
        if 'maps_to_apply_weights' in p.keys():
            self.freq_map_files_for_weights = p['maps_to_apply_weights']
            assert len(self.freq_map_files_for_weights) == self.N_freqs, "freq_map_files_for_weights"
            self.apply_weights_to_other_maps = True
        else:
            self.apply_weights_to_other_maps = False

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

        # resolution at which to perform the ILC (if unspecified, deafults to resolution of the highest-resolution input map)
        self.perform_ILC_at_beam = None
        if 'perform_ILC_at_beam' in p.keys():
            #perform_ILC_at_beam should be in arcmin. 
            self.perform_ILC_at_beam = p['perform_ILC_at_beam']  

        # N_side value of highest-resolution input map (and presumed output map N_side)
        # be conservative and assume N_side must be a power of 2 (stricly speaking, only necessary for nest-ordering)
        # https://healpy.readthedocs.io/en/latest/generated/healpy.pixelfunc.isnsideok.html
        self.N_side = p['N_side']
        assert hp.pixelfunc.isnsideok(self.N_side, nest=True), "invalid N_side"
        self.N_pix = 12*self.N_side**2

        # Do we only want to perform NILC on part of the sky? if so, include the mask
        self.mask_before_covariance_computation = None
        if 'mask_before_covariance_computation' in p.keys():
            self.mask_before_covariance_computation = hp.fitsfunc.read_map(p['mask_before_covariance_computation'][0],field=p['mask_before_covariance_computation'][1])
            assert hp.get_nside(self.mask_before_covariance_computation) >= self.N_side
            if hp.get_nside(self.mask_before_covariance_computation) >self.N_side:
                print("fsky before is",np.sum(self.mask_before_covariance_computation)/self.mask_before_covariance_computation.shape[0],flush=True)
                self.mask_before_covariance_computation = hp.ud_grade(self.mask_before_covariance_computation,self.N_side)
                self.mask_before_covariance_computation[self.mask_before_covariance_computation<1]=0
                print("fsky after is",np.sum(self.mask_before_covariance_computation)/self.mask_before_covariance_computation.shape[0],flush=True)


        # ILC: component to preserve
        self.ILC_preserved_comp = p['ILC_preserved_comp']
        assert self.ILC_preserved_comp in COMP_TYPES, "unsupported component type in ILC_preserved_comp"

        # ILC: bias tolerance
        self.ILC_bias_tol = 0.01
        if 'ILC_bias_tol' in p.keys():
            self.ILC_bias_tol = p['ILC_bias_tol']
        assert self.ILC_bias_tol > 0. and self.ILC_bias_tol < 1., "invalid ILC bias tolerance"
        self.override_ILCbiastol = False
        if 'override_ILCbiastol_threshold' in p.keys():
            if p['override_ILCbiastol_threshold'].lower() in ['true','yes']:
                self.override_ILCbiastol = True
        #manually set realspace areas (in radians)
        if 'FWHM_pix' in p.keys():
            self.FWHM_pix = p['FWHM_pix']
            assert type(self.FWHM_pix) is list
            assert len(self.FWHM_pix) == self.N_scales
            for x in self.FWHM_pix:
                sigma_pix_temp = x / np.sqrt(8.*np.log(2.))
                assert sigma_pix_temp < np.pi
        else:
            self.FWHM_pix = None

        # ILC: component(s) to deproject (if any)
        self.N_deproj = p['N_deproj']
        assert (type(self.N_deproj) is int) or (type(self.N_deproj) is list)
        # if an integer is input, deproject this at all scales
        if type(self.N_deproj) is int:
            assert type(self.N_deproj) is int and self.N_deproj >= 0, "N_deproj"
            if (self.N_deproj > 0):
                self.ILC_deproj_comps = p['ILC_deproj_comps']
                assert len(self.ILC_deproj_comps) == self.N_deproj, "ILC_deproj_comps"
                assert all(comp in COMP_TYPES for comp in self.ILC_deproj_comps), "unsupported component type in ILC_deproj_comps"
                assert((self.N_deproj + 1) <= self.N_freqs), "not enough frequency channels to deproject this many components"
                self.deproject_from_channels = {}
                for component in self.ILC_deproj_comps:
                    self.deproject_from_channels[component] = 'all'

        # If a list is input, assign each element the corresponding scale
        if type(self.N_deproj) is list:
            assert len(self.N_deproj) == self.N_scales
            ind = 0
            self.ILC_deproj_comps=[]
            self.deproject_from_channels = {}
            for N_deproj in self.N_deproj:
                assert type(N_deproj) is int and N_deproj >= 0, "N_deproj"
                if (N_deproj > 0):
                    self.ILC_deproj_comps.append(p['ILC_deproj_comps'][ind])
                    assert len(self.ILC_deproj_comps[ind]) == N_deproj, "ILC_deproj_comps"
                    assert all(comp in COMP_TYPES for comp in self.ILC_deproj_comps[ind]), "unsupported component type in ILC_deproj_comps"
                    assert((N_deproj + 1) <= self.N_freqs), "not enough frequency channels to deproject this many components"
                    for component in self.ILC_deproj_comps[ind]:
                        self.deproject_from_channels[component] = 'all'
                else:
                    self.ILC_deproj_comps.append([])
                ind = ind+1
        if 'deproject_from_channels' in p.keys():
            for component in p['deproject_from_channels'].keys():
                self.deproject_from_channels[component] = p['deproject_from_channels'][component]


        self.print_timing = False
        if 'print_timing' in p.keys():
            assert type(p['print_timing']) is str
            if p['print_timing'].lower() in ['true','yes','t','y']:
                self.print_timing = True

        self.use_numba = True
        if 'use_numba' in p.keys():
            assert type (p['use_numba']) is str
            if p['use_numba'].lower() in ['false','no','f','n']:
                self.use_numba = False

        if 'save_as' in p.keys():
            if p['save_as'] == 'fits':
                 self.save_as_fits = True
                 self.save_as_hdf5 = False
            elif p['save_as'] == 'hdf5':
                self.save_as_hdf5 = True
                self.save_as_fits = False
                self.covmaps_hdf5_filename = self.output_dir + self.output_prefix + '_covmaps'+'_crossILC'*self.cross_ILC+'.hdf5'
                self.invcovmaps_hdf5_filename = self.output_dir + self.output_prefix + '_invcovmaps'+'_crossILC'*self.cross_ILC+'.hdf5'
                self.wavelet_coeff_hdf5_filename = self.output_dir + self.output_prefix + '_waveletmaps.hdf5'

                self.weight_filename_hdf5 =  self.output_dir + self.output_prefix + '_weightmaps_component_'+self.ILC_preserved_comp+'_crossILC'*self.cross_ILC+self.output_suffix_weights+'.fits'
                if type(self.N_deproj )is int:
                    if self.N_deproj>0:
                        self.weight_filename_hdf5 =  self.output_dir+self.output_prefix+'_weightmaps_component_'+self.ILC_preserved_comp+'_deproject_'+'_'.join(self.ILC_deproj_comps)+'_crossILC'*self.cross_ILC+self.output_suffix_weights+'.fits'
                else:
                    if self.N_deproj[0]>0:
                        self.weight_filename_hdf5 =  self.output_dir+self.output_prefix+'_weightmaps_component_'+self.ILC_preserved_comp+'_deproject_'+'_'.join(self.ILC_deproj_comps[0])+'_crossILC'*self.cross_ILC+self.output_suffix_weights+'.fits' 

        else:
            self.save_as_fits = True
            self.save_as_hdf5 = False
        assert not (self.save_as_fits and self.save_as_hdf5)


        # Do we want to calculate the weights from the covmat with np.linalg.solve()
        # or the invcovmat with np.linalg.inv() and np.matmul()?
        self.weights_from_covmat = True
        self.weights_from_invcovmat = False
        if 'weights_from_invcovmat' in p.keys():
            assert type(p['weights_from_invcovmat']) is str
            if p['weights_from_invcovmat'].lower() in ['true','yes']:
                self.weights_from_covmat = False
                self.weights_from_invcovmat = True


        # recompute_covmat_for_ndeproj is a flagthat, when it is on, includes the number of deprojected components
        # in the filenames for the covmat. If it is off, it does not. This is important because the size of the real
        # space filters is set by calculating an area that includes enough modes to respect a userspecified ILC bias
        # tolerance, and this calculation changes depending on N_deproj. However, it is computationally intensive
        # to recompute the covmat, and this is likely a small effect, so it is often likely OK to just use the same 
        # covmat and not recompute all the time. So, if you don't want to recompute for different values of N_deproj, 
        # turn this off and it will just use the covmat calculated on the area appropriate for what N_deproj was the
        # first time you ran the code.
        if 'recompute_covmat_for_ndeproj' in p.keys(): 
            self.recompute_covmat_for_ndeproj = p['recompute_covmat_for_ndeproj']
        else:
            self.recompute_covmat_for_ndeproj = False

        ####################
        ### TODO: this block of code with SED parameters, etc is currently not used anywhere
        ###   instead, we currently just get the SED parameter info from fg_SEDs_default_params.yml
        ###   if we wanted to do something fancy like sample over SED parameters, we would want to make use of this code
        # ILC: SED parameters
        if 'N_SED_params' in p.keys():
            self.N_SED_params = p['N_SED_params']
        else:
             self.N_SED_params = 0
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
        if 'N_maps_xcorr' in p.keys():
            self.N_maps_xcorr = p['N_maps_xcorr']
        else:
             self.N_maps_xcorr = 0
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

        if not self.wavelet_maps_exist:

            self.maps = np.zeros((self.N_freqs,self.N_pix), dtype=np.float64)
            for i in range(self.N_freqs):
                # TODO: allow specification of nested or ring ordering (although will already work here if fits keyword ORDERING is present)
                temp_map = hp.fitsfunc.read_map(self.freq_map_files[i], field=self.freq_map_field)
                assert len(temp_map) <= self.N_pix, "input map at higher resolution than specified N_side"
                if (len(temp_map) == self.N_pix):
                    self.maps[i] = np.copy(temp_map)
                elif (len(temp_map) < self.N_pix):
                    # TODO: should probably upgrade in harmonic space to get pixel window correct
                    self.maps[i] = np.copy( hp.pixelfunc.ud_grade(temp_map, nside_out=self.N_side, order_out='RING', dtype=np.float64) )
            del(temp_map)
            # if cross-ILC read in the S1 and S2 maps
            if self.cross_ILC:
                self.maps_s1 = np.zeros((self.N_freqs,self.N_pix), dtype=np.float64)
                self.maps_s2 = np.zeros((self.N_freqs,self.N_pix), dtype=np.float64)
                for i in range(self.N_freqs):
                    # TODO: allow specification of nested or ring ordering (although will already work here if fits keyword ORDERING is present)
                    temp_map_s1 = hp.fitsfunc.read_map(self.freq_map_files_s1[i], field=self.freq_map_field)
                    assert len(temp_map_s1) <= self.N_pix, "input map at higher resolution than specified N_side"
                    temp_map_s2 = hp.fitsfunc.read_map(self.freq_map_files_s2[i], field=self.freq_map_field)
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
            # if you want to subtract something from the maps, do it here 
            if self.map_to_subtract is not None:
                map_to_subtract = hp.fitsfunc.read_map(self.map_to_subtract)
                assert hp.get_nside(map_to_subtract) >= self.N_side
                if hp.get_nside(map_to_subtract) > self.N_side:
                    map_to_subtract = hp.ud_grade(map_to_subtract,self.N_side)
                self.maps = self.maps - map_to_subtract[None,:]
                if self.cross_ILC:
                    self.maps_s1 = self.maps_s1 -  map_to_subtract[None,:]
                    self.maps_s2 = self.maps_s2 -  map_to_subtract[None,:]
            if self.maps_to_subtract is not None:
                 for freqind in range(self.N_freqs):
                    if self.maps_to_subtract[freqind] is not None:
                        map_to_subtract = hp.fitsfunc.read_map(self.maps_to_subtract[freqind])
                        assert hp.get_nside(map_to_subtract) >= self.N_side
                        if hp.get_nside(map_to_subtract) > self.N_side:
                            map_to_subtract = hp.ud_grade(map_to_subtract,self.N_side)
                        self.maps[freqind] = self.maps[freqind] - map_to_subtract
                        if self.cross_ILC:
                            self.maps_s1[freqind] = self.maps_s1[freqind] - map_to_subtract
                            self.maps_s2[freqind] = self.maps_s2[freqind] - map_to_subtract

        # if we need to apply weights to alternative maps, read them in
        if self.apply_weights_to_other_maps:
            print("reading in maps for weights",flush=True)
            self.maps_for_weights = np.zeros((self.N_freqs,self.N_pix), dtype=np.float64)
            for i in range(self.N_freqs):
                # TODO: allow specification of nested or ring ordering (although will already work here if fits keyword ORDERING is present)
                temp_map = hp.fitsfunc.read_map(self.freq_map_files_for_weights[i], field=self.freq_map_field)
                assert len(temp_map) <= self.N_pix, "input map at higher resolution than specified N_side"
                if (len(temp_map) == self.N_pix):
                    self.maps_for_weights[i] = np.copy(temp_map)
                elif (len(temp_map) < self.N_pix):
                    # TODO: should probably upgrade in harmonic space to get pixel window correct
                    self.maps_for_weights[i] = np.copy( hp.pixelfunc.ud_grade(temp_map, nside_out=self.N_side, order_out='RING', dtype=np.float64) )
            del(temp_map)
        # also read in maps with which to cross-correlate, if specified
        if self.N_maps_xcorr != 0:
            # maps
            self.maps_xcorr = np.zeros((self.N_maps_xcorr,self.N_pix), dtype=np.float64)
            for i in range(self.N_maps_xcorr):
                temp_map = hp.fitsfunc.read_map(self.maps_xcorr_files[i], field=0)
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
                    temp_map = hp.fitsfunc.read_map(self.masks_xcorr_files[i], field=0)
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
                if self.freq_bp_files[i] is not None:
                    if  self.freq_bp_files[i].lower()  !='delta':
                        (self.bandpasses).append(np.loadtxt(self.freq_bp_files[i], unpack=True, usecols=(0,1)))
                    else:
                        self.bandpasses.append('Delta')
                else:
                    self.bandpasses.append(None)


    # method for reading in beams
    # self.beams is a list of length self.N_freqs where each entry is an (ELLMAX+1) x 2 array
    def read_beams(self):
        if self.beam_type == 'Gaussians':
            self.beams = np.zeros((self.N_freqs,self.ELLMAX+1,2), dtype=np.float64)
            for i in range(self.N_freqs):
                self.beams[i] = np.transpose(np.array([np.arange(self.ELLMAX+1), hp.sphtfunc.gauss_beam(self.beam_FWHM_arcmin[i]*(np.pi/180.0/60.0), lmax=self.ELLMAX)]))
                # if self.perform_ILC_at_beam is specified, convolve all maps to the common_beam
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
    # method for turning maps to alms
    def maps2alms(self):
        self.alms=[]
        for freqind,mapp in enumerate(self.maps):
            filename = self.output_dir + self.output_prefix + '_alm_freq'+str(freqind)+'.fits'
            exists = os.path.isfile(filename)
            if exists:
                    self.alms.append(hp.fitsfunc.read_alm(filename))
            else:
                self.alms.append(hp.map2alm(mapp, lmax=self.ELLMAX))
                if self.save_alms:
                    hp.fitsfunc.write_alm(filename,self.alms[freqind])
        if self.cross_ILC:
            self.alms_s1 = []
            self.alms_s2 = []
            for mapp in self.maps_s1:
                self.alms_s1.append(hp.map2alm(mapp, lmax=self.ELLMAX))
            for mapp in self.maps_s2:
                self.alms_s2.append(hp.map2alm(mapp, lmax=self.ELLMAX))
    def alms2cls(self):
        self.cls = np.zeros((len(self.alms),len(self.alms),self.ELLMAX+1))
        new_beam = self.common_beam
        for a in range(len(self.maps)):
            inp_beam_a = (self.beams)[a]
            beam_fac_a = new_beam[:,1]/inp_beam_a[:,1]
            for b in range(a,len(self.maps)):
                 inp_beam_b = (self.beams)[b]
                 beam_fac_b = new_beam[:,1]/inp_beam_b[:,1]
                 self.cls[a,b]=self.cls[b,a] = hp.alm2cl(self.alms[a],self.alms[b],lmax=self.ELLMAX) * beam_fac_b * beam_fac_a 
        if self.cross_ILC:
            self.cls_s1s2= np.zeros((len(self.alms),len(self.alms),self.ELLMAX+1))
            for a in range(len(self.maps)):
                inp_beam_a = (self.beams)[a]
                beam_fac_a = new_beam[:,1]/inp_beam_a[:,1]
                for b in range(len(self.maps)):
                    inp_beam_b = (self.beams)[b]
                    beam_fac_b = new_beam[:,1]/inp_beam_b[:,1]

                    self.cls_s1s2[a,b]=hp.alm2cl(self.alms_s1[a],self.alms_s2[b],lmax=self.ELLMAX) * beam_fac_b * beam_fac_a 
