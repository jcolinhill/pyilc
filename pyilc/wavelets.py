from __future__ import print_function
import time
import numpy as np
import healpy as hp
from astropy.io import fits
import os
import matplotlib

import h5py #Fiona - we could make this optional if we don't want to include other dependencies


from numba import njit,prange

@njit(parallel=True)
def jit_linsolve(A, b):
    m = A.shape[0]
    n=A.shape[-1]
    k=b.shape[-1]
    ret = np.empty((m, n,k))
    for i in prange(m):
        ret[i, :] = np.linalg.solve(A[i], b)
    return ret

def my_numba_solver(a,b):
    return jit_linsolve(a,b)

@njit(parallel=True)
def jit_linsolve_parallelb(A, b):
    m = A.shape[0]
    n=A.shape[-1]
    k=b.shape[-1]
    ret = np.empty((m, k))
    for i in prange(m):
        ret[i, :] = np.linalg.solve(A[i], b[i])
    return ret

def my_numba_solver_parallelb(a,b):
    return jit_linsolve_parallelb(a,b)

@njit(parallel=True)
def jit_det(A,):
    m = A.shape[0]
    n=A.shape[0]
    ret = np.empty((m))
    for i in prange(m):
        ret[i] = np.linalg.det( A[i])
    #ret = np.einsum('ajp,bj->abp', a, b)
    return ret

def numba_det(a,):
    return jit_det(a,)

@njit(parallel=True)
def jit_matmul(A,b):
    m = A.shape[0]
    n=A.shape[1]
    ret = np.empty((m,n,n))
    for i in prange(m):
        ret[i] = np.dot(A[i],b)
    #ret = np.einsum('ajp,bj->abp', a, b)
    return ret

def numba_matmul(a,b):
    return jit_matmul(a,b)


matplotlib.use('pdf')
matplotlib.rc('font', family='serif', serif='cm10')
matplotlib.rc('text', usetex=True)
fontProperties = {'family':'sans-serif',
                  'weight' : 'normal', 'size' : 16}
import matplotlib.pyplot as plt
from input import ILCInfo
from fg import get_mix, get_mix_bandpassed
import time
"""
this module constructs the Wavelets class, which contains the
harmonic-space filters defining a set of wavelets, as well as
some associated methods.  three example types of wavelets are
explicitly constructed:
- Gaussian needlets:
  - Planck 2015 NILC y-map needlets
  - Planck 2016 GNILC dust map needlets
- cosine needlets 
- Top-hat needlets
- scale-discretized wavelets [TODO]

relevant parameters:
- N_scales: number of wavelet scales
- ELLMAX: maximum multipole

the module also contains wavelet transformation functions

the module also contains the wavelet ILC function
"""

class Wavelets(object):
    # initialize the filters to unity
    # construct non-trivial filters (or user can define)
    def __init__(self, N_scales=10, ELLMAX=4097, tol=1.e-6, taper_width=200):
        self.N_scales = N_scales #number of needlet filters
        self.ELLMAX = ELLMAX #maximum multipole
        self.tol = tol #tolerance for transmission condition
        # Include option to apply a taper near ELLMAX to avoid aliasing of small-scale power 
        # due to sharp truncation. Set taper_width to zero for no taper
        self.taper_width = taper_width 
        assert(self.N_scales > 0)
        assert(type(self.N_scales) is int)
        assert(self.ELLMAX > 0)
        assert(type(self.ELLMAX) is int)
        # initialize filters
        self.ell = np.linspace(0, self.ELLMAX, num=ELLMAX+1, endpoint=True, retstep=False)
        self.filters = np.ones((self.N_scales,self.ELLMAX+1),dtype=float)

    # Planck 2015 NILC y-map Gaussian needlet filters: [600', 300', 120', 60', 30', 15', 10', 7.5', 5']
    # Planck 2016 GNILC Gaussian needlet filters: [300' , 120' , 60' , 45' , 30' , 15' , 10' , 7.5' , 5']
    # (Thanks to Mathieu Remazeilles for providing these numbers (2/22/19) -- although the y-map filters are still slightly different at low ell than those in the paper)
    # For the details of the construction,
    #   see Eqs. (A.29)-(A.32) of http://arxiv.org/pdf/1605.09387.pdf
    # Note that these can be constructed for different (user-specified) choices of N_scales and ELLMAX also.
    # Define the FWHM values used in the Gaussians -- default = Planck 2015 NILC y-map values
    def GaussianNeedlets(self, FWHM_arcmin=np.array([600., 300., 120., 60., 30., 15., 10., 7.5, 5.])):
        # FWHM need to be in strictly decreasing order, otherwise you'll get nonsense
        if ( any( i <= j for i, j in zip(FWHM_arcmin, FWHM_arcmin[1:]))):
            raise AssertionError
        # check consistency with N_scales
        assert(len(FWHM_arcmin) == self.N_scales - 1)
        FWHM = FWHM_arcmin * np.pi/(180.*60.)
        # define gaussians
        Gaussians = np.zeros((self.N_scales-1,self.ELLMAX+1))
        for i in range(self.N_scales-1):
            Gaussians[i] = hp.gauss_beam(FWHM[i], lmax=self.ELLMAX)
        # define needlet filters in harmonic space
        self.filters[0] = Gaussians[0]
        for i in range(1,self.N_scales-1):
            self.filters[i] = np.sqrt(Gaussians[i]**2. - Gaussians[i-1]**2.)
        self.filters[self.N_scales-1] = np.sqrt(1. - Gaussians[self.N_scales-2]**2.)
        # simple check to ensure that sum of squared transmission is unity as needed for NILC algorithm
        assert (np.absolute( np.sum( self.filters**2., axis=0 ) - np.ones(self.ELLMAX+1,dtype=float)) < self.tol).all(), "wavelet filter transmission check failed"
        return self.ell, self.filters

    # the cosine needlet filters used in the Planck 2015/2018 CMB analysis are described in
    #    App. B of https://arxiv.org/pdf/1502.05956.pdf and App. B of https://arxiv.org/pdf/1807.06208.pdf
    def CosineNeedlets(self,ellmin=None, ellpeaks=None, ): # Fiona added CosineNeedlets
        '''

        '''

        ellpeaks = [ellmin]+list(ellpeaks)
        # ellpeaks need to be in strictly increasing order, otherwise you'll get nonsense
        if ( any( i >= j for i, j in zip(ellpeaks, ellpeaks[1:]))):
            raise AssertionError

         # check consistency with N_scales
        assert self.N_scales == len(ellpeaks)

        assert ellpeaks[-1] == self.ELLMAX+1

        self.filters= np.zeros((self.N_scales,self.ELLMAX+1))
        ells=np.arange(self.ELLMAX+1)

        for i in range(0,self.N_scales-1):
            filt1 = np.logical_and(ells<ellpeaks[i],ells>=ellpeaks[i-1])
            filt2 = np.logical_and(ells<ellpeaks[i+1],ells>=ellpeaks[i])
            self.filters[i,filt1] = np.cos(np.pi/2*(ellpeaks[i]-ells[filt1])/(ellpeaks[i]-ellpeaks[i-1]))#hp.gauss_beam(FWHM[i], lmax=ELLMAX)
            self.filters[i,filt2] = np.cos(np.pi/2*(ells[filt2]-ellpeaks[i])/(ellpeaks[i+1]-ellpeaks[i]))#hp.gauss_beam(FWHM[i], lmax=ELLMAX)

        i=i+1
        filt1 = np.logical_and(ells<ellpeaks[i],ells>=ellpeaks[i-1])

        self.filters[i,filt1] = np.cos(np.pi/2*(ellpeaks[i]-ells[filt1])/(ellpeaks[i]-ellpeaks[i-1]))#hp.gauss_beam(FWHM[i], lmax=ELLMAX)

        # simple check to ensure that sum of squared transmission is unity as needed for NILC algorithm 
        assert (np.absolute( np.sum( self.filters**2., axis=0 ) - np.ones(self.ELLMAX+1,dtype=float)) < self.tol).all(), "wavelet filter transmission check failed"
        self.ell = ells
        return self.ell, self.filters

    # tapered top hat wavelets
    def TaperedTopHats(self,ellboundaries = None,taperwidths = None):
        if ( any( i >= j for i, j in zip(ellboundaries, ellboundaries[1:]))):
            raise AssertionError
        assert self.N_scales ==  len(ellboundaries)+1

        all_ell_boundaries = np.array([0]+list(ellboundaries)+[self.ELLMAX+1])
        self.filters= np.zeros((self.N_scales,self.ELLMAX+1))

        ells = np.arange(self.ELLMAX+1)

        for i in range(0,self.N_scales-1):
            filt = np.ones(self.ELLMAX+1)
            taper =  (1.0 - 0.5*(np.tanh(2/taperwidths[i]*(ells- (ellboundaries[i]))) + 1.0))

            self.filters[i] = np.sqrt(np.ones(self.ELLMAX+1) * taper - np.sum(self.filters[:i]**2,axis=0))
            self.filters[i][np.isnan(self.filters[i])]=0.0
        self.filters[-1] = np.sqrt(np.ones(self.ELLMAX+1)  - np.sum(self.filters[:-1]**2,axis=0))
        self.filters[-1][np.isnan(self.filters[-1])]=0.0
        assert (np.absolute( np.sum( self.filters**2., axis=0 ) - np.ones(self.ELLMAX+1,dtype=float)) < self.tol).all(), "wavelet filter transmission check failed"
        self.ell = ells
        return self.ell, self.filters
    # scale-discretized wavelets
    # tophat wavelets
    def TopHats(self,ellboundaries =None):

        # ellboundaries need to be in strictly increasing order, otherwise you'll get nonsense
        assert ellboundaries[0]==0
        if ( any( i >= j for i, j in zip(ellboundaries, ellboundaries[1:]))):
            raise AssertionError

        # check consistency with N_scales
        assert self.N_scales == len(ellboundaries)-1

        assert ellboundaries[-1] == self.ELLMAX+1

        self.filters= np.zeros((self.N_scales,self.ELLMAX+1))
        ells=np.arange(self.ELLMAX+1)

        for i in range(0,self.N_scales):
            filt = np.logical_and(ells < ellboundaries[i+1], ells >= ellboundaries[i])
            self.filters[i,filt] = 1


        # simple check to ensure that sum of squared transmission is unity as needed for NILC algorithm 
        assert (np.absolute( np.sum( self.filters**2., axis=0 ) - np.ones(self.ELLMAX+1,dtype=float)) < self.tol).all(), "wavelet filter transmission check failed"
        self.ell = ells
        return self.ell, self.filters

    def TopHatHarmonic(self, ellbins):

        self.filters = np.zeros((len(ellbins)-1,self.ELLMAX+1),dtype=float)
        for i in range(0,len(ellbins)-1):
            self.filters[i] = np.zeros(self.ELLMAX+1)
            self.filters[i,ellbins[i]:ellbins[i+1]] = 1
        self.filters[-1,ellbins[i+1]:]=1
        # simple check to ensure that sum of squared transmission is unity as needed for NILC algorithm
        assert (np.absolute( np.sum( self.filters**2., axis=0 ) - np.ones(self.ELLMAX+1,dtype=float)) < self.tol).all(), "wavelet filter transmission check failed"
        return self.ell, self.filters

    def plot_wavelets(self, filename='example_wavelet_plot', log_or_lin='log'):
        plt.clf()
        if log_or_lin == 'log':
            for i in range(self.N_scales):
                plt.semilogx(self.ell, self.filters[i], 'k', lw=1.)
        elif log_or_lin == 'lin':
            for i in range(self.N_scales):
                plt.plot(self.ell, self.filters[i], 'k', lw=1.)
        else:
            raise AssertionError
        plt.xlim(left=1, right=self.ELLMAX+1)
        plt.ylim(0.0, 1.0)
        plt.xlabel(r'$\ell$', fontsize=18)
        plt.ylabel(r'$h^j_{\ell}$', fontsize=18)
        plt.grid(alpha=0.5)
        plt.savefig(filename+log_or_lin+'.pdf')

class scale_info(object):
    ## It might be better to let this inherit from the wavelets class, and just use one scale_info object instead of both wavelets
    ## and scale_info - possibly a future edit
    def __init__(self,wv,info,):
        ##########################
        # criterion to determine which frequency maps to use for each wavelet filter scale
        # require multipole ell_F where wavelet filter F(ell_F) = wavelet_beam_criterion (on its decreasing side)
        #   to be less than the multipole ell_B where the beam B(ell_B) = wavelet_beam_criterion
        # note that this assumes monotonicity of the beam
        # and assumes filter function has a decreasing side, which is generally not true for the smallest-scale wavelet filter
        self.freqs_to_use = np.full((wv.N_scales,info.N_freqs), False)
        self.N_freqs_to_use = np.zeros(wv.N_scales,dtype=int)
        self.N_side_to_use = np.ones(wv.N_scales,dtype=int)*info.N_side #initialize all of the internal, per-scale N_side values to the output N_side
        ell_F = np.zeros(wv.N_scales)
        ell_B = np.zeros(info.N_freqs)
        if len(ell_F)>1:
            for i in range(wv.N_scales-1):
                ell_peak = np.argmax(wv.filters[i]) #we'll use this to ensure we're on the decreasing side of the filter
                ell_F[i] = ell_peak + (np.abs( wv.filters[i][ell_peak:] - info.wavelet_beam_criterion )).argmin()
                if ell_F[i] > wv.ELLMAX:
                    ell_F[i] = wv.ELLMAX
            ell_F[-1] = ell_F[-2] #just use the second-to-last criterion for the last one #TODO: improve this
        else:
             for i in range(wv.N_scales):
                ell_peak = np.argmax(wv.filters[i]) #we'll use this to ensure we're on the decreasing side of the filter
                ell_F[i] = ell_peak + (np.abs( wv.filters[i][ell_peak:] - info.wavelet_beam_criterion )).argmin()
                if ell_F[i] > wv.ELLMAX:
                    ell_F[i] = wv.ELLMAX

        for j in range(info.N_freqs):
            ell_B[j] = (np.abs( (info.beams[j])[:,1] - info.wavelet_beam_criterion )).argmin()
        for i in range(wv.N_scales):
            for j in range(info.N_freqs):
                if ell_F[i] <= ell_B[j]:
                    if j not in info.drop_channels:
                        self.freqs_to_use[i][j] = True
                        self.N_freqs_to_use[i] += 1
                    else:
                        self.freqs_to_use[i][j] = False
                else:
                    self.freqs_to_use[i][j] = False
            # Fiona override Nfreqstouse
            if info.override_N_freqs_to_use:
                self.N_freqs_to_use[i] = info.N_freqs_to_use[i]
                for j in range(self.N_freqs_to_use[i]):
                    self.freqs_to_use[i][info.N_freqs-j-1]=True
                for j in range(info.N_freqs-self.N_freqs_to_use[i]):
                    self.freqs_to_use[i][j] = False
                print("overriding nfreqs to use",flush=True)
            # check that number of frequencies is non-zero
            assert(self.N_freqs_to_use[i] > 0), "insufficient number of channels for high-resolution filter(s)"
            # check that we still have enough frequencies for desired deprojection at each filter scale
            if type(info.N_deproj) is int:
                assert((info.N_deproj + 1) <= self.N_freqs_to_use[i]), "not enough frequency channels to deproject this many components at scale "+str(i)
            else:
                assert((info.N_deproj[i] + 1) <= self.N_freqs_to_use[i]), "not enough frequency channels to deproject this many components at scale "+ str()
            # determine N_side value to use for each filter scale, by finding the smallest valid N_side larger than ell_F[i]
            for j in range(20):
                if (ell_F[i] < 2**j):
                    self.N_side_to_use[i] = int(2**j)
                    break
            if (self.N_side_to_use[i] > info.N_side):
                self.N_side_to_use[i] = info.N_side
        self.N_pix_to_use = 12*(self.N_side_to_use)**2
        ##########################
        ##########################
        # criterion to determine the real-space gaussian FWHM used in wavelet ILC
        # based on ILC bias mode-counting
        self.FWHM_pix = np.zeros(wv.N_scales,dtype=float)
        if info.wavelet_type == 'GaussianNeedlets':
            ell, filts = wv.GaussianNeedlets(info.GN_FWHM_arcmin)
        elif info.wavelet_type == 'TopHatHarmonic':
            ell, filts = wv.TopHatHarmonic(info.ellbins)
        elif info.wavelet_type == 'CosineNeedlets':
            ell, filts = wv.CosineNeedlets(ellmin = info.ellmin,ellpeaks = info.ellpeaks)
        elif info.wavelet_type =='TaperedTopHats':
            ell,filts = wv.TaperedTopHats(ellboundaries = info.ellboundaries,taperwidths= info.taperwidths)
        elif info.wavelet_type == 'TopHats':
            ell, filts = wv.TopHats(ellboundaries = info.ellboundaries)
        else:
            raise NotImplementedError
        # compute effective number of modes associated with each filter (on the full sky)
        # note that the weights we use are filt^2, since this is the quantity that sums to unity at each ell
        N_modes = np.zeros(wv.N_scales, dtype=float)
        for i in range(wv.N_scales):
            N_modes[i] = np.sum( (2.*ell + np.ones(wv.ELLMAX+1)) * (filts[i])**2. )
        # now find real-space Gaussian s.t. number of modes in that area satisfies ILC bias threshold
        # we use the flat-sky approximation here -- TODO: could improve this


        if info.FWHM_pix is None:

            for i in range(wv.N_scales):
                # this expression comes from noting that ILC_bias_tol = (N_deproj+1 - N_freqs)/N_modes_eff
                #   where N_modes_eff = N_modes * (2*pi*sigma_pix^2)/(4*pi)
                #   and then solving for sigma_pix
                # note that this corrects an error in Eq. 3 of Planck 2015 y-map paper -- the numerator should be (N_ch - 2) in their case (if they're deprojecting CMB)
                if type(info.N_deproj) is int:
                    sigma_pix_temp = np.sqrt( np.absolute( 2.*(float( (info.N_deproj + 1) - self.N_freqs_to_use[i] )) / (N_modes[i] * info.ILC_bias_tol) ) ) #result is in radians
                else:
                    sigma_pix_temp = np.sqrt( np.absolute( 2.*(float( (info.N_deproj[i] + 1) - self.N_freqs_to_use[i] )) / (N_modes[i] * info.ILC_bias_tol) ) ) #result is in radians
                if info.override_ILCbiastol:
                    if sigma_pix_temp > np.pi:
                        print ("not enough modes to satisfy ILC_bias_tol at scale " +str(i) ,flush=True)
                        sigma_pix_temp = np.pi
                else:
                     assert sigma_pix_temp < np.pi, "not enough modes to satisfy ILC_bias_tol at scale " +str(i) #don't want real-space gaussian to be the full sky or close to it
                # note that sigma_pix_temp can come out zero if N_deproj+1 = N_freqs_to_use (formally bias vanishes in this case because the problem is fully constrained)
                # for now, just set equal to case where N_freqs_to_use = N_deproj
                if sigma_pix_temp == 0.:
                    sigma_pix_temp = np.sqrt( np.absolute( 2. / (N_modes[i] * info.ILC_bias_tol) ) ) #result is in radians
                self.FWHM_pix[i] = np.sqrt(8.*np.log(2.)) * sigma_pix_temp #in radians
        else:
            self.FWHM_pix=info.FWHM_pix

    def mixing_matrix_at_scale_j(self,j,info):
            ### compute the mixing matrix A_{i\alpha} ###
            # this is the alpha^th component's SED evaluated at the i^th frequency
            # units of A_mix are K_CMB
            # Note: only include channels that are being used for this filter scale
            if type(info.N_deproj) is int:
                N_deproj = info.N_deproj
                if N_deproj>0:
                    ILC_deproj_comps = info.ILC_deproj_comps
            else:
                N_deproj = info.N_deproj[j]
                if N_deproj>0:
                    ILC_deproj_comps = info.ILC_deproj_comps[j]
            N_comps = (N_deproj + 1)
            A_mix = np.zeros((int(self.N_freqs_to_use[j]),N_comps))
            countt = 0
            for a in range(info.N_freqs):
                if (self.freqs_to_use[j][a] == True):
                    for b in range(N_comps):
                        if (b == 0): # zero^th component is special (this is the one being preserved in the ILC)
                            if (info.bandpass_type == 'DeltaBandpasses'):
                                # N.B. get_mix and get_mix_bandpassed assume the input maps are in uK_CMB, i.e., responses are computed in uK_CMB, but we are assuming in this code that all maps are in K_CMB, hence factor of 1.e-6 below
                                # However, note that as a consequence an output NILC CMB map from this code has units of uK_CMB!
                                A_mix[countt][b] = 1.e-6 * (get_mix([info.freqs_delta_ghz[a]], info.ILC_preserved_comp, param_dict_file=info.param_dict_file, param_dict_override=None, dust_beta_param_name='beta_CIB', radio_beta_param_name='beta_radio',
                                    radio_beta1_param_name='beta1_radio',
                                    radio_beta2_param_name='beta2_radio'))[0] #convert to K from uK
                            elif (info.bandpass_type == 'ActualBandpasses'):
                                get_delta_bandpass = False
                                if info.freq_bp_files[a] is None:
                                    get_delta_bandpass = False
                                elif info.freq_bp_files[a].lower() == 'delta':
                                    get_delta_bandpass = True
                                else:
                                    get_delta_bandpass = False
                                if get_delta_bandpass:

                                    A_mix[countt][b] = 1.e-6 * (get_mix([info.freqs_delta_ghz[a]], info.ILC_preserved_comp, param_dict_file=info.param_dict_file, param_dict_override=None, dust_beta_param_name='beta_CIB', radio_beta_param_name='beta_radio',
                                    radio_beta1_param_name='beta1_radio',
                                    radio_beta2_param_name='beta2_radio'))[0] #convert to K from uK
                                else:

                                    A_mix[countt][b] = 1.e-6 * (get_mix_bandpassed([info.freq_bp_files[a]], info.ILC_preserved_comp, param_dict_file=info.param_dict_file, param_dict_override=None, dust_beta_param_name='beta_CIB', radio_beta_param_name='beta_radio',
                                    radio_beta1_param_name='beta1_radio',
                                    radio_beta2_param_name='beta2_radio'))[0] #convert to K from uK
                        else:
                            if (info.bandpass_type == 'DeltaBandpasses'):
                                A_mix[countt][b] = 1.e-6 * (get_mix([info.freqs_delta_ghz[a]], ILC_deproj_comps[b-1], param_dict_file=info.param_dict_file, param_dict_override=None, dust_beta_param_name='beta_CIB', radio_beta_param_name='beta_radio',
                                    radio_beta1_param_name='beta1_radio',
                                    radio_beta2_param_name='beta2_radio'))[0] #convert to K from uK
                            elif (info.bandpass_type == 'ActualBandpasses'):
                                get_delta_bandpass = False
                                if info.freq_bp_files[a] is None:
                                    get_delta_bandpass = False
                                elif info.freq_bp_files[a].lower() == 'delta':
                                    get_delta_bandpass = True
                                else:
                                    get_delta_bandpass = False
                                if get_delta_bandpass:
                                    A_mix[countt][b] = 1.e-6 * (get_mix([info.freqs_delta_ghz[a]], ILC_deproj_comps[b-1], param_dict_file=info.param_dict_file, param_dict_override=None, dust_beta_param_name='beta_CIB', radio_beta_param_name='beta_radio',
                                    radio_beta1_param_name='beta1_radio',
                                    radio_beta2_param_name='beta2_radio'))[0] #convert to K from uK
                                else:
                                    A_mix[countt][b] = 1.e-6 * (get_mix_bandpassed([info.freq_bp_files[a]], ILC_deproj_comps[b-1], param_dict_file=info.param_dict_file, param_dict_override=None, dust_beta_param_name='beta_CIB', radio_beta_param_name='beta_radio',
                                    radio_beta1_param_name='beta1_radio',
                                    radio_beta2_param_name='beta2_radio'))[0] #convert to K from uK
                    countt += 1
            # normalize the columns of A_mix corresponding to the deprojected components so that they have values near unity
            if (N_deproj != 0):
                for b in range(1,N_deproj+1):
                    max_temp = np.amax(A_mix[:,b])
                    A_mix[:,b] = A_mix[:,b]/max_temp

            # set to zero the channels that we have told not to deproject in
            if (N_deproj != 0):
                for b in range(1,N_deproj+1):
                    component = ILC_deproj_comps[b-1]
            return A_mix

    def compute_covariance_at_scale_j(self,info,j,FWHM_pix):
        cov_maps_temp=[]
        #Fiona add option to include a mask at the level of covariance computation:
        #todo: this needs refinement (and the hard-coding of this part needs to be removed)
        if info.mask_before_covariance_computation is not None:
            print("fsky of whole mask is "+str(np.sum(info.mask_before_covariance_computation)/info.mask_before_covariance_computation.shape[0]),flush=True)
            dgraded_mask = hp.ud_grade(info.mask_before_covariance_computation,self.N_side_to_use[j])
            dgraded_mask[dgraded_mask!=0]=1
            print("fsky at scale "+str(j)+" is "+str(np.sum(dgraded_mask)/dgraded_mask.shape[0]),flush=True)
            if np.sum(dgraded_mask)== 0:
                assert self.N_side_to_use[j] < 256
                mask1 = hp.ud_grade(info.mask_before_covariance_computation,256)
                dgraded_mask = hp.ud_grade(mask1, self.N_side_to_use[j])
                dgraded_mask[dgraded_mask!=0]=1
                print("fsky at scale "+str(j)+" is "+str(np.sum(dgraded_mask)/dgraded_mask.shape[0]),flush=True)
                assert np.sum(dgraded_mask)>0

            smoothed_mask = hp.sphtfunc.smoothing(dgraded_mask,FWHM_pix[j])
            fskyinv = np.zeros(smoothed_mask.shape)
            fskyinv[smoothed_mask!=0] = 1.0/smoothed_mask[smoothed_mask!=0]
            fskyinv[smoothed_mask==0] = 1.0e100

        else:
            fskyinv = 1.0
            dgraded_mask = 1.0

        smoothed_maps_A = {}
        unsmoothed_maps_A = {}
        if info.cross_ILC:
            smoothed_maps_B = {}
            unsmoothed_maps_B = {}
        for a in range(info.N_freqs):
            if (self.freqs_to_use[j][a] == True) :
                if not info.cross_ILC:
                    season_A = None
                else:
                    season_A = 1
                    season_B = 2
                wavelet_map_A = dgraded_mask * self.load_wavelet_coeff_map(a,j,info,season=season_A) 
                smoothed_maps_A[a] = dgraded_mask*hp.sphtfunc.smoothing(wavelet_map_A, FWHM_pix[j])*fskyinv
                unsmoothed_maps_A[a] = wavelet_map_A
                if info.cross_ILC:
                    wavelet_map_B = dgraded_mask * self.load_wavelet_coeff_map(a,j,info,season=season_B) 
                    smoothed_maps_B[a] = hp.sphtfunc.smoothing(wavelet_map_B, FWHM_pix[j])
                    unsmoothed_maps_B[a] = wavelet_map_B

        for a in range(info.N_freqs):
                    start_at = a
                    for b in range(start_at, info.N_freqs):
                        exists,cov_filename = self.load_covmap(a,b,j,info,query_exists = True)
                        if not exists:
                            if (self.freqs_to_use[j][a] == True) and (self.freqs_to_use[j][b] == True):
                                # read in wavelet coefficient maps constructed in previous step above
                                wavelet_map_A = unsmoothed_maps_A[a]
                                wavelet_map_A_smoothed = smoothed_maps_A[a]
                                if info.cross_ILC: 
                                    wavelet_map_B = unsmoothed_maps_B[b]
                                    wavelet_map_B_smoothed = smoothed_maps_B[b]
                                else:
                                    wavelet_map_B = unsmoothed_maps_A[b]
                                    wavelet_map_B_smoothed = smoothed_maps_A[b]

                                assert len(wavelet_map_A) == len(wavelet_map_B), "cov mat map calculation: wavelet coefficient maps have different N_side"
                                if not info.cross_ILC:
                                    cov_map_temp = hp.sphtfunc.smoothing( (wavelet_map_A - wavelet_map_A_smoothed)*(wavelet_map_B - wavelet_map_B_smoothed) , FWHM_pix[j])
                                else:
                                    cov_map_temp_AB = (hp.sphtfunc.smoothing( (wavelet_map_A - wavelet_map_A_smoothed)*(wavelet_map_B - wavelet_map_B_smoothed) , FWHM_pix[j]))
                                    cov_map_temp_BA = (hp.sphtfunc.smoothing( (wavelet_map_B - wavelet_map_B_smoothed)*(wavelet_map_A - wavelet_map_A_smoothed) , FWHM_pix[j]))

                                    cov_map_temp = 0.5*(cov_map_temp_AB+ cov_map_temp_BA)


                                cov_maps_temp.append( cov_map_temp  * fskyinv)

                                self.save_covmap(a,b,j,info,cov_map_temp,)
                        else:
                            if (self.freqs_to_use[j][a] == True) and (self.freqs_to_use[j][b] == True):

                                covmat_loaded = self.load_covmap(a,b,j,info)
                                cov_maps_temp.append(covmat_loaded * fskyinv)

        print('done computing all covariance maps at scale'+str(j),flush=True)
        return cov_maps_temp


    def weights_from_covmat_at_scale_j(self,info,j,cov_maps_temp,A_mix,resp_tol):

        subtract_comp = 0
        subtract_columns = []
        for ncomp in range(A_mix.shape[1]):
            if np.all(A_mix[:,ncomp]==0):
                assert not ncomp == 0
                subtract_columns += [ncomp]
                subtract_comp += 1
                
        A_mix = np.delete(A_mix,tuple(subtract_columns),axis=1)


        if type(info.N_deproj) is int:
            N_deproj = info.N_deproj
        else:
            N_deproj = info.N_deproj[j]
        N_deproj = N_deproj - subtract_comp

        N_comps = (N_deproj + 1)
        for a in range(info.N_freqs):
            if (self.freqs_to_use[j][a] == True):
                a_min = a
                break
                
        # get the mask that we don't care about computing outside of:
        #todo: this needs to be removed, and the hardcoding of this needs to be removed
        t1=time.time()
        if info.mask_before_covariance_computation is not None:

            dgraded_mask = hp.ud_grade(info.mask_before_covariance_computation,self.N_side_to_use[j])
            dgraded_mask[dgraded_mask!=0]=1
            if np.sum(dgraded_mask)== 0:
                assert self.N_side_to_use[j] < 256
                mask1 = hp.ud_grade(info.mask_before_covariance_computation,256)
                dgraded_mask = hp.ud_grade(mask1, self.N_side_to_use[j])
                dgraded_mask[dgraded_mask!=0]=1
                assert np.sum(dgraded_mask)>0
            dgraded_mask = dgraded_mask.astype(bool)
            len_unmasked_pix = np.sum(dgraded_mask)
            apply_mask = True
            print("did all the mask stuff in",time.time()-t1,flush=True)
        else:
            len_unmasked_pix = self.N_pix_to_use[j]
            apply_mask = False

        covmat_temp_sliced = np.zeros((int(self.N_freqs_to_use[j]),int(self.N_freqs_to_use[j]), len_unmasked_pix))
        count=0
        for a in range(np.sum(self.freqs_to_use[j])):
            for b in range(a, np.sum(self.freqs_to_use[j])):
              #  if (self.freqs_to_use[j][a] == True) and (self.freqs_to_use[j][b] == True):
                    # cov_maps_temp is in order 00, 01, 02, ..., 0(N_freqs_to_use[j]-1), 11, 12, ..., 1(N_freqs_to_use[j]-1), 22, 23, ...
                    if apply_mask:
                        covmat_temp_sliced[a,b] = cov_maps_temp[count][dgraded_mask] #by construction we're going through inv_cov_maps_temp in the same order as it was populated when computed (see below)
                    else:
                        covmat_temp_sliced[a,b] = cov_maps_temp[count]
                    if a != b:
                        covmat_temp_sliced[b,a] = covmat_temp_sliced[a,b] #symmetrize
                    count+=1

        covmat_temp_transpose = np.transpose(covmat_temp_sliced,(2,1,0))
        del(covmat_temp_sliced)

        tmp1 = np.zeros((N_comps,self.N_freqs_to_use[j],len_unmasked_pix))
        t1=time.time()
        if not info.use_numba:
            tmp1 = np.linalg.solve(covmat_temp_transpose,A_mix[None,:,:]) 
        else:
            tmp1 = my_numba_solver(covmat_temp_transpose,A_mix[:,:]) 
        if info.print_timing:
            print("got tmp1 in ",time.time()-t1,flush=True)
        tmp1 = np.transpose(tmp1)

        ### construct the matrix Q_{alpha beta} defined in Eq. 30 of McCarthy & Hill 2023 for each pixel at this wavelet scale and evaluate Eq. 29 to get weights ###
        t1=time.time()
        if not info.use_numba:
            Qab_pix = np.einsum('ajp,bj->abp', tmp1, np.transpose(A_mix))
        else:
            Qab_pix = np.transpose(numba_matmul(np.transpose(tmp1,(2,0,1)),A_mix),(1,2,0))
        if info.print_timing:
            print("got Qab in",time.time()-t1,flush=True)

        t1=time.time()
        tempvec = np.zeros((N_comps,len_unmasked_pix))
        # treat the no-deprojection case separately, since QSa_temp is empty in this case
        if (N_comps == 1):
            tempvec[0] = np.ones(len_unmasked_pix)
        else:
            for a in range(N_comps):
                QSa_temp = np.delete(np.delete(Qab_pix, a, 0), 0, 1) #remove the a^th row and zero^th column
                if not info.use_numba:
                    tempvec[a] = (-1.0)**float(a) * np.linalg.det(np.transpose(QSa_temp,(2,0,1)))
                else:
                     tempvec[a] = (-1.0)**float(a) * numba_det(np.transpose(QSa_temp,(2,0,1)))
        if info.print_timing:
            print("got tempvec in",time.time()-t1,flush=True)

        t1=time.time()
        tmp2 = np.einsum('ia,ap->ip', A_mix, tempvec) #todo: paralelize with numba? this is not very long though
        if info.print_timing:
            print("got tmp2 in",time.time()-t1,flush=True)
        t1=time.time()
        if not info.use_numba:
            tmp3 =  np.transpose(np.linalg.solve(covmat_temp_transpose,np.transpose(tmp2)))
        else:
            tmp3 =  np.transpose(my_numba_solver_parallelb(covmat_temp_transpose,np.transpose(tmp2)))
        del(covmat_temp_transpose)
        if info.print_timing: 
            print("got tmp3 in",time.time()-t1,flush=True)


        t1=time.time()
        if not info.use_numba:
             weights_sliced = 1.0/np.linalg.det(np.transpose(Qab_pix,(2,0,1)))[:,None]*np.transpose(tmp3) 
        else:
            weights_sliced = 1.0/numba_det(np.transpose(Qab_pix,(2,0,1)))[:,None]*np.transpose(tmp3) 
        if info.print_timing:
             print("got weights in",time.time()-t1,flush=True)

        t1=time.time()
        # response verification
        response = np.einsum('pi,ia->ap', weights_sliced, A_mix) #dimensions N_comps x N_pix_to_use[j]
        optimal_response_preserved_comp = np.ones(len_unmasked_pix)#preserved component, want response=1
        optimal_response_deproj_comp = np.zeros((N_comps-1, len_unmasked_pix)) #deprojected components, want response=0

        if not (np.absolute(response[0,]-optimal_response_preserved_comp) < resp_tol).all():
            print(f'preserved component response failed at wavelet scale {j}',str(np.max(np.absolute(response[0,]-optimal_response_preserved_comp)))+'>'+str(resp_tol),flush=True)
            quit()
        if response.shape[0]>1:
            if not (np.absolute(response[1:,]-optimal_response_deproj_comp) < resp_tol).all():
                print(f'deprojected component response failed at wavelet scale {j}',str(np.max(np.absolute(response[1:,]-optimal_response_deproj_comp) ))+'>'+str(resp_tol),flush=True)
                quit()
        if info.print_timing:
            print("done response verification in",time.time()-t1,flush=True)

        if apply_mask:
            weights = np.zeros((int(self.N_pix_to_use[j]),int(self.N_freqs_to_use[j])))
            weights[dgraded_mask] = weights_sliced
            return weights #N.B. 'weights' here only includes channels that passed beam_thresh criterion,
        else:
            return weights_sliced #N.B. 'weights' here only includes channels that passed beam_thresh criterion,


    def save_weights_fits(self,weights,j,info,map_images = False):
            count=0
            for a in range(info.N_freqs):
                if (self.freqs_to_use[j][a] == True):
                    weight_filename = _weights_filename(info,a,j)
                    hp.write_map(weight_filename, weights[:,count], nest=False, dtype=np.float64, overwrite=False)
                    if map_images == True: #save images if requested
                        plt.clf()
                        hp.mollview(weights[:,count], unit="1/K", title="Needlet ILC Weight Map, Frequency "+str(a)+" Scale "+str(j))
                        if N_deproj == 0:
                            plt.savefig(info.output_dir+info.output_prefix+'_needletILCweightmap_freq'+str(a)+'_scale'+str(j)+'_component_'+info.ILC_preserved_comp+'_crossILC'*info.cross_ILC+info.output_suffix+'.pdf')
                        else:
                            plt.savefig(info.output_dir+info.output_prefix+'_needletILCweightmap_freq'+str(a)+'_scale'+str(j)+'_component_'+info.ILC_preserved_comp+'_deproject_'+'_'.join(ILC_deproj_comps)+'_crossILC'*info.cross_ILC+info.output_suffix+'.pdf')

                    count+=1

    def save_weights_hdf5(self,weights,j,info):

        weight_filename = info.weight_filename_hdf5
        f = h5py.File(weight_filename,'a')
        count=0
        for a in range(info.N_freqs):
            if (self.freqs_to_use[j][a] == True):
                f.create_dataset("freq"+str(a)+'_scale'+str(j),weights[:,count].shape,data=weights[:,count])   

                count+=1
        f.close()


    def save_weights_at_scale_j(self,weights,j,info,map_images=False):
        if info.save_as_fits:
            self.save_weights_fits(weights,j,info,map_images=map_images)
        elif info.save_as_hdf5:
            self.save_weights_hdf5(weights,j,info)


    def compute_weights_at_scale_j_from_covmat(self,j,info,resp_tol,map_images = False):

            # Computes the ILC weights at scale j without ever computing the invcovmat (using np.linalg.solve instead of np.linalg.inv)

            t1=time.time()

            if type(info.N_deproj) is int:
                N_deproj = info.N_deproj
                if N_deproj>0:
                    ILC_deproj_comps = info.ILC_deproj_comps
            else:
                N_deproj = info.N_deproj[j]
                if N_deproj>0:
                    ILC_deproj_comps = info.ILC_deproj_comps[j]

            # Get the mixing matrix (the matrix of SEDs)
            A_mix = self.mixing_matrix_at_scale_j(j,info)

            ##############################
            ##############################
            # for each filter scale, compute (or load) maps of the smoothed real-space frequency-frequency covariance matrix using the Gaussian FWHMs
            cov_maps_temp = []
            flag=True
            # First we try to load the covmats
            for a in range(info.N_freqs):
                    start_at = a
                    for b in range(start_at, info.N_freqs):
                        if (self.freqs_to_use[j][a] == True) and (self.freqs_to_use[j][b] == True and flag==True):
                            exists,cov_filename = self.load_covmap(a,b,j,info,query_exists = True)
                            if exists:
                                cov_maps_temp.append(  self.load_covmap(a,b,j,info))
                            else:
                                print('needlet coefficient covariance map not previously computed; computing all covariance maps at scale '+str(j)+' now...',flush=True)
                                flag=False
                                break

            # If they didn't exist, we calculate them:
            if flag == False:
                    # Calculate the covmats
                    cov_maps_temp = self.compute_covariance_at_scale_j(info,j,self.FWHM_pix)
            else:
                print('needlet coefficient covariance map already exists at scale',j,":", cov_filename,flush=True)

            if info.print_timing:
                print("got covmats in",time.time()-t1,flush=True)    

            # compute the weights
            t1=time.time()
            weights = self.weights_from_covmat_at_scale_j(info,j,cov_maps_temp,A_mix,resp_tol)
            if info.print_timing:
                print("calcweights in",time.time()-t1,flush=True)

            del cov_maps_temp #free up memory
            print('done computing all ILC weights at scale '+str(j),flush=True)

            ##########################
            # only save these maps of the ILC weights if requested
            if (info.save_weights == 'yes' or info.save_weights == 'Yes' or info.save_weights == 'YES'):
                self.save_weights_at_scale_j(weights,j,info,map_images=map_images)
            return weights

    def compute_weights_at_scale_j_from_invcovmat(self,j,info,resp_tol,map_images = False):
            print("computin gfrom invcov",flush=True)
            # Computes the ILC weights at scale j  by computing the invcovmat (using np.linalg.inv and np.matmul instead of np.linalg.solve)

            t1=time.time()

            if type(info.N_deproj) is int:
                N_deproj = info.N_deproj
                if N_deproj>0:
                    ILC_deproj_comps = info.ILC_deproj_comps
            else:
                N_deproj = info.N_deproj[j]
                if N_deproj>0:
                    ILC_deproj_comps = info.ILC_deproj_comps[j]

            # Get the mixing matrix (the matrix of SEDs)
            A_mix = self.mixing_matrix_at_scale_j(j,info)

            ##############################
            ##############################
            # for each filter scale, compute (or load) maps of the smoothed real-space frequency-frequency inverse covariance matrix using the Gaussian FWHMs
            cov_maps_temp = []
            flag=True
            # First we try to load the covmats
            t1=time.time()
            inv_cov_maps_temp = []
            for a in range(info.N_freqs):
                    start_at = a
                    #if info.cross_ILC:
                     #   start_at = 0
                    for b in range(start_at, info.N_freqs):
                        if (self.freqs_to_use[j][a] == True) and (self.freqs_to_use[j][b] == True and flag==True):
                            exists,inv_cov_filename = self.load_invcovmap(a,b,j,info,query_exists = True)
                            if exists:
                                inv_cov_maps_temp.append(  self.load_invcovmap(a,b,j,info))
                            else:
                                print(inv_cov_filename,'does not exist',flush=True)
                                print('needlet coefficient inverse covariance map not previously computed; computing all inverse covariance maps at scale '+str(j)+' now...',flush=True)
                                flag=False
                                break
            # If they didn't exist, we calculate them:
            if flag == False:
                    # Calculate the covmats
                    inv_cov_maps_temp = self.compute_inv_covariance_at_scale_j(info,j,self.FWHM_pix)
            else:
                print('needlet coefficient inverse covariance map already exists at scale',j,":", inv_cov_filename,flush=True)

            print("got inv covmats in",time.time()-t1,flush=True)

            # compute the weights
            t1=time.time()
            weights = self.weights_from_invcovmat_at_scale_j(info,j,inv_cov_maps_temp,A_mix,resp_tol)
            print("calcweights in",time.time()-t1,flush=True)

            del inv_cov_maps_temp #free up memory
            print('done computing all ILC weights at scale '+str(j),flush=True)

            ##########################
            # only save these maps of the ILC weights if requested
            if (info.save_weights == 'yes' or info.save_weights == 'Yes' or info.save_weights == 'YES'):
                self.save_weights_at_scale_j(weights,j,info)
            return weights


    def load_weights_at_scale_j_fits(self,info,j,query_exists=False):
            ## Loads and returns the precomputed weights at scale j
            weights = np.zeros((int(self.N_pix_to_use[j]),int(self.N_freqs_to_use[j])))
            count=0
            flag = True
            for a in range(info.N_freqs):
                if (self.freqs_to_use[j][a] == True):
                    weight_filename = _weights_filename(info,a,j)
                    exists = os.path.isfile(weight_filename)
                    if not query_exists:
                          weights[:,count] = hp.read_map(weight_filename, dtype=np.float64, )
                    else:
                        if not exists:
                            return False
                    count+=1
            if query_exists:
                return True
            return weights

    def load_weights_at_scale_j_hdf5(self,info,j,query_exists=False):
        weight_filename = info.weight_filename_hdf5
        weights = np.zeros((int(self.N_pix_to_use[j]),int(self.N_freqs_to_use[j])))
        exists = os.path.isfile(weight_filename)
        if query_exists:
            if not exists:
                print("file",weight_filename,"does not exist",flush=True)
                return False
        else:
            assert exists

        f = h5py.File(weight_filename,'r')
        count=0
        flag = True
        for a in range(info.N_freqs):
            if (self.freqs_to_use[j][a] == True):
                if not query_exists:
                    weights[:,count] =  f["freq"+str(a)+'_scale'+str(j)][()]
                else:
                    if "freq"+str(a)+'_scale'+str(j) not in f.keys():
                        flag = False
                count+=1
        f.close()
        if query_exists:
            if False:
                print('Weights do not exist at stale',j)
            return flag
        return weights


    def load_weights_at_scale_j(self,info,j,query_exists = False):
        if info.save_as_fits:
            return self.load_weights_at_scale_j_fits(info,j,query_exists=query_exists)
        elif info.save_as_hdf5:
            return self.load_weights_at_scale_j_hdf5(info,j,query_exists=query_exists)

    def fits_load_wavelet_coeff_map(self,frequency,scale,info,season=None,query_exists = False):
        filename_wavelet_coeff_map = _needletcoeffmap_filename(info,frequency,scale,season=season)

        exists = os.path.isfile(filename_wavelet_coeff_map)

        if query_exists:
            return exists,filename_wavelet_coeff_map
        elif exists:
            return hp.read_map(filename_wavelet_coeff_map, dtype=np.float64)

    def hdf5_load_wavelet_coeff_map(self,frequency,scale,info,season=None,query_exists = False):

        filename_wavelet_coeff_dataset = info.wavelet_coeff_hdf5_filename

        if query_exists:
            if not os.path.isfile(filename_wavelet_coeff_dataset):
                exists = False
            else:
                f = h5py.File(filename_wavelet_coeff_dataset,'r')
                if season is None:
                    keyy = "freq"+str(frequency)+'_scale'+str(scale)
                else:
                    keyy = "freq"+str(frequency)+'_scale'+str(scale) +'_S'+str(season)
                if keyy not in f.keys():
                    exists = False
                else:
                    exists = True
                f.close()
            return exists,filename_wavelet_coeff_dataset

        else:
            f = h5py.File(filename_wavelet_coeff_dataset,'r')

            if season is None:
                    keyy = "freq"+str(frequency)+'_scale'+str(scale)
            else:
                    keyy = "freq"+str(frequency)+'_scale'+str(scale) +'_S'+str(season)


            wavelet_coeff_map =f[keyy][()]

            f.close()

            return wavelet_coeff_map

    def load_wavelet_coeff_map(self,frequency,scale,info,season=None,query_exists = False):
            if info.save_as_fits:

                return self.fits_load_wavelet_coeff_map(frequency,scale,info,season=season,query_exists = query_exists)

            elif info.save_as_hdf5:

                return self.hdf5_load_wavelet_coeff_map(frequency,scale,info,season=season,query_exists = query_exists)

    def save_wavelet_coeff_map(self,frequency,scale,info,wavelet_coeff_map,season=None):

        if info.save_as_fits:

            filename_wavelet_coeff_map = _needletcoeffmap_filename(info,frequency,scale,season=season)

            hp.write_map(filename_wavelet_coeff_map, wavelet_coeff_map,nest=False, dtype=np.float64, overwrite=False)

        elif info.save_as_hdf5:

            filename_wavelet_coeff_dataset =  info.wavelet_coeff_hdf5_filename

            f = h5py.File(filename_wavelet_coeff_dataset,'a')

            if season is None:

                f.create_dataset("freq"+str(frequency)+'_scale'+str(scale),wavelet_coeff_map.shape,data=wavelet_coeff_map)
            else:
                f.create_dataset("freq"+str(frequency)+'_scale'+str(scale)+"_S"+str(season),wavelet_coeff_map.shape,data=wavelet_coeff_map)


            f.close()

    def fits_load_covmap(self,frequency1,frequency2,scale,info,query_exists = False):
        filename = _cov_filename(info,frequency1,frequency2,scale)
        exists = os.path.isfile(filename)
        if query_exists:
            return exists,filename
        elif exists:
            return hp.read_map(filename, dtype=np.float64)

    def hdf5_load_covmap(self,frequency1,frequency2,scale,info,query_exists = False):
        filename_covmaps_dataset = info.covmaps_hdf5_filename

        if query_exists:

            if not os.path.isfile(filename_covmaps_dataset):
                exists = False
            else:
                f = h5py.File(filename_covmaps_dataset,'r')
                if "freq"+str(frequency1)+'_'+str(frequency2)+'_scale'+str(scale) not in f.keys():
                    exists = False
                else:
                    exists = True
                f.close()
            return exists,filename_covmaps_dataset

        else:
            f = h5py.File(filename_covmaps_dataset,'r')
            covmap = f["freq"+str(frequency1)+'_'+str(frequency2)+'_scale'+str(scale)][()]

            f.close()
            return covmap

    def load_covmap(self,frequency1,frequency2,scale,info,query_exists = False):

        if info.save_as_fits:

            return self.fits_load_covmap(frequency1,frequency2,scale,info,query_exists = query_exists)

        elif info.save_as_hdf5:

            return self.hdf5_load_covmap(frequency1,frequency2,scale,info,query_exists = query_exists)

    def save_covmap(self,frequency1,frequency2,scale,info,covmap,query_exists = False):

        if info.save_as_fits:

            filename = _cov_filename(info,frequency1,frequency2,scale)

            hp.write_map(filename, covmap,nest=False, dtype=np.float64, overwrite=False)

        elif info.save_as_hdf5:

            filename_covmaps_dataset = info.covmaps_hdf5_filename

            f = h5py.File(filename_covmaps_dataset,'a')

            f.create_dataset("freq"+str(frequency1)+'_'+str(frequency2)+'_scale'+str(scale),covmap.shape,data=covmap)

            f.close()

    def fits_load_invcovmap(self,frequency1,frequency2,scale,info,query_exists = False):
        filename = _inv_cov_filename(info,frequency1,frequency2,scale)
        exists = os.path.isfile(filename)
        if query_exists:
            return exists,filename
        elif exists:
            return hp.read_map(filename, dtype=np.float64)

    def hdf5_load_invcovmap(self,frequency1,frequency2,scale,info,query_exists = False):

        filename_invcovmaps_dataset = info.invcovmaps_hdf5_filename

        if query_exists:

            if not os.path.isfile(filename_invcovmaps_dataset):
                exists = False
            else:
                f = h5py.File(filename_invcovmaps_dataset,'r')
                if "freq"+str(frequency1)+'_'+str(frequency2)+'_scale'+str(scale) not in f.keys():
                    exists = False
                else:
                    exists = True
                f.close()
            return exists,filename_invcovmaps_dataset

        else:
            f = h5py.File(filename_invcovmaps_dataset,'r')
            invcovmap = f["freq"+str(frequency1)+'_'+str(frequency2)+'_scale'+str(scale)][()]

            f.close()
            return invcovmap


    def load_invcovmap(self,frequency1,frequency2,scale,info,query_exists = False):

        if info.save_as_fits:

            return self.fits_load_invcovmap(frequency1,frequency2,scale,info,query_exists = query_exists)

        elif info.save_as_hdf5:

            return self.hdf5_load_invcovmap(frequency1,frequency2,scale,info,query_exists = query_exists)

    def save_invcovmap(self,frequency1,frequency2,scale,info,invcovmap,query_exists = False):

        if info.save_as_fits:

            filename = _inv_cov_filename(info,frequency1,frequency2,scale)

            hp.write_map(filename, invcovmap,nest=False, dtype=np.float64, overwrite=False)

        elif info.save_as_hdf5:

            filename_invcovmaps_dataset = info.invcovmaps_hdf5_filename

            f = h5py.File(filename_invcovmaps_dataset,'a')

            f.create_dataset("freq"+str(frequency1)+'_'+str(frequency2)+'_scale'+str(scale),invcovmap.shape,data=invcovmap)

            f.close()

    def ILC_at_scale_j(self,j,info,resp_tol,map_images=False,maps_for_weights_needlets=None):
        ## Performs the full ILC at scale j

        if type(info.N_deproj) is int:
            N_deproj = info.N_deproj
            if N_deproj>0:
                ILC_deproj_comps = info.ILC_deproj_comps
        else:
            N_deproj = info.N_deproj[j]
            if N_deproj>0:
                ILC_deproj_comps = info.ILC_deproj_comps[j]

        # first, check if the weights already exist, and skip everything if so
        weights_exist = True
        count=0

        weights_exist = self.load_weights_at_scale_j(info,j,query_exists = True)
        if (weights_exist == False):
            print("weights do not exist; recalculating weights at scale",j)
            weights = self.compute_weights_at_scale_j_from_covmat(j,info,resp_tol,map_images = map_images)
        else:
            print("weights exist; loading weights at scale",j)
            weights = self.load_weights_at_scale_j(info,j)


        ##########################
        # apply these ILC weights to the needlet coefficient maps to get the per-needlet-scale ILC maps
        t1=time.time()
        ILC_map_temp = np.zeros(int(self.N_pix_to_use[j]))
        count=0
        for a in range(info.N_freqs):
            if (self.freqs_to_use[j][a] == True):
                if not info.apply_weights_to_other_maps:
                    wavelet_coeff_map = self.load_wavelet_coeff_map(a,j,info)
                else:
                    wavelet_coeff_map = maps_for_weights_needlets[a][j]
                if info.subtract_means_before_sums[j]:
                    dgraded_mask = hp.ud_grade(info.mask_before_covariance_computation,self.N_side_to_use[j])
                    dgraded_mask[dgraded_mask!=0]=1
                    print("fsky at scale "+str(j)+" is "+str(np.sum(dgraded_mask)/dgraded_mask.shape[0]),flush=True)
                    if np.sum(dgraded_mask)== 0:
                         assert self.N_side_to_use[j] < 256
                         mask1 = hp.ud_grade(info.mask_before_covariance_computation,256)
                         dgraded_mask = hp.ud_grade(mask1, self.N_side_to_use[j])
                         dgraded_mask[dgraded_mask!=0]=1
                         print("fsky at scale "+str(j)+" is "+str(np.sum(dgraded_mask)/dgraded_mask.shape[0]),flush=True)
                    assert np.sum(dgraded_mask)>0
                    fsky = hp.smoothing(dgraded_mask,self.FWHM_pix[j])
                    fsky[fsky==0] = 1e-100
                    wavelet_coeff_map_minus_smoothed = wavelet_coeff_map - hp.smoothing(dgraded_mask * wavelet_coeff_map,self.FWHM_pix[j])/fsky
                    ILC_map_temp += weights[:,count] * (wavelet_coeff_map_minus_smoothed)
                else:
                    ILC_map_temp += weights[:,count] * wavelet_coeff_map
                count+=1
        if info.print_timing:
            print("finished summing at scale",j,time.time()-t1,flush=True)
        return ILC_map_temp



# apply wavelet transform (i.e., filters) to a map
def waveletize(inp_map=None, wv=None, rebeam=False, inp_beam=None, new_beam=None, wv_filts_to_use=None, N_side_to_use=None):
    assert inp_map is not None, "no input map specified"
    N_pix = len(inp_map)
    N_side_inp = hp.npix2nside(N_pix)
    assert wv is not None, "wavelets not defined"
    assert type(wv) is Wavelets, "Wavelets TypeError"
    assert wv.ELLMAX < 3*N_side_inp-1, "ELLMAX too high"
    if (N_side_to_use is None):
        N_side_to_use = np.ones(wv.N_scales, dtype=int)*N_side_inp
    if(rebeam):
        assert inp_beam is not None, "no input beam defined"
        assert new_beam is not None, "no new beam defined"
        assert len(inp_beam) == len(new_beam), "input and new beams have different ell_max"
        assert inp_beam[0][0] == 0 and new_beam[0][0] == 0, "beam profiles must start at ell=0"
        assert inp_beam[-1][0] == wv.ELLMAX and new_beam[-1][0] == wv.ELLMAX, "beam profiles must end at ELLMAX"
        beam_fac = new_beam[:,1]/inp_beam[:,1]
    else:
        beam_fac = np.ones(wv.ELLMAX+1,dtype=float)
    if(wv.taper_width):
        assert wv.ELLMAX - wv.taper_width > 10., "desired taper is too broad for given ELLMAX"
        taper_func = (1.0 - 0.5*(np.tanh(0.025*(wv.ell - (wv.ELLMAX - wv.taper_width))) + 1.0)) #smooth taper to zero from ELLMAX-taper_width to ELLMAX
    else:
        taper_func = np.ones(wv.ELLMAX+1,dtype=float)
    # convert map to alm, apply wavelet filters (and taper and rebeam)
    inp_map_alm = hp.map2alm(inp_map, lmax=wv.ELLMAX)
    wv_maps = []
    if wv_filts_to_use is not None: #allow user to only output maps for some of the filter scales
        assert len(wv_filts_to_use) == wv.N_scales, "wv_filts_to_use has wrong shape"
        assert len(N_side_to_use) == wv.N_scales, "N_side_to_use has wrong shape"
        for j in range(wv.N_scales):
            assert N_side_to_use[j] <= N_side_inp, "N_side_to_use > N_side_inp"
            if wv_filts_to_use[j] == True:
                wv_maps.append( hp.alm2map( hp.almxfl( inp_map_alm, (wv.filters[j])*taper_func*beam_fac), nside=N_side_to_use[j]) )
    else:
        assert len(N_side_to_use) == wv.N_scales, "N_side_to_use has wrong shape"
        for j in range(wv.N_scales):
            assert N_side_to_use[j] <= N_side_inp, "N_side_to_use > N_side_inp"
            wv_maps.append( hp.alm2map( hp.almxfl( inp_map_alm, (wv.filters[j])*taper_func*beam_fac), nside=N_side_to_use[j]) )
    # return maps of wavelet coefficients (i.e., filtered maps) -- N.B. each one can have a different N_side
    return wv_maps




# Find nth wavelet coefficients of a map
def find_nth_wavelet_coefficient(scale,inp_map=None,inp_map_alm=None, wv=None, rebeam=False, inp_beam=None, new_beam=None, wv_filts_to_use=None, N_side_to_use=None):
    assert inp_map is not None, "no input alms specified"
    N_pix = len(inp_map)
    N_side_inp = hp.npix2nside(N_pix)
    assert wv is not None, "wavelets not defined"
    assert type(wv) is Wavelets, "Wavelets TypeError"
    assert wv.ELLMAX < 3*N_side_inp-1, "ELLMAX too high"
    assert wv.ELLMAX > 200, "ELLMAX too low"
    if (N_side_to_use is None):
        N_side_to_use = np.ones(wv.N_scales, dtype=int)*N_side_inp
    if(rebeam):
        assert inp_beam is not None, "no input beam defined"
        assert new_beam is not None, "no new beam defined"
        assert len(inp_beam) == len(new_beam), "input and new beams have different ell_max"
        assert inp_beam[0][0] == 0 and new_beam[0][0] == 0, "beam profiles must start at ell=0"
        assert inp_beam[-1][0] == wv.ELLMAX and new_beam[-1][0] == wv.ELLMAX, "beam profiles must end at ELLMAX"
        beam_fac = new_beam[:,1]/inp_beam[:,1]
    else:
        beam_fac = np.ones(wv.ELLMAX+1,dtype=float)
    if(wv.taper_width):
        assert wv.ELLMAX - wv.taper_width > 10., "desired taper is too broad for given ELLMAX"
        taper_func = (1.0 - 0.5*(np.tanh(0.025*(wv.ell - (wv.ELLMAX - wv.taper_width))) + 1.0)) #smooth taper to zero from ELLMAX-taper_width to ELLMAX
    else:
        taper_func = np.ones(wv.ELLMAX+1,dtype=float)
    # convert map to alm, apply wavelet filters (and taper and rebeam)
    if inp_map_alm is not None:
        inp_map_alm = hp.map2alm(inp_map, lmax=wv.ELLMAX)
    wv_maps = []
    if wv_filts_to_use is not None: #allow user to only output maps for some of the filter scales
        assert len(wv_filts_to_use) == wv.N_scales, "wv_filts_to_use has wrong shape"
        assert len(N_side_to_use) == wv.N_scales, "N_side_to_use has wrong shape"
        for j in [scale]:
            assert N_side_to_use[j] <= N_side_inp, "N_side_to_use > N_side_inp"
            if wv_filts_to_use[j] == True:
                wv_maps.append( hp.alm2map( hp.almxfl( inp_map_alm, (wv.filters[j])*taper_func*beam_fac), nside=N_side_to_use[j]) )
    else:
        assert len(N_side_to_use) == wv.N_scales, "N_side_to_use has wrong shape"
        for j in [scale]:
            assert N_side_to_use[j] <= N_side_inp, "N_side_to_use > N_side_inp"
            wv_maps.append( hp.alm2map( hp.almxfl( inp_map_alm, (wv.filters[j])*taper_func*beam_fac), nside=N_side_to_use[j]) )
    # return maps of wavelet coefficients (i.e., filtered maps) -- N.B. each one can have a different N_side
    return wv_maps[0]

# synthesize map from wavelet coefficients
# (N.B. power will be lost near ELLMAX if a taper has been applied in waveletize)
# note that you will get nonsense if different wavelets are used here than in waveletize (obviously)
def synthesize(wv_maps=None, wv=None, N_side_out=None):
    assert wv is not None, "wavelets not defined"
    assert type(wv) is Wavelets, "Wavelets TypeError"
    assert N_side_out is not None, "N_side_out must be specified"
    assert N_side_out > 0, "N_side_out must be positive"
    assert hp.pixelfunc.isnsideok(N_side_out, nest=True), "invalid N_side_out"
    assert wv.ELLMAX < 3*N_side_out-1, "ELLMAX too high"
    N_pix_out = 12*N_side_out**2
    out_map = np.zeros(N_pix_out)
    for j in range(wv.N_scales):
        N_pix_temp = len(wv_maps[j])
        N_side_temp = hp.npix2nside(N_pix_temp)
        temp_alm = hp.map2alm(wv_maps[j], lmax=np.amin(np.array([wv.ELLMAX, 3*N_side_temp-1])))
        if (3*N_side_temp-1 < wv.ELLMAX):
            temp_alm_filt = hp.almxfl(temp_alm, (wv.filters[j])[:3*N_side_temp])
        else:
            temp_alm_filt = hp.almxfl(temp_alm, wv.filters[j])
        out_map += hp.alm2map(temp_alm_filt, nside=N_side_out)
    return out_map

def waveletize_input_maps(info,scale_info_wvs,wv,map_images = False):
        ##########################
        # compute wavelet decomposition of all frequency maps used at each filter scale
        # save the filtered maps (aka maps of "wavelet coefficients")
        # remember to re-convolve all maps to the highest resolution map being used when passing into needlet filtering, or to the user-specified input beam at which to compute the ILC
        # possibly make this a routine of the object scale_info_wvs ?

        freqs_to_use = scale_info_wvs.freqs_to_use
        N_freqs_to_use = scale_info_wvs.N_freqs_to_use
        FWHM_pix = scale_info_wvs.FWHM_pix
        N_side_to_use = scale_info_wvs.N_side_to_use
        N_pix_to_use = scale_info_wvs.N_pix_to_use

        have_read_input_maps = False

        for i in range(info.N_freqs):
            # N.B. maps are assumed to be in strictly decreasing order of FWHM! i.e. info.beams[-1] is highest-resolution beam
            wv_maps_temp = []
            flag=True
            for j in range(wv.N_scales):
                if freqs_to_use[j][i] == True:
                    exists,filename = scale_info_wvs.load_wavelet_coeff_map(i,j,info,query_exists = True)
                    if exists:
                        #print('needlet coefficient map already exists at scale'+str(j)+' :', filename,flush=True)
                        wv_maps_temp.append( scale_info_wvs.load_wavelet_coeff_map(i,j,info))
                    else:
                        print('needlet coefficient map not previously computed; computing all maps for frequency '+str(i)+' now...',flush=True)
                        flag=False
                        break
            if flag == False:
                # if we have not read in the input files we need to do so
                if not have_read_input_maps:
                    info.read_maps()
                    have_read_input_maps = True
                print("waveletizing frequency ", i, "...",flush=True)
                if info.mask_before_wavelet_computation is not None:
                    mask = info.mask_before_wavelet_computation
                else:
                    mask = 1

                wv_maps_temp = waveletize(inp_map=mask*(info.maps)[i], wv=wv, rebeam=True, inp_beam=(info.beams)[i], new_beam=info.common_beam, wv_filts_to_use=freqs_to_use[:,i], N_side_to_use=N_side_to_use)
                for j in range(wv.N_scales):
                    if freqs_to_use[j][i] == True:
                        scale_info_wvs.save_wavelet_coeff_map(i,j,info,wv_maps_temp[j],season=None)
                print("done waveletizing frequency ", i, "...",flush=True)
            if map_images == True:
                for j in range(wv.N_scales):
                    if freqs_to_use[j][i] == True:
                        plt.clf()
                        hp.mollview(wv_maps_temp[j], unit="K", title="Needlet Coefficient Map, Frequency "+str(i)+" Scale "+str(j), min=np.mean(wv_maps_temp[j])-2*np.std(wv_maps_temp[j]), max=np.mean(wv_maps_temp[j])+2*np.std(wv_maps_temp[j]))
                        plt.savefig(info.output_dir+info.output_prefix+'_needletcoeffmap_freq'+str(i)+'_scale'+str(j)+'.pdf')
            if info.cross_ILC:
                print("doing crossILC waveletizing now")

                for season in [1,2]:
                    print("doing crossILC season",season,flush=True)
                    flag = True
                    wv_maps_temp = []
                    for j in range(wv.N_scales):
                        if freqs_to_use[j][i] == True:
                            exists,filename  = scale_info_wvs.load_wavelet_coeff_map(i,j,info,query_exists = True,season=season)
                            if exists:
                                print('needlet coefficient map already exists:', filename,flush=True)
                                wv_maps_temp.append( scale_info_wvs.load_wavelet_coeff_map(i,j,info,season=season))
                            else:
                                print('needlet coefficient map not previously computed; computing all season '+str(season)+' maps for frequency '+str(i)+' now...',flush=True)
                                flag=False
                                break
                    if flag == False:
                        if season==1:
                            maps = info.maps_s1
                        elif season==2:
                            maps = info.maps_s2
                        if info.perform_ILC_at_beam is not None:
                            newbeam = info.common_beam
                        else:
                            newbeam = (info.beams)[-1]
                        wv_maps_temp = waveletize(inp_map=(maps)[i], wv=wv, rebeam=True, inp_beam=(info.beams)[i], new_beam=newbeam, wv_filts_to_use=freqs_to_use[:,i], N_side_to_use=N_side_to_use)
                        for j in range(wv.N_scales):
                            if freqs_to_use[j][i] == True:
                                scale_info_wvs.save_wavelet_coeff_map(i,j,info,wv_maps_temp[j],season=season) #need to check if they exist first
                        del(maps) #free up memory
                    print("done waveletizing season"+str(season)+" maps at frequency" + str(i),flush=True)
            del wv_maps_temp #free up memory
            
def _needletcoeffmap_filename(info,freq,scale,season=None):

    if season is None:
        filename_wavelet_coeff_map = info.output_dir+info.output_prefix+'_needletcoeffmap_freq'+str(freq)+'_scale'+str(scale)+'.fits'
    else:
        assert season in [1,2]
        filename_wavelet_coeff_map = info.output_dir+info.output_prefix+'_needletcoeffmap_freq'+str(freq)+'_scale'+str(scale)+'_S'+str(season)+'.fits'

    return filename_wavelet_coeff_map


def _cov_filename(info,freq1,freq2,scale):

    a = freq1
    b = freq2
    j = scale

    cov_filename = info.output_dir+info.output_prefix+'_needletcoeff_covmap_freq'+str(a)+'_freq'+str(b)+'_scale'+str(j)+'_crossILC'*info.cross_ILC+'.fits'

    return cov_filename

def _ILC_scale_filename(info,j,scale_info_wvs):


    frequencies_included = ''
    for x in range(info.N_freqs):
        if scale_info_wvs.freqs_to_use[j][x]:
            frequencies_included += str(x)

    ILC_map_filename = info.output_dir+info.output_prefix+'needletILCmap_scale'+str(j)+'_component_'+info.ILC_preserved_comp+'_crossILC'*info.cross_ILC+'_includechannels'+frequencies_included+info.output_suffix+'.fits'
    if type(info.N_deproj) is int:
        if info.N_deproj>0:
            ILC_map_filename = info.output_dir+info.output_prefix+'needletILCmap_scale'+str(j)+'_component_'+info.ILC_preserved_comp+'_deproject_'+'_'.join(info.ILC_deproj_comps)+'_crossILC'*info.cross_ILC+'_includechannels'+frequencies_included+info.output_suffix+'.fits'
    else:
         if info.N_deproj[0]>0:
            # NOTE: ILCdeprojected file name is not so descriptive here. Need to describe it more in info.output_suffix.
            ILC_map_filename = info.output_dir+info.output_prefix+'needletILCmap_scale'+str(j)+'_component_'+info.ILC_preserved_comp+'_deproject_'+'_'.join(info.ILC_deproj_comps[0])+'_crossILC'*info.cross_ILC+'_includechannels'+frequencies_included+info.output_suffix+'.fits'

    return ILC_map_filename


def _weights_filename(info,freq,scale):
                a = freq
                j = scale
                weight_filename = info.output_dir+info.output_prefix+'weightmap_freq'+str(a)+'_scale'+str(j)+'_component_'+info.ILC_preserved_comp+'_crossILC'*info.cross_ILC+info.output_suffix_weights+'.fits'
                if type(info.N_deproj )is int:
                    if info.N_deproj>0:
                        weight_filename = info.output_dir+info.output_prefix+'weightmap_freq'+str(a)+'_scale'+str(j)+'_component_'+info.ILC_preserved_comp+'_deproject_'+'_'.join(info.ILC_deproj_comps)+'_crossILC'*info.cross_ILC+info.output_suffix_weights+'.fits'
                else:
                    if info.N_deproj[j]>0:
                        weight_filename = info.output_dir+info.output_prefix+'weightmap_freq'+str(a)+'_scale'+str(j)+'_component_'+info.ILC_preserved_comp+'_deproject_'+'_'.join(info.ILC_deproj_comps[j])+'_crossILC'*info.cross_ILC+info.output_suffix_weights+'.fits'
                '''
                if info.recompute_covmat_for_ndeproj:
                    if type(info.N_deproj) is int:
                        N_deproj = info.N_deproj
                    else:
                        N_deproj = info.N_deproj[j]
                    weight_filename = weight_filename[:-5] +'_Ndeproj'+str(N_deproj)+'.fits'
                '''
                weight_filename = weight_filename[:-5]+info.output_suffix+'.fits'
                return weight_filename

def _ILC_map_filename(info):
    ILC_map_filename = info.output_dir+info.output_prefix+'needletILCmap'+'_component_'+info.ILC_preserved_comp+'_crossILC'*info.cross_ILC+info.output_suffix+'.fits'
    if type(info.N_deproj) is int:
        if info.N_deproj>0:
            ILC_map_filename = info.output_dir+info.output_prefix+'needletILCmap'+'_component_'+info.ILC_preserved_comp+'_deproject_'+'_'.join(info.ILC_deproj_comps)+'_crossILC'*info.cross_ILC+info.output_suffix+'.fits'
    else:
         if info.N_deproj[0]>0:
            # NOTE: ILCdeprojected file name is not so descriptive here. Need to describe it more in info.output_suffix.
            ILC_map_filename = info.output_dir+info.output_prefix+'needletILCmap'+'_component_'+info.ILC_preserved_comp+'_deproject_'+'_'.join(info.ILC_deproj_comps[0])+'_crossILC'*info.cross_ILC+info.output_suffix+'.fits'

    return ILC_map_filename


def plot_ILC_map(info,ILC_map):
        plt.clf()
        hp.mollview(ILC_map, unit='dimensionless', title='Needlet ILC Map, Component '+info.ILC_preserved_comp)
        if type(info.N_deproj) is int:
            if info.N_deproj==0:
                plt.savefig(info.output_dir+info.output_prefix+'needletILCmap'+'_component_'+info.ILC_preserved_comp+'_crossILC'*info.cross_ILC+info.output_suffix+'.pdf')
            else:
                plt.savefig(info.output_dir+info.output_prefix+'needletILCmap'+'_component_'+info.ILC_preserved_comp+'_deproject_'+'_'.join(info.ILC_deproj_comps)+'_crossILC'*info.cross_ILC+info.output_suffix+'.pdf')
        else:
            if info.N_deproj[0]==0:
                plt.savefig(info.output_dir+info.output_prefix+'needletILCmap'+'_component_'+info.ILC_preserved_comp+'_crossILC'*info.cross_ILC+info.output_suffix+'.pdf')
            else:
                plt.savefig(info.output_dir+info.output_prefix+'needletILCmap'+'_component_'+info.ILC_preserved_comp+'_deproject_'+'_'.join(info.ILC_deproj_comps[0])+'_crossILC'*info.cross_ILC+info.output_suffix+'.pdf')


def wavelet_ILC(wv=None, info=None,  resp_tol=1.e-3, map_images=False,return_ILC_map=False):
    assert wv is not None, "wavelets not defined"
    assert type(wv) is Wavelets, "Wavelets TypeError"
    assert info is not None, "ILC info not defined"
    assert type(info) is ILCInfo, "ILCInfo TypeError"
    assert wv.N_scales == info.N_scales, "N_scales must match"
    assert wv.ELLMAX == info.ELLMAX, "ELLMAX must match"
    assert(info.wavelet_beam_criterion > 0. and info.wavelet_beam_criterion < 1.)
    assert info.N_side > 0, "N_side cannot be negative or zero"

    print("doing wavelet ILC",flush=True)

    ## Define the properties of the scales etc that we need, 
    scale_info_wvs = scale_info(wv,info,)

    if info.perform_ILC_at_beam is not None:
        newbeam = info.common_beam
    else:
        newbeam = (info.beams)[-1]

    # Calculate (or load) the wavelet coefficients of the input maps:
    if not info.wavelet_maps_exist:
        t1=time.time()
        waveletize_input_maps(info,scale_info_wvs,wv,map_images = map_images)
        if info.print_timing:
            print("waveletized in",time.time()-t1,flush=True)

    # If you want to apply the weights to OTHER maps, calculate their wavelet coefficients here
    if info.apply_weights_to_other_maps:
         maps_for_weights_needlets=[]
         info.read_maps()
         for i in range(info.N_freqs):
             print("waveletizing maps to apply weights", i,flush=True)
             maps_for_weights_needlets.append(waveletize(inp_map=(info.maps_for_weights)[i], wv=wv, rebeam=True, inp_beam=(info.beams)[i], new_beam=newbeam, wv_filts_to_use=scale_info_wvs.freqs_to_use[:,i], N_side_to_use=scale_info_wvs.N_side_to_use))
             print("waveletized ", i,flush=True)
    else:
        print("not waveletizing any other maps",flush=True)
        maps_for_weights_needlets = None


    # Perform the full ILC calculation at each scale
    ILC_maps_per_scale = []
    for j in range(wv.N_scales):
        # first, check if the ILC has been done already at this scale:
        ILCscalefilename = _ILC_scale_filename(info,j,scale_info_wvs)
        exists = os.path.isfile(ILCscalefilename)
        if exists:
            ILC_map_temp = hp.fitsfunc.read_map(ILCscalefilename)
        else:
            # if it has not been done, do the ILC at this scale
            ILC_map_temp = scale_info_wvs.ILC_at_scale_j(j,info,resp_tol,map_images = map_images,maps_for_weights_needlets=maps_for_weights_needlets)
    
        ILC_maps_per_scale.append(ILC_map_temp)

        ILCscalefilename = _ILC_scale_filename(info,j,scale_info_wvs)
        # save the ILC at this scale for future use
        if not exists:
            hp.fitsfunc.write_map(ILCscalefilename,ILC_map_temp)
        del(ILC_map_temp)

    # synthesize the per-needlet-scale ILC maps into the final combined ILC map (apply each needlet filter again and add them all together -- have to upgrade to all match the same Nside -- done in synthesize)
    print("synthesizing the full ILC map",flush=True)
    ILC_map = synthesize(wv_maps=ILC_maps_per_scale, wv=wv, N_side_out=info.N_side)
    # save the final ILC map
    ILC_map_filename = _ILC_map_filename(info)
    print("done! saving the map as",ILC_map_filename,flush=True)

    hp.write_map(ILC_map_filename, ILC_map, nest=False, dtype=np.float64, overwrite=False)

    # make image if requested
    if map_images:
         plot_ILC_map(info,ILCmap)

    # cross-correlate with map specified in input file (if requested; e.g., useful for simulation analyses) -- TODO
    print("done wavelet ILC",flush=True)
    if return_ILC_map: 
        return ILC_map
    return 1
    ##########################
    ##########################

# harmonic ILC
def harmonic_ILC(wv=None, info=None, resp_tol=1.e-3, map_images=False):
    # This function is copy-and-pasted from wavelet_ILC() above and edited.
    # It would be much better to avoid such hard-coding by writing one wavelet_ILC() function that can do both (in progress on local branch).
    # However, the harmonic_ILC() function itself contains a lot of hard coded code-snippets (i.e., it is a very long function with very few calls to subroutines).
    # Thus, the HILC had a lot very long if statements of the kind "if info.wavelet_type =='HILC': {do x} else {do y}" and it was very hard to follow the overall
    # function. So really, what should be done is to split wavelet_ILC() into many smaller subroutines and then it would be much easier to read the whole function
    # in this way. However, this remains TODO for an update.
    assert wv is not None, "wavelets not defined"
    assert type(wv) is Wavelets, "Wavelets TypeError"
    assert info is not None, "ILC info not defined"
    assert type(info) is ILCInfo, "ILCInfo TypeError"
    assert wv.N_scales == info.N_scales, "N_scales must match"
    assert wv.ELLMAX == info.ELLMAX, "ELLMAX must match"
    assert(info.wavelet_beam_criterion > 0. and info.wavelet_beam_criterion < 1.)
    assert info.N_side > 0, "N_side cannot be negative or zero"
    ##########################
    # criterion to determine which frequency maps to use for each wavelet filter scale
    # require multipole ell_F where wavelet filter F(ell_F) = info.wavelet_beam_criterion (on its decreasing side)
    #   to be less than the multipole ell_B where the beam B(ell_B) = info.wavelet_beam_criterion
    # note that this assumes monotonicity of the beam
    # and assumes filter function has a decreasing side, which is generally not true for the smallest-scale wavelet filter
    freqs_to_use = np.full((wv.N_scales,info.N_freqs), False)
    N_freqs_to_use = np.zeros(wv.N_scales,dtype=int)
    N_side_to_use = np.ones(wv.N_scales,dtype=int)*info.N_side #initialize all of the internal, per-scale N_side values to the output N_side
    ell_F = np.zeros(wv.N_scales)
    ell_B = np.zeros(info.N_freqs)
    for i in range(wv.N_scales-1):
        ell_peak = np.argmax(wv.filters[i]) #we'll use this to ensure we're on the decreasing side of the filter
        ell_F[i] = ell_peak + (np.abs( wv.filters[i][ell_peak:] - info.wavelet_beam_criterion )).argmin()
        if ell_F[i] > wv.ELLMAX:
            ell_F[i] = wv.ELLMAX
    ell_F[-1] = ell_F[-2] #just use the second-to-last criterion for the last one #TODO: improve this
    for j in range(info.N_freqs):
        ell_B[j] = (np.abs( (info.beams[j])[:,1] - info.wavelet_beam_criterion )).argmin()
    for i in range(wv.N_scales):
        for j in range(info.N_freqs):
            if ell_F[i] <= ell_B[j]:
                freqs_to_use[i][j] = True
                N_freqs_to_use[i] += 1
            else:
                freqs_to_use[i][j] = False
        # check that number of frequencies is non-zero
        assert(N_freqs_to_use[i] > 0), "insufficient number of channels for high-resolution filter(s)"
        # check that we still have enough frequencies for desired deprojection at each filter scale
        if type(info.N_deproj) is int:
            assert((info.N_deproj + 1) <= N_freqs_to_use[i]), "not enough frequency channels to deproject this many components"
        else:
            assert((info.N_deproj[i] + 1) <= N_freqs_to_use[i]), "not enough frequency channels to deproject this many components at scale "+str(i)
        # determine N_side value to use for each filter scale, by finding the smallest valid N_side larger than ell_F[i]
        for j in range(20):
            if (ell_F[i] < 2**j):
                N_side_to_use[i] = int(2**j)
                break
        if (N_side_to_use[i] > info.N_side):
            N_side_to_use[i] = info.N_side
    N_pix_to_use = 12*(N_side_to_use)**2
    ##########################
    ##########################
    # criterion to determine the real-space gaussian FWHM used in wavelet ILC
    # based on ILC bias mode-counting
    ell, filts = wv.TopHatHarmonic(info.ellbins)

    ##########################
    ### MAIN ILC CALCULATION ###
    # TODO -- memory management could probably be improved here (reduce file I/O overhead, reduce number of smoothing operations, etc...)
    ILC_maps_per_scale = []
    ILC_alms_per_scale = []
    ILC_alms = np.zeros(int(hp.sphtfunc.Alm.getsize(wv.ELLMAX)),dtype=np.complex_)
    ILC_filters = []
    for a in range(info.N_freqs):
        ILC_filters.append(np.zeros(wv.ELLMAX+1))
    print("doing main ILC!!",flush=True)
    ells=np.arange(wv.ELLMAX+1)

    new_beam = info.common_beam
    if(wv.taper_width):
        assert wv.ELLMAX - wv.taper_width > 10., "desired taper is too broad for given ELLMAX"
        taper_func = (1.0 - 0.5*(np.tanh(0.025*(wv.ell - (wv.ELLMAX - wv.taper_width))) + 1.0)) #smooth taper to zero from ELLMAX-taper_width to ELLMAX
    else:
        taper_func = np.ones(wv.ELLMAX+1,dtype=float)
    import time
    for j in range(wv.N_scales):
        t1j=time.time()
        # first, check if the weights already exist, and skip everything if so
        weights_exist = True
        if type(info.N_deproj) is int:
            N_deproj = info.N_deproj
            if info.N_deproj>0:
                ILC_deproj_comps = info.ILC_deproj_comps
        else:
            N_deproj = info.N_deproj[j]
            if N_deproj>0:
                ILC_deproj_comps = info.ILC_deproj_comps[j]
        weight_filename = info.output_dir+info.output_prefix+'weightvector_scale'+str(j)+'_component_'+info.ILC_preserved_comp+'_crossILC'*info.cross_ILC+'.txt'


        if N_deproj>0:
            weight_filename = info.output_dir+info.output_prefix+'weightvector_scale'+str(j)+'_component_'+info.ILC_preserved_comp+'_deproject_'+'_'.join(ILC_deproj_comps)+'_crossILC'*info.cross_ILC+'.txt'
        exists = os.path.isfile(weight_filename)
        if exists:
            print('weight vector already exists:', weight_filename,flush=True)
        else:
            weights_exist = False
            #break
        if (weights_exist == False):
            ### compute the mixing matrix A_{i\alpha} ###
            # this is the alpha^th component's SED evaluated at the i^th frequency
            # units of A_mix are K_CMB
            # Note: only include channels that are being used for this filter scale
            N_comps = (N_deproj + 1)
            A_mix = np.zeros((int(N_freqs_to_use[j]),N_comps))
            countt = 0
            for a in range(info.N_freqs):
                if (freqs_to_use[j][a] == True):
                    for b in range(N_comps):
                        if (b == 0): # zero^th component is special (this is the one being preserved in the ILC)
                            if (info.bandpass_type == 'DeltaBandpasses'):
                                # N.B. get_mix and get_mix_bandpassed assume the input maps are in uK_CMB, i.e., responses are computed in uK_CMB, but we are assuming in this code that all maps are in K_CMB, hence factor of 1.e-6 below
                                # However, note that as a consequence an output NILC CMB map from this code has units of uK_CMB!
                                A_mix[countt][b] = 1.e-6 * (get_mix([info.freqs_delta_ghz[a]], info.ILC_preserved_comp, param_dict_file=info.param_dict_file, param_dict_override=None, dust_beta_param_name='beta_CIB', radio_beta_param_name='beta_radio',
                                    radio_beta1_param_name='beta1_radio',
                                    radio_beta2_param_name='beta2_radio'

                                    ))[0] #convert to K from uK
                            elif (info.bandpass_type == 'ActualBandpasses'):
                                if info.freq_bp_files[a] is not None:
                                    A_mix[countt][b] = 1.e-6 * (get_mix_bandpassed([info.freq_bp_files[a]], info.ILC_preserved_comp, param_dict_file=info.param_dict_file, param_dict_override=None, dust_beta_param_name='beta_CIB', radio_beta_param_name='beta_radio',
                                    radio_beta1_param_name='beta1_radio',
                                    radio_beta2_param_name='beta2_radio'
))[0] #convert to K from uK
                                else:
                                    print("getting none amix",flush=True)
                                
                                    A_mix[countt][b] = 1.e-6 * (get_mix([None], info.ILC_preserved_comp, param_dict_file=info.param_dict_file, param_dict_override=None, dust_beta_param_name='beta_CIB', radio_beta_param_name='beta_radio',
                                    radio_beta1_param_name='beta1_radio',
                                    radio_beta2_param_name='beta2_radio'

                                        ))[0] #convert to K from uK

                        else:
                            if (info.bandpass_type == 'DeltaBandpasses'):
                                A_mix[countt][b] = 1.e-6 * (get_mix([info.freqs_delta_ghz[a]], ILC_deproj_comps[b-1], param_dict_file=info.param_dict_file, param_dict_override=None, dust_beta_param_name='beta_CIB', radio_beta_param_name='beta_radio',
                                    
                                    radio_beta1_param_name='beta1_radio',
                                    radio_beta2_param_name='beta2_radio'

                                    ))[0] #convert to K from uK
                            elif (info.bandpass_type == 'ActualBandpasses'):
                                A_mix[countt][b] = 1.e-6 * (get_mix_bandpassed([info.freq_bp_files[a]], ILC_deproj_comps[b-1], param_dict_file=info.param_dict_file, param_dict_override=None, dust_beta_param_name='beta_CIB', radio_beta_param_name='beta_radio',
                                    
                                    radio_beta1_param_name='beta1_radio',
                                    radio_beta2_param_name='beta2_radio'

                                    ))[0] #convert to K from uK
                    countt += 1
            # normalize the columns of A_mix corresponding to the deprojected components so that they have values near unity
            if (N_deproj != 0):
                for b in range(1,N_deproj+1):
                    max_temp = np.amax(A_mix[:,b])
                    A_mix[:,b] = A_mix[:,b]/max_temp
            ##############################
            ##############################
            # for each filter scale, compute maps of the smoothed real-space frequency-frequency covariance matrix using the Gaussians determined above
            cov_maps_temp = []
            flag=True
            cont = False
            for a in range(info.N_freqs):
                start_at = a
                if info.cross_ILC:
                    start_at = 0
                for b in range(start_at, info.N_freqs):
                    if (freqs_to_use[j][a] == True) and (freqs_to_use[j][b] == True and flag==True and not cont):
                        # Note that for HILC the covmat has no pixel index and only needs {freq1, freq2} indices at every scale. So we save it at every scale in a .txt file as a 2-d numpy array
                        cov_filename = info.output_dir+info.output_prefix+'_needletcoeff_covmap_scale'+str(j)+'_crossILC'*info.cross_ILC+'.txt' 
                        exists = os.path.isfile(cov_filename)
                        if exists:
                            cov_matrix_harmonic = np.loadtxt(cov_filename)
                            cont = True
                        else:
                            flag=False
                            break
            if flag == False:
                cov_matrix_harmonic = np.zeros((int(N_freqs_to_use[j]),int(N_freqs_to_use[j])))
                counta = 0
                for a in range(info.N_freqs):
                    countb = 0
                    for b in range(0, info.N_freqs): #  Could probably do this quicker by starting at a instead of 0 when not doing cross_ILC but would have to keep track of count_a and count_b
                        if (freqs_to_use[j][a] == True) and (freqs_to_use[j][b] == True):
                            cov_matrix_harmonic[counta,countb] = np.sum((2*ells+1)/(4*np.pi)*info.cls[a,b]* (wv.filters[j])**2*taper_func**2)/np.sum(wv.filters[j]**2) # thanks to Jack Kwok for noticing a bug here
                            countb +=1
                    if (freqs_to_use[j][a] == True):
                        counta +=1
                cov_filename = info.output_dir+info.output_prefix+'_needletcoeff_covmap_scale'+str(j)+'_crossILC'*info.cross_ILC+'.txt'
                if info.save_harmonic_covmat:
                    print("saving covmat",cov_filename,flush=True)
                    np.savetxt(cov_filename,cov_matrix_harmonic)

            ##########################
            ##########################
            # invert the cov matrix for each filter scale
            if info.cross_ILC: # symmetrize the covmat
                    cov_matrix_harmonic= (cov_matrix_harmonic+ np.transpose(cov_matrix_harmonic))/2.
            inv_covmat_harmonic= np.linalg.inv(cov_matrix_harmonic) # we don't need to bother saving this because it is not expensive to invert this covmat (TODO: check this)

            identity = np.eye(N_freqs_to_use[j])
            assert np.allclose(np.matmul(inv_covmat_harmonic,cov_matrix_harmonic),identity,rtol=1.e-3, atol=1.e-3)
            inv_covmat_temp = inv_covmat_harmonic[:,:,None]

            ### for each filter scale, perform cov matrix inversion and compute maps of the ILC weights using the inverted cov matrix maps
            count=0
            ### construct the matrix Q_{alpha beta} defined in Eq. 30 of McCarthy & Hill 2023 for each pixel at this wavelet scale and evaluate Eq. 29 to get weights ###
            tmp1 = np.einsum('ai,jip->ajp', np.transpose(A_mix), inv_covmat_temp)
            Qab_pix = np.einsum('ajp,bj->abp', tmp1, np.transpose(A_mix))
            # compute weights 
            tempvec = np.zeros((N_comps, 1))
            # treat the no-deprojection case separately, since QSa_temp is empty in this case
            if (N_comps == 1):
                tempvec[0] = [1.0]*int(1)
            else:
                for a in range(N_comps):
                    QSa_temp = np.delete(np.delete(Qab_pix, a, 0), 0, 1) #remove the a^th row and zero^th column
                    tempvec[a] = (-1.0)**float(a) * numba_det(np.transpose(QSa_temp,(2,0,1)))
            tmp2 = np.einsum('ia,ap->ip', A_mix, tempvec)
            tmp3 = np.einsum('jip,ip->jp', inv_covmat_temp, tmp2)
            weights = 1.0/numba_det(np.transpose(Qab_pix,(2,0,1)))[:,None] * np.transpose(tmp3) #N.B. 'weights' here only includes channels that passed beam_thresh criterion
            # response verification
            response = np.einsum('pi,ia->ap', weights, A_mix) #dimensions N_comps x N_pix_to_use[j]
            optimal_response_preserved_comp = np.ones(1)
            optimal_response_deproj_comp = np.zeros((N_comps-1, 1))
            if not (np.absolute(response[0]-optimal_response_preserved_comp) < resp_tol).all():
                print('preserved component response failed at wavelet scale '+str(j)+'; these should be zero: '+str(np.absolute(response[0]-optimal_response_preserved_comp))+' (tol is '+str(resp_tol)+')',flush=True)
                quit()
            if not (np.absolute(response[1:]-optimal_response_deproj_comp) < resp_tol).all():
                print('deprojected component response failed at wavelet scale '+str(j)+'; these should be zero:'+str(np.absolute(response[1:]-optimal_response_deproj_comp))+' (tol is '+str(resp_tol)+')',flush=True)
                quit()


            ##########################
            # only save these ILC weights if requested
            if (info.save_weights == 'yes' or info.save_weights == 'Yes' or info.save_weights == 'YES'):
                weight_filename = info.output_dir+info.output_prefix+'weightvector_scale'+str(j)+'_component_'+info.ILC_preserved_comp+'_crossILC'*info.cross_ILC+'.txt'
                if N_deproj>0:
                    weight_filename = info.output_dir+info.output_prefix+'weightvector_scale'+str(j)+'_component_'+info.ILC_preserved_comp+'_deproject_'+'_'.join(ILC_deproj_comps)+'_crossILC'*info.cross_ILC+'.txt'
                np.savetxt(weight_filename, weights,)
        else:
            weight_filename = info.output_dir+info.output_prefix+'weightvector_scale'+str(j)+'_component_'+info.ILC_preserved_comp+'_crossILC'*info.cross_ILC+'.txt'
            if N_deproj>0:
                weight_filename = info.output_dir+info.output_prefix+'weightvector_scale'+str(j)+'_component_'+info.ILC_preserved_comp+'_deproject_'+'_'.join(ILC_deproj_comps)+'_crossILC'*info.cross_ILC+'.txt'

            weights = np.loadtxt(weight_filename)
        ##########################
        # apply these ILC weights to the needlet coefficient maps to get the per-needlet-scale ILC maps
        #ILC_map_temp = np.zeros(int(N_pix_to_use[j]))
        #ILC_alms = np.zeros(int(hp.sphtfunc.Alm.getsize(wv.ELLMAX)))
        count=0
        for a in range(info.N_freqs):
            if (freqs_to_use[j][a] == True):
                #wavelet_coeff_alm = info.alms[a]
                inp_beam = (info.beams)[a]
                beam_fac = new_beam[:,1]/inp_beam[:,1]
                ILC_filters[a] += weights[:,count]*taper_func*beam_fac*wv.filters[j]
                #ILC_alms += hp.almxfl(wavelet_coeff_alm ,weights[:,count]*taper_func*beam_fac*wv.filters[j])
                count+=1
        #ILC_alms_per_scale.append(ILC_alm_temp)
    ##########################
    # synthesize the per-needlet-scale ILC maps into the final combined ILC map (apply each needlet filter again and add them all together -- have to upgrade to all match the same Nside -- done in synthesize)
    ILC_alm = np.zeros(int(hp.sphtfunc.Alm.getsize(wv.ELLMAX)),dtype=np.complex_)
    for a in range(info.N_freqs):
        if not info.apply_weights_to_other_maps:
            wavelet_coeff_alm = info.alms[a]
        else: 
            wavelet_coeff_alm = info.alms_to_apply_weights[a]
        ILC_alm += hp.almxfl(wavelet_coeff_alm ,ILC_filters[a])
    ILC_map = hp.alm2map(ILC_alm,nside=info.N_side)
    #ILC_map = synthesize(wv_maps=ILC_maps_per_scale, wv=wv, N_side_out=info.N_side)
    # save the final ILC map
    ILC_map_filename = info.output_dir+info.output_prefix+'needletILCmap'+'_component_'+info.ILC_preserved_comp+'_crossILC'*info.cross_ILC+info.output_suffix+'.fits'
    if type(info.N_deproj) is int:
        if info.N_deproj>0:
            ILC_map_filename = info.output_dir+info.output_prefix+'needletILCmap'+'_component_'+info.ILC_preserved_comp+'_deproject_'+'_'.join(info.ILC_deproj_comps)+'_crossILC'*info.cross_ILC+info.output_suffix+'.fits'
    else:
        if info.N_deproj[0]>0:
            ILC_map_filename = info.output_dir+info.output_prefix+'needletILCmap'+'_component_'+info.ILC_preserved_comp+'_deproject_'+'_'.join(info.ILC_deproj_comps[0])+'_crossILC'*info.cross_ILC+info.output_suffix+'.fits'
    hp.write_map(ILC_map_filename, ILC_map, nest=False, dtype=np.float64, overwrite=False)
    # make image if requested
    if map_images == True:
        plt.clf()
        hp.mollview(ILC_map, unit='dimensionless', title='Needlet ILC Map, Component '+info.ILC_preserved_comp)
        if type(info.N_deproj) is int:
            if info.N_deproj==0:
                plt.savefig(info.output_dir+info.output_prefix+'needletILCmap'+'_component_'+info.ILC_preserved_comp+'_crossILC'*info.cross_ILC+'.pdf')
            else:
                plt.savefig(info.output_dir+info.output_prefix+'needletILCmap'+'_component_'+info.ILC_preserved_comp+'_deproject_'+'_'.join(info.ILC_deproj_comps)+'_crossILC'*info.cross_ILC+'.pdf')
        else:
            if info.N_deproj[0]==0:
                plt.savefig(info.output_dir+info.output_prefix+'needletILCmap'+'_component_'+info.ILC_preserved_comp+'_crossILC'*info.cross_ILC+'.pdf')
            else:
                plt.savefig(info.output_dir+info.output_prefix+'needletILCmap'+'_component_'+info.ILC_preserved_comp+'_deproject_'+'_'.join(info.ILC_deproj_comps[0])+'_crossILC'*info.cross_ILC+'.pdf')


    # cross-correlate with map specified in input file (if requested; e.g., useful for simulation analyses) -- TODO
    return 1
    ##########################
    ##########################












##############################################################################
# only cruft below here
##############################################################################
#wv = Wavelets()
#ell, filts = wv.GaussianNeedlets()
##print(wv.N_scales,wv.ELLMAX)
##print(wv.filters[0])
#assert type(wv) is Wavelets, "Wavelets TypeError"

# # plot -- match Fig. A.2 of https://arxiv.org/pdf/1605.09387.pdf
# # can also match Fig. 1 of Planck 2015 y-map paper by also including 10 arcmin beam
# bl10arcmin = hp.sphtfunc.gauss_beam(10.* np.pi/(180.*60.), lmax=len(ell)-1) #FWHM=10 arcmin (in radian)
# plt.clf()
# for i in xrange(10):
#     plt.semilogx(ell, filts[i], 'k', lw=0.75)
#     plt.semilogx(ell, (filts[i])**2.0 * bl10arcmin, 'b')
# plt.xlim(left=1, right=1e4)
# plt.ylim(0.0, 1.0)
# plt.xlabel(r'$\ell$', fontsize=18)
# plt.ylabel(r'$B^{\alpha}_{\ell}$', fontsize=18)
# plt.grid()
# #plt.legend(loc=0)
# plt.savefig('NILC_bands_FigA2ofGNILCpaper.pdf')

# # # read in NILC filter scales from 2015 y-map paper
# ELLMAX_NILC = 4097
# ell_NILC = np.arange(0,ELLMAX_NILC+1)
# Nbands_NILC = 10
# bl10arcmin = hp.sphtfunc.gauss_beam(0.00290888208, lmax=len(ell_NILC)-1) #FWHM=10 arcmin (in radian)
# NILC_bands = np.zeros((Nbands_NILC,ELLMAX_NILC+1))
# NILC_bands[0][0] = 1.0
# for i in xrange(1,Nbands_NILC):
#     NILC_bands[i][0] = 0.0
# hdulist = fits.open('/data/jch/Planckdata/COM_CompMap_YSZ_R2.00.fits/nilc_bands.fits')
# for i in xrange(Nbands_NILC):
#     for j in xrange(1,ELLMAX_NILC+1):
#         NILC_bands[i][j] = (hdulist[1].data)[j-1][i]
# hdulist.close()

# # plot the NILC bands -- note that Fig. 1 of the 2015 y-map paper is actually a plot of (h^j_ell)^2 * b_ell (filter bands squared times 10 arcmin beam) [info from Mathieu Remazeilles]
# plt.clf()
# for i in xrange(Nbands_NILC):
#     plt.semilogx(ell_NILC, NILC_bands[i], 'k', lw=0.75)
#     plt.semilogx(ell_NILC, (NILC_bands[i])**2.0 * bl10arcmin, 'b')
# plt.xlim(left=1, right=1e4)
# plt.ylim(0.0, 1.0)
# plt.xlabel(r'$\ell$', fontsize=18)
# plt.ylabel(r'$B^{\alpha}_{\ell}$', fontsize=18)
# plt.grid()
# #plt.legend(loc=0)
# plt.savefig('NILC_bands_Fig1ofymappaper.pdf')

# # plot of 2015 y-map NILC bands vs. the GNILC bands
# plt.clf()
# for i in xrange(Nbands_NILC):
#     if (i==0):
#         plt.semilogx(ell_NILC, NILC_bands[i], 'r', lw=1.5, label='2015 y-map')
#         #plt.semilogx(ell, filts[i], 'k', lw=1., ls='--', label='2016 GNILC')
#         plt.semilogx(ell, filts[i], 'k', lw=1., ls='--', label='2015 y-map (mine)')
#     else:
#         plt.semilogx(ell_NILC, NILC_bands[i], 'r', lw=1.5)
#         plt.semilogx(ell, filts[i], 'k', lw=1., ls='--')
# plt.xlim(left=1, right=4e3)
# plt.ylim(0.0, 1.0)
# plt.xlabel(r'$\ell$', fontsize=18)
# plt.ylabel(r'$h^{j}_{\ell}$', fontsize=18)
# plt.grid()
# plt.legend(loc=0, fontsize=10, ncol=2)
# #plt.savefig('NILC_bands_ymap_vs_GNILC.pdf')
# plt.savefig('NILC_bands_ymap_check.pdf')
