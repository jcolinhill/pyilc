from __future__ import print_function
import numpy as np
import healpy as hp
from astropy.io import fits
import os
import matplotlib
matplotlib.use('pdf')
matplotlib.rc('font', family='serif', serif='cm10')
matplotlib.rc('text', usetex=True)
fontProperties = {'family':'sans-serif',
                  'weight' : 'normal', 'size' : 16}
import matplotlib.pyplot as plt
from input import ILCInfo
from fg import get_mix, get_mix_bandpassed
"""
this module constructs the Wavelets class, which contains the
harmonic-space filters defining a set of wavelets, as well as
some associated methods.  three example types of wavelets are
explicitly constructed:
- Gaussian needlets:
  - Planck 2015 NILC y-map needlets
  - Planck 2016 GNILC dust map needlets
- cosine needlets [TODO]
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
    #    TODO: (not yet implemented here)
    #def CosineNeedlets(self, ellmin=None, ellpeak=None, ellmax=None):
    #    TODO: (not yet implemented here)
    #    # simple check to ensure that sum of squared transmission is unity as needed for NILC algorithm
    #    assert (np.absolute( np.sum( self.filters**2., axis=0 ) - np.ones(self.ELLMAX+1,dtype=float)) < self.tol).all(), "wavelet filter transmission check failed"
    #    return self.ell, self.filters

    # scale-discretized wavelets
    #def ScaleDiscretizedWavelets(self, TODO):
    #    TODO: (not yet implemented here)
    #    # simple check to ensure that sum of squared transmission is unity as needed for NILC algorithm
    #    assert (np.absolute( np.sum( self.filters**2., axis=0 ) - np.ones(self.ELLMAX+1,dtype=float)) < self.tol).all(), "wavelet filter transmission check failed"
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


# apply wavelet transform (i.e., filters) to a map
def waveletize(inp_map=None, wv=None, rebeam=False, inp_beam=None, new_beam=None, wv_filts_to_use=None, N_side_to_use=None):
    assert inp_map is not None, "no input map specified"
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

def _cov_filename(info,freq1,freq2,scale):

    a = freq1
    b = freq2
    j = scale

    cov_filename = info.output_dir+info.output_prefix+'_needletcoeff_covmap_freq'+str(a)+'_freq'+str(b)+'_scale'+str(j)+'_crossILC'*info.cross_ILC+'.fits'
    if info.recompute_covmat_for_ndeproj:
        if type(info.N_deproj) is int:
            N_deproj = info.N_deproj
        else:
            N_deproj = info.N_deproj[j]
        cov_filename = info.output_dir+info.output_prefix+'_needletcoeff_covmap_freq'+str(a)+'_freq'+str(b)+'_scale'+str(j)+'_crossILC'*info.cross_ILC+'_Ndeproj'+str(N_deproj)+'.fits'

    return cov_filename

def _inv_cov_filename(info,scale,freq1,freq2):
    a = freq1
    b = freq2
    j = scale
    inv_cov_filename = info.output_dir+info.output_prefix+'_needletcoeff_invcovmap_freq'+str(a)+'_freq'+str(b)+'_scale'+str(j)+'_crossILC'*info.cross_ILC+'.fits'
    if info.recompute_covmat_for_ndeproj:
                    if type(info.N_deproj) is int:
                        N_deproj = info.N_deproj
                    else:
                        N_deproj = info.N_deproj[j]
                    inv_cov_filename = info.output_dir+info.output_prefix+'_needletcoeff_invcovmap_freq'+str(a)+'_freq'+str(b)+'_scale'+str(j)+'_crossILC'*info.cross_ILC+'_Ndeproj'+str(N_deproj)+'.fits'
    return inv_cov_filename


def _weights_filename(info,freq,scale):
                a = freq
                j = scale
                weight_filename = info.output_dir+info.output_prefix+'weightmap_freq'+str(a)+'_scale'+str(j)+'_component_'+info.ILC_preserved_comp+'_crossILC'*info.cross_ILC+'.fits'
                if type(info.N_deproj )is int:
                    if info.N_deproj>0:
                        weight_filename = info.output_dir+info.output_prefix+'weightmap_freq'+str(a)+'_scale'+str(j)+'_component_'+info.ILC_preserved_comp+'_deproject_'+'_'.join(info.ILC_deproj_comps)+'_crossILC'*info.cross_ILC+'.fits'
                else:
                    if info.N_deproj[j]>0:
                        weight_filename = info.output_dir+info.output_prefix+'weightmap_freq'+str(a)+'_scale'+str(j)+'_component_'+info.ILC_preserved_comp+'_deproject_'+'_'.join(info.ILC_deproj_comps[j])+'_crossILC'*info.cross_ILC+'.fits'
                if info.recompute_covmat_for_ndeproj:
                    if type(info.N_deproj) is int:
                        N_deproj = info.N_deproj
                    else:
                        N_deproj = info.N_deproj[j]
                    weight_filename = weight_filename[:-5] +'_Ndeproj'+str(N_deproj)+'.fits'

                weight_filename = weight_filename[:-5]+info.output_suffix+'.fits'
                return weight_filename



# wavelet ILC
def wavelet_ILC(wv=None, info=None, wavelet_beam_criterion=1.e-3, resp_tol=1.e-3, map_images=False):
    assert wv is not None, "wavelets not defined"
    assert type(wv) is Wavelets, "Wavelets TypeError"
    assert info is not None, "ILC info not defined"
    assert type(info) is ILCInfo, "ILCInfo TypeError"
    assert wv.N_scales == info.N_scales, "N_scales must match"
    assert wv.ELLMAX == info.ELLMAX, "ELLMAX must match"
    assert(wavelet_beam_criterion > 0. and wavelet_beam_criterion < 1.)
    assert info.N_side > 0, "N_side cannot be negative or zero"
    ##########################
    # criterion to determine which frequency maps to use for each wavelet filter scale
    # require multipole ell_F where wavelet filter F(ell_F) = wavelet_beam_criterion (on its decreasing side)
    #   to be less than the multipole ell_B where the beam B(ell_B) = wavelet_beam_criterion
    # note that this assumes monotonicity of the beam
    # and assumes filter function has a decreasing side, which is generally not true for the smallest-scale wavelet filter
    freqs_to_use = np.full((wv.N_scales,info.N_freqs), False)
    N_freqs_to_use = np.zeros(wv.N_scales,dtype=int)
    N_side_to_use = np.ones(wv.N_scales,dtype=int)*info.N_side #initialize all of the internal, per-scale N_side values to the output N_side
    ell_F = np.zeros(wv.N_scales)
    ell_B = np.zeros(info.N_freqs)
    for i in range(wv.N_scales-1):
        ell_peak = np.argmax(wv.filters[i]) #we'll use this to ensure we're on the decreasing side of the filter
        ell_F[i] = ell_peak + (np.abs( wv.filters[i][ell_peak:] - wavelet_beam_criterion )).argmin()
        if ell_F[i] > wv.ELLMAX:
            ell_F[i] = wv.ELLMAX
    ell_F[-1] = ell_F[-2] #just use the second-to-last criterion for the last one #TODO: improve this
    for j in range(info.N_freqs):
        ell_B[j] = (np.abs( (info.beams[j])[:,1] - wavelet_beam_criterion )).argmin()
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
            assert((info.N_deproj[i] + 1) <= N_freqs_to_use[i]), "not enough frequency channels to deproject this many components at scale "+ str()
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
    FWHM_pix = np.zeros(wv.N_scales,dtype=float)
    if info.wavelet_type == 'GaussianNeedlets':
        ell, filts = wv.GaussianNeedlets(info.GN_FWHM_arcmin)
    elif info.wavelet_type == 'TopHatHarmonic':
        ell, filts = wv.TopHatHarmonic(info.ellbins)
    # TODO: implement these
    #elif info.wavelet_type == 'CosineNeedlets':
    #elif info.wavelet_type == 'ScaleDiscretizedWavelets':
    else:
        raise NotImplementedError
    # compute effective number of modes associated with each filter (on the full sky)
    # note that the weights we use are filt^2, since this is the quantity that sums to unity at each ell
    N_modes = np.zeros(wv.N_scales, dtype=float)
    for i in range(wv.N_scales):
        N_modes[i] = np.sum( (2.*ell + np.ones(wv.ELLMAX+1)) * (filts[i])**2. )
    # now find real-space Gaussian s.t. number of modes in that area satisfies ILC bias threshold
    # we use the flat-sky approximation here -- TODO: could improve this
    for i in range(wv.N_scales):
        # this expression comes from noting that ILC_bias_tol = (N_deproj+1 - N_freqs)/N_modes_eff
        #   where N_modes_eff = N_modes * (2*pi*sigma_pix^2)/(4*pi)
        #   and then solving for sigma_pix
        # note that this corrects an error in Eq. 3 of Planck 2015 y-map paper -- the numerator should be (N_ch - 2) in their case (if they're deprojecting CMB)
        if type(info.N_deproj) is int:
            sigma_pix_temp = np.sqrt( np.absolute( 2.*(float( (info.N_deproj + 1) - N_freqs_to_use[i] )) / (N_modes[i] * info.ILC_bias_tol) ) ) #result is in radians
        else:
            sigma_pix_temp = np.sqrt( np.absolute( 2.*(float( (info.N_deproj[i] + 1) - N_freqs_to_use[i] )) / (N_modes[i] * info.ILC_bias_tol) ) ) #result is in radians
        assert sigma_pix_temp < np.pi, "not enough modes to satisfy ILC_bias_tol" #don't want real-space gaussian to be the full sky or close to it
        # note that sigma_pix_temp can come out zero if N_deproj+1 = N_freqs_to_use (formally bias vanishes in this case because the problem is fully constrained)
        # for now, just set equal to case where N_freqs_to_use = N_deproj
        if sigma_pix_temp == 0.:
            sigma_pix_temp = np.sqrt( np.absolute( 2. / (N_modes[i] * info.ILC_bias_tol) ) ) #result is in radians
        FWHM_pix[i] = np.sqrt(8.*np.log(2.)) * sigma_pix_temp #in radians
    ##########################
    ##########################
    # compute wavelet decomposition of all frequency maps used at each filter scale
    # save the filtered maps (aka maps of "wavelet coefficients")
    # remember to re-convolve all maps to the highest resolution map being used when passing into needlet filtering, or to the user-specified input beam at which to compute the ILC
    for i in range(info.N_freqs):
        # N.B. maps are assumed to be in strictly decreasing order of FWHM! i.e. info.beams[-1] is highest-resolution beam
        print("waveletizing frequency ", i, "...")
        wv_maps_temp = []
        flag=True
        for j in range(wv.N_scales):
            if freqs_to_use[j][i] == True:
                filename = info.output_dir+info.output_prefix+'_needletcoeffmap_freq'+str(i)+'_scale'+str(j)+'.fits'
                exists = os.path.isfile(filename)
                if exists:
                    print('needlet coefficient map already exists:', filename)
                    wv_maps_temp.append( hp.read_map(filename, dtype=np.float64) )
                else:
                    print('needlet coefficient map not previously computed; re-computing all maps for frequency '+str(i)+' now...')
                    flag=False
                    break
        if flag == False:
            wv_maps_temp = waveletize(inp_map=(info.maps)[i], wv=wv, rebeam=True, inp_beam=(info.beams)[i], new_beam=info.common_beam, wv_filts_to_use=freqs_to_use[:,i], N_side_to_use=N_side_to_use)
            for j in range(wv.N_scales):
                if freqs_to_use[j][i] == True:
                    filename = info.output_dir+info.output_prefix+'_needletcoeffmap_freq'+str(i)+'_scale'+str(j)+'.fits'
                    hp.write_map(filename, wv_maps_temp[j], nest=False, dtype=np.float64, overwrite=False)
        if map_images == True:
            print(len(wv_maps_temp))
            for j in range(wv.N_scales):
                if freqs_to_use[j][i] == True:
                    plt.clf()
                    hp.mollview(wv_maps_temp[j], unit="K", title="Needlet Coefficient Map, Frequency "+str(i)+" Scale "+str(j), min=np.mean(wv_maps_temp[j])-2*np.std(wv_maps_temp[j]), max=np.mean(wv_maps_temp[j])+2*np.std(wv_maps_temp[j]))
                    plt.savefig(info.output_dir+info.output_prefix+'_needletcoeffmap_freq'+str(i)+'_scale'+str(j)+'.pdf')
        print("done waveletizing frequency ", i, "...")
        if info.cross_ILC:
            for season in [1,2]:
                flag = True
                wv_maps_temp = []
                for j in range(wv.N_scales):
                    if freqs_to_use[j][i] == True:
                        filename = info.output_dir+info.output_prefix+'_needletcoeffmap_freq'+str(i)+'_scale'+str(j)+'_S'+str(season)+'.fits'
                        exists = os.path.isfile(filename)
                        if exists:
                            print('needlet coefficient map already exists:', filename,)
                            wv_maps_temp.append( hp.read_map(filename, dtype=np.float64) )
                        else:
                            print('needlet coefficient map not previously computed; re-computing all '+str(season)+'maps for frequency '+str(i)+' now...',)
                            flag=False
                            break
                if flag == False:
                    if season==1:
                        maps = info.maps_s1
                    elif season==2:
                        maps = info.maps_s2

                    wv_maps_temp = waveletize(inp_map=(maps)[i], wv=wv, rebeam=True, inp_beam=(info.beams)[i], new_beam=newbeam, wv_filts_to_use=freqs_to_use[:,i], N_side_to_use=N_side_to_use)
                    for j in range(wv.N_scales):
                        if freqs_to_use[j][i] == True:
                            filename = info.output_dir+info.output_prefix+'_needletcoeffmap_freq'+str(i)+'_scale'+str(j)+'_S'+str(season)+'.fits'
                            exists2 = os.path.isfile(filename)
                            if not exists2:
                                hp.write_map(filename, wv_maps_temp[j], nest=False, dtype=np.float64, overwrite=False)
                    del(maps) #free up memory
        del wv_maps_temp #free up memory
    ##########################
    ##########################
    ### MAIN ILC CALCULATION ###
    # TODO -- memory management could certainly be improved here (reduce file I/O overhead, reduce number of smoothing operations, etc...)

    if info.apply_weights_to_other_maps:
         maps_for_weights_needlets=[]
         for i in range(info.N_freqs):
             print("waveletizing maps to apply weights", i)
             if info.perform_ILC_at_beam is not None:
                    newbeam = info.common_beam
             else:
                    newbeam = (info.beams)[-1]
             maps_for_weights_needlets.append(waveletize(inp_map=(info.maps_for_weights)[i], wv=wv, rebeam=True, inp_beam=(info.beams)[i], new_beam=newbeam, wv_filts_to_use=freqs_to_use[:,i], N_side_to_use=N_side_to_use))
             print("waveletized ", i)
    else:
        print("not waveletizing any other maps")

    ILC_maps_per_scale = []
    for j in range(wv.N_scales):
        # first, check if the weights already exist, and skip everything if so
        if type(info.N_deproj) is int:
            N_deproj = info.N_deproj
            if N_deproj>0:
                ILC_deproj_comps = info.ILC_deproj_comps
        else:
            N_deproj = info.N_deproj[j]
            if N_deproj>0:
                ILC_deproj_comps = info.ILC_deproj_comps[j]
        weights_exist = True
        count=0
        for a in range(info.N_freqs):
            if (freqs_to_use[j][a] == True):
                weight_filename = _weights_filename(info,a,j)
                exists = os.path.isfile(weight_filename)
                if exists:
                    print('weight map already exists:', weight_filename)
                    count += 1
                else:
                    weights_exist = False
                    break
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
                                A_mix[countt][b] = 1.e-6 * (get_mix([info.freqs_delta_ghz[a]], info.ILC_preserved_comp, param_dict_file=info.param_dict_file, param_dict_override=None, dust_beta_param_name='beta_CIB', radio_beta_param_name='beta_radio'))[0] #convert to K from uK
                            elif (info.bandpass_type == 'ActualBandpasses'):
                                A_mix[countt][b] = 1.e-6 * (get_mix_bandpassed([info.freq_bp_files[a]], info.ILC_preserved_comp, param_dict_file=info.param_dict_file, param_dict_override=None, dust_beta_param_name='beta_CIB', radio_beta_param_name='beta_radio'))[0] #convert to K from uK
                        else:
                            if (info.bandpass_type == 'DeltaBandpasses'):
                                A_mix[countt][b] = 1.e-6 * (get_mix([info.freqs_delta_ghz[a]], ILC_deproj_comps[b-1], param_dict_file=info.param_dict_file, param_dict_override=None, dust_beta_param_name='beta_CIB', radio_beta_param_name='beta_radio'))[0] #convert to K from uK
                            elif (info.bandpass_type == 'ActualBandpasses'):
                                A_mix[countt][b] = 1.e-6 * (get_mix_bandpassed([info.freq_bp_files[a]], ILC_deproj_comps[b-1], param_dict_file=info.param_dict_file, param_dict_override=None, dust_beta_param_name='beta_CIB', radio_beta_param_name='beta_radio'))[0] #convert to K from uK
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
            for a in range(info.N_freqs):
                start_at = a
                if info.cross_ILC:
                    start_at = 0
                for b in range(start_at, info.N_freqs):
                    if (freqs_to_use[j][a] == True) and (freqs_to_use[j][b] == True and flag==True):
                        cov_filename = _cov_filename(info,a,b,j)
                        exists = os.path.isfile(cov_filename)
                        if exists:
                            print('needlet coefficient covariance map already exists:', cov_filename)
                            cov_maps_temp.append( hp.read_map(cov_filename, dtype=np.float64) )
                        else:
                            print('needlet coefficient covariance map not previously computed; re-computing all covariance maps at scale'+str(j)+' now...')
                            flag=False
                            break
            if flag == False:
                for a in range(info.N_freqs):
                    start_at = a
                    if info.cross_ILC:
                        start_at = 0
                    for b in range(start_at, info.N_freqs):
                        if (freqs_to_use[j][a] == True) and (freqs_to_use[j][b] == True):
                            cov_filename = _cov_filename(info,a,b,j)
                            # read in wavelet coefficient maps constructed in previous step above
                            if not info.cross_ILC:
                                filename_A = info.output_dir+info.output_prefix+'_needletcoeffmap_freq'+str(a)+'_scale'+str(j)+'.fits'
                                filename_B = info.output_dir+info.output_prefix+'_needletcoeffmap_freq'+str(b)+'_scale'+str(j)+'.fits'
                            else:
                                filename_A = info.output_dir+info.output_prefix+'_needletcoeffmap_freq'+str(a)+'_scale'+str(j)+'_S1.fits'
                                filename_B = info.output_dir+info.output_prefix+'_needletcoeffmap_freq'+str(b)+'_scale'+str(j)+'_S2.fits'
                            wavelet_map_A = hp.read_map(filename_A, dtype=np.float64)
                            wavelet_map_B = hp.read_map(filename_B, dtype=np.float64)
                            assert len(wavelet_map_A) == len(wavelet_map_B), "cov mat map calculation: wavelet coefficient maps have different N_side"
                            # first perform smoothing operation to get the "mean" maps
                            wavelet_map_A_smoothed = hp.sphtfunc.smoothing(wavelet_map_A, FWHM_pix[j])
                            wavelet_map_B_smoothed = hp.sphtfunc.smoothing(wavelet_map_B, FWHM_pix[j])
                            # then construct the smoothed real-space freq-freq cov matrix element for this pair of frequency maps
                            # note that the overall normalization of this cov matrix is irrelevant for the ILC weight calculation (it always cancels out)
                            cov_map_temp = hp.sphtfunc.smoothing( (wavelet_map_A - wavelet_map_A_smoothed)*(wavelet_map_B - wavelet_map_B_smoothed) , FWHM_pix[j])
                            cov_maps_temp.append( cov_map_temp )
                            hp.write_map(cov_filename, cov_map_temp, nest=False, dtype=np.float64, overwrite=False)
            print('done computing all covariance maps at scale'+str(j),flush=True)
            ##########################
            ##########################
            # invert the cov matrix in each pixel for each filter scale
            inv_cov_maps_temp = np.zeros((len(cov_maps_temp), int((N_pix_to_use[j]))))
            # determine lowest-resolution frequency used in the analysis for this scale -- N.B. freq maps are assumed to be in order from lowest to highest resolution (as elsewhere in the code)
            for a in range(info.N_freqs):
                if (freqs_to_use[j][a] == True):
                    a_min = a
                    break
            ### for each filter scale, perform cov matrix inversion and compute maps of the ILC weights using the inverted cov matrix maps
            weights = np.zeros((int(N_pix_to_use[j]),int(N_freqs_to_use[j])))
            flag=True #flag for whether inverse covariance maps already exist
            count=0
            for a in range(info.N_freqs):
                for b in range(a, info.N_freqs):
                    if (freqs_to_use[j][a] == True) and (freqs_to_use[j][b] == True and flag==True):
                        inv_cov_filename = _inv_cov_filename(info,j,a,b)
                        exists = os.path.isfile(inv_cov_filename)
                        if exists:
                            print('needlet coefficient inverse covariance map already exists:', inv_cov_filename)
                            inv_cov_maps_temp[count] = hp.read_map(inv_cov_filename, dtype=np.float64) #by construction we're going through cov_maps_temp in the same order as it was populated above
                            count+=1
                        else:
                            print('needlet coefficient inverse covariance map not previously computed; re-computing all inverse covariance maps at scale'+str(j)+' now...')
                            flag=False
                            break
            if (flag==True):
                ### construct the matrix Q_{alpha beta} defined in Eq. 30 of McCarthy & Hill 2023 for each pixel at this wavelet scale and evaluate Eq. 29 to get weights ###
                inv_covmat_temp = np.zeros((int(N_freqs_to_use[j]),int(N_freqs_to_use[j]), int(N_pix_to_use[j])))
                count=0
                for a in range(info.N_freqs):
                    for b in range(a, info.N_freqs):
                        if (freqs_to_use[j][a] == True) and (freqs_to_use[j][b] == True):
                            # inv_cov_maps_temp is in order 00, 01, 02, ..., 0(N_freqs_to_use[j]-1), 11, 12, ..., 1(N_freqs_to_use[j]-1), 22, 23, ...
                            inv_covmat_temp[a-a_min][b-a_min] = inv_cov_maps_temp[count] #by construction we're going through inv_cov_maps_temp in the same order as it was populated when computed (see below)
                            if (a-a_min) != (b-a_min):
                                inv_covmat_temp[b-a_min][a-a_min] = inv_covmat_temp[a-a_min][b-a_min] #symmetrize
                            count+=1
                tmp1 = np.einsum('ai,jip->ajp', np.transpose(A_mix), inv_covmat_temp)
                Qab_pix = np.einsum('ajp,bj->abp', tmp1, np.transpose(A_mix))
                # compute weights 
                tempvec = np.zeros((N_comps, int(N_pix_to_use[j])))
                # treat the no-deprojection case separately, since QSa_temp is empty in this case
                if (N_comps == 1):
                    tempvec[0] = [1.0]*int(N_pix_to_use[j])
                else:
                    for a in range(N_comps):
                        QSa_temp = np.delete(np.delete(Qab_pix, a, 0), 0, 1) #remove the a^th row and zero^th column
                        tempvec[a] = (-1.0)**float(a) * np.linalg.det(np.transpose(QSa_temp,(2,0,1))) 
                tmp2 = np.einsum('ia,ap->ip', A_mix, tempvec)
                tmp3 = np.einsum('jip,ip->jp', inv_covmat_temp, tmp2)
                weights = 1.0/np.linalg.det(np.transpose(Qab_pix,(2,0,1)))[:,None]*np.transpose(tmp3) #N.B. 'weights' here only includes channels that passed beam_thresh criterion, 
                # response verification
                response = np.einsum('pi,ia->ap', weights, A_mix) #dimensions N_comps x N_pix_to_use[j]
                optimal_response_preserved_comp = np.ones(int(N_pix_to_use[j]))  #preserved component, want response=1
                optimal_response_deproj_comp = np.zeros((N_comps-1, int(N_pix_to_use[j]))) #deprojected components, want response=0
                if not (np.absolute(response[0]-optimal_response_preserved_comp) < resp_tol).all():
                    print(f'preserved component response failed at wavelet scale {j}')
                    quit()
                if not (np.absolute(response[1:]-optimal_response_deproj_comp) < resp_tol).all():
                    print(f'deprojected component response failed at wavelet scale {j}')
                    quit()


            ##########
            ### if inverse covariance maps don't already exist ###
            if (flag == False):
                covmat = np.zeros((int(N_freqs_to_use[j]),int(N_freqs_to_use[j]), int(N_pix_to_use[j])))
                count=0
                for a in range(info.N_freqs):
                    start_at = a
                    if info.cross_ILC:
                        start_at = 0
                    for b in range(start_at, info.N_freqs):
                        if (freqs_to_use[j][a] == True) and (freqs_to_use[j][b] == True):
                            # cov_maps_temp is in order 00, 01, 02, ..., 0(N_freqs_to_use[j]-1), 11, 12, ..., 1(N_freqs_to_use[j]-1), 22, 23, ...
                            covmat[a-a_min][b-a_min] = cov_maps_temp[count] #by construction we're going through cov_maps_temp in the same order as it was populated above
                            # TODO: maybe symmetrize it before saving so that we don't have to save twice as many covmats?
                            if (a-a_min) != (b-a_min) and not info.cross_ILC:
                                covmat[b-a_min][a-a_min] = covmat[a-a_min][b-a_min] #symmetrize
                            count+=1
                # cross-ILC : symmetrize the covmat 
                if info.cross_ILC:
                    covmat = (covmat + np.transpose(covmat,(1,0,2)))/2
                inv_covmat = np.linalg.inv(np.transpose(covmat,(2,0,1)))
                inv_covmat = np.transpose(inv_covmat, axes=[1,2,0]) #new dim freq, freq, pix

                assert np.allclose(np.einsum('ijp,jkp->pik', inv_covmat, covmat), np.transpose(np.repeat(np.eye(N_freqs_to_use[j])[:,:,None],N_pix_to_use[j],axis=2),(2,0,1)), rtol=1.e-2, atol=1.e-2), "covmat inversion failed for scale "+str(j) #, covmat, inv_covmat, np.dot(inv_covmat, covmat)-np.eye(int(N_freqs_to_use[j]))
                count=0
                for a in range(info.N_freqs):
                    for b in range(a, info.N_freqs):
                        if (freqs_to_use[j][a] == True) and (freqs_to_use[j][b] == True):
                            # inv_cov_maps_temp is in order 00, 01, 02, ..., 0(N_freqs_to_use[j]-1), 11, 12, ..., 1(N_freqs_to_use[j]-1), 22, 23, ...
                            inv_cov_maps_temp[count] = inv_covmat[a-a_min][b-a_min] #by construction we're going through cov_maps_temp in the same order as it was populated above
                            count+=1

                tmp1 = np.einsum('ai,jip->ajp', np.transpose(A_mix), inv_covmat)
                Qab_pix = np.einsum('ajp,bj->abp', tmp1, np.transpose(A_mix))
                # compute weights 
                tempvec = np.zeros((N_comps, int(N_pix_to_use[j])))
                # treat the no-deprojection case separately, since QSa_temp is empty in this case
                if (N_comps == 1):
                    tempvec[0] = [1.0]*int(N_pix_to_use[j])
                else:
                    for a in range(N_comps):
                        QSa_temp = np.delete(np.delete(Qab_pix, a, 0), 0, 1) #remove the a^th row and zero^th column
                        tempvec[a] = (-1.0)**float(a) * np.linalg.det(np.transpose(QSa_temp,(2,0,1)))
                tmp2 = np.einsum('ia,ap->ip', A_mix, tempvec)
                tmp3 = np.einsum('jip,ip->jp', inv_covmat, tmp2)
                weights = 1.0/np.linalg.det(np.transpose(Qab_pix,(2,0,1)))[:,None]*np.transpose(tmp3) #N.B. 'weights' here only includes channels that passed beam_thresh criterion, 
                # response verification
                response = np.einsum('pi,ia->ap', weights, A_mix) #dimensions N_comps x N_pix_to_use[j]
                optimal_response_preserved_comp = np.ones(int(N_pix_to_use[j]))  #preserved component, want response=1
                optimal_response_deproj_comp = np.zeros((N_comps-1, int(N_pix_to_use[j]))) #deprojected components, want response=0
                if not (np.absolute(response[0]-optimal_response_preserved_comp) < resp_tol).all():
                    print(f'preserved component response failed at wavelet scale {j}')
                    quit()
                if not (np.absolute(response[1:]-optimal_response_deproj_comp) < resp_tol).all():
                    print(f'deprojected component response failed at wavelet scale {j}')
                    quit()

                # save inverse covariance maps for future use
                count=0
                for a in range(info.N_freqs):
                    for b in range(a, info.N_freqs):
                        if (freqs_to_use[j][a] == True) and (freqs_to_use[j][b] == True):
                            inv_cov_filename = _inv_cov_filename(info,j,a,b)
                            hp.write_map(inv_cov_filename, inv_cov_maps_temp[count], nest=False, dtype=np.float64, overwrite=False)
                            count+=1
                print('done computing all inverse covariance maps at scale'+str(j))
                del cov_maps_temp #free up memory
            del inv_cov_maps_temp #free up memory
            print('done computing all ILC weights at scale'+str(j))
            ##########################
            # only save these maps of the ILC weights if requested
            if (info.save_weights == 'yes' or info.save_weights == 'Yes' or info.save_weights == 'YES'):
                count=0
                for a in range(info.N_freqs):
                    if (freqs_to_use[j][a] == True):
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
        else:
            weights = np.zeros((int(N_pix_to_use[j]),int(N_freqs_to_use[j])))
            count=0
            for a in range(info.N_freqs):
                if (freqs_to_use[j][a] == True):
                    weight_filename = _weights_filename(info,a,j)
                    weights[:,count] = hp.read_map(weight_filename, dtype=np.float64, )
                    count+=1
        ##########################
        # apply these ILC weights to the needlet coefficient maps to get the per-needlet-scale ILC maps
        ILC_map_temp = np.zeros(int(N_pix_to_use[j]))
        count=0
        for a in range(info.N_freqs):
            if (freqs_to_use[j][a] == True):
                filename_wavelet_coeff_map = info.output_dir+info.output_prefix+'_needletcoeffmap_freq'+str(a)+'_scale'+str(j)+'.fits'
                if not info.apply_weights_to_other_maps:
                    wavelet_coeff_map = hp.read_map(filename_wavelet_coeff_map, dtype=np.float64)
                else:
                    wavelet_coeff_map =maps_for_weights_needlets[a][j]
                wavelet_coeff_map = hp.read_map(filename_wavelet_coeff_map, dtype=np.float64)
                ILC_map_temp += weights[:,count] * wavelet_coeff_map
                count+=1
        ILC_maps_per_scale.append(ILC_map_temp)
    ##########################
    # synthesize the per-needlet-scale ILC maps into the final combined ILC map (apply each needlet filter again and add them all together -- have to upgrade to all match the same Nside -- done in synthesize)
    ILC_map = synthesize(wv_maps=ILC_maps_per_scale, wv=wv, N_side_out=info.N_side)
    # save the final ILC map
    ILC_map_filename = info.output_dir+info.output_prefix+'needletILCmap'+'_component_'+info.ILC_preserved_comp+'_crossILC'*info.cross_ILC+info.output_suffix+'.fits'
    if type(info.N_deproj) is int:
        if N_deproj>0:
            ILC_map_filename = info.output_dir+info.output_prefix+'needletILCmap'+'_component_'+info.ILC_preserved_comp+'_deproject_'+'_'.join(info.ILC_deproj_comps)+'_crossILC'*info.cross_ILC+info.output_suffix+'.fits'
    else:
         if info.N_deproj[0]>0:
            # NOTE: ILCdeprojected file name is not so descriptive here. Need to describe it more in info.output_suffix.
            ILC_map_filename = info.output_dir+info.output_prefix+'needletILCmap'+'_component_'+info.ILC_preserved_comp+'_deproject_'+'_'.join(info.ILC_deproj_comps[0])+'_crossILC'*info.cross_ILC+info.output_suffix+'.fits'


    hp.write_map(ILC_map_filename, ILC_map, nest=False, dtype=np.float64, overwrite=False)
    # make image if requested
    if map_images == True:
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


    # cross-correlate with map specified in input file (if requested; e.g., useful for simulation analyses) -- TODO
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
    # require multipole ell_F where wavelet filter F(ell_F) = wavelet_beam_criterion (on its decreasing side)
    #   to be less than the multipole ell_B where the beam B(ell_B) = wavelet_beam_criterion
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
    FWHM_pix = np.zeros(wv.N_scales,dtype=float)
    ell, filts = wv.TopHatHarmonic(info.ellbins)

    # compute effective number of modes associated with each filter (on the full sky)
    # note that the weights we use are filt^2, since this is the quantity that sums to unity at each ell
    N_modes = np.zeros(wv.N_scales, dtype=float)
    for i in range(wv.N_scales):
        N_modes[i] = np.sum( (2.*ell + np.ones(wv.ELLMAX+1)) * (filts[i])**2. )
    # we use the entire sphere to calculate the covmat - there is no realspace filter
    # so just set these to a large number
    for i in range(wv.N_scales):
        sigma_pix_temp = np.pi * 4
        FWHM_pix[i] = np.sqrt(8.*np.log(2.)) * sigma_pix_temp #in radians
    ##########################
    ##########################
    # compute wavelet decomposition of all frequency maps used at each filter scale
    # save the filtered maps (aka maps of "wavelet coefficients")
    for i in range(info.N_freqs):
        # N.B. maps are assumed to be in strictly decreasing order of FWHM! i.e. info.beams[-1] is highest-resolution beam
        print("waveletizing frequency ", i, "...")
        wv_maps_temp = []
        flag=True
        for j in range(wv.N_scales):
            if freqs_to_use[j][i] == True:
                filename = info.output_dir+info.output_prefix+'_needletcoeffmap_freq'+str(i)+'_scale'+str(j)+'.fits'
                exists = os.path.isfile(filename)
                if exists:
                    print('needlet coefficient map already exists:', filename)
                    wv_maps_temp.append( hp.read_map(filename, dtype=np.float64) )
                else:
                    print('needlet coefficient map not previously computed; re-computing all maps for frequency '+str(i)+' now...')
                    flag=False
                    break
        if flag == False:
            wv_maps_temp = waveletize(inp_map=(info.maps)[i], wv=wv,  rebeam=True, inp_beam=(info.beams)[i], new_beam=info.common_beam, wv_filts_to_use=freqs_to_use[:,i], N_side_to_use=N_side_to_use)
            for j in range(wv.N_scales):
                if freqs_to_use[j][i] == True:
                    filename = info.output_dir+info.output_prefix+'_needletcoeffmap_freq'+str(i)+'_scale'+str(j)+'.fits'
                    hp.write_map(filename, wv_maps_temp[j], nest=False, dtype=np.float64, overwrite=False)
        if map_images == True:
            for j in range(wv.N_scales):
                if freqs_to_use[j][i] == True:
                    plt.clf()
                    hp.mollview(wv_maps_temp[j], unit="K", title="Needlet Coefficient Map, Frequency "+str(i)+" Scale "+str(j), min=np.mean(wv_maps_temp[j])-2*np.std(wv_maps_temp[j]), max=np.mean(wv_maps_temp[j])+2*np.std(wv_maps_temp[j]))
                    plt.savefig(info.output_dir+info.output_prefix+'_needletcoeffmap_freq'+str(i)+'_scale'+str(j)+'.pdf')
        print("done waveletizing frequency ", i, "...")
        if info.cross_ILC:
            for season in [1,2]:
                flag = True
                wv_maps_temp = []
                for j in range(wv.N_scales):
                    if freqs_to_use[j][i] == True:
                        filename = info.output_dir+info.output_prefix+'_needletcoeffmap_freq'+str(i)+'_scale'+str(j)+'_S'+str(season)+'.fits'
                        exists = os.path.isfile(filename)
                        if exists:
                            wv_maps_temp.append( hp.read_map(filename, dtype=np.float64) )
                        else:
                            flag=False
                            break
                if flag == False:
                    if season==1:
                        maps = info.maps_s1
                    elif season==2:
                        maps = info.maps_s2

                    wv_maps_temp = waveletize(inp_map=(maps)[i], wv=wv, rebeam=True, inp_beam=(info.beams)[i], new_beam=newbeam, wv_filts_to_use=freqs_to_use[:,i], N_side_to_use=N_side_to_use)
                    for j in range(wv.N_scales):
                        if freqs_to_use[j][i] == True:
                            filename = info.output_dir+info.output_prefix+'_needletcoeffmap_freq'+str(i)+'_scale'+str(j)+'_S'+str(season)+'.fits'
                            exists2 = os.path.isfile(filename)

                            if not exists2:
                                hp.write_map(filename, wv_maps_temp[j], nest=False, dtype=np.float64, overwrite=False)
                    del(maps) #free up memory
        del wv_maps_temp #free up memory
    ##########################
    ##########################
    ### MAIN ILC CALCULATION ###
    # TODO -- memory management could probably be improved here (reduce file I/O overhead, reduce number of smoothing operations, etc...)
    ILC_maps_per_scale = []
    print("doing main ILC!!",flush=True)
    for j in range(wv.N_scales):
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
            print('weight vector already exists:', weight_filename)
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
                                A_mix[countt][b] = 1.e-6 * (get_mix([info.freqs_delta_ghz[a]], info.ILC_preserved_comp, param_dict_file=info.param_dict_file, param_dict_override=None, dust_beta_param_name='beta_CIB', radio_beta_param_name='beta_radio'))[0] #convert to K from uK
                            elif (info.bandpass_type == 'ActualBandpasses'):
                                A_mix[countt][b] = 1.e-6 * (get_mix_bandpassed([info.freq_bp_files[a]], info.ILC_preserved_comp, param_dict_file=info.param_dict_file, param_dict_override=None, dust_beta_param_name='beta_CIB', radio_beta_param_name='beta_radio'))[0] #convert to K from uK
                        else:
                            if (info.bandpass_type == 'DeltaBandpasses'):
                                A_mix[countt][b] = 1.e-6 * (get_mix([info.freqs_delta_ghz[a]], ILC_deproj_comps[b-1], param_dict_file=info.param_dict_file, param_dict_override=None, dust_beta_param_name='beta_CIB', radio_beta_param_name='beta_radio'))[0] #convert to K from uK
                            elif (info.bandpass_type == 'ActualBandpasses'):
                                A_mix[countt][b] = 1.e-6 * (get_mix_bandpassed([info.freq_bp_files[a]], ILC_deproj_comps[b-1], param_dict_file=info.param_dict_file, param_dict_override=None, dust_beta_param_name='beta_CIB', radio_beta_param_name='beta_radio'))[0] #convert to K from uK
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
            for a in range(info.N_freqs):
                start_at = a
                if info.cross_ILC:
                    start_at = 0
                for b in range(start_at, info.N_freqs):
                    if (freqs_to_use[j][a] == True) and (freqs_to_use[j][b] == True and flag==True):
                        # Note that for HILC the covmat has no pixel index and only needs {freq1, freq2} indices at every scale. So we save it at every scale in a .txt file as a 2-d numpy array
                        cov_filename = info.output_dir+info.output_prefix+'_needletcoeff_covmap_scale'+str(j)+'_crossILC'*info.cross_ILC+'.txt' 
                        exists = os.path.isfile(cov_filename)
                        if exists:
                            print('needlet coefficient covariance map already exists:', cov_filename)
                            cov_matrix_harmonic = np.loadtxt(cov_filename)
                        else:
                            print('needlet coefficient covariance map not previously computed; re-computing all covariance maps at scale'+str(j)+' now...')
                            flag=False
                            break
            if flag == False:
                for a in range(info.N_freqs):
                    cov_matrix_harmonic = np.zeros((int(N_freqs_to_use[j]),int(N_freqs_to_use[j])))
                    counta = 0
                    all_maps_A = []
                    all_maps_B = []

                    all_maps_A_smoothed = []
                    all_maps_B_smoothed = []
                    for a in range(0,info.N_freqs):
                        if freqs_to_use[j][a] :
                            if not info.cross_ILC:
                                map_A = hp.fitsfunc.read_map(info.output_dir+info.output_prefix+'_needletcoeffmap_freq'+str(a)+'_scale'+str(j)+'.fits')
                                map_B = map_A.copy()
                                smooth_map_A = smooth_map_B = hp.sphtfunc.smoothing(map_A, FWHM_pix[j])

                            else:
                                map_A = hp.fitsfunc.read_map(info.output_dir+info.output_prefix+'_needletcoeffmap_freq'+str(a)+'_scale'+str(j)+'_S1.fits')
                                map_B = hp.fitsfunc.read_map(info.output_dir+info.output_prefix+'_needletcoeffmap_freq'+str(a)+'_scale'+str(j)+'_S2.fits')
                                smooth_map_A =  hp.sphtfunc.smoothing(map_A, FWHM_pix[j])
                                smooth_map_B =  hp.sphtfunc.smoothing(map_B, FWHM_pix[j])
                            all_maps_A.append(map_A)
                            all_maps_B.append(map_B)
                            all_maps_A_smoothed.append(smooth_map_A)
                            all_maps_B_smoothed.append(smooth_map_B)
                        else:
                            all_maps_A.append(0)
                            all_maps_B.append(0)
                            all_maps_A_smoothed.append(0)
                            all_maps_B_smoothed.append(0)


                for a in range(info.N_freqs):
                            countb = 0
                            for b in range(0, info.N_freqs): #  Could probably do this quicker by starting at a instead of 0 when not doing cross_ILC but would have to keep track of count_a and count_b
                                if (freqs_to_use[j][a] == True) and (freqs_to_use[j][b] == True):
                                    wavelet_map_A = all_maps_A[a].copy()
                                    wavelet_map_B = all_maps_B[b].copy()
                                    assert len(wavelet_map_A) == len(wavelet_map_B), "cov mat map calculation: wavelet coefficient maps have different N_side"
                                    wavelet_map_A_smoothed = all_maps_A_smoothed[a].copy()
                                    wavelet_map_B_smoothed = all_maps_B_smoothed[b].copy()
                                    # then construct the smoothed real-space freq-freq cov matrix element for this pair of frequency maps
                                    # note that the overall normalization of this cov matrix is irrelevant for the ILC weight calculation (it always cancels out)
                                    cov_map_temp = hp.sphtfunc.smoothing( (wavelet_map_A - wavelet_map_A_smoothed)*(wavelet_map_B - wavelet_map_B_smoothed) , FWHM_pix[j])
                                    cov_matrix_harmonic[counta,countb] = np.mean(cov_map_temp) # Actually the map is a constant so we don't need to take the mean, we could also take any arbitrary value
                                    countb +=1
                            if (freqs_to_use[j][a] == True):
                                 counta +=1
                cov_filename = info.output_dir+info.output_prefix+'_needletcoeff_covmap_scale'+str(j)+'_crossILC'*info.cross_ILC+'.txt'
                print("saving covmat",cov_filename)
                np.savetxt(cov_filename,cov_matrix_harmonic)

            print('done computing the covariance matrix at scale'+str(j),flush=True)
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
                    tempvec[a] = (-1.0)**float(a) * np.linalg.det(np.transpose(QSa_temp,(2,0,1)))
            tmp2 = np.einsum('ia,ap->ip', A_mix, tempvec)
            tmp3 = np.einsum('jip,ip->jp', inv_covmat_temp, tmp2)
            weights = 1.0/np.linalg.det(np.transpose(Qab_pix,(2,0,1)))[:,None] * np.transpose(tmp3) #N.B. 'weights' here only includes channels that passed beam_thresh criterion
            # response verification
            response = np.einsum('pi,ia->ap', weights, A_mix) #dimensions N_comps x N_pix_to_use[j]
            optimal_response_preserved_comp = np.ones(1)
            optimal_response_deproj_comp = np.zeros((N_comps-1, 1))
            if not (np.absolute(response[0]-optimal_response_preserved_comp) < resp_tol).all():
                print(f'preserved component response failed at wavelet scale {j}')
                quit()
            if not (np.absolute(response[1:]-optimal_response_deproj_comp) < resp_tol).all():
                print(f'deprojected component response failed at wavelet scale {j}')
                quit()


            print('done computing all ILC weights at scale'+str(j))
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
        ILC_map_temp = np.zeros(int(N_pix_to_use[j]))
        count=0
        for a in range(info.N_freqs):
            if (freqs_to_use[j][a] == True):
                filename_wavelet_coeff_map = info.output_dir+info.output_prefix+'_needletcoeffmap_freq'+str(a)+'_scale'+str(j)+'.fits'
                wavelet_coeff_map = hp.read_map(filename_wavelet_coeff_map, dtype=np.float64)
                ILC_map_temp += weights[:,count] * wavelet_coeff_map
                count+=1
        ILC_maps_per_scale.append(ILC_map_temp)
    ##########################
    # synthesize the per-needlet-scale ILC maps into the final combined ILC map (apply each needlet filter again and add them all together -- have to upgrade to all match the same Nside -- done in synthesize)
    ILC_map = synthesize(wv_maps=ILC_maps_per_scale, wv=wv, N_side_out=info.N_side)
    # save the final ILC map
    ILC_map_filename = info.output_dir+info.output_prefix+'needletILCmap'+'_component_'+info.ILC_preserved_comp+'_crossILC'*info.cross_ILC+'.fits'
    if type(info.N_deproj) is int:
        if info.N_deproj>0:
            ILC_map_filename = info.output_dir+info.output_prefix+'needletILCmap'+'_component_'+info.ILC_preserved_comp+'_deproject_'+'_'.join(info.ILC_deproj_comps)+'_crossILC'*info.cross_ILC+'.fits'
    else:
        if info.N_deproj[0]>0:
            ILC_map_filename = info.output_dir+info.output_prefix+'needletILCmap'+'_component_'+info.ILC_preserved_comp+'_deproject_'+'_'.join(info.ILC_deproj_comps[0])+'_crossILC'*info.cross_ILC+'.fits'
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
