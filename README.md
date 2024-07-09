# pyilc

pyilc is a pure-python implementation of the needlet internal linear combination (NILC) algorithm for CMB component separation.  Harmonic-space ILC is also implemented in the code.  For details, see McCarthy & Hill (2023) [arXiv:2307.01043](https://arxiv.org/abs/2307.01043).

## diffusive_inpaint
This repository also includes an inpainting code, diffusive_inpaint, that diffusively inpaints a masked region with the mean of the unmasked neighboring pixels. This README is intended for pyilc, not diffusive_inpaint; in the diffusive_inpaint sub-directory, we include a sample .py file `diffusive_inpaint/diffusive_inpaint_example.py`, which should make clear how to use `diffusive_inpaint/diffusive_inpaint.py`.

# Requirements

`pyilc` requires python3, [numpy](https://numpy.readthedocs.io/en/latest/), [matplotlib](https://matplotlib.org), and [healpy](https://healpy.readthedocs.io/en/latest/) (and all of their requirements). 

# Using the code

`pyilc` is public; if you use it in a publication, please cite the paper [arXiv:2307.01043](https://arxiv.org/abs/2307.01043) and (optionally) [arXiv:2308.16260](https://arxiv.org/2308.16260). 

Additionally, if you use NILC you should cite the original NILC reference, https://ui.adsabs.harvard.edu/abs/arXiv:0807.0773.

If you use an ILC that deprojects some component, please cite the constrained ILC references https://ui.adsabs.harvard.edu/abs/2009ApJ...694..222C/abstract and https://ui.adsabs.harvard.edu/abs/arXiv:1006.5599.  

If using a moment-based deprojection, please cite https://ui.adsabs.harvard.edu/abs/arXiv:1701.00274.

See [here](https://ed1b6aae-4cdf-4f9b-a94d-f1aadd0cab78.filesusr.com/ugd/6acf9d_de251625176e4fb1bd3cb40e737ec096.pdf) for slides with an overview of how to use pyilc, presented to the SO FG AWG (Simons Observatory Foreground Analysis Working Group).

# Basic usage

To run `pyilc`, a `yaml` input file is required. `pyilc` is run by running the `main.py` file using python with this input file as an argument:
```
python pyilc/main.py sample_input.yml
```

We have included a sample input file `pyilc_input_example_general.yml`, which serves as documentation of the different input options. The main ones are described here.

We go into detail about the input and output structure below. In general, an input file will contain a list of input frequency maps **in units of K_CMB** on which the NILC is to be performed, along with a path specifying what directory the output should be saved in and a prefix and suffix with which to save the products. The output products that are saved are the needlet coefficients of the input maps, the (maps of) covariance and inverse frequency-frequency covariance matrices, and the ILC maps (and ILC weights if requested in the input file). Before performing NILC, the code will check whether these products already exist in the specified output directory with the specified **prefix** (in the case of the input needlet coefficients and the covariance and inverse covariance matrices) and the specified **prefix AND suffix** (in the case of the weights and final ILC map): 
* If the map or weights exist, the code will not compute anything as the products already exist.  
* If the covariance/inverse covariance matrices exist, the code will load these and use these to compute the final ILC weights and map, then save the weights/map with the specified **prefix AND suffix**. 
* If no covariance products exist, the code will compute these, save them with the specified **prefix**, then use them to compute the weights and map and save them with the specified **prefix AND suffix**. 

As the covariance matrix computation/inversion is often the most computationally expensive step in performing NILC, this allows several versions of ILC maps (different deprojections, SEDs, etc.) to be computed with the same input without recomputing the covariances.


### Examples

There are some examples of usage in the notebooks/ folder, where we explicitly run a .yaml file from the input/ folder which will reproduce the ILC calculation undeprojected y map in [arXiv:2307.01043](https://arxiv.org/abs/2307.01043) . Note that it first downloads the preprocessed input single-frequency maps from https://users.flatironinstitute.org/~fmccarthy/ymaps_PR4_McCH23/inpainted_input_maps/ (the preprocessing is done with diffusive_inpaint.py and is described in  [arXiv:2307.01043](https://arxiv.org/abs/2307.01043) ). There is also an example notebook which performs HILC on the Planck data to create a temperature map. The HILC calculation is much faster than the NILC calculation, and this can be run very quickly.

## Input structure


### Specifying the input maps

The maps on which the (N)ILC will be performed should all be saved separately at some location `/path/to/location/map_freq_X.fits`. These files should be included in the .yml file as a list of strings:

```
freq_map_files: ['/path/to/location/map_freq_X.fits',
                 '/path/to/location/map_freq_Y.fits',...]
```

Note that there is another input parameter `N_freqs` that is required in the input file, and which **must be** equal to the length of the `freq_map_files` list, or else an error will be thrown.  The maps must be listed in order of decreasing resolution, as specified by the user-specified input beams, as described below.
#### Beams
There is also additional metadata about the input maps that must be included in the input file. In particular the **beams** with which the maps are convolved must be specified. There are two options: Gaussian beams, or more general, one-dimensional ($\ell$-dependent) beams. Gaussian beams are specified as follows:
```
beam_type: 'Gaussians'  
beam_FWHM_arcmin: [FWHM_X,FWHM_Y,...]
```
where FWHM_X is the FWHM of the beam, in arcminutes, of the map at `/path/to/location/map_freq_X.fits`. **The FWHMS must be listed in the same order as the maps they correspond to in `freq_map_files`**. Also, **the FWHMS must be decreasing**, i.e., the maps must be read in in order from lowest-resolution to highest-resolution, or else an error will be thrown (note this might mean that they are **not** read in in a monotonically increasing frequency order, e.g., if the maps come from different instruments).

Alternatively, 1D beams can be specified as follows:
```
beam_type: '1DBeams'
beam_files:['/path/to/location/beam_freq_X.txt',
            '/path/to/location/beam_freq_Y.txt',...]
```
where '/path/to/location/beam_freq_X.txt' contains an array of shape (LMAX,2) where the first column specifies the $\ell$ and the second column specifies the beam at $\ell$. LMAX should be at least as high as the ELLMAX (maximum multipole) at which the NILC is being performed (this is a user-specified parameter in the input file -- see below).

**The maps should all be in units of CMB thermodynamic temperature, $\mathrm{K}_{\mathrm{CMB}}$**.


#### Frequency coverage

In order to calculate the SED of a given component, the code needs to know what frequencies the maps correspond to. There are two options: delta-function passbands and performing passband integration. The former can be indicated by including the following in the input file:
```
bandpass_type: 'DeltaBandpasses'
freqs_delta_ghz: [freq_X, freq_Y, ...]
```
where freq_X refers to the frequency (**in GHz**) of the appropriate input map (the order should be the same as that of freq_map_files). To perform passband integration, the following should be indicated in the input file:
```
bandpass_type: 'ActualBandpasses'
freq_bp_files: ['/path/to/location/bandpass_freq_X.txt',
                '/path/to/location/bandpass_freq_Y.txt',,...]
               
```
where '/path/to/location/bandpass_freq_X.txt' is a file containing the passband (first column = frequency in GHz, second column = transmission in the same convention as used in Planck) of the appropriate input map (again, the order should be the same as that of freq_map_files).

### Specifying the resolution of the output ILC map

The maximum multipole (ELLMAX) used in all spherical harmonic transformations should be specified with
```
ELLMAX: ELLMAX
```
The N_side of the output map should be specified with
```
N_side: N_side
```
This should be the same N_side as the highest-resolution input map.

Optionally, the FWHM (in arcminutes) at which NILC should be performed (and with which the output map is convolved) can be specified according to 
```
perform_ILC_at_beam: FWHM
```
If this is **not** specified, the ILC will be performed on maps which are all convolved to the same beam as that of the highest-resolution input map.

### Specifying the type of ILC to perform

Currently, `pyilc` can perform two types of ILC: NILC with Gaussian needlets and harmonic ILC. 

#### Gaussian NILC

For Gaussian NILC with N_scales needlet scales, constructed from Gaussians with FWHMs of FWHM_1,..., the following needs to be included in the input file:
```
wavelet_type: 'GaussianNeedlets' 
N_scales: N_scales
GH_FWHM_arcmin: [FWHM_1,FWHM_2,...,]
```
GH_FWHM_arcmin should be a list of the FWHMs in arcminutes of the Gaussians which are used to define the needlet scales. There must be N_scales-1 entries. They must be in decreasing order.

#### Real-space filter size

The size of the real-space domains used for calculating the covariances is defined adaptively in the code by specifying a threshold for the ILC bias and calculating the area required such that the number of modes within the area will be enough to unsure the fractional ILC bias is lower than this threshold. This threshold can be modified by specifying
```
ILC_bias_tol: b_tol
```
If unspecified, this is set by default to 0.01.

#### HILC
For harmonic ILC (HILC), the following needs to be included in the input file

```
wavelet_type: 'TopHatHarmonic'
BinSize: 20
```
BinSize is **one integer** which specifies the width in $\ell$-space (i.e., $\Delta_\ell$) of the bins on which the HILC is calculated. (Functionality for more general bins will perhaps be included in a later release; however this can be easily modified in pyilc/input.py if a user needs, by modifying the specification of `ILCInfo.ell_bins`.)

### Specifying the components to preserve and deproject

By default, `pyilc` can preserve and deproject any of the following components:

```
['CMB','kSZ','tSZ','rSZ','mu','CIB','CIB_dbeta','CIB_dT','radio']
```
where 'CMB' and 'kSZ' both refer to a blackbody (CMB+kSZ) component; 'tSZ' refers to the Compton-y distortion; 'rSZ' refers to the relativistic thermal SZ (modeled with a third-order Taylor expansion in kT_e/(m_e c^2)); 'mu' refers to the $\mu$-distortion; 'CIB' refers to the cosmic infrared background, which is specified by a modified blackbody SED with a spectral index beta and temperature T; 'CIB_dbeta' refers to the first moment with respect to beta of this modified blackbody; 'CIB_dT' refers to the first moment with respect to T of this modified blackbody; and 'radio' refers to a power-law SED (in specific intensity units).  The CIB and radio SEDs are also available in Jy/sr rather than the default uK_CMB units that are used internally in `pyilc/fg.py`. (Note: as stated above, input frequency maps are always assumed to be in K_CMB.)

In all cases, the SED is calculated at the frequencies specified in freqs_delta_ghz (for delta-function passbands) or integrated over the passband of the maps specified in freq_bp_files (for actual realistic passbands). This is computed in `pyilc/fg.py`. 

**Output units**: Due to the internal use of uK_CMB units in the code, a CMB-preserved output ILC map is in units of **uK_CMB**.  A kSZ-preserved output ILC map is also in uK_CMB.  A tSZ-preserved output ILC map is in dimensionless Compton-y units.  Output mu-distortion or rSZ ILC maps are also in analogous dimensionless units associated with these distortions.  For all other components (CIB, radio, etc.), an ILC map that preserves one of these components will not have a meaningful absolute normalization (note that such a normalization is *not* needed in order to deproject these components).  If you would like to make an ILC map that preserves one of these component SEDs, consult the modeling in `pyilc/fg.py` to determine how you would like to normalize the output.

**Some of these SEDs depend on specified parameters**. The values of the parameters should be saved in a spearate yaml file; we provide a default `input/fg_SEDs_default_params.yml` . If a different parameter file is desired, this can be read in in the overall input file as follows:
```
param_dict_file: /path/to/input/alternative_paramfile.yml
```
If this is unspecified, the default values will be used.  The parameters that can be changed are the temperature for the relativistic SZ evaluation (`kT_e_keV`); and the effective dust temperature and spectral index for the CIB modified blackbody (`Tdust_CIB` and `beta_CIB` respectively). 

For an ILC preserving component 'AAA', specify:
```
ILC_preserved_comp: 'AAA'
```
in the input file. To deproject some number N_deproj of components in an ILC, include:
```
N_deproj: N_deproj
ILC_deproj_comps: ['BBB','CCC',...]
```
ILC_deproj_comps should be a list of length N_deproj. For an unconstrained ILC, this should read
```
N_deproj: 0
ILC_deproj_comps: []
```
All of the components should be in the list COMP_TYPES in `pyilc/input.py`, or else an error will be thrown.  (You can also easily add your own desired components there.)

#### Different deprojections on different scales

Instead of specifying one set of components to deproject on all scales, one can specify different components to deproject on different needlet scales. 
In this case, N_deproj should be modified to be a list of length N_scales, with each entry specifying the number of components to deproject at that scale.
ILC_deproj_comps should similarly be modified to be a list (of length N_scales) of lists, with each entry specifying the list of components (of length corresponding to N_deproj at that scale) to deproject at that scale.

#### Specifying new components to preserve and deproject

If you want to preserve or deproject a new component whose SED you can parametrize, you should do the following:

* Add a string specifying your component to the list COMP_TYPES in `pyilc/input.py`
* Add the SED to `fg.py` by following the implementation of any of the other components. Note that you will have to specify the SED both for delta-function passbands and the SED that should be integrated over a passband for the actual-passbands case. Note that as the internal SED computations in `fg.py` are all done in units of $\mu\mathrm{K}_{\mathrm{CMB}}$, you may need to convert from specific intensity to temperature units with the conversion function dBnudT(nu).  (As stated above, your input frequency maps to the code itself should all be in units of K_CMB -- yes, this differs from the internal computations in `fg.py', for primarily legacy-code-related reasons.)


## Output structure

**Note: this section only refers to the case when NILC is performed (not HILC)**

The ILC products (needlet coefficients, covariance matrices, inverse covariance matrices, ILC maps, and ILC weights if requested) will be saved in an output directory `/path/to/output/` with a prefix `output_prefix` specified in the input file by strings. A suffix `output_suffix`, which will **only** be appended to the ILC map and weight filenames (i.e., not covariance products) can also be specified:

```
output_dir: '/path/to/output/'
output_prefix: 'output_prefix'
output_suffix: 'output_suffix'
```

All output products will be saved in `/path/to/output/` as follows:

### Needlet coefficients of input maps

The needlet coefficients of the input maps are computed for each frequency (labelled by $X\in 0,..., N_{freq}-1$ and each scale $A\in 0,...,N_{scales}-1$) and saved as:

```
/path/to/output/output_prefix_needletcoeffmap_freqX_scaleA.fits
```

### Covariances and inverse covariances
Elements of covariance matrices are saved as 
```
/path/to/output/output_prefix_needletcoeff_covmap_freqX_freqY_scaleA.fits
```
In NILC, the covariance matrices are computed **at every pixel**, so each pixel has an N_freq x N_freq symmetric covariance matrix associated with it. These are saved as $\frac{N_{freq}\times (N_{freq}+1)}{2}$ healpix maps, with each map (labeled by $X, Y$, for $X, Y \in 0,...,N_{freq}-1$ and by the needlet scale $A\in 0,...,N_{scales}-1$.

Similarly, the **inverse** covariance matrices are saved as 
```
/path/to/output/output_prefix_needletcoeff_invcovmap_freqX_freqY_scaleA.fits
```
Note that the total covariance matrices are inverted **in frequency space** , i.e., **separately for each pixel**, before being saved.

Also note that the number of frequencies is different for different needlet scales, as determined by a beam threshold criterion specified in the code (one does not want to use frequency maps with low-resolution beams in the ILC on high-$\ell$ needlet scales).  Thus the dimensionality of the covariance matrix changes as a function of needlet scale.  In addition, `pyilc` minimizes memory usage by downgrading the maps used at low-$\ell$ needlet scales.  Thus the number of pixels in the covariance and inverse covariance matrix maps is lower for low-$\ell$ needlet scales than high-$\ell$ needlet scales.

### ILC weights and maps

#### ILC weights

To save the ILC weights, include
```
save_weights: 'yes'
```
in the input file; otherwise, 
```
save_weights: 'no'
```
should be specified. The weights depend on the component being deprojected. For an unconstrained ILC that preserves component AAA, they will be saved at
```
/path/to/output/output_prefix_needletcoeff_weightmap_freqX_scaleA_component_AAA_output_suffix.fits
```
For a constrained ILC that preserves component AAA and deprojects components BBB,CCC,DDD, the weights will be saved at
```
/path/to/output/output_prefix_needletcoeff_weightmap_freqX_scaleA_componentAAA_deproject_BBB_CCC_DDD_output_suffix.fits
```
Note that the use of output_suffix here (and not in the covariance matrices) allows the same covariance matrices to be used to construct different versions of the output maps, for example by modifying the SED of a component to be deprojected.  This allows rapid construction of many ILC maps after computing the covariances and inverse covariances only once.

#### ILC maps

For an unconstrained ILC that preserves component AAA, the final ILC map will be saved at
```
/path/to/output/output_prefix_needletILCmap_component_AAA_output_suffix.fits
```
For a constrained ILC that preserves component AAA and deprojects components BBB,CCC,DDD, the ILC map will be saved at
```
/path/to/output/output_prefix_needletILCmap_component_AAA_deproject_BBB_CCC_DDD_output_suffix.fits
```
Again the use of output_suffix here (and not in the covariance matrices) allows the same covariance matrices to be used to construct different versions of the output maps, for example by modifying the SED of a component to be deprojected.


# More complicated usage

`pyilc` has various extended functionality, if required. We list some of the other abilities of the code here.

## Applying weights to alternative maps

Sometimes it is useful to apply the weights calculated from one map to a separate map, for example when testing on simulations to quantify exactly how much of a specific component exists in the final output map. To propagate the weights directly through another map, the following can be specified in the input file:
```
maps_to_apply_weights: [...]
```
where maps_to_apply_weights is a list of filenames in the same format as freq_map_files. If the weights have been previously calculated and saved, they will be read in and applied to these maps. If they have not been, they will be calculated from the maps in input_maps and applied to the maps in maps_to_apply_weights.

Note that it is important to directly change the output suffix if you are using this option, as doing this does not automatically change the output file name of the ILC map (recall that changing the output suffix does not change the covariance files read in as long as the output prefix is unchanged).


## Cross-ILC

`pyilc` can also perform Cross-ILC (see https://ui.adsabs.harvard.edu/abs/2014JCAP...02..030H/abstract), where the covariance matrices are computed only from independent splits of the maps; the overall statistic to be minimized in this case is the variance caused by **foregrounds**, and not instrumental noise. In order to specify that `pyilc` should perform cross-ILC, inlcude:
```
cross_ILC: 'True'
```
in the input (note that this `True' is a **string** and not the boolean True). In this case, two independent splits of each frequency map should be read in.  They should be included as follows:
```
freq_map_files_s1: [...]
freq_map_files_s2: [...]
```
where freq_map_files_s1 and freq_map_files_s2 are both lists with the same format as freq_map_files but with filenames that point to the appropriate split maps. Note that freq_map_files should still be included, as the weights will still be applied to the maps in freq_map_files. Also note that the covariance, inverse covariance, and ILC map filenames will all be modified to include the term '_crossILC'.

# Acknowledgments

If using the code, please cite [arXiv:2307.01043](https://arxiv.org/abs/2307.01043) and (optionally) [arXiv:2308.16260](https://arxiv.org/abs/2308.16260).  The primary authors of `pyilc` are Colin Hill (jch2200@columbia.edu) and Fiona McCarthy (fmccarthy@flatironinstitute.org).  Additional contributors include Mathew Madhavacheril and Kristen Surrao.  We thank Mathieu Remazeilles for sharing details about the Planck NILC y-map analysis and Will Coulton for useful NILC comparisons.  We thank Jens Chluba for discussions about the moment expansion method.  We additionally thank Aleksandra Kusiak, Blake Sherwin, and David Spergel for useful conversations, as well as Shivam Pandey for a helpful check of our NILC y-maps.  We are also grateful to Boris Bolliet for useful discussions regarding the halo model and [class_sz](https://github.com/CLASS-SZ/class_sz).
