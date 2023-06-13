# pyilc

Needlet ILC in Python

# Requirements

`pyilc` requires python3, [numpy](https://numpy.readthedocs.io/en/latest/), [matplotlib](https://matplotlib.org) and [healpy](https://healpy.readthedocs.io/en/latest/) (and all of their requirements). 

# Basic usage

To run `pyilc`, a `.yaml` input file is required. `pyilc` is run by running the `main.py` file using python with this input file as an argument:
```
python pyilc/main.py sample_input.yaml
```

We have included a sample input file `pyilc_input_example_Fiona.yml` (Fiona to-do: change this name) which serves as documentation of the different input options. The main ones are described here.

We go into detail about the input and output structure below. 
In general, an input file will contain a list of input frequency maps on which the NILC is to be performed, along with a path specifying what directory the output should be saved in and a prefix and suffix to save the products with. 
The output products that are saved are the needlet coefficients of the input maps; the covariance and inverse covariance matrices; and the ILC maps (and ILC weights if requested in the input file). 
Before performing NILC, the code will check whether these products already exist in the specified output directory with the specified **prefix** (in the case of the input needlet coefficients and the covariance and inverse covariance matrices) and the specified **prefix AND suffix** (in the case of the weights and final ILC map): 
* If the map or weights exist, the code will not compute anything as the products already exist.  
* If the covariance/inverse covariance matrices exist, the code will load these and use these to compute the final ILC weights and map, then save the weights/map with the specified **prefix AND suffix**. 
* If no covariance products exist, the code will compute these, save them with the specified **prefix**, then use them to compute the weights and map and save them with the specified **prefix AND suffix**. 

As the covariance matrix computation/inversion is often the costliest step in performing a NILC, this allows several versions of ILC maps (different deprojections, SEDs, etc...) to be computed with the same input without recomputing the covariances.


## Input structure


### Specifying the input maps

The maps on which the (N)ILC will be performed should all be saved separately at some location `/path/to/location/map_freq_X.fits`. These files should be included in the .yaml file as a list of strings:

```
freq_map_files: ['/path/to/location/map_freq_X.fits',
                 '/path/to/location/map_freq_Y.fits',...]
```

Note that there is another input paramater `N_freqs` which is required in the input file, and which **must be** equal to the length of the `freq_map_files` list, or else an error will be thrown.  The maps must be listed in order of decreasing resolution, as specified by the user-specified input beams, as described below.
#### Beams
There is also some additional metadata about the input maps that must be included in the input file. In particular the **beams** with which the maps are convolved must be specified. There are two options: Gaussian beams, or more general, 1-dimensional ($\ell$-dependent) beams. Gaussian beams are specified as follows:
```
beam_type: 'Gaussians'  
beam_FWHM_arcmin: [FWHM_X,FWHM_Y,...]
```
where FWHM_X is the FWHM of the beam, in arcminutes, of the map at `/path/to/location/map_freq_X.fits`. **The FWHMS must be listed in the same order as the maps they correspond to in `freq_map_files`**. Also, **The FWHMS must be decreasing**, ie the maps must be read in in order from lowest-resolution to highest-resolution, or else an error will be thrown (note this might mean that they are **not** read in in a monotonic frequency order).

Alternatively, 1D beams can be specified as follows:
```
beam_type: '1DBeams'
beam_files:['/path/to/location/beam_freq_X.txt',
            '/path/to/location/beam_freq_Y.txt',...]
```
where '/path/to/location/beam_freq_X.txt' contains an array of shape (LMAX,2) where the first column specifies the $ell$s and the second column specifies the beam at $\ell$. LMAX should be at least as high as the LMAX at which the NILC is being performed (this is a user-specified parameter in the input file).

**The maps should all be in units of $\mu \mathrm{K}_{\mathrm{CMB}}$**


#### Frequency coverage

In order to calculate the SED of a given component, the code needs to know what frequencies the maps correspond to. There are two options: delta-bandpasses and performing bandpass integration. The former can be indicated by including the following in the input file:
```
bandpass_type: 'DeltaBandpasses'
freqs_delta_ghz: [freq_X, freq_Y, ...]
```
where freq_X refers to the frequency (**in gHZ**) of the appropriate input map (the order should be the same as that of freq_map_files). To perform bandpass interation, the following should be indicated in the input file:
```
bandpass_type: 'ActualBandpasses'
freq_bp_files: ['/path/to/location/bandpass_freq_X.txt',
                '/path/to/location/bandpass_freq_Y.txt',,...]
               
```
where '/path/to/location/bandpass_freq_X.txt' is a file containing the bandpass (check format!!) of the appropriate input map (again, the order should be the same as that of freq_map_files).

### Specifying the resolution of the output ILC map

the Lmax used in all spherical harmonic transformations should be specified with
```
ELLMAX: Lmax
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
If this is **not** specified, the ILC will be performed on maps which have are all convolved to the same beam as that of the highest-resolution input map.

### Specifying the type of ILC to perform

Currently, pyilc can perform two types of ILC: NILC with Gaussian needlets, and harmonic ILC. 

#### Gaussian NILC

For Gaussian NILC with N_scales needlet scales with FWHMs of FWHM_1,..., the following needs to be included in the input file:
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
If unspecified, this is set to 0.01.

#### HILC
For HILC, the following needs to be included in the input file

```
wavelet_type: 'TopHatHarmonic'
BinSize: 20
```
BinSize is **one integer** which specifies the width in $\ell$-space of ($\Delta_ell$) of the bins on which the HILC is calculated. (Functionality for more general bins will perhaps be included in a later release; however this can be easily modified in pyilc/input.py if a user needs, by modifying the specification of `ILCInfo.ell_bins`)

### Specifying the components to preserve and deproject

By default, pyilc can preserve and deproject any of the following components:

```
['CMB','kSZ','tSZ','rSZ','mu','CIB', 'CIB_dbeta','CIB_dT']
```
where 'CMB' and 'kSZ' both refer to a black-body (CMB+kSZ) component;  'tSZ' refers to the Compton-$y$ distortion; 'rSZ' refers to the relativistic thermal SZ; 'mu' refers to the $\mu$-distortion; 'CIB' refers to the Cosmic Infrared background, which is specified by a modified black body SED; 'CIB_dbeta' refers to the first moment with respect to beta (the spectral index) of this modified black body; and $CIB_dT$ refers to the first moment with respect to T (the temperature) of this modified black body.

In any case, the SED is calculated at the frequencies specified in freqs_delta_ghz (for delta-bandpasses) or integrated over the bandpass of the maps specified in freq_bp_files (for actual-bandpasses). This is done in `pyilc/fg.py`. 

**Some of these SEDs depend on specified parameters**. The values of the parameters should be saved in a spearate yaml file; we provide a default `input/fg_SEDs_default_params.yml` . If a different parameter file is desired, this can be read in in the overall input file as follows:
```
param_dict_file: /path/to/input/alternative_paramfile.yml
```
If this is unspecified, the default will be used.  The parameters that can be changed are the temperature for the relativistic SZ evaluation (`kT_e_keV`); and the effective dust temperature and spectral index for the CIB modified black body (`Tdust_CIB` and `beta_CIB` respectively). 

For ILC preserving component 'AAA', specify:
```
ILC_preserved_comp: 'AAA'
```
in the input file. To deproject some number N_deproj of components, include:
```
N_deproj: N_deproj
ILC_deproj_comps: ['BBB','CCC',...]
```
ILC_deproj_comps should be a list of length N_deproj. For unconstrained ILC, this should read
```
N_deproj: 0
ILC_deproj_comps: []
```
All of the components should be in the list  COMP_TYPES in pyilc/input.py, or else an error will be thrown.

#### Different deprojections on different scales

Instead of specifying one deprojected components, one can specify different components to deproject on different needlet scales. 
In this case, N_deproj should be modified to be a list of length N_scales , with each entry specifying the number of 

#### Specifying new components to preserve and deproject

If you want to preserve or deproject a new component whose SED you can parametrize, you should do the following:

* Add a string specifying your component to the list COMP_TYPES in pyilc/input.py
* Add the SED to the fg.py by following the implementation of any of the other components. Note that you will have to specify the SED both for delta-bandpasses and the SED that should be integrated over a bandpass for the actual-bandpasses case. Note that as the maps are always read in in $\mu\mathrm{K}_{\mathrm{CIB}}$  you may need to convert from intensity to temperature with the conversion function dBnudT(nu).



## Output structure

**Note: this section only refers to the case when NILC is performed (not HILC)**

The ILC products (needlet coefficients, covariance matrices, inverse covariance matrices, ILC maps, and ILC weights if requested) will be saved in an output folder `/path/to/output/` with a prefix `output_prefix` specified in the input file by strings. A suffix `output_suffix`, which will **only** be appended to ILC map and weight filenames (ie, not covariance products) can also be specified:


```
output_dir: '/path/to/output/'
output_prefix: 'output_prefix'
output_suffix: 'output_suffix'
```

All output products will be saved in `/path/to/output/` as follows:

### Needlet coefficients of input maps

The needlet coefficients of the input maps are computed for each frequency (labelled by $X\in 0,..., N_freq-1$ and each scale $A\in 0,...,N_{scales}-1$) and saved as:

```
/path/to/output/output_prefix_needletcoeffmap_freqX_scaleA.fits
```

### Covariance and inverse covariances
Elements of covariance matrices as 
```
/path/to/output/output_prefix_needletcoeff_covmap_freqX_freqY_scaleA.fits
```
In NILC, the covariance matrices are computed **at every pixel**, so each pixel has an N_freq x N_freq symmetric covariance matrix associated with it. These are saved as $\frac{N_{freq}\times (N_{freq}+1)}{2}$ healpy maps, with each map (labeled by $X, Y$, for $X, Y \in 0,...,N_{freq}-1$ and by the needlet scale $A\in 0,...,N_{scales}-1$.

Similarly, the **inverse** covariance matrices are saved as 
```
/path/to/output/output_prefix_needletcoeff_invcovmap_freqX_freqY_scaleA.fits
```
Note that the total covariance matrices are inverted **in frequency space** , ie **separately for each pixel**, before being saved.

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
should be specified. The weights depend on the component being deprojected. For unconstrained ILC preserving component AAA, they will be saved at
```
/path/to/output/output_prefix_needletcoeff_weightmap_freqX_scaleA_component_AAA_output_suffix.fits.
```
For constrained ILC preserving component AAA and deprojecting components BBB,CCC,DDD, the weights will be saved at
```
/path/to/output/output_prefix_needletcoeff_weightmap_freqX_scaleA_componentAAA_deproject_BBB_CCC_DDD_output_suffix.fits
```
Note that the use of output_suffix (and not in the covariance matrices) here allows the same covariance matrices to be used to create different versions of the output maps, for example by modifying the SED of a component to be deprojected.

#### ILC maps

For unconstrained ILC preserving component AAA, the final ILC map will be saved at
```
/path/to/output/output_prefix_needletILCmap_component_AAA_output_suffix.fits.
```
Constrained ILC preserving component AAA and deprojecting components BBB,CCC,DDD will be saved at
```
/path/to/output/output_prefix_needletILCmap_component_AAA_deproject_BBB_CCC_DDD_output_suffix.fits.
```
Again the use of output_suffix (and not in the covariance matrices) here allows the same covariance matrices to be used to create different versions of the output maps, for example by modifying the SED of a component to be deprojected.


# More complicated usage

pyilc has various extended functionality, if required. We list some of the other abilities of pyilc here.

## Applying weights to alternative maps

Sometimes it is useful to apply the weights calculated from one map to a separate map, for example when testing on simulations to quantify exactly how much of a specific component there is in the final output. To propogate the weights directly through another map, the following can be specified in the input file:
```
maps_to_apply_weights: [...]
```
where maps_to_apply_weights is a list of filenames in the same format as freq_map_files. Note that it is importnat to directly change the output suffix if you are using this option as doing this does not automatically change the output file name of the ILC map (Recall that changing the output suffix does not change the covariance files read in as long as the output prefix is unchanged).

components deprojected at the relevant needlet scale. Also, ILC_deproj_comps should be a list of length N_scales where each entry is a list of length of the corresponding N_deproj for that scale.


## Cross-ILC

pyilc can perform Cross-ILC, where the covariance matrices are computed only from independent splits of the maps; the overall statistic to be minimized in this case is the variance caused by **foregrounds**, and not instrumental noise. In order to specify that pyilc should perform cross-ILC, inlcude:
```
cross_ILC: 'True'
```
in the input (note that this `True' is a **string** and not the boolean True). In this case, two independent splits should be read in; they should be included as follows:
```
freq_map_files_s1: [...]
freq_map_files_s2: [...]
```
where freq_map_files_s1 and freq_map_files_s2 are both lists with the same format as freq_map_files but with filenames that point to the appropriate split maps. Note that freq_map_files should still be included, as the weights will still be applied to the maps in freq_map_files. Also note that the covariance / inverse covariance and ILC map filenames will all be modified to include the term '_crossILC'.
