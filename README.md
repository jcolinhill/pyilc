# pyilc

Needlet ILC in Python

## Installation

pyilc does not need to be installed.

## Requirements

`pyilc` requires python3, [numpy](https://numpy.readthedocs.io/en/latest/), [matplotlib](https://matplotlib.org) and [healpy](https://healpy.readthedocs.io/en/latest/) (and all of their requirements). 

## Basic usage

To run `pyilc`, a `.yaml` input file is required. `pyilc` is run by running the `main.py` file using python with this input file as an argument:
```
python pyilc/main.py sample_input.yaml
```

We have included a sample input file `pyilc_input_example_Fiona.yml` (Fiona to-do: change this name) which serves as documentation of the different input options. The main ones are described here.

### Input/Output structure

The maps on which the (N)ILC will be performed should all be saved separately at some location /path/to/location/map_freq_X.fits . These files should be included in the .yaml file as a list of strings:

```
freq_map_files: ['/path/to/location/map_freq_X.fits',
                 '/path/to/location/map_freq_Y.fits',...]
```
.

Note that there is another input paramater `N_freqs` which is required in the input file, and which **must be** equal to the length of the `freq_map_files` list, or else an error will be thrown.

The ILC products (covariance matrices, inverse covariance matrices, ILC maps, and ILC weights if requested) will be saved in an output folder specified in the input file according to:

```
output_dir: '/path/to/output/'
```
.
