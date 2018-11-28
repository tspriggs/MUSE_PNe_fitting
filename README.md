# Detecting PNe within reduced MUSE data.

## Intro

The purpose of this pipeline is to first run a spaxel-by-spaxel fit for [OIII] 5007 Angstrom doublet (4959 Angstrom), then store files and save plots. Then having formed a x,y list of coordinates (pixel coordinates), fit the PNe with the 3D model fitting routine, using LMfit for the minimisation efforts.

## Directory Structure
Note: if you are willing to change the input and output locations for files, then you can use any naming / directory convention. For the purposes of ease, the convention used herein is explained in full.

### Data
Please store the input data files (residual cubes) in folders named: galaxyName_data/, an example would be FCC167_data/, or FCC255_data/. In this folder please place the residual cubes from gandalf fits (containining only the residuals of stellar continuum subtracted spectra, with nebulous emission lines.), as well the wavelength array (saved in npy format) if required.

### Plots
The folder Plots/ is where the saved figures / plots from the script will be saved. Inside Plots/ there needs to be a folder for each galaxy you intend to use with the pipeline, named as follows: GalaxyName/, (e.g. FCC167/ or FCC255/). Then inside each folder should be another folder, named "full_sepc_fits/", this is where the pipeline will save the ntegrated spectrum and 1D associated fit plot to.

### Exported_data
The final folder required here is "exported_data/", where the results of the spaxel-by-spaxel fit will be stored. Again a folder within "exported_data/", for each galaxy, needs to be made so as to keep track of exported files.

The directory will look something like this:
Working_directory/
    -FCC000_data/
        -FCC000_residauls_list.fits
        -FCC000_wavelength.npy
    -Plots/
        -FCC000/
            -full_spec_fits/
    -exported_data/
        -FCC000/

## Galaxy information
In this pipeline, a yaml file is used to store and read from the information of a galaxy that you intend to analyse and fit.
The format, as it stands, of the yaml entry for each galaxy is:

FCC000:
    Distance: 20 # this is in Mpc
    Galaxy name: FCC000
    emissions: # This is a dictionary of emissions with the required parameter setup.
        OIII_1: [100, null, null]
        ha: [5, null, 'wave_OIII_1 + 1556.375 * (1+{})']
        hb: [5, null, 'wave_OIII_1 - 145.518 * (1+{})']
        OIII_2: [5, "Amp_2D_OIII_1/3", 'wave_OIII_1 - 47.9399 * (1+{})']
    residual cube: FCC000_data/FCC000_residuals_list.fits # Load in the galaxy residuals data list
    fit 1D: N # Use Y to fit spaxel-by-spaxel, then after this the script will update to N
    z: 0.005 # redshift from simbad.

## Spaxel-by-Spaxel [OIII] fitting



## PNe 3D modelling
