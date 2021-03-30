# Detecting PNe within reduced MUSE data

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tspriggs/MUSE_PNe_fitting/b1dd21304663b97ed3c53771c7054e844c35bb77?filepath=%2FBinder_example.ipynb)
 [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3726795.svg)](https://doi.org/10.5281/zenodo.3726795)

## Binder Instructions

For instructions and a prepared working environment, please open `Binder_example.ipynb`, and read the notes.
To explore the variables, you can run `%whos` to return a list of stored variables from the executed scripts.

## Data requirements

This project uses the output of the GIST IFU pipeline ([[GIST]](https://abittner.gitlab.io/thegistpipeline/)). Where the stellar continuum has been modelled (ppxf), along with the emission lines of the diffuse ionised gas (Gandalf; Sarzi, M. 2006). Once the stellar continuum is subtracted (MUSE_spectra - (Best fit - emission fit) ), we are left with a residual data cube, where the emission lines of unresolved point sources can be mapped out, along with regions of diffuse ionised gas. The residual cube is what is needed for this workflow. To create a residual data cube, please use the gist_residual_data.py script, under `scripts/other`, this will load up the relevant files and create a residual data cube for use with the PNe detection scripts.

## Introduction

This software was first introduced in the method paper of [Spriggs, T. W. et al. 2020](https://ui.adsabs.harvard.edu/abs/2020A%26A...637A..62S/abstract).

The purpose of this pipeline is to first run a spaxel-by-spaxel fit for \[OIII] 5007 Angstrom doublet (4959 Angstrom), then store files and save plots (exported_data/ and Plots/). Then having formed a x,y list of coordinates (pixel coordinates) from running SEP on the A/rN map, fit the PNe with the 1D+2D model fitting routine, using LMfit for the minimisation efforts.

## Directory Structure

Note: if you are willing to change the input and output locations for files, then you can use any naming / directory convention. For the purposes of ease, the convention used herein is explained in full.

### **galaxy_data**

Please store the input data files (residual lists) in folders named: galaxy_data/FCC000_data/, an example would be FCC167_data/. In this folder please place the residual lists .fits file from gandalf fits (containining only the residuals of stellar continuum subtracted spectra, with nebulous emission lines.), as well the wavelength array (saved in npy format) if required.

### **Plots**

The folder Plots/ is where the saved figures / plots from the script are stored. Inside Plots/ there needs to be a folder for each galaxy you intend to use with the pipeline, named as follows: GalaxyName/ (e.g. `Plots/FCC167/` ).

### **exported_data**

The final folder required here is "exported_data/", where the results of the spaxel-by-spaxel fit will be stored. Again a folder within "exported_data/", for each galaxy, needs to be made so as to keep track of exported files.

The directory will look something like this:

```
    MUSE_PNe_fitting/
    │
    ├── galaxy_data/
    │   ├── FCC000_data/
    │   └── FCC000center_residauls_cube.fits
    │   
    ├── Plots/
    │   └──FCC000/
    │                     
    └── exported_data/
        └─ FCC000/
```

## Galaxy information

For storing the galaxy information that is required to detect and fit for PNe, a yaml file is used. The format, as it stands, of the yaml entry for each galaxy is:


    FCCtest_center:
        Galaxy name: FCCtest
        F_corr: 1.0                 # Flux calibration correction
        velocity: 1369              # km/s
        eff_r: 28.6                 # Effective Radius, in arc-secons
        FWHM: 4.0                   # Moffat FWHM and error
        FWHM_err: 0.05
        beta: 2.5                   # Moffat beta parameter and error
        beta_err: 0.1
        LSF: 2.8                    # Gaussian line spread function (FWHM), and error
        LSF_err: 0.01
        centre: [220, 220]          # pixel coordinates for the centre of the galaxy.
        gal_mask: [1,1,1,1,1]       # ellipsoid mask paramters for the galaxy (xe, ye, length, width, angle)
        star_mask: [[1,1,1]]        # circle mask parameters to mask out stars, list of lists. (xc, yc, radius)
        over_lum: []                # index values for over-luminous sources
        impostor_filter: [[],[],[]] # index values for identified impostors: [[SNR], [HII], [impostor]]
        interloper_filter: []       # index values for velocity offset interlopers
        emissions:                  # This is a dictionary of emissions with the required parameter setup.
          OIII_1: [750, null, null]
          OIII_2: [null, Amp_2D_OIII_1/3, 'wave_OIII_1 - 47.9399 * (1+{})']


## Spaxel-by-Spaxel [OIII] fitting

This routine will take in the galaxy FCC000's residual cube data file, fit spaxel by spaxel for \[OIII] doublets, and save the output files and plots in the relevant files. The shape of the galaxy (x and y lengths) are stored in the header now and easy to access.

## PNe detection and PSF analysis

Once you have run MUSE_spaxel_fit.py, then you will want to run SEP on the A/rN map (this is not included in the scripts yet, only in master_book.ipynb). SEP will save a list of x,y pixel coordinates of the detected sources. Once they are read in as minicubes, you need to run the PSF routine to get values for FWHM, beta and LSF, with the relevant errors. On the first fit, to determine the signal to noise ratio, the PSF is defaulted to FWHM=4.0, beta=2.5 and LSF=3.0. For the PSF part, you will need to select a suitable number of sources via the brightest and work downwards. 5 or 10 objects is adequate. This will be incorporated later into a script.

## PNe 3D modelling

After you have a list of PNe coordinates, and the PSF values saved into the galaxy_info.yaml file, you are ready to run the PNe_fitting.py file. To run this script, use:

```bash
$ python scripts/other/PNe_fitting.py FCC000 # command line use
```

### **PNe source fitting with 3D model**

The PNe_fitting.py script uses the PNe minicubes from the MUSE_spaxel_fit.py script, as detected from signal to noise maps, to fit the \[OIII] emission lines using our 3D model. Once all the sources are fitted, the initial catalogue is trimmed of any objects that do not pass the first filters: signal to noise $\geq$ 3, or a $\chi^{2}$ within 3 sigma (99.73 %).

### **Point Spread Function fitting**

This script is also capable of choosing the 4 brightest PNe and fitting them simultaneously to evaluate the Point Spread Function (PSF) for the current pointing.

### **Impostor identification, using GIST**

Beyond the previously mentioned filter, there is also functionality to produce PSF weighted datacubes for the PNe, taken from the reduced (raw) MUSE data, and be ready for further evaluation by the GIST pipeline for impostor determination. This is achieved through emission line diagnostic ratio investigation. Here, sources that meet the associated criteria are labelled as Supernova Remnants (SNR), compact HII regions (HII), or unknowns.

### **Planetary Nebulae Luminosity Function (PNLF) fitting**

The final catalogue of PNe are then passed to a function that fits the Ciardullo et al. 1989 analytical PNLF formulae to the data. The only free parameter of this $\chi^{2}$ minimisation effort is the distance modulus. The result of the PNLF modelling method produces both a distance to the galaxy hosting the observed PNe, it also calculates the expected number of PNe to lie within 2.5 magntitudes of the cut-off.
