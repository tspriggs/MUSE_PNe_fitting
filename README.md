# Detecting PNe within reduced MUSE data

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tspriggs/MUSE_PNe_fitting/fec5a221e1759e9671fcdb75acba88f46b36e525?urlpath=lab) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3698303.svg)](https://doi.org/10.5281/zenodo.3698303)

# Binder Instructions

For instructions and a prepared working environment, please open `notebooks/Binder_example.ipynb`, and read the notes.
To explore the variables, you can run `%whos` to return a list of stored variables from the executed scripts.

## Data requirements

This project uses the output of the GIST IFU pipeline ([[GIST]](https://abittner.gitlab.io/thegistpipeline/)). Where the stellar continuum has been modelled (ppxf), along with the emission lines of the diffuse ionised gas (Gandalf; Sarzi, M. 2006). Once the stellar continuum is subtracted (MUSE_spectra - (Best fit - emission fit) ), we are left with a residual data cube, where the emission lines of unresolved point sources can be mapped out, along with regions of diffuse ionised gas. The residual cube is what is needed for this workflow. To create a residual data cube, please use the gist_residual_data.py script, under `scripts/other`, this will load up the relevant files and create a residual data cube for use with the PNe detection scripts.

## Introduction - WIP

The purpose of this pipeline is to first run a spaxel-by-spaxel fit for [OIII] 5007 Angstrom doublet (4959 Angstrom), then store files and save plots (exported_data/ and Plots/). Then having formed a x,y list of coordinates (pixel coordinates) from running SEP on the A/rN map, fit the PNe with the 1D+2D model fitting routine, using LMfit for the minimisation efforts.

## Directory Structure
Note: if you are willing to change the input and output locations for files, then you can use any naming / directory convention. For the purposes of ease, the convention used herein is explained in full.

### Data
Please store the input data files (residual lists) in folders named: galaxy_data/FCC000_data/, an example would be FCC167_data/. In this folder please place the residual lists .fits file from gandalf fits (containining only the residuals of stellar continuum subtracted spectra, with nebulous emission lines.), as well the wavelength array (saved in npy format) if required.

### Plots
The folder Plots/ is where the saved figures / plots from the script are stored. Inside Plots/ there needs to be a folder for each galaxy you intend to use with the pipeline, named as follows: GalaxyName/, (e.g. FCC167/ ).

### Exported_data
The final folder required here is "exported_data/", where the results of the spaxel-by-spaxel fit will be stored. Again a folder within "exported_data/", for each galaxy, needs to be made so as to keep track of exported files.

The directory will look something like this:

    MUSE_PNe_fitting/
        galaxy_data/

            FCC000_data/
            
                FCC000_residauls_cube.fits
                
            
        Plots/
        
            FCC000/
                            
        exported_data/
        
            FCC000/

## Galaxy information

For storing the galaxy information that is required to detect and fit for PNe, a yaml file is used. The format, as it stands, of the yaml entry for each galaxy is:

    FCC000_center:
        Distance: 20 # this is in Mpc
        Galaxy name: FCC000
        velocity: 1500 # km/s
        FWHM: 4.00 # Moffat FWHM and error
        FWHM_err: 0.01
        beta: 2.5 # Moffat beta parameter and error
        beta_err: 0.1
        LSF: 3.0 # Gaussian line spread function (FWHM)
        LSF_err: 0.1
        gal_mask: [1,1,1,1,1] # Used for making an ellipsoid for masking regions of the galaxy (xe, ye, length, width, angle)
        star_mask: [[1,1,1],[1,1,1]] # used for masking out stars, list of lists required, even if one list in a list. (xc, yc, radius)
        emissions: # This is a dictionary of emissions with the required parameter setup.
            OOIII_1: [200, null, null, null]
            OIII_2: [null, Amp_2D_OIII_1/2.85, 'wave_OIII_1 - 47.9399 * (1+{})', null]

## Spaxel-by-Spaxel [OIII] fitting

This routine will take in the galaxy FCC000's residual cube data file, fit spaxel by spaxel for [OIII] doublets, and save the output files and plots in the relevant files. The shape of the galaxy (x and y lengths) are stored in the header now and easy to access.

## PNe detection and PSF analysis

Once you have run MUSE_spaxel_fit.py, then you will want to run SEP on the A/rN map (this is not included in the scripts yet, only in master_book.ipynb). SEP will save a list of x,y pixel coordinates of the detected sources. Once they are read in as minicubes, you need to run the PSF routine to get values for FWHM, beta and LSF, with the relevant errors. On the first fit, to determine the signal to noise ratio, the PSF is defaulted to FWHM=4.0, beta=2.5 and LSF=3.0. For the PSF part, you will need to select a suitable number of sources via the brightest and work downwards. 5 or 10 objects is adequate. This will be incorporated later into a script.

## PNe 3D modelling

After you have a list of PNe coordinates, and the PSF values saved into the galaxy_info.yaml file, you are ready to run the PNe_fitting.py file. To run this script, use:
*python PNe_fitting.py FCC000 # terminal use
*ipython PNe_fitting.py FCC000 # ipython terminal
*%run PNe_fitting.py FCC000 # inside ipython

This will open up the galaxy, extract the PNe minicubes, and perform fits on them using the PSF values previously set. Once fitted, the script will evaluate the PNLF and then the bolometric luminosity of the integrated MUSE spectra (i.e. stars). This uses the same masks as for PNe detection.
Depending on RAM, this may either fail due to RAM limitaitons, or take a while, however with 8-16GB, it should be farely rapid (5-10 mins maximum).

Furhter documentation is in the works, with functions and changes within the scripts also being documented within the script itself.
