import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, Circle
from astropy.io import ascii, fits
from astropy.table import Table
from lmfit import minimize, Minimizer, report_fit, Model, Parameters
import lmfit
import pandas as pd
from MUSE_Models import MUSE_3D_OIII, MUSE_3D_residual, PNextractor, PSF_residuals
from ppxf import robust_sigma

#First load in the relevant data
hdulist = fits.open("FCC167_data/FCC167_OIII_line_center.fits")
hdr = hdulist[0].header
raw_data = hdulist[0].data
y_data = hdr["NAXIS2"]
x_data = hdr["NAXIS1"]
wavelength = np.exp(hdr['CRVAL3']+np.arange(hdr["NAXIS3"])*hdr['CDELT3'])

# swap axes to y,x,wavelength - THIS MAY NO BE NEEDED
raw_data_list = np.array(raw_data).reshape(len(wavelength), x_data*y_data)
raw_data_list = np.swapaxes(raw_data_list, 1, 0)
# Check for nan values
raw_data_cube = raw_data_list.reshape(y_data, x_data, len(wavelength))

#Read in x and y coordinates of PNe - RE work
x_y_list = np.load("exported_data/FCC167/sep_x_y_list.npy")
x_PNe = np.array([x[0] for x in x_y_list])
y_PNe = np.array([y[1] for y in x_y_list])

# constants
n_pixels= 13
z = 0.006261
c = 299792458.0

coordinates = [(n,m) for n in range(n_pixels) for m in range(n_pixels)]
x_fit = np.array([item[0] for item in coordinates])
y_fit = np.array([item[1] for item in coordinates])

flatten = lambda l: [item for sublist in l for item in sublist]

# Retrieve the respective spectra for each PNe source
PNe_spectra = np.array([PNextractor(x, y, n_pixels, raw_data_cube, wave=wavelength, dim=2.0) for x,y in zip(x_PNe, y_PNe)])

# create Pandas data frame for values
PNe_df = pd.DataFrame(columns=("PNe number", "Total Flux", "Flux error", "V (km/s)", "m 5007", "M 5007", "M 5007 error","A/rN"))
PNe_df["PNe number"] = np.arange(1,len(x_PNe)+1)

check_for_1D_fit = input("Do you want to run the 1D fitter?: (y/n)")

if check_for_1D_fit == "y":
    # Run 1D fitter
elif check_for_1D_fit == "n":
    # load from saved files

#Run 1D fit of the spectra and save relevant outputs
## potentially check to see if 1D needs to be run, or load from files with an input() call


#run initial 3D fit on selected objects

#run psf fit using objective residuals

#determine PSF values and feed back into 3D fitter

#Fit PNe with updated PSF

#Run the rest of the analysis