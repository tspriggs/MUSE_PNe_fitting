import sys
import yaml
import lmfit
import argparse
import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from scipy.stats import norm
from astropy.table import Table
from astropy.io import ascii, fits
from astropy.wcs import WCS, utils, wcs
from astropy.coordinates import SkyCoord
from matplotlib.patches import Rectangle, Ellipse, Circle
from lmfit import minimize, Minimizer, report_fit, Model, Parameters
from MUSE_Models import PNe_residuals_3D, PNe_spectrum_extractor, PSF_residuals_3D, robust_sigma

with open("galaxy_info.yaml", "r") as yaml_data:
    galaxy_info = yaml.load(yaml_data, Loader=yaml.FullLoader)

# Queries sys arguments for galaxy name
my_parser = argparse.ArgumentParser()

my_parser.add_argument('--galaxy', action='store', type=str, required=True)
args = my_parser.parse_args()
galaxy_name = args.galaxy

galaxy_data = galaxy_info[galaxy_name]

DATA_DIR = "galaxy_data/"+galaxy_name+"_data/"+galaxy_name
EXPORT_DIR = "exported_data/"+galaxy_name+"/"+galaxy_name
PLOT_DIR = "Plots/"+galaxy_name+"/"+galaxy_name

# Load in the residual data, in list form
hdulist = fits.open(DATA_DIR+"_residuals_list.fits") # Path to data
res_hdr = hdulist[0].header # extract header from residual cube

# Check to see if the wavelength is in the fits fileby checking length of fits file.
if len(hdulist) == 2: # check to see if HDU data has 2 units (data, wavelength)
    wavelength = np.exp(hdulist[1].data)
    np.save(DATA_DIR+"_wavelength", wavelength)
else:
    wavelength = np.load(DATA_DIR+"_wavelength.npy")

# Will introduce method that takes from header the x and y dimensions
    
# Use the length of the data to return the size of the y and x dimensions of the spatial extent.
x_data = res_hdr["XAXIS"]
y_data = res_hdr["YAXIS"]
   
# Indexes where there is spectral data to fit. We check where there is data that doesn't start with 0.0 (spectral data should never be 0.0).
non_zero_index = np.squeeze(np.where(hdulist[0].data[:,0] != 0.))

# Constants
n_pixels = 9 # number of pixels to be considered for FOV x and y range
c = 299792458.0 # speed of light

gal_vel = galaxy_data["velocity"] 
z = gal_vel*1e3 / c 
D = galaxy_data["Distance"] # Distance in Mpc - from Simbad / NED - read in from yaml file
gal_mask = galaxy_data["gal_mask"]

# Construct the PNe FOV coordinate grid for use when fitting PNe.
coordinates = [(n,m) for n in range(n_pixels) for m in range(n_pixels)]
x_fit = np.array([item[0] for item in coordinates])
y_fit = np.array([item[1] for item in coordinates])

# Defines spaxel by spaxel fitting model
def spaxel_by_spaxel(params, x, data, error, spec_num):
    """
    Using a Gaussian double peaked model, fit the [OIII] lines at 4959 and 5007 Angstrom, found within Stellar continuum subtracted spectra, from MUSE.
    Inputs:
        Params - Using the LMfit python package, contruct the parameters needed and read them in:
                Amplitude of [OIII] at 5007 A.
                mean wavelength position of [OIII] 5007 A peak.
                FWHM of Gaussian profiles.
                Gaussian backrgound level of residuals.
                Gaussian gradient of background residuals.
        x - Wavelength array
        data - read in sprectrum by spectrum of data via list form.
        error - associated errors for each spectrum.
        spec_num - from enumerate, just the index number of spectrum, for storing value sin np array.

    Returns -  (Data - model) / error   for chi square minimiser.
    """
    Amp = params["Amp"]
    wave = params["wave"]
    FWHM = params["FWHM"]
    Gauss_bkg = params["Gauss_bkg"]
    Gauss_grad = params["Gauss_grad"]

    Gauss_std = FWHM / 2.35482 # FWHM to Standard Deviation calculation.

    model = ((Gauss_bkg + Gauss_grad * x) + Amp * np.exp(- 0.5 * (x - wave)** 2 / Gauss_std**2.) +
             (Amp/2.85) * np.exp(- 0.5 * (x - (wave - 47.9399*(1+z)))** 2 / Gauss_std**2.))

    # Saves both the Residual noise level of the fit, alongside the 'data residual' (data-model) array from the fit.
    list_of_rN[spec_num] = robust_sigma(data - model)
    data_residuals[spec_num] = data - model

    return (data - model) / error


################################################################################
#################################### Fit 1D ####################################
################################################################################

# Run Spaxel by Spaxel fit of the spectra within the .fits file.
# Check if fit_1D parameter, within the Galaxy_info.yaml file is set to Y (yes to fit), or N (no to fit - has been fitted before).
print("Spaxel by Spaxel fit underway...")

# Run Spaxel by Spaxel fitter
print("Fitting Spaxel by Spaxel for [OIII] doublet.")

list_of_std = np.abs([robust_sigma(dat) for dat in hdulist[0].data])
input_errors = [np.repeat(item, len(wavelength)) for item in list_of_std] # Intially use the standard deviation of each spectra as the uncertainty for the spaxel fitter.

# Setup numpy arrays for storage of best fit values.
gauss_A = np.zeros(len(hdulist[0].data))
list_of_rN = np.zeros(len(hdulist[0].data))
data_residuals = np.zeros((len(hdulist[0].data),len(wavelength)))
obj_residuals = np.zeros((len(hdulist[0].data),len(wavelength)))

# setup LMfit paramterts
spaxel_params = Parameters()
spaxel_params.add("Amp",value=150., min=0.001)
spaxel_params.add("wave", value=5006.77*(1+z), min=(5006.77*(1+z))-15, max=(500677*(1+z))+15) #Starting position calculated from redshift value of galaxy.
spaxel_params.add("FWHM", value=galaxy_data["LSF"], vary=False) # Line Spread Function
spaxel_params.add("Gauss_bkg", value=0.01)
spaxel_params.add("Gauss_grad", value=0.0001)

# Loop through spectra from list format of data.
for j,i in tqdm(enumerate(non_zero_index), total=len(non_zero_index)):
    #progbar(j, len(non_zero_index), 40)
    fit_results = minimize(spaxel_by_spaxel, spaxel_params, args=(wavelength, hdulist[0].data[i], input_errors[i], i), nan_policy="propagate")
    gauss_A[i] = fit_results.params["Amp"].value
    obj_residuals[i] = fit_results.residual

A_rN = np.array([A / rN for A,rN in zip(gauss_A, list_of_rN)])
Gauss_F = np.array(gauss_A) * np.sqrt(2*np.pi) * 1.19

# Save A/rN, Gauss A, Guass F and rN arrays as npy files. Change to .fits soon maybe
np.save(EXPORT_DIR+"_A_rN_cen", A_rN)
np.save(EXPORT_DIR+"_gauss_A_cen", gauss_A)
np.save(EXPORT_DIR+"_gauss_F_cen", Gauss_F)
np.save(EXPORT_DIR+"_rN", list_of_rN)

# save the data and obj res in fits file format to us memmapping.
hdu_data_res = fits.PrimaryHDU(data_residuals)
hdu_obj_res = fits.PrimaryHDU(obj_residuals)
hdu_data_res.writeto("exported_data/"+ galaxy_name +"/"+galaxy_name+"_resids_data.fits", overwrite=True)
hdu_obj_res.writeto("exported_data/"+ galaxy_name +"/"+galaxy_name+"_resids_obj.fits", overwrite=True)

print("Cube fitted, data saved.")

# Construct A/rN, A_5007 and F_5007 plots, and save in Plots/Galaxy_name/
# Plot A/rN
plt.figure(figsize=(20,20))
plt.imshow(A_rN.reshape(y_data, x_data), origin="lower", cmap="CMRmap", vmin=1, vmax=8)
plt.colorbar()
plt.savefig(PLOT_DIR+"_A_rN_map.png")

# Plot A_5007
plt.figure(figsize=(20,20))
plt.imshow(gauss_A.reshape(y_data, x_data), origin="lower", cmap="CMRmap", vmin=10, vmax=100)
plt.colorbar()
plt.savefig(PLOT_DIR+"_A_5007_map.png")

# Plot F_5007
plt.figure(figsize=(20,20))
plt.imshow(Gauss_F.reshape(y_data, x_data), origin="lower", cmap="CMRmap", vmin=10, vmax=100)
plt.colorbar()
plt.savefig(PLOT_DIR+"_F_5007_map.png")

print("Plots saved in Plots/"+galaxy_name)