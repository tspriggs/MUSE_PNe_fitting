import os
import sys
import sep
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
from photutils import CircularAperture
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
my_parser.add_argument('--loc', action="store", type=str, required=True)
my_parser.add_argument("--s", action="store_true", default=False)

args = my_parser.parse_args()

galaxy_name = args.galaxy
loc = args.loc
save_PNe = args.s

galaxy_data = galaxy_info[f"{galaxy_name}_{loc}"]

DATA_DIR = f"galaxy_data/{galaxy_name}_data/{galaxy_name}{loc}"
EXPORT_DIR = f"exported_data/{galaxy_name}/{galaxy_name}{loc}"
PLOT_DIR = f"Plots/{galaxy_name}/{galaxy_name}{loc}"

# Load in the residual data, in list form
hdulist = fits.open(DATA_DIR+"_residuals_list.fits") # Path to data
res_hdr = hdulist[0].header # extract header from residual cube

# Check to see if the wavelength is in the fits fileby checking length of fits file.
if len(hdulist) == 2: # check to see if HDU data has 2 units (data, wavelength)
    wavelength = np.exp(hdulist[1].data)
    np.save(DATA_DIR+"_wavelength", wavelength)
else:
    wavelength = np.load(DATA_DIR+"_wavelength.npy")
    
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
# D = galaxy_data["Distance"] # Distance in Mpc - from Simbad / NED - read in from yaml file
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
g_bkg  = np.zeros(len(hdulist[0].data))
g_grad = np.zeros(len(hdulist[0].data))
list_of_mean = np.zeros(len(hdulist[0].data))

# setup LMfit paramterts
spaxel_params = Parameters()
spaxel_params.add("Amp",value=150., min=0.001)
spaxel_params.add("wave", value=5006.77*(1+z), min=(5006.77*(1+z))-25, max=(500677*(1+z))+25) #Starting position calculated from redshift value of galaxy.
spaxel_params.add("FWHM", value=galaxy_data["LSF"], vary=False) # Line Spread Function
spaxel_params.add("Gauss_bkg", value=0.01)
spaxel_params.add("Gauss_grad", value=0.0001)

# Loop through spectra from list format of data.
if os.path.isfile(f"exported_data/{galaxy_name}/{galaxy_name}{loc}_A_rN_cen.npy") != True:
    for j,i in tqdm(enumerate(non_zero_index), total=len(non_zero_index)):
        #progbar(j, len(non_zero_index), 40)
        fit_results = minimize(spaxel_by_spaxel, spaxel_params, args=(wavelength, hdulist[0].data[i], input_errors[i], i), nan_policy="propagate")
        gauss_A[i] = fit_results.params["Amp"].value
        obj_residuals[i] = fit_results.residual
        g_bkg[i]  = fit_results.params["Gauss_bkg"].value
        g_grad[i] = fit_results.params["Gauss_grad"].value

    A_rN = np.array([A / rN for A,rN in zip(gauss_A, list_of_rN)])
    gauss_F = np.array(gauss_A) * np.sqrt(2*np.pi) * 1.19


    # Save A/rN, Gauss A, Guass F and rN arrays as npy files. Change to .fits soon maybe
    np.save(EXPORT_DIR+"_A_rN_cen", A_rN)
    np.save(EXPORT_DIR+"_gauss_A_cen", gauss_A)
    np.save(EXPORT_DIR+"_gauss_F_cen", gauss_F)
    np.save(EXPORT_DIR+"_rN", list_of_rN)

    # save the data and obj res in fits file format to us memmapping.
    hdu_data_res = fits.PrimaryHDU(data_residuals)
    hdu_obj_res = fits.PrimaryHDU(obj_residuals)
    hdu_data_res.writeto(f"{EXPORT_DIR}_resids_data.fits", overwrite=True)
    hdu_obj_res.writeto(f"{EXPORT_DIR}_resids_obj.fits", overwrite=True)

    print("Cube fitted, data saved.")

else:
    print(f"Spaxel fit data for {galaxy_name} {loc} already exist.")
    # load up gauss_A, gauss_F and A_rN
    gauss_A = np.load(f"{EXPORT_DIR}_gauss_A_cen.npy")
    gauss_F = np.load(f"{EXPORT_DIR}_gauss_F_cen.npy")
    A_rN    = np.load(f"{EXPORT_DIR}_A_rN_cen.npy")


# Construct A/rN, A_5007 and F_5007 plots, and save in Plots/Galaxy_name/
# Plot A/rN
plt.figure(figsize=(20,20))
plt.imshow(A_rN.reshape(y_data, x_data), origin="lower", cmap="CMRmap", vmin=1, vmax=8)
plt.title("A/rN")
plt.colorbar()
plt.savefig(PLOT_DIR+"_A_rN_map.png")

# Plot A_5007
plt.figure(figsize=(20,20))
plt.imshow(gauss_A.reshape(y_data, x_data), origin="lower", cmap="CMRmap", vmin=10, vmax=100)
plt.title("Amplitude")
plt.colorbar()
plt.savefig(PLOT_DIR+"_A_5007_map.png")

# Plot F_5007
plt.figure(figsize=(20,20))
plt.imshow(gauss_F.reshape(y_data, x_data), origin="lower", cmap="CMRmap", vmin=10, vmax=100)
plt.title("Flux")
plt.colorbar()
plt.savefig(PLOT_DIR+"_F_5007_map.png")

print("Plots saved in Plots/"+galaxy_name)



############################################################################
################################# run SEP  #################################
############################################################################

A_rN_img = A_rN.reshape(y_data,x_data)

# Where element is equal to element [0,0], set equal to 0.0, essentially making out of bound areas equal to 0.0
A_rN_img[A_rN_img == A_rN_img[0,0]] = 0.0

plt.figure(figsize=(20,20))

# analyse background noise using sep.background
bkg = sep.Background(A_rN_img, bw=7, bh=7, fw=3, fh=3)

bkg_image = bkg.rms()

gal_mask_params = galaxy_data["gal_mask"]
star_mask_params = galaxy_data["star_mask"]

Y, X = np.mgrid[:y_data, :x_data]

# set up the mask parameters, as taken from the yaml file. default is [0,0,0,0,0]
if loc == "middle" or loc == "halo":
    xe, ye, length, width, alpha = [0,0,0,0,0]
else:
    xe, ye, length, width, alpha = gal_mask_params

elip_mask_gal = (((X-xe) * np.cos(alpha) + (Y-ye) * np.sin(alpha)) / (width/2)) ** 2 + (((X-xe) * np.sin(alpha) - (Y-ye) * np.cos(alpha)) / (length/2)) ** 2 <= 1    

# mask out any known and selected stars
star_mask = np.sum([(Y - yc)**2 + (X - xc)**2 <= rc**2 for xc,yc,rc in star_mask_params],0).astype(bool)

# Use sep.extract to get the locations of sources
objects = sep.extract(A_rN_img-bkg, thresh=2.0, clean=True, minarea=6, err=bkg.globalrms, mask=elip_mask_gal+star_mask, deblend_nthresh=4,)
x_sep = objects["x"]
y_sep = objects["y"]

positions = (x_sep, y_sep)
apertures = CircularAperture(positions, r=4)
plt.figure(figsize=(16,16))
plt.imshow(A_rN_img-bkg, origin="lower", cmap="CMRmap", vmin=1, vmax=8.)
apertures.plot(color="green")

# Add on the eliptical mask (if there is one)
ax = plt.gca()
elip_gal = Ellipse((xe, ye), width, length, angle=alpha*(180/np.pi), fill=False, color="white")
ax.add_artist(elip_gal)

# store list of objects, and print number of detected objects
sep_x_y_list = [[x,y] for x,y in zip(x_sep, y_sep)]
print(len(x_sep))

x_y_list = np.array([[x,y] for x,y in zip(x_sep, y_sep)])
x_PNe = np.array([x[0] for x in x_y_list])
y_PNe = np.array([y[1] for y in x_y_list])

if save_PNe == True:
    np.save(EXPORT_DIR+"_PNe_x_y_list", sep_x_y_list)