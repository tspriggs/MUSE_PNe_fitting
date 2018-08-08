# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 16:00:51 2017

@author: TSpriggs
"""
import time
import numpy as np
from astropy.io import ascii, fits
from astropy.modeling import fitting
from astropy.table import Table
from MUSE_Models import Gaussian1D_OIII

# TODO Move to stddev of empty Flux density region as error.
# TODO work out new error propagation for Flux calculation.

#%%
# Read in list of Wavelength values from ascii file.
data = ascii.read("M87_data/wavelength.txt", names=["wavelength"])
wavelength = data["wavelength"]

# Open Fits file and assign to tdata
hdulist = fits.open("M87_data/M87_rescube.fits")
tdata = (hdulist[0].data)

std_MUSE = 1.16 # MUSE instrumental resolution
#%% Functions and classes


# Function to flatten an array.
flatten = lambda l: [item for sublist in l for item in sublist]

# Fitter
fitter_LevMar = fitting.LevMarLSQFitter()

# Chi Square function
def chi_square(residuals, A_var, std_var):
    chi_sq = [np.sum(item**2) for item in residuals]

    ndof = len(wavelength) - 4 # number of degrees of freedom.
    chi_2_r = np.divide(chi_sq, ndof) # Reduced Chi square
    factor = np.sqrt(chi_2_r)
    A_err = np.sqrt(A_var) * factor
    std_err = np.sqrt(std_var) * factor

    return chi_sq, chi_2_r, A_err, std_err


#%% 1D Gaussian modelling and value gathering for later manipulation and plotting.

start = time.time()
gauss_params = [] # for best-fitting parameters
list_of_residuals = np.zeros((len(tdata), len(wavelength))) # for residuals
cov_x = [] # storing the covariant matrix from each fit.
param_cov = []

list_of_std = [np.abs(np.std(spec[140:])) for spec in tdata]
list_of_PNe_errors = [np.repeat(list_of_std[i], len(wavelength)) for i in np.arange(0,len(list_of_std))]

# model_spectra = []
for i, spectra in enumerate(tdata):

    g_init = Gaussian1D_OIII(amplitude=20., mean=5007.,
                             stddev=0.0, bkg=0.1, grad=0.01, fixed={'stddev':True}) # Initial guesses.
    g_fit= fitter_LevMar(g_init, wavelength, spectra, maxiter=1000000000) # Fitting process
    #model_spectra.append(g_fit(wavelength))
    list_of_residuals[i,:] = fitter_LevMar.fit_info["fvec"] # Append an array of the residuals from the 1D fitter.
    gauss_params.append(g_fit.parameters) # Extract the A, Mean Lambda and stddev values for later evaluation.
    cov_x.append(fitter_LevMar.fit_info["cov_x"]) # Extract Covarient Matrix returned by the fit, append to cov_x.
    param_cov.append(fitter_LevMar.fit_info["param_cov"])


# For timing purposes
end = time.time()
elapsed = end-start
print(elapsed/60, "minutes")

#%% Section to manually look over the returned values, useful for finding statistical facts about each section.

g_amplitudes = [np.abs(item[0]) for item in gauss_params]
g_means = [item[1] for item in gauss_params]
g_stddev = [item[2] for item in gauss_params]

# retrieve from cov_x the error on A from each fit, if cov_x is None, then make it equal to 30.0
#cov_x_A_var = []
#for item in cov_x:
#    if item is not None:
#        cov_x_A_var.append(item[0][0])
#    elif item == None:
#        cov_x_A_var.append(0.0)


cov_x_A_var = []
cov_x_stddev_var = []
for item in param_cov:
    if item is not None:
        cov_x_A_var.append(item[0][0])
        cov_x_stddev_var.append(item[2][2])
    elif item == None:
        cov_x_A_var.append(0.0)
        cov_x_stddev_var.append(0.0)

# Calculate the Residual noise.
list_of_res_N = [np.std(item) for item in list_of_residuals]
list_of_res_N = np.array(list_of_res_N)

#%% Compute Chi Square statistic from function
chi_sq, chi_sq_r, A_err, std_err = chi_square(np.array(list_of_residuals), np.array(cov_x_A_var), np.array(cov_x_stddev_var))

#%%
# Gaussian amplitudes
Gauss_A = np.array(g_amplitudes)
#A_err = np.sqrt(cov_x)

# Gaussian standard deviations, applying instrumental
Gauss_std = np.sqrt(np.array(g_stddev)**2 + std_MUSE**2)

# 1D Gaussian Fluxes calculated from amplitudes and stddev
Gauss_F = np.array(Gauss_A) * np.sqrt(2*np.pi) * np.array(Gauss_std)
F_err = Gauss_F * np.sqrt( (A_err / Gauss_A)**2)

# The following checks are to clear inf and nan values 1D data, reducing potential 2D fitting errors.
F_err[F_err == 0.0] = 100
F_err[np.isnan(F_err)] = 100

# Calculate the wieghts for initial 2D fitting run.
F_weights = 1 / F_err

# Amplitude divided by Residual Noise
A_by_rN = Gauss_A / list_of_res_N

#%% Output Table for file writing.
data_table = Table(data=(Gauss_F, F_err, F_weights, A_by_rN, Gauss_A, A_err, g_means),
                   names=("Gaussian Fluxes", "Flux error"," Flux weights", "A/rN", "Best Fit Amplitude", "Amplitude Error", "mean wavelength"))

#ascii.write(data_table, "exported_data/Gaussian1D_data.txt", overwrite=True)
