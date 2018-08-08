# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from astropy.modeling import models, fitting
from astropy.modeling import Fittable1DModel, Parameter
#%%
#read in the data file and asign column headers from the file.
data1D = ascii.read("for_Thomas_1D_spec.txt", names =["lambda (Angstrom)", "flux density (10^-20 erg/s/cm2/A/arcsec2)"] )
data2D = ascii.read("for_Thomas_2d_ima.txt", names = ["x (arcsec)","y (arcsec)","surf.brightness 10^-20 erg/s/cm2/arcsec2"])

# assign each column to variables for easier reading later on
x_col_1D = "lambda (Angstrom)"
y_col_1D = "flux density (10^-20 erg/s/cm2/A/arcsec2)"

x_col_2D = "x (arcsec)"
y_col_2D = "y (arcsec)"
z_col_2D = "surf.brightness 10^-20 erg/s/cm2/arcsec2"

# assign x and y to the respective data column from 'data'
wavelength = data1D[x_col_1D]
flux_density = data1D[y_col_1D]

x_2D = data2D[x_col_2D]
y_2D = data2D[y_col_2D]
z_2D = data2D[z_col_2D]

#%%

# Define custom two peak Gaussian model using Astropy class type.
class Gaussian1D_OIII(Fittable1DModel):
    amplitude = Parameter()
    mean = Parameter()
    stddev = Parameter()
    a = Parameter()
    b = Parameter()
    
    # evaluate the model. This is composued using the Fittable1DModel class from Astropy.
    @staticmethod
    def evaluate(x, amplitude, mean, stddev, a, b):
        model = a + b*x + np.abs(amplitude) * np.exp(- 0.5 * (x - mean) ** 2 / (np.sqrt((stddev)**2. + 1.16**2.)) ** 2) + np.abs(amplitude)/3 * np.exp(- 0.5 * (x - (mean - 47.9399)) ** 2 / (np.sqrt((stddev)**2. + 1.16**2.)) ** 2)
        return model


#%%
# single gaussian plotted at lambda_max position.
# Fit the data using a Gaussian

# curve fit attempt
g_init = Gaussian1D_OIII(amplitude=max(flux_density), mean=5045, stddev=3.0, a=0., b=0.)#, fixed={"stddev":True})
fitter = fitting.LevMarLSQFitter()
g_fit = fitter(g_init, wavelength, flux_density, maxiter=1000000000)

plt.clf()
#plot
plt.plot(wavelength, flux_density, 'k', label= "Data")
plt.plot(wavelength, g_init(wavelength), 'g', label= "Data")
plt.plot(wavelength, g_fit(wavelength), 'r', label = "1D Gaussian Model fit")
plt.xlabel("Wavelength ($\AA$)")
plt.ylabel("Flux Density ($10^{-20}$ $erg s^{-1}$ $cm^{-2}$ $\AA^{-1}$ $arcsec^{-2}$)")
plt.title("Double 1D Gaussian fit of O[III] line")
plt.legend()
plt.show()

print(g_fit)
#%%
print("Param_cov: ", "\n", fitter.fit_info['param_cov'])
print("Cov_x: ", "\n", fitter.fit_info['cov_x'])
print(fitter.fit_info['message'])

print(' ')

