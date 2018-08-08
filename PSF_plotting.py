# -*- coding: utf-8 -*-
"""
Created on Tue May 16 13:10:28 2017

@author: TSpriggs
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from astropy.modeling import models, fitting
from astropy.modeling import Fittable2DModel, Parameter
#%%

data = ascii.read("M87_data/M87_PSF.txt")

x_psf = data["x_psf(\")"]
y_psf = data["y_psf(\")"]

psf = data["z_psf"]
psf_1 = data["z_psf_1"]
psf_2 = data["z_psf_2"]

#%% Moffat model

class Moffat2D_OIII(Fittable2DModel):
    amplitude = Parameter(default=1)
    x_0 = Parameter(default=0)
    y_0 = Parameter(default=0)
    gamma = Parameter(default=1)
    alpha = Parameter(default=1)
    bkg = Parameter(default=0)

    # Evaluate the model for the Moffat2D
    @staticmethod
    def evaluate(x,y, amplitude, x_0, y_0, gamma, alpha, bkg):
        rr_gg = ((x - x_0) ** 2 + (y - y_0) ** 2) / gamma ** 2
        return amplitude * (1 + rr_gg) ** (-alpha) + bkg


# Custom 2D Gaussian model
class Gaussian2D_OIII(Fittable2DModel):
    amplitude = Parameter(default=1)
    x_mean = Parameter(default=0)
    y_mean = Parameter(default=0)
    x_stddev = Parameter(default=1)
    y_stddev = Parameter(default=1)
    theta = Parameter(default=0.0)

    @staticmethod
    def evaluate(x, y, amplitude, x_mean, y_mean, x_stddev, y_stddev, theta):
        cost2 = np.cos(theta) ** 2
        sint2 = np.sin(theta) ** 2
        sin2t = np.sin(2. * theta)
        xstd2 = x_stddev ** 2
        ystd2 = y_stddev ** 2
        xdiff = x - x_mean
        ydiff = y - y_mean
        a = 0.5 * ((cost2 / xstd2) + (sint2 / ystd2))
        b = 0.5 * ((sin2t / xstd2) - (sin2t / ystd2))
        c = 0.5 * ((sint2 / xstd2) + (cost2 / ystd2))
        return amplitude * np.exp(-((a * xdiff ** 2) + (b * xdiff * ydiff) +
                                    (c * ydiff ** 2)))

#%%

psf_cube = np.array(psf).reshape(201,201)
psf_1_cube = np.array(psf_1).reshape(201,201)
psf_2_cube = np.array(psf_2).reshape(201,201)

#%%
plt.figure(1)
plt.imshow(psf_cube, vmax=0.0013)
plt.colorbar()

plt.figure(2)
plt.imshow(psf_1_cube, vmax=0.0013)
plt.colorbar()

plt.figure(3)
plt.imshow(psf_2_cube, vmax=0.0013)
plt.colorbar()

#%%

X,Y = np.mgrid[:201,:201]

coordinates = [(x,y) for x in range(201) for y in range(201)]

x = [item[0] for item in coordinates]
y = [item[1] for item in coordinates]

x = np.array(x)
y = np.array(y)
#%%
g_init = Gaussian2D_OIII(amplitude = max(psf), x_mean=100, y_mean=100, x_stddev=1., y_stddev=1.)
fitter = fitting.LevMarLSQFitter()
g_fit = fitter(g_init, x, y, psf)

plt.figure(4)
plt.imshow(g_fit(X, Y))
plt.colorbar()
#%%
M_init = Moffat2D_OIII(x_0=100, y_0=100 )
fitter = fitting.LevMarLSQFitter()
M_fit = fitter(M_init, x, y, psf)

x_c = 100.
y_c = 100.

# TODO Get Fluxes from Moffat2D fit
plt.figure(5)
plt.imshow(M_fit(X, Y))
plt.colorbar()
#%%
r = np.sqrt((x-x_c)**2+(y-y_c)**2)

plt.figure(6)
plt.scatter(r, psf, marker=".", color="r")
plt.scatter(r, M_fit(x, y), marker="x", color="b", alpha=0.5)
plt.xlim(0, 60)
plt.ylim(0, max(psf)*1.05)
plt.show()
