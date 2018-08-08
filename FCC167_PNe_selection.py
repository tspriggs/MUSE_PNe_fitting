# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 13:28:01 2018

@author: tspri
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii, fits
#from astropy.modeling import fitting, FittableModel, Fittable2DModel, Fittable1DModel, Parameter, custom_model


#%%
# functions
#Read in Wavelength and slice to region of interest.
#data = ascii.read("FCC167_data/FCC167_restframed_wavelength_forThomas.txt", names=["rest-frame wavelength, assuming a Vsys = 1860.35 km/s"])
#full_wavelength = np.array(data["rest-frame wavelength, assuming a Vsys = 1860.35 km/s"])
#wavelength = full_wavelength[175:344]

# Open Fits file and assign to raw_data
#hdulist = fits.open("FCC167_data/FCC167_residuals_forThomas.fits")
raw_data = hdulist[0].data
# Reshape into a datacube 318x315x271
Flux_data = raw_data[:,175:344]
Flux_data_shape = Flux_data.reshape((441, 444, len(wavelength)))

#%%

A_rN = np.load("exported_data/M87/A_rN.npy")
A_rN_shape = A_rN.reshape(318,315)

gauss_1D_A = np.load("exported_data/M87/gauss_A.npy")
Flux_1D = gauss_1D_A * np.sqrt(2*np.pi) * 1.16
Flux_1D_cube = Flux_1D.reshape(318, 315)

x_y_list = np.load("exported_data/M87/x_y_list.npy")
#%%

plt.figure(1,figsize=(20,20))
plt.imshow(A_rN_shape, interpolation="nearest", origin="lower", cmap="viridis", vmin=1.6, vmax=7)
cb = plt.colorbar()

ax = plt.gca()
cb.set_label("A/rN", fontsize=22)
print("Please select points, then press <ENTER> to proceed.")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.pause(0.001)
#for i, item in enumerate(x_y_list):
#    ax = plt.gca()
#    circ = plt.Circle((item[0],item[1]),6, color="white", fill=False)
#    ax.add_artist(circ)
#    if item[0]<240.:
#        ax.annotate(i+1, (item[0]+6, item[1]-8), color="white", size=12)
#    else:
#        ax.annotate(i+1, (item[0]+8, item[1]+2), color="white", size=12)
#    plt.draw()

x_y_list_new = plt.ginput(n=300,timeout=0, show_clicks=True)
x_y_list_new = np.array(x_y_list_new)
number = range(1,len(x_y_list_new))


    
#%%
    
plt.figure(2,figsize=(20,20))
plt.imshow(Flux_1D_cube, interpolation="nearest", origin="lower", cmap="CMRmap", vmin=0, vmax=150)
cb = plt.colorbar()

ax = plt.gca()
cb.set_label("A/rN", fontsize=22)
print("Please select points, then press <ENTER> to proceed.")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.pause(0.001)
#x_y_list = plt.ginput(n=300,timeout=0, show_clicks=True)
#x_y_list = np.array(x_y_list)
number = range(1,len(x_y_list))

for i, item in enumerate(x_y_list_new):
    ax = plt.gca()
    circ = plt.Circle((item[0],item[1]),6, color="white", fill=False)
    ax.add_artist(circ)
    ax.annotate(i+1, (item[0]-8, item[1]-16), color="white", size=12)
    plt.draw()