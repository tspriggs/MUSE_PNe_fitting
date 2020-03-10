####################################################
# RECOVERING THE DIMESION OF THE ORIGINAL MUSE CUBE #
#####################################################

import numpy as np
from astropy.io import fits
import sys
import os
import argparse
import glob


cvel      = 299792.458

######################################################
# Storing the data in .fits file
def save_cube(data, wave, hdr , fname, s):
    p_hdu  = fits.PrimaryHDU()
    data_hdu = fits.ImageHDU(data = np.copy(data), header=hdr, name="DATA", )
    wave_hdu = fits.ImageHDU(data=wave, name='WAVELENGTH',)
    # hdr = data_hdu.header
    # hdr.set("YAXIS", value=s[1])
    # hdr.set("XAXIS", value=s[2])

    print(f'Data cube recovered in --> {name}')

    # Create HDU list and write to file
    HDUList = fits.HDUList([p_hdu, data_hdu, wave_hdu])
    HDUList.writeto(fname, overwrite=True)
    
def save_list(data, header, name, s):
    hdu = fits.PrimaryHDU()
    hdu2 = fits.ImageHDU()
    hdu.data = np.copy(data)
    hdu2 = fits.ImageHDU(data = header, name="WAVELENGTH")
    hdr = hdu.header
    hdr.set("YAXIS", value=s[1])
    hdr.set("XAXIS", value=s[2])
    
    print("Data to be saved in a list format.")
    HDUList = fits.HDUList([hdu, hdu2])
    HDUList.writeto(name, overwrite=True)
########################################################


my_parser = argparse.ArgumentParser()

my_parser.add_argument('--galaxy', action='store', type=str, required=True) 
my_parser.add_argument('--loc', action='store', type=str, required=True)
my_parser.add_argument('--extra', action='store', type=str, required=True)


args = my_parser.parse_args()

galaxy = args.galaxy 
loc = args.loc
extra = args.extra


if loc == "0":
    loc = ""

if extra == "0":
    file_extra = ""
else:    
    file_extra = extra
    

WORK_DIR   = f"/local/tspriggs/re_reduced_F3D/gist_results/" # Where the GIST output files are located
RAW_DIR    = "/local/tspriggs/re_reduced_F3D/"
EXPORT_DIR = "/data/tspriggs/Jupyterlab_dir/Github/MUSE_PNe_fitting/galaxy_data/{galaxy}_data/" 

## extract Residual data
hdu_Allspec = fits.open(WORK_DIR+f"{galaxy}{loc}_AllSpectra.fits{file_extra}")
spectra = hdu_Allspec[1].data.SPEC.T
# 
hdu_bestfit = fits.open(WORK_DIR+f"{galaxy}{loc}_gandalf-bestfit_SPAXEL.fits{file_extra}")
hdu_emission = fits.open(WORK_DIR+f'{galaxy}{loc}_gandalf-emission_SPAXEL.fits{file_extra}')
# 
aux = np.array(hdu_bestfit[1].data, dtype=float) - np.array(hdu_emission[1].data, dtype=float)
residuals = spectra - aux.T    

with fits.open(RAW_DIR+"{galaxy}{loc}.fits") as raw_hdu:
    raw_hdr = raw_hdu[1].header
    raw_shape = np.shape(raw_hdu[1].data)

xaxis = np.arange(raw_shape[2])*raw_hdr['CD2_2']*3600.0
yaxis = np.arange(raw_shape[1])*raw_hdr['CD2_2']*3600.0

# Open the _table.fits file and get x_pix and y_pix
table_hdu = fits.open(WORK_DIR+f"{galaxy}{loc}_table.fits")
table_data = table_hdu[1].data

x_pix = table_data["X"]
y_pix = table_data["Y"]


# check where the index location of the x,y coordinates of the fitted pixels are, relative to the xaxis and yaxis coordinate systems
index_pix = np.zeros((len(x_pix),2))

for n,(i,j) in enumerate(zip(x_pix, y_pix)):
    index_pix[n] = [np.squeeze(np.where(xaxis == i)), np.squeeze(np.where(yaxis==j))]


# load up the cleaned_spaxel data and get wavelength
with fits.open(WORK_DIR+f"{galaxy}{loc}_gandalf-cleaned_SPAXEL.fits") as clean_hdulist:
    wavelength = np.exp(clean_hdulist[2].data.LOGLAM)


# Make an empty pointing of the same shape as the input raw data cube: y, x, lambda
residual_cube = np.zeros((len(wavelength), raw_shape[1], raw_shape[2], ))

# Fill in the empty pointing, using the x,y index values of the fitted pixel locations.
for n, i in enumerate(index_pix):
    residual_cube[:,int(i[1]),int(i[0])] = residuals[:,n]

# Save residual data cube

# use wavelength shortening condition to reduce wavelength range to between 4900 and 5100
# rename empty_gal
cond = (wavelength >= np.log(float(4900.))) & (wavelength <= np.log(float(5100.)))
residual_cube[cond,:,:]

save_cube(residual_cube, wavelength, raw_hdr, WORK_DIR+f"{galaxy}{loc}_residual_cube.fits", raw_shape)



# WORK_DIR = f"/data/tspriggs/Jupyterlab_dir/Github/MUSE_PNe_fitting/galaxy_data/{galaxy}_data/"
# DATA_DIR = f"/local/tspriggs/Fornax_data_cubes/{galaxy}/"
# DATA_DIR = f"/local/tspriggs/muse/MILES_stars_Guerou/{galaxy}/{galaxy}{loc}_{loc}/"
# DATA_DIR = f"/local/tspriggs/muse/{galaxy}/{galaxy}{loc}_{loc}/"






####


# # hdu = fits.open(RAW_DIR+f"{galaxy}{loc}.fits")
# hdu = fits.open(RAW_DIR+f"{galaxy}center.fits")

# # huduu = fits.open(DATA_DIR+f"{galaxy}{loc}_AllSpectra.fits{file_extra}")
# huduu = fits.open(DATA_DIR+f"{galaxy}_AllSpectra.fits{file_extra}")

# data  = hdu[1].data
# #stat  = hdu[2].data
# hdr   = hdu[1].header
# s     = np.shape(data)
# spec  = np.reshape(data,[s[0],s[1]*s[2]])
# spec  = np.array(spec, dtype=float)
# #espec = np.reshape(stat,[s[0],s[1]*s[2]])

# hdrr   = huduu[2].data
# ss     = np.array(np.shape(data))
# ss[0]  = len(hdrr['LOGLAM'])

# # Getting the wavelength info
# wave = hdr['CRVAL3']+(np.arange(s[0])-hdr['CRPIX3'])*hdr['CD3_3']

# params = fits.open(DATA_DIR+f'USED_PARAMS.fits{file_extra}')[0].header
# vsys   = float(params['REDSHIFT'])

# # Applying some wavelength cuts
# idx       = (wave >= float(params["LMIN"])*(1.0 + vsys/cvel)) & \
#             (wave <= float(params["LMAX"])*(1.0 + vsys/cvel))
# spec      = spec[idx,:]

# idx_good = np.where( np.median(spec, axis=0) > 0.0 )[0]

# # Computing the SNR per spaxel
# signal = np.nanmedian(spec,axis=0)
# #noise  = np.abs(np.nanmedian(np.sqrt(espec),axis=0))
# #snr    = signal / noise

# hdu1 = fits.open(DATA_DIR+f'{galaxy}_gandalf-residuals_SPAXEL.fits{file_extra}')
# hdu2 = fits.open(DATA_DIR+f'{galaxy}_gandalf-emission_SPAXEL.fits{file_extra}')
# data1, data2    = hdu1[1].data, (hdu2[1].data)
# resid, emission = data1['RESIDUALS'], data2['EMISSION'].T

# hdu  = fits.open(DATA_DIR+f'{galaxy}_gandalf_SPAXEL.fits{file_extra}')
# data = hdu[2].data

# AoN = data['AoN'][:,0]

# resid[np.isnan(resid)], emission[np.isnan(emission)] = 0.0, 0.0

# #assert resid.shape[0] == idx_good.shape[0]

# # Construct the residual cube and emission cube over the original coordinates
# s = np.copy(ss)
# free_points = s[1]*s[2]
# empty = np.zeros((s[0],free_points))
# resid_tot    = np.copy(empty)
# emission_tot = np.copy(empty)
# #rN_tot = np.copy(empty)
# AoN_tot      = np.zeros(free_points)
# resid_tot[:,idx_good] = np.copy(resid)
# emission_tot[:,idx_good] = np.copy(emission)
# #rN_tot[:,idx_good] = np.copy(rN)
# AoN_tot[idx_good] = np.copy(AoN)


# wave_range = ("4900, 5100")#f"{args.lmin_max}" #4900, 5100
# lmin_lmax = [x.strip() for x in wave_range.split(',')]

# cond = (hdrr['LOGLAM'] >= np.log(float(lmin_lmax[0]))) & (hdrr['LOGLAM'] <= np.log(float(lmin_lmax[1])))
# tmp  = hdrr['LOGLAM'][cond]
# resid_tot = resid_tot[cond,:]
# #rN_tot = rN_tot[cond,:]
# s[0] = len(tmp)

# # Rearrange cube to list for saving to a list format .fits file
# resid_list = np.swapaxes(np.copy(resid_tot), 1, 0) # shape swapped to x*y, lambda

# # Save the data cubes
# save_list(resid_list, tmp, WORK_DIR+f"{galaxy}{loc}_residuals_list.fits", s)
# #save_cube(resid_tot.reshape((s[0],s[1],s[2])),tmp, directory + WORK_DIR+f"{galaxy}_residuals_cube.fits", s)



