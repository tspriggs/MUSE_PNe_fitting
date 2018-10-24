#####################################################
# RECOVERING THE DIMESION OF THE ORIGINAL MUSE CUBE #
#####################################################

import numpy as np
from astropy.io import fits

import os

import pdb

cvel      = 299792.458

def save_cube(data,header,name):
    hdu = fits.PrimaryHDU()
    hdu.data = np.copy(data)
    hdu.header['CRVAL3']  = min(np.exp(hdrr["LOGLAM"]))
    hdu.header['CRPIX3']  = 1.0
    hdu.header['CDELT3'] = np.exp(hdrr["LOGLAM"])[1]-np.exp(hdrr["LOGLAM"])[0]

    print('Data cube recovered in -->',name)

#    if os.path.exists(name):
#       os.remove(name)
    hdu.writeto(name, overwrite=True)

directory = 'FCC167_data/'
#filename = 'FCC153_center.fits'
filename = 'FCC167_DATACUBE_center.fits'
#filename = 'FCC177_center.fits'
galaxy   = filename[:6]

vsys = {'FCC153':1638, 'FCC170': 1769., 'FCC177':1567, "FCC167":1841}

hdu = fits.open(directory + filename)
huduu = fits.open(directory + galaxy + '_AllSpectra.fits')

data  = hdu[1].data
stat  = hdu[2].data
hdr   = hdu[1].header
s      = np.shape(data)
spec  = np.reshape(data,[s[0],s[1]*s[2]])
espec = np.reshape(stat,[s[0],s[1]*s[2]])

hdrr  = huduu[2].data
ss     = np.array(np.shape(data))
ss[0]  = len(hdrr['LOGLAM'])

# Getting the wavelength info
wave = hdr['CRVAL3']+(np.arange(s[0])-hdr['CRPIX3'])*hdr['CD3_3']

# Getting the spatial coordinates
xaxis = np.arange(s[2])*hdr['CD2_2']*3600.0
yaxis = np.arange(s[1])*hdr['CD2_2']*3600.0
x, y  = np.meshgrid(xaxis,yaxis)
x     = np.reshape(x,[s[1]*s[2]])
y     = np.reshape(y,[s[1]*s[2]])
pixelsize = x[1] - x[0]

# Applying some wavelength cuts
idx       = (wave >= 4700*(1.0 + vsys[galaxy]/cvel)) & \
            (wave <= 6900*(1.0 + vsys[galaxy]/cvel))
spec      = spec[idx,:]
espec     = espec[idx,:]
wave      = wave[idx]
velscale  = (wave[1]-wave[0])*cvel/np.mean(wave)
xy_extent = np.array( [data.shape[2], data.shape[1]] )

idx_good = np.where( np.median(spec, axis=0) > 0.0 )[0]
spec     = spec[:,idx_good]
espec    = espec[:,idx_good]
spec[np.isnan(spec)] = 0.0
espec[np.isnan(espec)] = 0.0
x        = x[idx_good]
y        = y[idx_good]

# Computing the SNR per spaxel
signal = np.nanmedian(spec,axis=0)
noise  = np.abs(np.nanmedian(np.sqrt(espec),axis=0))
snr    = signal / noise

# Open the residuals and the emission cube obtained with GandALF
hdu1, hdu2      = fits.open(directory+galaxy+'_gandalf-residuals_SPAXEL.fits'), fits.open(directory+galaxy+'_gandalf-emission_SPAXEL.fits')
data1, data2    = hdu1[1].data, (hdu2[1].data)
resid, emission = data1['RESIDUALS'], data2['EMISSION'].T

hdu  = fits.open(directory+galaxy+'_gandalf_SPAXEL.fits')
data = hdu[2].data
AoN  = data['AoN'][:,11]

resid[np.isnan(resid)], emission[np.isnan(emission)] = 0.0, 0.0


# Construct the residual cube and emission cube over the original coordinates
s = np.copy(ss)
free_points = s[1]*s[2]
empty = np.zeros((s[0],free_points))
resid_tot    = np.copy(empty)
emission_tot = np.copy(empty)
AoN_tot      = np.zeros(free_points)
resid_tot[:,idx_good] = np.copy(resid)
emission_tot[:,idx_good] = np.copy(emission)
AoN_tot[idx_good] = np.copy(AoN)
#pdb.set_trace()

# Save the data cubes

save_cube(resid_tot.reshape((s[0],s[1],s[2])),hdr,galaxy+'_residual_cube_pablo.fits')
save_cube(emission_tot.reshape((s[0],s[1],s[2])),hdr,galaxy+'_emission_cube_pablo.fits')
save_cube(AoN_tot.reshape((s[1],s[2])),hdr,galaxy+'_AoN_cube_pablo.fits')