####################################################
# RECOVERING THE DIMESION OF THE ORIGINAL MUSE CUBE #
#####################################################

import numpy as np
from astropy.io import fits

cvel      = 299792.458

######################################################
# Storing the data in .fits file
def save_cube(data,header,name):
    hdu  = fits.PrimaryHDU()
    hdu2 = fits.ImageHDU()
    hdu.data = np.copy(data)
    hdu2 = fits.ImageHDU(data=header, name='WAVELENGTH')

    print('Data cube recovered in -->',name)

    # Create HDU list and write to file
    HDUList = fits.HDUList([hdu, hdu2])
    HDUList.writeto(name, overwrite=True)
    
def save_list(data, header, name):
    hdu = fits.PrimaryHDU()
    hdu2 = fits.ImageHDU()
    hdu.data = np.copy(data)
    hdu2 = fits.ImageHDU(data = header, name="WAVELENGTH")
    
    print("Data to be saved in a list format.")
    HDUList = fits.HDUList([hdu, hdu2])
    HDUList.writeto(name, overwrite=True)
########################################################

galaxy = input("Please type in the galaxy name, in the format of FCC000:  ")
vel = input("Please type in the galxy velocity from Simbad (km/s):  ")
wave_range = input("Please type in the 2 wavelength ranges, used in the Config file, as follows: lmin, lmax:  ")
lmin_lmax = [x.strip() for x in wave_range.split(',')]
directory = galaxy + "_data/"
raw_galaxy = galaxy + 'center.fits'

# vsys from Simbad
#vsys = {'FCC153':1638, 'FCC170': 1769., 'FCC177':1567, "FCC167":1841}
vsys = {galaxy:float(vel)}

hdu = fits.open("../"+raw_galaxy)
huduu = fits.open(galaxy+"center_AllSpectra.fits")

data  = hdu[1].data
#stat  = hdu[2].data
hdr   = hdu[1].header
s     = np.shape(data)
spec  = np.reshape(data,[s[0],s[1]*s[2]])
#espec = np.reshape(stat,[s[0],s[1]*s[2]])

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
idx       = (wave >= float(lmin_lmax[0])*(1.0 + vsys[galaxy]/cvel)) & \
            (wave <= float(lmin_lmax[1])*(1.0 + vsys[galaxy]/cvel))
spec      = spec[idx,:]
#espec     = espec[idx,:]
wave      = wave[idx]
velscale  = (wave[1]-wave[0])*cvel/np.mean(wave)
xy_extent = np.array( [data.shape[2], data.shape[1]] )

idx_good = np.where( np.median(spec, axis=0) > 0.0 )[0]
spec     = spec[:,idx_good]
#espec    = espec[:,idx_good]
spec[np.isnan(spec)] = 0.0
#espec[np.isnan(espec)] = 0.0
x        = x[idx_good]
y        = y[idx_good]

# Computing the SNR per spaxel
signal = np.nanmedian(spec,axis=0)
#noise  = np.abs(np.nanmedian(np.sqrt(espec),axis=0))
#snr    = signal / noise

# Open the residuals and the emission cube obtained with GandALF
hdu1, hdu2      = fits.open(galaxy+'center_gandalf-residuals_SPAXEL.fits'), fits.open(galaxy+'center_gandalf-emission_SPAXEL.fits')
data1, data2    = hdu1[1].data, (hdu2[1].data)
resid, emission = data1['RESIDUALS'], data2['EMISSION'].T

hdu  = fits.open(galaxy+'center_gandalf_SPAXEL.fits')
data = hdu[2].data
#AoN  = data['AoN'][:,11]

resid[np.isnan(resid)], emission[np.isnan(emission)] = 0.0, 0.0


# Construct the residual cube and emission cube over the original coordinates
s = np.copy(ss)
free_points  = s[1]*s[2]
empty = np.zeros((s[0],free_points))
resid_tot    = np.copy(empty)
emission_tot = np.copy(empty)
#AoN_tot      = np.zeros(free_points)
resid_tot[:,idx_good]    = np.copy(resid) # shape = lambda, x*y
emission_tot[:,idx_good] = np.copy(emission)
#AoN_tot[idx_good]        = np.copy(AoN)

# Rearrange cube to list for saving to a list format .fits file
resid_list = np.swapaxes(np.copy(resid_tot), 1, 0) # shape swapped to x*y, lambda

# Save the data cubes

#save_cube(resid_tot.reshape((s[0],s[1],s[2])),hdrr['LOGLAM'],directory + "reshaped_cubes/"+galaxy+'_residuals.fits')
#save_cube(emission_tot.reshape((s[0],s[1],s[2])),hdrr['LOGLAM'],directory + "reshaped_cubes/"+galaxy+'_emission.fits')
#save_cube(AoN_tot.reshape((s[1],s[2])),hdrr['LOGLAM'],directory + "reshaped_cubes/"+galaxy+'_AoN.fits')

save_list(resid_list, hdrr["LOGLAM"], galaxy+"center_residuals_list_long.fits")