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



hdu = fits.open("../"+raw_galaxy)
huduu = fits.open(galaxy+"_AllSpectra.fits")

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

params = fits.open('USED_PARAMS.fits')[0].header
vsys   = float(params['REDSHIFT'])

# Applying some wavelength cuts
idx       = (wave >= float(lmin_lmax[0])*(1.0 + vsys[galaxy]/cvel)) & \
            (wave <= float(lmin_lmax[1])*(1.0 + vsys[galaxy]/cvel))
spec      = spec[idx,:]

idx_good = np.where( np.median(spec, axis=0) > 0.0 )[0]

# Computing the SNR per spaxel
signal = np.nanmedian(spec,axis=0)
#noise  = np.abs(np.nanmedian(np.sqrt(espec),axis=0))
#snr    = signal / noise

hdu1, hdu2      = fits.open(galaxy+'_gandalf-residuals_SPAXEL.fits'), fits.open(galaxy+'_gandalf-emission_SPAXEL.fits')
data1, data2    = hdu1[1].data, (hdu2[1].data)
resid, emission = data1['RESIDUALS'], data2['EMISSION'].T

hdu  = fits.open(galaxy+'_gandalf_SPAXEL.fits')
data = hdu[2].data

AoN = data['AoN'][:,0]

resid[np.isnan(resid)], emission[np.isnan(emission)] = 0.0, 0.0


# Construct the residual cube and emission cube over the original coordinates
s = np.copy(ss)
free_points = s[1]*s[2]
empty = np.zeros((s[0],free_points))
resid_tot    = np.copy(empty)
emission_tot = np.copy(empty)
#rN_tot = np.copy(empty)
AoN_tot      = np.zeros(free_points)
resid_tot[:,idx_good] = np.copy(resid)
emission_tot[:,idx_good] = np.copy(emission)
#rN_tot[:,idx_good] = np.copy(rN)
AoN_tot[idx_good] = np.copy(AoN)

cond = (hdrr['LOGLAM'] >= np.log(lmin_lmax[0])) & (hdrr['LOGLAM'] <= np.log(lmin_lmax[1]))
tmp  = hdrr['LOGLAM'][cond]
resid_tot = resid_tot[cond,:]
#rN_tot = rN_tot[cond,:]
s[0] = len(tmp)

# Rearrange cube to list for saving to a list format .fits file
resid_list = np.swapaxes(np.copy(resid_tot), 1, 0) # shape swapped to x*y, lambda

# Save the data cubes
save_list(resid_list, tmp, galaxy+"_residuals_list.fits")

