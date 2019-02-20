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

galaxies = ['FCC153',
'FCC170',
'FCC177']

for i in galaxies:
    for region in ['center','halo']:
        directory = i+'_'+region+'_SNR_50/'
        #filename = 'FCC153_center.fits'
        filename = i+'_AllSpectra.fits'
        #filename = 'FCC177_center.fits'

# vsys from Simbad
#vsys = {'FCC153':1638, 'FCC170': 1769., 'FCC177':1567, "FCC167":1841}
vsys = {galaxy:float(vel)}

hdu = fits.open(i+'_'+region+'.fits')
huduu = fits.open(directory + filename)

data  = hdu[1].data
#stat  = hdu[2].data
hdr   = hdu[1].header
s      = np.shape(data)
spec  = np.reshape(data,[s[0],s[1]*s[2]])
#espec = np.reshape(stat,[s[0],s[1]*s[2]])

hdrr  = huduu[2].data
ss     = np.array(np.shape(data))
ss[0]  = len(hdrr['LOGLAM'])

# Getting the wavelength info
wave = hdr['CRVAL3']+(np.arange(s[0])-hdr['CRPIX3'])*hdr['CD3_3']

params = fits.open(directory + 'USED_PARAMS.fits')[0].header
vsys   = float(params['REDSHIFT'])

# Applying some wavelength cuts
idx       = (wave >= float(params['LMIN'])*(1.0 + vsys/cvel)) & \
            (wave <= float(params['LMAX'])*(1.0 + vsys/cvel))
spec      = spec[idx,:]

idx_good = np.where( np.median(spec, axis=0) > 0.0 )[0]


# Open the residuals and the emission cube obtained with GandALF
hdu1, hdu2      = fits.open(directory+i+'_gandalf-residuals_SPAXEL.fits'), fits.open(directory+i+'_gandalf-emission_SPAXEL.fits')
data1, data2    = hdu1[1].data, (hdu2[1].data)
resid, emission = data1['RESIDUALS'], data2['EMISSION'].T

hdu  = fits.open(directory+i+'_gandalf_SPAXEL.fits')
data = hdu[2].data
if i != 'FCC153':
    AoN = data['AoN'][:,0]
else:
    AoN  = data['AoN'][:,1]

#rN = resid - emission # Computing the residual cube

resid[np.isnan(resid)], emission[np.isnan(emission)] = 0.0, 0.0

#pdb.set_trace()
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
#pdb.set_trace()

cond = (hdrr['LOGLAM'] >= np.log(4900.0)) & (hdrr['LOGLAM'] <= np.log(5100.0))
tmp  = hdrr['LOGLAM'][cond]
resid_tot = resid_tot[cond,:]
#rN_tot = rN_tot[cond,:]
s[0] = len(tmp)
#tmp = hdrr['LOGLAM']

# Computing the robust sigma (pPXF) to get the residual cube
#sigma = []
#for i in range(len(rN_tot[0])):
#    sigma.append(robust_sigma(rN_tot[:,i]))
#rN_tot = np.array(sigma)
#rN_tot[np.isnan(rN_tot)] = 0.0

# Save the data cubes
save_cube(resid_tot.reshape((s[0],s[1],s[2])),tmp,directory+i+'_residuals.fits')
#save_cube(emission_tot.reshape((s[0],s[1],s[2])),hdrr['LOGLAM'],directory+galaxy+'_emission.fits')
save_cube(AoN_tot.reshape((s[1],s[2])),tmp,directory+i+'_AoN.fits')
#save_cube(rN_tot.reshape((s[1],s[2])),tmp,directory+galaxy+'_rN.fits')

# Rearrange cube to list for saving to a list format .fits file
resid_list = np.swapaxes(np.copy(resid_tot), 1, 0) # shape swapped to x*y, lambda

# Save the data cubes
save_list(resid_list, tmp, directory+i+"_residuals_list.fits")