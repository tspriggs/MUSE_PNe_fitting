####################################################
# RECOVERING THE DIMESION OF THE ORIGINAL MUSE CUBE #
#####################################################

import numpy as np
from astropy.io import fits
import sys
import os

cvel      = 299792.458

######################################################
# Storing the data in .fits file
def save_cube(data,header,name, s):
    hdu  = fits.PrimaryHDU()
    hdu2 = fits.ImageHDU()
    hdu.data = np.copy(data)
    hdu2 = fits.ImageHDU(data=header, name='WAVELENGTH')
    hdr = hdu.header
    hdr.set("YAXIS", value=s[1])
    hdr.set("XAXIS", value=s[2])

    print('Data cube recovered in -->',name)

    # Create HDU list and write to file
    HDUList = fits.HDUList([hdu, hdu2])
    HDUList.writeto(name, overwrite=True)
    
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

galaxy = sys.argv[1] # FCC000
#vel = sys.argv[2] # (km/s)
wave_range = f"{sys.argv[2]},{sys.argv[3]}" #4900, 5100
lmin_lmax = [x.strip() for x in wave_range.split(',')]
centre = sys.argv[4]
extra = sys.argv[5]

if centre == "1":
    cen = "center"
else:
    cen = ""
    print("Everything is fine")

if extra == "0":
    file_extra = ""
    print("Everything is fine")
else:    
    file_extra = extra
    
    


WORK_DIR = f"/data/tspriggs/Jupyterlab_dir/Github/MUSE_PNe_fitting/galaxy_data/{galaxy}_data/"
RAW_DIR  = "/local/tspriggs/Fornax_data_cubes/"
DATA_DIR = f"/local/tspriggs/Fornax_data_cubes/{galaxy}/"

# Check for and download the FCC000centre.fits file
if os.path.isfile(f"/local/tspriggs/Fornax_data_cubes/{galaxy}center.fits") != True:
    if (galaxy == "FCC167") | (galaxy == "FCC219"): # or
        os.system(f"scp tspriggs@uhhpc.herts.ac.uk:/data/ralf/muse/MILES_stars_Guerou/{galaxy}/{galaxy}center.fits {RAW_DIR}")
    else:
        os.system(f"scp tspriggs@uhhpc.herts.ac.uk:/data/ralf/muse/{galaxy}/{galaxy}center.fits {RAW_DIR}")
else:
    print(f"{galaxy}center.fits already exits")

# list of files needed
files_needed = [f"{galaxy}{cen}_AllSpectra.fits{file_extra}", 
                f"USED_PARAMS.fits{file_extra}", 
                f"{galaxy}{cen}_gandalf-residuals_SPAXEL.fits{file_extra}", 
                f"{galaxy}{cen}_gandalf-emission_SPAXEL.fits{file_extra}", 
                f"{galaxy}{cen}_gandalf_SPAXEL.fits{file_extra}",
                f"{galaxy}{cen}_ppxf_SPAXELS.fits{file_extra}",
                f"{galaxy}{cen}_table.fits{file_extra}"]

# check to see if folder of galaxy already exits
if os.path.isdir(f"{RAW_DIR}{galaxy}/") == True:
    print(f"{galaxy} folder already exists.")
else:
    os.system(f"mkdir {RAW_DIR}{galaxy}")

# for file in list "files_needed", check for and download files needed.
for file in files_needed:
    if os.path.isfile(f"{DATA_DIR}/{file}") == True:
        print(f"{file} already exists.")
    else:
        if (galaxy == "FCC167") | (galaxy == "FCC219"):
            os.system(f"scp tspriggs@uhhpc.herts.ac.uk:/data/ralf/muse/MILES_stars_Guerou/{galaxy}/{galaxy}center_center/{file} /local/tspriggs/Fornax_data_cubes/{galaxy}/")
        else:
            os.system(f"scp tspriggs@uhhpc.herts.ac.uk:/data/ralf/muse/{galaxy}/{galaxy}center_center/{file} /local/tspriggs/Fornax_data_cubes/{galaxy}/")
    

hdu = fits.open(RAW_DIR+f"{galaxy}center.fits")
huduu = fits.open(DATA_DIR+f"{galaxy}{cen}_AllSpectra.fits{file_extra}")

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

params = fits.open(DATA_DIR+f'USED_PARAMS.fits{file_extra}')[0].header
vsys   = float(params['REDSHIFT'])

# Applying some wavelength cuts
idx       = (wave >= float(params["LMIN"])*(1.0 + vsys/cvel)) & \
            (wave <= float(params["LMAX"])*(1.0 + vsys/cvel))
spec      = spec[idx,:]

idx_good = np.where( np.median(spec, axis=0) > 0.0 )[0]

# Computing the SNR per spaxel
signal = np.nanmedian(spec,axis=0)
#noise  = np.abs(np.nanmedian(np.sqrt(espec),axis=0))
#snr    = signal / noise

hdu1 = fits.open(DATA_DIR+f'{galaxy}{cen}_gandalf-residuals_SPAXEL.fits{file_extra}')
hdu2 = fits.open(DATA_DIR+f'{galaxy}{cen}_gandalf-emission_SPAXEL.fits{file_extra}')
data1, data2    = hdu1[1].data, (hdu2[1].data)
resid, emission = data1['RESIDUALS'], data2['EMISSION'].T

hdu  = fits.open(DATA_DIR+f'{galaxy}{cen}_gandalf_SPAXEL.fits{file_extra}')
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

cond = (hdrr['LOGLAM'] >= np.log(float(lmin_lmax[0]))) & (hdrr['LOGLAM'] <= np.log(float(lmin_lmax[1])))
tmp  = hdrr['LOGLAM'][cond]
resid_tot = resid_tot[cond,:]
#rN_tot = rN_tot[cond,:]
s[0] = len(tmp)

# Rearrange cube to list for saving to a list format .fits file
resid_list = np.swapaxes(np.copy(resid_tot), 1, 0) # shape swapped to x*y, lambda

# Save the data cubes
save_list(resid_list, tmp, WORK_DIR+f"{galaxy}_residuals_list.fits", s)
#save_cube(resid_tot.reshape((s[0],s[1],s[2])),tmp, directory + WORK_DIR+f"{galaxy}_residuals_cube.fits", s)



