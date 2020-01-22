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


my_parser = argparse.ArgumentParser()

my_parser.add_argument('--galaxy', action='store', type=str, required=True) 
# my_parser.add_argument('--lmin_max', action='store', type=str, required=True)
my_parser.add_argument('--loc', action='store', type=str, required=True)
my_parser.add_argument('--extra', action='store', type=str, required=True)
#my_parser.add_argument("--dir", action="store", type=str, required=True)

args = my_parser.parse_args()

galaxy = args.galaxy # FCC000
wave_range = ("4900, 5100")#f"{args.lmin_max}" #4900, 5100
lmin_lmax = [x.strip() for x in wave_range.split(',')]
local = args.loc
extra = args.extra

#external_dir = args.dir # used for external data dir path, i.e. here we extract from uhhpc cluster

# galaxy = sys.argv[1] # FCC000
#vel = sys.argv[2] # (km/s)
# wave_range = f"{sys.argv[2]},{sys.argv[3]}" #4900, 5100
# centre = sys.argv[4]
# extra = sys.argv[5]


if local == "0":
    loc = ""
else:
    loc = local

if extra == "0":
    file_extra = ""
else:    
    file_extra = extra
    


WORK_DIR = f"/data/tspriggs/Jupyterlab_dir/Github/MUSE_PNe_fitting/galaxy_data/{galaxy}_data/"
RAW_DIR  = "/local/tspriggs/Fornax_data_cubes/"
# DATA_DIR = f"/local/tspriggs/Fornax_data_cubes/{galaxy}/"
DATA_DIR = f"/local/tspriggs/muse/MILES_stars_Guerou/{galaxy}/{galaxy}{loc}_{loc}/"


# file_names = [glob.glob("*_AllSpectra.fits*"),
#               glob.glob(f"USED_PARAMS_{loc}.fits*"),
#               glob.glob("*_gandalf-emission_SPAXEL.fits*"),
#               glob.glob("*_gandalf-SPAXEL.fits*"),
#               glob.glob("*_gandalf-bestfit_SPAXEL.fits*")]

# Check for and download the FCC000centre.fits file
# if os.path.isfile(f"/local/tspriggs/Fornax_data_cubes/{galaxy}{loc}.fits") != True:
# #     if (galaxy == "FCC167") | (galaxy == "FCC219") | (galaxy == "FCC153") | (galaxy == "FCC170") | (galaxy == "FCC177"): # or
# # #         os.system(f"scp tspriggs@uhhpc.herts.ac.uk:/data/ralf/muse/MILES_stars_Guerou/{galaxy}/{galaxy}center.fits {RAW_DIR}")
# #         os.system(f"scp tspriggs@uhhpc.herts.ac.uk:/car-data/muse/MILES_stars_Guerou/{galaxy}/{galaxy}center.fits {RAW_DIR}")
# #     else:
#     os.system(f"scp tspriggs@uhhpc.herts.ac.uk:/data/ralf/muse/{galaxy}/{galaxy}{loc}.fits {RAW_DIR}")
# else:
#     print(f"{galaxy}{loc}.fits already exits")
# 
# # list of files needed
# files_needed = [f"{galaxy}{loc}_AllSpectra.fits{file_extra}", 
#                 f"USED_PARAMS.fits{file_extra}", 
#                 f"{galaxy}{loc}_gandalf-residuals_SPAXEL.fits{file_extra}", 
#                 f"{galaxy}{loc}_gandalf-emission_SPAXEL.fits{file_extra}", 
#                 f"{galaxy}{loc}_gandalf_SPAXEL.fits{file_extra}",
#                 f"{galaxy}{loc}_gandalf-bestfit_SPAXEL.fits{file_extra}"]
#                 f"{galaxy}{cen}_ppxf_SPAXELS.fits{file_extra}",
#                 f"{galaxy}{cen}_table.fits{file_extra}"]



# check to see if folder of galaxy already exits
# if os.path.isdir(f"{RAW_DIR}{galaxy}/") == True:
#     print(f"{galaxy} folder already exists.")
# else:
#     os.system(f"mkdir {RAW_DIR}{galaxy}")

# # for file in list "files_needed", check for and download files needed.
# for file in files_needed:
#     if os.path.isfile(f"{DATA_DIR}/{file}") == True:
#         print(f"{file} already exists.")
#     else:
#         if (galaxy == "FCC167") | (galaxy == "FCC219") | (galaxy == "FCC153") | (galaxy == "FCC170") | (galaxy == "FCC177"):
#             os.system(f"scp tspriggs@uhhpc.herts.ac.uk:/data/ralf/muse/MILES_stars_Guerou/{galaxy}/{galaxy}{loc}_{loc}/{file} /local/tspriggs/Fornax_data_cubes/{galaxy}/")
#         else:
# #             os.system(f"scp tspriggs@uhhpc.herts.ac.uk:/data/ralf/muse/{galaxy}/{galaxy}center_center/{file} /local/tspriggs/Fornax_data_cubes/{galaxy}/")
#             os.system(f"scp tspriggs@uhhpc.herts.ac.uk:/data/ralf/muse/{file} /local/tspriggs/Fornax_data_cubes/{galaxy}/")

# os.system(f"mv /local/tspriggs/Fornax_data_cubes/{galaxy}/USED_PARAMS.fits /local/tspriggs/Fornax_data_cubes/{galaxy}/USED_PARAMS_{loc}.fits")
## extract Residual data
# hdu_Allspec = fits.open(DATA_DIR+f"{galaxy}{cen}_AllSpectra.fits{file_extra}")
# spectra = hdu_Allspec[1].data.SPEC.T
# 
# hdu_bestfit = fits.open(DATA_DIR+f"{galaxy}{cen}_gandalf-bestfit_SPAXEL.fits{file_extra}")
# hdu_emission = fits.open(DATA_DIR+f'{galaxy}{cen}_gandalf-emission_SPAXEL.fits{file_extra}')
# 
# aux = np.array(hdu_bestfit[1].data, dtype=float) - np.array(hdu_emission[1].data, dtype=float)
# resid = spectra - aux.T    
# # save residuals?
# ####
# 
# outfits = DATA_DIR+f"{galaxy}{cen}_gandalf-residuals_SPAXEL.fits{file_extra}"
# 
# # Primary HDU
# priHDU = fits.PrimaryHDU()
# #
# ## Extension 1: Table HDU with optimal templates
# cols = []
# cols.append( fits.Column(name='RESIDUALS', format=str(resid.shape[1])+'D', array=resid ) )
# dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
# dataHDU.name = 'RESIDUALS'
# #
# ## Create HDU list and write to file
# HDUList = fits.HDUList([priHDU, dataHDU])
# HDUList.writeto(outfits, overwrite=True)
# 

####


# hdu = fits.open(RAW_DIR+f"{galaxy}{loc}.fits")
hdu = fits.open(RAW_DIR+f"{galaxy}center.fits")

# huduu = fits.open(DATA_DIR+f"{galaxy}{loc}_AllSpectra.fits{file_extra}")
huduu = fits.open(DATA_DIR+f"{galaxy}_AllSpectra.fits{file_extra}")

data  = hdu[1].data
#stat  = hdu[2].data
hdr   = hdu[1].header
s     = np.shape(data)
spec  = np.reshape(data,[s[0],s[1]*s[2]])
spec  = np.array(spec, dtype=float)
#espec = np.reshape(stat,[s[0],s[1]*s[2]])

hdrr   = huduu[2].data
ss     = np.array(np.shape(data))
ss[0]  = len(hdrr['LOGLAM'])

# Getting the wavelength info
wave = hdr['CRVAL3']+(np.arange(s[0])-hdr['CRPIX3'])*hdr['CD3_3']

# params = fits.open(DATA_DIR+f'USED_PARAMS_{loc}.fits{file_extra}')[0].header
params = fits.open(DATA_DIR+f'USED_PARAMS.fits')[0].header
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

# hdu1 = fits.open(DATA_DIR+f'{galaxy}{loc}_gandalf-residuals_SPAXEL.fits{file_extra}')
hdu1 = fits.open(DATA_DIR+f'{galaxy}_gandalf-residuals_SPAXEL.fits{file_extra}')
# hdu2 = fits.open(DATA_DIR+f'{galaxy}{loc}_gandalf-emission_SPAXEL.fits{file_extra}')
hdu2 = fits.open(DATA_DIR+f'{galaxy}_gandalf-emission_SPAXEL.fits{file_extra}')
data1, data2    = hdu1[1].data, (hdu2[1].data)
# data2    = (hdu2[1].data)
resid, emission = data1['RESIDUALS'], data2['EMISSION'].T
# emission = data2['EMISSION'].T

# hdu  = fits.open(DATA_DIR+f'{galaxy}{loc}_gandalf_SPAXEL.fits{file_extra}')
hdu  = fits.open(DATA_DIR+f'{galaxy}_gandalf_SPAXEL.fits{file_extra}')
data = hdu[2].data

AoN = data['AoN'][:,0]

resid[np.isnan(resid)], emission[np.isnan(emission)] = 0.0, 0.0

#assert resid.shape[0] == idx_good.shape[0]

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
save_list(resid_list, tmp, WORK_DIR+f"{galaxy}{loc}_residuals_list.fits", s)
#save_cube(resid_tot.reshape((s[0],s[1],s[2])),tmp, directory + WORK_DIR+f"{galaxy}_residuals_cube.fits", s)



