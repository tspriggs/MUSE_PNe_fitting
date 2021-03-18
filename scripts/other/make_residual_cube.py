from astropy.io import fits
import numpy as np 
import argparse
import glob


my_parser = argparse.ArgumentParser()

my_parser.add_argument('--galaxy', action='store', type=str, required=True)
my_parser.add_argument('--loc', action='store', type=str, required=True)
my_parser.add_argument('--origin', action='store', type=str, required=True)


args = my_parser.parse_args()

galaxy = args.galaxy
loc = args.loc
origin = args.origin


def save_cube(data, wave, hdr, fname, s):
    p_hdu = fits.PrimaryHDU()

    data_hdu = fits.ImageHDU(data=np.copy(data), header=hdr, name="DATA", )
    wave_hdu = fits.ImageHDU(data=wave, name='WAVELENGTH',)

    # Create HDU list and write to file
    HDUList = fits.HDUList([p_hdu, data_hdu, wave_hdu])
    HDUList.writeto(fname, overwrite=True)
    print(f'Residual cube saved to --> {fname}')




# Directory where the final residual cube will be saved to, change as you see fit, to match your working directory
EXPORT_DIR = f"/data/tspriggs/Jupyterlab_dir/Github/MUSE_PNe_fitting/galaxy_data/{galaxy}_data/"

# Decide where to read data from, as determined from command line input arguments
if origin == "re_reduced":
    # Where the GIST output files are located
    WORK_DIR = f"/local/tspriggs/re_reduced_F3D/gist_results/{galaxy}{loc}_{loc}/"
    RAW_DIR = "/local/tspriggs/re_reduced_F3D/"
elif origin == "F3D_gist":
    if galaxy in ["FCC153", "FCC167", "FCC170", "FCC177", "FCC219"]:
        WORK_DIR = f"/local/tspriggs/muse/MILES_stars_Guerou/{galaxy}/{galaxy}{loc}_{loc}/"
        RAW_DIR =  f"/local/tspriggs/muse/{galaxy}/"
    else:
        WORK_DIR = f"/local/tspriggs/muse/{galaxy}/{galaxy}{loc}_{loc}/"
        RAW_DIR =  f"/local/tspriggs/muse/{galaxy}/"



hdu_Allspec = fits.open(glob.glob(WORK_DIR+f"{galaxy}*_AllSpectra.fits*")[0])
spectra = hdu_Allspec[1].data.SPEC.T
# extract wavelength from AllSpec fits file
log_wavelength = hdu_Allspec[2].data["LOGLAM"]
wavelength = np.exp(log_wavelength)

hdu_bestfit = fits.open(
    glob.glob(WORK_DIR+f"{galaxy}*_gandalf-bestfit_SPAXEL.fits*")[0])
hdu_emission = fits.open(
    glob.glob(WORK_DIR+f"{galaxy}*_gandalf-emission_SPAXEL.fits*")[0])

# Stellar light = Gandalf-BESTFIT - Gandalf-EMISSION
stellar = hdu_bestfit[1].data.BESTFIT - hdu_emission[1].data.EMISSION
# Residuals (with nebulous emissions) = raw_spectra - stellar light
residuals = spectra - stellar.T

# Open raw data to extract fits header, and shape of raw data cube
with fits.open(RAW_DIR+f"{galaxy}{loc}.fits") as raw_hdu:
    raw_hdr = raw_hdu[1].header
    raw_shape = np.shape(raw_hdu[1].data)

#wavelength = raw_hdr['CRVAL3']+(np.arange(raw_shape[0])-raw_hdr['CRPIX3'])*raw_hdr['CD3_3']

# If number of spectra in residuals less than raw data x-axis length, times y-axis length
# then project residuals to original pixel locations.
if np.shape(residuals)[1] < raw_shape[1]*raw_shape[2]:
    yaxis = np.arange(raw_shape[1])*raw_hdr['CD2_2']*3600.0
    xaxis = np.arange(raw_shape[2])*raw_hdr['CD2_2']*3600.0

    # Open the _table.fits file and get x_pix and y_pix index values for re shaping
    table_hdu = fits.open(glob.glob(WORK_DIR+f"{galaxy}*_table.fits*")[0])
    table_data = table_hdu[1].data

    y_pix = table_data["Y"]
    x_pix = table_data["X"]

    # check where the index location of the x,y coordinates of the fitted pixels are, relative to the xaxis and yaxis coordinate systems
    index_pix = np.zeros((len(x_pix), 2))

    for n, (i, j) in enumerate(zip(x_pix, y_pix)):
        if (i not in xaxis) or (j not in yaxis):
            continue
        else:
            index_pix[n] = [np.where(xaxis == i)[0], np.where(yaxis == j)[0]]


    # Make an empty cube, with the same shape as the input raw data cube: lambda, y, x, 
    residual_cube = np.zeros((len(wavelength), raw_shape[1], raw_shape[2]))

    # Fill in the empty cube, using the y,x index values of the fitted pixel locations.
    for n, i in enumerate(index_pix):
        residual_cube[:, int(i[1]), int(i[0])] = residuals[:, n]

# If number of spectra in residuals matches x-axis time y-axis length from raw data
# Simply reshape residuals into cube: lambda, y, x
elif np.shape(residuals)[1] == raw_shape[1]*raw_shape[2]:
    residual_cube = residuals.reshape(len(wavelength), raw_shape[1], raw_shape[2])
    


# Shorten the wavelength range to that between 4900 and 5100, to encapsulate the region containing [OIII] emissions
#raw_wave = raw_hdr['CRVAL3']+(np.arange(raw_shape[0])-raw_hdr['CRPIX3'])*raw_hdr['CD3_3']
cond = (log_wavelength >= np.log(float(4900.))) & (log_wavelength <= np.log(float(5100.)))
#cond = (raw_wave >= float(4900.)) & (raw_wave <= float(5100.))

# apply cond filter to the residual cube, and to the wavelength array
residual_cube_cond = residual_cube[cond, :, :]
wavelength_cond = wavelength[cond]
#wavelength_cond = raw_wave[cond]

# Save residuals in cube format
save_cube(residual_cube_cond, wavelength_cond, raw_hdr,
          EXPORT_DIR+f"{galaxy}{loc}_residual_cube.fits", raw_shape)
